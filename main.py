import os
import sys
import queue
import threading
import time
import json
import logging
import traceback
import shutil

# ==========================================
# LOGGING SETUP (must be before any other imports)
# ==========================================
base_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(base_dir, "app.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

def handle_exception(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    logger.critical("未捕捉の例外が発生しました", exc_info=(exc_type, exc_value, exc_tb))

sys.excepthook = handle_exception

# ==========================================
# NVIDIA CUDA DLL PRELOAD
# ==========================================
if sys.platform == 'win32':
    try:
        import ctypes as _ctypes
        import glob as _glob
        import site as _site

        _cuda_dll_names = [
            # CUDA Runtime
            'cudart64_12.dll', 'cudart64_121.dll', 'cudart64_120.dll',
            # cuBLAS
            'cublas64_12.dll', 'cublasLt64_12.dll',
            'cublas64_11.dll', 'cublasLt64_11.dll',
            # cuDNN
            'cudnn64_9.dll', 'cudnn64_8.dll',
            'cudnn_ops64_9.dll', 'cudnn_graph64_9.dll',
            'cudnn_ops_infer64_8.dll', 'cudnn_cnn_infer64_8.dll',
            'cudnn_adv_infer64_8.dll',
            # NVRTC / JIT / ONNX Runtime dependencies
            'nvJitLink_120_0.dll', 'nvJitLink.dll',
            'nvrtc64_120_0.dll', 'nvrtc64_121_0.dll', 'nvrtc64_12.dll',
            'nvrtc-builtins64_121.dll', 'nvrtc-builtins64_120.dll',
            # Other CUDA libs
            'curand64_10.dll',
            'cusolver64_11.dll', 'cusolver64_12.dll',
            'cusparse64_12.dll',
            'cufft64_11.dll', 'cufftw64_11.dll',
            'nppc64_12.dll', 'nppial64_12.dll',
        ]

        _search_dirs = []
        _sp_paths = list(_site.getsitepackages())
        if hasattr(_site, 'getusersitepackages'):
            _sp_paths.append(_site.getusersitepackages())
        _venv_sp = os.path.join(sys.prefix, 'Lib', 'site-packages')
        if _venv_sp not in _sp_paths:
            _sp_paths.append(_venv_sp)
            
        for _sp in _sp_paths:
            if not os.path.isdir(_sp):
                continue
            _search_dirs += _glob.glob(os.path.join(_sp, 'nvidia', '*', 'bin'))
            _search_dirs += _glob.glob(os.path.join(_sp, 'nvidia', '*', 'lib'))
            _search_dirs.append(os.path.join(_sp, 'llama_cpp', 'lib'))
            _search_dirs += _glob.glob(os.path.join(_sp, 'ctranslate2', '*.dll'))
            _search_dirs.append(os.path.join(_sp, 'ctranslate2'))
            # ONNX Runtime library paths
            _search_dirs += _glob.glob(os.path.join(_sp, 'onnxruntime', 'capi'))

        import glob as _glob2
        for _ver in ['v12.1', 'v12.0', 'v12.2', 'v12.3', 'v12.4', 'v11.8', 'v11.7']:
            _cuda_bin = fr'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{_ver}\bin'
            if os.path.isdir(_cuda_bin):
                _search_dirs.append(_cuda_bin)
        _search_dirs.append(r'C:\Windows\System32')

        _unique_dirs = []
        for _d in _search_dirs:
            _abs_d = os.path.abspath(_d)
            if os.path.isdir(_abs_d) and _abs_d not in _unique_dirs:
                _unique_dirs.append(_abs_d)
        _search_dirs = _unique_dirs

        _loaded = []
        for _dll_name in _cuda_dll_names:
            for _d in _search_dirs:
                _dll_path = os.path.join(_d, _dll_name)
                if os.path.exists(_dll_path):
                    try:
                        _ctypes.CDLL(_dll_path)
                        _loaded.append(_dll_name)
                        logger.debug(f"Preloaded CUDA DLL: {_dll_path}")
                        break
                    except Exception as _e:
                        logger.debug(f"Failed to preload {_dll_path}: {_e}")

        if _loaded:
            logger.info(f"Preloaded CUDA DLLs: {_loaded}")
        else:
            logger.warning("No CUDA DLLs found to preload. GPU support may not work.")

        for _d in _search_dirs:
            if os.path.isdir(_d):
                try:
                    os.add_dll_directory(_d)
                except Exception:
                    pass
        os.environ['PATH'] = ';'.join(d for d in _search_dirs if os.path.isdir(d)) \
                             + ';' + os.environ.get('PATH', '')

    except Exception:
        logger.warning("CUDA DLL preload setup failed", exc_info=True)

# ==========================================
# THIRD-PARTY IMPORTS
# ==========================================
try:
    import numpy as np
    import sounddevice as sd
    import customtkinter as ctk
    from faster_whisper import WhisperModel
    import moonshine_onnx
    from llama_cpp import Llama
    from pythonosc import udp_client
    from huggingface_hub import hf_hub_download
except Exception:
    logger.critical("ライブラリのインポートに失敗しました", exc_info=True)
    sys.exit(1)

# ==========================================
# CONFIGURATION
# ==========================================
OSC_IP = "127.0.0.1"
OSC_PORT = 9000
WHISPER_MODEL_SIZE = "tiny"
LLAMA_REPO = "tencent/Hy-MT2-1.8B-GGUF"
LLAMA_FILE = "Hy-MT2-1.8B-Q4_K_M.gguf"
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 0.5
CONFIG_FILE = os.path.join(base_dir, "config.json")

STT_LANGUAGES = ["Auto (自動認識)", "Japanese (日本語)", "English (英語)", "Spanish (スペイン語)", "Chinese (中国語)", "Korean (韓国語)", "Russian (ロシア語)"]
MT_LANGUAGES = ["Japanese (日本語)", "English (英語)", "Spanish (スペイン語)", "Chinese (中国語)", "Korean (韓国語)", "Russian (ロシア語)"]

LANG_CODE_MAP = {
    "Auto (自動認識)": None,
    "Japanese (日本語)": "ja",
    "English (英語)": "en",
    "Spanish (スペイン語)": "es",
    "Chinese (中国語)": "zh",
    "Korean (韓国語)": "ko",
    "Russian (ロシア語)": "ru"
}

LANG_NAME_MAP = {
    "Japanese (日本語)": "Japanese",
    "English (英語)": "English",
    "Spanish (スペイン語)": "Spanish",
    "Chinese (中国語)": "Chinese",
    "Korean (韓国語)": "Korean",
    "Russian (ロシア語)": "Russian"
}

UI_LOCALIZATION = {
    "US": {
        "win_title": "Two-Way Realtime Translator (Moonshine High-Speed Edition)",
        "header": "Two-Way Realtime Translator\n(Moonshine + Whisper Hybrid)",
        "listen_frame": "【Partner's Voice (Listening / Moonshine)】",
        "speak_frame": "【Your Voice (Speaking / Whisper)】",
        "device": "Device:",
        "stt": "STT Lang:",
        "mt": "➡ Target:",
        "osc": "Send to VRChat ChatBox (OSC)",
        "start_btn": "Start (翻訳開始)",
        "stop_btn": "Stop (停止)",
        "init_log": "System started. Please select devices, languages, and click 'Start'.",
        "err_no_device": "Error: Configured devices not found. Check connection.",
        "initializing": "\n--- Initializing ---",
        "stopping": "Stopping...",
        "stopped": "--- Stopped ---",
        "llm_loading": "Loading Hy-MT2 (LLM)...",
        "gpu_loaded": "-> Loaded in GPU mode",
        "gpu_fail_cpu_retry": "-> GPU mode failed. Retrying in CPU mode... ",
        "cpu_loaded": "-> Loaded in CPU mode",
        "cpu_fail": "-> CPU mode also failed: ",
        "whisper_loading": "Loading Faster-Whisper...",
        "whisper_gpu": "-> Running in GPU (CUDA) mode",
        "whisper_cpu_fail": "-> GPU (CUDA) init failed. Running in CPU mode. ",
        "moonshine_loading_en": "Loading Moonshine (EN / Super-fast)...",
        "moonshine_loaded_en": "-> Loaded Moonshine (for English)",
        "moonshine_fail_en": "-> Moonshine EN init failed: ",
        "moonshine_prep_es": "Preparing Moonshine (ES / Super-fast) model...",
        "moonshine_loading_es": "Loading Moonshine (ES / Super-fast)...",
        "moonshine_loaded_es": "-> Loaded Moonshine (for Spanish)",
        "moonshine_fail_es": "-> Moonshine ES init failed: ",
        "ready": "--- Ready! Translation started ---",
        "partner": "Partner",
        "me": "Me",
        "translation": "Translation",
        "chatbox": "ChatBox"
    },
    "JP": {
        "win_title": "双方向リアルタイム翻訳 (Moonshine高速版)",
        "header": "双方向リアルタイム翻訳\n(Moonshine + Whisper ハイブリッド)",
        "listen_frame": "【相手の音声 (リスニング・英語推奨/Moonshine)】",
        "speak_frame": "【自分の音声 (スピーキング・日本語推奨/Whisper)】",
        "device": "デバイス:",
        "stt": "認識言語:",
        "mt": "➡ 翻訳先:",
        "osc": "VRChat ChatBox (OSC) に送信する",
        "start_btn": "翻訳開始 (Start)",
        "stop_btn": "停止 (Stop)",
        "init_log": "システムを起動しました。デバイスと言語を選択して「開始」を押してください。",
        "err_no_device": "エラー: 設定されたデバイスが見つかりません。接続を確認してください。",
        "initializing": "\n--- 初期化中 ---",
        "stopping": "停止処理を行っています...",
        "stopped": "--- 停止しました ---",
        "llm_loading": "Hy-MT2 (LLM) をロード中...",
        "gpu_loaded": "-> GPU モードでロードしました",
        "gpu_fail_cpu_retry": "-> GPU モード失敗。CPU モードで再試行します... ",
        "cpu_loaded": "-> CPU モードでロードしました",
        "cpu_fail": "-> CPU モードも失敗しました: ",
        "whisper_loading": "Faster-Whisper (スピーキング用) をロード中...",
        "whisper_gpu": "-> Faster-Whisper GPU モードでロードしました",
        "whisper_cpu_fail": "-> Faster-Whisper GPU初期化失敗。CPUモードで実行します。 ",
        "moonshine_loading_en": "Moonshine (英語音声認識・爆速) をロード中...",
        "moonshine_loaded_en": "-> Moonshine (英語用) をロードしました",
        "moonshine_fail_en": "-> Moonshine 英語初期化失敗: ",
        "moonshine_prep_es": "Moonshine (スペイン語音声認識・爆速) モデルを準備中...",
        "moonshine_loading_es": "Moonshine (スペイン語音声認識・爆速) をロード中...",
        "moonshine_loaded_es": "-> Moonshine (スペイン語用) をロードしました",
        "moonshine_fail_es": "-> Moonshine スペイン語初期化失敗: ",
        "ready": "--- 準備完了！ 翻訳を開始しました ---",
        "partner": "相手",
        "me": "自分",
        "translation": "翻訳",
        "chatbox": "ChatBox"
    },
    "ES": {
        "win_title": "Traductor en tiempo real bidireccional (Edición rápida Moonshine)",
        "header": "Traductor en tiempo real bidireccional\n(Híbrido Moonshine + Whisper)",
        "listen_frame": "【Voz del compañero (Escucha / Moonshine)】",
        "speak_frame": "【Tu voz (Habla / Whisper)】",
        "device": "Dispositivo:",
        "stt": "Idioma STT:",
        "mt": "➡ Destino:",
        "osc": "Enviar a VRChat ChatBox (OSC)",
        "start_btn": "Iniciar traducción (Start)",
        "stop_btn": "Detener (Stop)",
        "init_log": "Sistema iniciado. Seleccione los dispositivos, los idiomas y haga clic en 'Iniciar'.",
        "err_no_device": "Error: Dispositivos configurados no encontrados. Verifique la conexión.",
        "initializing": "\n--- Inicializando ---",
        "stopping": "Deteniendo...",
        "stopped": "--- Detenido ---",
        "llm_loading": "Cargando Hy-MT2 (LLM)...",
        "gpu_loaded": "-> Cargado en modo GPU",
        "gpu_fail_cpu_retry": "-> Error en GPU. Reintentando en modo CPU... ",
        "cpu_loaded": "-> Cargado en modo CPU",
        "cpu_fail": "-> También falló el modo CPU: ",
        "whisper_loading": "Cargando Faster-Whisper...",
        "whisper_gpu": "-> Ejecutándose en modo GPU (CUDA)",
        "whisper_cpu_fail": "-> Falló la inicialización de GPU (CUDA). Ejecutándose en modo CPU. ",
        "moonshine_loading_en": "Cargando Moonshine (EN / Superrápido)...",
        "moonshine_loaded_en": "-> Moonshine (para inglés) cargado",
        "moonshine_fail_en": "-> Falló la inicialización de Moonshine EN: ",
        "moonshine_prep_es": "Preparando el modelo Moonshine (ES / Superrápido)...",
        "moonshine_loading_es": "Cargando Moonshine (ES / Superrápido)...",
        "moonshine_loaded_es": "-> Moonshine (para español) cargado",
        "moonshine_fail_es": "-> Falló la inicialización de Moonshine ES: ",
        "ready": "--- ¡Preparado! Traducción iniciada ---",
        "partner": "Compañero",
        "me": "Yo",
        "translation": "Traducción",
        "chatbox": "ChatBox"
    },
    "RU": {
        "win_title": "Двусторонний переводчик реального времени (Быстрая версия Moonshine)",
        "header": "Двусторонний переводчик реального времени\n(Гибрид Moonshine + Whisper)",
        "listen_frame": "【Голос собеседника (Прослушивание / Moonshine)】",
        "speak_frame": "【Ваш голос (Говорение / Whisper)】",
        "device": "Устройство:",
        "stt": "Язык STT:",
        "mt": "➡ Перевод:",
        "osc": "Отправлять в VRChat ChatBox (OSC)",
        "start_btn": "Начать перевод (Start)",
        "stop_btn": "Остановить (Stop)",
        "init_log": "Система запущена. Выберите устройства, языки и нажмите «Начать».",
        "err_no_device": "Ошибка: Настроенные устройства не найдены. Проверьте подключение.",
        "initializing": "\n--- Инициализация ---",
        "stopping": "Остановка...",
        "stopped": "--- Останавливать ---",
        "llm_loading": "Загрузка Hy-MT2 (LLM)...",
        "gpu_loaded": "-> Загружено в режиме GPU",
        "gpu_fail_cpu_retry": "-> Ошибка GPU. Повторная попытка в режиме CPU... ",
        "cpu_loaded": "-> Загруровано в режиме CPU",
        "cpu_fail": "-> Ошибка также и в режиме CPU: ",
        "whisper_loading": "Загрузка Faster-Whisper...",
        "whisper_gpu": "-> Работает в режиме GPU (CUDA)",
        "whisper_cpu_fail": "-> Ошибка инициализации GPU (CUDA). Работает в режиме CPU. ",
        "moonshine_loading_en": "Загрузка Moonshine (EN / Супербыстро)...",
        "moonshine_loaded_en": "-> Moonshine (для английского) загружен",
        "moonshine_fail_en": "-> Ошибка инициализации Moonshine EN: ",
        "moonshine_prep_es": "Подготовка модели Moonshine (ES / Супербыстро)...",
        "moonshine_loading_es": "Загрузка Moonshine (ES /  Супербыстро)...",
        "moonshine_loaded_es": "-> Moonshine (для испанского) загружен",
        "moonshine_fail_es": "-> Ошибка инициализации Moonshine ES: ",
        "ready": "--- Готово! Перевод начат ---",
        "partner": "Собеседник",
        "me": "Я",
        "translation": "Перевод",
        "chatbox": "ChatBox"
    },
    "中": {
        "win_title": "双向实时翻译 (Moonshine 高速版)",
        "header": "双向实时翻译\n(Moonshine + Whisper 混合驱动)",
        "listen_frame": "【对方语音 (听译 / Moonshine)】",
        "speak_frame": "【我的语音 (说译 / Whisper)】",
        "device": "设备:",
        "stt": "识别语言:",
        "mt": "➡ 翻译至:",
        "osc": "发送至 VRChat ChatBox (OSC)",
        "start_btn": "开始翻译 (Start)",
        "stop_btn": "停止翻译 (Stop)",
        "init_log": "系统已启动。请选择设备与语言，然后点击“开始”。",
        "err_no_device": "错误: 未找到配置的设备。请检查连接。",
        "initializing": "\n--- 初始化中 ---",
        "stopping": "正在停止...",
        "stopped": "--- 已停止 ---",
        "llm_loading": "正在加载 Hy-MT2 (LLM)...",
        "gpu_loaded": "-> 已在 GPU 模式下加载",
        "gpu_fail_cpu_retry": "-> GPU 模式启动失败。正在尝试 CPU 模式... ",
        "cpu_loaded": "-> 已在 CPU 模式下加载",
        "cpu_fail": "-> CPU 模式加载也失败: ",
        "whisper_loading": "正在加载 Faster-Whisper...",
        "whisper_gpu": "-> 正在 GPU (CUDA) 模式下运行",
        "whisper_cpu_fail": "-> GPU (CUDA) 初始化失败。正在 CPU 模式下运行。 ",
        "moonshine_loading_en": "正在加载 Moonshine (英文 / 超快速)...",
        "moonshine_loaded_en": "-> 英文 Moonshine 已加载",
        "moonshine_fail_en": "-> 英文 Moonshine 初始化失败: ",
        "moonshine_prep_es": "正在准备 Moonshine (西班牙文 / 超快速) 模型...",
        "moonshine_loading_es": "正在加载 Moonshine (西班牙文 / 超快速)...",
        "moonshine_loaded_es": "-> 西班牙文 Moonshine 已加载",
        "moonshine_fail_es": "-> 西班牙文 Moonshine 初始化失败: ",
        "ready": "--- 准备就绪！翻译已开始 ---",
        "partner": "对方",
        "me": "自己",
        "translation": "翻译",
        "chatbox": "ChatBox"
    },
    "한": {
        "win_title": "양방향 실시간 번역기 (Moonshine 고속 에디션)",
        "header": "양방향 실시간 번역기\n(Moonshine + Whisper 하이브리드)",
        "listen_frame": "【상대방 음성 (듣기 / Moonshine)】",
        "speak_frame": "【내 음성 (말하기 / Whisper)】",
        "device": "장치:",
        "stt": "인식 언어:",
        "mt": "➡ 번역 대상:",
        "osc": "VRChat ChatBox (OSC)로 전송",
        "start_btn": "번역 시작 (Start)",
        "stop_btn": "정지 (Stop)",
        "init_log": "시스템이 시작되었습니다. 장치와 언어를 선택한 다음 '시작'을 누르세요.",
        "err_no_device": "오류: 설정된 장치를 찾을 수 없습니다. 연결을 확인하세요.",
        "initializing": "\n--- 초기화 중 ---",
        "stopping": "정지 처리 중...",
        "stopped": "--- 정지됨 ---",
        "llm_loading": "Hy-MT2 (LLM) 로드 중...",
        "gpu_loaded": "-> GPU 모드로 로드되었습니다",
        "gpu_fail_cpu_retry": "-> GPU 모드 실패. CPU 모드로 재시도합니다... ",
        "cpu_loaded": "-> CPU 모드로 로드되었습니다",
        "cpu_fail": "-> CPU 모드도 실패했습니다: ",
        "whisper_loading": "Faster-Whisper 로드 중...",
        "whisper_gpu": "-> GPU (CUDA) 모드로 실행 중",
        "whisper_cpu_fail": "-> GPU (CUDA) 초기화 실패. CPU 모드로 실행합니다. ",
        "moonshine_loading_en": "Moonshine (영어 / 초고속) 로드 중...",
        "moonshine_loaded_en": "-> 영어용 Moonshine이 로드되었습니다",
        "moonshine_fail_en": "-> 영어 Moonshine 초기화 실패: ",
        "moonshine_prep_es": "Moonshine (스페인어 / 초고속) 모델 준비 중...",
        "moonshine_loading_es": "Moonshine (스페인어 / 초고속) 로드 중...",
        "moonshine_loaded_es": "-> 스페인어용 Moonshine이 로드되었습니다",
        "moonshine_fail_es": "-> 스페인어 Moonshine 초기화 실패: ",
        "ready": "--- 준비 완료! 번역이 시작되었습니다 ---",
        "partner": "상대방",
        "me": "자신",
        "translation": "번역",
        "chatbox": "ChatBox"
    }
}

# ==========================================
# HELPER CLASSES
def get_device_name_only(combo_value):
    if "]" in combo_value:
        return combo_value.split("]", 1)[1].strip()
    return combo_value

def find_best_device_match(saved_name, combo_values):
    if not saved_name:
        return combo_values[0] if combo_values else ""
    if saved_name in combo_values:
        return saved_name
    
    clean_saved = saved_name
    if clean_saved.startswith("[") and "]" in clean_saved:
        clean_saved = clean_saved.split("]", 1)[1].strip()
        
    for val in combo_values:
        clean_val = val
        if clean_val.startswith("[") and "]" in clean_val:
            clean_val = clean_val.split("]", 1)[1].strip()
        if clean_saved.lower() in clean_val.lower() or clean_val.lower() in clean_saved.lower():
            return val
            
    return combo_values[0] if combo_values else ""

def load_config():
    default_config = {
        "speaker_device": "", 
        "mic_device": "",
        "listen_stt": "English (英語)",
        "listen_mt": "Japanese (日本語)",
        "speak_stt": "Japanese (日本語)",
        "speak_mt": "English (英語)",
        "use_osc": True,
        "ui_lang": "JP"
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                default_config.update(loaded)
        except:
            pass
    return default_config

def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

MIN_MODEL_SIZE_BYTES = 500 * 1024 * 1024  # 500MB未満は破損とみなす

def download_model_if_needed(log_callback):
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, LLAMA_FILE)

    needs_download = False
    if not os.path.exists(model_path):
        needs_download = True
    else:
        size = os.path.getsize(model_path)
        if size < MIN_MODEL_SIZE_BYTES:
            log_callback(f"モデルファイルが破損しています ({size//1024//1024}MB)。再ダウンロードします...")
            logger.warning(f"Model file too small ({size} bytes), re-downloading")
            os.remove(model_path)
            needs_download = True
        else:
            log_callback(f"モデル確認OK: {model_path} ({size//1024//1024}MB)")

    if needs_download:
        log_callback(f"[{LLAMA_FILE}] のダウンロードを開始します (約1GB)...")
        logger.info(f"Downloading {LLAMA_REPO}/{LLAMA_FILE} to {models_dir}")
        downloaded_path = hf_hub_download(repo_id=LLAMA_REPO, filename=LLAMA_FILE, local_dir=models_dir)
        size = os.path.getsize(downloaded_path)
        log_callback(f"ダウンロード完了: {downloaded_path} ({size//1024//1024}MB)")
        logger.info(f"Download complete: {downloaded_path} ({size} bytes)")
        model_path = downloaded_path
    return model_path

def download_moonshine_es_if_needed(log_callback):
    models_dir = os.path.join(base_dir, "models", "moonshine-es")
    os.makedirs(models_dir, exist_ok=True)
    
    files = ["encoder_model.onnx", "decoder_model_merged.onnx"]
    repo_id = "UsefulSensors/moonshine-es"
    subfolder = "onnx/merged/base/float"
    
    downloaded_paths = []
    for f in files:
        target_path = os.path.join(models_dir, f)
        if not os.path.exists(target_path):
            log_callback(f"[{repo_id}] から {f} をダウンロードしています...")
            logger.info(f"Downloading {f} from {repo_id}")
            path = hf_hub_download(repo_id=repo_id, filename=f, subfolder=subfolder, local_dir=models_dir)
            downloaded_paths.append(path)
        else:
            downloaded_paths.append(target_path)
            
    # local_dir の保存先（subfolder以下）から直下に配置
    for f in files:
        src = os.path.join(models_dir, subfolder, f)
        dst = os.path.join(models_dir, f)
        if os.path.exists(src) and not os.path.exists(dst):
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                logger.error(f"Failed to copy {f}: {e}")
            
    return models_dir

class AudioRecorder:
    def __init__(self, device_idx, name, sample_rate=16000, silence_duration=1.0, threshold=0.01, max_duration=8.0):
        self.device_idx = device_idx
        self.name = name
        self.target_sample_rate = sample_rate
        self.sample_rate = sample_rate  # Will be adjusted if 16000Hz is not supported
        self.silence_duration = silence_duration
        self.threshold = threshold
        self.max_duration = max_duration
        self.audio_chunks = []
        self.silence_chunks = 0
        self.is_recording_active = False
        self.output_queue = queue.Queue()
        self.stream = None
        
    def resample(self, audio, orig_sr, target_sr):
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        num_target_samples = int(duration * target_sr)
        return np.interp(
            np.linspace(0, len(audio), num_target_samples, endpoint=False),
            np.arange(len(audio)),
            audio
        ).astype(np.float32)

    def callback(self, indata, frames, time_info, status):
        rms = np.sqrt(np.mean(indata**2))
        
        if rms > self.threshold:
            self.is_recording_active = True
            self.silence_chunks = 0
            self.audio_chunks.append(indata[:, 0].copy())
        else:
            if self.is_recording_active:
                self.silence_chunks += 1
                self.audio_chunks.append(indata[:, 0].copy())
                silence_limit = int((self.silence_duration * self.sample_rate) / frames)
                if self.silence_chunks > silence_limit:
                    audio_data = np.concatenate(self.audio_chunks, axis=0)
                    resampled = self.resample(audio_data, self.sample_rate, self.target_sample_rate)
                    self.output_queue.put(resampled)
                    self.audio_chunks = []
                    self.is_recording_active = False
                    self.silence_chunks = 0
                    return

        # 強制区切り (最大録音時間を超えた場合)
        if self.is_recording_active:
            total_samples = sum(len(c) for c in self.audio_chunks)
            if total_samples >= self.max_duration * self.sample_rate:
                audio_data = np.concatenate(self.audio_chunks, axis=0)
                resampled = self.resample(audio_data, self.sample_rate, self.target_sample_rate)
                self.output_queue.put(resampled)
                self.audio_chunks = []
                self.is_recording_active = False
                self.silence_chunks = 0

    def start(self, log_callback):
        # 1. Try target sample rate (16000Hz) first
        try:
            self.stream = sd.InputStream(device=self.device_idx,
                                         channels=1,
                                         samplerate=self.target_sample_rate,
                                         callback=self.callback)
            self.stream.start()
            self.sample_rate = self.target_sample_rate
            log_callback(f"[{self.name}] 16000Hz で録音を開始しました。")
            return
        except Exception as e:
            logger.warning(f"Failed to start stream at 16000Hz on device {self.device_idx}: {e}")

        # 2. Try default sample rate as fallback
        try:
            dev_info = sd.query_devices(self.device_idx)
            default_sr = int(dev_info['default_samplerate'])
            self.stream = sd.InputStream(device=self.device_idx,
                                         channels=1,
                                         samplerate=default_sr,
                                         callback=self.callback)
            self.stream.start()
            self.sample_rate = default_sr
            log_callback(f"[{self.name}] デフォルトレート {default_sr}Hz で録音を開始しました（自動リサンプリング有効）。")
        except Exception as e2:
            logger.critical(f"Failed to start stream at default rate on device {self.device_idx}: {e2}")
            raise e2

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()


def is_hallucination_or_excessive_repeat(text):
    if not text:
        return False
    # Check 1: Length is too short, no need to check
    if len(text) < 25:
        return False
        
    # Check 2: Low unique character variety in long text (e.g. "hahahaha...", "はっはっはっ...")
    unique_chars = set(text)
    if len(unique_chars) <= 3:
        return True
        
    # Check 3: Ratio of unique characters is extremely low in long text
    if len(text) >= 40 and (len(unique_chars) / len(text)) < 0.08:
        return True
        
    # Check 4: Substring repetitions (e.g. "エッティエッティエッティ...")
    # Try to find repeating patterns of length 2 to 6
    for pattern_len in range(2, 7):
        pattern = text[:pattern_len]
        # If the text is mostly filled with this pattern
        expected_repeats = len(text) // pattern_len
        if expected_repeats >= 4:
            matched_len = 0
            for i in range(expected_repeats):
                if text[i*pattern_len : (i+1)*pattern_len] == pattern:
                    matched_len += pattern_len
                else:
                    break
            if matched_len >= 20 and (matched_len / len(text)) > 0.8:
                return True
                
    return False

def translate_text(llm, text, source_lang, target_lang, lock=None):
    if source_lang == target_lang:
        return text
    prompt_text = f"Translate the following {source_lang} text to {target_lang}. Output ONLY the translated text without any explanation or quotes.\n\nText: {text}\n\nTranslation:"
    
    if lock:
        with lock:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful and highly skilled translator."},
                    {"role": "user", "content": prompt_text}
                ],
                max_tokens=256,
                temperature=0.1
            )
    else:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are a helpful and highly skilled translator."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=256,
            temperature=0.1
        )
    result = response['choices'][0]['message']['content'].strip()
    return result

# ==========================================
# GUI APPLICATION
# ==========================================
class VRChatTranslatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Two-Way Realtime Translator (Moonshine High-Speed Edition)")
        self.geometry("750x700")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        self.is_running = False
        self.whisper_model = None
        self.moonshine_model_en = None
        self.moonshine_model_es = None
        self.llm = None
        self.llm_lock = threading.Lock()
        self.speaker_recorder = None
        self.mic_recorder = None
        
        self.devices = []
        self.device_names = []
        self.populate_devices()
        self.config = load_config()
        
        self.setup_ui()
        self.update_ui_text()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def populate_devices(self):
        devs = sd.query_devices()
        hostapis = sd.query_hostapis()
        
        self.devices_input = []   # list of (index, name, hostapi)
        self.devices_output = []  # list of (index, name, hostapi)
        
        self.combo_speaker_values = []
        self.combo_mic_values = []
        
        for i, dev in enumerate(devs):
            api_name = hostapis[dev['hostapi']]['name']
            # Limit to MME and Windows WASAPI to avoid duplicate clutter and WDM-KS issues
            if api_name not in ["MME", "Windows WASAPI"]:
                continue
                
            raw_name = dev['name']
            clean_name = raw_name
            try:
                # Fix encoding mismatch commonly returned by sounddevice on Windows
                clean_name = raw_name.encode('latin1').decode('cp932')
            except Exception:
                pass
                
            display_name = f"[{api_name}] {clean_name}"
            
            # Input devices
            if dev['max_input_channels'] > 0:
                self.devices_input.append((i, clean_name, api_name))
                self.combo_mic_values.append(display_name)
                
            # Output devices
            if dev['max_output_channels'] > 0:
                self.devices_output.append((i, clean_name, api_name))
                self.combo_speaker_values.append(display_name)
                
        # Also add all input devices to speaker combo values so users can choose direct loopback input or cable outputs
        for i, clean_name, api_name in self.devices_input:
            display_name = f"[{api_name}] {clean_name}"
            if display_name not in self.combo_speaker_values:
                self.combo_speaker_values.append(display_name)
                
    def setup_ui(self):
        # Header
        self.header = ctk.CTkLabel(self, text="Two-Way Realtime Translator\n(Moonshine + Whisper Hybrid)", font=ctk.CTkFont(size=22, weight="bold"))
        self.header.pack(pady=(15, 5))
        
        # Language Switch Buttons Container at Top Right
        self.lang_btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.lang_btn_frame.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)
        
        self.lang_buttons = {}
        languages = ["US", "JP", "ES", "RU", "中", "한"]
        for lang in languages:
            btn = ctk.CTkButton(
                self.lang_btn_frame, text=lang, width=32, height=20,
                corner_radius=4, font=ctk.CTkFont(size=10, weight="bold"),
                fg_color="#4A5568", hover_color="#2D3748",
                command=lambda l=lang: self.change_ui_lang(l)
            )
            btn.pack(side="left", padx=2)
            self.lang_buttons[lang] = btn
        
        # --- Listen Settings (相手の音声) ---
        self.frame_listen = ctk.CTkFrame(self)
        self.frame_listen.pack(pady=5, padx=20, fill="x")
        self.lbl_listen_title = ctk.CTkLabel(self.frame_listen, text="【相手の音声 (リスニング・英語固定/Moonshine)】", font=ctk.CTkFont(weight="bold"))
        self.lbl_listen_title.grid(row=0, column=0, columnspan=4, pady=(10,5), padx=10, sticky="w")
        
        self.lbl_listen_device = ctk.CTkLabel(self.frame_listen, text="デバイス:")
        self.lbl_listen_device.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.combo_speaker = ctk.CTkComboBox(self.frame_listen, values=self.combo_speaker_values, width=300, command=self.on_setting_changed)
        self.combo_speaker.grid(row=1, column=1, columnspan=3, padx=10, pady=5, sticky="w")
        spk_name = self.config.get("speaker_device", "")
        matched_spk = find_best_device_match(spk_name, self.combo_speaker_values)
        if matched_spk:
            self.combo_speaker.set(matched_spk)
            
        self.lbl_listen_stt = ctk.CTkLabel(self.frame_listen, text="認識言語:")
        self.lbl_listen_stt.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.combo_listen_stt = ctk.CTkComboBox(self.frame_listen, values=STT_LANGUAGES, width=150, command=self.on_setting_changed)
        self.combo_listen_stt.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        if self.config.get("listen_stt") in STT_LANGUAGES:
            self.combo_listen_stt.set(self.config["listen_stt"])
        else:
            self.combo_listen_stt.set("English (英語)")
            
        self.lbl_listen_mt = ctk.CTkLabel(self.frame_listen, text="➡ 翻訳先:")
        self.lbl_listen_mt.grid(row=2, column=2, padx=10, pady=5, sticky="w")
        self.combo_listen_mt = ctk.CTkComboBox(self.frame_listen, values=MT_LANGUAGES, width=150, command=self.on_setting_changed)
        self.combo_listen_mt.grid(row=2, column=3, padx=10, pady=5, sticky="w")
        if self.config["listen_mt"] in MT_LANGUAGES: self.combo_listen_mt.set(self.config["listen_mt"])
 
        # --- Speak Settings (自分の音声) ---
        self.frame_speak = ctk.CTkFrame(self)
        self.frame_speak.pack(pady=5, padx=20, fill="x")
        self.lbl_speak_title = ctk.CTkLabel(self.frame_speak, text="【自分の音声 (スピーキング・日本語推奨/Whisper)】", font=ctk.CTkFont(weight="bold"))
        self.lbl_speak_title.grid(row=0, column=0, columnspan=4, pady=(10,5), padx=10, sticky="w")
        
        self.lbl_speak_device = ctk.CTkLabel(self.frame_speak, text="デバイス:")
        self.lbl_speak_device.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.combo_mic = ctk.CTkComboBox(self.frame_speak, values=self.combo_mic_values, width=300, command=self.on_setting_changed)
        self.combo_mic.grid(row=1, column=1, columnspan=3, padx=10, pady=5, sticky="w")
        mic_name = self.config.get("mic_device", "")
        matched_mic = find_best_device_match(mic_name, self.combo_mic_values)
        if matched_mic:
            self.combo_mic.set(matched_mic)
            
        self.lbl_speak_stt = ctk.CTkLabel(self.frame_speak, text="認識言語:")
        self.lbl_speak_stt.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.combo_speak_stt = ctk.CTkComboBox(self.frame_speak, values=STT_LANGUAGES, width=150, command=self.on_setting_changed)
        self.combo_speak_stt.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        if self.config["speak_stt"] in STT_LANGUAGES: self.combo_speak_stt.set(self.config["speak_stt"])
            
        self.lbl_speak_mt = ctk.CTkLabel(self.frame_speak, text="➡ 翻訳先:")
        self.lbl_speak_mt.grid(row=2, column=2, padx=10, pady=5, sticky="w")
        self.combo_speak_mt = ctk.CTkComboBox(self.frame_speak, values=MT_LANGUAGES, width=150, command=self.on_setting_changed)
        self.combo_speak_mt.grid(row=2, column=3, padx=10, pady=5, sticky="w")
        if self.config["speak_mt"] in MT_LANGUAGES: self.combo_speak_mt.set(self.config["speak_mt"])
            
        # VRChat OSC Send Option
        self.check_osc = ctk.CTkCheckBox(self.frame_speak, text="VRChat ChatBox (OSC) に送信する", command=self.on_setting_changed)
        self.check_osc.grid(row=3, column=0, columnspan=4, padx=10, pady=10, sticky="w")
        if self.config.get("use_osc", True):
            self.check_osc.select()
        else:
            self.check_osc.deselect()
            
        # --- Buttons ---
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(pady=10)
        
        self.btn_start = ctk.CTkButton(self.btn_frame, text="翻訳開始 (Start)", font=ctk.CTkFont(weight="bold"), fg_color="#2FA572", hover_color="#108552", command=self.on_start)
        self.btn_start.pack(side="left", padx=10)
        
        self.btn_stop = ctk.CTkButton(self.btn_frame, text="停止 (Stop)", font=ctk.CTkFont(weight="bold"), fg_color="#D35B58", hover_color="#B33B38", state="disabled", command=self.on_stop)
        self.btn_stop.pack(side="left", padx=10)
        
        # --- Log Box ---
        self.log_box = ctk.CTkTextbox(self, width=710, height=200, font=ctk.CTkFont(size=14))
        self.log_box.pack(pady=(5, 10), padx=20, fill="both", expand=True)
        self.log_box.configure(state="disabled")
        
        self.log_localized("init_log")
 
    def log(self, message):
        logger.info(message)
        def _append():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", message + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.after(0, _append)

    def log_localized(self, key, suffix=""):
        lang = self.config.get("ui_lang", "JP")
        loc = UI_LOCALIZATION.get(lang, UI_LOCALIZATION["JP"])
        msg = loc.get(key, key) + suffix
        self.log(msg)

    def change_ui_lang(self, lang):
        self.config["ui_lang"] = lang
        save_config(self.config)
        self.update_ui_text()

    def update_ui_text(self):
        lang = self.config.get("ui_lang", "JP")
        loc = UI_LOCALIZATION.get(lang, UI_LOCALIZATION["JP"])
        
        self.title(loc["win_title"])
        self.header.configure(text=loc["header"])
        self.lbl_listen_title.configure(text=loc["listen_frame"])
        self.lbl_listen_device.configure(text=loc["device"])
        self.lbl_listen_stt.configure(text=loc["stt"])
        self.lbl_listen_mt.configure(text=loc["mt"])
        
        self.lbl_speak_title.configure(text=loc["speak_frame"])
        self.lbl_speak_device.configure(text=loc["device"])
        self.lbl_speak_stt.configure(text=loc["stt"])
        self.lbl_speak_mt.configure(text=loc["mt"])
        
        self.check_osc.configure(text=loc["osc"])
        self.btn_start.configure(text=loc["start_btn"])
        self.btn_stop.configure(text=loc["stop_btn"])
        
        for l, btn in self.lang_buttons.items():
            if l == lang:
                btn.configure(fg_color="#3B82F6", hover_color="#1D4ED8")
            else:
                btn.configure(fg_color="#4A5568", hover_color="#2D3748")
 
    def lock_ui(self, lock):
        state = "disabled" if lock else "normal"
        self.combo_speaker.configure(state=state)
        self.combo_mic.configure(state=state)
        self.combo_listen_stt.configure(state=state)
        self.combo_listen_mt.configure(state=state)
        self.combo_speak_stt.configure(state=state)
        self.combo_speak_mt.configure(state=state)
        self.check_osc.configure(state=state)
        self.btn_start.configure(state=state)
        self.btn_stop.configure(state="normal" if lock else "disabled")
 
    def on_setting_changed(self, event=None):
        self.config["speaker_device"] = self.combo_speaker.get()
        self.config["mic_device"] = self.combo_mic.get()
        self.config["listen_stt"] = self.combo_listen_stt.get()
        self.config["listen_mt"] = self.combo_listen_mt.get()
        self.config["speak_stt"] = self.combo_speak_stt.get()
        self.config["speak_mt"] = self.combo_speak_mt.get()
        self.config["use_osc"] = bool(self.check_osc.get())
        save_config(self.config)
 
    def on_closing(self):
        self.on_setting_changed()
        if self.is_running:
            self.on_stop()
        self.destroy()
 
    def on_start(self):
        self.on_setting_changed()
        
        # Get selected full names
        spk_val = self.combo_speaker.get()
        mic_val = self.combo_mic.get()
        
        logger.info(f"on_start - Speaker selected: {spk_val}")
        logger.info(f"on_start - Mic selected: {mic_val}")
        
        # Parse API name and device name
        def parse_combo_val(val):
            if val.startswith("[") and "]" in val:
                parts = val.split("]", 1)
                api = parts[0][1:].strip()
                name = parts[1].strip()
                return api, name
            return None, val
            
        spk_api, spk_name = parse_combo_val(spk_val)
        mic_api, mic_name = parse_combo_val(mic_val)
        
        self.speaker_idx = None
        self.mic_idx = None
        
        # 1. Resolve mic index (input device)
        for i, name, api in self.devices_input:
            if name == mic_name and api == mic_api:
                self.mic_idx = i
                break
                
        # Fallback for mic: if api didn't match, match only by clean name
        if self.mic_idx is None:
            for i, name, api in self.devices_input:
                if name == mic_name:
                    self.mic_idx = i
                    logger.info(f"Mic resolved via fallback (name-only): {name} -> index {i}")
                    break
                
        # 2. Resolve speaker index
        # First check if the selected speaker is direct input (like CABLE Output or Stereo Mix)
        for i, name, api in self.devices_input:
            if name == spk_name and api == spk_api:
                self.speaker_idx = i
                break
                
        # Fallback for speaker direct input (name-only match)
        if self.speaker_idx is None:
            for i, name, api in self.devices_input:
                if name == spk_name:
                    self.speaker_idx = i
                    logger.info(f"Speaker resolved direct input via fallback (name-only): {spk_name} -> index {i}")
                    break
                
        # If not, it is an output device (like Speakers). We map it to its loopback input device.
        if self.speaker_idx is None:
            target_loopback_names = []
            if "cable input" in spk_name.lower():
                # VB-Cable output mapping
                target_loopback_names.append(spk_name.lower().replace("cable input", "cable output"))
            
            # WASAPI Loopback name mappings
            target_loopback_names.append(spk_name.lower() + " (loopback)")
            target_loopback_names.append(spk_name.lower() + " (ループバック)")
            
            # First try matching target loopback names with API name
            for i, name, api in self.devices_input:
                if api == "Windows WASAPI" or api == spk_api:
                    if name.lower() in target_loopback_names or any(t in name.lower() for t in target_loopback_names):
                        self.speaker_idx = i
                        break
                        
            # Fallback loopback search ignoring API name
            if self.speaker_idx is None:
                for i, name, api in self.devices_input:
                    if name.lower() in target_loopback_names or any(t in name.lower() for t in target_loopback_names):
                        self.speaker_idx = i
                        logger.info(f"Speaker loopback resolved via fallback (ignoring API): {name} -> index {i}")
                        break
                        
            # Fallback containing speaker name (with API match)
            if self.speaker_idx is None:
                for i, name, api in self.devices_input:
                    if spk_name.lower() in name.lower() and (api == "Windows WASAPI" or api == spk_api):
                        self.speaker_idx = i
                        break
                        
            # Fallback containing speaker name (ignoring API match)
            if self.speaker_idx is None:
                for i, name, api in self.devices_input:
                    if spk_name.lower() in name.lower():
                        self.speaker_idx = i
                        logger.info(f"Speaker resolved containing fallback (ignoring API): {name} -> index {i}")
                        break
                        
        if self.speaker_idx is None or self.mic_idx is None:
            # Try to fallback to system default input device
            try:
                import sounddevice as sd
                default_input_idx = sd.default.device[0]
                if default_input_idx >= 0:
                    if self.speaker_idx is None:
                        self.speaker_idx = default_input_idx
                        logger.warning(f"Speaker loopback resolution failed. Falling back to default input index {default_input_idx}")
                        self.log("警告: スピーカー録音デバイスの解決に失敗したため、システム規定の入力を使用します。")
                    if self.mic_idx is None:
                        self.mic_idx = default_input_idx
                        logger.warning(f"Mic resolution failed. Falling back to default input index {default_input_idx}")
                        self.log("警告: マイクデバイスの解決に失敗したため、システム規定の入力を使用します。")
            except Exception as fe:
                logger.error(f"Fallback resolution failed: {fe}")

        if self.speaker_idx is None or self.mic_idx is None:
            err_details = []
            if self.speaker_idx is None:
                err_details.append(f"Speaker ({spk_val}) not resolved")
            if self.mic_idx is None:
                err_details.append(f"Mic ({mic_val}) not resolved")
            logger.error(f"Device resolution failed: {', '.join(err_details)}")
            self.log(f"-> {', '.join(err_details)}")
            self.log_localized("err_no_device")
            return
 
        self.lock_ui(True)
        self.btn_stop.configure(state="disabled") # Temporary while loading
        self.log_localized("initializing")
        
        self.is_running = True
        threading.Thread(target=self.init_and_run, daemon=True).start()
 
    def on_stop(self):
        self.is_running = False
        self.log_localized("stopping")
        if self.speaker_recorder:
            self.speaker_recorder.stop()
        if self.mic_recorder:
            self.mic_recorder.stop()
            
        self.lock_ui(False)
        self.log_localized("stopped")
 
    def init_and_run(self):
        try:
            # 1. Download Model
            model_path = download_model_if_needed(self.log)
            if not self.is_running: return
            
            # 2. Load LLM
            if self.llm is None:
                model_size_mb = os.path.getsize(model_path) // 1024 // 1024
                logger.info(f"Model file: {model_path} ({model_size_mb} MB)")
                try:
                    with open(model_path, 'rb') as _f:
                        _magic = _f.read(4)
                    if _magic != b'GGUF':
                        self.log_localized("gpu_fail_cpu_retry", f" (Invalid GGUF)")
                        self.after(0, self.on_stop)
                        return
                except Exception as _ve:
                    self.log_localized("cpu_fail", f" ({_ve})")
                    self.after(0, self.on_stop)
                    return
 
                self.log_localized("llm_loading")
                try:
                    self.llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=2048, verbose=False)
                    self.log_localized("gpu_loaded")
                except Exception as gpu_e:
                    self.log_localized("gpu_fail_cpu_retry", f"({gpu_e})")
                    try:
                        self.llm = Llama(model_path=model_path, n_gpu_layers=0, n_ctx=2048, verbose=True)
                        self.log_localized("cpu_loaded")
                    except Exception as cpu_e:
                        self.log_localized("cpu_fail", f"{cpu_e}")
                        raise cpu_e
            if not self.is_running: return
                
            # 3. Load Speech-to-Text Engines
            listen_stt_val = self.config.get("listen_stt", "English (英語)")
            if listen_stt_val == "English (英語)":
                if self.moonshine_model_en is None:
                    self.log_localized("moonshine_loading_en")
                    try:
                        self.moonshine_model_en = moonshine_onnx.MoonshineOnnxModel(model_name="useful-sensors/moonshine/tiny")
                        self.log_localized("moonshine_loaded_en")
                    except Exception as e:
                        self.log_localized("moonshine_fail_en", f"{e}")
                        raise e
            elif listen_stt_val == "Spanish (スペイン語)":
                if self.moonshine_model_es is None:
                    self.log_localized("moonshine_prep_es")
                    try:
                        es_dir = download_moonshine_es_if_needed(self.log)
                        self.log_localized("moonshine_loading_es")
                        self.moonshine_model_es = moonshine_onnx.MoonshineOnnxModel(models_dir=es_dir, model_name="moonshine-es-base")
                        self.log_localized("moonshine_loaded_es")
                    except Exception as e:
                        self.log_localized("moonshine_fail_es", f"{e}")
                        raise e
 
            # Load Faster-Whisper (Japanese STT)
            if self.whisper_model is None:
                self.log_localized("whisper_loading")
                try:
                    self.whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
                    self.log_localized("whisper_gpu")
                except Exception as e:
                    self.log_localized("whisper_cpu_fail", f"({e})")
                    self.whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=1, num_workers=1)
            if not self.is_running: return
                    
            # 4. Start Recorders
            self.speaker_recorder = AudioRecorder(self.speaker_idx, "相手の音声", sample_rate=SAMPLE_RATE, threshold=SILENCE_THRESHOLD)
            self.mic_recorder = AudioRecorder(self.mic_idx, "自分の音声", sample_rate=SAMPLE_RATE, threshold=SILENCE_THRESHOLD)
            
            self.speaker_recorder.start(self.log)
            self.mic_recorder.start(self.log)
            
            self.osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
            
            self.log_localized("ready")
            self.after(0, lambda: self.btn_stop.configure(state="normal"))
            
            # 5. Start Processing Threads
            t1 = threading.Thread(target=self.pipeline_listen, daemon=True)
            t2 = threading.Thread(target=self.pipeline_speak, daemon=True)
            t1.start()
            t2.start()
            
        except Exception as e:
            logger.exception("Initialization error")
            self.log(f"Error: {e}")
            self.after(0, self.on_stop)
 
    def pipeline_listen(self):
        mt_target = LANG_NAME_MAP[self.config["listen_mt"]]
        
        while self.is_running:
            try:
                audio_data = self.speaker_recorder.output_queue.get(timeout=1.0)
                if not self.is_running: break
                
                listen_stt_val = self.config.get("listen_stt", "English (英語)")
                if listen_stt_val == "English (英語)":
                    text_list = moonshine_onnx.transcribe(audio_data, model=self.moonshine_model_en)
                    text = "".join(text_list).strip()
                    source_lang_name = "English"
                elif listen_stt_val == "Spanish (スペイン語)":
                    text_list = moonshine_onnx.transcribe(audio_data, model=self.moonshine_model_es)
                    text = "".join(text_list).strip()
                    source_lang_name = "Spanish"
                else:
                    stt_lang_code = LANG_CODE_MAP.get(listen_stt_val)
                    segments, info = self.whisper_model.transcribe(
                        audio_data,
                        beam_size=1,
                        best_of=1,
                        language=stt_lang_code,
                        vad_filter=True,
                        condition_on_previous_text=False,
                        temperature=[0.0],
                        log_prob_threshold=-0.8,
                        compression_ratio_threshold=1.8,
                        no_speech_threshold=0.5
                    )
                    text = "".join([segment.text for segment in segments]).strip()
                    
                    source_lang_name = "English"
                    if info:
                        for k, v in LANG_CODE_MAP.items():
                            if v == info.language:
                                source_lang_name = LANG_NAME_MAP.get(k, "English")
                                break
                
                if text:
                    # Limit maximum length to prevent GGUF parser stack overflow
                    if len(text) > 400:
                        text = text[:400] + "..."
                        
                    lang_key = self.config.get("ui_lang", "JP")
                    loc = UI_LOCALIZATION.get(lang_key, UI_LOCALIZATION["JP"])
                    partner_lbl = loc["partner"]
                    translation_lbl = loc["translation"]
                    
                    self.log(f"[{partner_lbl} - {source_lang_name}] {text}")
                    
                    if is_hallucination_or_excessive_repeat(text):
                        logger.warning(f"Skipping translation for repeated text to prevent LLM crash: {text}")
                        self.log(f"  -> [{translation_lbl}] (繰り返し音声を検知したため、翻訳をスキップしました)\n")
                    else:
                        translated = translate_text(self.llm, text, source_lang_name, mt_target, self.llm_lock)
                        self.log(f"  -> [{translation_lbl} ({mt_target})] {translated}\n")
            except queue.Empty:
                continue
            except Exception as e:
                if self.is_running:
                    logger.exception("Listening Pipeline Error")
                    self.log(f"Listening Pipeline Error: {e}")
 
    def pipeline_speak(self):
        stt_lang_code = LANG_CODE_MAP[self.config["speak_stt"]]
        mt_target = LANG_NAME_MAP[self.config["speak_mt"]]
        
        while self.is_running:
            try:
                audio_data = self.mic_recorder.output_queue.get(timeout=1.0)
                if not self.is_running: break
                
                # 日本語スピーキングは WhisperModel で高精度に認識
                segments, info = self.whisper_model.transcribe(
                    audio_data,
                    beam_size=1,
                    best_of=1,
                    language=stt_lang_code,
                    vad_filter=True,
                    condition_on_previous_text=False,
                    temperature=[0.0],
                    log_prob_threshold=-0.8,
                    compression_ratio_threshold=1.8,
                    no_speech_threshold=0.5
                )
                text = "".join([segment.text for segment in segments]).strip()
                
                if text:
                    # Limit maximum length to prevent GGUF parser stack overflow
                    if len(text) > 400:
                        text = text[:400] + "..."
                        
                    lang_key = self.config.get("ui_lang", "JP")
                    loc = UI_LOCALIZATION.get(lang_key, UI_LOCALIZATION["JP"])
                    me_lbl = loc["me"]
                    chatbox_lbl = loc["chatbox"]
                    
                    # Determine input language name
                    source_lang_name = "English"
                    for k, v in LANG_CODE_MAP.items():
                        if v == info.language:
                            source_lang_name = LANG_NAME_MAP.get(k, "English")
                            break
                    
                    self.log(f"[{me_lbl} - {info.language}] {text}")
                    
                    if is_hallucination_or_excessive_repeat(text):
                        logger.warning(f"Skipping translation for repeated text to prevent LLM crash: {text}")
                        self.log(f"  -> [{chatbox_lbl}] (繰り返し音声を検知したため、翻訳をスキップしました)\n")
                    else:
                        translated = translate_text(self.llm, text, source_lang_name, mt_target, self.llm_lock)
                        self.log(f"  -> [{chatbox_lbl} ({mt_target})] {translated}\n")
                        if self.config.get("use_osc", True):
                            self.osc_client.send_message("/chatbox/input", [translated, True])
            except queue.Empty:
                continue
            except Exception as e:
                if self.is_running:
                    logger.exception("Speaking Pipeline Error")
                    self.log(f"Speaking Pipeline Error: {e}")
 
if __name__ == "__main__":
    app = VRChatTranslatorApp()
    app.mainloop()
