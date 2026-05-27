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

# ==========================================
# HELPER CLASSES
def get_device_name_only(combo_value):
    if "]" in combo_value:
        return combo_value.split("]", 1)[1].strip()
    return combo_value

def load_config():
    default_config = {
        "speaker_device": "", 
        "mic_device": "",
        "listen_stt": "English (英語)",
        "listen_mt": "Japanese (日本語)",
        "speak_stt": "Japanese (日本語)",
        "speak_mt": "English (英語)",
        "use_osc": True
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
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration
        self.threshold = threshold
        self.max_duration = max_duration
        self.audio_chunks = []
        self.silence_chunks = 0
        self.is_recording_active = False
        self.output_queue = queue.Queue()
        self.stream = None
        
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
                    self.output_queue.put(audio_data)
                    self.audio_chunks = []
                    self.is_recording_active = False
                    self.silence_chunks = 0
                    return

        # 強制区切り (最大録音時間を超えた場合)
        if self.is_recording_active:
            total_samples = sum(len(c) for c in self.audio_chunks)
            if total_samples >= self.max_duration * self.sample_rate:
                audio_data = np.concatenate(self.audio_chunks, axis=0)
                self.output_queue.put(audio_data)
                self.audio_chunks = []
                self.is_recording_active = False
                self.silence_chunks = 0

    def start(self, log_callback):
        self.stream = sd.InputStream(device=self.device_idx,
                                     channels=1,
                                     samplerate=self.sample_rate,
                                     callback=self.callback)
        self.stream.start()
        log_callback(f"[{self.name}] 録音を開始しました。")

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()

def translate_text(llm, text, source_lang, target_lang):
    if source_lang == target_lang:
        return text
    prompt_text = f"Translate the following {source_lang} text to {target_lang}. Output ONLY the translated text without any explanation or quotes.\n\nText: {text}\n\nTranslation:"
    
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
        self.speaker_recorder = None
        self.mic_recorder = None
        
        self.devices = []
        self.device_names = []
        self.populate_devices()
        self.config = load_config()
        
        self.setup_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def populate_devices(self):
        devs = sd.query_devices()
        for i, dev in enumerate(devs):
            if dev['max_input_channels'] > 0:
                self.devices.append((i, dev['name']))
                self.device_names.append(f"[{i}] {dev['name']}")
                
    def setup_ui(self):
        # Header
        self.header = ctk.CTkLabel(self, text="Two-Way Realtime Translator\n(Moonshine + Whisper Hybrid)", font=ctk.CTkFont(size=22, weight="bold"))
        self.header.pack(pady=(15, 5))
        
        # --- Listen Settings (相手の音声) ---
        self.frame_listen = ctk.CTkFrame(self)
        self.frame_listen.pack(pady=5, padx=20, fill="x")
        ctk.CTkLabel(self.frame_listen, text="【相手の音声 (リスニング・英語固定/Moonshine)】", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=4, pady=(10,5), padx=10, sticky="w")
        
        ctk.CTkLabel(self.frame_listen, text="デバイス:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.combo_speaker = ctk.CTkComboBox(self.frame_listen, values=self.device_names, width=300, command=self.on_setting_changed)
        self.combo_speaker.grid(row=1, column=1, columnspan=3, padx=10, pady=5, sticky="w")
        spk_name = self.config.get("speaker_device", "")
        if spk_name:
            for name in self.device_names:
                if get_device_name_only(name) == spk_name:
                    self.combo_speaker.set(name)
                    break
            
        ctk.CTkLabel(self.frame_listen, text="認識言語:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.combo_listen_stt = ctk.CTkComboBox(self.frame_listen, values=STT_LANGUAGES, width=150, command=self.on_setting_changed)
        self.combo_listen_stt.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        if self.config.get("listen_stt") in STT_LANGUAGES:
            self.combo_listen_stt.set(self.config["listen_stt"])
        else:
            self.combo_listen_stt.set("English (英語)")
            
        ctk.CTkLabel(self.frame_listen, text="➡ 翻訳先:").grid(row=2, column=2, padx=10, pady=5, sticky="w")
        self.combo_listen_mt = ctk.CTkComboBox(self.frame_listen, values=MT_LANGUAGES, width=150, command=self.on_setting_changed)
        self.combo_listen_mt.grid(row=2, column=3, padx=10, pady=5, sticky="w")
        if self.config["listen_mt"] in MT_LANGUAGES: self.combo_listen_mt.set(self.config["listen_mt"])

        # --- Speak Settings (自分の音声) ---
        self.frame_speak = ctk.CTkFrame(self)
        self.frame_speak.pack(pady=5, padx=20, fill="x")
        ctk.CTkLabel(self.frame_speak, text="【自分の音声 (スピーキング・日本語推奨/Whisper)】", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=4, pady=(10,5), padx=10, sticky="w")
        
        ctk.CTkLabel(self.frame_speak, text="デバイス:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.combo_mic = ctk.CTkComboBox(self.frame_speak, values=self.device_names, width=300, command=self.on_setting_changed)
        self.combo_mic.grid(row=1, column=1, columnspan=3, padx=10, pady=5, sticky="w")
        mic_name = self.config.get("mic_device", "")
        if mic_name:
            for name in self.device_names:
                if get_device_name_only(name) == mic_name:
                    self.combo_mic.set(name)
                    break
            
        ctk.CTkLabel(self.frame_speak, text="認識言語:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.combo_speak_stt = ctk.CTkComboBox(self.frame_speak, values=STT_LANGUAGES, width=150, command=self.on_setting_changed)
        self.combo_speak_stt.grid(row=2, column=1, padx=10, pady=5, sticky="w")
        if self.config["speak_stt"] in STT_LANGUAGES: self.combo_speak_stt.set(self.config["speak_stt"])
            
        ctk.CTkLabel(self.frame_speak, text="➡ 翻訳先:").grid(row=2, column=2, padx=10, pady=5, sticky="w")
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
        
        self.log("システムを起動しました。デバイスと言語を選択して「開始」を押してください。")
 
    def log(self, message):
        logger.info(message)
        def _append():
            self.log_box.configure(state="normal")
            self.log_box.insert("end", message + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.after(0, _append)
 
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
        self.config["speaker_device"] = get_device_name_only(self.combo_speaker.get())
        self.config["mic_device"] = get_device_name_only(self.combo_mic.get())
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
        
        # Get current indexes based on selected device names
        spk_name = get_device_name_only(self.combo_speaker.get())
        mic_name = get_device_name_only(self.combo_mic.get())
        
        self.speaker_idx = None
        self.mic_idx = None
        
        devs = sd.query_devices()
        for i, dev in enumerate(devs):
            if dev['max_input_channels'] > 0:
                if dev['name'] == spk_name and self.speaker_idx is None:
                    self.speaker_idx = i
                if dev['name'] == mic_name and self.mic_idx is None:
                    self.mic_idx = i
                    
        if self.speaker_idx is None or self.mic_idx is None:
            self.log("エラー: 設定されたデバイスが見つかりません。接続を確認してください。")
            return

        self.lock_ui(True)
        self.btn_stop.configure(state="disabled") # Temporary while loading
        self.log("\n--- 初期化中 ---")
        
        self.is_running = True
        threading.Thread(target=self.init_and_run, daemon=True).start()
 
    def on_stop(self):
        self.is_running = False
        self.log("停止処理を行っています...")
        if self.speaker_recorder:
            self.speaker_recorder.stop()
        if self.mic_recorder:
            self.mic_recorder.stop()
            
        self.lock_ui(False)
        self.log("--- 停止しました ---")
 
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
                        self.log(f"エラー: モデルファイルが破損しています (magic={_magic.hex()})。")
                        self.log("models フォルダ内の .gguf ファイルを削除して再起動してください。")
                        self.after(0, self.on_stop)
                        return
                except Exception as _ve:
                    self.log(f"エラー: モデルファイルを読み取れません: {_ve}")
                    self.after(0, self.on_stop)
                    return
 
                self.log("Hy-MT2 (LLM) をロード中...")
                try:
                    self.llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=2048, verbose=False)
                    self.log("-> GPU モードでロードしました")
                except Exception as gpu_e:
                    self.log(f"-> GPU モード失敗。CPU モードで再試行します... ({gpu_e})")
                    try:
                        self.llm = Llama(model_path=model_path, n_gpu_layers=0, n_ctx=2048, verbose=True)
                        self.log("-> CPU モードでロードしました")
                    except Exception as cpu_e:
                        self.log(f"-> CPU モードも失敗しました: {cpu_e}")
                        raise cpu_e
            if not self.is_running: return
                
            # 3. Load Speech-to-Text Engines
            listen_stt_val = self.config.get("listen_stt", "English (英語)")
            if listen_stt_val == "English (英語)":
                if self.moonshine_model_en is None:
                    self.log("Moonshine (英語音声認識・爆速) をロード中...")
                    try:
                        self.moonshine_model_en = moonshine_onnx.MoonshineOnnxModel(model_name="useful-sensors/moonshine/tiny")
                        self.log("-> Moonshine (英語用) をロードしました")
                    except Exception as e:
                        self.log(f"-> Moonshine 英語初期化失敗: {e}")
                        raise e
            elif listen_stt_val == "Spanish (スペイン語)":
                if self.moonshine_model_es is None:
                    self.log("Moonshine (スペイン語音声認識・爆速) モデルを準備中...")
                    try:
                        es_dir = download_moonshine_es_if_needed(self.log)
                        self.log("Moonshine (スペイン語音声認識・爆速) をロード中...")
                        self.moonshine_model_es = moonshine_onnx.MoonshineOnnxModel(models_dir=es_dir, model_name="moonshine-es-base")
                        self.log("-> Moonshine (スペイン語用) をロードしました")
                    except Exception as e:
                        self.log(f"-> Moonshine スペイン語初期化失敗: {e}")
                        raise e

            # Load Faster-Whisper (Japanese STT)
            if self.whisper_model is None:
                self.log("Faster-Whisper (スピーキング用) をロード中...")
                try:
                    self.whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
                    self.log("-> Faster-Whisper GPU モードでロードしました")
                except Exception as e:
                    self.log(f"-> Faster-Whisper GPU初期化失敗。CPUモードで実行します。({e})")
                    self.whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=1, num_workers=1)
            if not self.is_running: return
                    
            # 4. Start Recorders
            self.speaker_recorder = AudioRecorder(self.speaker_idx, "相手の音声", sample_rate=SAMPLE_RATE, threshold=SILENCE_THRESHOLD)
            self.mic_recorder = AudioRecorder(self.mic_idx, "自分の音声", sample_rate=SAMPLE_RATE, threshold=SILENCE_THRESHOLD)
            
            self.speaker_recorder.start(self.log)
            self.mic_recorder.start(self.log)
            
            self.osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
            
            self.log("--- 準備完了！ 翻訳を開始しました ---")
            self.after(0, lambda: self.btn_stop.configure(state="normal"))
            
            # 5. Start Processing Threads
            t1 = threading.Thread(target=self.pipeline_listen, daemon=True)
            t2 = threading.Thread(target=self.pipeline_speak, daemon=True)
            t1.start()
            t2.start()
            
        except Exception as e:
            logger.exception("初期化中にエラーが発生しました")
            self.log(f"初期化中にエラーが発生しました: {e}")
            self.after(0, self.on_stop)
 
    def pipeline_listen(self):
        mt_target = LANG_NAME_MAP[self.config["listen_mt"]]
        
        while self.is_running:
            try:
                audio_data = self.speaker_recorder.output_queue.get(timeout=1.0)
                if not self.is_running: break
                
                listen_stt_val = self.config.get("listen_stt", "English (英語)")
                if listen_stt_val == "English (英語)":
                    # 英語は Moonshine を用いて超高速・暴走なしで文字起こし
                    text_list = moonshine_onnx.transcribe(audio_data, model=self.moonshine_model_en)
                    text = "".join(text_list).strip()
                    source_lang_name = "English"
                elif listen_stt_val == "Spanish (スペイン語)":
                    # スペイン語も Moonshine を用いて超高速・暴走なしで文字起こし
                    text_list = moonshine_onnx.transcribe(audio_data, model=self.moonshine_model_es)
                    text = "".join(text_list).strip()
                    source_lang_name = "Spanish"
                else:
                    # 英語以外は WhisperModel で文字起こし
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
                    self.log(f"[相手 - {source_lang_name}] {text}")
                    translated = translate_text(self.llm, text, source_lang_name, mt_target)
                    self.log(f"  -> [翻訳 ({mt_target})] {translated}\n")
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
                    self.log(f"[自分 - {info.language}] {text}")
                    source_lang_name = "English"
                    for k, v in LANG_CODE_MAP.items():
                        if v == info.language:
                            source_lang_name = LANG_NAME_MAP.get(k, "English")
                            break
                    translated = translate_text(self.llm, text, source_lang_name, mt_target)
                    self.log(f"  -> [ChatBox ({mt_target})] {translated}\n")
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
