import os
import sys
import subprocess
import shutil
import glob

def check_nvidia_gpu():
    """Check if an NVIDIA GPU is present by trying to run nvidia-smi."""
    try:
        res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            return True
    except Exception:
        pass
    return False

def check_cuda_toolkit():
    """Check if CUDA Toolkit v12.x or v11.x is installed in the default location."""
    for ver in ['v12.1', 'v12.0', 'v12.2', 'v12.3', 'v12.4', 'v11.8', 'v11.7']:
        cuda_bin = fr'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{ver}\bin'
        if os.path.isdir(cuda_bin):
            if glob.glob(os.path.join(cuda_bin, "cudart64_*.dll")):
                return cuda_bin
    return None

def main():
    # 実行環境が PyInstaller (Frozen) かどうかでベースディレクトリを判定
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(os.path.abspath(sys.executable))
    else:
        base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    
    os.chdir(base_dir)
    
    venv_dir = os.path.join(base_dir, "venv")
    python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
    
    # 仮想環境 (venv) の作成に用いる Python コマンドの判定
    if getattr(sys, 'frozen', False):
        python_cmd = "python"
    else:
        python_cmd = sys.executable

    # 1. 既存の仮想環境 (venv) が正常に動作するかチェック
    if os.path.exists(python_exe):
        try:
            result = subprocess.run([python_exe, "-c", "import sys"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                raise Exception("venv python is not working correctly")
        except Exception:
            print("[INFO] 既存の仮想環境 (venv) が現在のPC環境と一致しないため、再構築します...")
            try:
                shutil.rmtree(venv_dir)
            except Exception as e:
                print(f"[WARNING] 既存の venv フォルダの削除に失敗しました: {e}")

    # 2. 仮想環境が存在しない場合は作成
    if not os.path.exists(python_exe):
        print("[INFO] 初回セットアップを実行中... 仮想環境を作成しています。")
        try:
            subprocess.run([python_cmd, "--version"], capture_output=True, check=True)
        except Exception:
            print("[ERROR] システムに Python がインストールされていないか、PATHが通っていません。")
            print("Python 3.10 または 3.11 をインストールし、「Add Python.exe to PATH」にチェックを入れてください。")
            input("続行するには何かキーを押してください . . .")
            sys.exit(1)
            
        try:
            subprocess.run([python_cmd, "-m", "venv", "venv"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 仮想環境の作成に失敗しました: {e}")
            input("続行するには何かキーを押してください . . .")
            sys.exit(1)
        
    # 3. 必要なライブラリのインストール・更新
    print("[INFO] 必要なライブラリを確認・更新しています...")
    try:
        # pip 自体のアップグレード
        subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip", "--quiet"], check=True)
        
        ver_res = subprocess.run([python_exe, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], capture_output=True, text=True)
        py_version = ver_res.stdout.strip()
        print(f"[INFO] 仮想環境の Python バージョン: {py_version}")
        
        has_gpu = check_nvidia_gpu()
        use_cuda = False
        
        if has_gpu:
            print("[INFO] NVIDIA GPU が検出されました。")
            cuda_dir = check_cuda_toolkit()
            
            # CUDA Toolkit がない場合は pip 経由で CUDA runtime DLLs をインストールする
            if not cuda_dir:
                print("[INFO] システムに CUDA Toolkit が見つかりません。pip経由でCUDAランタイムを自動導入します...")
                try:
                    subprocess.run([python_exe, "-m", "pip", "install", 
                                    "nvidia-cuda-runtime-cu12", 
                                    "nvidia-cublas-cu12", 
                                    "nvidia-cudnn-cu12",
                                    "--quiet"], check=True)
                    print("[INFO] CUDAランタイムの自動インストールに成功しました。")
                except subprocess.CalledProcessError as e:
                    print(f"[WARNING] CUDAランタイムの pip インストールに失敗しました: {e}")
            
            # llama-cpp-python のインストール
            print("[INFO] CUDA (GPU) 対応版の llama-cpp-python をインストールしています...")
            cuda_install = subprocess.run([python_exe, "-m", "pip", "install", "llama-cpp-python>=0.3.20", 
                                           "--prefer-binary", "--no-cache-dir", 
                                           "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu121", 
                                           "--upgrade"])
            if cuda_install.returncode == 0:
                use_cuda = True
                print("[INFO] CUDA (GPU) 対応版 llama-cpp-python のインストールに成功しました。")
            else:
                print("[WARNING] CUDA版のインストールに失敗しました。CPU版にフォールバックします...")
        
        if not use_cuda:
            print("[INFO] CPU版の llama-cpp-python をインストールしています...")
            cpu_install = subprocess.run([python_exe, "-m", "pip", "install", "llama-cpp-python>=0.3.20", 
                                           "--prefer-binary", "--no-cache-dir", 
                                           "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu", 
                                           "--upgrade"])
            if cpu_install.returncode != 0:
                print("\n[ERROR] llama-cpp-python のインストールに失敗しました。")
                print("【原因】Python バージョンが新しすぎる（例: 3.12以上）か、環境がサポートされていません。")
                print("【解決策】Python 3.10.x または 3.11.x をインストールし直してください。")
                input("続行するには何かキーを押してください . . .")
                sys.exit(1)

        # その他の依存パッケージのインストール
        print("[INFO] その他の依存ライブラリをインストールしています...")
        subprocess.run([python_exe, "-m", "pip", "install", "-r", "requirements.txt", "--upgrade"], check=True)

        # 4. DLL コピー処理
        if use_cuda:
            print("[INFO] CUDA DLL のコピーを行っています...")
            site_packages_res = subprocess.run([python_exe, "-c", "import site; print(site.getsitepackages()[0])"], capture_output=True, text=True)
            sp_dir = site_packages_res.stdout.strip()
            
            llama_lib_dir = os.path.join(sp_dir, "llama_cpp", "lib")
            os.makedirs(llama_lib_dir, exist_ok=True)
            
            # コピー元候補
            search_dirs = []
            nvidia_packages = glob.glob(os.path.join(sp_dir, "nvidia", "*", "bin"))
            search_dirs.extend(nvidia_packages)
            cuda_sys = check_cuda_toolkit()
            if cuda_sys:
                search_dirs.append(cuda_sys)
            search_dirs.append(r'C:\Windows\System32')
            
            dll_patterns = [
                "cudart64*.dll", "cublas64*.dll", "cublasLt64*.dll", "cudnn*.dll",
                "nvJitLink*.dll", "nvrtc*.dll", "curand64*.dll", "cusolver64*.dll",
                "cusparse64*.dll", "cufft64*.dll"
            ]
            
            copied_count = 0
            for d in search_dirs:
                if not os.path.isdir(d):
                    continue
                for pat in dll_patterns:
                    for f in glob.glob(os.path.join(d, pat)):
                        dest_file = os.path.join(llama_lib_dir, os.path.basename(f))
                        if not os.path.exists(dest_file):
                            try:
                                shutil.copy2(f, dest_file)
                                copied_count += 1
                            except Exception:
                                pass
            print(f"[INFO] {copied_count} 個の CUDA DLL を llama_cpp/lib に配置しました。")
            
            # ctranslate2 の DLL コピー
            ct_dir = os.path.join(sp_dir, "ctranslate2")
            if os.path.isdir(ct_dir):
                for f in glob.glob(os.path.join(ct_dir, "*.dll")):
                    try:
                        shutil.copy2(f, llama_lib_dir)
                    except Exception:
                        pass

            # テストロード
            test_res = subprocess.run([python_exe, "-c", "from llama_cpp import Llama"], capture_output=True)
            if test_res.returncode != 0:
                print("[WARNING] CUDA 版 llama-cpp-python の読み込みテストに失敗しました。CPU版に切り替えます。")
                subprocess.run([python_exe, "-m", "pip", "install", "llama-cpp-python", 
                                "--prefer-binary", "--no-cache-dir", 
                                "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu", 
                                "--force-reinstall"], check=True)

    except subprocess.CalledProcessError as e:
        print(f"[WARNING] セットアップ中にエラーが発生しました: {e}")
    
    # 5. main.py の実行
    print("[INFO] VRChat Translator を起動します...")
    if not os.path.exists(os.path.join(base_dir, "main.py")):
        print("[ERROR] main.py が見つかりません。")
        input("続行するには何かキーを押してください . . .")
        sys.exit(1)
        
    try:
        subprocess.run([python_exe, "main.py"])
    except Exception as e:
        print(f"[ERROR] プログラムの起動に失敗しました: {e}")
        input("続行するには何かキーを押してください . . .")

if __name__ == "__main__":
    main()
