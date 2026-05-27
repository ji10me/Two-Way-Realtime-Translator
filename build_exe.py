import sys
import os
import subprocess

# 1. Base directory setup
base_dir = os.path.dirname(os.path.abspath(__file__))
ico_path = os.path.join(base_dir, "icon.ico")

# 2. Install pyinstaller
subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"])

# 3. Build exe
os.chdir(base_dir)
subprocess.run([sys.executable, "-m", "PyInstaller", "--onefile", f"--icon={ico_path}", "--name=VRC_Translator", "launcher.py"])
