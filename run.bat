@echo off
chcp 65001 >nul
set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%venv

REM 1. Python check
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python 3.10 or 3.11 and check "Add Python to PATH" during installation.
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 2. Rebuild venv if broken
if exist "%VENV_DIR%\Scripts\python.exe" (
    "%VENV_DIR%\Scripts\python.exe" -c "import sys" >nul 2>nul
    if errorlevel 1 (
        echo [INFO] Virtual environment venv is broken. Rebuilding...
        rmdir /s /q "%VENV_DIR%"
    )
)

REM 3. Create venv
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment venv...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM 4. Activate venv
echo [INFO] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

REM 5. Run launcher.py to manage setup and execution
python "%SCRIPT_DIR%launcher.py"
if errorlevel 1 (
    echo [ERROR] Program exited with an error. Check app.log or install.log for details.
    pause
    exit /b 1
)
