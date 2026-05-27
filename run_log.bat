@echo off
chcp 65001 >nul
echo ============================================
echo  VRC Translator - Setup with Log Capture
echo  Log: %~dp0install.log
echo ============================================
echo.
echo [INFO] セットアップを実行中です... (数分かかる場合があります)
echo [INFO] 出力は install.log に保存されます
echo.

set "RUNBAT=%~dp0run.bat"
set "LOGFILE=%~dp0install.log"

REM PowerShell のパイプを介さず、単純なリダイレクトにする
call "%RUNBAT%" > "%LOGFILE%" 2>&1

if errorlevel 1 (
    echo.
    echo [ERROR] セットアップ中にエラーが発生しました。詳細なログ(install.log)を表示します:
    echo.
    type "%LOGFILE%"
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  完了。ログは install.log に保存されました。
echo ============================================
pause
