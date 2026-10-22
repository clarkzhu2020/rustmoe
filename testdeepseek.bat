@echo off
setlocal
cd /d "%~dp0"
echo Testing RustMoE OpenAI API...
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\test-deepseek.ps1" %*
echo.
pause
