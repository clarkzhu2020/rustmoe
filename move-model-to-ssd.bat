@echo off
setlocal
cd /d "%~dp0"
echo Moving the IQ1_S model to a fast disk.
echo The source is kept; only a copy is made.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\move-model-to-ssd.ps1" %*
echo.
pause
