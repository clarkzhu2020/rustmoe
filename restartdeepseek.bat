@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "PORT=8080"
if defined RUSTMOE_ADDR for /f "tokens=2 delims=:" %%P in ("%RUSTMOE_ADDR%") do set "PORT=%%P"

set "FOUND="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:":%PORT% .*LISTENING"') do (
    if not defined FOUND (
        tasklist /fi "PID eq %%P" 2>nul | findstr /i "rustmoe.exe" >nul
        if not errorlevel 1 set "FOUND=%%P"
    )
)
if defined FOUND (
    echo [restart] stopping rustmoe PID !FOUND! on port %PORT%
    taskkill /F /PID !FOUND! >nul
    timeout /t 2 /nobreak >nul
) else (
    echo [restart] no rustmoe process is listening on port %PORT%
)
call "%~dp0startdeepseek.bat" %*
