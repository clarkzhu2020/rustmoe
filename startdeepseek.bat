@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title RustMoE - DeepSeek-V4.1 - RTX 4090 24GB / 64GB RAM
rem Usage:
rem   startdeepseek.bat                 default 128K context, auto placement
rem   startdeepseek.bat 65536           force 64K context
rem   startdeepseek.bat --check         verify files and GPU, do not start
rem
rem Environment overrides:
rem   RUSTMOE_THREADS   CPU expert threads, default 24
rem   RUSTMOE_ADDR      listen address, default 127.0.0.1:8080
set "MODEL="
if defined RUSTMOE_MODEL set "MODEL=%RUSTMOE_MODEL%"
if not defined MODEL if exist ".models\model-path.txt" set /p MODEL=<".models\model-path.txt"
if defined MODEL if not exist "%MODEL%" set "MODEL="
if not defined MODEL if exist "C:\rustmoe-models\UD-IQ1_S\DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf" set "MODEL=C:\rustmoe-models\UD-IQ1_S\DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf"
if not defined MODEL if exist "G:\rustmoe-models\UD-IQ1_S\DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf" set "MODEL=G:\rustmoe-models\UD-IQ1_S\DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf"
if not defined MODEL set "MODEL=.models\UD-IQ1_S\DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00005.gguf"
set "EXE=target\release\rustmoe.exe"
if not defined RUSTMOE_ADDR set "RUSTMOE_ADDR=127.0.0.1:8080"
if not defined RUSTMOE_THREADS set "RUSTMOE_THREADS=24"
if not defined RUSTMOE_MOE_CACHE set "RUSTMOE_MOE_CACHE=0"
if not defined RUSTMOE_AGENT_ROOT set "RUSTMOE_AGENT_ROOT=%CD%\workspace"
if not defined RUSTMOE_SKILLS_DIR set "RUSTMOE_SKILLS_DIR=%CD%\skills"
if not defined RUSTMOE_AGENT_ALLOW_SHELL set "RUSTMOE_AGENT_ALLOW_SHELL=0"
set "AGENT_SHELL="
if "%RUSTMOE_AGENT_ALLOW_SHELL%"=="1" set "AGENT_SHELL=--agent-allow-shell"
set "CONTEXT=%~1"
if "%CONTEXT%"=="" set "CONTEXT=131072"
if /i "%CONTEXT%"=="--check" set "CHECK_ONLY=1"
if /i "%CONTEXT%"=="-c" set "CHECK_ONLY=1"
if defined CHECK_ONLY set "CONTEXT=0"
echo.
echo ============================================================
echo   RustMoE - real DeepSeek-V4.1-Flash-1017
echo   Target: RTX 4090 24GB + 64GB RAM
echo ============================================================
echo.
if not exist "%EXE%" (
    echo [build] missing %EXE%
    echo [build] cargo build --release -p rustmoe-server --features llama
    where cargo >nul 2>nul
    if errorlevel 1 (
        echo [error] cargo was not found on PATH
        goto :fail
    )
    cargo build --release -p rustmoe-server --features llama --bin rustmoe
    if errorlevel 1 (
        echo [error] build failed
        goto :fail
    )
    if not exist "%EXE%" (
        echo [error] build completed but %EXE% is still missing
        goto :fail
    )
)
if not exist "%MODEL%" (
    echo [error] missing model shard:
    echo         %MODEL%
    echo         download with: scripts\download-iq1-s.ps1
    goto :fail
)
set "DISK_INFO="
for /f "usebackq delims=" %%D in (`powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\disk-info.ps1" -Path "%MODEL%"`) do set "DISK_INFO=%%D"
if defined DISK_INFO echo [disk] !DISK_INFO!
echo !DISK_INFO! | findstr /i "HDD" >nul
if not errorlevel 1 (
    echo [warn] the model is on a hard disk; a cold load can take 15-25 minutes.
    echo        for interactive use, run move-model-to-ssd.bat once, then restart.
)
for %%D in (llama.dll ggml.dll ggml-base.dll ggml-cpu.dll ggml-cuda.dll) do (
    if not exist "target\release\%%D" (
        echo [error] missing runtime DLL: target\release\%%D
        echo         rebuild with --features llama
        goto :fail
    )
)
rem Pick the physical 4090 index. CUDA_VISIBLE_DEVICES uses numeric indices.
set "GPU_INDEX="
where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    for /f "tokens=1,2 delims=," %%A in ('nvidia-smi "--query-gpu=index,name" "--format=csv,noheader"') do (
        set "IDX=%%A"
        set "GNAME=%%B"
        set "GNAME=!GNAME: =!"
        echo !GNAME! | findstr /i "4090" >nul
        if not errorlevel 1 if not defined GPU_INDEX set "GPU_INDEX=!IDX!"
    )
)
if defined GPU_INDEX (
    echo [gpu] found RTX 4090 at physical index !GPU_INDEX!
    if defined CUDA_VISIBLE_DEVICES (
        echo [gpu] respecting existing CUDA_VISIBLE_DEVICES=!CUDA_VISIBLE_DEVICES!
    ) else (
        set "CUDA_VISIBLE_DEVICES=!GPU_INDEX!"
        echo [gpu] CUDA_VISIBLE_DEVICES=!GPU_INDEX!
    )
) else (
    echo [gpu] no RTX 4090 found by nvidia-smi
    echo [gpu] auto placement will use the visible GPU with the most free VRAM
)
echo [ram] 64GB target: auto placement reads actual available RAM
echo [ctx] %CONTEXT% tokens  ^(0 = choose from free VRAM^)
echo [cpu] %RUSTMOE_THREADS% expert threads
echo [url] http://%RUSTMOE_ADDR%
echo [ui]  http://%RUSTMOE_ADDR%/status
echo [chat]  http://%RUSTMOE_ADDR%/chat
echo [agent] http://%RUSTMOE_ADDR%/agent  root=%RUSTMOE_AGENT_ROOT%  shell=%RUSTMOE_AGENT_ALLOW_SHELL%  skills=%RUSTMOE_SKILLS_DIR%
echo [api] base URL: http://%RUSTMOE_ADDR%   endpoint: /v1/chat/completions
echo [api] do not use .../v1/v1/chat/completions or a trailing slash as base URL
echo.
if defined CHECK_ONLY (
    echo [check] files, DLLs, and GPU detection passed; not starting the server.
    exit /b 0
)
set "PORT=8080"
for /f "tokens=2 delims=:" %%P in ("%RUSTMOE_ADDR%") do set "PORT=%%P"
set "PORT_PID="
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /r /c:":%PORT% .*LISTENING"') do if not defined PORT_PID set "PORT_PID=%%P"
if defined PORT_PID (
    echo [error] port %PORT% is already in use by PID !PORT_PID!
    tasklist /fi "PID eq !PORT_PID!"
    echo         run restartdeepseek.bat, or set RUSTMOE_ADDR to another port
    goto :fail
)
if not defined RUST_LOG set "RUST_LOG=error"
echo Loading the 217 GiB IQ1_S model. NVMe cold load is 1-5 min; HDD can take 15-25 min.
echo Wait for the HTTP server to report ready, then run:
echo   testdeepseek.bat
echo or:
echo   curl http://%RUSTMOE_ADDR%/health
echo.
"%EXE%" serve "%MODEL%" ^
    --backend llama ^
    --addr "%RUSTMOE_ADDR%" ^
    --context "%CONTEXT%" ^
    --threads "%RUSTMOE_THREADS%" ^
    --moe-cache-mib "%RUSTMOE_MOE_CACHE%" ^
    --agent-root "%RUSTMOE_AGENT_ROOT%" --agent-max-steps 24 ^
    --skills-dir "%RUSTMOE_SKILLS_DIR%" %AGENT_SHELL% ^
    --llama-batch 512 ^
    --load-mode 1 ^
    --lazy-mode on ^
    --model-id "deepseek-v4.1-flash-1017"
set "RC=%ERRORLEVEL%"
echo.
echo [exit] rustmoe returned %RC%
if not "%RC%"=="0" goto :fail
pause
exit /b 0
:fail
echo.
echo Startup failed. See the messages above.
pause
exit /b 1
