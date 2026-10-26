# Dual engine: Strata fast + RustMoE deep

Both model engines expose an OpenAI-compatible API on the same address
(127.0.0.1:8080 by default), but they cannot share the RTX 4090 at the same time.
Strata uses about 23.5 GB of the 24 GB card, leaving no room for DeepSeek.

The fast agent is different: it loads no model and runs on port 8081 beside
Strata. It calls Strata for tokens and keeps RustMoE's local tools.

## Use

    engine-status.bat     show engine, fast agent, and VRAM
    engine-fast.bat       stop RustMoE, start Strata + fast agent
    engine-deep.bat       stop fast agent and Strata, start RustMoE

The wrappers call scripts/engine.ps1:

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\engine.ps1 status
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\engine.ps1 fast
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\engine.ps1 deep

## Ports and URLs

| Port | Fast mode | Deep mode |
|---|---|---|
| 8080 | Strata chat/API | RustMoE status/chat/agent/API |
| 8081 | RustMoE fast agent | not running |

Fast mode:

    http://127.0.0.1:8080          Strata chat and OpenAI API
    http://127.0.0.1:8080/v1       Strata base URL
    http://127.0.0.1:8081/agent    RustMoE Dev Agent using Strata

Deep mode:

    http://127.0.0.1:8080/status   RustMoE live dashboard
    http://127.0.0.1:8080/chat     RustMoE chat
    http://127.0.0.1:8080/agent    RustMoE Dev Agent using DeepSeek
    http://127.0.0.1:8080/v1       RustMoE OpenAI API

## Fast mode

    engine-fast.bat

What it does:

1. If RustMoE is running, stop it.
2. If Strata is not running, start C:\Strata\START-HERE.bat.
3. Wait for Strata /health with service=strata.
4. Start target\release\rustmoe-agent.exe on 8081 with upstream
   http://127.0.0.1:8080.
5. Wait for the fast agent /health with service=rustmoe-fast-agent.

Model:

    Qwen3.8-Flash-Next IQ3_S
    262144 context
    int8 KV
    MTP speculative decoding
    expert profile / hot expert cache
    observed 90-140 tok/s

Use for code completion, research, website work, ComfyUI workflows, and fast
agent iterations.

## Fast agent

The fast agent is a separate binary:

    target\release\rustmoe-agent.exe --addr 127.0.0.1:8081 ^
      --upstream http://127.0.0.1:8080 ^
      --model qwen3.8-flash-next-iq3_s ^
      --agent-root workspace --skills-dir skills --agent-allow-shell

It implements the same local tool loop as the deep agent:

- list/read/write/mkdir/delete/glob/grep
- PowerShell shell with explicit allow
- 17 skills
- project directories and metadata
- interrupted task detection and resume
- ZIP export and delete
- SSE events start/thinking/step/tool/finish/error

Only the model source changes. A two-step file-creation test completed in
seconds and wrote the expected file.

The fast agent has no local model and therefore uses no VRAM. It can run beside
Strata as long as the machine has enough host RAM.

## Deep mode

    engine-deep.bat

What it does:

1. Stop rustmoe-agent on 8081.
2. Stop strata and strata-vision.
3. Wait for the port and VRAM to be released.
4. Start startdeepseek.bat with automatic placement.
5. Wait for RustMoE /health with backend=llama.cpp.

Model:

    DeepSeek-V4.1-Flash-1017 IQ1_S
    748B parameters, 216.97 GiB tensor bytes
    131072 context
    3 GPU expert layers + 37 host layers
    0 MiB MoE cache
    observed 4-5 tok/s

Deep model loading can take 1-5 minutes. The script waits up to 900 seconds and
then prints the ready line. If it times out, check the loader window or the
startdeepseek log.

## Strata startup tuning

Strata needs headroom after loading the expert cache. On this 24 GB card the
working settings are:

    --vram-reserve-mib 1200
    --prefill auto:4096

With 700 MiB reserve Strata loaded but exited with only 333 MiB VRAM free. At
1200 MiB it started with about 1.3 GB free. These values live in
C:\Strata\strata-iq3_s.json. A smaller prompt chunk borrows fewer GPU cache
slots during prefill.

If engine-fast times out, read C:\Strata\strata-iq3_s.log and look for the last
VRAM free line. Also remove stale Python setup.py launchers from failed starts.

## Why not both

    Strata VRAM: about 23.5 GB
    RustMoE dense + KV + 3 expert layers: about 21 GB
    Total: far above 24 GB

A 24 GB card cannot hold both. The model switch is exclusive. There is no
configuration flag that makes both models resident.

The fast agent is the exception because it has no weights.

## Verified status

At the time of writing:

    engine-status.bat
    [engine] FAST  Strata
      model: qwen3.8-flash-next-iq3_s
      ctx: 262144
    [agent] FAST  http://127.0.0.1:8081/agent
      model: qwen3.8-flash-next-iq3_s upstream: http://127.0.0.1:8080
    GPU 0: about 23.5 GB used

The fast agent wrote and finished a test file in two tool steps.

## Configuration

The scripts read:

    STRATA_HOME          default C:\Strata
    RUSTMOE_ADDR         default 127.0.0.1:8080
    RUSTMOE_FAST_MODEL   default qwen3.8-flash-next-iq3_s
    RUSTMOE_AGENT_ROOT   default <repo>\workspace
    RUSTMOE_SKILLS_DIR   default <repo>\skills

The fast agent binary must exist:

    cargo build --release -p rustmoe-server --bin rustmoe-agent

If it is missing, engine-fast.bat reports that command instead of starting a
stale process.

## Failure modes

Symptom: Strata health is up but 8081 says not running.

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\engine.ps1 fast

Check target\release\rustmoe-agent.exe exists.

Symptom: engine-deep times out at /health.

    Check the hidden launcher output or run startdeepseek.bat manually.
    Check nvidia-smi shows Strata stopped.
    Check the model path and llama DLLs.

Symptom: port 8080 is occupied by an unknown process.

    Get-NetTCPConnection -LocalPort 8080 | Select-Object OwningProcess
    Get-Process -Id <pid>

Do not kill an unknown process blindly.

## Safety

- engine-deep stops strata and strata-vision. Let an active generation finish
  first if you need its output.
- engine-fast stops rustmoe.
- Both scripts force-stop the previous engine to free VRAM.
- Model files are never moved or modified.
- The fast agent runs with shell enabled in the provided script. Review the
  security section in docs/agent.md.
- Do not run both launch scripts at the same time.
