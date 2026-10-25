# Troubleshooting

## Check the active engine first

    engine-status.bat

This prints:

- service on 8080: Strata FAST or RustMoE DEEP;
- fast agent on 8081;
- current GPU used/free.

Most "slow" or "wrong page" reports are simply the other engine owning 8080.

## 8080 is Strata but I expected RustMoE

That is fast mode. Switch:

    engine-deep.bat

Wait for the ready line. Deep model load takes 1-5 minutes.

## 8080 is RustMoE but I expected Strata

That is deep mode. Switch:

    engine-fast.bat

This stops RustMoE, starts Strata, and starts the fast agent on 8081.

## Fast agent is not running on 8081

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\engine.ps1 status
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\engine.ps1 fast

If it reports a missing binary:

    cargo build --release -p rustmoe-server --bin rustmoe-agent

Check the port:

    Get-NetTCPConnection -LocalPort 8081

Only one process can bind 8081.

## Fast agent returns an upstream error

The fast agent forwards model calls to Strata at 8080. Check:

    Invoke-RestMethod http://127.0.0.1:8080/health

It must return service=strata. If not, start fast mode. If Strata is busy, the
agent waits for the model response; check /status on 8080.

If Strata requires an API key, start the agent with:

    --upstream-api-key KEY

## Deep server says rustmoe.exe is missing

startdeepseek.bat now attempts a release build automatically when the binary is
missing. If that fails, run:

    cargo build --release -p rustmoe-server --features llama --bin rustmoe

Check that cargo is on PATH and that target\release is writable.

## Deep server fails to load

Common causes:

- Strata still owns the GPU: engine-status should show no Strata process.
- The model path or a shard is missing.
- llama.dll or CUDA runtime DLLs are missing beside rustmoe.exe.
- The planner returned INSUFFICIENT for the current free VRAM.

Check:

    nvidia-smi
    target\release\rustmoe.exe --version
    startdeepseek.bat --check

Do not set a larger context or more GPU layers just to force a fit. The planner
is allowed to refuse.

## API returns 404

Use the base URL, not the full endpoint:

    base_url = http://127.0.0.1:8080/v1
    path     = /chat/completions

Do not use:

    http://127.0.0.1:8080/v1/chat/completions/chat/completions
    http://127.0.0.1:8080/v1/v1/chat/completions

## API returns 401

Deep mode with --api-key requires:

    Authorization: Bearer KEY

The agent page stores the key in localStorage. Fast agent requires its own
--api-key only when configured.

## Agent is interrupted

Open /agent. The banner offers Continue. Continue resumes at the saved step.
If the page was reloaded, select the project and use Continue/Use.

The state file is:

    workspace/<project>/.rustmoe-agent-state.json

Do not hand-edit it unless you have a backup.

## Agent shell tool is disabled

Start with:

    --agent-allow-shell

or set RUSTMOE_AGENT_ALLOW_SHELL=1 before startdeepseek.bat. The fast agent
script already passes this flag; the setting applies to the deep agent.

## Skills do not appear

Check the directory:

    Invoke-RestMethod http://127.0.0.1:8081/v1/agent/skills

Each skill needs a SKILL.md with YAML frontmatter:

    ---
    name: my-skill
    description: What it does.
    ---

The loader scans recursively. Restart is not needed for a new run.

## ComfyUI skill cannot connect

ComfyUI must be running at:

    http://127.0.0.1:8188

Test:

    Invoke-WebRequest http://127.0.0.1:8188/system_stats

A skill cannot install or start ComfyUI for you unless the model runs its helper
commands.

## Low tokens per second in deep mode

Expected: 4-5 tok/s on the 748B IQ1_S profile. Check placement in /status:

    3 GPU expert layers
    37 host layers
    24 threads
    0 MiB MoE cache

If the plan shows 2 GPU layers, another process is using VRAM. Stop Strata,
close other GPU apps, and restart deep mode. Do not expect 100 tok/s from 748B
on one 24 GB card.

## Port already in use

    Get-NetTCPConnection -LocalPort 8080 | Select-Object OwningProcess
    Get-Process -Id <pid>

Stop the previous engine through engine-fast/engine-deep rather than killing an
unknown process.
