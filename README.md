# RustMoE

RustMoE is a Rust control plane for very large GGUF MoE models on
workstation-class hardware. It treats weight placement across VRAM, host RAM,
and NVMe as a first-class runtime concern instead of assuming the model fits in
VRAM.

It has two model engines plus a separate fast agent:

| Mode | Model | Speed on this host | Entry point |
|---|---|---|---|
| **Deep** | DeepSeek-V4.1-Flash-1017, 748B, IQ1_S | ~4-5 tok/s | http://127.0.0.1:8080 |
| **Fast** | Strata + Qwen3.8-Flash-Next, 125B, IQ3_S | ~90-140 tok/s | http://127.0.0.1:8080 |
| **Fast Agent** | Strata model + RustMoE local tools | follows Strata | http://127.0.0.1:8081/agent |

The two model engines cannot occupy the 24 GB RTX 4090 at the same time.
engine-fast.bat and engine-deep.bat switch between them. The fast agent does
not load a model and can run beside Strata.

[Quick start](#quick-start) · [Dual engine](docs/dual-engine.md) · [Agent](docs/agent.md) · [Skills](docs/skills.md) · [API](docs/api.md) · [Performance](docs/performance.md) · [Troubleshooting](docs/troubleshooting.md)

![RustMoE status dashboard](docs/status-dashboard.png)

## Quick start

Check what is running:

    engine-status.bat

Start the fast stack (Strata + Qwen + RustMoE fast agent):

    engine-fast.bat

Open:

    http://127.0.0.1:8080          Strata chat / OpenAI API
    http://127.0.0.1:8081/agent    RustMoE fast Dev Agent

Start the deep stack (RustMoE + DeepSeek):

    engine-deep.bat

Open:

    http://127.0.0.1:8080/status   live system and API dashboard
    http://127.0.0.1:8080/chat     chat with sessions
    http://127.0.0.1:8080/agent    Deep Dev Agent
    http://127.0.0.1:8080/v1       OpenAI-compatible API

Deep model loading can take 1-5 minutes on NVMe. The switch scripts wait for
/health before returning.

## Why two engines?

Strata's 100+ tok/s and RustMoE's 4-5 tok/s are not the same workload:

| | Strata / Qwen | RustMoE / DeepSeek |
|---|---|---|
| Parameters | ~125B | 748B |
| Weight size | 35-55 GB | 217 GiB |
| GPU expert placement | hot expert cache, 80-90% hit | 3 of 40 layers |
| Speculative decoding | MTP, 4 drafts | not implemented |
| Prompt reuse | 140K-token prefix reuse | not implemented |
| KV | int8 | F16 |
| Decode | 90-140 tok/s | 4-5 tok/s |

See [docs/dual-engine.md](docs/dual-engine.md) and
[docs/performance.md](docs/performance.md) for the measured logs and analysis.

## What this version does

- Parses GGUF v2/v3 headers without touching tensor payloads.
- Discovers split shards, validates offsets, alignment, shapes, and quantization.
- Knows DeepSeek-V4.1: 40 layers, 5120 hidden, MLA, 384 routed experts, top-6
  plus a shared expert, DSpark metadata, and the Engram/PLE table.
- Plans VRAM, RAM, NVMe, KV, and expert cache capacity before loading weights.
- Executes real DeepSeek tokens through the Unsloth llama.cpp dsv41-all branch
  behind a narrow C ABI. RustMoE owns lifecycle, tokenizer, API, and metrics.
- Auto-selects context, GPU expert layers, host layers, and MoE cache from free
  VRAM/RAM and CUDA_VISIBLE_DEVICES.
- Serves OpenAI-compatible chat/completions with SSE streaming, cancellation,
  Prometheus metrics, API key auth, and a live dashboard.
- Provides a browser chat with streaming, auto max-token sizing, auto-continue,
  and persistent sessions stored in localStorage.
- Provides a local Dev Agent with file, shell, skill, project, export, delete,
  and interrupted-task resume tools.
- Loads 17 SKILL.md packs: web research, ComfyUI workflows, computer control,
  website maintenance, project delivery, and DeepSeek Harness packs.
- Runs the Dev Agent against either the local DeepSeek engine or an
  OpenAI-compatible Strata upstream. The fast agent is a separate binary and
  uses no VRAM.


## What this version does not do

- It is not a native Rust/CUDA DeepSeek forward pass. Real execution is
  llama.cpp behind a C ABI.
- It does not implement DeepSeek DSpark speculative decoding, per-expert
  hotness placement, or prompt prefix KV reuse.
- It does not implement a full official Jinja chat template, tools, images,
  reasoning_content, JSON mode, n>1, or /v1/responses.
- It does not run DeepSeek and Strata simultaneously. One GPU, one loaded model.
- It does not guarantee production-quality generated code. Review agent output.

## Current hardware and measured placement

Host:

    GPU:  NVIDIA GeForce RTX 4090 24 GB (index 0)
          NVIDIA GeForce RTX 3060 12 GB (index 1)
    CPU:  Intel i9-14900KF
    RAM:  64 GB
    Disk: C: WD Blue SN570 NVMe (model), D: SATA SSD, G:/H: SATA SSD

Deep 128K baseline:

    Model:             DeepSeek-V4.1-Flash-1017 UD-IQ1_S
    Shards:            5
    Tensor bytes:      216.97 GiB
    Context:           131072
    Threads:           24
    MoE cache:         0 MiB
    GPU expert layers: 3
    Host expert layers:37
    VRAM:              about 20.9-21.1 GB
    Decode:            about 4-5 tok/s
    TTFT:              about 4-6 s for a short request

At 64K, four GPU expert layers were tested and measured 3.22 tok/s, slower than
the 128K three-layer baseline. Do not assume more layers are always faster.

A short 8-token request measured:

    prefill 3.98 s
    decode  1.55 s
    5.16 tok/s

Longer prompts and generated code are slower; 4-5 tok/s is the practical range.

The 12 GB RTX 3060 path remains a low-speed compatibility target at roughly
0.07-0.72 tok/s. It is useful for correctness, not interactive chat.

## Fast mode and the fast agent

Fast mode is Strata, already installed at C:\Strata:

    engine-fast.bat

It starts:

    8080  Strata + Qwen3.8-Flash-Next IQ3_S, 262144 context, int8 KV, MTP
    8081  RustMoE fast agent, no GPU model

The fast agent is a standalone Rust binary:

    rustmoe-agent.exe --addr 127.0.0.1:8081 ^
      --upstream http://127.0.0.1:8080 ^
      --model qwen3.8-flash-next-iq3_s ^
      --agent-root workspace --skills-dir skills --agent-allow-shell

It keeps the local tool loop:

- read/list/write/mkdir/delete/glob/grep
- PowerShell with explicit allow
- skill loading and helper scripts
- project directory management
- interrupted task detection and resume
- ZIP export
- same 17 skills as Deep mode

The only change is where the model tokens come from: Strata instead of the local
DeepSeek engine. A verified two-step task created fast.txt and finished in
seconds.

## Dev Agent

Both agents use the same project model.

Every new run creates:

    workspace/<task-slug>-YYYYMMDD-HHMMSS/

Inside:

    .rustmoe-project.json
    .rustmoe-agent-state.json
    PLAN.md
    ARCHITECTURE.md
    source files, UI, docs, deployment files

Project state is saved after every step. If the server stops, the run is
cancelled, or max steps are reached, the agent page shows:

    Interrupted task found
    <task> - <project> - step 3/24
    [Continue] [Dismiss]

Continue resumes from the saved step with the transcript and existing files.
New projects are isolated by directory; old projects can be continued, exported,
or deleted. See [docs/agent.md](docs/agent.md).

### Skills

The skills directory contains 17 packs:

    web-research
    comfyui-video
    computer-control
    website-maintenance
    project-delivery
    agent-experience
    cordis-composition-reference
    cordis-plugin-development
    editing-cordis-compositions
    diagnose-windows-sandbox-acl
    office
    office-docx
    office-export-pdf
    office-pptx
    office-recalculate
    office-render
    office-xlsx

Each pack is a SKILL.md with frontmatter. The agent sees a catalog and loads the
body with the skill tool. Web search/fetch and ComfyUI are helper scripts run
through the shell. See [docs/skills.md](docs/skills.md).

Skills are instructions, not plugins. Office/Cordis skills can require Python
packages, LibreOffice Kit, or the DSH Cordis host. ComfyUI requires a running
server at http://127.0.0.1:8188.


## API

Deep and fast agent expose related APIs.

| Method | Path | Purpose |
|---|---|---|
| GET | /health | liveness, model, engine |
| GET | /status | dashboard (Deep only) |
| GET | /chat | chat UI (Deep only) |
| GET | /agent | Dev Agent UI |
| GET | /v1/models | model list and context fields |
| GET | /v1/runtime | plan, caches, metrics (Deep) |
| GET | /v1/system | hardware snapshot (Deep) |
| GET | /v1/telemetry | CPU/GPU/RAM/activity feed (Deep) |
| POST | /v1/chat/completions | chat or SSE stream |
| POST | /v1/completions | text completion or SSE |
| POST | /v1/tokenize | GGUF BPE tokenization |
| POST | /v1/cache/reset | reset cache counters |
| POST | /v1/agent/run | agent SSE run |
| POST | /v1/agent/cancel | request cancellation |
| GET | /v1/agent/projects | list projects |
| DELETE | /v1/agent/projects/{name} | delete project |
| GET | /v1/agent/projects/{name}/export | ZIP download |
| GET | /v1/agent/skills | list skills |
| GET | /v1/agent/skills/{name} | skill body |

Base URLs:

    http://127.0.0.1:8080/v1    OpenAI SDK base_url
    http://127.0.0.1:8080       DeepSeek clients that append /chat/completions

Do not use the full endpoint as base_url and do not duplicate /v1.

Example:

    curl http://127.0.0.1:8080/v1/chat/completions ^
      -H "Content-Type: application/json" ^
      -d "{\"model\":\"deepseek-v4.1-flash-1017\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64}"

With an API key:

    -H "Authorization: Bearer YOUR_KEY"

The status dashboard's API access card shows the live base URL, model, auth
state, endpoint links, and copyable cURL commands.

## Build from source

Prerequisites: Rust stable, CMake, Visual Studio with CUDA, and the reference
llama.cpp branch.

    git clone --depth 1 --branch dsv41-all https://github.com/unslothai/llama.cpp.git .models/llama.cpp
    cmake -S .models/llama.cpp -B .models/llama-build -G "Visual Studio 18 2026" -A x64 ^
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89" ^
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
    cmake --build .models/llama-build --config Release --target llama llama-cli llama-server -- /m:16
    cargo build --release -p rustmoe-server --features llama

This also builds the fast agent:

    cargo build --release -p rustmoe-server --bin rustmoe-agent

The helper scripts scripts/build-reference.ps1 and scripts/download-iq1-s.ps1
perform the reference setup. The build applies
patches/llama-moe-cache-stats.patch for MoE cache telemetry.

Model download:

    https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/UD-IQ1_S

## Model facts

| Fact | Observed |
|---|---|
| Architecture key | deepseek41 |
| Layers | 40 |
| Hidden size | 5120 |
| Attention | MLA, q_lora 1280, latent/key 512, rope 64, 64 heads, 8 groups |
| Routed experts | 384/layer, top-6, one shared expert, sqrtsoftplus |
| Context | 1,048,576 native, sliding window 128 |
| Engram/PLE | layers 1 and 14, 768,022,850 rows, head dim 256 |
| DSpark | separate optional drafter, block 5, target layers 37..39 |
| IQ1_S tensor bytes | 216.97 GiB |
| Tensors | 1,043 |

The Q4_K_XL directory is 500.7 GB and is not the model used for the 24 GB
profile. Its routed experts are MXFP4 and the Engram table is MXFP8. IQ1_S is
the low-memory profile.

## Memory planning

    rustmoe plan MODEL.gguf --context 131072 --vram 24GiB --ram 64GiB

The planner reports:

- usable VRAM after display, runtime, safety, KV, and workspace reserves;
- dense spine placement;
- GPU/host expert cache capacity;
- KV bytes per component and per token;
- cold NVMe/PCIe time per token, explicitly not a throughput promise;
- a reduced prefill chunk without truncating prompts.

12 GB VRAM with 32 GB RAM is a compatibility target. The planner is allowed to
return INSUFFICIENT.


## Architecture

    CLI / OpenAI HTTP / SSE / Status / Chat / Agent
                         |
                  Runtime assembly
                         |
        +----------------+----------------+
        | GGUF parser    | Memory planner  |
        | splits, shapes | VRAM/RAM/NVMe   |
        +----------------+----------------+
                         |
        +----------------+----------------+
        | Expert scheduler | GPU/host/SSD  |
        | top-K preserved  | caches        |
        +----------------+----------------+
                         |
              Inference backend boundary
                  /                 \
       llama.cpp C ABI          Strata HTTP
       local DeepSeek           fast agent only
                         |
                    Metrics / telemetry

The agent server is separate because it must run while Strata owns the GPU:

    rustmoe-agent (8081, no model)
        |-- local file/shell/skill/project tools
        |-- HTTP -> Strata 8080 /v1/chat/completions

## Repository layout

    crates/gguf           GGUF v2/v3 and split discovery
    crates/quant          tensor layouts and dequantizers
    crates/model          DeepSeek-V4.1 facts and validation
    crates/attention      MLA geometry and KV budget
    crates/memory         VRAM/RAM/disk planner
    crates/io-engine      positioned reads and mmap expert store
    crates/expert-cache   GPU/host cache bookkeeping
    crates/scheduler      top-K preserving scheduling
    crates/backend        backend trait and simulator
    crates/backend-llama  llama.cpp C shim
    crates/tokenizer      GGUF BPE
    crates/inference      prefill, decode, sampling, UTF-8 streaming
    crates/telemetry      metrics and Prometheus
    crates/server         CLI, HTTP, dashboard, chat, agent, fast agent
    skills/               17 SKILL.md packs and helpers
    scripts/              build, engine switch, import, diagnostics
    fixtures/tiny-ds41    synthetic test model
    docs/                 architecture, API, agent, skills, performance

## Tests

    cargo fmt --check
    cargo clippy --workspace --all-targets
    cargo test --workspace

The server suite covers GGUF parsing, quantization, planning, scheduler/cache
invariants, tokenizer/UTF-8, API shape, SSE, auth, telemetry, status/chat/agent
pages, and agent/skill/project/resume unit tests.

## Security and safety

- Unsupported quantization fails before load.
- Split shards and tensor bounds are validated.
- Router top-K is a hard contract; cache pressure never drops an expert.
- Prefill chunks shrink; prompts are not silently truncated.
- The agent is restricted to --agent-root and rejects .., absolute paths, and
  escaping symlinks.
- Shell is disabled unless --agent-allow-shell is set.
- The agent can delete files and run commands. Use a dedicated workspace and
  review generated code.
- Fast agent and deep agent both bind to localhost by default. Add --api-key
  before exposing them.

## Status and roadmap

| Area | Status |
|---|---|
| GGUF v2/v3 and split validation | implemented |
| DeepSeek-V4.1 metadata/catalog | implemented |
| Memory planning | implemented |
| OpenAI API, SSE, auth, metrics | implemented |
| Real DeepSeek via llama.cpp | implemented |
| 12 GB real decode | validated, low-speed |
| 24 GB real decode | validated, 4-5 tok/s at 128K |
| Status dashboard + API access | implemented |
| Chat sessions | implemented |
| Dev Agent, projects, resume | implemented |
| 17 skills, web/ComfyUI helpers | implemented |
| Fast agent over Strata | implemented |
| Dual-engine switch | implemented |
| Native Rust/CUDA forward | not implemented |
| DSpark speculative decoding | not implemented |
| Per-expert hot cache | not implemented |
| Prompt prefix KV reuse | not implemented |

Next engineering priorities:

1. Prefix KV reuse for agent/chat.
2. DeepSeek DSpark/speculative decoding.
3. Per-expert profile cache in the llama.cpp path.
4. Fused low-bit CUDA kernels.
5. Optional gateway that routes one endpoint to fast or deep based on health.

## Documentation

- [Documentation index](docs/README.md)
- [docs/dual-engine.md](docs/dual-engine.md) - fast/deep switching
- [docs/troubleshooting.md](docs/troubleshooting.md) - common failures
- [docs/agent.md](docs/agent.md) - projects, resume, security
- [docs/skills.md](docs/skills.md) - 17 skills and adding new ones
- [docs/api.md](docs/api.md) - HTTP/SSE compatibility
- [docs/performance.md](docs/performance.md) - measured TPS and Strata comparison
- [docs/benchmark.md](docs/benchmark.md) - benchmark details
- [docs/real-backend.md](docs/real-backend.md) - llama.cpp setup and flags
- [docs/memory-planning.md](docs/memory-planning.md) - budgets and placement
- [docs/architecture.md](docs/architecture.md) - crate architecture
- [docs/model-snapshot.md](docs/model-snapshot.md) - verified model facts
- [docs/supported-formats.md](docs/supported-formats.md) - GGUF/quantization
- [docs/status-dashboard.png](docs/status-dashboard.png)
- [docs/chat.png](docs/chat.png)
- [docs/agent.png](docs/agent.png)

## License

MIT. See LICENSE.

Created by clarkzhu.

- GitHub: https://github.com/clarkzhu2020
- Blog: https://clarkzhu.blog
