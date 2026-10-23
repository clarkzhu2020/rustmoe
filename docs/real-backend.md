# Real execution backend

RustMoE can execute real DeepSeek-V4.1-Flash tokens through the Unsloth
llama.cpp reference branch. This is Route A: llama.cpp owns model loading, MLA,
Engram, routing, and kernels; RustMoE owns the GGUF control plane, memory plan,
tokenizer boundary, CLI, HTTP/SSE API, metrics, and lifecycle.

The backend is a C++ shim with a small C ABI. It is not a native Rust/CUDA
implementation. Do not describe it as one.

## Validated configuration

| Item | Value |
|---|---|
| Model | DeepSeek-V4.1-Flash-1017 UD-IQ1_S |
| Model size | 5 shards, 216.97 GiB tensor bytes |
| Current GPU | RTX 4090 24 GB, compute capability 8.9 |
| Historical GPU | RTX 3060 12 GB, compute capability 8.6 |
| Host RAM | 64 GB (32 GB minimum target) |
| Model disk | C: WD Blue SN570 NVMe; 1-5 min cold load observed |
| Reference | unslothai/llama.cpp, dsv41-all, commit d1781c3 |
| CUDA | 13.4 build against CUDA 12 runtime DLLs |
| Context | 131072 current; 2048 for the first 12 GB runs |

The 12 GB results below are historical. The current 24 GB profile runs 3 GPU
expert layers, 37 host layers, 24 threads, 0 MiB MoE cache, and 4-5 tok/s.

The original 12 GB validation host had 68 GB installed; the current host has 64
GB. Both satisfy the 32 GB minimum. Peak process commit in the 12 GB run was
about 11.8 GB and the 256-token working set about 19.3 GB. Do not enable mlock
on a 32 GB host.

## Windows 4090 24 GB / 64 GB launcher

startdeepseek.bat detects the physical RTX 4090 index, sets
CUDA_VISIBLE_DEVICES, and starts the real backend with automatic placement.

    startdeepseek.bat --check
    startdeepseek.bat
    startdeepseek.bat 65536
    testdeepseek.bat
    restartdeepseek.bat

The live dashboard is at http://127.0.0.1:8080/status. It refreshes /health,
/v1/models, /v1/runtime, and /v1/system every two seconds.
    move-model-to-ssd.bat
    disks.bat

During model load a terminal spinner shows elapsed seconds. After the HTTP
listener binds, the server prints a ready banner with URL, model, backend,
context, GPU placement, and cache size.

--check validates files and GPU detection without loading weights. The default
context is 131072 (128K); startdeepseek.bat 65536 requests 64K explicitly. Both
64K and 128K were executed on the RTX 4090. The launcher now uses 24 CPU expert
threads and 0 MiB MoE cache by default; set RUSTMOE_THREADS and
RUSTMOE_MOE_CACHE to override. RUSTMOE_ADDR changes the listen address.
testdeepseek.bat verifies /health, /v1/models, and a real chat request.

For the fast/deep switch and the 8081 fast agent, see docs/dual-engine.md.

For OpenAI clients the base URL is http://127.0.0.1:8080/v1, not the full
/chat/completions endpoint. /v1/responses is not implemented. The current build
also accepts a trailing slash on the completion routes and returns a 404 body
that lists valid endpoints.

## Automatic placement

Automatic placement is the default for the real backend. It reads free VRAM and
available RAM, filters the GPU list through CUDA_VISIBLE_DEVICES (numeric
indices), and chooses context, expert-layer placement, and MoE cache size. Override it with
--no-auto-device and explicit flags. A context of 0 means auto.

The policy reserves 2 GiB for compute, 1 GiB safety, and the MLA KV estimate;
places the dense spine first; then assigns complete expert layers to the GPU;
and gives half of the remaining VRAM to the hot-expert cache, capped at 4 GiB.

For IQ1_S the planner produces approximate targets:

| Free VRAM | Context | GPU expert layers | Host expert layers | MoE cache |
|---:|---:|---:|---:|---:|
| 12 GiB | 4096 | 0 | 40 | 1143 MiB |
| 16 GiB | 4096 | 1 | 39 | 1431 MiB |
| 24 GiB | 8192 | 3 | 37 | 1812 MiB |
| 32 GiB | 16384 | 6 | 34 | 332 MiB |

The 12 GiB row was executed on an RTX 3060 with CUDA_VISIBLE_DEVICES=1 and
completed. A 24 GB RTX 4090 with 21.60 GiB free executed the planner's
three-expert-layer plan at context 8192, with 573 MiB cache; observed GPU use was
about 17.6 GB. The 16 and 32 GiB rows are unit-tested plans only; those GPUs were
not available. The policy uses free, not total, VRAM, so a busy GPU gets a
smaller plan or fails before load.

For code agents, set context explicitly. On the RTX 4090 with 21.1 GiB free:

| Requested context | GPU expert layers | Host expert layers | MoE cache | Observed GPU |
|---:|---:|---:|---:|---:|
| 65536 (64K) | 3 | 37 | 0 MiB | ~18.1 GB |
| 131072 (128K) | 3 | 37 | 0 MiB | ~20.9 GB |

Both executed successfully. The 128K case leaves roughly 3.6 GB of the 24 GB
card free; 64K leaves more headroom. startdeepseek.bat defaults to 128K and
accepts `startdeepseek.bat 65536` for 64K. The C shim splits prefill into
backend-sized batches; a 700-word prompt at 128K completed with HTTP 200 after
the split fix. `/v1/models` reports the actual
runtime context, so code agents that require context_length > 64K see 65536 or
131072, not 8192.

## Storage mapping on the test host

| Drive | Device | Media | Bus | Role |
|---|---|---|---|---|
| C: | WD Blue SN570 1TB | SSD | NVMe | system; only NVMe volume |
| D: | WDC WDS100T2G0A | SSD | SATA | software |
| E:, F: | ST8000VN004 | HDD | SATA | historical model store |
| G:, H: | Samsung 870 EVO 2TB | SSD | SATA | project/backup; not NVMe |
| X: | Seagate BUP Slim | HDD | USB | backup |

The model was originally on F:, an HDD, where a cold load took about 12 minutes.
It now lives on C:, the only NVMe volume; cold load is 1-5 minutes. G:/H: are
SATA SSDs, much faster than the HDD but slower than C:.

## Flags

- --backend llama selects the reference backend.
- --no-auto-device disables the automatic plan; otherwise placement is derived
  from the visible device.
- --context 0 asks automatic placement to choose a context; an explicit value
  overrides it.
- --gpu-layers 999 offloads dense tensors and attention to the GPU; in automatic
  mode this is filled from the device plan.
- --cpu-moe-layers=-1 overrides all routed expert tensors to the host. This is
  the key to fitting 12 GB. Positive N keeps the first N layers on the host;
  0 keeps all experts on the GPU.
- --moe-cache-mib 1024 allocates a 1 GiB device cache for hot host experts.
- --llama-batch 512 sets the logical batch.
- --load-mode 1 selects mmap.
- --lazy-mode on allows on-demand rows for the 202 GB Engram/PLE table.
- --threads 24 controls CPU expert computation; the current 4090 profile uses 24.

The 1 GiB cache figures below are historical 12 GB measurements. Dense weights
consumed about 6.3 GB, CUDA compute buffers about 1.3 GB, and the rest was
KV/context/runtime. The current 128K profile uses 0 MiB cache because the small
cache had no hits and slowed decode. Do not raise the cache without measuring
backend hit rate and free VRAM.

Cold startup is slow. Mapping metadata and dense weights over a 233 GB cold
store can take minutes; warm OS page cache makes later starts much faster. On
the F: HDD the observed load was about 12 minutes; a 200 MB/s full-read estimate
is 18.5 minutes. The same model on a 3 GB/s NVMe should load in 1-3 minutes.
The server is not ready until /health returns 200. startdeepseek.bat reports the
drive type and a worst-case estimate before loading; move-model-to-ssd.bat
copies the model to C: and writes a pointer.

## Measured output

Eight-token continuation, temperature 0:

    prompt:     "The quick brown fox"
    completion: ". Jumps. Over the lazy dog"
    prompt:      4 tokens
    prefill:    53.19 s
    decode:     76.22 s for 8 tokens
    throughput: 0.105 tokens/s
    first token:53.20 s

Raw completion, temperature 0:

    prompt:     "The capital of France is"
    completion: " Paris. The capital"
    prompt:      5 tokens
    prefill:    60.16 s
    decode:     24.38 s for 4 tokens
    throughput: 0.164 tokens/s
    first token:60.17 s

Chat request, temperature 0:

    user:       "What is the capital of France? Answer briefly."
    assistant:  "Paris."
    prompt:     14 tokens
    prefill:    159.4 s
    decode:     28.6 s for 2 tokens
    throughput: 0.070 tokens/s

These numbers are low-speed compatibility. The model is correct enough to answer
the prompt, but it is not interactive. The dominant cost is CPU expert compute,
not just NVMe bandwidth. The 12 GB card cannot hold enough experts to avoid it.

## Why both mmap and CPU expert overrides

mmap maps the 233 GB cold store without requiring 233 GB resident. CPU overrides
tell llama.cpp that expert tensors must live in host-mapped memory rather than
VRAM. The OS page cache and llama.cpp's host MoE cache keep the working set
bounded. The warning:

    tensor overrides to CPU are used with mmap enabled - consider using
    --load-mode none for better performance

is expected. --load-mode none would try to make the full cold store resident and
is not suitable for 32 GB RAM.

## Build

The repository build script for the backend uses these environment variables:

- LLAMA_CPP_DIR: default .models/llama.cpp
- LLAMA_BUILD_DIR: default .models/llama-build

The Rust build script compiles the shim, links llama/ggml import libraries, and
copies the needed DLLs beside the Rust executable. A binary built with
--features llama will not start correctly if llama.dll, ggml.dll,
ggml-base.dll, ggml-cpu.dll, ggml-cuda.dll, and the CUDA 12 runtime DLLs are
missing.

## 256-token stability

A 256-token greedy run with prompt "Once", context 2048, and a 1 GiB MoE cache
completed without OOM, corruption, or deadlock:

    completion:       256 tokens, finish length
    prefill:          22.04 s
    decode:           355.16 s (0.72 tokens/s)
    working set:      about 19.3 GB
    backend_cache:    0 hits, 61440 misses, 460.91 GB uploaded

The sampled text was 256 newline tokens. This validates length and stability, not
output quality. A larger context or a better prompt is required for a meaningful
long-form sample.

## Chat template

Real mode uses a basic DeepSeek-V4.1 text chat encoding:

    <BOS><User>message<Assistant></think>

System messages and assistant history are handled. The encoder closes thinking
mode for direct chat. It does not implement DSML tool calls, image blocks,
reasoning budgets, or tool-result turns. Use /v1/completions for raw prompts
when exact prompt control matters.

## MoE cache telemetry

The reference source is patched with a small stats accessor. RustMoE exposes
cumulative llama.cpp MoE cache counters in the generation result and in
/v1/runtime:

    "backend_cache": {
      "hits": 0,
      "misses": 240,
      "uploaded_bytes": 1800437760,
      "copied_hits": 0,
      "copied_bytes": 0
    }

From a one-token run with a 1 GiB GPU MoE cache: all 240 selected experts
missed and about 1.80 GB was uploaded. A four-token run reported 1399 misses,
zero hits, and 10.32 GB uploaded. The 1 GiB cache is too small for even one
token's working set, so it provides no reuse in these short runs. This makes the
12 GB behavior visible instead of inferred.

A 2 GiB cache produced 221 hits, 1315 misses, 9.92 GB uploaded, and 12.04 s
decode versus 12.35 s at 1 GiB. It fit on the 12 GB card but gave little wall
clock benefit for four tokens. Every GiB given to the MoE cache is unavailable
to KV, compute buffers, or additional dense layers; use backend_cache to measure
before increasing it.

llama.cpp still owns the router. RustMoE does not see per-layer expert ids, so
its own scheduler/cache counters remain zero in real mode. A router callback is
the next instrumentation step. The patch is not part of the upstream project;
it is a local observability patch.

## Observability limitation

RustMoE's scheduler counters are inactive in real mode. Use backend_cache for
the actual llama.cpp MoE cache behavior.

## Failure modes

- Out of VRAM: lower --moe-cache-mib, reduce --gpu-layers, or lower --context.
- Out of host RAM: keep mmap and lazy on; lower the OS page-cache pressure.
- Model not found: pass shard 1; llama.cpp discovers the remaining shards.
- Missing DLLs: rebuild with --features llama or set the DLL search path.
- Garbage or empty chat: the simple chat encoder may not match every thinking
  or tool configuration. Use /v1/completions to isolate the model.
