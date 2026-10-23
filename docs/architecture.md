# Architecture

## Goal

RustMoE controls the placement and movement of a very large GGUF MoE model so
that it can decode on a workstation whose VRAM and RAM are much smaller than the
model. The runtime must be schedulable, observable, and verifiable. It must not
silently change model semantics to make the budget close.

## Non-goals

- Replacing every llama.cpp kernel immediately.
- Training or quantization.
- Multi-user batching.
- Multi-GPU execution.
- Running DeepSeek and a second GPU model at the same time.
- Claiming throughput without measurement.

## Layers

    User interface
      CLI, OpenAI HTTP, SSE
    Runtime
      GGUF catalog, memory planner, expert scheduler, caches
    Execution
      Backend trait; llama.cpp real backend; simulation for tests
    Fast agent
      local tools plus Strata HTTP model
    Storage
      NVMe split files; optional mmap; explicit positioned reads
    Devices
      One CUDA device when a real backend is connected

The Rust layers own lifecycle, validation, placement, policy, and
observability. Numeric kernels must eventually run on CUDA. Rust is not used as
a substitute for GPU matrix math.

## Crates

| Crate | Responsibility |
|---|---|
| rustmoe-gguf | streaming header parser, split discovery, tensor directory |
| rustmoe-quant | GGML type table, byte layout math, reference dequantizers |
| rustmoe-model | DeepSeek-V4.1 facts, tensor roles, shape validation |
| rustmoe-attention | MLA geometry and conservative KV budget |
| rustmoe-memory | VRAM/RAM/disk planner and diagnostics |
| rustmoe-io-engine | positioned reads, mmap store, expert byte ranges |
| rustmoe-expert-cache | cost-aware LRU/LFU hybrid bookkeeping |
| rustmoe-scheduler | top-K validation, cache resolution, read batching |
| rustmoe-backend | backend trait, deterministic simulation |
| rustmoe-tokenizer | GGUF GPT-2 byte-level BPE and UTF-8 decode |
| rustmoe-inference | prefill chunking, decode loop, sampling, stream events |
| rustmoe-telemetry | atomic counters, latency buckets, Prometheus |
| rustmoe-backend-llama | narrow C++ shim for the DeepSeek llama.cpp branch |
| rustmoe-server | CLI, OpenAI API, SSE, auth, dashboards, agent, fast agent |

## Load sequence

1. Discover shards from the entry filename, or resolve a directory containing
   exactly one split set.
2. Parse each header with a hard byte limit. Tensor payloads are not read.
3. Validate split indices, counts, alignment, tensor ranges, and duplicate
   tensor names.
4. Parse DeepSeek-V4.1 metadata into typed configuration.
5. Build a tensor catalog: role, layer, expert count, bytes per expert.
6. Validate expected tensor names and shapes, expert consistency, and
   quantization support.
7. Compute the memory budget: VRAM reserves, KV, workspace, dense spine, expert
   caches, RAM reserve, and disk.
8. Only then is a backend allowed to allocate or read weights.

A metadata-only shard can be inspected with --single-file. It cannot produce a
weight-backed plan because no tensor directory exists.

## Scheduler invariants

For every layer route:

- the number of routed experts equals metadata expert_used_count;
- expert ids are unique within the layer;
- the shared expert is marked as computed when the model has one;
- all layers 0..block_count-1 appear exactly once;
- expert ids are inside the declared expert count.

A violation returns an error. It never degrades to fewer experts.

For each selected expert:

1. Resolve GPU hit, host hit, or cold miss.
2. On a host hit, account for H2D bytes and promote to the GPU tier.
3. On a cold miss, account for NVMe bytes, insert into host, insert into GPU.
4. If an expert is larger than the GPU tier, it is bypassed for that step; it
   is still present in the computation.
5. Collect cold ranges into per-layer, per-shard read batches.
6. Coalesce adjacent ranges only when the gap is below the configured limit.

The scheduler currently computes the plan and cache statistics. A production
backend will use the same plan to issue actual asynchronous reads.

## Cache policy

The GPU hot tier and host warm tier use the same policy. The default is a
cost-aware hybrid:

    value = (frequency_weight * ln(1 + hits)
             + recency_weight * recency
             + cost_weight * ln(1 + load_cost)) / sqrt(bytes)

The lowest value is evicted. LRU and LFU are available for diagnosis. This is
not plain LRU: a repeatedly loaded, expensive expert should survive a one-shot
expert even if its last access is older.

## KV and attention state

DeepSeek-V4.1 uses MLA plus compressed K-only caches. The exact cache has raw
sliding-window state, compressed source-layer state, index state, and backend
workspace. RustMoE v0.1 estimates these components from metadata to reserve
memory. The estimate is intentionally conservative. A backend must replace it
with exact allocation accounting.

The planner never treats the 1M context limit as free. It reports KV bytes,
per-token bytes, and the context that the budget can support.

## Backend boundary

    trait InferenceBackend {
        fn name(&self) -> &str;
        fn load_model(&mut self, model: &ModelDescriptor) -> Result<(), BackendError>;
        fn prefill(&mut self, tokens: &[u32], state: &mut ModelState)
            -> Result<BackendOutput, BackendError>;
        fn decode_step(&mut self, token: u32, state: &mut ModelState)
            -> Result<BackendOutput, BackendError>;
        fn reset(&mut self);
    }

BackendOutput contains logits and one route per layer. The interface is
deliberately small. A production version needs device-resident logits,
asynchronous transfers, KV lifecycle, and batched execution; these are roadmap
items.

## Route A and Route B

Route A is implemented. At startup the memory crate computes an automatic device
plan from free VRAM, available RAM, and CUDA_VISIBLE_DEVICES; the RustMoE backend
receives the resulting context, expert-layer split, and MoE cache size. The
rustmoe-backend-llama crate compiles a small C++ shim against the Unsloth DeepSeek-V4.1 llama.cpp branch and exposes only opaque
sessions and token-level calls to Rust. llama.cpp owns MLA, Engram, routing, and
kernels. RustMoE owns the API, lifecycle, plan, tokenizer boundary, and metrics.
The real backend is selected with --backend llama and requires the llama Cargo
feature.

Route A still has a gap: llama.cpp owns the router and MoE cache, so RustMoE's
expert-cache counters are inactive in real mode. Instrumenting router decisions
is the next control-plane task.

Route B replaces execution with native Rust control plus CUDA kernels. It must
proceed one operator family at a time with parity gates. The MXFP4 GEMV prototype
in kernels/cuda is the first step, not the finish line.

## Agent and dual engine

The deep server embeds the agent in crates/server/src/agent.rs. The agent owns
project directories, state, skills, path guards, tools, and SSE. Model calls go
through one of two paths:

    AgentModel::Local(Runtime) -> llama.cpp DeepSeek
    AgentModel::Http(HttpAgent) -> OpenAI-compatible /v1/chat/completions

The standalone rustmoe-agent binary is an HTTP agent server with no local model.
It binds 127.0.0.1:8081, forwards model calls to Strata on 8080, and executes all
tools locally. This is how the fast agent works while Strata owns the GPU.

engine-fast.bat starts Strata and the fast agent. engine-deep.bat stops both and
starts the deep RustMoE server on 8080. The two model engines are exclusive; the
fast agent is not, because it has no weights.

## Observability

Per request the runtime records:

- prompt and completion token counts;
- time-to-first-token, prefill time, total decode time;
- GPU, host, and cold expert counts;
- NVMe bytes read and H2D bytes;
- read batch count, prefetch issued and used;
- eviction count;
- llama.cpp MoE cache hits, misses, uploaded bytes, and copied experts when the
  real backend is active;
- approximate latency histograms.

Prometheus names are stable and documented in docs/api.md.
