# Memory planning

## Principle

The planner does not accept a single --gpu-memory number and hope. It computes
what remains after fixed reserves and refuses to start when the budget does not
close.

For VRAM:

    V_usable  = V_total - V_display - V_runtime - V_safety
    V_fixed   = V_spine + V_kv + V_workspace
    V_experts = max(0, V_usable - V_fixed)

For host RAM:

    R_base  = R_available - R_reserve - R_io - R_offloaded_spine
    R_cache = min(requested_or_75_percent_of_base, R_base, all_expert_bytes)

The dense spine is placed in VRAM when it fits with KV and workspace. Otherwise
it is offloaded to RAM and a warning is emitted. Passing --host-spine forces the
offload. Passing a GPU-backed spine that does not fit produces an error.

## Reserves

Defaults, all overridable in code and partially on the CLI:

| Reserve | Default | Purpose |
|---|---:|---|
| display | 512 MiB | desktop / display engine |
| runtime | 768 MiB | CUDA context and backend runtime |
| safety | 512 MiB | allocator fragmentation and driver overhead |
| workspace | auto, at least 1.5 GiB | attention/compute scratch and prefill |
| RAM reserve | 8 GiB | OS, services, page cache pressure |
| I/O buffers | 512 MiB | read staging and transfer queues |

The planner also compares reported free VRAM with the amount required now. If
another process is occupying the device, the plan fails instead of assuming it
will go away.

## Current 24 GB result

The RTX 4090 / 64 GB host with 128K context produces:

    GPU expert layers: 3
    host expert layers: 37
    MoE cache: 0 MiB
    KV: F16 MLA state
    observed VRAM: about 20.9-21.1 GB
    predicted headroom: small, positive

The planner deliberately reserves compute and safety headroom. Four layers at
64K did not improve decode; it measured 3.22 tok/s versus 4.03-5.16 at 128K and
three layers.

## KV estimate

The DSV4.1 cache is not a Llama KV cache. The planner estimates:

- raw MLA sliding-window state for every layer, rows capped by
  attention.sliding_window;
- one compressed latent row per compressed position for each
  kv_source_layers entry, with each layer compression ratio;
- one indexer key row per compressed position for each indexer source layer;
- a small Engram position-history array when Engram is present.

The estimate is deliberately conservative and excludes backend-specific
workspace. The planner reports each component, total bytes, and bytes per token.
A production backend must replace this estimate with exact allocations.

## Prefill

    activation_per_token = max(32 KiB, hidden_size * 16)
    workspace            = max(1 GiB, requested_prefill * activation_per_token) + 512 MiB
    max_chunk            = workspace / activation_per_token

If the requested prefill batch is larger than max_chunk, the chunk is reduced
and a warning is emitted. The engine processes every prompt chunk in order. The
prompt is never truncated to satisfy a budget.

## Expert I/O estimate

    cold_bytes_per_step = avg_expert_bytes * experts_per_token
    experts_per_token   = expert_used_count * block_count

The estimate intentionally uses the cold case. Static planning cannot predict
router reuse, so it does not invent a hit rate. It reports:

- GPU cache capacity in experts and as a fraction of all experts;
- host cache capacity in experts and fraction;
- NVMe and PCIe milliseconds per cold token;
- read batch count from the trace when a simulation is available.

## The real Q4_K_XL numbers

From the verified 9-shard snapshot:

| Quantity | Value |
|---|---:|
| Total tensor bytes | 500.70 GB |
| Routed expert bytes | 288.78 GB |
| Engram/PLE table | 202.76 GB |
| Dense Q8 + F32 | 9.17 GB |
| Experts per layer | 384 |
| Layers | 40 |
| Routed experts total | 15,360 |
| Bytes per expert | 18,800,640 |
| Bytes per layer, all experts | 7.219 GB |
| Routed bytes in a top-6 token | 4.512 GB |

Cold expert traffic alone is 4.51 GB per token, before attention, dense layers,
Engram lookup, or PCIe overhead. At a synthetic 7 GB/s NVMe and perfect overlap,
that is about 0.64 seconds per token, or 1.55 tokens/s as an I/O-only bound. The
real result will be lower. This is why the project must report low-speed
compatibility honestly.

In Q4_K_XL the PLE table is 202.76 GB; in IQ1_S it is 103.00 GB. Neither can be
made resident on a 32-128 GB workstation. The planner keeps it on NVMe and
estimates only the per-ngram lookup traffic separately.

## IQ1_S on 12 GB VRAM and 32 GB RAM

The real 12 GB validation used UD-IQ1_S, not Q4_K_XL:

| Quantity | Value |
|---|---:|
| Total tensor bytes | 216.97 GiB |
| Routed expert bytes | 107.31 GiB |
| Engram/PLE table | 103.00 GiB |
| Dense spine | 5.47 GiB |
| Avg expert bytes | 7.15 MiB |
| Cold top-6 bytes/token | 1.68 GiB |
| KV at 2048 | 18.26 MiB |
| Plan status | LOW-SPEED COMPATIBLE |

The planner reports a 3.26 GiB GPU expert cache after reserves, about 466
experts or 3.03 percent of all experts. The host cache target is 16 GiB, about
2290 experts or 14.91 percent. The cold I/O-only estimate is 257 ms/token; real
CPU expert compute made observed decode much slower, 6-14 s/token.

The reference backend used a smaller 1 GiB GPU MoE cache and still fit below
12 GB. The planner cache number is a native-backend budget, not a claim about
llama.cpp's allocation.

## Automatic device plan

The real backend is not configured with fixed 12/24/32 GB constants. At startup
it reads the free VRAM of the visible GPU, filters the device list through
CUDA_VISIBLE_DEVICES, and computes:

1. context from free VRAM (4096 below 20 GiB, 8192 at 20-28 GiB, 16384 above);
2. 2 GiB compute reserve, 1 GiB safety reserve, and the MLA KV estimate;
3. the dense spine, with a 1.20 safety factor;
4. complete expert layers on the GPU, with a 1.35 safety factor;
5. half of the remaining VRAM as a hot-expert cache, capped at 4 GiB.

For IQ1_S the plans are approximately:

| Free VRAM | Context | GPU experts | Host experts | Cache |
|---:|---:|---:|---:|---:|
| 12 GiB | 4096 | 0 layers | 40 layers | 1143 MiB |
| 16 GiB | 4096 | 1 layer | 39 layers | 1431 MiB |
| 24 GiB | 8192 | 3 layers | 37 layers | 1812 MiB |
| 32 GiB | 16384 | 6 layers | 34 layers | 332 MiB |

The 12 GiB and 24 GiB plans were executed on the RTX 3060 and RTX 4090. The
16 GiB and 32 GiB plans are unit-tested only. The planner uses free, not total,
VRAM; a busy GPU gets a smaller plan.

Host RAM changes page-cache capacity and is reported in the plan, but it does not
change expert arithmetic: host experts remain mmap-backed on NVMe. A 32 GiB host
has enough room for the measured 19.3 GB working set but little page cache; a
256 GiB host keeps much more of the 233 GB store warm.

## Hardware tiers

The 12 GB row is now measured with UD-IQ1_S and the reference backend. The other
rows remain design targets; no 16-48 GB real-model matrix has been run.

| Tier | Placement strategy | Target |
|---|---|---|
| 12 GB VRAM + 32 GB RAM | dense on GPU, all experts host, 1 GiB GPU MoE cache, mmap/lazy | measured with IQ1_S: real tokens, 0.07-0.16 tokens/s |
| 16-24 GB VRAM + 64 GB RAM | resident spine, larger GPU hot cache, active experts in host | single-request text generation |
| 24-32 GB VRAM + 128 GB RAM | larger GPU and host caches, fewer cold reads | improve interactive latency |
| 40-48 GB VRAM + 128 GB RAM | keep most hot experts resident | stable throughput and longer context |

The planner's job is to reject impossible combinations and quantify the slow
ones, not to promise interactivity.

## Example

    rustmoe plan /models/DS41-UD-Q4_K_XL-00001-of-00009.gguf \
      --context 8192 --vram 24GiB --ram 64GiB \
      --gpu-cache 10GiB --host-cache 40GiB

The output includes status COMPATIBLE, LOW-SPEED COMPATIBLE, or INSUFFICIENT,
plus every reserve and diagnostic. Low-speed compatibility is a valid outcome.
Insufficient is a valid outcome too.
