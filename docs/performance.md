# Performance notes and Strata comparison

## Measured on the RTX 4090 / 64 GB host

Model: DeepSeek-V4.1-Flash-1017 UD-IQ1_S on C: NVMe, 128K context, 3 GPU
expert layers, 37 host expert layers.

| Change | Decode result |
|---|---:|
| 16 CPU expert threads | 3.21 tokens/s |
| 24 CPU expert threads | 4.03 tokens/s |
| automatic 279 MiB MoE cache, 24 threads | 1.86 tokens/s |
| cache 0, 24 threads | 4.13 tokens/s |

The cache result is counterintuitive but repeatable: the 279 MiB cache had no
hits and added bookkeeping/upload work. The launcher now defaults to 24 threads
and RUSTMOE_MOE_CACHE=0 for 128K. The live dashboard reports 3-4 tokens/s
depending on prompt and cache state.

## Why Strata is much faster

Strata (Qwen3.8-Flash-Next, 125B class) reports 50-100 tokens/s on 12-24 GB
cards. The main differences are model size and placement, not the dashboard:

- Strata's model is roughly 125B parameters and 35-55 GB; DeepSeek-V4.1 is 748B
  and 217 GiB even at IQ1_S.
- Strata keeps hot experts on the GPU and all experts in RAM. On this host only
  3 of 40 DeepSeek expert layers fit in 24 GB; the other 37 run on the CPU.
- Strata uses MTP speculative decoding (draft head plus verification), hot
  experts in VRAM, fused int8 tensor-core prompt kernels, KV streaming above 64K,
  and optional 8/4-bit KV. RustMoE has none of the first four and uses F16 KV.
- Strata's monitor is a UI feature, not a speed feature. RustMoE now has a
  comparable /status dashboard.

## Borrowable ideas

1. Per-expert GPU hotness instead of whole-layer CPU overrides. This needs router
   traces and a custom expert cache in the llama.cpp path; it is the largest
   structural borrow from Strata.
2. Speculative decoding. DeepSeek metadata includes DSpark, but RustMoE does not
   execute it. A draft model plus verification can help decode if acceptance is
   high.
3. Smaller checkpoints or a coder distill. A 125B-class model is inherently much
   faster than 748B on the same card.
4. Parallel request batching. This improves aggregate throughput, but a single
   code-agent request does not benefit and each answer gets slower on 24 GB.

## Practical knobs

- Keep the model on NVMe, not the HDD.
- Use 24 expert threads on the i9-14900KF.
- At 128K, use cache 0 unless the backend reports real hits.
- A 64K + 4 GPU-layer test measured 3.22 tokens/s (decode 2.48 s for 8
  tokens), slower than 128K + 3 layers at 4.1-5.2. Do not assume the fourth layer
  is a win; measure it on the actual card.
- 32K/16K frees more VRAM for expert layers or a cache, but most coding agents
  require at least 64K.
- --llama-batch and prefill chunk affect prompt ingestion, not single-token
  decode speed.

## References

- Strata project: https://github.com/Niko1221/Strata
- Strata model/placement description:
  https://github.com/Niko1221/Strata#how-does-it-work
