# Performance notes and Strata comparison

## Measured on the RTX 4090 / 64 GB host

Model: DeepSeek-V4.1-Flash-1017 UD-IQ1_S on C: NVMe, 128K context, 3 GPU
expert layers, 37 host expert layers, 24 CPU threads, 0 MiB MoE cache.

| Test | Prefill | Decode | Decode tok/s |
|---|---:|---:|---:|
| 8 tokens, 128K, 16 threads | 2.49 s | 2.49 s | 3.21 |
| 8 tokens, 128K, 24 threads | 2.46 s | 1.99 s | 4.03 |
| 8 tokens, 128K, 24 threads, 279 MiB cache | 2.46 s | 4.30 s | 1.86 |
| 8 tokens, 128K, 24 threads, cache 0 | 2.46 s | 1.94 s | 4.13 |
| short 8-token request | 3.98 s | 1.55 s | 5.16 |
| 64K, 4 GPU layers | - | 2.48 s | 3.22 |

The practical range is 4-5 tok/s for short requests. Longer prompts, generated
code, and thermal pressure move it toward 3-4 tok/s.

The small MoE cache result is counterintuitive but repeatable: 279 MiB had no
hits and added bookkeeping/upload work. The 128K launcher defaults to 24 threads
and MoE cache 0. Set a cache only after checking backend hit rate.

The 64K + 4-layer configuration was slower than 128K + 3 layers. More offloaded
layers are not automatically faster because the 4090 has little headroom left
for compute and the fourth layer changes memory traffic.

## Why Strata is much faster

Strata's current log on this host:

    prompt 140705 tokens
      = 140425 reused + 280 read in 549 ms (509.8 tok/s)
    318 generated in 3540 ms (89.8 tok/s)
    decode expert cache: 86.5% hits
    suffix drafts: 122/128 accepted

Another run:

    prompt 134177 tokens
      = 134151 reused + 26 read in 218 ms (119.3 tok/s)
    893 generated in 6672 ms (133.8 tok/s)
    drafts accepted 601 of 770
    decode expert cache hit rate 88.5%

This is not the same workload:

| | Strata / Qwen | RustMoE / DeepSeek |
|---|---|---|
| Parameters | ~125B | 748B |
| Weight bytes | 35-55 GB | 217 GiB |
| GPU expert placement | hot expert cache 80-90% hit | 3 complete layers |
| Speculative decoding | MTP 4 drafts | none |
| Prompt prefix reuse | 140K tokens | none |
| KV | int8 | F16 |
| Decode | 90-140 tok/s | 4-5 tok/s |

Four effects compound:

1. Six times fewer parameters and far less weight traffic.
2. Per-expert hotness keeps 80-90% of routed experts on the GPU. RustMoE can
   only place 3 of 40 complete expert layers in 24 GB.
3. MTP speculative decoding accepts 70-85% of drafts, so one verification pass
   yields multiple tokens.
4. Prompt reuse avoids re-prefilling 140K tokens. RustMoE re-prefills the prompt
   per request.

The monitor UI is not a speed feature. RustMoE's /status is already comparable.

## Borrowable ideas and realistic gains

| Idea | RustMoE path | Expected gain |
|---|---|---:|
| Prefix KV reuse | keep KV, rewind to common prefix | TTFT and agent iteration, not decode |
| MTP/DSpark speculative | draft head plus verify | 1.3-1.8x decode |
| Per-expert hot cache | router trace plus custom cache | 1.5-3x, but only 3 GB headroom |
| Fused int8/int4 kernels | custom CUDA | 1.1-1.3x |
| Smaller model | Qwen/Strata | 10-20x |

Even with all four RustMoE changes, the realistic 748B ceiling is about 8-12
tok/s, not 100. The model and the amount of GPU-resident expert data set the
limit.

## Fast agent

The fast agent removes the model bottleneck from the tool loop:

    Strata model call: 90-140 tok/s
    RustMoE local tools: file/shell/skills, no VRAM
    agent URL: http://127.0.0.1:8081/agent

A two-step file-creation task completes in seconds. The agent still spends time
on shell commands, file I/O, and the model's JSON action generation, but it no
longer waits on 4-5 tok/s DeepSeek decode.

## Practical knobs

- Keep the model on NVMe, not the HDD.
- Use 24 expert threads on the i9-14900KF.
- At 128K, use cache 0 unless the backend reports real hits.
- Do not assume a fourth GPU layer is faster; measure.
- --llama-batch and prefill chunk affect prompt ingestion, not single-token
  decode.
- Stop Strata before starting RustMoE. It uses 23.5 GB of the 24 GB card.
- Use engine-fast.bat for interactive code work and engine-deep.bat for
  DeepSeek-specific tasks.

## References

- Strata project: https://github.com/Niko1221/Strata
- Strata model/placement description:
  https://github.com/Niko1221/Strata#how-does-it-work
- Dual engine: docs/dual-engine.md
