# Benchmark notes

## Current 24 GB baseline

The production profile is now RTX 4090 24 GB, 128K context, 24 threads, 3 GPU
expert layers, 37 host layers, and 0 MiB MoE cache. Short requests measure
4-5 tok/s; the 8-token benchmark measured 4.03-5.16 tok/s. See
[performance.md](performance.md) for the current table and the Strata
comparison. The 12 GB sections below are the historical low-memory validation
and are not the interactive profile.

## Real IQ1_S benchmark on 12 GB

RustMoE now executes real DeepSeek-V4.1-Flash-1017 through the llama.cpp
reference backend. The first measured configuration is deliberately the hard
one: UD-IQ1_S, all routed experts on the host, dense tensors on a 12 GB RTX
3060, 68 GB host RAM, mmap plus lazy loading, 1 GiB GPU MoE cache, context 2048.

Model: 5 shards, 233.0 GB decimal. Reference: unslothai/llama.cpp dsv41-all,
commit d1781c3. The 4090 in the test machine was occupied by another process, so
the run was pinned to the 3060.

### Raw completion

Command:

    rustmoe simulate ...-00001-of-00005.gguf --backend llama --context 2048
      --prompt "The capital of France is" --max-tokens 4 --gpu-layers 999
      --cpu-moe-layers=-1 --moe-cache-mib 1024 --llama-batch 512 --threads 16
      --load-mode 1 --lazy-mode on

Result:

    completion: " Paris. The capital"
    prompt tokens: 5
    prefill: 60.16 s
    decode: 24.38 s for 4 tokens
    TTFT: 60.17 s
    decode throughput: 0.164 tokens/s

### Eight-token continuation

Prompt: "The quick brown fox"

    completion: ". Jumps. Over the lazy dog"
    prompt tokens: 4
    completion tokens: 8
    prefill: 53.19 s
    decode: 76.22 s
    TTFT: 53.20 s
    decode throughput: 0.105 tokens/s
    finish: length

This is a coherent continuation, not just one sampled token. It is still
low-speed.

### Automatic placement, measured

RTX 3060 12 GB, visible through CUDA_VISIBLE_DEVICES=1:

    selected: RTX 3060, 11.83 GiB free
    context: 4096; GPU expert layers: 0; host: 40; cache: 1143 MiB
    headroom: 1.12 GiB; prefill: 10.35 s for one prompt token

RTX 4090 24 GB, visible through CUDA_VISIBLE_DEVICES=0, with 21.60 GiB free:

    selected: RTX 4090
    context: 8192; GPU expert layers: 3; host: 37; cache: 573 MiB
    headroom: 0.57 GiB; observed GPU use: about 17.6 GB
    prefill: 15.72 s for one prompt token

Both runs completed real decoding. The 16 GiB and 32 GiB plans are unit-tested
only; those GPUs were not available.

### 256-token length/stability run

Prompt: "Once", greedy, max_tokens=256, 1 GiB MoE cache.

    completion tokens: 256
    finish: length
    prefill: 22.04 s
    decode: 355.16 s
    decode throughput: 0.72 tokens/s
    backend_cache: hits 0, misses 61440, uploaded 460.91 GB
    observed working set: about 19.3 GB
    output: 256 newline tokens

The 256-token run completed without OOM, corruption, or deadlock. It is a
length/stability test only: the sampled sequence was newline-only because the
one-token prompt "Once" was not enough context. Do not present it as a quality
sample. The zero hit rate over 61,440 requests shows that the 1 GiB cache was
useless for this access pattern and uploaded 460.91 GB over the run.

### 128K thread benchmark

RTX 4090, 128K context, 3 GPU expert layers, 37 host layers, 0 MiB MoE
cache, prompt "The quick brown fox", 8 generated tokens:

| CPU expert threads | prefill | decode | decode tok/s |
|---:|---:|---:|---:|
| 16 | 2.49 s | 2.49 s | 3.21 |
| 24 | 2.46 s | 1.99 s | 4.03 |

24 threads is about 20 percent faster on decode for this CPU. The launcher
defaults to 24 expert threads. This is a short decode test, not a throughput
guarantee.

A second effect was larger. With the same 24 threads, 128K context, and 8-token
chat request:

| MoE cache | decode | decode tok/s |
|---:|---:|---:|
| 279 MiB (automatic) | 4.30 s | 1.86 |
| 0 MiB | 1.94 s | 4.13 |

The small cache had no hits and added bookkeeping/upload work. For 128K, the
launcher now defaults RUSTMOE_MOE_CACHE=0. Set it to a larger value only after
checking the backend cache hit rate.

### MoE cache telemetry

The patched reference backend exposes cumulative MoE cache counters. With a
1 GiB GPU cache:

    one-token run:  hits 0, misses 240, uploaded 1.80 GB
    four-token run: hits 0, misses 1399, uploaded 10.32 GB

The four-token run took 32.40 s prefill and 12.35 s decode. Zero hits means the
1 GiB cache could not retain the selected experts between steps. A 2 GiB cache
produced 221 hits and 1315 misses (14.4 percent hit rate), uploaded 9.92 GB, and
took 33.16 s prefill and 12.04 s decode. The extra GiB bought a measurable hit
rate but almost no wall-clock improvement for four tokens. It remained under the
12 GB device budget. This is a tuning tradeoff, not a free speedup.

### Chat completion

Prompt: "What is the capital of France? Answer briefly."

Result:

    completion: "Paris."
    prompt tokens: 14
    prefill: 159.4 s
    decode: 28.6 s for 2 tokens
    TTFT: 159.4 s
    decode throughput: 0.070 tokens/s

These are real model tokens. The run is low-speed compatibility. CPU expert
GEMM dominates; the 12 GB card cannot hold enough experts to make this
interactive. See docs/real-backend.md for the configuration.

## Runtime-control-plane measurements

The tokenizer and fixture numbers below are not model performance.



## Verified tokenizer measurement

Input shard: real Q4_K_XL shard 1, 5,780,589 bytes, metadata and tokenizer only.

    rustmoe tokenize ...-00001-of-00009.gguf --single-file --text "Hello, world!"

Output:

    ids:     [19923, 14, 2058, 3]
    tokens:  ["Hello", ",", "Ġworld", "!"]
    decoded: "Hello, world!"
    pre:     joyai-llm (exact=false)

This validates vocabulary parsing, byte-level mapping, merges, and decode for a
common ASCII prompt. It does not prove exact compatibility with joyai-llm for
all inputs.

## Tiny fixture measurements

Fixture: synthetic deepseek41 catalog, 28 tensors, 263.43 KiB, one layer.

    rustmoe validate .../tiny-ds41-00001-of-00001.gguf

    28 tensors, 0 errors, 0 warnings

    rustmoe simulate .../tiny-ds41-00001-of-00001.gguf \
      --context 64 --prompt "Hello world" --max-tokens 8 \
      --gpu-cache 128KiB --host-cache 256KiB

Observed on this machine:

    prompt=11 completion=8 finish=Length
    ttft=0.08 ms prefill=0.03 ms decode=0.14 ms total=0.19 ms
    gpu=12 hits, host=0, disk=4 misses, read=96 KiB, batches=2

These numbers only show that the control plane is not introducing a large
overhead into the simulation path. They say nothing about matrix math or disk
bandwidth at model scale.

## Real-model benchmark protocol

For reference and native backend runs, report these separately for each context.
The single 12 GB run above is an initial compatibility measurement, not a
complete matrix.

1. Hardware: GPU model and VRAM, CPU, RAM, NVMe model, driver, OS.
2. Model: exact repo commit, quant directory, all shard SHA-256 values.
3. Backend commit and build flags.
4. Prompt length, number of prefill chunks, generated token count, seed, sampler.
5. Peak VRAM and peak RSS, measured by an external process.
6. GPU cache capacity, host cache capacity, and observed hit/miss counts.
7. NVMe bytes read and H2D bytes per token.
8. First-token latency, prefill tokens/s, decode tokens/s, p50/p95 inter-token
   latency.
9. Output checksum or reference token ids.

Run each point at least three times. Record the median and spread, not a best
run. Test contexts 4K, 8K, 32K, 64K, and 128K. Report 12 GB, 24 GB, and 48 GB
VRAM configurations separately. A configuration that completes but takes seconds
per token must be labelled low-speed compatibility, not interactive.

## Acceptance gates for a native backend

A native backend is not accepted until:

- fixed prompts produce logits within a declared tolerance of the reference;
- at least 256 consecutive tokens decode with correct EOS handling;
- repeated runs show no OOM, corruption, or deadlock;
- every shard is read and every tensor range is validated;
- cache hit and byte counters reconcile with actual I/O;
- the official model semantics are preserved: no dropped experts and no changed
  top-K;
- the benchmark protocol above is reproducible from recorded checksums.
