# RustMoE documentation

Start with the repository [README](../README.md).

## Operating modes

- [Dual engine](dual-engine.md) - switch Strata fast and RustMoE deep, plus the
  8081 fast agent.
- [Performance](performance.md) - measured 748B TPS, why Strata is faster, and
  realistic optimization limits.
- [Troubleshooting](troubleshooting.md) - ports, loading, auth, agent, TPS.

## Agent and skills

- [Dev Agent](agent.md) - local tools, project isolation, resume, security.
- [Skills](skills.md) - the 17 installed packs and how to add one.

## API and UI

- [HTTP API](api.md) - OpenAI/SSE compatibility, agent endpoints, errors.
- [Status dashboard](status-dashboard.png)
- [Chat page](chat.png)
- [Agent page](agent.png)

## Model and runtime

- [Architecture](architecture.md) - crates and runtime boundaries.
- [Real backend](real-backend.md) - llama.cpp setup, flags, placement.
- [Memory planning](memory-planning.md) - VRAM/RAM/NVMe budgets.
- [Model snapshot](model-snapshot.md) - verified DeepSeek facts.
- [Supported formats](supported-formats.md) - GGUF and quantization.
- [Benchmark](benchmark.md) - detailed historical measurements.

## Quick commands

    engine-status.bat
    engine-fast.bat
    engine-deep.bat

    cargo test --workspace
    cargo clippy --workspace --all-targets
