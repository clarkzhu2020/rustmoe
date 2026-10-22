# Tiny DeepSeek-V4.1 fixture

This directory contains a generated, synthetic GGUF used by the test suite and
the README quick start. It is not a language model and is not derived from model
weights.

- architecture: deepseek41
- 1 layer, hidden 64, 2 attention heads, key length 8
- 4 routed experts, top-2, one shared expert
- 256-token byte-level vocabulary, no merges
- 28 tensors, all F32
- 277,376 bytes, one split file

It exercises metadata parsing, split handling, tokenizer loading, tensor shape
validation, expert layout, scheduler cache accounting, CLI output, HTTP, SSE,
auth, and metrics. It does not produce meaningful text.

Regenerate with:

    python scripts/make_tiny_fixture.py

The generator writes the file deterministically. The fixture is committed so
the quick start does not require Python.
