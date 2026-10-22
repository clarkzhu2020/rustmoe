# CUDA kernels

This directory is a development target, not a complete inference backend.
RustMoE v0.1 does not link these kernels. The runtime currently ships a
simulation backend so scheduling, caching, validation, API, and streaming can
be tested before numerical kernels are trusted.

## What is here

- mxfp4_gemv.cu: a readable MXFP4 GEMV kernel for the 17-byte block layout
  used by the DeepSeek-V4.1 Q4_K_XL expert shards. It decodes E8M0 scales and
  E2M1 nibbles directly into a dot product.
- build.ps1: compiles the kernel to a cubin for SM 89.

## What is deliberately absent

- MLA decode and compressed attention kernels.
- Engram n-gram lookup and gate kernels.
- Router sqrtsoftplus and top-6 kernels.
- Superposed hyper-connection kernels.
- Device-side sampling and logits retention.
- Multi-stream transfer overlap and CUDA graphs.
- Numerical parity tests against llama.cpp.

Those are Phase 4 items. Adding them without tensor-level parity tests would
make failures impossible to localize.

## Build

    cd kernels/cuda
    ./build.ps1

or:

    nvcc -O3 -arch=sm_89 -cubin mxfp4_gemv.cu -o mxfp4_gemv.cubin

The kernel is unvalidated against a reference implementation. Treat it as a
starting point, not a verified fast path.
