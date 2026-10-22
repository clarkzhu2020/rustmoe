// Reference MXFP4 GEMV kernel for RustMoE.
//
// This file is a standalone kernel development target. It is not linked into
// the Rust runtime yet. Do not describe it as a complete DeepSeek-V4.1 backend.
//
// Layout: row-major MXFP4, 32 values per 17-byte block:
//   byte 0     E8M0 scale, value = 2^(e - 127)
//   bytes 1-16 32 E2M1 nibbles; low nibbles are columns 0..15,
//               high nibbles are columns 16..31.
//
// Build: nvcc -O3 -arch=sm_89 -cubin mxfp4_gemv.cu -o mxfp4_gemv.cubin

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

__device__ __forceinline__ float e8m0_to_float(uint8_t e) {
    return ldexpf(1.0f, (int)e - 127);
}

__device__ __forceinline__ float e2m1_nibble(uint8_t q) {
    const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float v = table[q & 0x7];
    return (q & 0x8) ? -v : v;
}

__global__ void mxfp4_gemv_kernel(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int cols) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const int blocks_per_row = cols / 32;
    const uint8_t* wrow = weights + (size_t)row * blocks_per_row * 17;
    float acc = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const int block = col / 32;
        const int within = col % 32;
        const uint8_t* b = wrow + block * 17;
        const float scale = e8m0_to_float(b[0]);
        const uint8_t byte = b[1 + (within & 15)];
        const uint8_t q = (within < 16) ? (byte & 0x0f) : (byte >> 4);
        acc += scale * e2m1_nibble(q) * x[col];
    }

    // Block reduction.
    __shared__ float smem[32];
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        smem[warp] = acc;
    }
    __syncthreads();
    if (warp == 0) {
        acc = (lane < (blockDim.x + 31) / 32) ? smem[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            y[row] = acc;
        }
    }
}

extern "C" cudaError_t rustmoe_mxfp4_gemv(
    const uint8_t* weights,
    const float* x,
    float* y,
    int rows,
    int cols,
    cudaStream_t stream) {
    if (cols <= 0 || cols % 32 != 0) {
        return cudaErrorInvalidValue;
    }
    mxfp4_gemv_kernel<<<rows, 256, 0, stream>>>(weights, x, y, rows, cols);
    return cudaGetLastError();
}
