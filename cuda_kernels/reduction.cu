/**
 * reduction.cu — Parallel reduction with shared memory.
 *
 * Implements progressively optimized reduction algorithms:
 * 1. Naive: interleaved addressing (bank conflicts)
 * 2. Sequential addressing: no bank conflicts
 * 3. Warp-level unrolling: avoid __syncthreads for last warp
 * 4. Multiple elements per thread: better instruction-level parallelism
 *
 * Reduction is the canonical example of a memory-bound kernel where
 * shared memory and warp-level optimizations make a large difference.
 *
 * Compile: nvcc -o reduction reduction.cu
 * Run:     ./reduction
 */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define BLOCK_SIZE 256

/**
 * Kernel 1: Naive reduction with interleaved addressing.
 * Has shared memory bank conflicts due to strided access.
 */
__global__ void reduce_naive(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load from global to shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Interleaved reduction (has bank conflicts)
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

/**
 * Kernel 2: Sequential addressing — eliminates bank conflicts.
 * Threads access contiguous shared memory locations.
 */
__global__ void reduce_sequential(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Sequential addressing: no bank conflicts
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

/**
 * Warp-level reduction using shuffle instructions.
 * No __syncthreads needed within a warp (32 threads execute in lockstep).
 */
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/**
 * Kernel 3: Warp-unrolled reduction with multiple elements per thread.
 * Each thread loads and sums multiple elements before the tree reduction.
 */
__global__ void reduce_warp_unrolled(
    const float* input, float* output, int n, int elements_per_thread
) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int base_idx = blockIdx.x * blockDim.x * elements_per_thread + tid;

    // Each thread accumulates multiple elements
    float sum = 0.0f;
    for (int i = 0; i < elements_per_thread; i++) {
        unsigned int idx = base_idx + i * blockDim.x;
        if (idx < n) {
            sum += input[idx];
        }
    }
    sdata[tid] = sum;
    __syncthreads();

    // Tree reduction in shared memory until we reach warp size
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction (no sync needed)
    if (tid < 32) {
        float val = sdata[tid];
        if (blockDim.x >= 64) val += sdata[tid + 32];
        val = warp_reduce_sum(val);
        if (tid == 0) output[blockIdx.x] = val;
    }
}

/**
 * Host-side two-pass reduction.
 * Pass 1: reduce within blocks. Pass 2: reduce block results.
 */
float gpu_reduce(const float* d_input, int n, int kernel_type) {
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t smem_bytes = BLOCK_SIZE * sizeof(float);

    float* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, grid_size * sizeof(float)));

    // Pass 1: per-block reduction
    switch (kernel_type) {
        case 0:
            reduce_naive<<<grid_size, BLOCK_SIZE, smem_bytes>>>(d_input, d_partial, n);
            break;
        case 1:
            reduce_sequential<<<grid_size, BLOCK_SIZE, smem_bytes>>>(d_input, d_partial, n);
            break;
        case 2: {
            int elems_per_thread = 4;
            int effective_grid = (n + BLOCK_SIZE * elems_per_thread - 1) / (BLOCK_SIZE * elems_per_thread);
            reduce_warp_unrolled<<<effective_grid, BLOCK_SIZE, smem_bytes>>>(
                d_input, d_partial, n, elems_per_thread
            );
            grid_size = effective_grid;
            break;
        }
    }

    // Pass 2: reduce partial sums (always use sequential for simplicity)
    float* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(float)));
    if (grid_size > 1) {
        reduce_sequential<<<1, BLOCK_SIZE, smem_bytes>>>(d_partial, d_result, grid_size);
    } else {
        CUDA_CHECK(cudaMemcpy(d_result, d_partial, sizeof(float), cudaMemcpyDeviceToDevice));
    }

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFree(d_result));
    return result;
}

int main() {
    printf("=== CUDA Parallel Reduction ===\n\n");

    int n = 1 << 20;  // 1M elements
    size_t bytes = n * sizeof(float);

    // Initialize host data
    float* h_input = (float*)malloc(bytes);
    double cpu_sum = 0.0;
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;  // Simple test: sum should equal n
        cpu_sum += h_input[i];
    }

    // Allocate and copy to device
    float* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    const char* kernel_names[] = {
        "Naive (interleaved)",
        "Sequential addressing",
        "Warp-unrolled (4 elem/thread)"
    };

    // Benchmark each kernel
    for (int k = 0; k < 3; k++) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // Warmup
        gpu_reduce(d_input, n, k);

        CUDA_CHECK(cudaEventRecord(start));
        float result = gpu_reduce(d_input, n, k);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

        float error = fabsf(result - (float)cpu_sum);
        printf("%-35s sum=%.0f  error=%.2e  time=%.3f ms  BW=%.1f GB/s\n",
               kernel_names[k], result, error, ms,
               bytes / (ms * 1e6));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    CUDA_CHECK(cudaFree(d_input));
    free(h_input);
    return 0;
}
