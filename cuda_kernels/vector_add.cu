/**
 * vector_add.cu — Basic CUDA vector addition kernel.
 *
 * Demonstrates the fundamental CUDA programming model:
 * - Grid/block/thread hierarchy
 * - Global memory access patterns
 * - Boundary checking for non-aligned sizes
 * - Host-device memory transfer
 *
 * This is the "hello world" of CUDA: each thread computes one element
 * of the output vector C[i] = A[i] + B[i].
 *
 * Compile: nvcc -o vector_add vector_add.cu
 * Run:     ./vector_add
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/**
 * CUDA kernel: element-wise vector addition.
 *
 * Each thread computes one output element. The grid is 1D with
 * enough blocks to cover all N elements.
 *
 * @param a     Input vector A (device pointer)
 * @param b     Input vector B (device pointer)
 * @param c     Output vector C = A + B (device pointer)
 * @param n     Number of elements
 */
__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * Vectorized kernel using float4 for coalesced memory access.
 * Processes 4 elements per thread, improving memory throughput.
 */
__global__ void vector_add_float4_kernel(
    const float4* a, const float4* b, float4* c, int n_float4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_float4) {
        float4 va = a[idx];
        float4 vb = b[idx];
        c[idx] = make_float4(
            va.x + vb.x,
            va.y + vb.y,
            va.z + vb.z,
            va.w + vb.w
        );
    }
}

/**
 * Host function: allocate, transfer, launch kernel, verify results.
 */
void run_vector_add(int n) {
    size_t bytes = n * sizeof(float);

    // Allocate host memory
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);

    // Initialize with test data
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(2 * i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // Copy host -> device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    // Timing with CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vector_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Copy device -> host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float expected = h_a[i] + h_b[i];
        if (fabsf(h_c[i] - expected) > 1e-5f) {
            errors++;
            if (errors <= 5) {
                printf("Mismatch at %d: got %f, expected %f\n", i, h_c[i], expected);
            }
        }
    }

    printf("Vector Add: n=%d, time=%.3f ms, errors=%d\n", n, ms, errors);
    printf("Effective bandwidth: %.2f GB/s\n",
           3.0 * bytes / (ms * 1e6));  // 2 reads + 1 write

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
}

int main() {
    printf("=== CUDA Vector Addition ===\n\n");

    // Print device info
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory bandwidth: %.0f GB/s\n\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);

    // Run at different sizes
    int sizes[] = {1024, 1 << 16, 1 << 20, 1 << 24};
    for (int i = 0; i < 4; i++) {
        run_vector_add(sizes[i]);
    }

    return 0;
}
