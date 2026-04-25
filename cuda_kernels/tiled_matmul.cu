/**
 * tiled_matmul.cu — Tiled matrix multiplication with shared memory.
 *
 * Demonstrates the tiling optimization for matrix multiplication:
 * - Naive: each thread reads entire rows/columns from global memory
 * - Tiled: threads cooperatively load tiles into shared memory, then
 *   compute from the fast on-chip SRAM instead of slow HBM
 *
 * For tile size T, data reuse is T-fold: each element loaded into shared
 * memory is used T times, reducing global memory traffic by T.
 *
 * Shared memory padding (+1 column) avoids bank conflicts on stores.
 *
 * Compile: nvcc -o tiled_matmul tiled_matmul.cu
 * Run:     ./tiled_matmul
 */

#include <stdio.h>
#include <stdlib.h>
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

#define TILE_SIZE 16

/**
 * Naive matrix multiplication: C = A * B
 * Each thread computes one element of C by reading a full row of A
 * and a full column of B from global memory.
 *
 * Arithmetic intensity: O(1) — each element read once per output element.
 */
__global__ void matmul_naive(
    const float* A, const float* B, float* C, int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/**
 * Tiled matrix multiplication using shared memory.
 *
 * Each thread block loads TILE_SIZE x TILE_SIZE tiles of A and B into
 * shared memory, computes partial products, and iterates over K.
 *
 * Shared memory is padded (+1) to avoid bank conflicts:
 * without padding, threads in the same warp accessing the same column
 * of a TILE_SIZE-wide array hit the same bank, serializing access.
 *
 * Arithmetic intensity: O(TILE_SIZE) — each loaded element reused TILE_SIZE times.
 */
__global__ void matmul_tiled(
    const float* A, const float* B, float* C, int M, int N, int K
) {
    // Padded shared memory to avoid bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;

    // Iterate over tiles along K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Cooperative load: each thread loads one element of each tile
        int a_col = t * TILE_SIZE + tx;
        int b_row = t * TILE_SIZE + ty;

        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // Compute partial product from shared memory
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/**
 * Initialize matrix with deterministic values for verification.
 */
void init_matrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(i % 7) * 0.1f;  // Small values to avoid overflow
    }
}

/**
 * CPU reference matrix multiplication for verification.
 */
void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * Verify GPU result against CPU reference.
 */
int verify(const float* gpu_result, const float* cpu_result, int n, float tol) {
    int errors = 0;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(gpu_result[i] - cpu_result[i]);
        float rel = diff / (fabsf(cpu_result[i]) + 1e-8f);
        if (rel > tol) {
            errors++;
            if (errors <= 3) {
                printf("  Mismatch at %d: gpu=%.6f cpu=%.6f rel_err=%.2e\n",
                       i, gpu_result[i], cpu_result[i], rel);
            }
        }
    }
    return errors;
}

int main() {
    printf("=== CUDA Tiled Matrix Multiplication ===\n\n");

    int M = 512, N = 512, K = 512;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // Host allocation
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C_naive = (float*)malloc(bytes_C);
    float* h_C_tiled = (float*)malloc(bytes_C);
    float* h_C_cpu = (float*)malloc(bytes_C);

    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // CPU reference
    matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);

    // Device allocation
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Benchmark naive kernel
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);  // warmup
    CUDA_CHECK(cudaEventRecord(start));
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_naive;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, bytes_C, cudaMemcpyDeviceToHost));

    // Benchmark tiled kernel
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);  // warmup
    CUDA_CHECK(cudaEventRecord(start));
    matmul_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_tiled;
    CUDA_CHECK(cudaEventElapsedTime(&ms_tiled, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, bytes_C, cudaMemcpyDeviceToHost));

    // Results
    double gflops = 2.0 * M * N * K / 1e9;
    printf("Matrix size: %d x %d x %d\n", M, N, K);
    printf("Tile size: %d\n\n", TILE_SIZE);

    int err_naive = verify(h_C_naive, h_C_cpu, M * N, 1e-3f);
    printf("Naive:  %.3f ms  %.1f GFLOP/s  errors=%d\n",
           ms_naive, gflops / (ms_naive * 1e-3), err_naive);

    int err_tiled = verify(h_C_tiled, h_C_cpu, M * N, 1e-3f);
    printf("Tiled:  %.3f ms  %.1f GFLOP/s  errors=%d\n",
           ms_tiled, gflops / (ms_tiled * 1e-3), err_tiled);

    printf("\nSpeedup (tiled/naive): %.2fx\n", ms_naive / ms_tiled);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);
    free(h_C_cpu);

    return 0;
}
