"""Benchmark tiled matmul vs PyTorch native matmul.

Compares the tiled matrix multiplication reference implementation against
torch.matmul across different matrix sizes. Measures latency, throughput
(GFLOP/s), and arithmetic intensity.

Works on CPU — GPU benchmarks activate automatically when CUDA is available.
"""

import torch
import time
import sys
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from triton_kernels.matmul import triton_matmul, matmul_reference


@dataclass
class MatmulBenchResult:
    """Result of a single matmul benchmark."""
    name: str
    m: int
    n: int
    k: int
    latency_ms: float
    gflops: float
    arithmetic_intensity: float


def compute_matmul_flops(m: int, n: int, k: int) -> int:
    """Compute FLOPs for matrix multiplication C[M,N] = A[M,K] @ B[K,N].

    Args:
        m: Rows of output.
        n: Columns of output.
        k: Inner dimension.

    Returns:
        Total floating-point operations (2*M*N*K for multiply-add).
    """
    return 2 * m * n * k


def compute_arithmetic_intensity(m: int, n: int, k: int, dtype_bytes: int = 4) -> float:
    """Compute arithmetic intensity (FLOPs / bytes transferred).

    Assumes each matrix is read once from memory (no tiling benefit at this level).

    Args:
        m: Rows of A and C.
        n: Columns of B and C.
        k: Inner dimension.
        dtype_bytes: Bytes per element (4 for float32).

    Returns:
        Arithmetic intensity in FLOPs/byte.
    """
    flops = 2 * m * n * k
    bytes_transferred = (m * k + k * n + m * n) * dtype_bytes
    return flops / bytes_transferred


def benchmark_fn(fn, args: tuple, warmup: int = 2, repeats: int = 5) -> float:
    """Time a function with warmup.

    Args:
        fn: Function to benchmark.
        args: Arguments tuple.
        warmup: Warmup iterations.
        repeats: Timed iterations.

    Returns:
        Average latency in milliseconds.
    """
    for _ in range(warmup):
        fn(*args)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return sum(times) / len(times)


def run_benchmarks(
    sizes: Optional[list[tuple[int, int, int]]] = None,
    block_size: int = 32,
) -> list[MatmulBenchResult]:
    """Run matmul benchmarks across matrix sizes.

    Args:
        sizes: List of (M, N, K) tuples. Defaults to common sizes.
        block_size: Tile size for the tiled implementation.

    Returns:
        List of benchmark results.
    """
    if sizes is None:
        sizes = [
            (64, 64, 64),
            (128, 128, 128),
            (256, 256, 256),
            (512, 512, 512),
        ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: list[MatmulBenchResult] = []

    print(f"Device: {device}")
    print(f"Block size: {block_size}")
    print(f"{'Method':<25} {'Size':>15} {'Latency':>10} {'GFLOP/s':>10} {'AI':>8}")
    print("-" * 72)

    for m, n, k in sizes:
        torch.manual_seed(42)
        a = torch.randn(m, k, device=device)
        b = torch.randn(k, n, device=device)

        flops = compute_matmul_flops(m, n, k)
        ai = compute_arithmetic_intensity(m, n, k)
        size_str = f"{m}x{k} @ {k}x{n}"

        # Tiled reference
        latency_tiled = benchmark_fn(
            matmul_reference, (a, b, block_size, block_size, block_size)
        )
        gflops_tiled = flops / (latency_tiled * 1e6)
        results.append(MatmulBenchResult("Tiled Reference", m, n, k, latency_tiled, gflops_tiled, ai))
        print(f"{'Tiled Reference':<25} {size_str:>15} {latency_tiled:>8.2f}ms {gflops_tiled:>9.2f} {ai:>7.1f}")

        # PyTorch native (cuBLAS on GPU, MKL/OpenBLAS on CPU)
        latency_native = benchmark_fn(torch.matmul, (a, b))
        gflops_native = flops / (latency_native * 1e6)
        results.append(MatmulBenchResult("torch.matmul", m, n, k, latency_native, gflops_native, ai))
        print(f"{'torch.matmul':<25} {size_str:>15} {latency_native:>8.2f}ms {gflops_native:>9.2f} {ai:>7.1f}")

        # Verify correctness
        result_tiled = matmul_reference(a, b, block_size, block_size, block_size)
        result_native = torch.matmul(a, b)
        max_err = (result_tiled - result_native).abs().max().item()
        print(f"{'  → max error':<25} {max_err:.2e}")
        print()

    return results


def print_summary(results: list[MatmulBenchResult]) -> None:
    """Print performance summary.

    Args:
        results: List of benchmark results.
    """
    print("=== Summary ===\n")
    print("Arithmetic intensity increases with matrix size, moving the operation")
    print("from memory-bound toward compute-bound on the roofline model.\n")

    sizes = sorted(set((r.m, r.n, r.k) for r in results))
    for m, n, k in sizes:
        tiled = next((r for r in results if r.name == "Tiled Reference" and r.m == m), None)
        native = next((r for r in results if r.name == "torch.matmul" and r.m == m), None)
        if tiled and native:
            ratio = native.gflops / tiled.gflops if tiled.gflops > 0 else 0
            print(f"  {m}x{k}: torch.matmul is {ratio:.1f}x faster "
                  f"(AI={tiled.arithmetic_intensity:.1f} FLOP/byte)")


if __name__ == "__main__":
    print("=== Matrix Multiplication Benchmark ===\n")
    results = run_benchmarks()
    print_summary(results)
