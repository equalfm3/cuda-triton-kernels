"""Benchmark flash attention vs PyTorch native attention.

Compares the tiled flash attention implementation against standard
scaled dot-product attention across different sequence lengths.
Works on CPU with reference implementations — GPU benchmarks run
automatically when CUDA is available.

Metrics: latency (ms), throughput (GFLOP/s), memory usage.
"""

import torch
import time
import math
import sys
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from triton_kernels.flash_attention import flash_attention, _standard_attention


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    seq_len: int
    latency_ms: float
    gflops: float
    memory_bytes: int


def compute_attention_flops(batch: int, heads: int, seq_len: int, head_dim: int) -> int:
    """Compute FLOPs for attention: QK^T + softmax + AV.

    Args:
        batch: Batch size.
        heads: Number of attention heads.
        seq_len: Sequence length.
        head_dim: Head dimension.

    Returns:
        Total floating-point operations.
    """
    # QK^T: batch * heads * seq_len * seq_len * (2 * head_dim)
    qk_flops = batch * heads * seq_len * seq_len * 2 * head_dim
    # Softmax: ~5 ops per element (max, sub, exp, sum, div)
    softmax_flops = batch * heads * seq_len * seq_len * 5
    # AV: batch * heads * seq_len * head_dim * (2 * seq_len)
    av_flops = batch * heads * seq_len * head_dim * 2 * seq_len
    return qk_flops + softmax_flops + av_flops


def benchmark_fn(
    fn,
    args: tuple,
    warmup: int = 3,
    repeats: int = 10,
) -> float:
    """Time a function with warmup and averaging.

    Args:
        fn: Function to benchmark.
        args: Arguments to pass to fn.
        warmup: Number of warmup iterations.
        repeats: Number of timed iterations.

    Returns:
        Average latency in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn(*args)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return sum(times) / len(times)


def run_benchmarks(
    batch: int = 2,
    heads: int = 8,
    head_dim: int = 64,
    seq_lengths: Optional[list[int]] = None,
) -> list[BenchmarkResult]:
    """Run attention benchmarks across sequence lengths.

    Args:
        batch: Batch size.
        heads: Number of attention heads.
        head_dim: Dimension per head.
        seq_lengths: List of sequence lengths to test.

    Returns:
        List of benchmark results.
    """
    if seq_lengths is None:
        seq_lengths = [64, 128, 256, 512]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: list[BenchmarkResult] = []

    print(f"Device: {device}")
    print(f"Config: batch={batch}, heads={heads}, head_dim={head_dim}")
    print(f"{'Method':<25} {'SeqLen':>8} {'Latency':>10} {'GFLOP/s':>10} {'Memory':>10}")
    print("-" * 70)

    for seq_len in seq_lengths:
        torch.manual_seed(42)
        q = torch.randn(batch, heads, seq_len, head_dim, device=device)
        k = torch.randn(batch, heads, seq_len, head_dim, device=device)
        v = torch.randn(batch, heads, seq_len, head_dim, device=device)

        flops = compute_attention_flops(batch, heads, seq_len, head_dim)

        # Standard attention
        latency_std = benchmark_fn(_standard_attention, (q, k, v))
        gflops_std = flops / (latency_std * 1e6)
        mem_std = batch * heads * seq_len * seq_len * 4  # attention matrix
        results.append(BenchmarkResult("Standard Attention", seq_len, latency_std, gflops_std, mem_std))
        print(f"{'Standard Attention':<25} {seq_len:>8} {latency_std:>8.2f}ms {gflops_std:>9.1f} {mem_std/1024:>8.1f}KB")

        # Flash attention (tiled)
        block_q = min(32, seq_len)
        block_kv = min(32, seq_len)
        latency_flash = benchmark_fn(flash_attention, (q, k, v, block_q, block_kv))
        gflops_flash = flops / (latency_flash * 1e6)
        mem_flash = batch * heads * seq_len * head_dim * 4  # no attention matrix
        results.append(BenchmarkResult("Flash Attention", seq_len, latency_flash, gflops_flash, mem_flash))
        print(f"{'Flash Attention':<25} {seq_len:>8} {latency_flash:>8.2f}ms {gflops_flash:>9.1f} {mem_flash/1024:>8.1f}KB")

        # PyTorch native (F.scaled_dot_product_attention)
        try:
            latency_native = benchmark_fn(
                torch.nn.functional.scaled_dot_product_attention, (q, k, v)
            )
            gflops_native = flops / (latency_native * 1e6)
            results.append(BenchmarkResult("PyTorch Native", seq_len, latency_native, gflops_native, mem_flash))
            print(f"{'PyTorch Native':<25} {seq_len:>8} {latency_native:>8.2f}ms {gflops_native:>9.1f} {mem_flash/1024:>8.1f}KB")
        except (AttributeError, RuntimeError):
            pass  # F.scaled_dot_product_attention not available in older PyTorch

        print()

    return results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a summary comparing methods at each sequence length.

    Args:
        results: List of benchmark results.
    """
    print("\n=== Summary ===\n")

    seq_lengths = sorted(set(r.seq_len for r in results))
    for seq_len in seq_lengths:
        seq_results = [r for r in results if r.seq_len == seq_len]
        if len(seq_results) >= 2:
            std = next((r for r in seq_results if "Standard" in r.name), None)
            flash = next((r for r in seq_results if "Flash" in r.name), None)
            if std and flash:
                speedup = std.latency_ms / flash.latency_ms if flash.latency_ms > 0 else 0
                mem_ratio = std.memory_bytes / flash.memory_bytes if flash.memory_bytes > 0 else 0
                print(f"SeqLen={seq_len}: Flash is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}, "
                      f"{mem_ratio:.1f}x less memory")


if __name__ == "__main__":
    print("=== Attention Benchmark ===\n")
    results = run_benchmarks()
    print_summary(results)
