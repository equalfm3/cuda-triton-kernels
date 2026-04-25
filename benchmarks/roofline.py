"""Roofline model analysis for GPU kernels.

Computes arithmetic intensity for each kernel type and plots theoretical
vs achieved performance against the hardware roofline. This reveals whether
a kernel is memory-bound or compute-bound and how much optimization headroom
remains.

The roofline model: P = min(π, β * I)
  - π: peak compute (FLOP/s)
  - β: peak memory bandwidth (bytes/s)
  - I: arithmetic intensity (FLOPs / bytes transferred)
  - Ridge point: I* = π / β (transition from memory-bound to compute-bound)

Works on CPU with estimated hardware parameters. Uses actual GPU specs
when CUDA is available.
"""

import torch
import time
import math
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class HardwareSpec:
    """Hardware performance specifications."""
    name: str
    peak_flops: float  # FLOP/s (float32)
    peak_bandwidth: float  # bytes/s
    ridge_point: float = field(init=False)

    def __post_init__(self) -> None:
        self.ridge_point = self.peak_flops / self.peak_bandwidth


@dataclass
class KernelProfile:
    """Performance profile of a single kernel."""
    name: str
    flops: int
    bytes_transferred: int
    latency_ms: float
    arithmetic_intensity: float = field(init=False)
    achieved_flops: float = field(init=False)
    achieved_bandwidth: float = field(init=False)

    def __post_init__(self) -> None:
        self.arithmetic_intensity = self.flops / max(self.bytes_transferred, 1)
        self.achieved_flops = self.flops / (self.latency_ms * 1e-3) if self.latency_ms > 0 else 0
        self.achieved_bandwidth = self.bytes_transferred / (self.latency_ms * 1e-3) if self.latency_ms > 0 else 0


def get_hardware_spec() -> HardwareSpec:
    """Detect hardware and return performance specs.

    Returns:
        HardwareSpec with peak compute and bandwidth.
    """
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        # Estimate peak FLOP/s: cores * clock * 2 (FMA)
        peak_flops = props.multi_processor_count * 128 * props.clock_rate * 1e3 * 2
        # Memory bandwidth: clock * bus_width * 2 (DDR)
        peak_bw = props.memory_clock_rate * 1e3 * (props.total_memory // (1024**3)) * 1e9 / 10
        # Use reasonable defaults if estimates seem off
        peak_bw = max(peak_bw, 500e9)  # At least 500 GB/s for modern GPUs
        return HardwareSpec(name=props.name, peak_flops=peak_flops, peak_bandwidth=peak_bw)

    # CPU estimates (conservative)
    return HardwareSpec(
        name="CPU (estimated)",
        peak_flops=200e9,  # ~200 GFLOP/s for modern CPU with AVX
        peak_bandwidth=50e9,  # ~50 GB/s DDR4/DDR5
    )


def roofline_bound(hw: HardwareSpec, arithmetic_intensity: float) -> float:
    """Compute the roofline performance bound.

    Args:
        hw: Hardware specifications.
        arithmetic_intensity: FLOPs per byte transferred.

    Returns:
        Maximum achievable FLOP/s at this arithmetic intensity.
    """
    return min(hw.peak_flops, hw.peak_bandwidth * arithmetic_intensity)


def profile_kernel(
    name: str,
    fn,
    args: tuple,
    flops: int,
    bytes_transferred: int,
    warmup: int = 3,
    repeats: int = 10,
) -> KernelProfile:
    """Profile a kernel's performance.

    Args:
        name: Kernel name for display.
        fn: Function to profile.
        args: Arguments to pass.
        flops: Total FLOPs for one invocation.
        bytes_transferred: Total bytes read + written.
        warmup: Warmup iterations.
        repeats: Timed iterations.

    Returns:
        KernelProfile with measured performance.
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

    avg_ms = sum(times) / len(times)
    return KernelProfile(name=name, flops=flops, bytes_transferred=bytes_transferred, latency_ms=avg_ms)


def run_roofline_analysis() -> tuple[HardwareSpec, list[KernelProfile]]:
    """Profile all kernels and compute roofline positions.

    Returns:
        Tuple of (hardware spec, list of kernel profiles).
    """
    from triton_kernels.fused_softmax import softmax_reference
    from triton_kernels.gelu import gelu_reference
    from triton_kernels.matmul import matmul_reference
    from triton_kernels.layernorm import layernorm_reference
    from triton_kernels.rmsnorm import rmsnorm_reference

    hw = get_hardware_spec()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    profiles: list[KernelProfile] = []

    torch.manual_seed(42)
    N = 1024
    D = 512

    # 1. GELU — elementwise, very memory-bound
    x_gelu = torch.randn(N, D, device=device)
    gelu_flops = N * D * 8  # tanh approx: ~8 ops per element
    gelu_bytes = N * D * 4 * 2  # 1 read + 1 write
    profiles.append(profile_kernel("GELU", gelu_reference, (x_gelu,), gelu_flops, gelu_bytes))

    # 2. Softmax — reduction + elementwise, memory-bound
    x_soft = torch.randn(N, D, device=device)
    soft_flops = N * D * 5  # max, sub, exp, sum, div
    soft_bytes = N * D * 4 * 3  # read input twice (max pass + exp pass) + write
    profiles.append(profile_kernel("Softmax", softmax_reference, (x_soft,), soft_flops, soft_bytes))

    # 3. LayerNorm — reduction + elementwise
    x_ln = torch.randn(N, D, device=device)
    gamma = torch.ones(D, device=device)
    beta = torch.zeros(D, device=device)
    ln_flops = N * D * 8  # mean, var, normalize, scale, shift
    ln_bytes = N * D * 4 * 3 + D * 4 * 2  # input (2 passes) + output + gamma + beta
    profiles.append(profile_kernel("LayerNorm", layernorm_reference, (x_ln, gamma, beta), ln_flops, ln_bytes))

    # 4. RMSNorm — simpler reduction
    x_rms = torch.randn(N, D, device=device)
    gamma_rms = torch.ones(D, device=device)
    rms_flops = N * D * 5  # square, sum, sqrt, div, scale
    rms_bytes = N * D * 4 * 2 + D * 4  # input + output + gamma
    profiles.append(profile_kernel("RMSNorm", rmsnorm_reference, (x_rms, gamma_rms), rms_flops, rms_bytes))

    # 5. Matrix multiplication — compute-bound for large sizes
    M_mat, K_mat, N_mat = 256, 256, 256
    a = torch.randn(M_mat, K_mat, device=device)
    b = torch.randn(K_mat, N_mat, device=device)
    mm_flops = 2 * M_mat * K_mat * N_mat
    mm_bytes = (M_mat * K_mat + K_mat * N_mat + M_mat * N_mat) * 4
    profiles.append(profile_kernel("MatMul (256)", matmul_reference, (a, b), mm_flops, mm_bytes))

    return hw, profiles


def print_roofline_table(hw: HardwareSpec, profiles: list[KernelProfile]) -> None:
    """Print roofline analysis as a formatted table.

    Args:
        hw: Hardware specifications.
        profiles: List of kernel profiles.
    """
    print(f"Hardware: {hw.name}")
    print(f"Peak compute: {hw.peak_flops / 1e9:.0f} GFLOP/s")
    print(f"Peak bandwidth: {hw.peak_bandwidth / 1e9:.0f} GB/s")
    print(f"Ridge point: {hw.ridge_point:.1f} FLOP/byte\n")

    header = f"{'Kernel':<15} {'AI':>8} {'Roofline':>12} {'Achieved':>12} {'Efficiency':>10} {'Bound':<12}"
    print(header)
    print("-" * len(header))

    for p in profiles:
        roof = roofline_bound(hw, p.arithmetic_intensity)
        efficiency = p.achieved_flops / roof * 100 if roof > 0 else 0
        bound = "memory" if p.arithmetic_intensity < hw.ridge_point else "compute"

        print(f"{p.name:<15} {p.arithmetic_intensity:>7.1f} "
              f"{roof / 1e9:>10.1f}G "
              f"{p.achieved_flops / 1e9:>10.1f}G "
              f"{efficiency:>8.1f}% "
              f"{bound:<12}")


def generate_roofline_data(hw: HardwareSpec, profiles: list[KernelProfile]) -> str:
    """Generate data suitable for plotting the roofline model.

    Args:
        hw: Hardware specifications.
        profiles: Kernel profiles.

    Returns:
        Formatted string with plot data.
    """
    lines = ["# Roofline Plot Data", f"# Hardware: {hw.name}",
             f"# Peak FLOP/s: {hw.peak_flops:.2e}", f"# Peak BW: {hw.peak_bandwidth:.2e}",
             "", "# Roofline curve (AI, max_FLOP/s)"]

    # Generate roofline curve points
    ai_points = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    for ai in ai_points:
        perf = roofline_bound(hw, ai)
        lines.append(f"roofline,{ai:.2f},{perf:.2e}")

    lines.append("")
    lines.append("# Kernel measurements (name, AI, achieved_FLOP/s)")
    for p in profiles:
        lines.append(f"kernel,{p.name},{p.arithmetic_intensity:.2f},{p.achieved_flops:.2e}")

    return "\n".join(lines)


if __name__ == "__main__":
    print("=== Roofline Model Analysis ===\n")

    hw, profiles = run_roofline_analysis()
    print_roofline_table(hw, profiles)

    print("\n=== Interpretation ===\n")
    print("Kernels below the ridge point are memory-bound: optimize by reducing")
    print("memory traffic (fusion, tiling, caching in shared memory).")
    print("Kernels above the ridge point are compute-bound: optimize by using")
    print("tensor cores, vectorized instructions, or algorithmic improvements.\n")

    # Print plot data
    data = generate_roofline_data(hw, profiles)
    print("=== Plot Data (for external visualization) ===\n")
    print(data)
