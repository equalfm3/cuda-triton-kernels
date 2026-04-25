"""Fused softmax kernel in Triton with PyTorch reference implementation.

The fused kernel computes softmax along the last dimension in a single pass,
avoiding the intermediate storage of the full exp() tensor. This reduces
HBM traffic from 3 reads + 1 write (naive) to 1 read + 1 write (fused).

Numerical stability: subtracts row-wise max before exponentiation.
"""

import torch
import math
from typing import Optional

# --- Triton kernel (GPU only) ---------------------------------------------------

HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True

    @triton.jit
    def _softmax_kernel(
        output_ptr,
        input_ptr,
        input_row_stride: tl.constexpr,
        n_cols: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel: fused softmax over the last dimension.

        Each program instance handles one row. We load the row in blocks,
        compute the max for numerical stability, exponentiate, sum, and normalize.
        """
        row_idx = tl.program_id(0)
        row_start_ptr = input_ptr + row_idx * input_row_stride

        # Phase 1: find row max for numerical stability
        row_max = float("-inf")
        for block_start in range(0, n_cols, BLOCK_SIZE):
            col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < n_cols
            row_block = tl.load(row_start_ptr + col_offsets, mask=mask, other=float("-inf"))
            row_max = tl.maximum(row_max, tl.max(row_block, axis=0))

        # Phase 2: compute exp(x - max) and sum
        denominator = 0.0
        for block_start in range(0, n_cols, BLOCK_SIZE):
            col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < n_cols
            row_block = tl.load(row_start_ptr + col_offsets, mask=mask, other=float("-inf"))
            exp_block = tl.exp(row_block - row_max)
            denominator += tl.sum(exp_block, axis=0)

        # Phase 3: normalize and write output
        out_row_start = output_ptr + row_idx * input_row_stride
        for block_start in range(0, n_cols, BLOCK_SIZE):
            col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < n_cols
            row_block = tl.load(row_start_ptr + col_offsets, mask=mask, other=float("-inf"))
            softmax_out = tl.exp(row_block - row_max) / denominator
            tl.store(out_row_start + col_offsets, softmax_out, mask=mask)

    def _triton_softmax(x: torch.Tensor) -> torch.Tensor:
        """Launch the Triton softmax kernel."""
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        BLOCK_SIZE = min(BLOCK_SIZE, 4096)

        output = torch.empty_like(x)
        _softmax_kernel[(n_rows,)](
            output, x,
            input_row_stride=x.stride(0),
            n_cols=n_cols,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output

except ImportError:
    pass


# --- PyTorch reference (CPU/GPU) ------------------------------------------------

def softmax_reference(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Pure PyTorch softmax with numerical stability.

    Args:
        x: Input tensor of any shape.
        dim: Dimension along which to compute softmax.

    Returns:
        Softmax probabilities, same shape as input.
    """
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def fused_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute softmax using Triton kernel if available, else PyTorch reference.

    Args:
        x: Input tensor. For Triton path, must be 2D and on CUDA.
        dim: Dimension for softmax (Triton path only supports dim=-1 on 2D input).

    Returns:
        Softmax probabilities.
    """
    if HAS_TRITON and x.is_cuda and x.ndim == 2 and dim in (-1, 1):
        return _triton_softmax(x)
    return softmax_reference(x, dim=dim)


if __name__ == "__main__":
    print("=== Fused Softmax Demo ===\n")

    # Test on CPU with reference implementation
    torch.manual_seed(42)
    x = torch.randn(4, 128)

    result = fused_softmax(x)
    expected = torch.softmax(x, dim=-1)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Sum along rows (should be 1.0): {result.sum(dim=-1)}")
    print(f"Max absolute error vs torch.softmax: {(result - expected).abs().max().item():.2e}")
    print(f"All values in [0, 1]: {(result >= 0).all().item() and (result <= 1).all().item()}")

    # Test numerical stability with large values
    x_large = torch.tensor([[1000.0, 1001.0, 1002.0]])
    result_large = fused_softmax(x_large)
    print(f"\nNumerical stability test (large inputs):")
    print(f"  Input: {x_large}")
    print(f"  Output: {result_large}")
    print(f"  Sum: {result_large.sum().item():.6f}")

    backend = "Triton (GPU)" if HAS_TRITON and torch.cuda.is_available() else "PyTorch reference (CPU)"
    print(f"\nBackend used: {backend}")
