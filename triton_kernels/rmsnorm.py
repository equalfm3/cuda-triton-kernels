"""RMSNorm kernel in Triton with PyTorch reference.

RMSNorm (Root Mean Square Layer Normalization) simplifies LayerNorm by
removing the mean subtraction step. It normalizes by the RMS of the input
only, which is cheaper to compute and works well in practice (used in
LLaMA, Gemma, and other modern architectures).

    RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps)

Compared to LayerNorm, RMSNorm:
- Skips mean computation (one fewer reduction)
- Has ~15% fewer FLOPs
- Produces comparable model quality
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
    def _rmsnorm_kernel(
        output_ptr,
        input_ptr,
        gamma_ptr,
        input_row_stride: tl.constexpr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """RMSNorm kernel: normalize by root mean square, then scale.

        Each program handles one row.
        """
        row_idx = tl.program_id(0)
        row_start = input_ptr + row_idx * input_row_stride

        # Compute sum of squares
        sum_sq = 0.0
        for block_start in range(0, n_cols, BLOCK_SIZE):
            col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < n_cols
            x = tl.load(row_start + col_offsets, mask=mask, other=0.0)
            sum_sq += tl.sum(x * x, axis=0)

        # RMS and inverse
        rms = tl.sqrt(sum_sq / n_cols + eps)
        rrms = 1.0 / rms

        # Normalize and scale
        out_start = output_ptr + row_idx * input_row_stride
        for block_start in range(0, n_cols, BLOCK_SIZE):
            col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < n_cols
            x = tl.load(row_start + col_offsets, mask=mask, other=0.0)
            gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
            out = gamma * x * rrms
            tl.store(out_start + col_offsets, out, mask=mask)

    def _triton_rmsnorm(
        x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """Launch the Triton RMSNorm kernel."""
        n_rows, n_cols = x.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        BLOCK_SIZE = min(BLOCK_SIZE, 4096)

        output = torch.empty_like(x)
        _rmsnorm_kernel[(n_rows,)](
            output, x, gamma,
            input_row_stride=x.stride(0),
            n_cols=n_cols,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output

except ImportError:
    pass


# --- PyTorch reference (CPU/GPU) ------------------------------------------------

def rmsnorm_reference(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Pure PyTorch RMSNorm implementation.

    Args:
        x: Input tensor of shape (..., D).
        gamma: Scale parameter of shape (D,).
        eps: Small constant for numerical stability.

    Returns:
        RMS-normalized tensor, same shape as input.
    """
    # Compute RMS: sqrt(mean(x^2) + eps)
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return gamma * x / rms


def fused_rmsnorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """RMSNorm using Triton if available, else PyTorch reference.

    Args:
        x: Input tensor. Triton path requires 2D CUDA tensor.
        gamma: Scale parameter matching last dimension of x.
        eps: Numerical stability constant.

    Returns:
        RMS-normalized tensor.
    """
    if HAS_TRITON and x.is_cuda and x.ndim == 2:
        return _triton_rmsnorm(x, gamma, eps)
    return rmsnorm_reference(x, gamma, eps)


if __name__ == "__main__":
    print("=== RMSNorm Demo ===\n")

    torch.manual_seed(42)
    batch, dim = 8, 256
    x = torch.randn(batch, dim)
    gamma = torch.ones(dim)

    result = fused_rmsnorm(x, gamma)

    # Manual verification
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    expected = x / rms

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Max error vs manual computation: {(result - expected).abs().max().item():.2e}")

    # Compare with LayerNorm (RMSNorm has no mean subtraction)
    ln_result = torch.nn.functional.layer_norm(x, [dim])
    print(f"\nRMSNorm vs LayerNorm difference: {(result - ln_result).abs().mean().item():.4f}")
    print(f"RMSNorm output RMS (should be ~1): {result.pow(2).mean(dim=-1).sqrt().mean().item():.4f}")

    # Test with learned gamma
    gamma2 = torch.randn(dim) * 0.5 + 1.0
    result2 = fused_rmsnorm(x, gamma2)
    expected2 = gamma2 * x / rms
    print(f"\nWith learned gamma:")
    print(f"Max error: {(result2 - expected2).abs().max().item():.2e}")

    backend = "Triton (GPU)" if HAS_TRITON and torch.cuda.is_available() else "PyTorch reference (CPU)"
    print(f"\nBackend used: {backend}")
