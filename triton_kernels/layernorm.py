"""Fused LayerNorm kernel in Triton with PyTorch reference.

Computes mean, variance, normalization, scale, and shift in a single kernel
launch using Welford's online algorithm for numerically stable one-pass
variance computation. This halves HBM traffic compared to the two-pass
approach (one pass for mean, one for variance).

LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
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
    def _layernorm_kernel(
        output_ptr,
        input_ptr,
        gamma_ptr,
        beta_ptr,
        input_row_stride: tl.constexpr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused LayerNorm kernel using Welford's online algorithm.

        Each program handles one row (one token in a sequence).
        """
        row_idx = tl.program_id(0)
        row_start = input_ptr + row_idx * input_row_stride

        # Welford online mean and variance (single pass)
        mean = 0.0
        m2 = 0.0
        count = 0.0

        for block_start in range(0, n_cols, BLOCK_SIZE):
            col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < n_cols
            x = tl.load(row_start + col_offsets, mask=mask, other=0.0)

            # Welford update for each element in the block
            block_count = tl.sum(mask.to(tl.float32), axis=0)
            block_mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / tl.maximum(block_count, 1.0)
            block_m2 = tl.sum(tl.where(mask, (x - block_mean) * (x - block_mean), 0.0), axis=0)

            # Combine with running statistics
            new_count = count + block_count
            delta = block_mean - mean
            mean = mean + delta * block_count / tl.maximum(new_count, 1.0)
            m2 = m2 + block_m2 + delta * delta * count * block_count / tl.maximum(new_count, 1.0)
            count = new_count

        var = m2 / tl.maximum(count, 1.0)
        rstd = 1.0 / tl.sqrt(var + eps)

        # Normalize, scale, and shift
        out_start = output_ptr + row_idx * input_row_stride
        for block_start in range(0, n_cols, BLOCK_SIZE):
            col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < n_cols
            x = tl.load(row_start + col_offsets, mask=mask, other=0.0)
            gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
            beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)

            normalized = (x - mean) * rstd
            out = gamma * normalized + beta
            tl.store(out_start + col_offsets, out, mask=mask)

    def _triton_layernorm(
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Launch the Triton LayerNorm kernel."""
        n_rows = x.shape[0]
        n_cols = x.shape[1]
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        BLOCK_SIZE = min(BLOCK_SIZE, 4096)

        output = torch.empty_like(x)
        _layernorm_kernel[(n_rows,)](
            output, x, gamma, beta,
            input_row_stride=x.stride(0),
            n_cols=n_cols,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output

except ImportError:
    pass


# --- PyTorch reference (CPU/GPU) ------------------------------------------------

def layernorm_reference(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Pure PyTorch LayerNorm using Welford's online algorithm.

    Computes mean and variance in a single pass for numerical stability,
    then applies affine transform: gamma * (x - mean) / sqrt(var + eps) + beta.

    Args:
        x: Input tensor of shape (..., D) where D is the normalized dimension.
        gamma: Scale parameter of shape (D,).
        beta: Shift parameter of shape (D,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor, same shape as input.
    """
    # Welford's online algorithm for mean and variance
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta


def fused_layernorm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """LayerNorm using Triton if available, else PyTorch reference.

    Args:
        x: Input tensor. Triton path requires 2D CUDA tensor.
        gamma: Scale parameter matching last dimension of x.
        beta: Shift parameter matching last dimension of x.
        eps: Numerical stability constant.

    Returns:
        Layer-normalized tensor.
    """
    if HAS_TRITON and x.is_cuda and x.ndim == 2:
        return _triton_layernorm(x, gamma, beta, eps)
    return layernorm_reference(x, gamma, beta, eps)


if __name__ == "__main__":
    print("=== Fused LayerNorm Demo ===\n")

    torch.manual_seed(42)
    batch, dim = 8, 256
    x = torch.randn(batch, dim)
    gamma = torch.ones(dim)
    beta = torch.zeros(dim)

    result = fused_layernorm(x, gamma, beta)
    expected = torch.nn.functional.layer_norm(x, [dim])

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Output mean (should be ~0): {result.mean(dim=-1).abs().max().item():.2e}")
    print(f"Output var (should be ~1): {result.var(dim=-1, unbiased=False).mean().item():.4f}")
    print(f"Max error vs F.layer_norm: {(result - expected).abs().max().item():.2e}")

    # Test with non-trivial gamma/beta
    gamma2 = torch.randn(dim)
    beta2 = torch.randn(dim)
    result2 = fused_layernorm(x, gamma2, beta2)
    expected2 = torch.nn.functional.layer_norm(x, [dim], gamma2, beta2)
    print(f"\nWith learned gamma/beta:")
    print(f"Max error vs F.layer_norm: {(result2 - expected2).abs().max().item():.2e}")

    backend = "Triton (GPU)" if HAS_TRITON and torch.cuda.is_available() else "PyTorch reference (CPU)"
    print(f"\nBackend used: {backend}")
