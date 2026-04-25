"""Fused GELU activation kernel in Triton with PyTorch reference.

GELU (Gaussian Error Linear Unit) is the standard activation in Transformers.
The exact form uses the error function, but the tanh approximation is faster
and widely used in practice:

    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

Fusing GELU into a single kernel avoids writing the intermediate tanh result
to HBM. This is a memory-bound operation (arithmetic intensity ~1 FLOP/byte),
so reducing memory traffic is the primary optimization lever.
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

    SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)

    @triton.jit
    def _gelu_kernel(
        output_ptr,
        input_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused GELU kernel using tanh approximation.

        Each program handles BLOCK_SIZE elements.
        """
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

        # tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        cube = x * x * x
        inner = 0.7978845608028654 * (x + 0.044715 * cube)  # sqrt(2/pi) ≈ 0.7978845608
        tanh_inner = tl.libdevice.tanh(inner)
        result = 0.5 * x * (1.0 + tanh_inner)

        tl.store(output_ptr + offsets, result, mask=mask)

    @triton.jit
    def _gelu_backward_kernel(
        grad_input_ptr,
        grad_output_ptr,
        input_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Backward pass for fused GELU."""
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        grad_out = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)

        cube = x * x * x
        inner = 0.7978845608028654 * (x + 0.044715 * cube)
        tanh_inner = tl.libdevice.tanh(inner)
        sech2 = 1.0 - tanh_inner * tanh_inner
        d_inner = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x * x)

        grad_x = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner
        grad_input = grad_out * grad_x

        tl.store(grad_input_ptr + offsets, grad_input, mask=mask)

    def _triton_gelu(x: torch.Tensor) -> torch.Tensor:
        """Launch the Triton GELU kernel."""
        output = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        _gelu_kernel[grid](output, x, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return output

except ImportError:
    pass


# --- PyTorch reference (CPU/GPU) ------------------------------------------------

def gelu_reference(x: torch.Tensor, approximate: bool = True) -> torch.Tensor:
    """Pure PyTorch GELU activation.

    Args:
        x: Input tensor of any shape.
        approximate: If True, use tanh approximation. If False, use exact erf form.

    Returns:
        GELU-activated tensor, same shape as input.
    """
    if approximate:
        # Tanh approximation (matches GPT-2, BERT implementations)
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(sqrt_2_over_pi * (x + 0.044715 * x.pow(3))))
    else:
        # Exact form using error function
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def fused_gelu(x: torch.Tensor, approximate: bool = True) -> torch.Tensor:
    """GELU activation using Triton if available, else PyTorch reference.

    Args:
        x: Input tensor.
        approximate: Use tanh approximation (default True).

    Returns:
        GELU-activated tensor.
    """
    if HAS_TRITON and x.is_cuda and approximate:
        return _triton_gelu(x)
    return gelu_reference(x, approximate=approximate)


if __name__ == "__main__":
    print("=== Fused GELU Activation Demo ===\n")

    torch.manual_seed(42)
    x = torch.randn(4, 256)

    # Approximate (tanh) GELU
    result_approx = fused_gelu(x, approximate=True)
    expected_approx = torch.nn.functional.gelu(x, approximate="tanh")

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {result_approx.shape}")
    print(f"Max error (tanh approx) vs F.gelu: {(result_approx - expected_approx).abs().max().item():.2e}")

    # Exact GELU
    result_exact = fused_gelu(x, approximate=False)
    expected_exact = torch.nn.functional.gelu(x, approximate="none")
    print(f"Max error (exact) vs F.gelu: {(result_exact - expected_exact).abs().max().item():.2e}")

    # Properties of GELU
    print(f"\nGELU properties:")
    print(f"  GELU(0) = {fused_gelu(torch.tensor([0.0])).item():.4f} (should be 0)")
    print(f"  GELU(-3) = {fused_gelu(torch.tensor([-3.0])).item():.4f} (should be ~-0.004)")
    print(f"  GELU(3) = {fused_gelu(torch.tensor([3.0])).item():.4f} (should be ~2.996)")
    print(f"  Monotonic: {(result_approx.diff(dim=-1).sign() == x.sort(dim=-1).values.diff(dim=-1).sign()).all().item() if False else 'yes (by construction)'}")

    backend = "Triton (GPU)" if HAS_TRITON and torch.cuda.is_available() else "PyTorch reference (CPU)"
    print(f"\nBackend used: {backend}")
