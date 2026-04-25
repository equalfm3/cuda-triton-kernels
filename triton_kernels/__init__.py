"""Triton GPU kernels for core deep learning operations.

Each module provides:
1. A Triton kernel (requires GPU + triton package)
2. A pure PyTorch reference implementation (works on CPU)
3. A unified interface that dispatches to the best available backend
"""

from triton_kernels.fused_softmax import fused_softmax, softmax_reference
from triton_kernels.matmul import triton_matmul, matmul_reference
from triton_kernels.layernorm import fused_layernorm, layernorm_reference
from triton_kernels.flash_attention import flash_attention, attention_reference
from triton_kernels.gelu import fused_gelu, gelu_reference
from triton_kernels.rmsnorm import fused_rmsnorm, rmsnorm_reference

__all__ = [
    "fused_softmax", "softmax_reference",
    "triton_matmul", "matmul_reference",
    "fused_layernorm", "layernorm_reference",
    "flash_attention", "attention_reference",
    "fused_gelu", "gelu_reference",
    "fused_rmsnorm", "rmsnorm_reference",
]
