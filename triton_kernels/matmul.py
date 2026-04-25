"""Tiled matrix multiplication in Triton with PyTorch reference.

Implements block-level tiling with configurable BLOCK_M, BLOCK_N, BLOCK_K.
Each Triton program computes a BLOCK_M x BLOCK_N tile of the output matrix
by iterating over K in chunks of BLOCK_K, accumulating partial products
in registers (fast) rather than global memory (slow).

Arithmetic intensity: O(BLOCK_K) reuse per loaded element.
"""

import torch
from typing import Optional

# --- Triton kernel (GPU only) ---------------------------------------------------

HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True

    @triton.jit
    def _matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        stride_am: tl.constexpr, stride_ak: tl.constexpr,
        stride_bk: tl.constexpr, stride_bn: tl.constexpr,
        stride_cm: tl.constexpr, stride_cn: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Tiled matmul kernel: C[M,N] = A[M,K] @ B[K,N].

        Each program computes one BLOCK_M x BLOCK_N output tile.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Offsets for this tile
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # Pointers to first tiles of A and B
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        # Accumulator in registers (float32 for precision)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # Iterate over K dimension in blocks
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k
            # Mask out-of-bounds accesses
            a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
            b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)

            a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

            acc += tl.dot(a_tile, b_tile)

            # Advance pointers
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        # Write output tile
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)

    def _triton_matmul(
        a: torch.Tensor,
        b: torch.Tensor,
        block_m: int = 64,
        block_n: int = 64,
        block_k: int = 32,
    ) -> torch.Tensor:
        """Launch the Triton tiled matmul kernel."""
        assert a.ndim == 2 and b.ndim == 2
        M, K = a.shape
        K2, N = b.shape
        assert K == K2, f"Inner dimensions must match: {K} vs {K2}"

        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

        _matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        )
        return c

except ImportError:
    pass


# --- PyTorch reference (CPU/GPU) ------------------------------------------------

def matmul_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
) -> torch.Tensor:
    """Tiled matrix multiplication in pure PyTorch (reference implementation).

    Demonstrates the tiling algorithm using explicit loops over blocks.
    For production use, torch.matmul is faster — this exists for clarity.

    Args:
        a: Left matrix of shape (M, K).
        b: Right matrix of shape (K, N).
        block_m: Tile size along M dimension.
        block_n: Tile size along N dimension.
        block_k: Tile size along K dimension.

    Returns:
        Result matrix of shape (M, N).
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"

    c = torch.zeros(M, N, dtype=a.dtype, device=a.device)

    for m_start in range(0, M, block_m):
        m_end = min(m_start + block_m, M)
        for n_start in range(0, N, block_n):
            n_end = min(n_start + block_n, N)
            for k_start in range(0, K, block_k):
                k_end = min(k_start + block_k, K)
                # Accumulate partial product for this tile
                c[m_start:m_end, n_start:n_end] += (
                    a[m_start:m_end, k_start:k_end] @ b[k_start:k_end, n_start:n_end]
                )
    return c


def triton_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
) -> torch.Tensor:
    """Matrix multiply using Triton if available, else tiled PyTorch reference.

    Args:
        a: Left matrix (M, K).
        b: Right matrix (K, N).
        block_m: Tile size along M.
        block_n: Tile size along N.
        block_k: Tile size along K.

    Returns:
        Result matrix (M, N).
    """
    if HAS_TRITON and a.is_cuda:
        return _triton_matmul(a, b, block_m, block_n, block_k)
    return matmul_reference(a, b, block_m, block_n, block_k)


if __name__ == "__main__":
    print("=== Tiled Matrix Multiplication Demo ===\n")

    torch.manual_seed(42)
    M, K, N = 128, 256, 64
    a = torch.randn(M, K)
    b = torch.randn(K, N)

    # Tiled reference
    result = triton_matmul(a, b, block_m=32, block_n=32, block_k=32)
    expected = a @ b

    print(f"A shape: {a.shape}, B shape: {b.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Max absolute error vs torch.matmul: {(result - expected).abs().max().item():.2e}")

    # Test non-aligned dimensions
    a2 = torch.randn(100, 77)
    b2 = torch.randn(77, 50)
    result2 = triton_matmul(a2, b2, block_m=32, block_n=32, block_k=16)
    expected2 = a2 @ b2
    print(f"\nNon-aligned test ({a2.shape} @ {b2.shape}):")
    print(f"Max absolute error: {(result2 - expected2).abs().max().item():.2e}")

    backend = "Triton (GPU)" if HAS_TRITON and torch.cuda.is_available() else "PyTorch reference (CPU)"
    print(f"\nBackend used: {backend}")
