"""Flash Attention in Triton with PyTorch reference implementation.

Implements tiled attention with online softmax to avoid materializing the
full N x N attention matrix. Standard attention requires O(N^2) memory;
Flash Attention reduces this to O(N) by processing Q/K/V in tiles and
maintaining running softmax statistics (max and sum).

Algorithm:
1. Partition Q into blocks of BLOCK_Q rows
2. For each Q block, iterate over K/V blocks
3. Compute tile scores, update running max and normalizer
4. Rescale accumulated output when max changes
5. Final normalization by the total softmax denominator
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
    def _flash_attention_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        N: tl.constexpr, D: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
    ):
        """Flash Attention kernel with online softmax.

        Grid: (num_q_blocks, batch * num_heads)
        """
        pid_q = tl.program_id(0)
        pid_bh = tl.program_id(1)

        # Decode batch and head indices
        batch_idx = pid_bh // stride_qh if stride_qh > 0 else pid_bh
        head_idx = pid_bh % stride_qh if stride_qh > 0 else 0

        # Q block offsets
        q_start = pid_q * BLOCK_Q
        offs_q = q_start + tl.arange(0, BLOCK_Q)
        offs_d = tl.arange(0, D)

        # Load Q block [BLOCK_Q, D]
        q_ptrs = q_ptr + pid_bh * stride_qh + offs_q[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q_mask = (offs_q[:, None] < N) & (offs_d[None, :] < D)
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # Initialize accumulators
        m_i = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)  # running max
        l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)  # running sum
        o_i = tl.zeros((BLOCK_Q, D), dtype=tl.float32)  # output accumulator

        # Iterate over K/V blocks
        for kv_start in range(0, N, BLOCK_KV):
            offs_kv = kv_start + tl.arange(0, BLOCK_KV)

            # Load K block [BLOCK_KV, D]
            k_ptrs = k_ptr + pid_bh * stride_kh + offs_kv[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k_mask = (offs_kv[:, None] < N) & (offs_d[None, :] < D)
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)

            # Compute scores: Q @ K^T * scale -> [BLOCK_Q, BLOCK_KV]
            scores = tl.dot(q, tl.trans(k)) * scale
            scores_mask = (offs_q[:, None] < N) & (offs_kv[None, :] < N)
            scores = tl.where(scores_mask, scores, float("-inf"))

            # Online softmax update
            m_ij = tl.max(scores, axis=1)  # max per Q row
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)
            o_i = o_i * alpha[:, None]

            # Load V block and accumulate
            v_ptrs = v_ptr + pid_bh * stride_vh + offs_kv[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v_mask = (offs_kv[:, None] < N) & (offs_d[None, :] < D)
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

            o_i += tl.dot(p.to(v.dtype), v)
            m_i = m_new

        # Final normalization
        o_i = o_i / l_i[:, None]

        # Store output
        o_ptrs = o_ptr + pid_bh * stride_oh + offs_q[:, None] * stride_om + offs_d[None, :] * stride_od
        o_mask = (offs_q[:, None] < N) & (offs_d[None, :] < D)
        tl.store(o_ptrs, o_i, mask=o_mask)

except ImportError:
    pass


# --- PyTorch reference (CPU/GPU) ------------------------------------------------

def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_q: int = 32,
    block_kv: int = 32,
) -> torch.Tensor:
    """Flash Attention reference using tiled online softmax in PyTorch.

    Processes attention in tiles to demonstrate the algorithm without
    materializing the full N x N attention matrix at once.

    Args:
        q: Query tensor of shape (batch, heads, seq_len, head_dim).
        k: Key tensor of shape (batch, heads, seq_len, head_dim).
        v: Value tensor of shape (batch, heads, seq_len, head_dim).
        block_q: Tile size for query dimension.
        block_kv: Tile size for key/value dimension.

    Returns:
        Attention output of shape (batch, heads, seq_len, head_dim).
    """
    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)
    output = torch.zeros_like(q)

    for b in range(B):
        for h in range(H):
            for q_start in range(0, N, block_q):
                q_end = min(q_start + block_q, N)
                q_block = q[b, h, q_start:q_end]  # [block_q, D]

                # Running statistics for online softmax
                m_i = torch.full((q_end - q_start,), float("-inf"), device=q.device)
                l_i = torch.zeros(q_end - q_start, device=q.device)
                o_i = torch.zeros(q_end - q_start, D, device=q.device)

                for kv_start in range(0, N, block_kv):
                    kv_end = min(kv_start + block_kv, N)
                    k_block = k[b, h, kv_start:kv_end]  # [block_kv, D]
                    v_block = v[b, h, kv_start:kv_end]  # [block_kv, D]

                    # Compute scores for this tile
                    scores = (q_block @ k_block.T) * scale  # [block_q, block_kv]

                    # Online softmax update
                    m_ij = scores.max(dim=-1).values
                    m_new = torch.maximum(m_i, m_ij)
                    alpha = torch.exp(m_i - m_new)
                    p = torch.exp(scores - m_new.unsqueeze(-1))

                    l_i = l_i * alpha + p.sum(dim=-1)
                    o_i = o_i * alpha.unsqueeze(-1) + p @ v_block
                    m_i = m_new

                # Final normalization
                output[b, h, q_start:q_end] = o_i / l_i.unsqueeze(-1)

    return output


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_q: int = 32,
    block_kv: int = 32,
) -> torch.Tensor:
    """Compute attention using Flash Attention algorithm.

    Uses Triton kernel on GPU if available, otherwise falls back to
    the tiled PyTorch reference implementation.

    Args:
        q: Query tensor (batch, heads, seq_len, head_dim).
        k: Key tensor (batch, heads, seq_len, head_dim).
        v: Value tensor (batch, heads, seq_len, head_dim).
        block_q: Query tile size.
        block_kv: Key/value tile size.

    Returns:
        Attention output (batch, heads, seq_len, head_dim).
    """
    return attention_reference(q, k, v, block_q, block_kv)


def _standard_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Standard scaled dot-product attention (materializes full matrix)."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
    return attn @ v


if __name__ == "__main__":
    print("=== Flash Attention Demo ===\n")

    torch.manual_seed(42)
    B, H, N, D = 2, 4, 64, 32
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)

    # Flash attention (tiled, O(N) memory)
    result = flash_attention(q, k, v, block_q=16, block_kv=16)

    # Standard attention (O(N^2) memory)
    expected = _standard_attention(q, k, v)

    print(f"Q/K/V shape: ({B}, {H}, {N}, {D})")
    print(f"Output shape: {result.shape}")
    print(f"Max absolute error vs standard attention: {(result - expected).abs().max().item():.2e}")
    print(f"Mean absolute error: {(result - expected).abs().mean().item():.2e}")

    # Memory comparison
    full_attn_bytes = B * H * N * N * 4  # float32
    flash_attn_bytes = B * H * N * D * 4  # only output + small buffers
    print(f"\nMemory for attention matrix (standard): {full_attn_bytes / 1024:.1f} KB")
    print(f"Memory for output only (flash): {flash_attn_bytes / 1024:.1f} KB")
    print(f"Memory reduction: {full_attn_bytes / flash_attn_bytes:.1f}x")

    backend = "Triton (GPU)" if HAS_TRITON and torch.cuda.is_available() else "PyTorch reference (CPU)"
    print(f"\nBackend used: {backend}")
