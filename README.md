# Custom CUDA Kernels & Triton GPU Programming

Hand-written CUDA kernels and Triton implementations for core deep learning operations: fused attention, matrix multiplication, LayerNorm, softmax, and custom activation functions. Benchmarked against PyTorch native ops.

## What This Covers

- Triton kernels: fused softmax, matmul, LayerNorm, GELU
- Flash Attention implementation in Triton
- CUDA C++ kernels: vector add, reduction, tiled matmul
- PyTorch custom op integration (torch.autograd.Function)
- Benchmarking: Triton vs PyTorch vs cuBLAS
- Memory bandwidth analysis and roofline model

## Structure

```
├── triton_kernels/
│   ├── fused_softmax.py       # Fused softmax kernel
│   ├── matmul.py              # Tiled matrix multiplication
│   ├── layernorm.py           # Fused LayerNorm
│   ├── flash_attention.py     # Flash Attention in Triton
│   ├── gelu.py                # Fused GELU activation
│   └── rmsnorm.py             # RMSNorm kernel
├── cuda_kernels/
│   ├── vector_add.cu          # Basic CUDA vector add
│   ├── reduction.cu           # Parallel reduction
│   ├── tiled_matmul.cu        # Tiled matrix multiplication
│   └── build.py               # Build script for CUDA extensions
├── benchmarks/
│   ├── bench_attention.py     # Flash attention benchmarks
│   ├── bench_matmul.py        # Matmul benchmarks
│   └── roofline.py            # Roofline model analysis
├── notebooks/
│   └── walkthrough.ipynb
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
# Run Triton fused softmax
python triton_kernels/fused_softmax.py
# Benchmark flash attention
python benchmarks/bench_attention.py
# Build and test CUDA kernels (requires nvcc)
python cuda_kernels/build.py
```
