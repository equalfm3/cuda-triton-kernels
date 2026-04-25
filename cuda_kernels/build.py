"""Build script for CUDA kernel extensions.

Compiles .cu files into shared libraries using torch.utils.cpp_extension
(if available) or falls back to direct nvcc invocation. Prints helpful
instructions if neither CUDA toolkit nor PyTorch CUDA is available.

Usage:
    python cuda_kernels/build.py           # Build all kernels
    python cuda_kernels/build.py --check   # Check CUDA availability only
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Optional


def find_nvcc() -> Optional[str]:
    """Locate the nvcc compiler.

    Returns:
        Path to nvcc binary, or None if not found.
    """
    # Check PATH first
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        return nvcc_path

    # Check common CUDA installation paths
    cuda_paths = [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/usr/local/cuda-11/bin/nvcc",
        os.path.expandvars("$CUDA_HOME/bin/nvcc"),
    ]
    for path in cuda_paths:
        if os.path.isfile(path):
            return path

    return None


def check_torch_cuda() -> bool:
    """Check if PyTorch is installed with CUDA support.

    Returns:
        True if torch.cuda is available.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def build_with_nvcc(cu_file: Path, output_dir: Path) -> bool:
    """Compile a .cu file directly with nvcc.

    Args:
        cu_file: Path to the CUDA source file.
        output_dir: Directory for the compiled binary.

    Returns:
        True if compilation succeeded.
    """
    nvcc = find_nvcc()
    if not nvcc:
        return False

    output_name = cu_file.stem
    output_path = output_dir / output_name

    cmd = [
        nvcc,
        str(cu_file),
        "-o", str(output_path),
        "-O2",
        "--use_fast_math",
    ]

    print(f"  Compiling {cu_file.name} -> {output_path}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print(f"  ✓ {output_name} compiled successfully")
            return True
        else:
            print(f"  ✗ Compilation failed: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Compilation timed out")
        return False
    except FileNotFoundError:
        print(f"  ✗ nvcc not found at {nvcc}")
        return False


def build_with_torch_extension(cu_file: Path, output_dir: Path) -> bool:
    """Build a CUDA extension using torch.utils.cpp_extension.

    This creates a Python-importable shared library that can be loaded
    with torch.ops or as a regular Python module.

    Args:
        cu_file: Path to the CUDA source file.
        output_dir: Directory for build artifacts.

    Returns:
        True if build succeeded.
    """
    try:
        from torch.utils.cpp_extension import load
    except ImportError:
        return False

    module_name = cu_file.stem
    print(f"  Building {cu_file.name} as PyTorch extension '{module_name}'...")

    try:
        module = load(
            name=module_name,
            sources=[str(cu_file)],
            build_directory=str(output_dir / "torch_extensions"),
            verbose=False,
        )
        print(f"  ✓ {module_name} built as PyTorch extension")
        return True
    except Exception as e:
        print(f"  ✗ PyTorch extension build failed: {str(e)[:200]}")
        return False


def print_instructions() -> None:
    """Print setup instructions when CUDA is not available."""
    print("\n" + "=" * 60)
    print("CUDA toolkit not found. To compile CUDA kernels:\n")
    print("Option 1: Install CUDA Toolkit")
    print("  - Download from https://developer.nvidia.com/cuda-downloads")
    print("  - Add nvcc to PATH: export PATH=/usr/local/cuda/bin:$PATH\n")
    print("Option 2: Use Google Colab or cloud GPU")
    print("  - Colab has CUDA pre-installed")
    print("  - Upload .cu files and compile with !nvcc\n")
    print("Option 3: Use Triton kernels instead (no nvcc needed)")
    print("  - python triton_kernels/fused_softmax.py")
    print("  - python triton_kernels/flash_attention.py")
    print("=" * 60)


def main() -> None:
    """Build all CUDA kernels in the cuda_kernels directory."""
    parser = argparse.ArgumentParser(description="Build CUDA kernel extensions")
    parser.add_argument(
        "--check", action="store_true",
        help="Only check CUDA availability, don't build"
    )
    args = parser.parse_args()

    # Determine paths
    script_dir = Path(__file__).parent
    cu_files = sorted(script_dir.glob("*.cu"))
    build_dir = script_dir / "build"

    print("=== CUDA Kernel Build Script ===\n")

    # Check environment
    nvcc = find_nvcc()
    has_torch_cuda = check_torch_cuda()

    print(f"nvcc found: {'yes — ' + nvcc if nvcc else 'no'}")
    print(f"PyTorch CUDA: {'yes' if has_torch_cuda else 'no'}")
    print(f"CUDA source files: {len(cu_files)}")

    if args.check:
        if not nvcc and not has_torch_cuda:
            print_instructions()
        return

    if not cu_files:
        print("\nNo .cu files found in", script_dir)
        return

    if not nvcc and not has_torch_cuda:
        print("\nCannot compile: no CUDA toolkit or PyTorch CUDA found.")
        print_instructions()
        print("\nCUDA source files are available for inspection:")
        for f in cu_files:
            lines = f.read_text().count("\n")
            print(f"  {f.name} ({lines} lines)")
        return

    # Build
    build_dir.mkdir(exist_ok=True)
    print(f"\nBuild directory: {build_dir}\n")

    results: dict[str, bool] = {}
    for cu_file in cu_files:
        print(f"[{cu_file.name}]")
        # Try torch extension first (produces importable module), fall back to nvcc
        success = False
        if has_torch_cuda:
            success = build_with_torch_extension(cu_file, build_dir)
        if not success and nvcc:
            success = build_with_nvcc(cu_file, build_dir)
        results[cu_file.name] = success
        print()

    # Summary
    print("Build Summary:")
    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {name}")

    n_success = sum(results.values())
    print(f"\n{n_success}/{len(results)} kernels compiled successfully")


if __name__ == "__main__":
    main()
