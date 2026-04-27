#!/usr/bin/env python3
"""
FP8 GEMM Level 1 — Test & Benchmark

D (BF16, M×N) = A (FP8 E4M3, M×K) × B^T (FP8 E4M3, N×K)

Scale factors use the kernel's GRAN_K setting and are stored as
UE8M0 values (4 E8M0 bytes packed into one int32).
"""

import os
import torch
import ctypes
import time
import subprocess

# ============================================================================
# Compilation
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRIPT_DIR, "matmul.cu")
LIB = os.path.join(SCRIPT_DIR, "matmul.so")

def compile_kernel():
    if os.path.exists(LIB) and os.path.getmtime(LIB) > os.path.getmtime(SRC):
        print("Kernel already compiled, skipping.")
        return
    cmd = [
        "nvcc", "--shared", "-Xcompiler", "-fPIC",
        "-gencode", "arch=compute_100a,code=sm_100a",
        "-O3", "-std=c++17",
        "-lcuda",
        SRC, "-o", LIB
    ]
    print("Compiling:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Compilation done.")

# ============================================================================
# Load library
# ============================================================================

def load_lib():
    lib = ctypes.CDLL(LIB)
    lib.launch_fp8_gemm.argtypes = [
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # B
        ctypes.c_void_p,  # SFA
        ctypes.c_void_p,  # SFB
        ctypes.c_void_p,  # D
        ctypes.c_int,     # M
        ctypes.c_int,     # N
        ctypes.c_int,     # K
    ]
    lib.launch_fp8_gemm.restype = None
    return lib

# ============================================================================
# Test data generation
# ============================================================================

GRAN_K = 128
SF_PACK_WIDTH = 4
MXFP8_BLOCK_K = 32

def pack_scale_factors(scale_groups):
    """Pack 4 E8M0 values into one int32 for the kernel."""
    num_rows, num_groups = scale_groups.shape
    sf_k = (num_groups + SF_PACK_WIDTH - 1) // SF_PACK_WIDTH

    padded = torch.ones(
        num_rows, sf_k * SF_PACK_WIDTH, dtype=torch.float32, device=scale_groups.device
    )
    padded[:, :num_groups] = scale_groups

    scale_bytes = (
        padded.to(torch.float8_e8m0fnu)
        .view(torch.uint8)
        .view(num_rows, sf_k, SF_PACK_WIDTH)
    )
    packed_scales = (
        scale_bytes[..., 0].to(torch.int32)
        | (scale_bytes[..., 1].to(torch.int32) << 8)
        | (scale_bytes[..., 2].to(torch.int32) << 16)
        | (scale_bytes[..., 3].to(torch.int32) << 24)
    )
    return packed_scales.transpose(0, 1).contiguous()

def pack_scaled_mm_scales(scale_groups):
    """Pack logical 1x32 scale blocks for torch._scaled_mm."""
    rows, cols = scale_groups.shape
    n_row_blocks = (rows + 127) // 128
    n_col_blocks = (cols + SF_PACK_WIDTH - 1) // SF_PACK_WIDTH

    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * SF_PACK_WIDTH

    padded = torch.zeros(
        (padded_rows, padded_cols),
        dtype=torch.float8_e8m0fnu,
        device=scale_groups.device,
    )
    padded[:rows, :cols] = scale_groups.to(torch.float8_e8m0fnu)

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, SF_PACK_WIDTH).permute(
        0, 2, 1, 3
    )
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten().contiguous()

def create_scale_tensors(num_rows, K, device="cuda"):
    """Create simple scale factors that vary across GRAN_K blocks."""
    num_groups = (K + GRAN_K - 1) // GRAN_K
    pattern = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32, device=device)
    scale_groups = pattern.repeat((num_groups + SF_PACK_WIDTH - 1) // SF_PACK_WIDTH)[:num_groups]
    scale_groups = scale_groups.unsqueeze(0).expand(num_rows, -1).contiguous()

    kernel_scales = pack_scale_factors(scale_groups)

    repeats = GRAN_K // MXFP8_BLOCK_K
    num_groups_32 = (K + MXFP8_BLOCK_K - 1) // MXFP8_BLOCK_K
    scaled_mm_groups = scale_groups.repeat_interleave(repeats, dim=1)[:, :num_groups_32]
    scaled_mm_scales = pack_scaled_mm_scales(scaled_mm_groups)

    return kernel_scales, scaled_mm_scales

def create_test_data(M, N, K, device="cuda"):
    """Create FP8 matrices and block scales matching matmul.cu."""
    # Random FP8 E4M3 matrices
    # Generate small float values, then cast to FP8
    A_fp32 = torch.randn(M, K, device=device) * 0.1
    B_fp32 = torch.randn(N, K, device=device) * 0.1
    A = A_fp32.to(torch.float8_e4m3fn).contiguous()
    B = B_fp32.to(torch.float8_e4m3fn).contiguous()

    SFA, scale_a = create_scale_tensors(M, K, device=device)
    SFB, scale_b = create_scale_tensors(N, K, device=device)

    # Output
    D = torch.zeros(M, N, dtype=torch.bfloat16, device=device).contiguous()

    return A, B, SFA, SFB, D, scale_a, scale_b

def reference_matmul(A_fp8, B_fp8, scale_a_mxfp8, scale_b_mxfp8):
    """Compute reference with PyTorch block-scaled matmul."""
    return torch._scaled_mm(
        A_fp8,
        B_fp8.T,
        scale_a_mxfp8,
        scale_b_mxfp8,
        out_dtype=torch.bfloat16,
    )

# ============================================================================
# Run kernel
# ============================================================================

def run_kernel(lib, A, B, SFA, SFB, D, M, N, K):
    lib.launch_fp8_gemm(
        A.data_ptr(), B.data_ptr(),
        SFA.data_ptr(), SFB.data_ptr(),
        D.data_ptr(), M, N, K)
    torch.cuda.synchronize()

# ============================================================================
# Correctness check
# ============================================================================

def check_correctness(lib, M=1024, N=1024, K=1024):
    print(f"\n=== Correctness Test: M={M}, N={N}, K={K} ===")
    print(f"  GRAN_K: {GRAN_K}")

    A, B, SFA, SFB, D, scale_a, scale_b = create_test_data(M, N, K)
    ref = reference_matmul(A, B, scale_a, scale_b).bfloat16()

    run_kernel(lib, A, B, SFA, SFB, D, M, N, K)

    # Check
    max_err = (D.float() - ref.float()).abs().max().item()
    mean_err = (D.float() - ref.float()).abs().mean().item()

    # Cosine similarity
    d_flat = D.float().flatten()
    r_flat = ref.float().flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        d_flat.unsqueeze(0), r_flat.unsqueeze(0)).item()

    print(f"  Max error:  {max_err:.4f}")
    print(f"  Mean error: {mean_err:.6f}")
    print(f"  Cosine sim: {cos_sim:.6f}")

    # FP8 has low precision, so allow larger errors
    if cos_sim > 0.99:
        print("  ✓ PASSED")
        return True
    else:
        print("  ✗ FAILED")
        # Show some samples
        print("  First 5 elements:")
        print(f"    Kernel: {D.flatten()[:5].tolist()}")
        print(f"    Ref:    {ref.flatten()[:5].tolist()}")
        return False

# ============================================================================
# Benchmark
# ============================================================================

def benchmark(lib, M, N, K, warmup=5, iters=20):
    A, B, SFA, SFB, D, _, _ = create_test_data(M, N, K)

    # Warmup
    for _ in range(warmup):
        run_kernel(lib, A, B, SFA, SFB, D, M, N, K)

    # Timing
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        run_kernel(lib, A, B, SFA, SFB, D, M, N, K)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / iters * 1000
    tflops = 2.0 * M * N * K / (avg_ms / 1000) / 1e12
    return avg_ms, tflops

def run_benchmarks(lib):
    print("\n=== Benchmarks ===")
    print(f"{'M':>6} {'N':>6} {'K':>6} | {'Time(ms)':>10} {'TFLOPS':>10}")
    print("-" * 52)

    sizes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (4096, 7168, 4096),
        (4096, 7168, 7168),
        (8192, 8192, 8192),
    ]

    for M, N, K in sizes:
        try:
            ms, tflops = benchmark(lib, M, N, K)
            print(f"{M:>6} {N:>6} {K:>6} | {ms:>10.3f} {tflops:>10.1f}")
        except Exception as e:
            print(f"{M:>6} {N:>6} {K:>6} | ERROR: {e}")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    compile_kernel()
    lib = load_lib()

    # Correctness tests
    all_pass = True
    for m, n, k in [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]:
        if not check_correctness(lib, m, n, k):
            all_pass = False

    if all_pass:
        run_benchmarks(lib)
    else:
        print("\nSkipping benchmarks due to correctness failures.")
