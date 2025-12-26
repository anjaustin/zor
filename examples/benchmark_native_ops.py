#!/usr/bin/env python3
"""
TriX Native Ops Benchmark

Compares self-hosted C operations vs NumPy.
Demonstrates that ternary matmul needs no actual multiplication.
"""

import numpy as np
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trix.native.ops import TrixOps

WARMUP = 5
ITERATIONS = 50


def benchmark(name: str, fn, iterations: int = ITERATIONS):
    """Run benchmark and return (time_ms, result)."""
    # Warmup
    for _ in range(WARMUP):
        result = fn()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        result = fn()
    elapsed = time.perf_counter() - start
    
    ms_per_iter = (elapsed / iterations) * 1000
    return ms_per_iter, result


def print_comparison(name: str, native_ms: float, numpy_ms: float):
    """Print benchmark comparison."""
    speedup = numpy_ms / native_ms if native_ms > 0 else float('inf')
    faster = "TriX" if native_ms < numpy_ms else "NumPy"
    
    print(f"  {name}:")
    print(f"    TriX Native: {native_ms:8.3f} ms")
    print(f"    NumPy:       {numpy_ms:8.3f} ms")
    print(f"    Winner:      {faster} ({abs(speedup):.2f}x)")
    print()


def main():
    print("=" * 60)
    print("TriX Native Ops vs NumPy Benchmark")
    print("=" * 60)
    print()
    
    ops = TrixOps()
    print(f"TriX Native Ops v{ops.version}")
    print(f"Iterations: {ITERATIONS} (after {WARMUP} warmup)")
    print()
    
    # =========================================================================
    # FROZEN SHAPES
    # =========================================================================
    print("=" * 60)
    print("FROZEN SHAPES")
    print("=" * 60)
    print()
    
    # ReLU
    x = np.random.randn(1_000_000).astype(np.float32)
    
    native_ms, _ = benchmark("ReLU", lambda: ops.relu(x))
    numpy_ms, _ = benchmark("ReLU", lambda: np.maximum(x, 0))
    print_comparison("ReLU (1M elements)", native_ms, numpy_ms)
    
    # XOR polynomial
    a = np.random.rand(1_000_000).astype(np.float32)
    b = np.random.rand(1_000_000).astype(np.float32)
    
    native_ms, _ = benchmark("XOR", lambda: ops.xor(a, b))
    numpy_ms, _ = benchmark("XOR", lambda: a + b - 2 * a * b)
    print_comparison("XOR polynomial (1M elements)", native_ms, numpy_ms)
    
    # =========================================================================
    # REDUCTIONS
    # =========================================================================
    print("=" * 60)
    print("REDUCTIONS (via Adder Trees)")
    print("=" * 60)
    print()
    
    # Sum
    x = np.random.randn(1_000_000).astype(np.float32)
    
    native_ms, native_sum = benchmark("Sum", lambda: ops.sum(x))
    numpy_ms, numpy_sum = benchmark("Sum", lambda: np.sum(x))
    print_comparison("Sum (1M elements)", native_ms, numpy_ms)
    print(f"    Results match: {np.isclose(native_sum, numpy_sum)}")
    print()
    
    # Norm
    native_ms, native_norm = benchmark("Norm", lambda: ops.norm(x))
    numpy_ms, numpy_norm = benchmark("Norm", lambda: np.linalg.norm(x))
    print_comparison("L2 Norm (1M elements)", native_ms, numpy_ms)
    print(f"    Results match: {np.isclose(native_norm, numpy_norm)}")
    print()
    
    # =========================================================================
    # BINARIZATION
    # =========================================================================
    print("=" * 60)
    print("BINARIZATION")
    print("=" * 60)
    print()
    
    x = np.random.randn(1_000_000).astype(np.float32)
    
    native_ms, _ = benchmark("Binarize", lambda: ops.binarize(x))
    
    def numpy_binarize(x):
        packed = np.packbits((x >= 0).astype(np.uint8))
        return packed
    
    numpy_ms, _ = benchmark("Binarize", lambda: numpy_binarize(x))
    print_comparison("Binarize (1M -> 125K bytes)", native_ms, numpy_ms)
    
    # =========================================================================
    # HAMMING ROUTING
    # =========================================================================
    print("=" * 60)
    print("HAMMING ROUTING")
    print("=" * 60)
    print()
    
    dim = 512
    packed_dim = dim // 8
    num_sigs = 128
    num_queries = 1000
    
    sigs = np.random.randint(0, 256, (num_sigs, packed_dim), dtype=np.uint8)
    queries = np.random.randint(0, 256, (num_queries, packed_dim), dtype=np.uint8)
    
    def native_route_all():
        results = []
        for i in range(num_queries):
            results.append(ops.route_hamming(queries[i], sigs))
        return results
    
    def numpy_route_all():
        results = []
        for i in range(num_queries):
            # XOR each query with all signatures
            xor = np.bitwise_xor(queries[i], sigs)
            # Count bits (popcount via lookup)
            dists = np.unpackbits(xor, axis=1).sum(axis=1)
            results.append(np.argmin(dists))
        return results
    
    native_ms, native_routes = benchmark("Routing", native_route_all, iterations=10)
    numpy_ms, numpy_routes = benchmark("Routing", numpy_route_all, iterations=10)
    
    print_comparison(f"Hamming Route ({num_queries} queries, {num_sigs} sigs)", native_ms, numpy_ms)
    print(f"    Results match: {native_routes == numpy_routes}")
    print()
    
    # =========================================================================
    # TERNARY VS DENSE MATMUL (THEORETICAL)
    # =========================================================================
    print("=" * 60)
    print("TERNARY VS DENSE MATMUL (Theoretical)")
    print("=" * 60)
    print()
    
    print("  For a [512 x 512] weight matrix:")
    print()
    print("  Dense MatMul (float32):")
    print("    - 512 * 512 = 262,144 multiplications")
    print("    - 512 * 512 = 262,144 additions")
    print("    - Memory: 512 * 512 * 4 = 1 MB")
    print()
    print("  Ternary MatMul (no multiply):")
    print("    - 0 multiplications (weights are routing signals)")
    print("    - ~170,000 additions (2/3 weights non-zero)")
    print("    - Memory: 512 * 512 * 2 bits = 64 KB")
    print()
    print("  Speedup potential: ~1.5x compute, 16x memory")
    print()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print("  TriX Native Ops provides:")
    print("    1. Self-hosted computation (no external math libraries)")
    print("    2. Ternary weights = routing, not multiplication")
    print("    3. Frozen shapes as mathematical truths")
    print("    4. Hamming distance routing via XOR + popcount")
    print()
    print("  The key insight:")
    print("    For weights in {-1, 0, +1}, matmul becomes:")
    print("      y = Σ_{w=+1} x  -  Σ_{w=-1} x")
    print("    Just additions and subtractions. Zero multiplies.")
    print()


if __name__ == "__main__":
    main()
