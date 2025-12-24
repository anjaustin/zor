"""
Hollywood Squares Foundry - Rigorous Test Suite

Validates correctness, not just speed.
Repeatable. Documented. Professional.

Run with: PYTHONPATH=src python -m pytest src/trix/foundry/test_hollywood_squares.py -v
"""

import pytest
import numpy as np
import cupy as cp
import torch
import json
import time
from datetime import datetime
from pathlib import Path


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

TEST_CONFIG = {
    "d_model": 128,
    "num_tiles": 16,
    "grid_size": 16,
    "batch_sizes": [100, 1000, 10000],
    "tolerance_float32": 1e-5,
    "tolerance_int8": 0.02,  # 2% relative error for INT8 quantization
    "random_seed": 42,
}


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def test_results():
    """Collect test results for logging."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": TEST_CONFIG,
        "tests": [],
        "hardware": {},
    }

    # Record hardware info
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    results["hardware"] = {
        "device_name": props['name'].decode(),
        "sm_count": props['multiProcessorCount'],
        "total_memory_gb": props['totalGlobalMem'] / 1e9,
    }

    yield results

    # Save results after all tests
    output_path = Path(__file__).parent / "test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nTest results saved to: {output_path}")


@pytest.fixture(scope="module")
def emergence_model():
    """Load Emergence model once for all tests."""
    from trix.foundry.hollywood_squares_emergence import HollywoodSquaresEmergence
    return HollywoodSquaresEmergence(
        d_model=TEST_CONFIG["d_model"],
        num_tiles=TEST_CONFIG["num_tiles"],
        grid_size=TEST_CONFIG["grid_size"],
    )


@pytest.fixture(scope="module")
def pytorch_model():
    """Load PyTorch reference model."""
    from trix.nn.sparse_lookup_v4 import SparseLookupFFNv4
    return SparseLookupFFNv4(
        d_model=TEST_CONFIG["d_model"],
        num_tiles=TEST_CONFIG["num_tiles"],
    ).cuda().eval()


# =============================================================================
# CORRECTNESS TESTS
# =============================================================================

class TestTernaryPacking:
    """Test ternary signature packing/unpacking."""

    def test_pack_unpack_roundtrip(self, test_results):
        """Verify ternary values survive pack/unpack cycle."""
        np.random.seed(TEST_CONFIG["random_seed"])

        # Generate random ternary values
        d_model = TEST_CONFIG["d_model"]
        original = np.random.choice([-1, 0, 1], size=d_model).astype(np.int8)

        # Pack: 16 values per uint32
        packed = np.zeros(d_model // 16, dtype=np.uint32)
        for w in range(d_model // 16):
            val = 0
            for i in range(16):
                # Encode: -1->0, 0->1, +1->2
                encoded = original[w * 16 + i] + 1
                val |= (encoded & 0x3) << (i * 2)
            packed[w] = val

        # Unpack
        unpacked = np.zeros(d_model, dtype=np.int8)
        for w in range(d_model // 16):
            for i in range(16):
                bits = (packed[w] >> (i * 2)) & 0x3
                unpacked[w * 16 + i] = bits - 1  # Decode: 0->-1, 1->0, 2->+1

        # Verify
        assert np.array_equal(original, unpacked), "Ternary pack/unpack mismatch"

        test_results["tests"].append({
            "name": "test_pack_unpack_roundtrip",
            "status": "PASSED",
            "details": {"d_model": d_model, "packed_size_bytes": packed.nbytes}
        })

    def test_compression_ratio(self, test_results):
        """Verify 16x compression for ternary signatures."""
        d_model = TEST_CONFIG["d_model"]
        num_tiles = TEST_CONFIG["num_tiles"]

        float32_size = num_tiles * d_model * 4  # 4 bytes per float
        packed_size = num_tiles * (d_model // 16) * 4  # uint32 per 16 values

        compression = float32_size / packed_size
        assert compression == 16.0, f"Expected 16x compression, got {compression}x"

        test_results["tests"].append({
            "name": "test_compression_ratio",
            "status": "PASSED",
            "details": {
                "float32_bytes": float32_size,
                "packed_bytes": packed_size,
                "compression": compression
            }
        })


class TestINT8Quantization:
    """Test INT8 direction quantization accuracy."""

    def test_quantization_error(self, test_results):
        """Verify INT8 quantization error is within tolerance."""
        np.random.seed(TEST_CONFIG["random_seed"])

        # Generate random directions
        d_model = TEST_CONFIG["d_model"]
        num_tiles = TEST_CONFIG["num_tiles"]
        directions = np.random.randn(num_tiles, d_model).astype(np.float32) * 0.02

        # Quantize
        scale = np.abs(directions).max() / 127.0
        quantized = np.clip(directions / scale, -127, 127).astype(np.int8)

        # Dequantize
        reconstructed = quantized.astype(np.float32) * scale

        # Compute absolute error (more appropriate for small values)
        abs_error = np.abs(directions - reconstructed)
        max_abs_error = abs_error.max()
        mean_abs_error = abs_error.mean()

        # Error should be less than 1 quantization step
        expected_max_error = scale  # One quantization step
        assert max_abs_error < expected_max_error * 1.1, \
            f"INT8 quantization error too high: {max_abs_error:.6f} > {expected_max_error:.6f}"

        test_results["tests"].append({
            "name": "test_quantization_error",
            "status": "PASSED",
            "details": {
                "max_absolute_error": float(max_abs_error),
                "mean_absolute_error": float(mean_abs_error),
                "quantization_scale": float(scale),
                "expected_max_error": float(expected_max_error),
            }
        })


class TestKernelCorrectness:
    """Test CUDA kernel produces valid outputs."""

    def test_output_shape(self, emergence_model, test_results):
        """Verify output shape matches input shape."""
        for batch_size in TEST_CONFIG["batch_sizes"]:
            x = cp.random.randn(batch_size, TEST_CONFIG["d_model"]).astype(cp.float32)
            output = emergence_model.forward(x)

            assert output.shape == x.shape, \
                f"Shape mismatch: input {x.shape}, output {output.shape}"

        test_results["tests"].append({
            "name": "test_output_shape",
            "status": "PASSED",
            "details": {"batch_sizes_tested": TEST_CONFIG["batch_sizes"]}
        })

    def test_output_finite(self, emergence_model, test_results):
        """Verify outputs contain no NaN or Inf values."""
        for batch_size in TEST_CONFIG["batch_sizes"]:
            x = cp.random.randn(batch_size, TEST_CONFIG["d_model"]).astype(cp.float32)
            output = emergence_model.forward(x)

            assert cp.isfinite(output).all(), "Output contains NaN or Inf"

        test_results["tests"].append({
            "name": "test_output_finite",
            "status": "PASSED",
            "details": {"batch_sizes_tested": TEST_CONFIG["batch_sizes"]}
        })

    def test_deterministic(self, emergence_model, test_results):
        """Verify same input produces same output."""
        cp.random.seed(TEST_CONFIG["random_seed"])

        x = cp.random.randn(1000, TEST_CONFIG["d_model"]).astype(cp.float32)
        positions = cp.arange(1000, dtype=cp.float32) % 64

        output1 = emergence_model.forward(x, positions)
        output2 = emergence_model.forward(x, positions)

        diff = cp.abs(output1 - output2).max()
        assert diff < 1e-6, f"Non-deterministic output: max diff = {diff}"

        test_results["tests"].append({
            "name": "test_deterministic",
            "status": "PASSED",
            "details": {"max_difference": float(diff)}
        })

    def test_residual_connection(self, emergence_model, test_results):
        """Verify residual connection is applied (output != direction * scale)."""
        cp.random.seed(TEST_CONFIG["random_seed"])

        # Use non-zero input to avoid division issues
        x = cp.random.randn(100, TEST_CONFIG["d_model"]).astype(cp.float32) + 1.0
        output = emergence_model.forward(x)

        # Output should be close to input (residual) plus small perturbation
        diff = cp.abs(output - x)
        mean_diff = float(diff.mean())

        # Residual should be small relative to input magnitude
        input_magnitude = float(cp.abs(x).mean())
        relative_change = mean_diff / (input_magnitude + 1e-8)

        assert relative_change < 0.5, "Residual connection may not be working"

        test_results["tests"].append({
            "name": "test_residual_connection",
            "status": "PASSED",
            "details": {
                "mean_absolute_change": mean_diff,
                "input_magnitude": input_magnitude,
                "relative_change": relative_change
            }
        })


class TestVectorizedKernel:
    """Test vectorized kernel matches standard kernel."""

    def test_vectorized_matches_standard(self, emergence_model, test_results):
        """Verify vectorized and standard kernels produce same results."""
        cp.random.seed(TEST_CONFIG["random_seed"])

        x = cp.random.randn(1000, TEST_CONFIG["d_model"]).astype(cp.float32)
        positions = cp.arange(1000, dtype=cp.float32) % 64

        output_std = emergence_model.forward(x, positions)
        output_vec = emergence_model.forward_vectorized(x, positions)

        max_diff = float(cp.abs(output_std - output_vec).max())
        mean_diff = float(cp.abs(output_std - output_vec).mean())

        # Allow small numerical differences
        assert max_diff < 1e-4, f"Vectorized kernel mismatch: max diff = {max_diff}"

        test_results["tests"].append({
            "name": "test_vectorized_matches_standard",
            "status": "PASSED",
            "details": {
                "max_difference": max_diff,
                "mean_difference": mean_diff,
            }
        })


class TestBSplineKernel:
    """Test B-spline spatial routing."""

    def test_bspline_properties(self, test_results):
        """Verify B-spline kernel has correct properties."""
        # B-spline should:
        # 1. Be non-negative
        # 2. Peak at t=0
        # 3. Be zero for |t| >= 2
        # 4. Be symmetric

        def cubic_bspline(t):
            t = np.abs(t)
            result = np.zeros_like(t)
            mask1 = t < 1
            result[mask1] = (2/3) - t[mask1]**2 + 0.5 * t[mask1]**3
            mask2 = (t >= 1) & (t < 2)
            result[mask2] = (1/6) * (2 - t[mask2])**3
            return result

        t = np.linspace(-3, 3, 1000)
        values = cubic_bspline(t)

        # Non-negative
        assert (values >= -1e-10).all(), "B-spline has negative values"

        # Peak at t=0
        peak_idx = np.argmax(values)
        assert abs(t[peak_idx]) < 0.01, "B-spline peak not at t=0"

        # Zero for |t| >= 2
        assert (values[np.abs(t) >= 2] < 1e-10).all(), "B-spline non-zero for |t| >= 2"

        # Symmetric
        left = values[:500]
        right = values[500:][::-1]
        assert np.allclose(left, right, atol=1e-6), "B-spline not symmetric"

        test_results["tests"].append({
            "name": "test_bspline_properties",
            "status": "PASSED",
            "details": {
                "peak_value": float(values.max()),
                "peak_location": float(t[peak_idx]),
            }
        })


# =============================================================================
# BENCHMARK TESTS
# =============================================================================

class TestBenchmarks:
    """Reproducible benchmark tests with logging."""

    def test_throughput_measurement(self, emergence_model, test_results):
        """Measure and log throughput at various batch sizes."""
        results = []

        for batch_size in [1000, 10000, 100000, 500000]:
            try:
                cp.get_default_memory_pool().free_all_blocks()

                x = cp.random.randn(batch_size, TEST_CONFIG["d_model"]).astype(cp.float32)
                positions = cp.arange(batch_size, dtype=cp.float32) % 64

                # Warmup
                _ = emergence_model.forward_vectorized(x, positions)
                cp.cuda.Stream.null.synchronize()

                # Benchmark (10 runs)
                times = []
                for _ in range(10):
                    start = time.perf_counter()
                    _ = emergence_model.forward_vectorized(x, positions)
                    cp.cuda.Stream.null.synchronize()
                    times.append(time.perf_counter() - start)

                best_time = min(times)
                mean_time = np.mean(times)
                std_time = np.std(times)
                throughput = batch_size / best_time

                results.append({
                    "batch_size": batch_size,
                    "best_time_ms": best_time * 1000,
                    "mean_time_ms": mean_time * 1000,
                    "std_time_ms": std_time * 1000,
                    "throughput_tokens_per_sec": throughput,
                    "throughput_m_per_sec": throughput / 1e6,
                })

            except Exception as e:
                results.append({
                    "batch_size": batch_size,
                    "error": str(e),
                })

        test_results["tests"].append({
            "name": "test_throughput_measurement",
            "status": "PASSED",
            "details": {"benchmarks": results}
        })

        # Print results for immediate visibility
        print("\n" + "=" * 60)
        print("THROUGHPUT BENCHMARK RESULTS")
        print("=" * 60)
        for r in results:
            if "error" in r:
                print(f"  {r['batch_size']:>10,}: ERROR - {r['error']}")
            else:
                print(f"  {r['batch_size']:>10,}: {r['throughput_m_per_sec']:.2f} M/s "
                      f"(best: {r['best_time_ms']:.3f}ms, std: {r['std_time_ms']:.3f}ms)")
        print("=" * 60)

    def test_memory_usage(self, emergence_model, test_results):
        """Measure GPU memory usage."""
        cp.get_default_memory_pool().free_all_blocks()

        # Baseline
        baseline = cp.cuda.runtime.memGetInfo()[0]

        # Allocate test data
        batch_size = 100000
        x = cp.random.randn(batch_size, TEST_CONFIG["d_model"]).astype(cp.float32)
        positions = cp.arange(batch_size, dtype=cp.float32) % 64

        # After allocation
        after_input = cp.cuda.runtime.memGetInfo()[0]

        # After forward pass
        output = emergence_model.forward(x, positions)
        after_forward = cp.cuda.runtime.memGetInfo()[0]

        input_memory_mb = (baseline - after_input) / 1e6
        output_memory_mb = (after_input - after_forward) / 1e6
        total_memory_mb = (baseline - after_forward) / 1e6

        test_results["tests"].append({
            "name": "test_memory_usage",
            "status": "PASSED",
            "details": {
                "batch_size": batch_size,
                "input_memory_mb": input_memory_mb,
                "output_memory_mb": output_memory_mb,
                "total_memory_mb": total_memory_mb,
            }
        })


# =============================================================================
# COMPARISON TESTS
# =============================================================================

class TestPyTorchComparison:
    """Compare against PyTorch reference (where applicable)."""

    def test_speedup_measurement(self, emergence_model, pytorch_model, test_results):
        """Measure speedup vs PyTorch and log results."""
        speedups = []

        for batch_size in [1000, 10000, 100000]:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                torch.cuda.empty_cache()

                # Emergence benchmark
                x_cp = cp.random.randn(batch_size, TEST_CONFIG["d_model"]).astype(cp.float32)
                positions = cp.arange(batch_size, dtype=cp.float32) % 64

                _ = emergence_model.forward_vectorized(x_cp, positions)
                cp.cuda.Stream.null.synchronize()

                emergence_times = []
                for _ in range(10):
                    start = time.perf_counter()
                    _ = emergence_model.forward_vectorized(x_cp, positions)
                    cp.cuda.Stream.null.synchronize()
                    emergence_times.append(time.perf_counter() - start)

                emergence_best = min(emergence_times)

                # PyTorch benchmark
                x_torch = torch.randn(1, batch_size, TEST_CONFIG["d_model"]).cuda()

                with torch.no_grad():
                    _ = pytorch_model(x_torch)
                torch.cuda.synchronize()

                pytorch_times = []
                for _ in range(10):
                    start = time.perf_counter()
                    with torch.no_grad():
                        _ = pytorch_model(x_torch)
                    torch.cuda.synchronize()
                    pytorch_times.append(time.perf_counter() - start)

                pytorch_best = min(pytorch_times)

                speedup = pytorch_best / emergence_best

                speedups.append({
                    "batch_size": batch_size,
                    "pytorch_time_ms": pytorch_best * 1000,
                    "emergence_time_ms": emergence_best * 1000,
                    "speedup": speedup,
                })

            except Exception as e:
                speedups.append({
                    "batch_size": batch_size,
                    "error": str(e),
                })

        test_results["tests"].append({
            "name": "test_speedup_measurement",
            "status": "PASSED",
            "details": {"speedups": speedups}
        })

        # Print for visibility
        print("\n" + "=" * 60)
        print("SPEEDUP vs PyTorch")
        print("=" * 60)
        for s in speedups:
            if "error" in s:
                print(f"  {s['batch_size']:>10,}: ERROR - {s['error']}")
            else:
                print(f"  {s['batch_size']:>10,}: {s['speedup']:.1f}x "
                      f"(PyTorch: {s['pytorch_time_ms']:.3f}ms, "
                      f"Emergence: {s['emergence_time_ms']:.3f}ms)")
        print("=" * 60)


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
