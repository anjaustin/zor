"""
Rigorous Tests for TriX Native Ops (C Library)

Tests correctness, numerical stability, edge cases, and Python bindings.
"""

import pytest
import numpy as np
import time
from pathlib import Path

# Skip if library not built
LIB_PATH = Path(__file__).parent.parent / "src/trix/native/ops/libtrix_ops.so"
pytestmark = pytest.mark.skipif(
    not LIB_PATH.exists(),
    reason="libtrix_ops.so not built. Run 'make' in src/trix/native/ops/"
)


@pytest.fixture
def ops():
    from trix.native.ops import TrixOps
    return TrixOps()


# =============================================================================
# FROZEN SHAPES - Mathematical Correctness
# =============================================================================

class TestFrozenShapesCorrectness:
    """Verify frozen shapes match mathematical definitions."""
    
    def test_relu_definition(self, ops):
        """ReLU: max(0, x)"""
        x = np.array([-100, -1, -0.001, 0, 0.001, 1, 100], dtype=np.float32)
        expected = np.maximum(x, 0)
        result = ops.relu(x)
        np.testing.assert_allclose(result, expected)
    
    def test_xor_truth_table(self, ops):
        """XOR polynomial on binary inputs equals hardware XOR."""
        a = np.array([0, 0, 1, 1], dtype=np.float32)
        b = np.array([0, 1, 0, 1], dtype=np.float32)
        expected = np.array([0, 1, 1, 0], dtype=np.float32)
        result = ops.xor(a, b)
        np.testing.assert_allclose(result, expected)
    
    def test_and_truth_table(self, ops):
        """AND polynomial on binary inputs equals hardware AND."""
        a = np.array([0, 0, 1, 1], dtype=np.float32)
        b = np.array([0, 1, 0, 1], dtype=np.float32)
        expected = np.array([0, 0, 0, 1], dtype=np.float32)
        result = ops.and_(a, b)
        np.testing.assert_allclose(result, expected)
    
    def test_or_truth_table(self, ops):
        """OR polynomial on binary inputs equals hardware OR."""
        a = np.array([0, 0, 1, 1], dtype=np.float32)
        b = np.array([0, 1, 0, 1], dtype=np.float32)
        expected = np.array([0, 1, 1, 1], dtype=np.float32)
        result = ops.or_(a, b)
        np.testing.assert_allclose(result, expected)
    
    def test_not_truth_table(self, ops):
        """NOT polynomial on binary inputs equals hardware NOT."""
        a = np.array([0, 1], dtype=np.float32)
        expected = np.array([1, 0], dtype=np.float32)
        result = ops.not_(a)
        np.testing.assert_allclose(result, expected)
    
    def test_xor_continuous(self, ops):
        """XOR polynomial is continuous and differentiable."""
        # Test at non-binary values
        a = np.array([0.5, 0.25, 0.75], dtype=np.float32)
        b = np.array([0.5, 0.75, 0.25], dtype=np.float32)
        result = ops.xor(a, b)
        # a + b - 2ab
        expected = a + b - 2 * a * b
        np.testing.assert_allclose(result, expected)
    
    def test_demorgan_identity(self, ops):
        """Verify De Morgan's law: NOT(AND(a,b)) = OR(NOT(a), NOT(b))"""
        a = np.array([0, 0, 1, 1], dtype=np.float32)
        b = np.array([0, 1, 0, 1], dtype=np.float32)
        
        lhs = ops.not_(ops.and_(a, b))
        rhs = ops.or_(ops.not_(a), ops.not_(b))
        np.testing.assert_allclose(lhs, rhs)


# =============================================================================
# REDUCTIONS - Numerical Stability
# =============================================================================

class TestReductions:
    """Test sum and norm operations."""
    
    def test_sum_small(self, ops):
        x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        assert ops.sum(x) == pytest.approx(15.0)
    
    def test_sum_large(self, ops):
        x = np.ones(10000, dtype=np.float32)
        assert ops.sum(x) == pytest.approx(10000.0, rel=1e-4)
    
    def test_sum_empty(self, ops):
        x = np.array([], dtype=np.float32)
        assert ops.sum(x) == 0.0
    
    def test_sum_negative(self, ops):
        x = np.array([-1, -2, -3], dtype=np.float32)
        assert ops.sum(x) == pytest.approx(-6.0)
    
    def test_sum_mixed(self, ops):
        x = np.array([1, -1, 2, -2, 3, -3], dtype=np.float32)
        assert ops.sum(x) == pytest.approx(0.0)
    
    def test_norm_345(self, ops):
        """Classic 3-4-5 right triangle."""
        x = np.array([3, 4], dtype=np.float32)
        assert ops.norm(x) == pytest.approx(5.0)
    
    def test_norm_unit(self, ops):
        x = np.array([1, 0, 0], dtype=np.float32)
        assert ops.norm(x) == pytest.approx(1.0)
    
    def test_norm_zero(self, ops):
        x = np.array([0, 0, 0], dtype=np.float32)
        assert ops.norm(x) == pytest.approx(0.0)
    
    def test_norm_large_vector(self, ops):
        x = np.ones(1000, dtype=np.float32)
        expected = np.sqrt(1000)
        assert ops.norm(x) == pytest.approx(expected, rel=1e-4)


# =============================================================================
# BINARIZATION AND ROUTING
# =============================================================================

class TestBinarization:
    """Test float-to-binary conversion."""
    
    def test_binarize_simple(self, ops):
        x = np.array([1, -1, 1, -1, 1, -1, 1, -1], dtype=np.float32)
        binary = ops.binarize(x)
        # Bits: 1,0,1,0,1,0,1,0 = 0b01010101 = 0x55
        assert binary[0] == 0x55
    
    def test_binarize_all_positive(self, ops):
        x = np.ones(8, dtype=np.float32)
        binary = ops.binarize(x)
        assert binary[0] == 0xFF
    
    def test_binarize_all_negative(self, ops):
        x = -np.ones(8, dtype=np.float32)
        binary = ops.binarize(x)
        assert binary[0] == 0x00
    
    def test_binarize_zero_is_positive(self, ops):
        """Zero should map to 1 (>= 0)."""
        x = np.zeros(8, dtype=np.float32)
        binary = ops.binarize(x)
        assert binary[0] == 0xFF
    
    def test_binarize_length(self, ops):
        """Output should be ceil(n/8) bytes."""
        for n in [1, 7, 8, 9, 15, 16, 17, 64, 65]:
            x = np.random.randn(n).astype(np.float32)
            binary = ops.binarize(x)
            assert len(binary) == (n + 7) // 8


class TestHammingDistance:
    """Test Hamming distance computation."""
    
    def test_hamming_identical(self, ops):
        a = np.array([0xFF, 0x00, 0xAA], dtype=np.uint8)
        b = np.array([0xFF, 0x00, 0xAA], dtype=np.uint8)
        assert ops.hamming_distance(a, b) == 0
    
    def test_hamming_opposite(self, ops):
        a = np.array([0xFF], dtype=np.uint8)
        b = np.array([0x00], dtype=np.uint8)
        assert ops.hamming_distance(a, b) == 8
    
    def test_hamming_one_bit(self, ops):
        a = np.array([0x00], dtype=np.uint8)
        b = np.array([0x01], dtype=np.uint8)
        assert ops.hamming_distance(a, b) == 1
    
    def test_hamming_symmetric(self, ops):
        a = np.array([0xAB, 0xCD], dtype=np.uint8)
        b = np.array([0x12, 0x34], dtype=np.uint8)
        assert ops.hamming_distance(a, b) == ops.hamming_distance(b, a)


class TestRouting:
    """Test Hamming-based routing."""
    
    def test_route_exact_match(self, ops):
        """Should find exact match."""
        input_bin = np.array([0xAB], dtype=np.uint8)
        sigs = np.array([
            [0x00],
            [0xFF],
            [0xAB],  # Exact match
            [0x12],
        ], dtype=np.uint8)
        assert ops.route_hamming(input_bin, sigs) == 2
    
    def test_route_closest(self, ops):
        """Should find closest when no exact match."""
        input_bin = np.array([0xFF], dtype=np.uint8)
        sigs = np.array([
            [0x00],  # Distance 8
            [0xFE],  # Distance 1 (closest)
            [0x0F],  # Distance 4
        ], dtype=np.uint8)
        assert ops.route_hamming(input_bin, sigs) == 1
    
    def test_route_first_on_tie(self, ops):
        """Should return first match on tie."""
        input_bin = np.array([0x00], dtype=np.uint8)
        sigs = np.array([
            [0x01],  # Distance 1
            [0x02],  # Distance 1
            [0x04],  # Distance 1
        ], dtype=np.uint8)
        assert ops.route_hamming(input_bin, sigs) == 0
    
    def test_route_multibyte(self, ops):
        """Test with multi-byte signatures."""
        input_bin = np.array([0xFF, 0xFF], dtype=np.uint8)
        sigs = np.array([
            [0x00, 0x00],  # Distance 16
            [0xFF, 0x00],  # Distance 8
            [0xFF, 0xFF],  # Distance 0 (exact)
        ], dtype=np.uint8)
        assert ops.route_hamming(input_bin, sigs) == 2


# =============================================================================
# NUMERICAL EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test numerical edge cases and stability."""
    
    def test_relu_inf(self, ops):
        x = np.array([np.inf, -np.inf], dtype=np.float32)
        result = ops.relu(x)
        assert result[0] == np.inf
        assert result[1] == 0.0
    
    def test_sum_large_values(self, ops):
        """Sum of large values shouldn't overflow."""
        x = np.full(1000, 1e6, dtype=np.float32)
        result = ops.sum(x)
        assert result == pytest.approx(1e9, rel=1e-3)
    
    def test_sum_small_values(self, ops):
        """Sum of small values shouldn't underflow."""
        x = np.full(1000, 1e-6, dtype=np.float32)
        result = ops.sum(x)
        assert result == pytest.approx(1e-3, rel=1e-3)
    
    def test_norm_stability(self, ops):
        """Norm should be stable for various scales."""
        for scale in [1e-6, 1e-3, 1, 1e3, 1e6]:
            x = np.array([3, 4], dtype=np.float32) * scale
            result = ops.norm(x)
            assert result == pytest.approx(5 * scale, rel=1e-4)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple operations."""
    
    def test_binarize_then_route(self, ops):
        """Full pipeline: float -> binary -> route."""
        # Create input
        x = np.array([1, -1, 1, -1, 1, 1, -1, -1], dtype=np.float32)
        
        # Create signatures (as floats, then binarize)
        sig_floats = [
            np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            np.array([-1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
            np.array([1, -1, 1, -1, 1, 1, -1, -1], dtype=np.float32),  # Same as x
        ]
        
        input_bin = ops.binarize(x)
        sigs_bin = np.array([ops.binarize(s) for s in sig_floats])
        
        best = ops.route_hamming(input_bin, sigs_bin)
        assert best == 2  # Should match the identical signature
    
    def test_xor_chain(self, ops):
        """XOR is associative: (a XOR b) XOR c = a XOR (b XOR c)"""
        a = np.array([0, 0, 1, 1], dtype=np.float32)
        b = np.array([0, 1, 0, 1], dtype=np.float32)
        c = np.array([1, 1, 0, 0], dtype=np.float32)
        
        lhs = ops.xor(ops.xor(a, b), c)
        rhs = ops.xor(a, ops.xor(b, c))
        np.testing.assert_allclose(lhs, rhs)
    
    def test_full_adder_logic(self, ops):
        """Full adder via shapes: sum = a XOR b XOR cin."""
        for a in [0, 1]:
            for b in [0, 1]:
                for cin in [0, 1]:
                    a_arr = np.array([a], dtype=np.float32)
                    b_arr = np.array([b], dtype=np.float32)
                    cin_arr = np.array([cin], dtype=np.float32)
                    
                    # Sum via XOR chain
                    sum_shape = ops.xor(ops.xor(a_arr, b_arr), cin_arr)
                    
                    # Expected sum bit
                    expected_sum = (a + b + cin) % 2
                    
                    assert sum_shape[0] == pytest.approx(expected_sum)


# =============================================================================
# PERFORMANCE SANITY CHECKS
# =============================================================================

class TestPerformance:
    """Basic performance sanity checks."""
    
    def test_sum_performance(self, ops):
        """Sum of 1M elements should complete quickly."""
        x = np.random.randn(1_000_000).astype(np.float32)
        
        start = time.perf_counter()
        result = ops.sum(x)
        elapsed = time.perf_counter() - start
        
        # Should complete in < 100ms
        assert elapsed < 0.1, f"Sum took {elapsed:.3f}s"
        # Result should be reasonable
        assert np.isfinite(result)
    
    def test_binarize_performance(self, ops):
        """Binarize 1M elements should complete quickly."""
        x = np.random.randn(1_000_000).astype(np.float32)
        
        start = time.perf_counter()
        binary = ops.binarize(x)
        elapsed = time.perf_counter() - start
        
        assert elapsed < 0.1, f"Binarize took {elapsed:.3f}s"
        assert len(binary) == 125000  # 1M / 8
    
    def test_routing_performance(self, ops):
        """Route 10K queries to 256 signatures should be fast."""
        dim = 512
        packed_dim = dim // 8
        num_sigs = 256
        num_queries = 10000
        
        sigs = np.random.randint(0, 256, (num_sigs, packed_dim), dtype=np.uint8)
        queries = np.random.randint(0, 256, (num_queries, packed_dim), dtype=np.uint8)
        
        start = time.perf_counter()
        for i in range(num_queries):
            ops.route_hamming(queries[i], sigs)
        elapsed = time.perf_counter() - start
        
        routes_per_sec = num_queries / elapsed
        assert routes_per_sec > 10000, f"Only {routes_per_sec:.0f} routes/sec"
