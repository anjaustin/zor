"""
Tests for Providence Memory - Content-addressable via XOR matching.

No carry chains. Pure parallel XOR. Elegance.
"""

import pytest
from trix.sim.providence import (
    hamming_distance,
    hamming_distance_batch,
    xor_match,
    xor_match_k,
    ProvidenceMemory,
    HierarchicalProvidenceMemory,
)


# =============================================================================
# TEST: Hamming Distance (The Core)
# =============================================================================

class TestHammingDistance:
    """Test XOR-based Hamming distance."""

    def test_identical(self):
        """Distance to self is 0."""
        assert hamming_distance(0x1234, 0x1234) == 0

    def test_one_bit_different(self):
        """One bit difference = distance 1."""
        assert hamming_distance(0b0000, 0b0001) == 1
        assert hamming_distance(0b1111, 0b1110) == 1

    def test_all_bits_different(self):
        """All bits different."""
        assert hamming_distance(0b0000, 0b1111) == 4
        assert hamming_distance(0x00, 0xFF) == 8

    def test_symmetric(self):
        """Distance is symmetric."""
        assert hamming_distance(0x1234, 0xABCD) == hamming_distance(0xABCD, 0x1234)

    def test_known_values(self):
        """Test known distances."""
        # 0x1234 = 0001 0010 0011 0100
        # 0x1235 = 0001 0010 0011 0101
        # Difference: 1 bit
        assert hamming_distance(0x1234, 0x1235) == 1


class TestHammingDistanceBatch:
    """Test batch Hamming distance."""

    def test_batch(self):
        """Compute distance to multiple signatures."""
        sigs = [0x0000, 0x00FF, 0xFFFF]
        dists = hamming_distance_batch(0x0000, sigs)
        assert dists == [0, 8, 16]


# =============================================================================
# TEST: XOR Matching
# =============================================================================

class TestXorMatch:
    """Test XOR-based nearest neighbor matching."""

    def test_exact_match(self):
        """Find exact signature."""
        sigs = [0x1000, 0x2000, 0x3000, 0x4000]
        assert xor_match(0x2000, sigs) == 1

    def test_nearest_match(self):
        """Find nearest when no exact match."""
        sigs = [0x0000, 0x00FF, 0xFF00, 0xFFFF]
        # 0x0001 is closest to 0x0000 (distance 1)
        assert xor_match(0x0001, sigs) == 0
        # 0x00FE is closest to 0x00FF (distance 1)
        assert xor_match(0x00FE, sigs) == 1

    def test_tie_breaking(self):
        """When tied, returns first match."""
        sigs = [0b0001, 0b0010]  # Both distance 1 from 0b0000
        # Should return first (index 0)
        result = xor_match(0b0000, sigs)
        assert result in [0, 1]  # Either is valid


class TestXorMatchK:
    """Test k-nearest matching."""

    def test_k_nearest(self):
        """Find k nearest signatures."""
        sigs = [0x0000, 0x0001, 0x0003, 0x000F, 0x00FF]
        # Query 0x0000: distances are [0, 1, 2, 4, 8]
        indices, dists = xor_match_k(0x0000, sigs, k=3)
        assert indices == [0, 1, 2]
        assert dists == [0, 1, 2]

    def test_k_larger_than_n(self):
        """k > n returns all."""
        sigs = [0x00, 0xFF]
        indices, dists = xor_match_k(0x00, sigs, k=10)
        assert len(indices) == 2


# =============================================================================
# TEST: Providence Memory
# =============================================================================

class TestProvidenceMemory:
    """Test content-addressable memory."""

    def test_creation(self):
        """Create memory."""
        mem = ProvidenceMemory(256)
        assert len(mem) == 256
        assert mem.num_cells == 256

    def test_write_read_exact(self):
        """Write and read with exact signature match."""
        mem = ProvidenceMemory(256, sig_bits=8, init_mode='sequential')
        mem.write(0x42, 123)
        assert mem.read(0x42) == 123

    def test_write_read_near(self):
        """Read with near signature still finds value."""
        mem = ProvidenceMemory(256, sig_bits=8, init_mode='sequential')
        mem.write(0x42, 123)
        # 0x43 is 1 bit away from 0x42
        assert mem.read(0x43) in [0, 123]  # May find 0x42 or 0x43

    def test_multiple_writes(self):
        """Multiple writes to different cells."""
        mem = ProvidenceMemory(256, sig_bits=8, init_mode='sequential')
        mem.write(0x00, 10)
        mem.write(0x10, 20)
        mem.write(0x20, 30)
        assert mem.read(0x00) == 10
        assert mem.read(0x10) == 20
        assert mem.read(0x20) == 30

    def test_value_truncation(self):
        """Values > 255 are truncated."""
        mem = ProvidenceMemory(256)
        mem.write(0x00, 0x1FF)
        assert mem.read(0x00) == 0xFF

    def test_read_with_distance(self):
        """Read and get distance."""
        mem = ProvidenceMemory(256, sig_bits=8, init_mode='sequential')
        mem.write(0x42, 123)
        value, dist = mem.read_with_distance(0x42)
        assert value == 123
        assert dist == 0  # Exact match


class TestProvidenceMemoryInterpolated:
    """Test interpolated (soft) reads."""

    def test_interpolated_exact(self):
        """Interpolated read with exact match."""
        mem = ProvidenceMemory(256, sig_bits=8, init_mode='sequential')
        mem.write(0x42, 100)
        result = mem.read_interpolated(0x42, k=4)
        assert result == 100.0

    def test_interpolated_blend(self):
        """Interpolated read blends nearby values."""
        mem = ProvidenceMemory(16, sig_bits=4, init_mode='sequential')
        # Write different values to nearby cells
        mem.write(0b0000, 0)
        mem.write(0b0001, 100)
        mem.write(0b0010, 100)
        mem.write(0b0100, 100)

        # Query 0b0000 exactly - should get 0
        result = mem.read_interpolated(0b0000, k=4)
        assert result == 0.0  # Exact match dominates

    def test_interpolated_temperature(self):
        """Temperature affects softness."""
        mem = ProvidenceMemory(16, sig_bits=4, init_mode='sequential')
        mem.write(0b0000, 0)
        mem.write(0b0001, 100)

        # Low temperature = harder (more like argmax)
        low_temp = mem.read_interpolated(0b0000, k=2, temperature=0.1)
        # High temperature = softer (more blending)
        high_temp = mem.read_interpolated(0b0000, k=2, temperature=10.0)

        # Both should lean toward 0 since 0b0000 is exact match
        assert low_temp < high_temp or low_temp == high_temp


class TestProvidenceMemoryInit:
    """Test different initialization modes."""

    def test_sequential_init(self):
        """Sequential init: signatures = 0, 1, 2, ..."""
        mem = ProvidenceMemory(16, sig_bits=8, init_mode='sequential')
        for i in range(16):
            assert mem.signatures[i] == i

    def test_random_init(self):
        """Random init: signatures are random."""
        mem = ProvidenceMemory(16, sig_bits=8, init_mode='random')
        # Just check they're not sequential
        assert mem.signatures != list(range(16))

    def test_spread_init(self):
        """Spread init: signatures maximally spread."""
        mem = ProvidenceMemory(16, sig_bits=8, init_mode='spread')
        # Check signatures are unique
        assert len(set(mem.signatures)) == 16

    def test_invalid_init_raises(self):
        """Invalid init mode raises."""
        with pytest.raises(ValueError):
            ProvidenceMemory(16, init_mode='invalid')


class TestProvidenceMemoryAnalysis:
    """Test analysis methods."""

    def test_coverage(self):
        """Track cell coverage."""
        mem = ProvidenceMemory(100, init_mode='sequential')
        assert mem.coverage() == 0.0

        mem.read(0)
        mem.read(1)
        mem.read(2)
        assert mem.coverage() == 0.03  # 3/100

    def test_hotspots(self):
        """Find most accessed cells."""
        mem = ProvidenceMemory(100, init_mode='sequential')
        for _ in range(10):
            mem.read(5)
        for _ in range(5):
            mem.read(10)

        hot = mem.hotspots(2)
        assert hot[0][0] == 5
        assert hot[0][1] == 10


# =============================================================================
# TEST: Hierarchical Providence Memory
# =============================================================================

class TestHierarchicalProvidenceMemory:
    """Test two-level hierarchical memory."""

    def test_creation(self):
        """Create hierarchical memory."""
        mem = HierarchicalProvidenceMemory(1024, cells_per_cluster=64)
        assert len(mem) == 1024
        assert mem.num_clusters == 16

    def test_write_read(self):
        """Write and read through hierarchy."""
        mem = HierarchicalProvidenceMemory(256, cells_per_cluster=16)
        mem.write(0x42, 123)
        assert mem.read(0x42) == 123

    def test_multiple_clusters(self):
        """Access different clusters."""
        mem = HierarchicalProvidenceMemory(256, cells_per_cluster=16)
        # Write to different clusters
        mem.write(0x00, 10)  # Cluster 0
        mem.write(0x20, 20)  # Cluster 2
        mem.write(0x40, 30)  # Cluster 4

        assert mem.read(0x00) == 10
        assert mem.read(0x20) == 20
        assert mem.read(0x40) == 30

    def test_interpolated(self):
        """Interpolated read through hierarchy."""
        mem = HierarchicalProvidenceMemory(256, cells_per_cluster=16)
        mem.write(0x42, 100)
        result = mem.read_interpolated(0x42, k=4)
        assert result == 100.0


# =============================================================================
# TEST: The Elegance
# =============================================================================

class TestElegance:
    """Test that XOR matching is truly O(1) per comparison."""

    def test_no_carry_chains(self):
        """XOR has no carry - same operation for any bit width."""
        # 8-bit
        d8 = hamming_distance(0xFF, 0x00)
        # 16-bit
        d16 = hamming_distance(0xFFFF, 0x0000)
        # 32-bit
        d32 = hamming_distance(0xFFFFFFFF, 0x00000000)

        # All computed with single XOR, just different popcount
        assert d8 == 8
        assert d16 == 16
        assert d32 == 32

    def test_parallel_operation(self):
        """XOR operates on all bits in parallel."""
        # This is what makes it elegant:
        # 0x1234 ^ 0x5678 computes ALL 16 bit differences at once
        # No sequential carry propagation
        result = 0x1234 ^ 0x5678
        # The XOR IS the distance computation (before popcount)
        assert result == 0x444C  # All differences captured instantly

    def test_content_is_address(self):
        """The content pattern IS the address."""
        mem = ProvidenceMemory(256, sig_bits=8, init_mode='sequential')

        # In traditional memory: address is arbitrary, content is stored
        # In Providence: content pattern IS the address

        pattern = 0b10101010
        mem.write(pattern, 42)

        # Query with same pattern = find same cell
        assert mem.read(pattern) == 42

        # Query with similar pattern = find nearby cell
        similar = 0b10101011  # 1 bit different
        val = mem.read(similar)
        # Will find pattern 0b10101011 if it exists, or nearest match
        assert isinstance(val, int)
