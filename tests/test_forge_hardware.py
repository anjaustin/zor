"""
Tests for trix.forge.hardware module.

Tests hardware resource and timing estimation for XORPU shapes.
"""

import pytest

from trix.forge.hardware import (
    HardwareEstimate,
    estimate_shape,
    estimate_xorpu,
    estimate_luts,
    estimate_ffs,
    estimate_cycles,
    estimate_power,
    estimate_summary,
    _estimate_term_luts,
    _estimate_xor_reduction_luts,
    _estimate_bit_luts,
    _is_simple_shape,
)
from trix.forge.term import (
    Term, BitTerms, ShapeTerms,
    generate_xor_shape, generate_and_shape, generate_or_shape, generate_not_shape,
    generate_nand_shape, generate_nor_shape, generate_xnor_shape, generate_nop_shape,
    generate_add_shape, generate_all_shapes,
)


# =============================================================================
# HARDWARE ESTIMATE DATACLASS
# =============================================================================

class TestHardwareEstimate:
    """Test HardwareEstimate dataclass."""

    def test_creation(self):
        """Can create HardwareEstimate with all fields."""
        est = HardwareEstimate(
            luts=100,
            ffs=0,
            bram_kb=0.0,
            cycles=1,
            power_mw=10.0,
        )
        assert est.luts == 100
        assert est.ffs == 0
        assert est.cycles == 1

    def test_addition(self):
        """Two estimates can be added."""
        est1 = HardwareEstimate(luts=50, ffs=0, bram_kb=0.0, cycles=1, power_mw=5.0)
        est2 = HardwareEstimate(luts=30, ffs=10, bram_kb=1.0, cycles=2, power_mw=3.0)
        total = est1 + est2

        assert total.luts == 80
        assert total.ffs == 10
        assert total.bram_kb == 1.0
        assert total.cycles == 2  # Max of cycles (parallel)
        assert total.power_mw == 8.0

    def test_summary(self):
        """Summary produces readable string."""
        est = HardwareEstimate(luts=100, ffs=0, bram_kb=0.0, cycles=1, power_mw=10.5)
        summary = est.summary()

        assert "100 LUTs" in summary
        assert "0 FFs" in summary
        assert "1 cyc" in summary
        assert "10.5 mW" in summary

    def test_repr(self):
        """Repr is informative."""
        est = HardwareEstimate(luts=32, ffs=0, bram_kb=0.0, cycles=1, power_mw=3.2)
        r = repr(est)

        assert "HardwareEstimate" in r
        assert "luts=32" in r


# =============================================================================
# TERM LUT ESTIMATION
# =============================================================================

class TestTermLutEstimation:
    """Test _estimate_term_luts helper."""

    def test_constant_zero_luts(self):
        """Constant terms need 0 LUTs."""
        term = Term(1, ())
        assert _estimate_term_luts(term) == 0

    def test_single_var_zero_luts(self):
        """Single variable is just a wire."""
        term = Term(1, (0,))
        assert _estimate_term_luts(term) == 0

    def test_two_var_one_lut(self):
        """Two-variable AND needs 1 LUT."""
        term = Term(1, (0, 1))
        assert _estimate_term_luts(term) == 1

    def test_six_var_one_lut(self):
        """Six variables fits in one 6-LUT."""
        term = Term(1, (0, 1, 2, 3, 4, 5))
        assert _estimate_term_luts(term) == 1

    def test_seven_var_two_luts(self):
        """Seven variables needs 2 LUTs."""
        term = Term(1, (0, 1, 2, 3, 4, 5, 6))
        assert _estimate_term_luts(term) == 2


# =============================================================================
# XOR REDUCTION ESTIMATION
# =============================================================================

class TestXorReductionEstimation:
    """Test _estimate_xor_reduction_luts helper."""

    def test_single_term_no_reduction(self):
        """Single term needs no XOR reduction."""
        assert _estimate_xor_reduction_luts(1) == 0

    def test_two_terms_one_lut(self):
        """Two terms need 1 XOR."""
        assert _estimate_xor_reduction_luts(2) == 1

    def test_three_terms_two_luts(self):
        """Three terms need 2 XORs."""
        assert _estimate_xor_reduction_luts(3) == 2

    def test_four_terms_three_luts(self):
        """Four terms need 3 XORs (tree structure)."""
        assert _estimate_xor_reduction_luts(4) == 3


# =============================================================================
# SIMPLE SHAPE DETECTION
# =============================================================================

class TestSimpleShapeDetection:
    """Test _is_simple_shape helper."""

    def test_xor_is_simple(self):
        assert _is_simple_shape("xor") is True
        assert _is_simple_shape("XOR") is True

    def test_and_is_simple(self):
        assert _is_simple_shape("and") is True

    def test_or_is_simple(self):
        assert _is_simple_shape("or") is True

    def test_not_is_simple(self):
        assert _is_simple_shape("not") is True

    def test_nop_is_simple(self):
        assert _is_simple_shape("nop") is True

    def test_add_is_not_simple(self):
        assert _is_simple_shape("add") is False

    def test_sub_is_not_simple(self):
        assert _is_simple_shape("sub") is False


# =============================================================================
# LUT ESTIMATION
# =============================================================================

class TestLutEstimation:
    """Test estimate_luts function."""

    def test_xor_luts(self):
        """XOR uses 1 LUT per bit."""
        shape = generate_xor_shape(bits=8)
        assert estimate_luts(shape) == 8

        shape32 = generate_xor_shape(bits=32)
        assert estimate_luts(shape32) == 32

    def test_and_luts(self):
        """AND uses 1 LUT per bit."""
        shape = generate_and_shape(bits=16)
        assert estimate_luts(shape) == 16

    def test_not_luts(self):
        """NOT uses 1 LUT per bit."""
        shape = generate_not_shape(bits=8)
        assert estimate_luts(shape) == 8

    def test_add_luts_more_than_bits(self):
        """Complex shapes need more LUTs."""
        shape = generate_add_shape(bits=8)
        luts = estimate_luts(shape)
        assert luts >= 8  # At least 1 per output bit


# =============================================================================
# FF ESTIMATION
# =============================================================================

class TestFfEstimation:
    """Test estimate_ffs function."""

    def test_combinational_no_ffs(self):
        """Combinational logic has 0 FFs."""
        shape = generate_xor_shape(bits=32)
        assert estimate_ffs(shape, pipelined=False) == 0

    def test_pipelined_has_ffs(self):
        """Pipelined logic adds FFs."""
        shape = generate_xor_shape(bits=32)
        assert estimate_ffs(shape, pipelined=True, stages=2) == 64  # 32 * 2


# =============================================================================
# CYCLE ESTIMATION
# =============================================================================

class TestCycleEstimation:
    """Test estimate_cycles function."""

    def test_all_shapes_one_cycle(self):
        """All combinational shapes execute in 1 cycle."""
        shapes = [
            generate_xor_shape(bits=32),
            generate_and_shape(bits=32),
            generate_add_shape(bits=32),
        ]
        for shape in shapes:
            assert estimate_cycles(shape) == 1


# =============================================================================
# POWER ESTIMATION
# =============================================================================

class TestPowerEstimation:
    """Test estimate_power function."""

    def test_power_proportional_to_luts(self):
        """Power increases with LUT count."""
        small = generate_xor_shape(bits=8)
        large = generate_xor_shape(bits=32)

        power_small = estimate_power(small)
        power_large = estimate_power(large)

        assert power_large > power_small
        assert power_large / power_small == pytest.approx(4.0, rel=0.1)  # 32/8 = 4


# =============================================================================
# SHAPE ESTIMATION
# =============================================================================

class TestShapeEstimation:
    """Test estimate_shape function."""

    def test_returns_hardware_estimate(self):
        """Returns HardwareEstimate object."""
        shape = generate_xor_shape(bits=8)
        est = estimate_shape(shape)

        assert isinstance(est, HardwareEstimate)
        assert est.luts > 0
        assert est.ffs == 0
        assert est.cycles == 1

    def test_all_simple_shapes(self):
        """All simple shapes have reasonable estimates."""
        generators = [
            generate_xor_shape,
            generate_and_shape,
            generate_or_shape,
            generate_not_shape,
            generate_nand_shape,
            generate_nor_shape,
            generate_xnor_shape,
            generate_nop_shape,
        ]

        for gen in generators:
            shape = gen(bits=32)
            est = estimate_shape(shape)

            assert est.luts == 32  # 1 LUT per bit
            assert est.ffs == 0
            assert est.cycles == 1


# =============================================================================
# XORPU ESTIMATION
# =============================================================================

class TestXorpuEstimation:
    """Test estimate_xorpu function."""

    def test_sums_all_shapes(self):
        """XORPU estimate sums all shape estimates."""
        shapes = {
            'xor': generate_xor_shape(bits=8),
            'and': generate_and_shape(bits=8),
        }
        total = estimate_xorpu(shapes)

        # Should include shape LUTs plus MUX overhead
        assert total.luts >= 16  # At least 8 + 8

    def test_empty_shapes(self):
        """Empty shapes dict works."""
        total = estimate_xorpu({})
        assert total.luts == 0

    def test_single_shape(self):
        """Single shape estimate."""
        shapes = {'xor': generate_xor_shape(bits=8)}
        total = estimate_xorpu(shapes)
        assert total.luts == 8  # No MUX with single shape


# =============================================================================
# SUMMARY
# =============================================================================

class TestEstimateSummary:
    """Test estimate_summary function."""

    def test_produces_table(self):
        """Summary produces formatted table."""
        shapes = {
            'xor': generate_xor_shape(bits=8),
            'and': generate_and_shape(bits=8),
        }
        summary = estimate_summary(shapes)

        assert "XORPU Hardware Estimates" in summary
        assert "xor" in summary
        assert "and" in summary
        assert "TOTAL" in summary
        assert "LUTs" in summary

    def test_sorted_shape_names(self):
        """Shape names are sorted alphabetically."""
        shapes = {
            'zeta': generate_xor_shape(bits=8),
            'alpha': generate_and_shape(bits=8),
        }
        summary = estimate_summary(shapes)

        # Alpha should appear before zeta
        alpha_pos = summary.find('alpha')
        zeta_pos = summary.find('zeta')
        assert alpha_pos < zeta_pos


# =============================================================================
# IMPORT FROM FORGE
# =============================================================================

class TestForgeExports:
    """Test that hardware functions are exported from forge."""

    def test_hardware_estimate_import(self):
        from trix.forge import HardwareEstimate
        assert HardwareEstimate is not None

    def test_estimate_shape_import(self):
        from trix.forge import estimate_shape
        assert callable(estimate_shape)

    def test_estimate_xorpu_import(self):
        from trix.forge import estimate_xorpu
        assert callable(estimate_xorpu)

    def test_estimate_summary_import(self):
        from trix.forge import estimate_summary
        assert callable(estimate_summary)
