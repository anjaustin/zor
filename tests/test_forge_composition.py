"""
Tests for Composition Operators.

Tests cover:
    - seq(a, b): Sequential composition
    - par(a, b): Parallel composition
    - sel(*shapes): Selection composition
    - rep(shape, n): Repetition composition
"""

import pytest

from trix.forge import (
    Foundry,
    seq, par, sel, rep,
    Seq, Par, Sel, Rep,
    compose_sequential,
    compose_parallel,
    compose_selection,
    compose_repetition,
    generate_xor_shape,
    generate_and_shape,
)


# =============================================================================
# SEQUENTIAL COMPOSITION TESTS
# =============================================================================

class TestSeq:
    """Tests for sequential composition."""

    def test_seq_creates_composition(self):
        """Test that seq() creates Seq object."""
        composition = seq("xor", "and")
        assert isinstance(composition, Seq)
        assert composition.a == "xor"
        assert composition.b == "and"

    def test_seq_evaluate(self):
        """Test evaluating sequential composition."""
        foundry = Foundry(bits=4)
        foundry.atom("xor", lambda a, b: a ^ b)
        foundry.atom("and", lambda a, b: a & b)

        # seq(xor, and)(a, b) = and(xor(a, b), 0)
        composition = seq("xor", "and")
        result = composition.evaluate(foundry.atoms)

        assert result is not None
        assert "seq" in result.name

    def test_seq_unknown_shape(self):
        """Test that seq with unknown shape raises error."""
        foundry = Foundry(bits=8)
        foundry.atom("xor", lambda a, b: a ^ b)

        composition = seq("xor", "unknown")
        with pytest.raises(ValueError):
            composition.evaluate(foundry.atoms)


# =============================================================================
# PARALLEL COMPOSITION TESTS
# =============================================================================

class TestPar:
    """Tests for parallel composition."""

    def test_par_creates_composition(self):
        """Test that par() creates Par object."""
        composition = par("xor", "and")
        assert isinstance(composition, Par)
        assert composition.shapes == ("xor", "and")

    def test_par_multiple_shapes(self):
        """Test par with multiple shapes."""
        composition = par("xor", "and", "or")
        assert len(composition.shapes) == 3

    def test_par_evaluate(self):
        """Test evaluating parallel composition."""
        foundry = Foundry(bits=4)
        foundry.atom("xor", lambda a, b: a ^ b)
        foundry.atom("and", lambda a, b: a & b)

        composition = par("xor", "and")
        result = composition.evaluate(foundry.atoms)

        assert result is not None
        # Parallel composition doubles output bits
        assert result.output_bits == 8  # 4 + 4

    def test_par_unknown_shape(self):
        """Test that par with unknown shape raises error."""
        foundry = Foundry(bits=8)
        foundry.atom("xor", lambda a, b: a ^ b)

        composition = par("xor", "unknown")
        with pytest.raises(ValueError):
            composition.evaluate(foundry.atoms)


# =============================================================================
# SELECTION COMPOSITION TESTS
# =============================================================================

class TestSel:
    """Tests for selection composition."""

    def test_sel_creates_composition(self):
        """Test that sel() creates Sel object."""
        composition = sel("xor", "and", "or")
        assert isinstance(composition, Sel)
        assert composition.shapes == ("xor", "and", "or")

    def test_sel_evaluate(self):
        """Test evaluating selection composition."""
        foundry = Foundry(bits=8)
        foundry.atom("xor", lambda a, b: a ^ b)
        foundry.atom("and", lambda a, b: a & b)
        foundry.atom("or", lambda a, b: a | b)

        composition = sel("xor", "and", "or")
        result = composition.evaluate(foundry.atoms)

        assert result is not None
        assert "sel" in result.name

    def test_sel_unknown_shape(self):
        """Test that sel with unknown shape raises error."""
        foundry = Foundry(bits=8)
        foundry.atom("xor", lambda a, b: a ^ b)

        composition = sel("xor", "unknown")
        with pytest.raises(ValueError):
            composition.evaluate(foundry.atoms)


# =============================================================================
# REPETITION COMPOSITION TESTS
# =============================================================================

class TestRep:
    """Tests for repetition composition."""

    def test_rep_creates_composition(self):
        """Test that rep() creates Rep object."""
        composition = rep("xor", 4)
        assert isinstance(composition, Rep)
        assert composition.shape == "xor"
        assert composition.n == 4

    def test_rep_evaluate_once(self):
        """Test evaluating repetition with n=1."""
        foundry = Foundry(bits=4)
        foundry.atom("xor", lambda a, b: a ^ b)

        composition = rep("xor", 1)
        result = composition.evaluate(foundry.atoms)

        # With n=1, should be same as original
        assert result is not None

    def test_rep_unknown_shape(self):
        """Test that rep with unknown shape raises error."""
        foundry = Foundry(bits=8)

        composition = rep("unknown", 4)
        with pytest.raises(ValueError):
            composition.evaluate(foundry.atoms)

    def test_rep_invalid_count(self):
        """Test that rep with invalid count raises error."""
        foundry = Foundry(bits=8)
        foundry.atom("xor", lambda a, b: a ^ b)

        composition = rep("xor", 0)
        with pytest.raises(ValueError):
            composition.evaluate(foundry.atoms)


# =============================================================================
# COMPOSE FUNCTIONS TESTS
# =============================================================================

class TestComposeFunctions:
    """Tests for compose_* functions."""

    def test_compose_parallel_two_shapes(self):
        """Test compose_parallel with two shapes."""
        xor_shape = generate_xor_shape(bits=4)
        and_shape = generate_and_shape(bits=4)

        result = compose_parallel([xor_shape, and_shape])

        assert result.output_bits == 8  # 4 + 4
        assert result.input_bits == 8   # Same as inputs

    def test_compose_parallel_three_shapes(self):
        """Test compose_parallel with three shapes."""
        from trix.forge import generate_or_shape

        xor_shape = generate_xor_shape(bits=4)
        and_shape = generate_and_shape(bits=4)
        or_shape = generate_or_shape(bits=4)

        result = compose_parallel([xor_shape, and_shape, or_shape])

        assert result.output_bits == 12  # 4 + 4 + 4

    def test_compose_parallel_empty(self):
        """Test compose_parallel with empty list raises error."""
        with pytest.raises(ValueError):
            compose_parallel([])

    def test_compose_selection(self):
        """Test compose_selection."""
        xor_shape = generate_xor_shape(bits=8)
        and_shape = generate_and_shape(bits=8)

        result = compose_selection([xor_shape, and_shape])

        assert result is not None
        assert hasattr(result, '_selection_shapes')

    def test_compose_repetition(self):
        """Test compose_repetition."""
        xor_shape = generate_xor_shape(bits=4)

        result = compose_repetition(xor_shape, 1)
        assert result is not None

    def test_compose_repetition_multiple(self):
        """Test compose_repetition with n > 1."""
        xor_shape = generate_xor_shape(bits=4)

        result = compose_repetition(xor_shape, 3)
        assert "rep" in result.name


# =============================================================================
# FOUNDRY COMPOSE INTEGRATION TESTS
# =============================================================================

class TestFoundryCompose:
    """Integration tests for Foundry.compose()."""

    def test_foundry_compose_par(self):
        """Test composing with par in foundry."""
        foundry = Foundry(bits=4)
        foundry.atom("xor", lambda a, b: a ^ b)
        foundry.atom("and", lambda a, b: a & b)
        foundry.compose("xor_and_par", par("xor", "and"))

        assert "xor_and_par" in foundry.composites
        assert foundry.composites["xor_and_par"].output_bits == 8

    def test_foundry_compose_sel(self):
        """Test composing with sel in foundry."""
        foundry = Foundry(bits=8)
        foundry.atom("xor", lambda a, b: a ^ b)
        foundry.atom("and", lambda a, b: a & b)
        foundry.atom("or", lambda a, b: a | b)
        foundry.compose("alu", sel("xor", "and", "or"))

        assert "alu" in foundry.composites

    def test_foundry_compose_invalid_type(self):
        """Test that compose with invalid type raises error."""
        foundry = Foundry(bits=8)
        foundry.atom("xor", lambda a, b: a ^ b)

        with pytest.raises(TypeError):
            foundry.compose("invalid", "not_a_composition")

    def test_composed_shapes_in_system(self):
        """Test that composed shapes appear in built system."""
        foundry = Foundry(bits=4)
        foundry.atom("xor", lambda a, b: a ^ b)
        foundry.atom("and", lambda a, b: a & b)
        foundry.compose("xor_and_par", par("xor", "and"))

        system = foundry.build()

        # Both atoms and composites should be in system
        assert "xor" in system.shapes
        assert "and" in system.shapes
        assert "xor_and_par" in system.shapes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
