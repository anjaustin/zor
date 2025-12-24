"""
Tests for Frozen Shapes Library.

Tests cover:
- Core primitives (XOR, AND, OR, NOT)
- Derived logic shapes
- Comparison shapes
- Activation shapes
- Composition shapes
- Shape library
- Gradient flow
"""

import pytest
import torch
import math

from trix.nn.frozen_shapes import (
    Primitives,
    LogicShapes,
    ComparisonShapes,
    ActivationShapes,
    ArithmeticShapes,
    CompositionShapes,
    FrozenShapeLibrary,
    GeneralFrozenTile,
    get_library,
)


# =============================================================================
# TEST: Primitives
# =============================================================================

class TestPrimitives:
    """Test the four axioms."""

    def test_xor_truth_table(self):
        """XOR truth table is exact on {0, 1}."""
        a = torch.tensor([0., 0., 1., 1.])
        b = torch.tensor([0., 1., 0., 1.])
        expected = torch.tensor([0., 1., 1., 0.])
        result = Primitives.xor(a, b)
        assert torch.allclose(result, expected)

    def test_xor_formula(self):
        """XOR = a + b - 2ab"""
        a, b = 0.3, 0.7
        expected = a + b - 2 * a * b
        result = Primitives.xor(torch.tensor(a), torch.tensor(b))
        assert abs(result.item() - expected) < 1e-6

    def test_and_truth_table(self):
        """AND truth table is exact."""
        a = torch.tensor([0., 0., 1., 1.])
        b = torch.tensor([0., 1., 0., 1.])
        expected = torch.tensor([0., 0., 0., 1.])
        result = Primitives.and_op(a, b)
        assert torch.allclose(result, expected)

    def test_or_truth_table(self):
        """OR truth table is exact."""
        a = torch.tensor([0., 0., 1., 1.])
        b = torch.tensor([0., 1., 0., 1.])
        expected = torch.tensor([0., 1., 1., 1.])
        result = Primitives.or_op(a, b)
        assert torch.allclose(result, expected)

    def test_not_truth_table(self):
        """NOT truth table is exact."""
        a = torch.tensor([0., 1.])
        expected = torch.tensor([1., 0.])
        result = Primitives.not_op(a)
        assert torch.allclose(result, expected)

    def test_gradient_flows_xor(self):
        """Gradients flow through XOR."""
        a = torch.tensor([0.5], requires_grad=True)
        b = torch.tensor([0.5], requires_grad=True)
        result = Primitives.xor(a, b)
        result.backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_gradient_flows_and(self):
        """Gradients flow through AND."""
        a = torch.tensor([0.5], requires_grad=True)
        b = torch.tensor([0.5], requires_grad=True)
        result = Primitives.and_op(a, b)
        result.backward()
        assert a.grad is not None

    def test_gradient_flows_or(self):
        """Gradients flow through OR."""
        a = torch.tensor([0.5], requires_grad=True)
        b = torch.tensor([0.5], requires_grad=True)
        result = Primitives.or_op(a, b)
        result.backward()
        assert a.grad is not None


# =============================================================================
# TEST: Logic Shapes
# =============================================================================

class TestLogicShapes:
    """Test derived logic operations."""

    def test_nand_truth_table(self):
        """NAND = NOT(AND)"""
        a = torch.tensor([0., 0., 1., 1.])
        b = torch.tensor([0., 1., 0., 1.])
        expected = torch.tensor([1., 1., 1., 0.])
        result = LogicShapes.nand(a, b)
        assert torch.allclose(result, expected)

    def test_nor_truth_table(self):
        """NOR = NOT(OR)"""
        a = torch.tensor([0., 0., 1., 1.])
        b = torch.tensor([0., 1., 0., 1.])
        expected = torch.tensor([1., 0., 0., 0.])
        result = LogicShapes.nor(a, b)
        assert torch.allclose(result, expected)

    def test_xnor_truth_table(self):
        """XNOR = NOT(XOR)"""
        a = torch.tensor([0., 0., 1., 1.])
        b = torch.tensor([0., 1., 0., 1.])
        expected = torch.tensor([1., 0., 0., 1.])
        result = LogicShapes.xnor(a, b)
        assert torch.allclose(result, expected)

    def test_implies_truth_table(self):
        """IMPLIES: 0→0=1, 0→1=1, 1→0=0, 1→1=1"""
        a = torch.tensor([0., 0., 1., 1.])
        b = torch.tensor([0., 1., 0., 1.])
        expected = torch.tensor([1., 1., 0., 1.])
        result = LogicShapes.implies(a, b)
        assert torch.allclose(result, expected)

    def test_half_adder(self):
        """Half adder: sum = XOR, carry = AND"""
        a = torch.tensor([0., 0., 1., 1.])
        b = torch.tensor([0., 1., 0., 1.])
        sum_expected = torch.tensor([0., 1., 1., 0.])
        carry_expected = torch.tensor([0., 0., 0., 1.])

        sum_result = LogicShapes.half_adder_sum(a, b)
        carry_result = LogicShapes.half_adder_carry(a, b)

        assert torch.allclose(sum_result, sum_expected)
        assert torch.allclose(carry_result, carry_expected)

    def test_full_adder(self):
        """Full adder: (a, b, c) → (sum, carry)"""
        # 0 + 0 + 0 = 0, carry 0
        s, c = LogicShapes.full_adder(
            torch.tensor([0.]),
            torch.tensor([0.]),
            torch.tensor([0.])
        )
        assert s.item() == 0 and c.item() == 0

        # 1 + 1 + 1 = 1, carry 1
        s, c = LogicShapes.full_adder(
            torch.tensor([1.]),
            torch.tensor([1.]),
            torch.tensor([1.])
        )
        assert s.item() == 1 and c.item() == 1

        # 1 + 0 + 0 = 1, carry 0
        s, c = LogicShapes.full_adder(
            torch.tensor([1.]),
            torch.tensor([0.]),
            torch.tensor([0.])
        )
        assert s.item() == 1 and c.item() == 0

    def test_majority(self):
        """Majority: 1 if at least 2 of 3 are 1."""
        # All combinations
        cases = [
            ([0., 0., 0.], 0.),
            ([0., 0., 1.], 0.),
            ([0., 1., 0.], 0.),
            ([0., 1., 1.], 1.),
            ([1., 0., 0.], 0.),
            ([1., 0., 1.], 1.),
            ([1., 1., 0.], 1.),
            ([1., 1., 1.], 1.),
        ]
        for (a, b, c), expected in cases:
            result = LogicShapes.majority(
                torch.tensor([a]),
                torch.tensor([b]),
                torch.tensor([c])
            )
            assert abs(result.item() - expected) < 1e-6, f"majority({a},{b},{c}) != {expected}"


# =============================================================================
# TEST: Comparison Shapes
# =============================================================================

class TestComparisonShapes:
    """Test comparison operations."""

    def test_eq_identical(self):
        """Equal values give ~1."""
        result = ComparisonShapes.eq(torch.tensor([0.5]), torch.tensor([0.5]))
        assert result.item() > 0.99

    def test_eq_different(self):
        """Different values give ~0."""
        result = ComparisonShapes.eq(torch.tensor([0.]), torch.tensor([1.]))
        assert result.item() < 0.01

    def test_lt_less(self):
        """a < b gives ~1."""
        result = ComparisonShapes.lt(torch.tensor([0.2]), torch.tensor([0.8]))
        assert result.item() > 0.9

    def test_lt_greater(self):
        """a > b gives ~0."""
        result = ComparisonShapes.lt(torch.tensor([0.8]), torch.tensor([0.2]))
        assert result.item() < 0.1

    def test_gt_greater(self):
        """a > b gives ~1."""
        result = ComparisonShapes.gt(torch.tensor([0.8]), torch.tensor([0.2]))
        assert result.item() > 0.9

    def test_max_takes_larger(self):
        """Max returns larger value."""
        result = ComparisonShapes.max2(torch.tensor([0.3]), torch.tensor([0.7]))
        assert abs(result.item() - 0.7) < 0.1

    def test_min_takes_smaller(self):
        """Min returns smaller value."""
        result = ComparisonShapes.min2(torch.tensor([0.3]), torch.tensor([0.7]))
        assert abs(result.item() - 0.3) < 0.1

    def test_gradient_flows(self):
        """Gradients flow through comparisons."""
        a = torch.tensor([0.5], requires_grad=True)
        b = torch.tensor([0.5], requires_grad=True)
        result = ComparisonShapes.max2(a, b)
        result.backward()
        assert a.grad is not None


# =============================================================================
# TEST: Activation Shapes
# =============================================================================

class TestActivationShapes:
    """Test activation functions."""

    def test_relu_positive(self):
        """ReLU passes positive values."""
        x = torch.tensor([1., 2., 3.])
        result = ActivationShapes.relu(x)
        assert torch.allclose(result, x)

    def test_relu_negative(self):
        """ReLU zeros negative values."""
        x = torch.tensor([-1., -2., -3.])
        result = ActivationShapes.relu(x)
        assert torch.allclose(result, torch.zeros_like(x))

    def test_gelu_approx_shape(self):
        """GELU approximation has correct shape."""
        x = torch.randn(10)
        result = ActivationShapes.gelu_approx(x)
        assert result.shape == x.shape

    def test_gelu_at_zero(self):
        """GELU(0) ≈ 0."""
        result = ActivationShapes.gelu_approx(torch.tensor([0.]))
        assert abs(result.item()) < 0.01

    def test_swish_at_zero(self):
        """Swish(0) = 0."""
        result = ActivationShapes.swish(torch.tensor([0.]))
        assert result.item() == 0

    def test_sigmoid_range(self):
        """Sigmoid output in (0, 1)."""
        x = torch.tensor([-10., 0., 10.])
        result = ActivationShapes.sigmoid_poly(x)
        assert (result > 0).all()
        assert (result < 1).all()

    def test_tanh_range(self):
        """Tanh output in [-1, 1]."""
        x = torch.tensor([-10., 0., 10.])
        result = ActivationShapes.tanh_poly(x)
        assert (result >= -1).all()
        assert (result <= 1).all()

    def test_gradient_flows(self):
        """Gradients flow through activations."""
        x = torch.tensor([0.5], requires_grad=True)
        result = ActivationShapes.gelu_approx(x)
        result.backward()
        assert x.grad is not None


# =============================================================================
# TEST: Arithmetic Shapes
# =============================================================================

class TestArithmeticShapes:
    """Test arithmetic operations."""

    def test_add_8bit_no_carry(self):
        """Add two 8-bit numbers without carry."""
        # 5 + 3 = 8
        # 5 = 00000101, 3 = 00000011, 8 = 00001000
        a = torch.tensor([[1., 0., 1., 0., 0., 0., 0., 0.]])  # 5
        b = torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0.]])  # 3
        c_in = torch.tensor([0.])

        result, c_out = ArithmeticShapes.add_8bit(a, b, c_in)

        expected = torch.tensor([[0., 0., 0., 1., 0., 0., 0., 0.]])  # 8
        assert torch.allclose(result, expected, atol=1e-5)
        assert c_out.item() < 0.1

    def test_add_8bit_with_carry(self):
        """Add with input carry."""
        # 5 + 3 + 1 = 9
        a = torch.tensor([[1., 0., 1., 0., 0., 0., 0., 0.]])  # 5
        b = torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0.]])  # 3
        c_in = torch.tensor([1.])

        result, c_out = ArithmeticShapes.add_8bit(a, b, c_in)

        expected = torch.tensor([[1., 0., 0., 1., 0., 0., 0., 0.]])  # 9
        assert torch.allclose(result, expected, atol=1e-5)


# =============================================================================
# TEST: Composition Shapes
# =============================================================================

class TestCompositionShapes:
    """Test composition utilities."""

    def test_identity(self):
        """Identity returns input unchanged."""
        x = torch.randn(5)
        result = CompositionShapes.identity(x)
        assert torch.allclose(result, x)

    def test_constant(self):
        """Constant returns same value."""
        x = torch.randn(5)
        result = CompositionShapes.constant(x, 0.5)
        assert torch.allclose(result, torch.full_like(x, 0.5))

    def test_swap(self):
        """Swap reverses dimensions."""
        x = torch.tensor([1., 2., 3., 4.])
        expected = torch.tensor([4., 3., 2., 1.])
        result = CompositionShapes.swap(x)
        assert torch.allclose(result, expected)


# =============================================================================
# TEST: Shape Library
# =============================================================================

class TestFrozenShapeLibrary:
    """Test the shape library."""

    def test_creation(self):
        """Create library."""
        lib = FrozenShapeLibrary()
        assert len(lib) > 0

    def test_get_shape(self):
        """Get shape by name."""
        lib = FrozenShapeLibrary()
        xor_fn = lib.get('xor')
        result = xor_fn(torch.tensor([1.]), torch.tensor([0.]))
        assert result.item() == 1

    def test_attribute_access(self):
        """Access shapes as attributes."""
        lib = FrozenShapeLibrary()
        result = lib.xor(torch.tensor([1.]), torch.tensor([0.]))
        assert result.item() == 1

    def test_list_shapes(self):
        """List all shapes."""
        lib = FrozenShapeLibrary()
        shapes = lib.list_shapes()
        assert 'xor' in shapes
        assert 'and' in shapes
        assert 'relu' in shapes

    def test_list_by_category(self):
        """List shapes by category."""
        lib = FrozenShapeLibrary()
        primitives = lib.list_shapes('primitive')
        assert 'xor' in primitives
        assert 'relu' not in primitives

    def test_list_categories(self):
        """List available categories."""
        lib = FrozenShapeLibrary()
        categories = lib.list_categories()
        assert 'primitive' in categories
        assert 'logic' in categories
        assert 'activation' in categories

    def test_shape_info(self):
        """Get shape metadata."""
        lib = FrozenShapeLibrary()
        info = lib.info('xor')
        assert info.name == 'xor'
        assert info.category == 'primitive'
        assert info.n_inputs == 2

    def test_contains(self):
        """Check if shape exists."""
        lib = FrozenShapeLibrary()
        assert 'xor' in lib
        assert 'nonexistent' not in lib

    def test_unknown_shape_raises(self):
        """Unknown shape raises error."""
        lib = FrozenShapeLibrary()
        with pytest.raises(KeyError):
            lib.get('nonexistent')

    def test_global_library(self):
        """Global library is singleton."""
        lib1 = get_library()
        lib2 = get_library()
        assert lib1 is lib2


# =============================================================================
# TEST: General Frozen Tile
# =============================================================================

class TestGeneralFrozenTile:
    """Test frozen tiles using library shapes."""

    def test_creation(self):
        """Create tile from shape name."""
        tile = GeneralFrozenTile('xor', d_model=64)
        assert tile.shape_name == 'xor'
        assert tile.d_model == 64

    def test_signature_shape(self):
        """Signature has correct shape."""
        tile = GeneralFrozenTile('xor', d_model=64)
        sig = tile.get_signature()
        assert sig.shape == (64,)

    def test_signature_is_ternary(self):
        """Signature values are in {-1, +1}."""
        tile = GeneralFrozenTile('xor', d_model=64)
        sig = tile.get_signature()
        assert ((sig == -1) | (sig == 1)).all()

    def test_forward_binary_shape(self):
        """Forward works for binary shapes."""
        tile = GeneralFrozenTile('xor', d_model=64)
        a = torch.tensor([[1., 0., 1., 0., 1., 0., 1., 0.]])
        b = torch.tensor([[0., 1., 0., 1., 0., 1., 0., 1.]])
        result = tile(a, b)
        expected = torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1.]])
        assert torch.allclose(result, expected)

    def test_forward_unary_shape(self):
        """Forward works for unary shapes."""
        tile = GeneralFrozenTile('relu', d_model=64)
        x = torch.tensor([[-1., 0., 1., 2.]])
        result = tile(x)
        expected = torch.tensor([[0., 0., 1., 2.]])
        assert torch.allclose(result, expected)

    def test_unknown_shape_raises(self):
        """Unknown shape raises error."""
        with pytest.raises(ValueError):
            GeneralFrozenTile('nonexistent', d_model=64)


# =============================================================================
# TEST: Mathematical Properties
# =============================================================================

class TestMathematicalProperties:
    """Test mathematical properties of shapes."""

    def test_xor_commutative(self):
        """XOR(a, b) = XOR(b, a)"""
        a = torch.rand(10)
        b = torch.rand(10)
        assert torch.allclose(Primitives.xor(a, b), Primitives.xor(b, a))

    def test_and_commutative(self):
        """AND(a, b) = AND(b, a)"""
        a = torch.rand(10)
        b = torch.rand(10)
        assert torch.allclose(Primitives.and_op(a, b), Primitives.and_op(b, a))

    def test_or_commutative(self):
        """OR(a, b) = OR(b, a)"""
        a = torch.rand(10)
        b = torch.rand(10)
        assert torch.allclose(Primitives.or_op(a, b), Primitives.or_op(b, a))

    def test_double_negation(self):
        """NOT(NOT(a)) = a"""
        a = torch.rand(10)
        assert torch.allclose(Primitives.not_op(Primitives.not_op(a)), a)

    def test_de_morgan_1(self):
        """NOT(AND(a, b)) = OR(NOT(a), NOT(b))"""
        a = torch.rand(10)
        b = torch.rand(10)
        lhs = Primitives.not_op(Primitives.and_op(a, b))
        rhs = Primitives.or_op(Primitives.not_op(a), Primitives.not_op(b))
        assert torch.allclose(lhs, rhs)

    def test_de_morgan_2(self):
        """NOT(OR(a, b)) = AND(NOT(a), NOT(b))"""
        a = torch.rand(10)
        b = torch.rand(10)
        lhs = Primitives.not_op(Primitives.or_op(a, b))
        rhs = Primitives.and_op(Primitives.not_op(a), Primitives.not_op(b))
        assert torch.allclose(lhs, rhs)

    def test_xor_self_is_zero(self):
        """XOR(a, a) = 0"""
        a = torch.rand(10)
        result = Primitives.xor(a, a)
        # For binary inputs this is exactly 0
        # For continuous inputs: a + a - 2aa = 2a - 2a² = 2a(1-a)
        # This is 0 only when a=0 or a=1
        binary_a = (torch.rand(10) > 0.5).float()
        binary_result = Primitives.xor(binary_a, binary_a)
        assert torch.allclose(binary_result, torch.zeros_like(binary_result))
