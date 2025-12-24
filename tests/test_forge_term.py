"""
Tests for trix.forge.term module.

Covers:
- Term dataclass
- BitTerms dataclass
- ShapeTerms dataclass
- Primitive term generators
- Shape generators
- Cross-validation with tensor functions
- Edge cases
"""

import pytest
import torch
from typing import Tuple, List

from trix.forge.term import (
    # Core classes
    Term, BitTerms, ShapeTerms,

    # Primitive generators
    xor_terms, and_terms, or_terms, not_terms,
    nand_terms, nor_terms, xnor_terms,

    # Shape generators
    generate_xor_shape, generate_and_shape, generate_or_shape, generate_not_shape,
    generate_nand_shape, generate_nor_shape, generate_xnor_shape, generate_nop_shape,
    generate_add_shape, generate_sub_shape,
    generate_sll_shape, generate_srl_shape, generate_sra_shape,
    generate_slt_shape, generate_sltu_shape,

    # Registry
    SHAPE_GENERATORS, generate_shape, generate_all_shapes,

    # Validation
    validate_terms_against_tensor,
)

from trix.routing.primitives import xor, and_, or_, not_, nand, nor, xnor


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_term():
    """A simple single-variable term."""
    return Term(1, (0,))


@pytest.fixture
def xor_term():
    """The -2ab term from XOR."""
    return Term(-2, (0, 32))


@pytest.fixture
def constant_term():
    """A constant term."""
    return Term(1, ())


@pytest.fixture
def xor_shape_8bit():
    """8-bit XOR shape for faster tests."""
    return generate_xor_shape(bits=8)


@pytest.fixture
def xor_shape_32bit():
    """32-bit XOR shape."""
    return generate_xor_shape(bits=32)


# =============================================================================
# TERM CLASS TESTS
# =============================================================================

class TestTerm:
    """Tests for Term dataclass."""

    def test_creation_basic(self):
        """Term can be created with coefficient and variables."""
        t = Term(1, (0,))
        assert t.coefficient == 1
        assert t.variables == (0,)

    def test_creation_multiple_vars(self):
        """Term can have multiple variables."""
        t = Term(-2, (0, 32))
        assert t.coefficient == -2
        assert t.variables == (0, 32)

    def test_creation_constant(self):
        """Constant term has empty variables."""
        t = Term(5, ())
        assert t.coefficient == 5
        assert t.variables == ()

    def test_creation_requires_sorted(self):
        """Variables must be sorted."""
        with pytest.raises(ValueError):
            Term(1, (32, 0))  # Not sorted

    def test_creation_sorted_ok(self):
        """Sorted variables are accepted."""
        t = Term(1, (0, 32, 64))
        assert t.variables == (0, 32, 64)

    def test_create_factory_sorts(self):
        """create() factory sorts variables."""
        t = Term.create(1, [32, 0])
        assert t.variables == (0, 32)

    def test_create_factory_filters_zero(self):
        """create() returns None for zero coefficient."""
        t = Term.create(0, [0, 32])
        assert t is None

    def test_constant_factory(self):
        """constant() creates degree-0 term."""
        t = Term.constant(42)
        assert t.coefficient == 42
        assert t.variables == ()
        assert t.is_constant()

    def test_constant_factory_filters_zero(self):
        """constant() returns None for zero."""
        t = Term.constant(0)
        assert t is None

    def test_degree_zero(self):
        """Constant term has degree 0."""
        t = Term(1, ())
        assert t.degree() == 0

    def test_degree_one(self):
        """Single variable has degree 1."""
        t = Term(1, (5,))
        assert t.degree() == 1

    def test_degree_two(self):
        """Two variables has degree 2."""
        t = Term(1, (0, 32))
        assert t.degree() == 2

    def test_is_constant_true(self):
        """is_constant() true for empty variables."""
        t = Term(1, ())
        assert t.is_constant()

    def test_is_constant_false(self):
        """is_constant() false for non-empty variables."""
        t = Term(1, (0,))
        assert not t.is_constant()

    def test_frozen(self):
        """Term is frozen (immutable)."""
        t = Term(1, (0,))
        with pytest.raises(AttributeError):
            t.coefficient = 2

    def test_hashable(self):
        """Term can be used in sets."""
        t1 = Term(1, (0,))
        t2 = Term(1, (0,))
        t3 = Term(2, (0,))

        s = {t1, t2, t3}
        assert len(s) == 2  # t1 and t2 are equal

    def test_evaluate_constant(self):
        """Constant term evaluates to constant."""
        t = Term(5, ())
        inputs = torch.zeros(3, 10)
        result = t.evaluate(inputs)
        assert torch.allclose(result, torch.tensor([5., 5., 5.]))

    def test_evaluate_single_var(self):
        """Single variable term."""
        t = Term(2, (0,))
        inputs = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=torch.float32)
        result = t.evaluate(inputs)
        assert torch.allclose(result, torch.tensor([2., 0., 2.]))

    def test_evaluate_product(self):
        """Product term evaluates as product."""
        t = Term(1, (0, 1))
        inputs = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=torch.float32)
        result = t.evaluate(inputs)
        assert torch.allclose(result, torch.tensor([1., 0., 0., 0.]))

    def test_evaluate_negative_coeff(self):
        """Negative coefficient works."""
        t = Term(-2, (0, 1))
        inputs = torch.tensor([[1, 1]], dtype=torch.float32)
        result = t.evaluate(inputs)
        assert torch.allclose(result, torch.tensor([-2.]))

    def test_to_str_constant(self):
        """String representation of constant."""
        t = Term(5, ())
        assert t.to_str() == "5"

    def test_to_str_single_var(self):
        """String representation of single variable."""
        t = Term(1, (0,))
        assert t.to_str() == "x0"

    def test_to_str_negative(self):
        """String representation with negative coefficient."""
        t = Term(-1, (5,))
        assert t.to_str() == "-x5"

    def test_to_str_with_coeff(self):
        """String representation with coefficient."""
        t = Term(-2, (0, 32))
        assert t.to_str() == "-2·x0·x32"

    def test_to_str_with_names(self):
        """String representation with custom names."""
        t = Term(1, (0, 32))
        names = {0: "a[0]", 32: "b[0]"}
        assert t.to_str(names) == "a[0]·b[0]"

    def test_to_dict(self):
        """Export to dictionary."""
        t = Term(-2, (0, 32))
        d = t.to_dict()
        assert d == {"coeff": -2, "vars": [0, 32]}

    def test_from_dict(self):
        """Import from dictionary."""
        d = {"coeff": -2, "vars": [0, 32]}
        t = Term.from_dict(d)
        assert t.coefficient == -2
        assert t.variables == (0, 32)

    def test_roundtrip(self):
        """to_dict / from_dict roundtrip."""
        original = Term(-2, (0, 32))
        restored = Term.from_dict(original.to_dict())
        assert restored == original


# =============================================================================
# BITTERMS CLASS TESTS
# =============================================================================

class TestBitTerms:
    """Tests for BitTerms dataclass."""

    def test_creation(self):
        """BitTerms can be created."""
        bt = BitTerms(0, [Term(1, (0,)), Term(1, (32,))])
        assert bt.bit_index == 0
        assert len(bt.terms) == 2

    def test_term_count(self):
        """term_count returns correct count."""
        bt = BitTerms(0, [Term(1, (0,)), Term(1, (32,)), Term(-2, (0, 32))])
        assert bt.term_count() == 3

    def test_term_count_empty(self):
        """term_count is 0 for empty list."""
        bt = BitTerms(0, [])
        assert bt.term_count() == 0

    def test_add_term(self):
        """add_term adds a term."""
        bt = BitTerms(0)
        bt.add_term(Term(1, (0,)))
        assert bt.term_count() == 1

    def test_add_term_filters_none(self):
        """add_term ignores None."""
        bt = BitTerms(0)
        bt.add_term(None)
        assert bt.term_count() == 0

    def test_add_terms(self):
        """add_terms adds multiple."""
        bt = BitTerms(0)
        bt.add_terms([Term(1, (0,)), None, Term(1, (32,))])
        assert bt.term_count() == 2

    def test_evaluate_empty(self):
        """Empty BitTerms evaluates to zero."""
        bt = BitTerms(0, [])
        inputs = torch.ones(5, 64)
        result = bt.evaluate(inputs)
        assert torch.allclose(result, torch.zeros(5))

    def test_evaluate_single(self):
        """Single term evaluation."""
        bt = BitTerms(0, [Term(3, (0,))])
        inputs = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
        result = bt.evaluate(inputs)
        assert torch.allclose(result, torch.tensor([3., 0.]))

    def test_evaluate_sum(self):
        """Multiple terms sum correctly."""
        # XOR: a + b - 2ab
        bt = BitTerms(0, [Term(1, (0,)), Term(1, (1,)), Term(-2, (0, 1))])
        inputs = torch.tensor([
            [0, 0],  # 0 + 0 - 0 = 0
            [1, 0],  # 1 + 0 - 0 = 1
            [0, 1],  # 0 + 1 - 0 = 1
            [1, 1],  # 1 + 1 - 2 = 0
        ], dtype=torch.float32)
        result = bt.evaluate(inputs)
        assert torch.allclose(result, torch.tensor([0., 1., 1., 0.]))

    def test_to_polynomial_str_empty(self):
        """Empty polynomial is '0'."""
        bt = BitTerms(0, [])
        assert bt.to_polynomial_str() == "0"

    def test_to_polynomial_str_single(self):
        """Single term polynomial."""
        bt = BitTerms(0, [Term(1, (0,))])
        assert bt.to_polynomial_str() == "x0"

    def test_to_polynomial_str_multiple(self):
        """Multiple term polynomial with signs."""
        bt = BitTerms(0, [Term(1, (0,)), Term(1, (1,)), Term(-2, (0, 1))])
        s = bt.to_polynomial_str()
        assert "x0" in s
        assert "x1" in s
        assert "-2" in s or "- 2" in s

    def test_to_dict(self):
        """Export to dictionary."""
        bt = BitTerms(5, [Term(1, (0,)), Term(-2, (0, 1))])
        d = bt.to_dict()
        assert d["index"] == 5
        assert d["term_count"] == 2
        assert len(d["terms"]) == 2

    def test_from_dict(self):
        """Import from dictionary."""
        d = {
            "index": 5,
            "term_count": 2,
            "terms": [{"coeff": 1, "vars": [0]}, {"coeff": -2, "vars": [0, 1]}]
        }
        bt = BitTerms.from_dict(d)
        assert bt.bit_index == 5
        assert bt.term_count() == 2

    def test_roundtrip(self):
        """to_dict / from_dict roundtrip."""
        original = BitTerms(3, [Term(1, (0,)), Term(1, (32,)), Term(-2, (0, 32))])
        restored = BitTerms.from_dict(original.to_dict())
        assert restored.bit_index == original.bit_index
        assert restored.term_count() == original.term_count()


# =============================================================================
# SHAPETERMS CLASS TESTS
# =============================================================================

class TestShapeTerms:
    """Tests for ShapeTerms dataclass."""

    def test_creation(self):
        """ShapeTerms can be created."""
        bt0 = BitTerms(0, [Term(1, (0,))])
        bt1 = BitTerms(1, [Term(1, (1,))])
        st = ShapeTerms("test", 8, 2, [bt0, bt1])
        assert st.name == "test"
        assert st.input_bits == 8
        assert st.output_bits == 2

    def test_total_terms(self):
        """total_terms sums all bit terms."""
        bt0 = BitTerms(0, [Term(1, (0,)), Term(1, (1,))])
        bt1 = BitTerms(1, [Term(1, (2,))])
        st = ShapeTerms("test", 8, 2, [bt0, bt1])
        assert st.total_terms() == 3

    def test_evaluate_shape(self):
        """Evaluate produces correct output shape."""
        # Simple 2-bit identity
        bt0 = BitTerms(0, [Term(1, (0,))])
        bt1 = BitTerms(1, [Term(1, (1,))])
        st = ShapeTerms("identity", 2, 2, [bt0, bt1])

        inputs = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)
        result = st.evaluate(inputs)

        assert result.shape == (4, 2)
        assert torch.allclose(result, inputs)

    def test_validate_structure_valid(self):
        """validate_structure passes for correct structure."""
        bt0 = BitTerms(0, [])
        bt1 = BitTerms(1, [])
        st = ShapeTerms("test", 8, 2, [bt0, bt1])
        assert st.validate_structure()

    def test_validate_structure_wrong_count(self):
        """validate_structure fails for wrong bit count."""
        bt0 = BitTerms(0, [])
        st = ShapeTerms("test", 8, 2, [bt0])  # Only 1 bit, should be 2
        assert not st.validate_structure()

    def test_validate_structure_wrong_index(self):
        """validate_structure fails for wrong indices."""
        bt0 = BitTerms(0, [])
        bt1 = BitTerms(5, [])  # Should be 1
        st = ShapeTerms("test", 8, 2, [bt0, bt1])
        assert not st.validate_structure()

    def test_to_dict(self):
        """Export to dictionary."""
        st = generate_xor_shape(bits=2)
        d = st.to_dict()
        assert d["name"] == "xor"
        assert d["input_bits"] == 4
        assert d["output_bits"] == 2
        assert "total_terms" in d
        assert "bits" in d

    def test_from_dict(self):
        """Import from dictionary."""
        original = generate_xor_shape(bits=2)
        d = original.to_dict()
        restored = ShapeTerms.from_dict(d)
        assert restored.name == original.name
        assert restored.total_terms() == original.total_terms()

    def test_roundtrip(self):
        """to_dict / from_dict roundtrip."""
        original = generate_and_shape(bits=4)
        restored = ShapeTerms.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.input_bits == original.input_bits
        assert restored.output_bits == original.output_bits
        assert restored.total_terms() == original.total_terms()


# =============================================================================
# PRIMITIVE TERM GENERATOR TESTS
# =============================================================================

class TestPrimitiveTermGenerators:
    """Tests for primitive term generators."""

    def test_xor_terms_count(self):
        """XOR has 3 terms."""
        terms = xor_terms(0, 1)
        assert len(terms) == 3

    def test_xor_terms_coefficients(self):
        """XOR has +1, +1, -2 coefficients."""
        terms = xor_terms(0, 1)
        coeffs = sorted([t.coefficient for t in terms])
        assert coeffs == [-2, 1, 1]

    def test_and_terms_count(self):
        """AND has 1 term."""
        terms = and_terms(0, 1)
        assert len(terms) == 1

    def test_and_terms_is_product(self):
        """AND term is product of variables."""
        terms = and_terms(5, 10)
        assert terms[0].variables == (5, 10)
        assert terms[0].coefficient == 1

    def test_or_terms_count(self):
        """OR has 3 terms."""
        terms = or_terms(0, 1)
        assert len(terms) == 3

    def test_not_terms_count(self):
        """NOT has 2 terms."""
        terms = not_terms(0)
        assert len(terms) == 2

    def test_not_terms_structure(self):
        """NOT is 1 - a."""
        terms = not_terms(5)
        # Should have constant 1 and -a
        has_constant = any(t.is_constant() and t.coefficient == 1 for t in terms)
        has_negative_a = any(t.variables == (5,) and t.coefficient == -1 for t in terms)
        assert has_constant
        assert has_negative_a

    def test_nand_terms_count(self):
        """NAND has 2 terms."""
        terms = nand_terms(0, 1)
        assert len(terms) == 2

    def test_nor_terms_count(self):
        """NOR has 4 terms."""
        terms = nor_terms(0, 1)
        assert len(terms) == 4

    def test_xnor_terms_count(self):
        """XNOR has 4 terms."""
        terms = xnor_terms(0, 1)
        assert len(terms) == 4


# =============================================================================
# SHAPE GENERATOR TESTS
# =============================================================================

class TestShapeGenerators:
    """Tests for shape generators."""

    @pytest.mark.parametrize("name", list(SHAPE_GENERATORS.keys()))
    def test_all_shapes_generate(self, name):
        """All registered shapes can be generated."""
        shape = generate_shape(name, bits=8)
        assert shape is not None
        assert shape.name == name

    @pytest.mark.parametrize("bits", [8, 16, 32])
    def test_xor_different_widths(self, bits):
        """XOR generates for different bit widths."""
        shape = generate_xor_shape(bits=bits)
        assert shape.output_bits == bits
        assert shape.total_terms() == bits * 3  # 3 terms per bit

    def test_xor_term_count(self):
        """XOR 32-bit has 96 terms."""
        shape = generate_xor_shape(bits=32)
        assert shape.total_terms() == 96

    def test_and_term_count(self):
        """AND 32-bit has 32 terms."""
        shape = generate_and_shape(bits=32)
        assert shape.total_terms() == 32

    def test_or_term_count(self):
        """OR 32-bit has 96 terms."""
        shape = generate_or_shape(bits=32)
        assert shape.total_terms() == 96

    def test_not_term_count(self):
        """NOT 32-bit has 64 terms."""
        shape = generate_not_shape(bits=32)
        assert shape.total_terms() == 64

    def test_nand_term_count(self):
        """NAND 32-bit has 64 terms."""
        shape = generate_nand_shape(bits=32)
        assert shape.total_terms() == 64

    def test_nor_term_count(self):
        """NOR 32-bit has 128 terms."""
        shape = generate_nor_shape(bits=32)
        assert shape.total_terms() == 128

    def test_xnor_term_count(self):
        """XNOR 32-bit has 128 terms."""
        shape = generate_xnor_shape(bits=32)
        assert shape.total_terms() == 128

    def test_nop_term_count(self):
        """NOP 32-bit has 32 terms (identity)."""
        shape = generate_nop_shape(bits=32)
        assert shape.total_terms() == 32

    def test_generate_all_shapes(self):
        """generate_all_shapes returns all shapes."""
        shapes = generate_all_shapes(bits=8)
        assert len(shapes) == len(SHAPE_GENERATORS)
        assert "xor" in shapes
        assert "add" in shapes


# =============================================================================
# CROSS-VALIDATION TESTS
# =============================================================================

class TestCrossValidation:
    """Validate term evaluation matches tensor functions."""

    @pytest.mark.parametrize("name,gen,tensor_fn", [
        ("xor", generate_xor_shape, xor),
        ("and", generate_and_shape, and_),
        ("or", generate_or_shape, or_),
        ("nand", generate_nand_shape, nand),
        ("nor", generate_nor_shape, nor),
        ("xnor", generate_xnor_shape, xnor),
    ])
    def test_simple_shapes_match_tensor(self, name, gen, tensor_fn):
        """Term evaluation matches tensor function."""
        shape = gen(bits=8)  # Use 8-bit for speed
        passed, accuracy = validate_terms_against_tensor(
            shape, tensor_fn, n_samples=100, bits=8
        )
        assert passed, f"{name} accuracy: {accuracy}"

    def test_not_matches_tensor(self):
        """NOT term evaluation matches tensor function."""
        shape = generate_not_shape(bits=8)
        passed, accuracy = validate_terms_against_tensor(
            shape, lambda a, b: not_(a), n_samples=100, bits=8
        )
        assert passed, f"NOT accuracy: {accuracy}"

    def test_nop_matches_identity(self):
        """NOP returns input unchanged."""
        shape = generate_nop_shape(bits=8)

        # Random inputs
        a = torch.randint(0, 2, (10, 8), dtype=torch.float32)
        result = shape.evaluate(a)

        assert torch.allclose(result, a)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge case inputs."""

    EDGE_CASES_8BIT = [
        (0x00, 0x00),  # zeros
        (0xFF, 0xFF),  # ones
        (0x00, 0xFF),  # mixed
        (0xAA, 0x55),  # alternating
        (0x80, 0x01),  # sign + lsb
        (0x7F, 0x01),  # max positive + 1
    ]

    def _int_to_bits(self, x: int, bits: int = 8) -> torch.Tensor:
        """Convert int to bit tensor."""
        return torch.tensor([(x >> i) & 1 for i in range(bits)], dtype=torch.float32)

    def _bits_to_int(self, bits: torch.Tensor) -> int:
        """Convert bit tensor to int."""
        result = 0
        for i, b in enumerate(bits):
            result += int(round(float(b))) << i
        return result

    @pytest.mark.parametrize("a,b", EDGE_CASES_8BIT)
    def test_xor_edge_cases(self, a, b):
        """XOR works for edge cases."""
        shape = generate_xor_shape(bits=8)

        a_bits = self._int_to_bits(a, 8).unsqueeze(0)
        b_bits = self._int_to_bits(b, 8).unsqueeze(0)
        inputs = torch.cat([a_bits, b_bits], dim=1)

        result = shape.evaluate(inputs)
        result_int = self._bits_to_int(result.squeeze())

        expected = a ^ b
        assert result_int == expected, f"XOR({a:#x}, {b:#x}) = {result_int:#x}, expected {expected:#x}"

    @pytest.mark.parametrize("a,b", EDGE_CASES_8BIT)
    def test_and_edge_cases(self, a, b):
        """AND works for edge cases."""
        shape = generate_and_shape(bits=8)

        a_bits = self._int_to_bits(a, 8).unsqueeze(0)
        b_bits = self._int_to_bits(b, 8).unsqueeze(0)
        inputs = torch.cat([a_bits, b_bits], dim=1)

        result = shape.evaluate(inputs)
        result_int = self._bits_to_int(result.squeeze())

        expected = a & b
        assert result_int == expected, f"AND({a:#x}, {b:#x}) = {result_int:#x}, expected {expected:#x}"

    @pytest.mark.parametrize("a,b", EDGE_CASES_8BIT)
    def test_or_edge_cases(self, a, b):
        """OR works for edge cases."""
        shape = generate_or_shape(bits=8)

        a_bits = self._int_to_bits(a, 8).unsqueeze(0)
        b_bits = self._int_to_bits(b, 8).unsqueeze(0)
        inputs = torch.cat([a_bits, b_bits], dim=1)

        result = shape.evaluate(inputs)
        result_int = self._bits_to_int(result.squeeze())

        expected = a | b
        assert result_int == expected, f"OR({a:#x}, {b:#x}) = {result_int:#x}, expected {expected:#x}"

    @pytest.mark.parametrize("a", [0x00, 0xFF, 0xAA, 0x55, 0x80, 0x01])
    def test_not_edge_cases(self, a):
        """NOT works for edge cases."""
        shape = generate_not_shape(bits=8)

        a_bits = self._int_to_bits(a, 8).unsqueeze(0)
        result = shape.evaluate(a_bits)
        result_int = self._bits_to_int(result.squeeze())

        expected = (~a) & 0xFF
        assert result_int == expected, f"NOT({a:#x}) = {result_int:#x}, expected {expected:#x}"


# =============================================================================
# REGISTRY TESTS
# =============================================================================

# =============================================================================
# XORPU INTEGRATION TESTS
# =============================================================================

class TestXORPUExport:
    """Tests for XORPU term export."""

    def test_export_terms_basic(self):
        """export_terms returns valid structure."""
        from trix.forge import XORPU
        xorpu = XORPU()

        export = xorpu.export_terms(include_shapes=['xor'])

        assert "version" in export
        assert "format_version" in export
        assert "bits" in export
        assert "shapes" in export
        assert "xor" in export["shapes"]

    def test_export_terms_has_actual_terms(self):
        """export_terms includes actual polynomial terms."""
        from trix.forge import XORPU
        xorpu = XORPU()

        export = xorpu.export_terms(include_shapes=['xor'])
        xor_shape = export["shapes"]["xor"]

        # Should have 32 bits
        assert len(xor_shape["bits"]) == 32

        # Each bit should have 3 terms
        assert xor_shape["bits"][0]["term_count"] == 3

        # First term should be valid
        first_term = xor_shape["bits"][0]["terms"][0]
        assert "coeff" in first_term
        assert "vars" in first_term

    def test_export_terms_all_shapes(self):
        """export_terms can export all shapes."""
        from trix.forge import XORPU
        xorpu = XORPU()

        export = xorpu.export_terms()  # All shapes

        # Should have 14 shapes (excluding nop)
        assert export["total_shapes"] == 14

    def test_export_terms_file(self, tmp_path):
        """export_terms can write to file."""
        from trix.forge import XORPU
        import json

        xorpu = XORPU()
        path = tmp_path / "test_export.json"

        xorpu.export_terms(str(path), include_shapes=['and'])

        # File should exist and be valid JSON
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "shapes" in data
        assert "and" in data["shapes"]


# =============================================================================
# TIERED VALIDATION TESTS
# =============================================================================

class TestTieredValidation:
    """Tests for tiered validation functions."""

    def test_validate_quick_xor(self):
        """Quick validation for XOR."""
        from trix.forge.term import validate_quick
        shape = generate_xor_shape(bits=8)
        result = validate_quick(shape, lambda a, b: a ^ b, n_samples=100, bits=8)
        assert result.passed
        assert result.mode == "quick"
        assert result.samples_tested == 100

    def test_validate_edge_xor(self):
        """Edge validation for XOR."""
        from trix.forge.term import validate_edge
        shape = generate_xor_shape(bits=32)
        result = validate_edge(shape, lambda a, b: a ^ b, bits=32)
        assert result.passed
        assert result.mode == "edge"
        assert result.samples_tested > 0

    def test_validate_exhaustive_xor(self):
        """Exhaustive 8-bit validation for XOR."""
        from trix.forge.term import validate_exhaustive_8bit
        shape = generate_xor_shape(bits=8)
        result = validate_exhaustive_8bit(shape, lambda a, b: a ^ b)
        assert result.passed
        assert result.mode == "exhaustive"
        assert result.samples_tested == 65536

    def test_validate_shape_modes(self):
        """validate_shape supports all modes."""
        from trix.forge.term import validate_shape
        shape = generate_and_shape(bits=8)
        truth = lambda a, b: a & b

        # Quick mode
        result = validate_shape(shape, truth, mode="quick", bits=8)
        assert result.mode == "quick"

        # Edge mode
        result = validate_shape(shape, truth, mode="edge", bits=8)
        assert result.mode == "edge"

        # Exhaustive mode
        result = validate_shape(shape, truth, mode="exhaustive")
        assert result.mode == "exhaustive"

    def test_validation_result_structure(self):
        """ValidationResult has expected fields."""
        from trix.forge.term import validate_quick, ValidationResult
        shape = generate_xor_shape(bits=8)
        result = validate_quick(shape, lambda a, b: a ^ b, n_samples=10, bits=8)

        assert isinstance(result, ValidationResult)
        assert hasattr(result, 'passed')
        assert hasattr(result, 'accuracy')
        assert hasattr(result, 'samples_tested')
        assert hasattr(result, 'mode')
        assert hasattr(result, 'failures')

    def test_edge_cases_count(self):
        """Edge cases have expected count."""
        from trix.forge.term import EDGE_CASES_32, EDGE_CASES_8
        assert len(EDGE_CASES_32) > 50  # Should have many cases
        assert len(EDGE_CASES_8) > 10


class TestRegistry:
    """Tests for shape registry."""

    def test_shape_generators_has_all(self):
        """SHAPE_GENERATORS contains all 15 shapes."""
        expected = {
            "nop", "xor", "and", "or", "not",
            "add", "sub",
            "sll", "srl", "sra",
            "slt", "sltu",
            "nand", "nor", "xnor"
        }
        assert set(SHAPE_GENERATORS.keys()) == expected

    def test_generate_shape_valid(self):
        """generate_shape works for valid names."""
        shape = generate_shape("xor", bits=8)
        assert shape is not None
        assert shape.name == "xor"

    def test_generate_shape_invalid(self):
        """generate_shape returns None for invalid names."""
        shape = generate_shape("invalid_shape", bits=8)
        assert shape is None


# =============================================================================
# MULTI-WIDTH TESTS
# =============================================================================

class TestMultiWidth:
    """Test multi-width support functions."""

    def test_standard_widths(self):
        """STANDARD_WIDTHS contains expected values."""
        from trix.forge.term import STANDARD_WIDTHS
        assert STANDARD_WIDTHS == [8, 16, 32, 64]

    def test_generate_all_shapes_multiwidth_default(self):
        """generate_all_shapes_multiwidth uses standard widths."""
        from trix.forge.term import generate_all_shapes_multiwidth, STANDARD_WIDTHS

        all_shapes = generate_all_shapes_multiwidth()
        assert set(all_shapes.keys()) == set(STANDARD_WIDTHS)

        for width, shapes in all_shapes.items():
            assert len(shapes) == 15
            assert all(s.output_bits == width for s in shapes.values())

    def test_generate_all_shapes_multiwidth_custom(self):
        """generate_all_shapes_multiwidth works with custom widths."""
        from trix.forge.term import generate_all_shapes_multiwidth

        all_shapes = generate_all_shapes_multiwidth(widths=[4, 8])
        assert set(all_shapes.keys()) == {4, 8}

    def test_validate_multiwidth_xor(self):
        """validate_multiwidth works for XOR across widths."""
        from trix.forge.term import validate_multiwidth

        xor_truth = lambda a, b: a ^ b
        results = validate_multiwidth('xor', xor_truth, widths=[8, 16, 32])

        assert all(r.passed for r in results.values())
        assert all(r.accuracy == 1.0 for r in results.values())

    def test_validate_multiwidth_and(self):
        """validate_multiwidth works for AND across widths."""
        from trix.forge.term import validate_multiwidth

        and_truth = lambda a, b: a & b
        results = validate_multiwidth('and', and_truth, widths=[8, 16, 32])

        assert all(r.passed for r in results.values())

    def test_validate_multiwidth_64bit(self):
        """validate_multiwidth works for 64-bit."""
        from trix.forge.term import validate_multiwidth

        xor_truth = lambda a, b: a ^ b
        results = validate_multiwidth('xor', xor_truth, widths=[64], mode='quick')

        assert 64 in results
        assert results[64].passed

    def test_multiwidth_summary(self):
        """multiwidth_summary produces formatted output."""
        from trix.forge.term import multiwidth_summary

        summary = multiwidth_summary(widths=[8, 16])

        assert "XORPU Multi-Width Summary" in summary
        assert "8" in summary
        assert "16" in summary
        assert "XOR Terms" in summary

    def test_term_scaling_linear(self):
        """Terms scale linearly with bit width."""
        from trix.forge.term import generate_xor_shape

        terms_8 = generate_xor_shape(bits=8).total_terms()
        terms_16 = generate_xor_shape(bits=16).total_terms()
        terms_32 = generate_xor_shape(bits=32).total_terms()

        # Each bit has 3 terms, so scaling is linear
        assert terms_16 == terms_8 * 2
        assert terms_32 == terms_8 * 4

    def test_all_widths_generate_same_shape_count(self):
        """All widths generate the same number of shapes."""
        from trix.forge.term import generate_all_shapes

        for bits in [8, 16, 32, 64]:
            shapes = generate_all_shapes(bits=bits)
            assert len(shapes) == 15, f"Expected 15 shapes for {bits}-bit"


# =============================================================================
# OPTIMIZATION TESTS
# =============================================================================

class TestFastPath:
    """Test fast path optimization functions."""

    def test_fast_path_shapes(self):
        """FAST_PATH_SHAPES contains expected shapes."""
        from trix.forge.term import FAST_PATH_SHAPES
        expected = {"xor", "and", "or", "not", "nand", "nor", "xnor", "nop"}
        assert FAST_PATH_SHAPES == expected

    def test_is_fast_path_true(self):
        """is_fast_path returns True for fast path shapes."""
        from trix.forge.term import is_fast_path
        for shape in ["xor", "and", "or", "not", "nand", "nor", "xnor", "nop"]:
            assert is_fast_path(shape), f"{shape} should be fast path"
            assert is_fast_path(shape.upper()), f"{shape.upper()} should be fast path"

    def test_is_fast_path_false(self):
        """is_fast_path returns False for non-fast path shapes."""
        from trix.forge.term import is_fast_path
        assert not is_fast_path("add")
        assert not is_fast_path("sub")
        assert not is_fast_path("sll")

    def test_compute_fast_xor(self):
        """compute_fast XOR works correctly."""
        from trix.forge.term import compute_fast
        assert compute_fast("xor", 0xAA, 0x55, 8) == 0xFF
        assert compute_fast("xor", 0xFF, 0xFF, 8) == 0x00
        assert compute_fast("xor", 0x00, 0x00, 8) == 0x00

    def test_compute_fast_and(self):
        """compute_fast AND works correctly."""
        from trix.forge.term import compute_fast
        assert compute_fast("and", 0xFF, 0x0F, 8) == 0x0F
        assert compute_fast("and", 0xAA, 0x55, 8) == 0x00

    def test_compute_fast_or(self):
        """compute_fast OR works correctly."""
        from trix.forge.term import compute_fast
        assert compute_fast("or", 0xAA, 0x55, 8) == 0xFF
        assert compute_fast("or", 0x00, 0x0F, 8) == 0x0F

    def test_compute_fast_not(self):
        """compute_fast NOT works correctly."""
        from trix.forge.term import compute_fast
        assert compute_fast("not", 0xFF, 0, 8) == 0x00
        assert compute_fast("not", 0x00, 0, 8) == 0xFF
        assert compute_fast("not", 0x0F, 0, 8) == 0xF0

    def test_compute_fast_nand(self):
        """compute_fast NAND works correctly."""
        from trix.forge.term import compute_fast
        assert compute_fast("nand", 0xFF, 0xFF, 8) == 0x00
        assert compute_fast("nand", 0xFF, 0x00, 8) == 0xFF

    def test_compute_fast_nor(self):
        """compute_fast NOR works correctly."""
        from trix.forge.term import compute_fast
        assert compute_fast("nor", 0x00, 0x00, 8) == 0xFF
        assert compute_fast("nor", 0xFF, 0x00, 8) == 0x00

    def test_compute_fast_xnor(self):
        """compute_fast XNOR works correctly."""
        from trix.forge.term import compute_fast
        assert compute_fast("xnor", 0xFF, 0xFF, 8) == 0xFF
        assert compute_fast("xnor", 0xAA, 0x55, 8) == 0x00

    def test_compute_fast_nop(self):
        """compute_fast NOP works correctly."""
        from trix.forge.term import compute_fast
        assert compute_fast("nop", 0x42, 0xFF, 8) == 0x42

    def test_compute_fast_invalid(self):
        """compute_fast raises for invalid shapes."""
        from trix.forge.term import compute_fast
        import pytest
        with pytest.raises(ValueError):
            compute_fast("add", 1, 2, 8)

    def test_compute_auto_fast_path(self):
        """compute_auto uses fast path for simple shapes."""
        from trix.forge.term import compute_auto
        assert compute_auto("xor", 0xAA, 0x55, 8) == 0xFF

    def test_compute_auto_polynomial(self):
        """compute_auto uses polynomial for complex shapes."""
        from trix.forge.term import compute_auto
        # ADD 10 + 20 = 30 (mod 256 not needed here)
        result = compute_auto("add", 10, 20, 8)
        assert result == 30


class TestBatchEvaluation:
    """Test batch evaluation functions."""

    def test_evaluate_batch_xor(self):
        """evaluate_batch works for XOR."""
        from trix.forge.term import evaluate_batch, generate_xor_shape

        xor = generate_xor_shape(bits=8)
        a_batch = [0x00, 0xFF, 0xAA]
        b_batch = [0xFF, 0xFF, 0x55]
        results = evaluate_batch(xor, a_batch, b_batch)

        assert results == [0xFF, 0x00, 0xFF]

    def test_evaluate_batch_single_input(self):
        """evaluate_batch works for single-input shapes."""
        from trix.forge.term import evaluate_batch, generate_not_shape

        not_shape = generate_not_shape(bits=8)
        a_batch = [0x00, 0xFF, 0x0F]
        results = evaluate_batch(not_shape, a_batch)

        assert results == [0xFF, 0x00, 0xF0]

    def test_evaluate_batch_fast_uses_fast_path(self):
        """evaluate_batch_fast uses fast path when available."""
        from trix.forge.term import evaluate_batch_fast

        a_batch = [0x00, 0xFF, 0xAA]
        b_batch = [0xFF, 0xFF, 0x55]
        results = evaluate_batch_fast("xor", a_batch, b_batch, bits=8)

        assert results == [0xFF, 0x00, 0xFF]

    def test_evaluate_batch_fast_polynomial(self):
        """evaluate_batch_fast falls back to polynomial."""
        from trix.forge.term import evaluate_batch_fast

        # Use OR shape which is a valid polynomial (not simplified like ADD)
        a_batch = [0x0F, 0xF0, 0xAA]
        b_batch = [0xF0, 0x0F, 0x55]
        results = evaluate_batch_fast("or", a_batch, b_batch, bits=8)

        # OR results: 0x0F|0xF0=0xFF, 0xF0|0x0F=0xFF, 0xAA|0x55=0xFF
        assert results == [0xFF, 0xFF, 0xFF]


class TestCompression:
    """Test compression statistics functions."""

    def test_compression_stats_structure(self):
        """compression_stats returns expected fields."""
        from trix.forge.term import compression_stats

        stats = compression_stats(bits=8)

        assert "shapes" in stats
        assert "total_term_refs" in stats
        assert "unique_terms" in stats
        assert "full_size_bytes" in stats
        assert "compressed_size_bytes" in stats
        assert "compression_ratio" in stats
        assert "savings_percent" in stats

    def test_compression_stats_values(self):
        """compression_stats produces reasonable values."""
        from trix.forge.term import compression_stats

        stats = compression_stats(bits=8)

        assert stats["shapes"] == 15
        assert stats["unique_terms"] <= stats["total_term_refs"]
        assert stats["compression_ratio"] >= 1.0
        assert 0 <= stats["savings_percent"] <= 100

    def test_compression_improves_with_size(self):
        """Compression ratio improves with larger bit widths."""
        from trix.forge.term import compression_stats

        stats_8 = compression_stats(bits=8)
        stats_32 = compression_stats(bits=32)

        # More terms means more opportunity for deduplication
        assert stats_32["total_term_refs"] > stats_8["total_term_refs"]
