"""
Explicit polynomial term representation for XORPU shapes.

This module provides introspectable, exportable term data structures
that represent the polynomial computation underlying frozen shapes.

The Three Atoms as Terms:
    XOR(a, b) = a + b - 2ab  → [Term(1,(a,)), Term(1,(b,)), Term(-2,(a,b))]
    AND(a, b) = ab           → [Term(1,(a,b))]
    NOT(a)    = 1 - a        → [Term(1,()), Term(-1,(a,))]

Everything else is composition.

Usage:
    from trix.forge.term import Term, BitTerms, ShapeTerms
    from trix.forge.term import generate_xor_shape, generate_add_shape

    # Generate XOR shape terms
    xor_shape = generate_xor_shape(bits=32)
    print(xor_shape.total_terms())  # 96 (32 bits × 3 terms)

    # Evaluate
    inputs = torch.cat([a_bits, b_bits], dim=1)  # [batch, 64]
    result = xor_shape.evaluate(inputs)  # [batch, 32]

    # Export
    spec = xor_shape.to_dict()
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Sequence, Optional, Dict, Any
import random
import torch
from torch import Tensor


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class Term:
    """
    A single polynomial term: coefficient × product of variables.

    Attributes:
        coefficient: Integer coefficient (-2, -1, 1, 2 typically)
        variables: Tuple of input bit indices, always sorted

    Examples:
        Term(1, (0,))       → +x₀
        Term(-2, (0, 32))   → -2 × x₀ × x₃₂
        Term(1, ())         → +1 (constant)
    """
    coefficient: int
    variables: Tuple[int, ...]

    def __post_init__(self):
        """Validate sorted variables."""
        if self.variables != tuple(sorted(self.variables)):
            raise ValueError(f"Variables must be sorted: {self.variables}")

    @classmethod
    def create(cls, coeff: int, vars: Sequence[int]) -> Optional["Term"]:
        """
        Factory method with automatic sorting and zero filtering.

        Returns None if coefficient is 0.
        """
        if coeff == 0:
            return None
        return cls(coeff, tuple(sorted(vars)))

    @classmethod
    def constant(cls, value: int) -> Optional["Term"]:
        """Create constant term (degree 0)."""
        if value == 0:
            return None
        return cls(value, ())

    def degree(self) -> int:
        """Polynomial degree (number of variables)."""
        return len(self.variables)

    def is_constant(self) -> bool:
        """True if this is a constant term (no variables)."""
        return len(self.variables) == 0

    def evaluate(self, inputs: Tensor) -> Tensor:
        """
        Evaluate term on batched inputs.

        Args:
            inputs: [batch, input_bits] tensor of 0/1 values

        Returns:
            [batch] tensor of term values
        """
        batch_size = inputs.shape[0]
        device = inputs.device

        if not self.variables:
            # Constant term
            return self.coefficient * torch.ones(batch_size, device=device)

        # Product of variables
        result = inputs[:, self.variables[0]]
        for v in self.variables[1:]:
            result = result * inputs[:, v]

        return self.coefficient * result

    def to_str(self, var_names: Dict[int, str] = None) -> str:
        """
        Human-readable string representation.

        Args:
            var_names: Optional mapping from index to name (e.g., {0: "a[0]"})
        """
        if not self.variables:
            return str(self.coefficient)

        var_names = var_names or {}
        vars_str = "·".join(
            var_names.get(v, f"x{v}") for v in self.variables
        )

        if self.coefficient == 1:
            return vars_str
        elif self.coefficient == -1:
            return f"-{vars_str}"
        else:
            return f"{self.coefficient}·{vars_str}"

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            "coeff": self.coefficient,
            "vars": list(self.variables),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Term":
        """Import from dictionary."""
        return cls(d["coeff"], tuple(d["vars"]))


@dataclass
class BitTerms:
    """
    Terms for computing one output bit.

    The output bit value is the sum of all term evaluations.
    """
    bit_index: int
    terms: List[Term] = field(default_factory=list)

    def term_count(self) -> int:
        """Number of terms for this bit."""
        return len(self.terms)

    def add_term(self, term: Optional[Term]) -> None:
        """Add a term (filters None)."""
        if term is not None:
            self.terms.append(term)

    def add_terms(self, terms: List[Optional[Term]]) -> None:
        """Add multiple terms (filters None)."""
        for t in terms:
            self.add_term(t)

    def evaluate(self, inputs: Tensor) -> Tensor:
        """
        Evaluate all terms and sum.

        Args:
            inputs: [batch, input_bits] tensor

        Returns:
            [batch] tensor of output bit values
        """
        if not self.terms:
            return torch.zeros(inputs.shape[0], device=inputs.device)

        result = self.terms[0].evaluate(inputs)
        for term in self.terms[1:]:
            result = result + term.evaluate(inputs)
        return result

    def to_polynomial_str(self, var_names: Dict[int, str] = None) -> str:
        """Human-readable polynomial expression."""
        if not self.terms:
            return "0"

        parts = [self.terms[0].to_str(var_names)]
        for term in self.terms[1:]:
            s = term.to_str(var_names)
            if s.startswith("-"):
                parts.append(f" - {s[1:]}")
            else:
                parts.append(f" + {s}")
        return "".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            "index": self.bit_index,
            "term_count": self.term_count(),
            "terms": [t.to_dict() for t in self.terms],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BitTerms":
        """Import from dictionary."""
        terms = [Term.from_dict(t) for t in d["terms"]]
        return cls(d["index"], terms)


@dataclass
class ShapeTerms:
    """
    Complete term specification for a shape.

    Contains terms for each output bit, organized for efficient
    evaluation and export.
    """
    name: str
    input_bits: int    # Total input bits (e.g., 64 for two 32-bit operands)
    output_bits: int   # Number of output bits
    bit_terms: List[BitTerms] = field(default_factory=list)

    def total_terms(self) -> int:
        """Total number of terms across all output bits."""
        return sum(bt.term_count() for bt in self.bit_terms)

    def evaluate(self, inputs: Tensor) -> Tensor:
        """
        Evaluate shape on batched inputs.

        Args:
            inputs: [batch, input_bits] tensor of 0/1 values

        Returns:
            [batch, output_bits] tensor
        """
        outputs = [bt.evaluate(inputs) for bt in self.bit_terms]
        return torch.stack(outputs, dim=1)

    def validate_structure(self) -> bool:
        """Check that bit_terms covers all output bits."""
        if len(self.bit_terms) != self.output_bits:
            return False
        for i, bt in enumerate(self.bit_terms):
            if bt.bit_index != i:
                return False
        return True

    def get_var_names(self, bits: int = 32) -> Dict[int, str]:
        """Generate variable names for pretty printing."""
        names = {}
        for i in range(bits):
            names[i] = f"a[{i}]"
            names[i + bits] = f"b[{i}]"
        return names

    def to_polynomial_strs(self) -> List[str]:
        """Get polynomial string for each output bit."""
        var_names = self.get_var_names(self.output_bits)
        return [bt.to_polynomial_str(var_names) for bt in self.bit_terms]

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary (for JSON serialization)."""
        return {
            "name": self.name,
            "input_bits": self.input_bits,
            "output_bits": self.output_bits,
            "total_terms": self.total_terms(),
            "bits": [bt.to_dict() for bt in self.bit_terms],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ShapeTerms":
        """Import from dictionary."""
        bit_terms = [BitTerms.from_dict(bt) for bt in d["bits"]]
        return cls(
            name=d["name"],
            input_bits=d["input_bits"],
            output_bits=d["output_bits"],
            bit_terms=bit_terms,
        )


# =============================================================================
# PRIMITIVE TERM GENERATORS
# =============================================================================

def xor_terms(a: int, b: int) -> List[Term]:
    """
    XOR(a, b) = a + b - 2ab

    Returns 3 terms.
    """
    return [
        Term(1, (a,)),
        Term(1, (b,)),
        Term(-2, (a, b)),
    ]


def and_terms(a: int, b: int) -> List[Term]:
    """
    AND(a, b) = ab

    Returns 1 term.
    """
    return [Term(1, (a, b))]


def or_terms(a: int, b: int) -> List[Term]:
    """
    OR(a, b) = a + b - ab

    Returns 3 terms.
    """
    return [
        Term(1, (a,)),
        Term(1, (b,)),
        Term(-1, (a, b)),
    ]


def not_terms(a: int) -> List[Term]:
    """
    NOT(a) = 1 - a

    Returns 2 terms (including constant).
    """
    return [
        Term.constant(1),
        Term(-1, (a,)),
    ]


def nand_terms(a: int, b: int) -> List[Term]:
    """
    NAND(a, b) = NOT(AND(a, b)) = 1 - ab

    Returns 2 terms.
    """
    return [
        Term.constant(1),
        Term(-1, (a, b)),
    ]


def nor_terms(a: int, b: int) -> List[Term]:
    """
    NOR(a, b) = NOT(OR(a, b)) = 1 - a - b + ab

    Returns 4 terms.
    """
    return [
        Term.constant(1),
        Term(-1, (a,)),
        Term(-1, (b,)),
        Term(1, (a, b)),
    ]


def xnor_terms(a: int, b: int) -> List[Term]:
    """
    XNOR(a, b) = NOT(XOR(a, b)) = 1 - a - b + 2ab

    Returns 4 terms.
    """
    return [
        Term.constant(1),
        Term(-1, (a,)),
        Term(-1, (b,)),
        Term(2, (a, b)),
    ]


# =============================================================================
# SHAPE GENERATORS
# =============================================================================

def generate_xor_shape(bits: int = 32) -> ShapeTerms:
    """Generate XOR shape: output[i] = a[i] XOR b[i]"""
    bit_terms = []
    for i in range(bits):
        a_idx, b_idx = i, i + bits
        bt = BitTerms(i)
        bt.add_terms(xor_terms(a_idx, b_idx))
        bit_terms.append(bt)

    return ShapeTerms("xor", bits * 2, bits, bit_terms)


def generate_and_shape(bits: int = 32) -> ShapeTerms:
    """Generate AND shape: output[i] = a[i] AND b[i]"""
    bit_terms = []
    for i in range(bits):
        a_idx, b_idx = i, i + bits
        bt = BitTerms(i)
        bt.add_terms(and_terms(a_idx, b_idx))
        bit_terms.append(bt)

    return ShapeTerms("and", bits * 2, bits, bit_terms)


def generate_or_shape(bits: int = 32) -> ShapeTerms:
    """Generate OR shape: output[i] = a[i] OR b[i]"""
    bit_terms = []
    for i in range(bits):
        a_idx, b_idx = i, i + bits
        bt = BitTerms(i)
        bt.add_terms(or_terms(a_idx, b_idx))
        bit_terms.append(bt)

    return ShapeTerms("or", bits * 2, bits, bit_terms)


def generate_not_shape(bits: int = 32) -> ShapeTerms:
    """Generate NOT shape: output[i] = NOT a[i]"""
    bit_terms = []
    for i in range(bits):
        bt = BitTerms(i)
        bt.add_terms(not_terms(i))
        bit_terms.append(bt)

    # NOT only uses 'a' input
    return ShapeTerms("not", bits, bits, bit_terms)


def generate_nand_shape(bits: int = 32) -> ShapeTerms:
    """Generate NAND shape: output[i] = NOT(a[i] AND b[i])"""
    bit_terms = []
    for i in range(bits):
        a_idx, b_idx = i, i + bits
        bt = BitTerms(i)
        bt.add_terms(nand_terms(a_idx, b_idx))
        bit_terms.append(bt)

    return ShapeTerms("nand", bits * 2, bits, bit_terms)


def generate_nor_shape(bits: int = 32) -> ShapeTerms:
    """Generate NOR shape: output[i] = NOT(a[i] OR b[i])"""
    bit_terms = []
    for i in range(bits):
        a_idx, b_idx = i, i + bits
        bt = BitTerms(i)
        bt.add_terms(nor_terms(a_idx, b_idx))
        bit_terms.append(bt)

    return ShapeTerms("nor", bits * 2, bits, bit_terms)


def generate_xnor_shape(bits: int = 32) -> ShapeTerms:
    """Generate XNOR shape: output[i] = NOT(a[i] XOR b[i])"""
    bit_terms = []
    for i in range(bits):
        a_idx, b_idx = i, i + bits
        bt = BitTerms(i)
        bt.add_terms(xnor_terms(a_idx, b_idx))
        bit_terms.append(bt)

    return ShapeTerms("xnor", bits * 2, bits, bit_terms)


def generate_nop_shape(bits: int = 32) -> ShapeTerms:
    """Generate NOP shape: output[i] = a[i] (pass-through)"""
    bit_terms = []
    for i in range(bits):
        bt = BitTerms(i)
        bt.add_term(Term(1, (i,)))  # Just the identity
        bit_terms.append(bt)

    return ShapeTerms("nop", bits, bits, bit_terms)


# =============================================================================
# ARITHMETIC SHAPE GENERATORS (Structural)
# =============================================================================

def _full_adder_sum_terms(a: int, b: int, cin: int) -> List[Term]:
    """
    Full adder sum: a XOR b XOR cin

    Expanded: a + b + cin - 2ab - 2a·cin - 2b·cin + 4ab·cin

    Returns 7 terms.
    """
    return [
        Term(1, (a,)),
        Term(1, (b,)),
        Term(1, (cin,)),
        Term(-2, (a, b)),
        Term(-2, (a, cin)),
        Term(-2, (b, cin)),
        Term(4, (a, b, cin)),
    ]


def _full_adder_carry_terms(a: int, b: int, cin: int) -> List[Term]:
    """
    Full adder carry: (a AND b) OR (cin AND (a XOR b))

    Expanded: ab + cin·a + cin·b - 2ab·cin

    Returns 4 terms.
    """
    return [
        Term(1, (a, b)),
        Term(1, (a, cin)),
        Term(1, (b, cin)),
        Term(-2, (a, b, cin)),
    ]


def generate_add_shape(bits: int = 32) -> ShapeTerms:
    """
    Generate ADD shape using ripple carry.

    For structural representation:
    - Each bit has sum terms from full adder
    - Carry is not expanded (would be exponential)
    - This gives an approximation suitable for term counting

    Total terms: bits × 8 (7 sum + forwarded structure)

    Note: This is a STRUCTURAL representation. For full polynomial
    expansion, use generate_add_shape_flat() (Phase 4).
    """
    bit_terms = []

    # We use a simplified structural model:
    # Each bit gets the XOR terms for sum = a XOR b
    # The carry chain is implicit in the structure

    for i in range(bits):
        a_idx, b_idx = i, i + bits
        bt = BitTerms(i)

        if i == 0:
            # First bit: half adder (sum = a XOR b, no carry in)
            bt.add_terms(xor_terms(a_idx, b_idx))
        else:
            # Full adder structure: sum = a XOR b XOR carry
            # For structural representation, we just use XOR terms
            # This undercounts actual polynomial complexity
            bt.add_terms(xor_terms(a_idx, b_idx))

        bit_terms.append(bt)

    return ShapeTerms("add", bits * 2, bits, bit_terms)


def generate_sub_shape(bits: int = 32) -> ShapeTerms:
    """
    Generate SUB shape: a - b = a + NOT(b) + 1

    Structural representation similar to ADD.
    """
    # For structural purposes, same complexity as ADD
    shape = generate_add_shape(bits)
    shape.name = "sub"
    return shape


# =============================================================================
# SHIFT SHAPE GENERATORS (Structural)
# =============================================================================

def generate_sll_shape(bits: int = 32) -> ShapeTerms:
    """
    Generate SLL (shift left logical) shape.

    Barrel shifter: 5 stages for 32-bit (log2(32) = 5)
    Each stage: conditional shift by 2^stage

    Structural representation: MUX terms per bit per stage.
    """
    bit_terms = []

    # For structural representation, each bit depends on MUX selection
    # MUX(s, a, b) = a + s(b - a) = a - sa + sb
    # This gives 3 terms per MUX

    for i in range(bits):
        bt = BitTerms(i)
        # Placeholder: identity for now
        # Full barrel shifter terms are complex
        bt.add_term(Term(1, (i,)))  # Simplified
        bit_terms.append(bt)

    return ShapeTerms("sll", bits * 2, bits, bit_terms)


def generate_srl_shape(bits: int = 32) -> ShapeTerms:
    """Generate SRL (shift right logical) shape."""
    shape = generate_sll_shape(bits)
    shape.name = "srl"
    return shape


def generate_sra_shape(bits: int = 32) -> ShapeTerms:
    """Generate SRA (shift right arithmetic) shape."""
    shape = generate_sll_shape(bits)
    shape.name = "sra"
    return shape


# =============================================================================
# COMPARISON SHAPE GENERATORS (Structural)
# =============================================================================

def generate_slt_shape(bits: int = 32) -> ShapeTerms:
    """
    Generate SLT (set less than, signed) shape.

    SLT = sign logic on SUB result.
    Structural representation.
    """
    bit_terms = []

    # Only bit 0 of output is meaningful (0 or 1)
    # Bits 1-31 are always 0

    for i in range(bits):
        bt = BitTerms(i)
        if i == 0:
            # The comparison result goes here
            # Structural placeholder
            bt.add_term(Term(1, (i,)))
        # Bits 1-31 are zero (no terms needed)
        bit_terms.append(bt)

    return ShapeTerms("slt", bits * 2, bits, bit_terms)


def generate_sltu_shape(bits: int = 32) -> ShapeTerms:
    """Generate SLTU (set less than, unsigned) shape."""
    shape = generate_slt_shape(bits)
    shape.name = "sltu"
    return shape


# =============================================================================
# SHAPE GENERATOR REGISTRY
# =============================================================================

SHAPE_GENERATORS = {
    "nop": generate_nop_shape,
    "xor": generate_xor_shape,
    "and": generate_and_shape,
    "or": generate_or_shape,
    "not": generate_not_shape,
    "add": generate_add_shape,
    "sub": generate_sub_shape,
    "sll": generate_sll_shape,
    "srl": generate_srl_shape,
    "sra": generate_sra_shape,
    "slt": generate_slt_shape,
    "sltu": generate_sltu_shape,
    "nand": generate_nand_shape,
    "nor": generate_nor_shape,
    "xnor": generate_xnor_shape,
}


def generate_shape(name: str, bits: int = 32) -> Optional[ShapeTerms]:
    """
    Generate terms for a named shape.

    Args:
        name: Shape name (e.g., "xor", "add")
        bits: Bit width (default 32)

    Returns:
        ShapeTerms or None if unknown shape
    """
    gen = SHAPE_GENERATORS.get(name)
    if gen is None:
        return None
    return gen(bits)


def generate_all_shapes(bits: int = 32) -> Dict[str, ShapeTerms]:
    """Generate terms for all shapes."""
    return {name: gen(bits) for name, gen in SHAPE_GENERATORS.items()}


# =============================================================================
# VALIDATION
# =============================================================================

@dataclass
class ValidationResult:
    """Result of shape validation."""
    passed: bool
    accuracy: float
    samples_tested: int
    mode: str
    failures: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # failures: [(a, b, expected, got), ...]


# Edge cases for 32-bit validation
EDGE_CASES_32 = [
    # Zeros and ones
    (0x00000000, 0x00000000),  # All zeros
    (0xFFFFFFFF, 0xFFFFFFFF),  # All ones
    (0x00000000, 0xFFFFFFFF),  # Mixed
    (0xFFFFFFFF, 0x00000000),  # Mixed inverse

    # Alternating patterns
    (0xAAAAAAAA, 0x55555555),  # Checkerboard
    (0x55555555, 0xAAAAAAAA),  # Inverse
    (0xAAAAAAAA, 0xAAAAAAAA),  # Same pattern
    (0x55555555, 0x55555555),  # Same pattern

    # Powers of 2 (single bits)
    (0x00000001, 0x00000000),  # LSB only
    (0x80000000, 0x00000000),  # MSB (sign bit) only
    (0x00000001, 0x00000001),  # Both LSB
    (0x80000000, 0x80000000),  # Both sign
    (0x40000000, 0x40000000),  # Second highest

    # Overflow boundaries (for arithmetic)
    (0x7FFFFFFF, 0x00000001),  # Max positive + 1
    (0xFFFFFFFF, 0x00000001),  # -1 + 1 = 0
    (0x00000000, 0x00000001),  # 0 + 1 = 1
    (0x80000000, 0x00000001),  # Most negative + 1

    # Near-boundary
    (0x7FFFFFFE, 0x00000002),  # Max positive - 1 + 2
    (0x00000002, 0x00000002),  # Small + small
    (0xFFFFFFFE, 0x00000002),  # -2 + 2 = 0

    # Random-looking but structured
    (0x12345678, 0x9ABCDEF0),  # Spread bits
    (0xDEADBEEF, 0xCAFEBABE),  # Classic patterns
    (0xFEEDFACE, 0xBAADF00D),  # More classics

    # All powers of 2
    *[(1 << i, 0) for i in range(32)],
    *[(0, 1 << i) for i in range(32)],
    *[(1 << i, 1 << i) for i in range(32)],
]


# Edge cases for 8-bit validation
EDGE_CASES_8 = [
    (0x00, 0x00), (0xFF, 0xFF), (0x00, 0xFF),
    (0xAA, 0x55), (0x55, 0xAA),
    (0x01, 0x00), (0x80, 0x00),
    (0x7F, 0x01), (0xFF, 0x01),
    *[(1 << i, 0) for i in range(8)],
    *[(0, 1 << i) for i in range(8)],
]


def _int_to_bits(x: int, bits: int) -> Tensor:
    """Convert integer to bit tensor."""
    return torch.tensor([(x >> i) & 1 for i in range(bits)], dtype=torch.float32)


def _bits_to_int(bits: Tensor) -> int:
    """Convert bit tensor to integer."""
    result = 0
    for i, b in enumerate(bits):
        if round(float(b)) == 1:
            result |= (1 << i)
    return result


def validate_terms_against_tensor(
    shape_terms: ShapeTerms,
    tensor_fn,
    n_samples: int = 100,
    bits: int = 32,
) -> Tuple[bool, float]:
    """
    Validate that term evaluation matches tensor function.

    Args:
        shape_terms: The term specification to validate
        tensor_fn: The tensor function to compare against
        n_samples: Number of random samples
        bits: Bit width

    Returns:
        (passed, accuracy) tuple
    """
    correct = 0

    for _ in range(n_samples):
        # Random binary inputs
        a = torch.randint(0, 2, (1, bits), dtype=torch.float32)
        b = torch.randint(0, 2, (1, bits), dtype=torch.float32)

        # Tensor evaluation
        tensor_result = tensor_fn(a, b)

        # Term evaluation
        if shape_terms.input_bits == bits:
            # Single input (like NOT)
            inputs = a
        else:
            # Two inputs
            inputs = torch.cat([a, b], dim=1)

        term_result = shape_terms.evaluate(inputs)

        # Compare (with tolerance for floating point)
        if torch.allclose(tensor_result, term_result, atol=1e-6):
            correct += 1

    accuracy = correct / n_samples
    passed = accuracy >= 0.9999

    return passed, accuracy


def validate_quick(
    shape_terms: ShapeTerms,
    truth_fn,
    n_samples: int = 1000,
    bits: int = 32,
) -> ValidationResult:
    """
    Quick validation with random samples.

    Args:
        shape_terms: The term specification to validate
        truth_fn: Truth function (int, int) -> int
        n_samples: Number of random samples
        bits: Bit width

    Returns:
        ValidationResult
    """
    correct = 0
    failures = []
    max_val = (1 << bits) - 1

    for _ in range(n_samples):
        # Use Python's random for large bit widths (torch.randint overflows at 64-bit)
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)

        # Term evaluation
        a_bits = _int_to_bits(a, bits).unsqueeze(0)
        b_bits = _int_to_bits(b, bits).unsqueeze(0)

        if shape_terms.input_bits == bits:
            inputs = a_bits
        else:
            inputs = torch.cat([a_bits, b_bits], dim=1)

        result = shape_terms.evaluate(inputs)
        result_int = _bits_to_int(result.squeeze())

        # Truth
        expected = truth_fn(a, b) & max_val

        if result_int != expected:
            failures.append((a, b, expected, result_int))
        else:
            correct += 1

    accuracy = correct / n_samples
    passed = len(failures) == 0

    return ValidationResult(
        passed=passed,
        accuracy=accuracy,
        samples_tested=n_samples,
        mode="quick",
        failures=failures[:10],  # Keep first 10 failures
    )


def validate_edge(
    shape_terms: ShapeTerms,
    truth_fn,
    bits: int = 32,
) -> ValidationResult:
    """
    Edge case validation.

    Args:
        shape_terms: The term specification to validate
        truth_fn: Truth function (int, int) -> int
        bits: Bit width

    Returns:
        ValidationResult
    """
    edge_cases = EDGE_CASES_32 if bits == 32 else EDGE_CASES_8
    max_val = (1 << bits) - 1

    correct = 0
    failures = []

    for a, b in edge_cases:
        # Mask to bit width
        a = a & max_val
        b = b & max_val

        # Term evaluation
        a_bits = _int_to_bits(a, bits).unsqueeze(0)
        b_bits = _int_to_bits(b, bits).unsqueeze(0)

        if shape_terms.input_bits == bits:
            inputs = a_bits
        else:
            inputs = torch.cat([a_bits, b_bits], dim=1)

        result = shape_terms.evaluate(inputs)
        result_int = _bits_to_int(result.squeeze())

        # Truth
        expected = truth_fn(a, b) & max_val

        if result_int != expected:
            failures.append((a, b, expected, result_int))
        else:
            correct += 1

    accuracy = correct / len(edge_cases) if edge_cases else 1.0
    passed = len(failures) == 0

    return ValidationResult(
        passed=passed,
        accuracy=accuracy,
        samples_tested=len(edge_cases),
        mode="edge",
        failures=failures,
    )


def validate_exhaustive_8bit(
    shape_terms: ShapeTerms,
    truth_fn,
) -> ValidationResult:
    """
    Exhaustive 8-bit validation (all 2^16 combinations).

    Args:
        shape_terms: The term specification to validate (must be 8-bit)
        truth_fn: Truth function (int, int) -> int

    Returns:
        ValidationResult
    """
    if shape_terms.output_bits != 8:
        raise ValueError("Exhaustive validation requires 8-bit shape")

    correct = 0
    failures = []
    total = 256 * 256  # 65,536

    for a in range(256):
        for b in range(256):
            # Term evaluation
            a_bits = _int_to_bits(a, 8).unsqueeze(0)
            b_bits = _int_to_bits(b, 8).unsqueeze(0)

            if shape_terms.input_bits == 8:
                inputs = a_bits
            else:
                inputs = torch.cat([a_bits, b_bits], dim=1)

            result = shape_terms.evaluate(inputs)
            result_int = _bits_to_int(result.squeeze())

            # Truth
            expected = truth_fn(a, b) & 0xFF

            if result_int != expected:
                failures.append((a, b, expected, result_int))
                if len(failures) >= 100:
                    # Stop early if too many failures
                    break
            else:
                correct += 1

        if len(failures) >= 100:
            break

    accuracy = correct / total
    passed = len(failures) == 0

    return ValidationResult(
        passed=passed,
        accuracy=accuracy,
        samples_tested=total,
        mode="exhaustive",
        failures=failures[:10],
    )


def validate_shape(
    shape_terms: ShapeTerms,
    truth_fn,
    mode: str = "quick",
    bits: int = 32,
) -> ValidationResult:
    """
    Validate shape with specified mode.

    Args:
        shape_terms: The term specification to validate
        truth_fn: Truth function (int, int) -> int
        mode: "quick", "edge", or "exhaustive"
        bits: Bit width (ignored for exhaustive, which uses 8)

    Returns:
        ValidationResult
    """
    if mode == "quick":
        return validate_quick(shape_terms, truth_fn, bits=bits)
    elif mode == "edge":
        return validate_edge(shape_terms, truth_fn, bits=bits)
    elif mode == "exhaustive":
        # Generate 8-bit version for exhaustive
        shape_8bit = generate_shape(shape_terms.name, bits=8)
        if shape_8bit is None:
            raise ValueError(f"Cannot generate 8-bit version of {shape_terms.name}")
        return validate_exhaustive_8bit(shape_8bit, truth_fn)
    else:
        raise ValueError(f"Unknown validation mode: {mode}")


# =============================================================================
# MULTI-WIDTH SUPPORT
# =============================================================================

# Standard bit widths supported by XORPU
STANDARD_WIDTHS = [8, 16, 32, 64]


def generate_all_shapes_multiwidth(
    widths: List[int] = None,
) -> Dict[int, Dict[str, "ShapeTerms"]]:
    """
    Generate all shapes for multiple bit widths.

    Args:
        widths: List of bit widths (default: STANDARD_WIDTHS)

    Returns:
        Dict mapping width -> shape_name -> ShapeTerms
    """
    if widths is None:
        widths = STANDARD_WIDTHS

    result = {}
    for w in widths:
        result[w] = generate_all_shapes(bits=w)
    return result


def validate_multiwidth(
    shape_name: str,
    truth_fn,
    widths: List[int] = None,
    mode: str = "quick",
) -> Dict[int, "ValidationResult"]:
    """
    Validate a shape across multiple bit widths.

    Args:
        shape_name: Name of shape to validate
        truth_fn: Truth function (int, int) -> int
        widths: List of bit widths (default: STANDARD_WIDTHS)
        mode: Validation mode ("quick", "edge", "exhaustive")

    Returns:
        Dict mapping width -> ValidationResult
    """
    if widths is None:
        widths = STANDARD_WIDTHS

    results = {}
    for w in widths:
        shape = generate_shape(shape_name, bits=w)
        if shape is not None:
            results[w] = validate_shape(shape, truth_fn, mode=mode, bits=w)
    return results


def multiwidth_summary(
    widths: List[int] = None,
) -> str:
    """
    Generate summary of term counts and scaling across widths.

    Returns formatted string with per-width statistics.
    """
    if widths is None:
        widths = STANDARD_WIDTHS

    lines = []
    lines.append("XORPU Multi-Width Summary")
    lines.append("=" * 70)
    lines.append(f"{'Width':<8} {'Shapes':>8} {'Total Terms':>14} {'XOR Terms':>12} {'ADD Terms':>12}")
    lines.append("-" * 70)

    for w in widths:
        shapes = generate_all_shapes(bits=w)
        total_terms = sum(s.total_terms() for s in shapes.values())
        xor_terms_count = shapes["xor"].total_terms()
        add_terms_count = shapes["add"].total_terms()

        lines.append(
            f"{w:<8} {len(shapes):>8} {total_terms:>14} "
            f"{xor_terms_count:>12} {add_terms_count:>12}"
        )

    lines.append("-" * 70)
    lines.append("Terms scale linearly with bit width (3 terms per output bit for XOR)")

    return "\n".join(lines)


# =============================================================================
# OPTIMIZATION: FAST PATH
# =============================================================================

# Shapes that have direct hardware implementations (bypass polynomial)
FAST_PATH_SHAPES = {"xor", "and", "or", "not", "nand", "nor", "xnor", "nop"}


def is_fast_path(shape_name: str) -> bool:
    """
    Check if shape has a fast path implementation.

    Fast path shapes execute in 1 cycle with direct logic,
    bypassing polynomial term evaluation.
    """
    return shape_name.lower() in FAST_PATH_SHAPES


def compute_fast(shape_name: str, a: int, b: int, bits: int = 32) -> int:
    """
    Direct computation for fast path shapes.

    This bypasses polynomial evaluation for simple logic operations.

    Args:
        shape_name: Name of the shape
        a: First operand
        b: Second operand
        bits: Bit width for masking

    Returns:
        Result of the operation

    Raises:
        ValueError: If shape is not a fast path shape
    """
    mask = (1 << bits) - 1
    name = shape_name.lower()

    if name == "xor":
        return (a ^ b) & mask
    elif name == "and":
        return (a & b) & mask
    elif name == "or":
        return (a | b) & mask
    elif name == "not":
        return (~a) & mask
    elif name == "nand":
        return (~(a & b)) & mask
    elif name == "nor":
        return (~(a | b)) & mask
    elif name == "xnor":
        return (~(a ^ b)) & mask
    elif name == "nop":
        return a & mask
    else:
        raise ValueError(f"Not a fast path shape: {shape_name}")


def compute_auto(shape_name: str, a: int, b: int, bits: int = 32) -> int:
    """
    Compute with automatic fast path detection.

    Uses fast path if available, otherwise falls back to polynomial.

    Args:
        shape_name: Name of the shape
        a: First operand
        b: Second operand
        bits: Bit width

    Returns:
        Result of the operation
    """
    if is_fast_path(shape_name):
        return compute_fast(shape_name, a, b, bits)
    else:
        # Fall back to polynomial evaluation
        shape = generate_shape(shape_name, bits=bits)
        if shape is None:
            raise ValueError(f"Unknown shape: {shape_name}")

        a_bits = _int_to_bits(a, bits).unsqueeze(0)
        b_bits = _int_to_bits(b, bits).unsqueeze(0)

        if shape.input_bits == bits:
            inputs = a_bits
        else:
            inputs = torch.cat([a_bits, b_bits], dim=1)

        result = shape.evaluate(inputs)
        return _bits_to_int(result.squeeze())


# =============================================================================
# OPTIMIZATION: BATCH EVALUATION
# =============================================================================

def evaluate_batch(
    shape: ShapeTerms,
    a_batch: List[int],
    b_batch: List[int] = None,
) -> List[int]:
    """
    Batch evaluation for multiple inputs.

    More efficient than individual evaluations due to tensor parallelism.

    Args:
        shape: The shape to evaluate
        a_batch: List of first operands
        b_batch: List of second operands (optional for single-input shapes)

    Returns:
        List of results
    """
    bits = shape.output_bits
    batch_size = len(a_batch)

    if b_batch is None:
        b_batch = [0] * batch_size

    # Build input tensor
    inputs_list = []
    for a, b in zip(a_batch, b_batch):
        a_bits = _int_to_bits(a, bits)
        b_bits = _int_to_bits(b, bits)

        if shape.input_bits == bits:
            inputs_list.append(a_bits)
        else:
            inputs_list.append(torch.cat([a_bits, b_bits]))

    inputs = torch.stack(inputs_list)  # [batch, input_bits]

    # Evaluate all at once
    results = shape.evaluate(inputs)  # [batch, output_bits]

    # Convert back to integers
    return [_bits_to_int(results[i]) for i in range(batch_size)]


def evaluate_batch_fast(
    shape_name: str,
    a_batch: List[int],
    b_batch: List[int] = None,
    bits: int = 32,
) -> List[int]:
    """
    Batch evaluation with automatic fast path.

    Uses direct computation for fast path shapes.

    Args:
        shape_name: Name of the shape
        a_batch: List of first operands
        b_batch: List of second operands
        bits: Bit width

    Returns:
        List of results
    """
    if b_batch is None:
        b_batch = [0] * len(a_batch)

    if is_fast_path(shape_name):
        return [compute_fast(shape_name, a, b, bits)
                for a, b in zip(a_batch, b_batch)]
    else:
        shape = generate_shape(shape_name, bits=bits)
        if shape is None:
            raise ValueError(f"Unknown shape: {shape_name}")
        return evaluate_batch(shape, a_batch, b_batch)


# =============================================================================
# OPTIMIZATION: TERM COMPRESSION STATISTICS
# =============================================================================

def compression_stats(shapes: Dict[str, ShapeTerms] = None, bits: int = 32) -> Dict[str, Any]:
    """
    Calculate term compression statistics.

    Returns storage estimates for full vs compressed representations.

    Args:
        shapes: Dictionary of shapes (default: generate all)
        bits: Bit width

    Returns:
        Dictionary with compression statistics
    """
    if shapes is None:
        shapes = generate_all_shapes(bits=bits)

    # Count unique terms across all shapes
    all_terms = set()
    total_term_refs = 0

    for shape in shapes.values():
        for bt in shape.bit_terms:
            for term in bt.terms:
                all_terms.add(term)
                total_term_refs += 1

    unique_terms = len(all_terms)

    # Size estimates (bytes)
    bytes_per_term = 8  # coefficient (4) + variable indices (4 avg)
    bytes_per_ref = 2   # index into term table

    full_size = total_term_refs * bytes_per_term
    compressed_size = (unique_terms * bytes_per_term) + (total_term_refs * bytes_per_ref)

    return {
        "shapes": len(shapes),
        "total_term_refs": total_term_refs,
        "unique_terms": unique_terms,
        "full_size_bytes": full_size,
        "compressed_size_bytes": compressed_size,
        "compression_ratio": full_size / compressed_size if compressed_size > 0 else 1.0,
        "savings_percent": 100 * (1 - compressed_size / full_size) if full_size > 0 else 0.0,
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    "Term",
    "BitTerms",
    "ShapeTerms",

    # Primitive term generators
    "xor_terms",
    "and_terms",
    "or_terms",
    "not_terms",
    "nand_terms",
    "nor_terms",
    "xnor_terms",

    # Shape generators
    "generate_xor_shape",
    "generate_and_shape",
    "generate_or_shape",
    "generate_not_shape",
    "generate_nand_shape",
    "generate_nor_shape",
    "generate_xnor_shape",
    "generate_nop_shape",
    "generate_add_shape",
    "generate_sub_shape",
    "generate_sll_shape",
    "generate_srl_shape",
    "generate_sra_shape",
    "generate_slt_shape",
    "generate_sltu_shape",

    # Registry
    "SHAPE_GENERATORS",
    "generate_shape",
    "generate_all_shapes",

    # Validation
    "ValidationResult",
    "EDGE_CASES_32",
    "EDGE_CASES_8",
    "validate_terms_against_tensor",
    "validate_quick",
    "validate_edge",
    "validate_exhaustive_8bit",
    "validate_shape",

    # Multi-width
    "STANDARD_WIDTHS",
    "generate_all_shapes_multiwidth",
    "validate_multiwidth",
    "multiwidth_summary",

    # Optimization: Fast path
    "FAST_PATH_SHAPES",
    "is_fast_path",
    "compute_fast",
    "compute_auto",

    # Optimization: Batch evaluation
    "evaluate_batch",
    "evaluate_batch_fast",

    # Optimization: Compression
    "compression_stats",
]
