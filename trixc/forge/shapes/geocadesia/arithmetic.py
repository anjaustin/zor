"""
Geocadesia: Arithmetic Kingdom

Numeric computation without learning.
All shapes here are frozen.

"The building blocks of both classical and neural computation."
"""

from __future__ import annotations
from typing import Tuple
from .catalog import shape, Shape, Kingdom, Arity, ShapeType, FrozenStatus


# =============================================================================
# Elemental Arithmetic
# =============================================================================

@shape(
    name="add",
    kingdom=Kingdom.ARITHMETIC,
    formula="a + b",
    definition="Binary addition.",
    arity=Arity.BINARY,
)
def add(a: float, b: float) -> float:
    return a + b

class Add:
    def __call__(self, a: float, b: float) -> float:
        return a + b
    def __repr__(self):
        return "<Add: a + b>"


@shape(
    name="sub",
    kingdom=Kingdom.ARITHMETIC,
    formula="a - b",
    definition="Binary subtraction.",
    arity=Arity.BINARY,
)
def sub(a: float, b: float) -> float:
    return a - b

class Sub:
    def __call__(self, a: float, b: float) -> float:
        return a - b
    def __repr__(self):
        return "<Sub: a - b>"


@shape(
    name="mul",
    kingdom=Kingdom.ARITHMETIC,
    formula="a × b",
    definition="Binary multiplication.",
    arity=Arity.BINARY,
)
def mul(a: float, b: float) -> float:
    return a * b

class Mul:
    def __call__(self, a: float, b: float) -> float:
        return a * b
    def __repr__(self):
        return "<Mul: a × b>"


@shape(
    name="neg",
    kingdom=Kingdom.ARITHMETIC,
    formula="-a",
    definition="Unary negation.",
    arity=Arity.UNARY,
)
def neg(a: float) -> float:
    return -a

class Neg:
    def __call__(self, a: float) -> float:
        return -a
    def __repr__(self):
        return "<Neg: -a>"


# =============================================================================
# Compound Arithmetic
# =============================================================================

@shape(
    name="half_adder",
    kingdom=Kingdom.ARITHMETIC,
    formula="sum = a ⊕ b, carry = a ∧ b",
    definition="Adds two bits, produces sum and carry.",
    arity=Arity.BINARY,
    shape_type=ShapeType.COMPOUND,
    built_from=["xor", "and"],
    see_also=["full_adder"],
    examples=[
        ((0, 0), (0, 0)),
        ((0, 1), (1, 0)),
        ((1, 0), (1, 0)),
        ((1, 1), (0, 1)),
    ],
)
def half_adder(a: float, b: float) -> Tuple[float, float]:
    """
    Half adder: adds two bits.

    Returns (sum, carry).
    """
    sum_out = a + b - 2.0 * a * b  # XOR
    carry = a * b                   # AND
    return sum_out, carry


class HalfAdder:
    """
    Half Adder: XOR + AND

    The simplest arithmetic compound. Adds two bits,
    produces sum and carry. Called "half" because it
    doesn't accept a carry-in.
    """

    def __call__(self, a: float, b: float) -> Tuple[float, float]:
        return half_adder(a, b)

    @staticmethod
    def discrete(a: int, b: int) -> Tuple[int, int]:
        return (a ^ b, a & b)

    def __repr__(self):
        return "<HalfAdder: sum=XOR, carry=AND>"


@shape(
    name="full_adder",
    kingdom=Kingdom.ARITHMETIC,
    formula="sum = a ⊕ b ⊕ cin, cout = (a∧b) ∨ (cin∧(a⊕b))",
    definition="Adds two bits plus carry-in, produces sum and carry-out.",
    arity=Arity.TERNARY,
    shape_type=ShapeType.COMPOUND,
    built_from=["xor", "and", "or"],
    see_also=["half_adder"],
    examples=[
        ((0, 0, 0), (0, 0)),
        ((1, 1, 0), (0, 1)),
        ((1, 1, 1), (1, 1)),
    ],
)
def full_adder(a: float, b: float, c_in: float) -> Tuple[float, float]:
    """
    Full adder: adds two bits plus carry-in.

    Returns (sum, carry_out).
    """
    # First half adder: a XOR b
    sum1 = a + b - 2.0 * a * b
    carry1 = a * b

    # Second half adder: sum1 XOR c_in
    sum_out = sum1 + c_in - 2.0 * sum1 * c_in
    carry2 = sum1 * c_in

    # OR the carries
    c_out = carry1 + carry2 - carry1 * carry2

    return sum_out, c_out


class FullAdder:
    """
    Full Adder: Complete single-bit addition

    The minimal viable organism of arithmetic.
    Handles two operands and a carry. Chain them
    together for arbitrary-precision addition.
    """

    def __call__(self, a: float, b: float, c_in: float) -> Tuple[float, float]:
        return full_adder(a, b, c_in)

    @staticmethod
    def discrete(a: int, b: int, c_in: int) -> Tuple[int, int]:
        sum_out = a ^ b ^ c_in
        c_out = (a & b) | (c_in & (a ^ b))
        return sum_out, c_out

    def __repr__(self):
        return "<FullAdder: 2×HalfAdder + OR>"


# =============================================================================
# Binary Arithmetic (FrozenDB Support)
# =============================================================================

@shape(
    name="popcount",
    kingdom=Kingdom.ARITHMETIC,
    formula="popcount(x) = Σ bits(x)",
    definition="Population count: number of 1-bits in a binary value.",
    arity=Arity.UNARY,
    see_also=["hamming", "xor"],
    examples=[
        (0b0000, 0),
        (0b0001, 1),
        (0b0101, 2),
        (0b1111, 4),
        (0b11111111, 8),
    ],
)
def popcount(x: int) -> int:
    """
    Population count: number of 1-bits.

    Essential for Hamming distance computation.
    Hardware-accelerated on modern CPUs (POPCNT instruction).
    """
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


class Popcount:
    """
    Population Count: Count the 1-bits

    The foundation of Hamming distance. Count how many
    bits are set to 1 in a binary value.

    Used in: FrozenDB, XNOR-Net, error detection
    """

    def __call__(self, x: int) -> int:
        return popcount(x)

    @staticmethod
    def fast(x: int) -> int:
        """Brian Kernighan's algorithm - O(set bits)."""
        count = 0
        while x:
            x &= x - 1  # Clear lowest set bit
            count += 1
        return count

    def __repr__(self):
        return "<Popcount: Σ bits(x)>"


@shape(
    name="hamming",
    kingdom=Kingdom.ARITHMETIC,
    formula="hamming(a, b) = popcount(a ⊕ b)",
    definition="Hamming distance: number of bit positions where a and b differ.",
    arity=Arity.BINARY,
    shape_type=ShapeType.COMPOUND,
    built_from=["xor", "popcount"],
    see_also=["xor", "popcount", "xnor"],
    examples=[
        ((0b0000, 0b0000), 0),
        ((0b0000, 0b0001), 1),
        ((0b1010, 0b0101), 4),
        ((0b1111, 0b0000), 4),
    ],
)
def hamming(a: int, b: int) -> int:
    """
    Hamming distance: number of differing bit positions.

    The core metric of FrozenDB. Measures how different
    two binary signatures are.

    hamming(a, b) = popcount(XOR(a, b))
    """
    return popcount(a ^ b)


class Hamming:
    """
    Hamming Distance: XOR + Popcount

    The number of positions where two binary values differ.
    This is THE metric for FrozenDB vector search.

    At 512 bits: hamming = 0 means identical
                 hamming = 512 means completely opposite
                 hamming = 256 means random/orthogonal
    """

    def __call__(self, a: int, b: int) -> int:
        return hamming(a, b)

    @staticmethod
    def normalized(a: int, b: int, bits: int = 512) -> float:
        """Normalized Hamming distance in [0, 1]."""
        return hamming(a, b) / bits

    @staticmethod
    def similarity(a: int, b: int, bits: int = 512) -> float:
        """Hamming similarity in [0, 1]. Higher = more similar."""
        return 1.0 - (hamming(a, b) / bits)

    def __repr__(self):
        return "<Hamming: popcount(a ⊕ b)>"
