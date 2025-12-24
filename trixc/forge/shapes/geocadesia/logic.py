"""
Geocadesia: Logic Kingdom

Boolean operations — the foundation of discrete computation.
All shapes here are frozen and exact at binary inputs.

"The primordial shapes. All digital computation reduces to them."
"""

from __future__ import annotations
from .catalog import shape, Shape, Kingdom, Arity, ShapeType, FrozenStatus


# =============================================================================
# XOR - The shape that started it all
# =============================================================================

@shape(
    name="xor",
    kingdom=Kingdom.LOGIC,
    formula="a ⊕ b = a + b - 2ab",
    definition="Exclusive OR. Returns 1 when exactly one input is 1.",
    arity=Arity.BINARY,
    see_also=["and", "or", "xnor"],
    examples=[
        ((0, 0), 0),
        ((0, 1), 1),
        ((1, 0), 1),
        ((1, 1), 0),
    ],
)
def xor(a: float, b: float) -> float:
    """Differentiable XOR for continuous inputs in [0, 1]."""
    return a + b - 2.0 * a * b


# Class-style interface
class XOR:
    """
    XOR Gate: Exclusive OR

    The shape of difference. Two identical inputs produce 0.
    Two different inputs produce 1.

    Usage:
        gate = XOR()
        result = gate(0.5, 0.7)
    """

    def __call__(self, a: float, b: float) -> float:
        return xor(a, b)

    @staticmethod
    def discrete(a: int, b: int) -> int:
        return a ^ b

    def __repr__(self):
        return "<XOR: a ⊕ b = a + b - 2ab>"


# =============================================================================
# AND - Both must agree
# =============================================================================

@shape(
    name="and",
    kingdom=Kingdom.LOGIC,
    formula="a ∧ b = a · b",
    definition="Logical AND. Returns 1 only when both inputs are 1.",
    arity=Arity.BINARY,
    see_also=["or", "nand", "xor"],
    examples=[
        ((0, 0), 0),
        ((0, 1), 0),
        ((1, 0), 0),
        ((1, 1), 1),
    ],
)
def and_gate(a: float, b: float) -> float:
    """Differentiable AND for continuous inputs in [0, 1]."""
    return a * b


class AND:
    """
    AND Gate: Logical Conjunction

    Both inputs must be high for output to be high.
    The gatekeeper of agreement.
    """

    def __call__(self, a: float, b: float) -> float:
        return and_gate(a, b)

    @staticmethod
    def discrete(a: int, b: int) -> int:
        return a & b

    def __repr__(self):
        return "<AND: a ∧ b = a · b>"


# =============================================================================
# OR - Either suffices
# =============================================================================

@shape(
    name="or",
    kingdom=Kingdom.LOGIC,
    formula="a ∨ b = a + b - ab",
    definition="Logical OR. Returns 1 when at least one input is 1.",
    arity=Arity.BINARY,
    see_also=["and", "nor", "xor"],
    examples=[
        ((0, 0), 0),
        ((0, 1), 1),
        ((1, 0), 1),
        ((1, 1), 1),
    ],
)
def or_gate(a: float, b: float) -> float:
    """Differentiable OR (probabilistic) for continuous inputs in [0, 1]."""
    return a + b - a * b


class OR:
    """
    OR Gate: Logical Disjunction

    Either input being high makes output high.
    The permissive gate.
    """

    def __call__(self, a: float, b: float) -> float:
        return or_gate(a, b)

    @staticmethod
    def discrete(a: int, b: int) -> int:
        return a | b

    def __repr__(self):
        return "<OR: a ∨ b = a + b - ab>"


# =============================================================================
# NOT - The inverter
# =============================================================================

@shape(
    name="not",
    kingdom=Kingdom.LOGIC,
    formula="¬a = 1 - a",
    definition="Logical NOT. Inverts the input.",
    arity=Arity.UNARY,
    see_also=["nand", "nor"],
    examples=[
        (0, 1),
        (1, 0),
        (0.5, 0.5),
    ],
)
def not_gate(a: float) -> float:
    """Differentiable NOT for continuous inputs in [0, 1]."""
    return 1.0 - a


class NOT:
    """
    NOT Gate: Logical Negation

    The simplest shape: inversion.
    What was high becomes low.
    """

    def __call__(self, a: float) -> float:
        return not_gate(a)

    @staticmethod
    def discrete(a: int) -> int:
        return 1 - a

    def __repr__(self):
        return "<NOT: ¬a = 1 - a>"


# =============================================================================
# NAND - The universal gate
# =============================================================================

@shape(
    name="nand",
    kingdom=Kingdom.LOGIC,
    formula="¬(a ∧ b) = 1 - ab",
    definition="NOT AND. Universal gate: any logic can be built from NAND alone.",
    arity=Arity.BINARY,
    built_from=["not", "and"],
    see_also=["and", "nor"],
    examples=[
        ((0, 0), 1),
        ((0, 1), 1),
        ((1, 0), 1),
        ((1, 1), 0),
    ],
)
def nand(a: float, b: float) -> float:
    """Differentiable NAND for continuous inputs in [0, 1]."""
    return 1.0 - a * b


class NAND:
    """
    NAND Gate: NOT AND

    The universal gate. Any Boolean function can be built
    from NAND gates alone. The Apollo Guidance Computer
    used only NOR gates — NAND's dual.
    """

    def __call__(self, a: float, b: float) -> float:
        return nand(a, b)

    @staticmethod
    def discrete(a: int, b: int) -> int:
        return 1 - (a & b)

    def __repr__(self):
        return "<NAND: ¬(a ∧ b) = 1 - ab>"


# =============================================================================
# NOR - The other universal gate
# =============================================================================

@shape(
    name="nor",
    kingdom=Kingdom.LOGIC,
    formula="¬(a ∨ b) = (1-a)(1-b)",
    definition="NOT OR. Universal gate, used in the Apollo Guidance Computer.",
    arity=Arity.BINARY,
    built_from=["not", "or"],
    see_also=["or", "nand"],
    examples=[
        ((0, 0), 1),
        ((0, 1), 0),
        ((1, 0), 0),
        ((1, 1), 0),
    ],
)
def nor(a: float, b: float) -> float:
    """Differentiable NOR for continuous inputs in [0, 1]."""
    return (1.0 - a) * (1.0 - b)


class NOR:
    """
    NOR Gate: NOT OR

    The other universal gate. The Apollo Guidance Computer
    was built entirely from ~5,600 NOR gates.
    """

    def __call__(self, a: float, b: float) -> float:
        return nor(a, b)

    @staticmethod
    def discrete(a: int, b: int) -> int:
        return 1 - (a | b)

    def __repr__(self):
        return "<NOR: ¬(a ∨ b) = (1-a)(1-b)>"


# =============================================================================
# XNOR - Equivalence
# =============================================================================

@shape(
    name="xnor",
    kingdom=Kingdom.LOGIC,
    formula="¬(a ⊕ b) = 1 - a - b + 2ab",
    definition="NOT XOR. Equivalence gate: returns 1 when inputs are the same.",
    arity=Arity.BINARY,
    built_from=["not", "xor"],
    see_also=["xor", "and"],
    examples=[
        ((0, 0), 1),
        ((0, 1), 0),
        ((1, 0), 0),
        ((1, 1), 1),
    ],
)
def xnor(a: float, b: float) -> float:
    """Differentiable XNOR (equivalence) for continuous inputs in [0, 1]."""
    return 1.0 - a - b + 2.0 * a * b


class XNOR:
    """
    XNOR Gate: Equivalence

    Returns 1 when both inputs are the same.
    The shape of agreement. Core of XNOR-Net binary neural networks.
    """

    def __call__(self, a: float, b: float) -> float:
        return xnor(a, b)

    @staticmethod
    def discrete(a: int, b: int) -> int:
        return 1 - (a ^ b)

    def __repr__(self):
        return "<XNOR: ¬(a ⊕ b) = 1 - a - b + 2ab>"
