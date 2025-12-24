"""
Logic shapes - the frozen boolean operations.

Each shape is a polynomial that equals the hardware operation on binary inputs.
"""
from ..foundry import register_builtin, Shape


def _xor(a: int, b: int) -> int:
    """XOR: a + b - 2ab on binary, a ^ b on integers."""
    return a ^ b


def _and(a: int, b: int) -> int:
    """AND: ab on binary, a & b on integers."""
    return a & b


def _or(a: int, b: int) -> int:
    """OR: a + b - ab on binary, a | b on integers."""
    return a | b


def _not(a: int) -> int:
    """NOT: 1 - a on binary, ~a on integers."""
    return ~a & 0xFF


def _nand(a: int, b: int) -> int:
    """NAND: 1 - ab on binary."""
    return ~(a & b) & 0xFF


def _nor(a: int, b: int) -> int:
    """NOR: 1 - a - b + ab on binary."""
    return ~(a | b) & 0xFF


def _xnor(a: int, b: int) -> int:
    """XNOR: 1 - a - b + 2ab on binary."""
    return ~(a ^ b) & 0xFF


# Register all logic shapes
xor = register_builtin("xor", _xor, "a + b - 2ab")
and_ = register_builtin("and", _and, "ab")
or_ = register_builtin("or", _or, "a + b - ab")
not_ = register_builtin("not", _not, "1 - a", arity=1)
nand = register_builtin("nand", _nand, "1 - ab")
nor = register_builtin("nor", _nor, "1 - a - b + ab")
xnor = register_builtin("xnor", _xnor, "1 - a - b + 2ab")
