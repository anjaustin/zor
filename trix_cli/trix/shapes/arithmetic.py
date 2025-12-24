"""
Arithmetic shapes - frozen numeric operations.
"""
from ..foundry import register_builtin, Shape


def _add(a: int, b: int) -> int:
    """8-bit addition with wraparound."""
    return (a + b) & 0xFF


def _sub(a: int, b: int) -> int:
    """8-bit subtraction with wraparound."""
    return (a - b) & 0xFF


def _mul(a: int, b: int) -> int:
    """8-bit multiplication (low byte)."""
    return (a * b) & 0xFF


# Register arithmetic shapes
add = register_builtin("add", _add, "(a + b) mod 256")
sub = register_builtin("sub", _sub, "(a - b) mod 256")
mul = register_builtin("mul", _mul, "(a * b) mod 256")
