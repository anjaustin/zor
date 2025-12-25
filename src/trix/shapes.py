"""
trix.shapes - Function-call interface to frozen 6502 shapes.

Zero-ceremony access to the 16 frozen mathematical shapes that compute
exactly like a 6502 ALU.

Usage:
    >>> from trix.shapes import add, xor, inc
    >>> add(42, 13, carry=1)
    56
    >>> xor(0x55, 0xFF)
    170
    >>> add(127, 1, verbose=True)  # Shows the geometry

The shapes ARE the computation. No learning, just math.

This module is pure Python - no PyTorch, no NumPy, no dependencies.
The mathematical truths don't need a framework.
"""

from typing import Tuple, List, Optional


# =============================================================================
# SHAPE METADATA
# =============================================================================

SHAPE_INFO = {
    'RIPPLE_ADD': {
        'formula': 'sum = XOR(XOR(a, b), cin)\ncout = OR(AND(a, b), AND(XOR(a,b), cin))',
        'description': '8 chained full-adders propagating carry',
        'polynomial': 'XOR(a,b) = a + b - 2ab',
    },
    'RIPPLE_SUB': {
        'formula': 'diff = a - b - borrow',
        'description': '8 chained full-adders with inverted b',
        'polynomial': 'Uses NOT(b) = 1 - b, then adds',
    },
    'PARALLEL_AND': {
        'formula': 'result[i] = a[i] * b[i]',
        'description': 'Bitwise AND in parallel',
        'polynomial': 'AND(a,b) = ab',
    },
    'PARALLEL_OR': {
        'formula': 'result[i] = a[i] + b[i] - a[i]*b[i]',
        'description': 'Bitwise OR in parallel',
        'polynomial': 'OR(a,b) = a + b - ab',
    },
    'PARALLEL_XOR': {
        'formula': 'result[i] = a[i] + b[i] - 2*a[i]*b[i]',
        'description': 'Bitwise XOR in parallel',
        'polynomial': 'XOR(a,b) = a + b - 2ab',
    },
    'SHIFT_LEFT': {
        'formula': 'result = [0, a[0], a[1], ..., a[6]], carry = a[7]',
        'description': 'Arithmetic shift left, bit 7 to carry',
        'polynomial': 'Bit permutation (no arithmetic)',
    },
    'SHIFT_RIGHT': {
        'formula': 'result = [a[1], a[2], ..., a[7], 0], carry = a[0]',
        'description': 'Logical shift right, bit 0 to carry',
        'polynomial': 'Bit permutation (no arithmetic)',
    },
    'ROTATE_LEFT': {
        'formula': 'result = [cin, a[0], a[1], ..., a[6]], carry = a[7]',
        'description': 'Rotate left through carry',
        'polynomial': 'Bit permutation with carry insertion',
    },
    'ROTATE_RIGHT': {
        'formula': 'result = [a[1], a[2], ..., a[7], cin], carry = a[0]',
        'description': 'Rotate right through carry',
        'polynomial': 'Bit permutation with carry insertion',
    },
    'INCREMENT': {
        'formula': 'result = a + 1',
        'description': 'Add 1 using ripple carry',
        'polynomial': '8 full-adders with b=00000001',
    },
    'DECREMENT': {
        'formula': 'result = a - 1',
        'description': 'Subtract 1 using ripple borrow',
        'polynomial': '8 full-adders with b=11111111, cin=0',
    },
    'TRANSFER': {
        'formula': 'result = a',
        'description': 'Pass-through (identity)',
        'polynomial': 'f(a) = a',
    },
    'IDENTITY': {
        'formula': 'result = a',
        'description': 'No operation',
        'polynomial': 'f(a) = a',
    },
}


# =============================================================================
# VERBOSE OUTPUT
# =============================================================================

def _format_bits(value: int) -> str:
    """Format an 8-bit value as binary."""
    return f"{int(value) & 0xFF:08b}"


def _print_shape_info(
    shape_name: str,
    inputs: dict,
    result: int,
    carry_out: Optional[int] = None,
):
    """Print formatted geometry information."""
    info = SHAPE_INFO.get(shape_name, {})

    print(f"┌{'─' * 50}┐")
    print(f"│ Shape: {shape_name:<42}│")
    print(f"├{'─' * 50}┤")

    if 'description' in info:
        desc = info['description'][:46]
        print(f"│ {desc:<48} │")

    if 'polynomial' in info:
        poly = info['polynomial'][:46]
        print(f"│ {poly:<48} │")

    print(f"├{'─' * 50}┤")

    for name, value in inputs.items():
        if isinstance(value, (int, float)):
            v = int(value)
            line = f"│ {name}: {v} ({_format_bits(v)})"
            print(f"{line:<51}│")

    print(f"├{'─' * 50}┤")

    r = int(result)
    result_line = f"│ result: {r} ({_format_bits(r)})"
    print(f"{result_line:<51}│")

    if carry_out is not None:
        carry_line = f"│ carry_out: {int(carry_out)}"
        print(f"{carry_line:<51}│")

    print(f"└{'─' * 50}┘")


# =============================================================================
# ARITHMETIC SHAPES (carry chain)
# =============================================================================

def add(a: int, b: int, carry: int = 0, verbose: bool = False) -> int:
    """
    Add two 8-bit integers with carry.

    Uses RIPPLE_ADD shape: 8 chained full-adders.

    Args:
        a: First operand (0-255)
        b: Second operand (0-255)
        carry: Carry input (0 or 1)
        verbose: If True, print geometry info

    Returns:
        Result of a + b + carry (mod 256)

    Example:
        >>> add(42, 13)
        55
        >>> add(255, 1)  # Overflow wraps
        0
    """
    a = a & 0xFF
    b = b & 0xFF
    carry = carry & 1

    result = (a + b + carry) & 0xFF

    if verbose:
        _print_shape_info('RIPPLE_ADD', {'a': a, 'b': b, 'carry_in': carry}, result, (a + b + carry) >> 8)

    return result


def add_with_carry(a: int, b: int, carry: int = 0, verbose: bool = False) -> Tuple[int, int]:
    """
    Add two 8-bit integers, returning result and carry out.

    Args:
        a, b: Operands (0-255)
        carry: Carry input
        verbose: Print geometry

    Returns:
        (result, carry_out)
    """
    a = a & 0xFF
    b = b & 0xFF
    carry = carry & 1

    full = a + b + carry
    result = full & 0xFF
    carry_out = (full >> 8) & 1

    if verbose:
        _print_shape_info('RIPPLE_ADD', {'a': a, 'b': b, 'carry_in': carry}, result, carry_out)

    return result, carry_out


def sub(a: int, b: int, borrow: int = 0, verbose: bool = False) -> int:
    """
    Subtract: a - b - borrow.

    Uses RIPPLE_SUB shape.
    Note: borrow=0 means no borrow (carry=1 in 6502 terms).

    Args:
        a: Minuend (0-255)
        b: Subtrahend (0-255)
        borrow: Borrow input (0 or 1)
        verbose: Print geometry

    Returns:
        Result of a - b - borrow (mod 256)
    """
    a = a & 0xFF
    b = b & 0xFF
    borrow = borrow & 1

    result = (a - b - borrow) & 0xFF

    if verbose:
        _print_shape_info('RIPPLE_SUB', {'a': a, 'b': b, 'borrow': borrow}, result)

    return result


def inc(a: int, verbose: bool = False) -> int:
    """
    Increment: a + 1.

    Uses INCREMENT shape.

    Args:
        a: Value to increment (0-255)
        verbose: Print geometry

    Returns:
        a + 1 (mod 256)
    """
    a = a & 0xFF
    result = (a + 1) & 0xFF

    if verbose:
        _print_shape_info('INCREMENT', {'a': a}, result)

    return result


def dec(a: int, verbose: bool = False) -> int:
    """
    Decrement: a - 1.

    Uses DECREMENT shape.

    Args:
        a: Value to decrement (0-255)
        verbose: Print geometry

    Returns:
        a - 1 (mod 256)
    """
    a = a & 0xFF
    result = (a - 1) & 0xFF

    if verbose:
        _print_shape_info('DECREMENT', {'a': a}, result)

    return result


# =============================================================================
# LOGIC SHAPES (parallel bitwise)
# =============================================================================

def and_op(a: int, b: int, verbose: bool = False) -> int:
    """
    Bitwise AND: a & b.

    Uses PARALLEL_AND shape: AND(a,b) = ab

    Args:
        a, b: Operands (0-255)
        verbose: Print geometry

    Returns:
        a AND b
    """
    a = a & 0xFF
    b = b & 0xFF
    result = a & b

    if verbose:
        _print_shape_info('PARALLEL_AND', {'a': a, 'b': b}, result)

    return result


def or_op(a: int, b: int, verbose: bool = False) -> int:
    """
    Bitwise OR: a | b.

    Uses PARALLEL_OR shape: OR(a,b) = a + b - ab

    Args:
        a, b: Operands (0-255)
        verbose: Print geometry

    Returns:
        a OR b
    """
    a = a & 0xFF
    b = b & 0xFF
    result = a | b

    if verbose:
        _print_shape_info('PARALLEL_OR', {'a': a, 'b': b}, result)

    return result


def xor(a: int, b: int, verbose: bool = False) -> int:
    """
    Bitwise XOR: a ^ b.

    Uses PARALLEL_XOR shape: XOR(a,b) = a + b - 2ab

    Args:
        a, b: Operands (0-255)
        verbose: Print geometry

    Returns:
        a XOR b
    """
    a = a & 0xFF
    b = b & 0xFF
    result = a ^ b

    if verbose:
        _print_shape_info('PARALLEL_XOR', {'a': a, 'b': b}, result)

    return result


# =============================================================================
# SHIFT SHAPES
# =============================================================================

def asl(a: int, verbose: bool = False) -> Tuple[int, int]:
    """
    Arithmetic Shift Left.

    Uses SHIFT_LEFT shape. Bit 7 goes to carry, 0 enters bit 0.

    Args:
        a: Value to shift (0-255)
        verbose: Print geometry

    Returns:
        (result, carry_out)
    """
    a = a & 0xFF
    carry_out = (a >> 7) & 1
    result = (a << 1) & 0xFF

    if verbose:
        _print_shape_info('SHIFT_LEFT', {'a': a}, result, carry_out)

    return result, carry_out


def lsr(a: int, verbose: bool = False) -> Tuple[int, int]:
    """
    Logical Shift Right.

    Uses SHIFT_RIGHT shape. Bit 0 goes to carry, 0 enters bit 7.

    Args:
        a: Value to shift (0-255)
        verbose: Print geometry

    Returns:
        (result, carry_out)
    """
    a = a & 0xFF
    carry_out = a & 1
    result = (a >> 1) & 0xFF

    if verbose:
        _print_shape_info('SHIFT_RIGHT', {'a': a}, result, carry_out)

    return result, carry_out


def rol(a: int, carry: int = 0, verbose: bool = False) -> Tuple[int, int]:
    """
    Rotate Left through carry.

    Uses ROTATE_LEFT shape. Bit 7 goes to carry, old carry enters bit 0.

    Args:
        a: Value to rotate (0-255)
        carry: Carry input (0 or 1)
        verbose: Print geometry

    Returns:
        (result, carry_out)
    """
    a = a & 0xFF
    carry = carry & 1
    carry_out = (a >> 7) & 1
    result = ((a << 1) | carry) & 0xFF

    if verbose:
        _print_shape_info('ROTATE_LEFT', {'a': a, 'carry_in': carry}, result, carry_out)

    return result, carry_out


def ror(a: int, carry: int = 0, verbose: bool = False) -> Tuple[int, int]:
    """
    Rotate Right through carry.

    Uses ROTATE_RIGHT shape. Bit 0 goes to carry, old carry enters bit 7.

    Args:
        a: Value to rotate (0-255)
        carry: Carry input (0 or 1)
        verbose: Print geometry

    Returns:
        (result, carry_out)
    """
    a = a & 0xFF
    carry = carry & 1
    carry_out = a & 1
    result = ((a >> 1) | (carry << 7)) & 0xFF

    if verbose:
        _print_shape_info('ROTATE_RIGHT', {'a': a, 'carry_in': carry}, result, carry_out)

    return result, carry_out


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_shapes() -> List[str]:
    """Return list of all shape names."""
    return [
        'RIPPLE_ADD', 'RIPPLE_SUB', 'PARALLEL_AND', 'PARALLEL_OR',
        'PARALLEL_XOR', 'SHIFT_LEFT', 'SHIFT_RIGHT', 'ROTATE_LEFT',
        'ROTATE_RIGHT', 'INCREMENT', 'DECREMENT', 'TRANSFER', 'IDENTITY',
    ]


def show_formula(shape_name: str) -> None:
    """Print the formula for a shape."""
    name = shape_name.upper()
    if name not in SHAPE_INFO:
        print(f"Unknown shape: {shape_name}")
        print(f"Available: {', '.join(list_shapes())}")
        return

    info = SHAPE_INFO[name]
    print(f"Shape: {name}")
    print(f"Formula: {info.get('formula', 'N/A')}")
    print(f"Polynomial: {info.get('polynomial', 'N/A')}")
    print(f"Description: {info.get('description', 'N/A')}")


def demo():
    """Run a quick demonstration of the shapes."""
    print("=" * 52)
    print("  FROZEN SHAPES DEMO - Computation is Geometry")
    print("=" * 52)
    print()

    print("1. Addition: 42 + 13 = ?")
    result = add(42, 13, verbose=True)
    print(f"   Result: {int(result)}")
    print()

    print("2. XOR: 0x55 ^ 0xFF = ?")
    result = xor(0x55, 0xFF, verbose=True)
    print(f"   Result: {int(result)} (0x{int(result):02X})")
    print()

    print("3. Shift Left: 0x40 << 1 = ?")
    result, carry = asl(0x40, verbose=True)
    print(f"   Result: {int(result)} (0x{int(result):02X}), Carry: {int(carry)}")
    print()

    print("=" * 52)
    print("  The shapes ARE the computation. No learning.")
    print("  Pure Python. No frameworks. Just math.")
    print("=" * 52)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Arithmetic
    'add', 'add_with_carry', 'sub', 'inc', 'dec',
    # Logic
    'and_op', 'or_op', 'xor',
    # Shifts
    'asl', 'lsr', 'rol', 'ror',
    # Utility
    'list_shapes', 'show_formula', 'demo',
]
