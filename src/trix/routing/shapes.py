"""
Built-in frozen shapes for common operations.

Each shape is a polynomial that computes exactly on binary inputs.
Shapes take bit tensors [batch, bits] and return bit tensors.
"""

import torch
from torch import Tensor

from .primitives import xor, and_, or_, not_, full_adder


# =============================================================================
# 8-BIT ARITHMETIC
# =============================================================================

def add_8bit(a_bits: Tensor, b_bits: Tensor) -> Tensor:
    """
    8-bit addition using ripple-carry adder.

    Args:
        a_bits: [batch, 8] tensor of bits (LSB first)
        b_bits: [batch, 8] tensor of bits (LSB first)

    Returns:
        [batch, 8] result bits (LSB first)
    """
    result = []
    carry = torch.zeros_like(a_bits[:, 0])

    for i in range(8):
        s, carry = full_adder(a_bits[:, i], b_bits[:, i], carry)
        result.append(s)

    return torch.stack(result, dim=1)


def sub_8bit(a_bits: Tensor, b_bits: Tensor) -> Tensor:
    """
    8-bit subtraction: a - b = a + NOT(b) + 1 (two's complement).

    Args:
        a_bits: [batch, 8] tensor of bits (LSB first)
        b_bits: [batch, 8] tensor of bits (LSB first)

    Returns:
        [batch, 8] result bits (LSB first)
    """
    not_b = not_(b_bits)
    result = []
    carry = torch.ones_like(a_bits[:, 0])  # +1 for two's complement

    for i in range(8):
        s, carry = full_adder(a_bits[:, i], not_b[:, i], carry)
        result.append(s)

    return torch.stack(result, dim=1)


def inc_8bit(a_bits: Tensor, b_bits: Tensor = None) -> Tensor:
    """
    8-bit increment: a + 1.

    Note: b_bits is ignored (for consistent API).
    """
    result = []
    carry = torch.ones_like(a_bits[:, 0])  # Start with carry=1 to add 1

    for i in range(8):
        s = xor(a_bits[:, i], carry)
        carry = and_(a_bits[:, i], carry)
        result.append(s)

    return torch.stack(result, dim=1)


def dec_8bit(a_bits: Tensor, b_bits: Tensor = None) -> Tensor:
    """
    8-bit decrement: a - 1.

    Note: b_bits is ignored (for consistent API).
    """
    # a - 1 = a + 0xFF (all ones) with carry-in = 0
    # Simpler: a - 1 = a + NOT(0) + 0 = a + 0xFF + 0
    result = []
    borrow = torch.ones_like(a_bits[:, 0])  # Start borrowing

    for i in range(8):
        # Subtract 1: if borrowing, flip bit and continue borrow if was 0
        s = xor(a_bits[:, i], borrow)
        borrow = and_(not_(a_bits[:, i]), borrow)
        result.append(s)

    return torch.stack(result, dim=1)


# =============================================================================
# 8-BIT LOGIC
# =============================================================================

def xor_8bit(a_bits: Tensor, b_bits: Tensor) -> Tensor:
    """8-bit XOR: a ^ b"""
    return xor(a_bits, b_bits)


def and_8bit(a_bits: Tensor, b_bits: Tensor) -> Tensor:
    """8-bit AND: a & b"""
    return and_(a_bits, b_bits)


def or_8bit(a_bits: Tensor, b_bits: Tensor) -> Tensor:
    """8-bit OR: a | b"""
    return or_(a_bits, b_bits)


def not_8bit(a_bits: Tensor, b_bits: Tensor = None) -> Tensor:
    """
    8-bit NOT: ~a

    Note: b_bits is ignored (for consistent API).
    """
    return not_(a_bits)


# =============================================================================
# 8-BIT SHIFTS
# =============================================================================

def shl_8bit(a_bits: Tensor, b_bits: Tensor = None) -> Tensor:
    """
    8-bit shift left by 1: a << 1

    Note: b_bits is ignored.
    """
    # Shift left: [b0, b1, b2, b3, b4, b5, b6, b7] -> [0, b0, b1, b2, b3, b4, b5, b6]
    zeros = torch.zeros_like(a_bits[:, :1])
    return torch.cat([zeros, a_bits[:, :-1]], dim=1)


def shr_8bit(a_bits: Tensor, b_bits: Tensor = None) -> Tensor:
    """
    8-bit shift right by 1: a >> 1

    Note: b_bits is ignored.
    """
    # Shift right: [b0, b1, b2, b3, b4, b5, b6, b7] -> [b1, b2, b3, b4, b5, b6, b7, 0]
    zeros = torch.zeros_like(a_bits[:, :1])
    return torch.cat([a_bits[:, 1:], zeros], dim=1)


def rol_8bit(a_bits: Tensor, b_bits: Tensor = None) -> Tensor:
    """
    8-bit rotate left by 1.

    Note: b_bits is ignored.
    """
    # Rotate left: [b0, b1, ..., b7] -> [b7, b0, b1, ..., b6]
    return torch.cat([a_bits[:, -1:], a_bits[:, :-1]], dim=1)


def ror_8bit(a_bits: Tensor, b_bits: Tensor = None) -> Tensor:
    """
    8-bit rotate right by 1.

    Note: b_bits is ignored.
    """
    # Rotate right: [b0, b1, ..., b7] -> [b1, b2, ..., b7, b0]
    return torch.cat([a_bits[:, 1:], a_bits[:, :1]], dim=1)


# =============================================================================
# SHAPE REGISTRY
# =============================================================================

BUILTIN_SHAPES = {
    "add": (add_8bit, lambda a, b: (a + b) & 0xFF),
    "sub": (sub_8bit, lambda a, b: (a - b) & 0xFF),
    "inc": (inc_8bit, lambda a, b: (a + 1) & 0xFF),
    "dec": (dec_8bit, lambda a, b: (a - 1) & 0xFF),
    "xor": (xor_8bit, lambda a, b: a ^ b),
    "and": (and_8bit, lambda a, b: a & b),
    "or": (or_8bit, lambda a, b: a | b),
    "not": (not_8bit, lambda a, b: (~a) & 0xFF),
    "shl": (shl_8bit, lambda a, b: (a << 1) & 0xFF),
    "shr": (shr_8bit, lambda a, b: (a >> 1) & 0xFF),
    "rol": (rol_8bit, lambda a, b: ((a << 1) | (a >> 7)) & 0xFF),
    "ror": (ror_8bit, lambda a, b: ((a >> 1) | (a << 7)) & 0xFF),
}


def get_builtin(name: str):
    """Get a built-in shape and its truth function by name."""
    if name not in BUILTIN_SHAPES:
        available = ", ".join(BUILTIN_SHAPES.keys())
        raise ValueError(f"Unknown shape '{name}'. Available: {available}")
    return BUILTIN_SHAPES[name]
