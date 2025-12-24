"""
Polynomial primitives for frozen geometry.

These are the building blocks for all frozen shapes.
Each function computes exactly on binary inputs {0, 1}.
"""

import torch
from torch import Tensor


def xor(a: Tensor, b: Tensor) -> Tensor:
    """XOR as polynomial: a + b - 2ab"""
    return a + b - 2 * a * b


def and_(a: Tensor, b: Tensor) -> Tensor:
    """AND as polynomial: ab"""
    return a * b


def or_(a: Tensor, b: Tensor) -> Tensor:
    """OR as polynomial: a + b - ab"""
    return a + b - a * b


def not_(a: Tensor) -> Tensor:
    """NOT as polynomial: 1 - a"""
    return 1 - a


def nand(a: Tensor, b: Tensor) -> Tensor:
    """NAND as polynomial: 1 - ab"""
    return 1 - a * b


def nor(a: Tensor, b: Tensor) -> Tensor:
    """NOR as polynomial: 1 - (a + b - ab)"""
    return 1 - (a + b - a * b)


def xnor(a: Tensor, b: Tensor) -> Tensor:
    """XNOR as polynomial: 1 - (a + b - 2ab)"""
    return 1 - (a + b - 2 * a * b)


def mux(sel: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """MUX: if sel then b else a"""
    # sel * b + (1 - sel) * a
    return a + sel * (b - a)


def full_adder(a: Tensor, b: Tensor, cin: Tensor) -> tuple[Tensor, Tensor]:
    """
    Full adder from polynomial primitives.

    Returns:
        (sum_bit, carry_out)
    """
    p = xor(a, b)
    sum_bit = xor(p, cin)
    carry_out = or_(and_(a, b), and_(cin, p))
    return sum_bit, carry_out
