"""
TriX - Frozen Shape Foundry

Type a function. Get hardware.
"""

__version__ = "1.0.0"

from .shapes.logic import xor, and_, or_, not_, nand, nor, xnor
from .shapes.arithmetic import add, sub, mul
from .foundry import Shape, Foundry

__all__ = [
    'xor', 'and_', 'or_', 'not_', 'nand', 'nor', 'xnor',
    'add', 'sub', 'mul',
    'Shape', 'Foundry',
]
