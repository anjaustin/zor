"""
Geocadesia: The Kingdom of Shapes

A living library of elemental and compound shapes for neural computation.
Not just documentation — the shapes themselves, embodied in code.

Usage:
    from geocadesia import XOR, AND, ReLU, FullAdder
    from geocadesia import catalog

    # Use shapes directly
    result = XOR(0.5, 0.7)

    # Query the catalog
    catalog.list_kingdom("logic")
    catalog.find(frozen=True, arity="binary")

"It's all in the reflexes."
"""

from __future__ import annotations

# Import all shapes for convenient access
from .logic import (
    XOR, AND, OR, NOT, NAND, NOR, XNOR,
    xor, and_gate, or_gate, not_gate, nand, nor, xnor,
)

from .arithmetic import (
    HalfAdder, FullAdder,
    half_adder, full_adder,
    Add, Sub, Mul, Neg,
    Popcount, Hamming,
    popcount, hamming,
)

from .activation import (
    ReLU, Sigmoid, Tanh, GELU, Swish, Softmax, LeakyReLU,
    relu, sigmoid, tanh, gelu, swish, softmax, leaky_relu,
)

from .normalization import (
    LayerNorm, RMSNorm,
    layer_norm, rms_norm,
)

from .pooling import (
    MaxPool, AvgPool, SumPool, MinPool, Argmin, Argmax,
    max_pool, avg_pool, sum_pool, min_pool, argmin, argmax,
)

from .catalog import catalog, Shape, Kingdom

# Binary format support
from .binary import (
    FrozenShape, FrozenShapeHeader,
    Opcode, KingdomID, ShapeTypeID, ArityID, FSHFlags,
    SHAPES as BINARY_SHAPES,
    get_shape as get_binary_shape,
    list_shapes as list_binary_shapes,
    save_all as save_all_binary,
    print_opcode_table,
)

__version__ = "0.1.0"
__all__ = [
    # Logic Kingdom
    "XOR", "AND", "OR", "NOT", "NAND", "NOR", "XNOR",
    "xor", "and_gate", "or_gate", "not_gate", "nand", "nor", "xnor",

    # Arithmetic Kingdom
    "HalfAdder", "FullAdder", "Add", "Sub", "Mul", "Neg",
    "half_adder", "full_adder",
    "Popcount", "Hamming", "popcount", "hamming",  # FrozenDB support

    # Activation Kingdom
    "ReLU", "Sigmoid", "Tanh", "GELU", "Swish", "Softmax", "LeakyReLU",
    "relu", "sigmoid", "tanh", "gelu", "swish", "softmax", "leaky_relu",

    # Normalization Kingdom
    "LayerNorm", "RMSNorm",
    "layer_norm", "rms_norm",

    # Pooling Kingdom
    "MaxPool", "AvgPool", "SumPool", "MinPool", "Argmin", "Argmax",
    "max_pool", "avg_pool", "sum_pool", "min_pool", "argmin", "argmax",

    # Catalog
    "catalog", "Shape", "Kingdom",

    # Binary format
    "FrozenShape", "FrozenShapeHeader",
    "Opcode", "KingdomID", "ShapeTypeID", "ArityID", "FSHFlags",
    "BINARY_SHAPES",
    "get_binary_shape", "list_binary_shapes", "save_all_binary",
    "print_opcode_table",
]
