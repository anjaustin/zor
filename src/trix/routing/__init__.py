"""
trix.routing: Learn routing, freeze execution.

The core insight: separate WHAT to compute (learned routing) from
HOW to compute it (frozen polynomial shapes).

Example:
    from trix.routing import RoutingPipeline

    pipe = RoutingPipeline(bit_width=8)
    pipe.add_builtins(["add", "sub", "xor", "and"])
    result = pipe.train()
    print(f"Accuracy: {result.accuracy*100}%")  # 100%
    print(f"Epochs: {result.epochs}")           # 1-2
    print(f"Router params: {result.router_params}")  # 76

    # Use it
    assert pipe.compute(17, 38, "add") == 55

    # Export it
    pipe.save("calculator.pt")
"""

from .pipeline import RoutingPipeline, TrainResult
from .shapes import (
    add_8bit, sub_8bit, inc_8bit, dec_8bit,
    xor_8bit, and_8bit, or_8bit, not_8bit,
    shl_8bit, shr_8bit, rol_8bit, ror_8bit,
    BUILTIN_SHAPES, get_builtin
)
from .primitives import xor, and_, or_, not_, full_adder

__all__ = [
    # Main class
    "RoutingPipeline",
    "TrainResult",

    # Built-in shapes
    "add_8bit", "sub_8bit", "inc_8bit", "dec_8bit",
    "xor_8bit", "and_8bit", "or_8bit", "not_8bit",
    "shl_8bit", "shr_8bit", "rol_8bit", "ror_8bit",
    "BUILTIN_SHAPES", "get_builtin",

    # Primitives
    "xor", "and_", "or_", "not_", "full_adder",
]
