"""
TRIX Forge - Neural Compilation Framework

Watch neural networks think.
"""

from .model import ModelIR, Shape, TensorSpec
from .levers import Lever, LeverType
from .events import Event, EventSpine
from .compiler import Compiler, Target
from .nge import NGEFile, NGEHeader

__version__ = "1.0.0"
__all__ = [
    "ModelIR", "Shape", "TensorSpec",
    "Lever", "LeverType",
    "Event", "EventSpine",
    "Compiler", "Target",
    "NGEFile", "NGEHeader",
]
