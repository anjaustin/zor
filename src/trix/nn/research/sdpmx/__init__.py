"""
Vi's Synthesis: SDPMX Pipeline

S -> D -> P -> M -> X
Smooth -> Differentiate -> Project -> Mask -> XOR

"The geometry dictates the operator order. AB != BA."

Mathematical foundation:
- Robert: Hilbert space geometry
- V'Gem: Splines + XOR operators
- Vision: Ghost in the Machine

This represents a unique mathematical approach to neural computation.
Kept as research branch pending integration with Providence.

Status: Research (Vi's Synthesis)
Consolidated: December 27, 2025
"""

from .sdpmx_pipeline import (
    Spline1D as SDPMXSpline1D,
    SmoothingOperator,
    DifferentiationOperator,
    ProjectionOperator,
    MaskOperator,
    XOROperator,
    SDPMXPipeline,
)

__all__ = [
    "SDPMXSpline1D",
    "SmoothingOperator",
    "DifferentiationOperator",
    "ProjectionOperator",
    "MaskOperator",
    "XOROperator",
    "SDPMXPipeline",
]
