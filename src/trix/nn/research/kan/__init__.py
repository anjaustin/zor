"""
Track B: Kolmogorov-Arnold Networks

VGem's insight: 768-dim grid = Heat Death. 768 x 1D splines = Doable.

Kolmogorov-Arnold representation:
    f(x1, x2, ..., xn) = sum(phi_i(x_i))

Each phi_i is a 1D spline (piecewise linear function).

This is a separate research track from the main TriX architecture.
It may be integrated if proven valuable at scale.

Modules:
    additive_kan.py     - AdditiveKAN, ProductSplineKAN, SharedAdditiveKAN
    kan_hierarchical.py - HierarchicalKANFFN (KAN + O(sqrt(n)) routing)
    hybrid_kan.py       - Hybrid approaches

Status: Research (Track B)
Consolidated: December 27, 2025
"""

from .additive_kan import (
    Spline1D,
    KANTile as AdditiveKANTile,
    AdditiveKAN,
    ProductSplineKAN,
    SharedAdditiveKAN,
)

from .kan_hierarchical import (
    KANTile,  # This is the hierarchical version
    HierarchicalKANFFN,
)

from .hybrid_kan import (
    HybridKANTile,
    HybridKANFFN,
)

__all__ = [
    "Spline1D",
    "AdditiveKANTile",
    "KANTile",
    "AdditiveKAN",
    "ProductSplineKAN",
    "SharedAdditiveKAN",
    "HierarchicalKANFFN",
    "HybridKANTile",
    "HybridKANFFN",
]
