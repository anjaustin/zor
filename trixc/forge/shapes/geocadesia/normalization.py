"""
Geocadesia: Normalization Kingdom

Statistical transforms for training stability.
These shapes tame the wild activations.

"They exist because raw activations tend toward pathology."
"""

from __future__ import annotations
import math
from typing import List
from .catalog import shape, Shape, Kingdom, Arity, ShapeType, FrozenStatus


@shape(
    name="layer_norm",
    kingdom=Kingdom.NORMALIZATION,
    formula="(x - μ) / √(σ² + ε)",
    definition="Normalize across features. Stabilizes training.",
    arity=Arity.NARY,
    see_also=["rms_norm"],
)
def layer_norm(x: List[float], eps: float = 1e-5) -> List[float]:
    """Layer normalization (frozen, no affine params)."""
    n = len(x)

    # Mean
    mean = sum(x) / n

    # Variance
    var = sum((xi - mean) ** 2 for xi in x) / n

    # Normalize
    inv_std = 1.0 / math.sqrt(var + eps)
    return [(xi - mean) * inv_std for xi in x]


class LayerNorm:
    """
    Layer Normalization

    Normalizes across features (the hidden dimension).
    Essential for transformer stability.
    """

    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def __call__(self, x: List[float]) -> List[float]:
        return layer_norm(x, self.eps)

    def __repr__(self):
        return f"<LayerNorm: ε={self.eps}>"


@shape(
    name="rms_norm",
    kingdom=Kingdom.NORMALIZATION,
    formula="x / √(mean(x²) + ε)",
    definition="Normalize by RMS only. Simpler than LayerNorm.",
    arity=Arity.NARY,
    see_also=["layer_norm"],
)
def rms_norm(x: List[float], eps: float = 1e-5) -> List[float]:
    """RMS normalization (no mean subtraction)."""
    n = len(x)

    # RMS
    rms = math.sqrt(sum(xi * xi for xi in x) / n + eps)

    # Normalize
    return [xi / rms for xi in x]


class RMSNorm:
    """
    RMS Normalization

    Like LayerNorm but without mean subtraction.
    Simpler, often just as effective.
    Used in LLaMA and other modern architectures.
    """

    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def __call__(self, x: List[float]) -> List[float]:
        return rms_norm(x, self.eps)

    def __repr__(self):
        return f"<RMSNorm: ε={self.eps}>"
