"""
Geocadesia: Activation Kingdom

Nonlinearities that enable learning.
The shapes that break linearity and make depth meaningful.

"Without them, deep networks collapse to a single linear transformation."
"""

from __future__ import annotations
import math
from typing import List
from .catalog import shape, Shape, Kingdom, Arity, ShapeType, FrozenStatus


# =============================================================================
# ReLU - The workhorse
# =============================================================================

@shape(
    name="relu",
    kingdom=Kingdom.ACTIVATION,
    formula="max(0, x)",
    definition="Rectified Linear Unit. Passes positive, zeros negative.",
    arity=Arity.UNARY,
    see_also=["leaky_relu", "gelu", "swish"],
    examples=[
        (-1.0, 0.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (2.5, 2.5),
    ],
)
def relu(x: float) -> float:
    """Rectified Linear Unit."""
    return max(0.0, x)


class ReLU:
    """
    ReLU: Rectified Linear Unit

    The workhorse of deep learning. Simple, fast, effective.
    Enabled the deep learning revolution by solving vanishing gradients.
    """

    def __call__(self, x: float) -> float:
        return relu(x)

    def __repr__(self):
        return "<ReLU: max(0, x)>"


# =============================================================================
# Sigmoid - The S-curve
# =============================================================================

@shape(
    name="sigmoid",
    kingdom=Kingdom.ACTIVATION,
    formula="1 / (1 + e^(-x))",
    definition="Logistic sigmoid. Squashes to (0, 1).",
    arity=Arity.UNARY,
    see_also=["tanh", "softmax", "swish"],
    examples=[
        (-5.0, 0.007),
        (0.0, 0.5),
        (5.0, 0.993),
    ],
)
def sigmoid(x: float) -> float:
    """Logistic sigmoid (numerically stable)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


class Sigmoid:
    """
    Sigmoid: The Logistic Curve

    The classic S-curve. Squashes any value to (0, 1).
    Used for probabilities, gates, and historical neural networks.
    """

    def __call__(self, x: float) -> float:
        return sigmoid(x)

    def __repr__(self):
        return "<Sigmoid: 1/(1+e^-x)>"


# =============================================================================
# Tanh - Zero-centered
# =============================================================================

@shape(
    name="tanh",
    kingdom=Kingdom.ACTIVATION,
    formula="(e^x - e^(-x)) / (e^x + e^(-x))",
    definition="Hyperbolic tangent. Zero-centered, outputs (-1, 1).",
    arity=Arity.UNARY,
    see_also=["sigmoid", "gelu"],
    examples=[
        (-2.0, -0.964),
        (0.0, 0.0),
        (2.0, 0.964),
    ],
)
def tanh(x: float) -> float:
    """Hyperbolic tangent."""
    return math.tanh(x)


class Tanh:
    """
    Tanh: Hyperbolic Tangent

    Zero-centered sigmoid. Outputs in (-1, 1).
    Often preferred in RNNs for balanced activations.
    """

    def __call__(self, x: float) -> float:
        return tanh(x)

    def __repr__(self):
        return "<Tanh: (e^x - e^-x)/(e^x + e^-x)>"


# =============================================================================
# GELU - Transformer standard
# =============================================================================

@shape(
    name="gelu",
    kingdom=Kingdom.ACTIVATION,
    formula="x · Φ(x)",
    definition="Gaussian Error Linear Unit. Smooth, probabilistic gating.",
    arity=Arity.UNARY,
    see_also=["relu", "swish"],
    examples=[
        (-1.0, -0.159),
        (0.0, 0.0),
        (1.0, 0.841),
    ],
)
def gelu(x: float) -> float:
    """Gaussian Error Linear Unit (exact)."""
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def gelu_approx(x: float) -> float:
    """GELU approximation using tanh (faster)."""
    sqrt_2_pi = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(sqrt_2_pi * (x + 0.044715 * x**3)))


class GELU:
    """
    GELU: Gaussian Error Linear Unit

    The transformer activation. Smooth, probabilistic.
    Standard in BERT, GPT, and modern architectures.
    """

    def __call__(self, x: float) -> float:
        return gelu(x)

    @staticmethod
    def approx(x: float) -> float:
        return gelu_approx(x)

    def __repr__(self):
        return "<GELU: x·Φ(x)>"


# =============================================================================
# Swish - Self-gated
# =============================================================================

@shape(
    name="swish",
    kingdom=Kingdom.ACTIVATION,
    formula="x · sigmoid(x)",
    definition="Self-gated activation. Discovered by neural architecture search.",
    arity=Arity.UNARY,
    built_from=["sigmoid"],
    see_also=["gelu", "relu"],
    examples=[
        (-1.0, -0.269),
        (0.0, 0.0),
        (1.0, 0.731),
    ],
)
def swish(x: float) -> float:
    """Swish activation (SiLU)."""
    return x * sigmoid(x)


class Swish:
    """
    Swish: Self-Gated Linear Unit

    Also known as SiLU. The input gates itself.
    Discovered by neural architecture search at Google.
    """

    def __call__(self, x: float) -> float:
        return swish(x)

    def __repr__(self):
        return "<Swish: x·sigmoid(x)>"


# =============================================================================
# Softmax - Probabilities
# =============================================================================

@shape(
    name="softmax",
    kingdom=Kingdom.ACTIVATION,
    formula="e^(xᵢ) / Σⱼ e^(xⱼ)",
    definition="Converts logits to probability distribution.",
    arity=Arity.NARY,
    see_also=["sigmoid"],
    examples=[
        ([1, 1, 1], [0.333, 0.333, 0.333]),
        ([2, 1, 0], [0.659, 0.242, 0.099]),
    ],
)
def softmax(x: List[float]) -> List[float]:
    """Softmax (numerically stable)."""
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]


class Softmax:
    """
    Softmax: Logits to Probabilities

    The classification finale. Converts any vector
    to a probability distribution that sums to 1.
    Core of attention mechanisms.
    """

    def __call__(self, x: List[float]) -> List[float]:
        return softmax(x)

    @staticmethod
    def with_temperature(x: List[float], temperature: float) -> List[float]:
        """Softmax with temperature scaling."""
        scaled = [xi / temperature for xi in x]
        return softmax(scaled)

    def __repr__(self):
        return "<Softmax: e^xᵢ / Σe^xⱼ>"


# =============================================================================
# Leaky ReLU
# =============================================================================

@shape(
    name="leaky_relu",
    kingdom=Kingdom.ACTIVATION,
    formula="x if x > 0 else αx",
    definition="ReLU with small negative slope to prevent dying neurons.",
    arity=Arity.UNARY,
    see_also=["relu"],
    examples=[
        (-1.0, -0.01),
        (0.0, 0.0),
        (1.0, 1.0),
    ],
)
def leaky_relu(x: float, alpha: float = 0.01) -> float:
    """Leaky ReLU."""
    return x if x > 0 else alpha * x


class LeakyReLU:
    """
    Leaky ReLU: ReLU with negative slope

    Prevents dying neurons by allowing small gradient
    for negative inputs. The alpha parameter controls
    the negative slope.
    """

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def __call__(self, x: float) -> float:
        return leaky_relu(x, self.alpha)

    def __repr__(self):
        return f"<LeakyReLU: α={self.alpha}>"
