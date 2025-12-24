"""
Geocadesia: Shape Implementations (Python)

Reference implementations of all shapes in Geocadesia.
These are the canonical definitions — other implementations
should match this behavior.

"It's all in the reflexes."
"""

from __future__ import annotations
import math
from typing import Tuple, List

# =============================================================================
# LOGIC KINGDOM - Boolean operations (frozen, exact)
# =============================================================================

def xor(a: float, b: float) -> float:
    """
    Differentiable XOR for continuous inputs in [0, 1].

    XOR(a, b) = a + b - 2ab

    Returns 1 when exactly one input is high.
    """
    return a + b - 2.0 * a * b


def xor_discrete(a: int, b: int) -> int:
    """Discrete XOR for boolean inputs."""
    return a ^ b


def and_gate(a: float, b: float) -> float:
    """
    Differentiable AND for continuous inputs in [0, 1].

    AND(a, b) = a * b

    Returns high only when both inputs are high.
    """
    return a * b


def and_discrete(a: int, b: int) -> int:
    """Discrete AND for boolean inputs."""
    return a & b


def or_gate(a: float, b: float) -> float:
    """
    Differentiable OR for continuous inputs in [0, 1].

    OR(a, b) = a + b - ab (probabilistic OR)

    Returns high when at least one input is high.
    """
    return a + b - a * b


def or_discrete(a: int, b: int) -> int:
    """Discrete OR for boolean inputs."""
    return a | b


def not_gate(a: float) -> float:
    """
    Differentiable NOT for continuous inputs in [0, 1].

    NOT(a) = 1 - a

    Inverts the input.
    """
    return 1.0 - a


def not_discrete(a: int) -> int:
    """Discrete NOT for boolean inputs."""
    return 1 - a


def nand(a: float, b: float) -> float:
    """
    Differentiable NAND for continuous inputs in [0, 1].

    NAND(a, b) = 1 - ab

    Universal gate: any logic can be built from NAND alone.
    """
    return 1.0 - a * b


def nand_discrete(a: int, b: int) -> int:
    """Discrete NAND for boolean inputs."""
    return 1 - (a & b)


def nor(a: float, b: float) -> float:
    """
    Differentiable NOR for continuous inputs in [0, 1].

    NOR(a, b) = (1-a)(1-b)

    Universal gate: any logic can be built from NOR alone.
    """
    return (1.0 - a) * (1.0 - b)


def nor_discrete(a: int, b: int) -> int:
    """Discrete NOR for boolean inputs."""
    return 1 - (a | b)


def xnor(a: float, b: float) -> float:
    """
    Differentiable XNOR (equivalence) for continuous inputs in [0, 1].

    XNOR(a, b) = 1 - a - b + 2ab

    Returns 1 when both inputs are the same.
    """
    return 1.0 - a - b + 2.0 * a * b


def xnor_discrete(a: int, b: int) -> int:
    """Discrete XNOR for boolean inputs."""
    return 1 - (a ^ b)


# =============================================================================
# ARITHMETIC KINGDOM - Numeric computation (frozen)
# =============================================================================

def half_adder(a: float, b: float) -> Tuple[float, float]:
    """
    Differentiable half adder for continuous inputs in [0, 1].

    Returns (sum, carry):
        sum   = XOR(a, b) = a + b - 2ab
        carry = AND(a, b) = ab

    Half adder: adds two bits, produces sum and carry.
    """
    sum_out = a + b - 2.0 * a * b
    carry = a * b
    return sum_out, carry


def half_adder_discrete(a: int, b: int) -> Tuple[int, int]:
    """Discrete half adder."""
    return (a ^ b, a & b)


def full_adder(a: float, b: float, c_in: float) -> Tuple[float, float]:
    """
    Differentiable full adder for continuous inputs in [0, 1].

    Returns (sum, carry_out):
        sum    = a XOR b XOR c_in
        c_out  = (a AND b) OR (c_in AND (a XOR b))

    Full adder: adds two bits plus carry-in, produces sum and carry-out.
    """
    # First half adder: a XOR b
    sum1 = a + b - 2.0 * a * b
    carry1 = a * b

    # Second half adder: sum1 XOR c_in
    sum_out = sum1 + c_in - 2.0 * sum1 * c_in
    carry2 = sum1 * c_in

    # OR the carries
    c_out = carry1 + carry2 - carry1 * carry2

    return sum_out, c_out


def full_adder_discrete(a: int, b: int, c_in: int) -> Tuple[int, int]:
    """Discrete full adder."""
    sum_out = a ^ b ^ c_in
    c_out = (a & b) | (c_in & (a ^ b))
    return sum_out, c_out


# =============================================================================
# ACTIVATION KINGDOM - Nonlinearities (mostly frozen)
# =============================================================================

def relu(x: float) -> float:
    """
    Rectified Linear Unit.

    ReLU(x) = max(0, x)

    The workhorse activation of deep learning.
    """
    return max(0.0, x)


def sigmoid(x: float) -> float:
    """
    Logistic sigmoid function.

    sigmoid(x) = 1 / (1 + e^(-x))

    Squashes input to (0, 1). Used for probabilities and gates.
    """
    # Numerically stable version
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def tanh_act(x: float) -> float:
    """
    Hyperbolic tangent.

    tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    Zero-centered version of sigmoid, output in (-1, 1).
    """
    return math.tanh(x)


def gelu(x: float) -> float:
    """
    Gaussian Error Linear Unit.

    GELU(x) = x * Phi(x)

    Where Phi is the CDF of the standard normal distribution.
    Used in transformers (BERT, GPT).
    """
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def gelu_approx(x: float) -> float:
    """
    GELU approximation using tanh.

    Faster than exact GELU, used in many implementations.
    """
    sqrt_2_pi = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(sqrt_2_pi * (x + 0.044715 * x**3)))


def swish(x: float) -> float:
    """
    Swish activation (also known as SiLU).

    swish(x) = x * sigmoid(x)

    Self-gated activation discovered by neural architecture search.
    """
    return x * sigmoid(x)


def softmax(x: List[float]) -> List[float]:
    """
    Softmax function.

    softmax(x)_i = e^(x_i) / sum_j(e^(x_j))

    Converts logits to probability distribution.
    Numerically stable implementation.
    """
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]


def leaky_relu(x: float, alpha: float = 0.01) -> float:
    """
    Leaky ReLU.

    leaky_relu(x) = x if x > 0 else alpha * x

    Allows small gradient for negative values to prevent dying neurons.
    """
    return x if x > 0 else alpha * x


# =============================================================================
# POOLING KINGDOM - Reduction operations (frozen)
# =============================================================================

def max_pool(x: List[float]) -> float:
    """Maximum pooling - take the maximum value."""
    return max(x)


def avg_pool(x: List[float]) -> float:
    """Average pooling - take the mean value."""
    return sum(x) / len(x)


def sum_pool(x: List[float]) -> float:
    """Sum pooling - sum all values."""
    return sum(x)


def min_pool(x: List[float]) -> float:
    """Minimum pooling - take the minimum value."""
    return min(x)


# =============================================================================
# TESTING
# =============================================================================

def _test_logic():
    """Test logic gates."""
    # XOR truth table
    assert abs(xor(0.0, 0.0) - 0.0) < 1e-6
    assert abs(xor(0.0, 1.0) - 1.0) < 1e-6
    assert abs(xor(1.0, 0.0) - 1.0) < 1e-6
    assert abs(xor(1.0, 1.0) - 0.0) < 1e-6

    # AND truth table
    assert abs(and_gate(0.0, 0.0) - 0.0) < 1e-6
    assert abs(and_gate(0.0, 1.0) - 0.0) < 1e-6
    assert abs(and_gate(1.0, 0.0) - 0.0) < 1e-6
    assert abs(and_gate(1.0, 1.0) - 1.0) < 1e-6

    # OR truth table
    assert abs(or_gate(0.0, 0.0) - 0.0) < 1e-6
    assert abs(or_gate(0.0, 1.0) - 1.0) < 1e-6
    assert abs(or_gate(1.0, 0.0) - 1.0) < 1e-6
    assert abs(or_gate(1.0, 1.0) - 1.0) < 1e-6

    print("Logic tests passed!")


def _test_arithmetic():
    """Test arithmetic compounds."""
    # Half adder
    assert half_adder_discrete(0, 0) == (0, 0)
    assert half_adder_discrete(0, 1) == (1, 0)
    assert half_adder_discrete(1, 0) == (1, 0)
    assert half_adder_discrete(1, 1) == (0, 1)

    # Full adder
    assert full_adder_discrete(0, 0, 0) == (0, 0)
    assert full_adder_discrete(1, 1, 0) == (0, 1)
    assert full_adder_discrete(1, 1, 1) == (1, 1)

    print("Arithmetic tests passed!")


def _test_activations():
    """Test activation functions."""
    # ReLU
    assert relu(-1.0) == 0.0
    assert relu(0.0) == 0.0
    assert relu(1.0) == 1.0

    # Sigmoid
    assert abs(sigmoid(0.0) - 0.5) < 1e-6
    assert sigmoid(-100) < 0.01
    assert sigmoid(100) > 0.99

    # Softmax
    result = softmax([1.0, 1.0, 1.0])
    assert all(abs(r - 1/3) < 1e-6 for r in result)
    assert abs(sum(result) - 1.0) < 1e-6

    print("Activation tests passed!")


if __name__ == "__main__":
    _test_logic()
    _test_arithmetic()
    _test_activations()
    print("\nAll Geocadesia shape tests passed!")
    print('"It\'s all in the reflexes."')
