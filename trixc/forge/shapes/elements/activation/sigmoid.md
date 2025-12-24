# Sigmoid

*The logistic curve — Squashing to (0, 1)*

```
┌─────────────────────────────────────────────────────────────┐
│ SIGMOID                                                     │
├─────────────────────────────────────────────────────────────┤
│ Kingdom: Activation                                         │
│ Type: Elemental                                             │
│ Arity: Unary                                                │
│ Frozen: Yes                                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Definition

### Mathematical

```
σ(x) = 1 / (1 + e^(-x))
```

Equivalent forms:
```
σ(x) = e^x / (e^x + 1)
σ(x) = 0.5 + 0.5 * tanh(x/2)
```

### Prose

Sigmoid squashes any real number into the range (0, 1). Large positive inputs approach 1, large negative inputs approach 0, and 0 maps to 0.5. It's the classic "S-curve" that appears throughout nature and statistics.

---

## Visual

```
    output
      │
    1 ┤            ___________
      │         __╱
  0.5 ┼ ─ ─ ─ ╱─ ─ ─ ─ ─ ─ ─
      │     ╱
    0 ┼____╱──────────────────
      │
      └─┼──┼──┼──┼──┼──┼──┼─ input
       -4 -2  0  2  4  6  8

The characteristic S-curve (sigmoid = "S-shaped").
```

---

## Examples

```
sigmoid(-5.0) ≈ 0.007    # Very negative → near 0
sigmoid(-2.0) ≈ 0.119    # Negative → low
sigmoid(0.0)  = 0.5      # Zero → middle
sigmoid(2.0)  ≈ 0.881    # Positive → high
sigmoid(5.0)  ≈ 0.993    # Very positive → near 1
```

---

## Implementation

### Python

```python
import math

def sigmoid(x: float) -> float:
    """Logistic sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x))

# Numerically stable version
def sigmoid_stable(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)
```

### C

```c
#include <math.h>

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Numerically stable
static inline float sigmoid_stable(float x) {
    if (x >= 0) {
        return 1.0f / (1.0f + expf(-x));
    } else {
        float exp_x = expf(x);
        return exp_x / (1.0f + exp_x);
    }
}
```

---

## Relationships

### Built From

Sigmoid is elemental (built from exp, but exp isn't in Geocadesia as a shape).

### Used In

- **Binary classification** — Output layer for yes/no predictions
- **Gates (LSTM, GRU)** — Controls information flow
- **Attention weights** — Sometimes used before softmax
- **[swish](swish.md)** — swish(x) = x * sigmoid(x)

### See Also

- **[tanh](tanh.md)** — Rescaled sigmoid: tanh(x) = 2σ(2x) - 1
- **[softmax](softmax.md)** — Multi-class generalization
- **[swish](swish.md)** — Self-gated using sigmoid

---

## Use Cases

1. **Binary Classification**: Sigmoid outputs a probability in (0, 1). Perfect for "what's the probability this is class A?"

2. **Gating**: In LSTMs and GRUs, sigmoid gates control what information flows through. Values near 0 block, near 1 pass.

3. **Smooth Approximation**: Sigmoid smoothly approximates a step function. Useful when you need differentiability.

4. **Logistic Regression**: The original use case — modeling binary outcomes.

---

## Properties

- **Range**: (0, 1) — never exactly 0 or 1
- **Monotonic**: Strictly increasing
- **Symmetric**: σ(x) + σ(-x) = 1
- **Derivative**: σ'(x) = σ(x) · (1 - σ(x))
- **Fixed point**: σ(0) = 0.5

---

## The Vanishing Gradient Problem

Sigmoid saturates at extremes — the derivative approaches zero:

```
At x = 0:   σ'(0) = 0.25 (maximum)
At x = 5:   σ'(5) ≈ 0.007 (tiny)
At x = 10:  σ'(10) ≈ 0.00005 (vanishing)
```

In deep networks, these tiny gradients multiply, making early layers nearly untrainable. This is why ReLU replaced sigmoid as the default activation.

**Sigmoid is still used** for:
- Output layers (probabilities)
- Gates (where saturation is desired)
- Shallow networks

---

## Historical Significance

Sigmoid was THE activation function from the 1980s to 2010s. The entire field of neural networks was built on it:

- Backpropagation (1986) used sigmoid
- Universal approximation theorems assumed sigmoid-like functions
- Early deep learning papers used sigmoid

Then ReLU arrived and everything changed. But sigmoid remains essential for probabilities and gates.

---

## The Natural S-Curve

The sigmoid appears naturally in:
- Population growth (logistic equation)
- Drug dose-response curves
- Market adoption curves
- Any saturation phenomenon

It's not just a neural network trick — it's a fundamental pattern of bounded growth.

---

*"Son of a bitch must pay."*
