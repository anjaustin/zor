# Swish

*Self-Gated Linear Unit — x times sigmoid(x)*

```
┌─────────────────────────────────────────────────────────────┐
│ SWISH                                                       │
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
swish(x) = x · σ(x) = x · sigmoid(x) = x / (1 + e^(-x))
```

Also known as SiLU (Sigmoid Linear Unit).

### Prose

Swish multiplies the input by its own sigmoid. It's a "self-gated" activation: the input gates itself. Large positive values pass through nearly unchanged; negative values are suppressed but not completely zeroed.

---

## Visual

```
    output
      │
    2 ┤              ╱
      │            ╱
    1 ┤          ╱
      │        ╱╱
    0 ┼─────╱╱────────────
      │  ╲╱
  -0.3┤ ╱
      │
      └─┼──┼──┼──┼──┼──┼─ input
       -4 -2  0  1  2  4

Shape very similar to GELU.
```

---

## Examples

```
swish(-4.0) ≈ -0.072   # Negative → small negative
swish(-1.0) ≈ -0.269   # Negative → suppressed
swish(0.0)  = 0.0      # Zero → zero
swish(1.0)  ≈ 0.731    # Positive → most passes
swish(4.0)  ≈ 3.928    # Large positive → nearly linear
```

---

## Implementation

### Python

```python
import math

def swish(x: float) -> float:
    """Swish activation (SiLU)."""
    return x / (1.0 + math.exp(-x))

# Equivalent
def swish_v2(x: float) -> float:
    return x * sigmoid(x)
```

### C

```c
#include <math.h>

static inline float swish(float x) {
    return x / (1.0f + expf(-x));
}

// Using sigmoid
static inline float swish_v2(float x) {
    return x * sigmoid(x);
}
```

---

## Relationships

### Built From

Conceptually: `x * sigmoid(x)`

But treated as elemental due to its status as a discovered activation.

### Used In

- **EfficientNet** — Standard activation
- **Mobile architectures** — Good accuracy/compute tradeoff
- **Some transformers** — Alternative to GELU

### See Also

- **[GELU](gelu.md)** — Similar shape, different derivation
- **[sigmoid](sigmoid.md)** — The gating function
- **[ReLU](relu.md)** — Simpler predecessor

---

## Use Cases

1. **EfficientNet Family**: Swish is the default activation in EfficientNet, one of the most efficient CNN architectures.

2. **ReLU Replacement**: Drop-in replacement that often improves accuracy with minimal compute overhead.

3. **Self-Gated Networks**: The self-gating property can be seen as a form of attention at the neuron level.

4. **Smooth ReLU**: When you want ReLU-like behavior with smooth gradients.

---

## Properties

- **Smooth**: Infinitely differentiable
- **Non-monotonic**: Has a negative bump (minimum ≈ -0.28 at x ≈ -1.28)
- **Bounded below**: min ≈ -0.28
- **Unbounded above**: Grows linearly for large positive x
- **Derivative**: swish'(x) = sigmoid(x) + x · sigmoid(x) · (1 - sigmoid(x))

---

## Discovery by Neural Architecture Search

Swish was discovered (not designed) by Google Brain using reinforcement learning to search over activation function space. The search optimized for ImageNet accuracy.

```
Search space: combinations of unary/binary operations
Discovered: x * sigmoid(x)
Performance: Better than ReLU on ImageNet
```

This is remarkable: a machine learning algorithm discovered a better activation function for machine learning.

---

## Swish vs GELU

The two are nearly identical in shape:

```
x        swish(x)    GELU(x)     diff
-2.0     -0.238      -0.045      0.193
-1.0     -0.269      -0.159      0.110
0.0       0.000       0.000      0.000
1.0       0.731       0.841      0.110
2.0       1.762       1.955      0.193
```

GELU has a theoretical justification (stochastic regularization). Swish was found empirically. Both work well.

---

## Parametric Swish

Swish can be parameterized:

```
swish_β(x) = x · sigmoid(β · x)
```

- β = 0: Linear (swish → x/2)
- β = 1: Standard swish
- β → ∞: Approaches ReLU

Some architectures learn β during training.

---

## The Self-Gating Insight

Swish is "self-gated": the input determines its own gate.

```
gate = sigmoid(x)     # How much to let through
value = x             # What to let through
output = value * gate # Self-gated result
```

This is a primitive form of attention: the input "attends to itself" to decide how much to output.

---

*"I was born ready."*
