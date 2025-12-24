# GELU

*Gaussian Error Linear Unit — Smooth stochastic regularization*

```
┌─────────────────────────────────────────────────────────────┐
│ GELU                                                        │
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
GELU(x) = x · Φ(x)
```

Where Φ(x) is the CDF of the standard normal distribution:
```
Φ(x) = 0.5 · (1 + erf(x / √2))
```

### Approximation (commonly used)

```
GELU(x) ≈ 0.5x · (1 + tanh(√(2/π) · (x + 0.044715x³)))
```

### Prose

GELU multiplies the input by the probability that a standard normal random variable is less than the input. It's a smooth, probabilistic gate: values likely to be "on" (positive) pass through; values likely to be "off" (negative) are suppressed.

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
  -0.2┤ ╱
      │
      └─┼──┼──┼──┼──┼──┼─ input
       -3 -2 -1  0  1  2

Note the slight dip below zero around x ≈ -0.5.
```

---

## Examples

```
GELU(-3.0) ≈ -0.004   # Very negative → near zero
GELU(-1.0) ≈ -0.159   # Negative → slightly negative (!)
GELU(0.0)  = 0.0      # Zero → zero
GELU(1.0)  ≈ 0.841    # Positive → mostly passes
GELU(3.0)  ≈ 2.996    # Very positive → unchanged
```

---

## Implementation

### Python

```python
import math

def gelu(x: float) -> float:
    """Gaussian Error Linear Unit (exact)."""
    return x * 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def gelu_approx(x: float) -> float:
    """GELU approximation using tanh."""
    return 0.5 * x * (1.0 + math.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
    ))
```

### C

```c
#include <math.h>

static inline float gelu(float x) {
    return x * 0.5f * (1.0f + erff(x / sqrtf(2.0f)));
}

static inline float gelu_approx(float x) {
    const float sqrt_2_pi = 0.7978845608f;  // sqrt(2/pi)
    float inner = sqrt_2_pi * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}
```

---

## Relationships

### Built From

GELU is elemental (built from erf, but erf isn't a Geocadesia shape).

### Used In

- **Transformers (BERT, GPT)** — Default activation in feed-forward blocks
- **Vision Transformers** — Same as language transformers
- **Modern architectures** — Increasingly popular default

### See Also

- **[ReLU](relu.md)** — Simpler, piecewise linear
- **[swish](swish.md)** — Similar smooth shape
- **[sigmoid](sigmoid.md)** — The gating function in GELU

---

## Use Cases

1. **Transformer Feed-Forward**: GELU is the standard activation in transformer architectures (BERT, GPT-2/3/4, T5).

2. **When ReLU Is Too Harsh**: GELU's smooth transition can help when ReLU's hard zero causes problems.

3. **Stochastic Regularization**: GELU can be seen as an expectation over stochastic binary dropout with input-dependent probability.

4. **Modern Default**: When building a new model in 2024+, GELU is often the first choice.

---

## Properties

- **Smooth**: Infinitely differentiable (unlike ReLU)
- **Non-monotonic**: Has a small negative region
- **Asymptotically linear**: Approaches identity for large positive x
- **Zero-centered output**: Balanced around zero for typical inputs
- **Derivative**: GELU'(x) = Φ(x) + x · φ(x), where φ is the PDF

---

## GELU vs ReLU vs Swish

```
             ReLU          GELU           Swish
Smoothness:  Piecewise     Smooth         Smooth
Negative:    0             Small dip      Small dip
Formula:     max(0,x)      x·Φ(x)         x·σ(x)
Compute:     Cheap         Medium         Medium
Transformer: Rarely        Standard       Sometimes
```

GELU and Swish are similar in shape. GELU has theoretical justification (stochastic regularization), while Swish was found via architecture search.

---

## The Probabilistic Interpretation

GELU has a beautiful interpretation:

Imagine randomly zeroing out neurons with probability 1 - Φ(x). The expected output is:
```
E[x · Bernoulli(Φ(x))] = x · Φ(x) = GELU(x)
```

GELU is the *expected* output under this stochastic dropout. It's dropout "baked in" as a deterministic function.

---

## Why Transformers Use GELU

When BERT was developed (2018), the authors experimented with various activations. GELU performed best, possibly because:

1. **Smoothness**: Transformers have deep feed-forward paths; smooth gradients help
2. **Non-monotonicity**: The slight negative region may provide regularization
3. **Probabilistic nature**: Aligns with the stochastic training of transformers

Whatever the reason, it worked. GELU became the transformer standard.

---

*"You know what Jack Burton always says..."*
