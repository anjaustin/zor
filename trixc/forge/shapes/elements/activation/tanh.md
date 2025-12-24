# tanh

*Hyperbolic Tangent — Squashing to (-1, 1)*

```
┌─────────────────────────────────────────────────────────────┐
│ TANH                                                        │
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
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

Equivalent forms:
```
tanh(x) = 2·sigmoid(2x) - 1
tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
```

### Prose

Hyperbolic tangent squashes any real number into the range (-1, 1). Unlike sigmoid, it's zero-centered — negative inputs give negative outputs. This often helps training by keeping activations balanced around zero.

---

## Visual

```
    output
      │
    1 ┤            ___________
      │         __╱
    0 ┼ ─ ─ ─ ╱─ ─ ─ ─ ─ ─ ─
      │     ╱
   -1 ┼____╱──────────────────
      │
      └─┼──┼──┼──┼──┼──┼──┼─ input
       -4 -2  0  2  4  6  8

Zero-centered S-curve.
```

---

## Examples

```
tanh(-5.0) ≈ -0.9999   # Very negative → near -1
tanh(-2.0) ≈ -0.964    # Negative → negative
tanh(0.0)  = 0.0       # Zero → zero (centered!)
tanh(2.0)  ≈ 0.964     # Positive → positive
tanh(5.0)  ≈ 0.9999    # Very positive → near 1
```

---

## Implementation

### Python

```python
import math

def tanh(x: float) -> float:
    """Hyperbolic tangent."""
    return math.tanh(x)

# Manual implementation
def tanh_manual(x: float) -> float:
    exp_2x = math.exp(2 * x)
    return (exp_2x - 1) / (exp_2x + 1)
```

### C

```c
#include <math.h>

static inline float tanh_shape(float x) {
    return tanhf(x);
}

// Manual implementation
static inline float tanh_manual(float x) {
    float exp_2x = expf(2.0f * x);
    return (exp_2x - 1.0f) / (exp_2x + 1.0f);
}
```

---

## Relationships

### Built From

tanh is elemental (defined via exp, but treated as primitive).

### Used In

- **RNNs** — Often preferred over sigmoid for hidden states
- **LSTMs** — Cell state uses tanh, gates use sigmoid
- **Normalization** — Sometimes used to bound values
- **Output layers** — When output should be in (-1, 1)

### See Also

- **[sigmoid](sigmoid.md)** — tanh(x) = 2σ(2x) - 1
- **[ReLU](relu.md)** — Unbounded alternative
- **[GELU](gelu.md)** — Smooth tanh-like shape

---

## Use Cases

1. **RNN Hidden States**: tanh's zero-centering helps recurrent networks maintain stable activations over time.

2. **LSTM Cell State**: The candidate cell state uses tanh to produce values in (-1, 1) before gating.

3. **Feature Normalization**: Squash features to a bounded range without losing sign information.

4. **Output Scaling**: When predictions should be in (-1, 1), tanh is natural.

---

## Properties

- **Range**: (-1, 1) — symmetric around zero
- **Zero-centered**: tanh(0) = 0
- **Odd function**: tanh(-x) = -tanh(x)
- **Derivative**: tanh'(x) = 1 - tanh²(x)
- **Steeper than sigmoid**: Saturates faster

---

## tanh vs sigmoid

```
                sigmoid          tanh
Range:          (0, 1)          (-1, 1)
Center:         0.5             0
At x=0:         0.5             0
Derivative:     σ(1-σ)          1-tanh²
Relationship:   σ(x)            2σ(2x)-1
```

tanh is essentially a rescaled sigmoid. But the zero-centering matters:
- Sigmoid outputs are always positive → biased gradients
- tanh outputs are balanced → more stable training

---

## The Vanishing Gradient Problem (Again)

Like sigmoid, tanh saturates at extremes:

```
At x = 0:   tanh'(0) = 1.0 (maximum, better than sigmoid!)
At x = 2:   tanh'(2) ≈ 0.07
At x = 5:   tanh'(5) ≈ 0.00018
```

tanh has a steeper slope at the origin than sigmoid (1.0 vs 0.25), which helps. But it still vanishes for large inputs.

---

## Hyperbolic Functions

tanh is part of the hyperbolic function family:
- sinh(x) = (e^x - e^(-x)) / 2
- cosh(x) = (e^x + e^(-x)) / 2
- tanh(x) = sinh(x) / cosh(x)

These mirror sin/cos/tan but for hyperbolas instead of circles. The name comes from the relationship to the unit hyperbola x² - y² = 1.

---

*"What does that mean? 'China is here.' I don't even know what that means."*
