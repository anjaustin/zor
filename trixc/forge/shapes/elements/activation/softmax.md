# Softmax

*From scores to probabilities — The classification finale*

```
┌─────────────────────────────────────────────────────────────┐
│ SOFTMAX                                                     │
├─────────────────────────────────────────────────────────────┤
│ Kingdom: Activation                                         │
│ Type: Elemental                                             │
│ Arity: N-ary (vector → vector)                              │
│ Frozen: Yes                                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Definition

### Mathematical

```
softmax(x)ᵢ = e^(xᵢ) / Σⱼ e^(xⱼ)
```

For numerical stability, subtract the max first:
```
softmax(x)ᵢ = e^(xᵢ - max(x)) / Σⱼ e^(xⱼ - max(x))
```

### Prose

Softmax converts a vector of real numbers into a probability distribution. All outputs are positive and sum to 1. Larger inputs get larger probabilities, exponentially so — softmax amplifies differences.

---

## Visual

```
Input logits:        [2.0, 1.0, 0.1]

                      ┌──────────────┐
    2.0 ──────────────│              │──── 0.659 (65.9%)
    1.0 ──────────────│   SOFTMAX    │──── 0.242 (24.2%)
    0.1 ──────────────│              │──── 0.099 (9.9%)
                      └──────────────┘
                                     sum = 1.0

Probabilities reflect relative magnitudes of inputs.
```

---

## Examples

```
softmax([1.0, 1.0, 1.0]) = [0.333, 0.333, 0.333]  # Equal → uniform
softmax([2.0, 1.0, 0.0]) = [0.659, 0.242, 0.099]  # Largest wins
softmax([10, 1, 1])       = [0.9999, 0.00005, 0.00005]  # Extreme
softmax([0.0])            = [1.0]                  # Single element
```

---

## Implementation

### Python

```python
import math

def softmax(x: list[float]) -> list[float]:
    """Numerically stable softmax."""
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    return [e / sum_exp for e in exp_x]

# NumPy version
def softmax_np(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

### C

```c
#include <math.h>

void softmax(const float* x, float* out, int n) {
    // Find max for numerical stability
    float max_x = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_x) max_x = x[i];
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf(x[i] - max_x);
        sum += out[i];
    }

    // Normalize
    for (int i = 0; i < n; i++) {
        out[i] /= sum;
    }
}
```

---

## Relationships

### Built From

Softmax is elemental (uses exp, but as a vector operation it's conceptually atomic).

### Used In

- **Classification output** — Final layer of classifiers
- **Attention** — softmax(QK^T / √d) produces attention weights
- **Mixture models** — Soft routing decisions
- **Reinforcement learning** — Policy distributions

### See Also

- **[sigmoid](sigmoid.md)** — Binary version (2-class softmax)
- **Attention compounds** — Use softmax internally
- **Temperature scaling** — softmax(x/T) adjusts sharpness

---

## Use Cases

1. **Multi-class Classification**: Convert logits to class probabilities. The predicted class is argmax.

2. **Attention Weights**: In transformers, softmax produces attention distributions over keys.

3. **Soft Decisions**: When you need a differentiable approximation to argmax.

4. **Mixture of Experts**: Softmax gates select which expert(s) to use.

---

## Properties

- **Output range**: (0, 1) for each element
- **Normalization**: Outputs sum to 1
- **Shift invariance**: softmax(x + c) = softmax(x)
- **Scale sensitivity**: Larger differences → sharper distribution
- **Derivative**: ∂softmax(x)ᵢ/∂xⱼ = softmax(x)ᵢ(δᵢⱼ - softmax(x)ⱼ)

---

## Temperature Scaling

Softmax can be "sharpened" or "smoothed" with temperature:

```
softmax(x/T)ᵢ = e^(xᵢ/T) / Σⱼ e^(xⱼ/T)
```

- T → 0: Approaches argmax (one-hot)
- T = 1: Standard softmax
- T → ∞: Approaches uniform distribution

Temperature is crucial for:
- Knowledge distillation (high T for soft targets)
- Exploration in RL (high T for more random actions)
- Calibration (adjust T to fix overconfidence)

---

## Numerical Stability

Naive softmax overflows for large inputs:

```python
# Bad: exp(1000) = inf
softmax([1000, 1, 1])  # → [inf/inf, 0, 0] = [nan, 0, 0]
```

The fix: subtract max(x) before exp:

```python
# Good: exp(1000-1000) = exp(0) = 1
softmax([1000, 1, 1])  # → [1/(1+ε), ε, ε] ≈ [1, 0, 0]
```

This is mathematically equivalent (shift invariance) but numerically stable.

---

## Softmax vs Sigmoid

For binary classification:
```
softmax([x, 0]) = [sigmoid(x), 1-sigmoid(x)]
```

Softmax with 2 classes and one fixed at 0 reduces to sigmoid. They're the same thing!

For multi-class, softmax generalizes sigmoid to N classes.

---

## The Attention Connection

Transformers use softmax for attention:

```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

Softmax converts similarity scores (QK^T) into attention weights — a probability distribution over what to attend to.

Without softmax, attention would just be linear. Softmax makes it selective.

---

*"It's all in the reflexes."*
