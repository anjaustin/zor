# ReLU

*Rectified Linear Unit — The workhorse of deep learning*

```
┌─────────────────────────────────────────────────────────────┐
│ ReLU                                                        │
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
ReLU(x) = max(0, x) = { x  if x > 0
                      { 0  if x ≤ 0
```

### Prose

ReLU passes positive values unchanged and zeros out negative values. It's the simplest non-linearity that enables deep learning. The "rectified" refers to keeping only the positive part, like a rectifier in electronics.

---

## Visual

```
    output
      │
    2 ┤           ╱
      │         ╱
    1 ┤       ╱
      │     ╱
    0 ┼───╱─────────
      │
   -1 ┤
      │
      └─┼──┼──┼──┼──┼─ input
       -2 -1  0  1  2

The characteristic "hinge" at zero.
```

---

## Examples

```
ReLU(-2.0) = 0.0    # Negative → zero
ReLU(-0.5) = 0.0    # Negative → zero
ReLU(0.0)  = 0.0    # Zero → zero
ReLU(0.5)  = 0.5    # Positive → unchanged
ReLU(2.0)  = 2.0    # Positive → unchanged
```

---

## Implementation

### Python

```python
def relu(x: float) -> float:
    """Rectified Linear Unit."""
    return max(0.0, x)

# Vectorized (NumPy)
def relu_vec(x):
    return np.maximum(0, x)
```

### C

```c
static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// Branchless version
static inline float relu_branchless(float x) {
    return x * (x > 0.0f);
}
```

---

## Relationships

### Built From

ReLU is elemental.

(Could be seen as `max(0, x)`, but `max` with a constant is the ReLU concept itself.)

### Used In

- **Nearly every modern neural network** — Default activation
- **Residual networks (ResNet)** — ReLU after each residual block
- **Convolutional networks** — ReLU after each conv layer
- **Transformers** — ReLU or GELU in feed-forward blocks

### See Also

- **[Leaky ReLU](leaky_relu.md)** — Allows small negative slope
- **[GELU](gelu.md)** — Smooth approximation
- **[swish](swish.md)** — Self-gated alternative

---

## Use Cases

1. **Default Activation**: When in doubt, use ReLU. It works well, trains fast, and is computationally cheap.

2. **Sparsity**: ReLU creates sparse representations — many neurons output exactly zero. This can be beneficial for interpretability and efficiency.

3. **Gradient Flow**: Unlike sigmoid/tanh, ReLU doesn't saturate for positive values. Gradients flow freely in the positive region.

4. **Hardware Efficiency**: ReLU is just a comparison and conditional select — much faster than exp() or tanh().

---

## Properties

- **Non-linear**: Essential for deep learning
- **Monotonic**: Preserves order
- **Piecewise linear**: Two linear pieces joined at zero
- **Non-saturating (positive)**: Gradient = 1 for x > 0
- **Sparse**: Many outputs are exactly zero

---

## The Dying ReLU Problem

If a neuron's input is always negative, it outputs zero and receives zero gradient. The neuron "dies" — it stops learning.

```
           dead neurons (always negative)
                    ↓
input → [weights] → ReLU(x<0) → 0 → no gradient → weights frozen
```

Mitigations:
- **Leaky ReLU**: Small negative slope prevents death
- **Careful initialization**: Start with small positive biases
- **Lower learning rates**: Prevent sudden weight updates

---

## Why ReLU Won

Before ReLU (2010), sigmoid and tanh were standard. But they suffer from:
- **Vanishing gradients**: Saturate at extremes
- **Expensive computation**: exp() is slow
- **Non-sparse outputs**: Everything is nonzero

ReLU solved all three. It became the default activation almost overnight.

```
Before: sigmoid → slow training, vanishing gradients
After:  ReLU → fast training, deep networks possible
```

ReLU enabled the deep learning revolution.

---

## The Simplicity Principle

ReLU is almost embarrassingly simple: "if negative, zero it." Yet this simplicity is its strength:
- Easy to compute
- Easy to differentiate (0 or 1)
- Easy to implement in hardware
- Easy to reason about

Sometimes the right answer is the obvious one.

---

*"Okay. You people sit tight, hold the fort and keep the home fires burning."*
