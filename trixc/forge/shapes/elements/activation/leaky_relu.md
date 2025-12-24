# Leaky ReLU

*ReLU with a safety valve — No dying neurons*

```
┌─────────────────────────────────────────────────────────────┐
│ LEAKY RELU                                                  │
├─────────────────────────────────────────────────────────────┤
│ Kingdom: Activation                                         │
│ Type: Elemental                                             │
│ Arity: Unary                                                │
│ Frozen: Yes (or Partial if α is learned)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Definition

### Mathematical

```
LeakyReLU(x) = { x      if x > 0
              { αx     if x ≤ 0

where α is typically 0.01 (1% slope for negatives)
```

### Prose

Leaky ReLU is ReLU with a small negative slope. Instead of completely zeroing negative values, it allows a small gradient through. This prevents "dying neurons" — units that get stuck outputting zero.

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
      │  ╱  (small slope α)
  -0.2┼╱
      │
      └─┼──┼──┼──┼──┼─ input
       -2 -1  0  1  2

The negative side has slope α instead of 0.
```

---

## Examples

```python
# With α = 0.01 (default)
leaky_relu(2.0)  = 2.0
leaky_relu(0.0)  = 0.0
leaky_relu(-1.0) = -0.01
leaky_relu(-100) = -1.0

# With α = 0.1
leaky_relu(-1.0, α=0.1) = -0.1
```

---

## Implementation

### Python

```python
def leaky_relu(x: float, alpha: float = 0.01) -> float:
    """Leaky ReLU."""
    return x if x > 0 else alpha * x
```

### C

```c
static inline float leaky_relu(float x, float alpha) {
    return x > 0.0f ? x : alpha * x;
}
```

---

## Relationships

### Built From

Leaky ReLU is elemental (though conceptually extends ReLU).

### Used In

- **GANs** — Often preferred over ReLU
- **Deep networks** — When dying ReLU is a problem
- **Negative-sensitive tasks** — When negative features matter

### See Also

- **[relu](relu.md)** — The original (α = 0)
- **[GELU](gelu.md)** — Smooth alternative
- **PReLU** — Learnable α (parameterized)

---

## Use Cases

1. **GAN Discriminators**: Leaky ReLU is standard in discriminator networks.

2. **Deep Networks**: When ReLU neurons keep dying, switch to Leaky ReLU.

3. **Feature Preservation**: When negative activations carry information.

---

## Properties

- **Piecewise linear**: Two linear pieces with different slopes
- **Non-zero gradient everywhere**: No vanishing gradient for negatives
- **Reduces to ReLU**: When α = 0
- **Becomes linear**: When α = 1

---

## The Dying ReLU Problem

With standard ReLU:
```
If weights push all inputs negative → output always 0
→ gradient always 0 → weights never update → neuron "dies"
```

With Leaky ReLU:
```
If weights push all inputs negative → output is αx
→ gradient is α (small but nonzero) → weights can still update
→ neuron can recover
```

The "leak" is a safety valve that keeps gradients flowing.

---

## Choosing α

| α Value | Effect |
|---------|--------|
| 0.0 | Standard ReLU |
| 0.01 | Default Leaky ReLU |
| 0.1-0.3 | More information from negatives |
| 1.0 | Linear (no nonlinearity) |

Common choices:
- **0.01**: Conservative, widely used
- **0.2**: Used in some GAN architectures
- **Learned**: PReLU learns α during training

---

## PReLU: The Learnable Version

PReLU (Parametric ReLU) makes α a learnable parameter:

```
PReLU(x) = { x      if x > 0
           { αx     if x ≤ 0   where α is learned
```

This is a "partial" frozen shape — the ReLU structure is frozen, but α is parameterized.

---

*"A small leak prevents the dam from breaking."*
