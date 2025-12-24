# XNOR

*NOT XOR — The equivalence gate*

```
┌─────────────────────────────────────────────────────────────┐
│ XNOR                                                        │
├─────────────────────────────────────────────────────────────┤
│ Kingdom: Logic                                              │
│ Type: Elemental                                             │
│ Arity: Binary                                               │
│ Frozen: Yes                                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Definition

### Mathematical

```
XNOR(a, b) = ¬(a ⊕ b) = 1 - (a + b - 2·a·b)
           = 1 - a - b + 2·a·b
```

Also known as:
```
XNOR(a, b) = (a ∧ b) ∨ (¬a ∧ ¬b)   # Both same
```

### Prose

XNOR outputs 1 when both inputs are the same (both 0 or both 1). It's the equivalence operation — "do these agree?" In continuous form, it measures similarity between two signals.

---

## Visual

```
     ┌─────┐
a ───┤     │
     │ ⊙   ├─── a ⊙ b
b ───┤     │
     └─────┘

Standard logic gate symbol:

    a ──┐
        )>o── output  (XOR with bubble)
    b ──┘
```

---

## Truth Table

| a | b | XNOR |
|---|---|------|
| 0 | 0 | 1    |
| 0 | 1 | 0    |
| 1 | 0 | 0    |
| 1 | 1 | 1    |

Notice: XNOR is the inverse of XOR.

---

## Examples

### Discrete

```
XNOR(0, 0) = 1   # Both 0 → same → 1
XNOR(0, 1) = 0   # Different → 0
XNOR(1, 0) = 0   # Different → 0
XNOR(1, 1) = 1   # Both 1 → same → 1
```

### Continuous

```
XNOR(0.0, 0.0) = 1.0   # Perfect agreement at 0
XNOR(1.0, 1.0) = 1.0   # Perfect agreement at 1
XNOR(0.5, 0.5) = 0.5   # Uncertain agreement
XNOR(0.0, 1.0) = 0.0   # Perfect disagreement
XNOR(0.3, 0.3) = 0.82  # Partial agreement
```

---

## Implementation

### Python

```python
def xnor(a: float, b: float) -> float:
    """Differentiable XNOR for continuous inputs in [0, 1]."""
    return 1.0 - a - b + 2.0 * a * b

def xnor_discrete(a: int, b: int) -> int:
    """Discrete XNOR for boolean inputs."""
    return 1 - (a ^ b)
```

### C

```c
static inline float xnor_shape(float a, float b) {
    return 1.0f - a - b + 2.0f * a * b;
}

static inline int xnor_discrete(int a, int b) {
    return !(a ^ b);
}
```

---

## Relationships

### Built From

Conceptually: `NOT(XOR(a, b))`

### Used In

- **Equality comparison** — Are these bits the same?
- **Binary neural networks** — XNOR is the core operation in XNOR-Net
- **Similarity metrics** — Measures agreement between binary features

### See Also

- **[XOR](xor.md)** — XNOR = NOT(XOR)
- **[AND](and.md)** — Part of equivalence: (a AND b)
- **[NOR](nor.md)** — Part of equivalence: (NOT a AND NOT b)

---

## Use Cases

1. **Equality Testing**: XNOR tells you if two bits match. Chained across all bits, it detects if two numbers are equal.

2. **XNOR-Net**: Binary neural networks use XNOR as the core operation, replacing expensive multiply-accumulate with simple bit operations.

3. **Similarity**: In binary representations, XNOR measures how many positions agree. Pop-count of XNOR gives Hamming similarity.

4. **Parity Check Inverse**: While XOR gives parity, XNOR gives anti-parity — 1 when even number of 1s.

---

## Properties

- **Commutative**: `XNOR(a, b) = XNOR(b, a)`
- **Reflexive**: `XNOR(a, a) = 1` (anything equals itself)
- **Symmetric with XOR**: `XNOR = NOT(XOR)`
- **Identity**: `XNOR(a, 1) = a`, `XNOR(a, 0) = NOT(a)`

---

## XNOR-Net: Binary Neural Networks

XNOR-Net (2016) revolutionized efficient inference by observing that:

```
W · X ≈ XNOR(sign(W), sign(X)) · scale
```

When weights and activations are binarized to {-1, +1}, convolution reduces to XNOR and popcount. This enables:
- 32× memory reduction
- 58× speedup on CPU
- Runs on embedded devices

XNOR isn't just a logic gate — it's the foundation of binary deep learning.

---

## The Equivalence Philosophy

XOR measures difference. XNOR measures sameness.

They're two sides of the same coin — literally complements. Where XOR asks "do these disagree?", XNOR asks "do these agree?"

In a world of patterns and matching, XNOR is the shape of recognition.

---

*"Hollow? Hollow."*
