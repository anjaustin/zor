# XOR

*Exclusive OR — The shape that started it all*

```
┌─────────────────────────────────────────────────────────────┐
│ XOR                                                         │
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
XOR(a, b) = a ⊕ b = (a ∧ ¬b) ∨ (¬a ∧ b)
```

For continuous inputs in [0, 1]:
```
XOR(a, b) = a + b - 2·a·b
```

### Prose

XOR outputs 1 when exactly one of its inputs is 1. It's the "one or the other, but not both" operation. In continuous form, it measures disagreement between two signals.

---

## Visual

```
     ┌─────┐
a ───┤     │
     │ ⊕   ├─── a ⊕ b
b ───┤     │
     └─────┘

Standard logic gate symbol:

    a ──┐
        )>o── output
    b ──┘
```

The distinctive curved back differentiates XOR from OR.

---

## Truth Table

| a | b | a ⊕ b |
|---|---|-------|
| 0 | 0 | 0     |
| 0 | 1 | 1     |
| 1 | 0 | 1     |
| 1 | 1 | 0     |

---

## Examples

### Discrete

```
XOR(0, 0) = 0   # Neither → 0
XOR(0, 1) = 1   # One → 1
XOR(1, 0) = 1   # One → 1
XOR(1, 1) = 0   # Both → 0 (cancels out)
```

### Continuous

```
XOR(0.0, 0.0) = 0.0
XOR(0.0, 1.0) = 1.0
XOR(0.5, 0.5) = 0.5   # Maximum uncertainty
XOR(1.0, 1.0) = 0.0
XOR(0.3, 0.7) = 0.58  # Partial disagreement
```

---

## Implementation

### Python

```python
def xor(a: float, b: float) -> float:
    """Differentiable XOR for continuous inputs in [0, 1]."""
    return a + b - 2 * a * b

def xor_discrete(a: int, b: int) -> int:
    """Discrete XOR for boolean inputs."""
    return a ^ b
```

### C

```c
static inline float xor_shape(float a, float b) {
    return a + b - 2.0f * a * b;
}

static inline int xor_discrete(int a, int b) {
    return a ^ b;
}
```

---

## Relationships

### Built From

XOR is elemental — it cannot be decomposed within Geocadesia.

(Mathematically: `(a AND NOT b) OR (NOT a AND b)`, but we treat XOR as primitive.)

### Used In

- **[half_adder](../../compounds/arithmetic/half_adder.md)** — XOR computes the sum bit
- **[full_adder](../../compounds/arithmetic/full_adder.md)** — Two XORs compute sum with carry-in
- **Parity checking** — XOR chain detects odd number of 1s
- **Encryption** — XOR with key is reversible

### See Also

- **[XNOR](xnor.md)** — NOT XOR, equivalence gate
- **[AND](and.md)** — Together with XOR, forms half adder
- **[OR](or.md)** — Logical disjunction

---

## Use Cases

1. **Binary Arithmetic**: XOR computes the sum bit in addition without considering carry. It's the heart of the half adder.

2. **Neural XOR Problem**: The classic demonstration that single-layer perceptrons cannot learn XOR. Solving XOR was a milestone for multilayer networks.

3. **Parity Detection**: XORing all bits gives 1 if odd number of 1s, 0 if even. Used in error detection.

4. **Encryption**: `plaintext XOR key = ciphertext`. Applying XOR again with the same key recovers plaintext.

5. **Bit Manipulation**: Swapping without temporary: `a ^= b; b ^= a; a ^= b;`

---

## Properties

- **Commutative**: `a ⊕ b = b ⊕ a`
- **Associative**: `(a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)`
- **Self-inverse**: `a ⊕ a = 0`
- **Identity**: `a ⊕ 0 = a`
- **Nilpotent with self**: `a ⊕ a = 0` (XOR cancels itself)

---

## The XOR Philosophy

XOR is perhaps the most philosophically interesting gate. It represents *difference*. Two identical inputs produce 0 — sameness negates. Two different inputs produce 1 — difference affirms.

In the continuous relaxation, XOR measures the *degree of disagreement* between two signals. When both signals are 0.5, XOR outputs 0.5 — maximum uncertainty, maximum disagreement with itself.

XOR is the shape of distinction.

---

*"It's all in the reflexes."*
