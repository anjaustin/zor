# AND

*Logical Conjunction — Both must agree*

```
┌─────────────────────────────────────────────────────────────┐
│ AND                                                         │
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
AND(a, b) = a ∧ b = min(a, b)  [discrete]
AND(a, b) = a · b              [continuous/differentiable]
```

### Prose

AND outputs 1 only when both inputs are 1. It's the "both must be true" operation. In continuous form, it computes the product — the degree to which both signals are active.

---

## Visual

```
     ┌─────┐
a ───┤     │
     │  ∧  ├─── a ∧ b
b ───┤     │
     └─────┘

Standard logic gate symbol:

    a ──┐
        D── output
    b ──┘
```

The flat back distinguishes AND from OR.

---

## Truth Table

| a | b | a ∧ b |
|---|---|-------|
| 0 | 0 | 0     |
| 0 | 1 | 0     |
| 1 | 0 | 0     |
| 1 | 1 | 1     |

---

## Examples

### Discrete

```
AND(0, 0) = 0   # Neither → 0
AND(0, 1) = 0   # One missing → 0
AND(1, 0) = 0   # One missing → 0
AND(1, 1) = 1   # Both → 1
```

### Continuous

```
AND(0.0, 0.0) = 0.0
AND(0.5, 0.5) = 0.25   # Partial agreement
AND(0.8, 0.9) = 0.72   # Strong agreement
AND(1.0, 1.0) = 1.0
AND(0.0, 1.0) = 0.0    # One zero kills it
```

---

## Implementation

### Python

```python
def and_gate(a: float, b: float) -> float:
    """Differentiable AND for continuous inputs in [0, 1]."""
    return a * b

def and_discrete(a: int, b: int) -> int:
    """Discrete AND for boolean inputs."""
    return a & b
```

### C

```c
static inline float and_shape(float a, float b) {
    return a * b;
}

static inline int and_discrete(int a, int b) {
    return a & b;
}
```

---

## Relationships

### Built From

AND is elemental.

### Used In

- **[half_adder](../../compounds/arithmetic/half_adder.md)** — AND computes the carry bit
- **[full_adder](../../compounds/arithmetic/full_adder.md)** — ANDs contribute to carry logic
- **[NAND](nand.md)** — NOT(AND) is a universal gate
- **Attention masking** — AND with mask zeros out positions

### See Also

- **[OR](or.md)** — Logical disjunction
- **[NAND](nand.md)** — Negated AND, universal gate
- **[XOR](xor.md)** — Exclusive disjunction

---

## Use Cases

1. **Carry Generation**: In binary addition, carry occurs when both bits are 1. AND detects this.

2. **Feature Conjunction**: "Is this pixel red AND bright?" Product of activations answers conjunction queries.

3. **Masking**: AND with a mask (0 or 1) selectively zeros elements. Foundation of attention masks.

4. **Gating**: In gated networks (LSTM, GRU), AND-like operations control information flow.

---

## Properties

- **Commutative**: `a ∧ b = b ∧ a`
- **Associative**: `(a ∧ b) ∧ c = a ∧ (b ∧ c)`
- **Identity**: `a ∧ 1 = a`
- **Annihilator**: `a ∧ 0 = 0`
- **Idempotent**: `a ∧ a = a`

---

*"You know what ol' Jack Burton says at a time like this?"*
