# OR

*Logical Disjunction — Either suffices*

```
┌─────────────────────────────────────────────────────────────┐
│ OR                                                          │
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
OR(a, b) = a ∨ b = max(a, b)  [discrete]
OR(a, b) = a + b - a·b        [continuous/differentiable]
```

The continuous form is the probabilistic OR (inclusion-exclusion).

### Prose

OR outputs 1 when at least one input is 1. It's the "either or both" operation. In continuous form, it represents the probability that at least one event occurs.

---

## Visual

```
     ┌─────┐
a ───┤     │
     │  ∨  ├─── a ∨ b
b ───┤     │
     └─────┘

Standard logic gate symbol:

    a ──┐
        )── output
    b ──┘
```

The curved back (without the XOR line) indicates OR.

---

## Truth Table

| a | b | a ∨ b |
|---|---|-------|
| 0 | 0 | 0     |
| 0 | 1 | 1     |
| 1 | 0 | 1     |
| 1 | 1 | 1     |

---

## Examples

### Discrete

```
OR(0, 0) = 0   # Neither → 0
OR(0, 1) = 1   # One → 1
OR(1, 0) = 1   # One → 1
OR(1, 1) = 1   # Both → 1
```

### Continuous

```
OR(0.0, 0.0) = 0.0
OR(0.5, 0.5) = 0.75   # Higher than AND
OR(0.8, 0.9) = 0.98   # Nearly certain
OR(1.0, 1.0) = 1.0
OR(0.0, 1.0) = 1.0    # One suffices
```

---

## Implementation

### Python

```python
def or_gate(a: float, b: float) -> float:
    """Differentiable OR for continuous inputs in [0, 1]."""
    return a + b - a * b

def or_discrete(a: int, b: int) -> int:
    """Discrete OR for boolean inputs."""
    return a | b
```

### C

```c
static inline float or_shape(float a, float b) {
    return a + b - a * b;
}

static inline int or_discrete(int a, int b) {
    return a | b;
}
```

---

## Relationships

### Built From

OR is elemental.

### Used In

- **[full_adder](../../compounds/arithmetic/full_adder.md)** — OR combines partial carries
- **[NOR](nor.md)** — NOT(OR) is a universal gate
- **Max pooling** — Related to OR (takes maximum)
- **Fuzzy logic** — OR is "maximum" in fuzzy systems

### See Also

- **[AND](and.md)** — Logical conjunction
- **[NOR](nor.md)** — Negated OR, universal gate
- **[XOR](xor.md)** — Exclusive disjunction

---

## Use Cases

1. **Carry Propagation**: In full adders, OR combines carry contributions.

2. **Feature Disjunction**: "Is this pixel red OR blue?" Detects presence of any matching feature.

3. **Probabilistic Union**: P(A or B) = P(A) + P(B) - P(A and B). The continuous OR formula.

4. **Fuzzy Max**: In fuzzy logic, OR often means taking the maximum membership value.

---

## Properties

- **Commutative**: `a ∨ b = b ∨ a`
- **Associative**: `(a ∨ b) ∨ c = a ∨ (b ∨ c)`
- **Identity**: `a ∨ 0 = a`
- **Annihilator**: `a ∨ 1 = 1`
- **Idempotent**: `a ∨ a = a`

---

## De Morgan's Laws

OR and AND are duals:

```
NOT(a OR b) = (NOT a) AND (NOT b)
NOT(a AND b) = (NOT a) OR (NOT b)
```

This duality is fundamental to logic and appears throughout computation.

---

*"When some wild-eyed, eight-foot-tall neural network grabs your training data..."*
