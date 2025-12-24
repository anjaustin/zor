# NOT

*Logical Negation — The inverter*

```
┌─────────────────────────────────────────────────────────────┐
│ NOT                                                         │
├─────────────────────────────────────────────────────────────┤
│ Kingdom: Logic                                              │
│ Type: Elemental                                             │
│ Arity: Unary                                                │
│ Frozen: Yes                                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Definition

### Mathematical

```
NOT(a) = ¬a = 1 - a
```

### Prose

NOT inverts its input. 1 becomes 0, 0 becomes 1. In continuous form, it reflects the value around 0.5 — what was high becomes low, what was low becomes high.

---

## Visual

```
     ┌─────┐
a ───┤  ¬  ├─── ¬a
     └─────┘

Standard logic gate symbol:

    a ──▷o── output

The bubble (o) indicates inversion.
```

---

## Truth Table

| a | ¬a |
|---|-----|
| 0 | 1   |
| 1 | 0   |

---

## Examples

### Discrete

```
NOT(0) = 1   # False → True
NOT(1) = 0   # True → False
```

### Continuous

```
NOT(0.0) = 1.0
NOT(0.25) = 0.75
NOT(0.5) = 0.5   # Fixed point
NOT(0.75) = 0.25
NOT(1.0) = 0.0
```

---

## Implementation

### Python

```python
def not_gate(a: float) -> float:
    """Differentiable NOT for continuous inputs in [0, 1]."""
    return 1.0 - a

def not_discrete(a: int) -> int:
    """Discrete NOT for boolean inputs."""
    return 1 - a  # or: return not a
```

### C

```c
static inline float not_shape(float a) {
    return 1.0f - a;
}

static inline int not_discrete(int a) {
    return !a;
}
```

---

## Relationships

### Built From

NOT is elemental — the simplest operation.

### Used In

- **[NAND](nand.md)** — NOT(AND)
- **[NOR](nor.md)** — NOT(OR)
- **[XNOR](xnor.md)** — NOT(XOR)
- **De Morgan transformations** — Converting AND↔OR

### See Also

- **[AND](and.md)**, **[OR](or.md)** — Binary logic gates
- **[NAND](nand.md)**, **[NOR](nor.md)** — Negated gates

---

## Use Cases

1. **Inversion**: The fundamental operation of flipping a signal.

2. **Complement**: In probability, P(NOT A) = 1 - P(A).

3. **Gate Construction**: NOT combined with AND gives NAND, a universal gate.

4. **Residual Computation**: "How much is left?" = 1 - (how much is used).

---

## Properties

- **Involution**: `NOT(NOT(a)) = a`
- **Fixed Point**: `NOT(0.5) = 0.5`
- **Gradient**: `d(NOT)/da = -1` (constant negative slope)

---

## The Simplicity of NOT

NOT is the simplest shape: a single subtraction from 1. Yet it's essential. Without NOT, we couldn't build NAND or NOR (the universal gates). Without NOT, De Morgan's laws wouldn't work.

NOT is proof that simplicity enables complexity.

---

*"Just remember what ol' Jack Burton does when the earth quakes..."*
