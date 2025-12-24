# Mul

*Binary Multiplication — Scaling and combination*

```
┌─────────────────────────────────────────────────────────────┐
│ MUL                                                         │
├─────────────────────────────────────────────────────────────┤
│ Kingdom: Arithmetic                                         │
│ Type: Elemental                                             │
│ Arity: Binary                                               │
│ Frozen: Yes                                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Definition

### Mathematical

```
mul(a, b) = a × b = a · b
```

### Prose

Binary multiplication. Scales one value by another. The foundation of linear transformations and the continuous form of AND.

---

## Visual

```
     ┌─────┐
a ───┤     │
     │  ×  ├─── a × b
b ───┤     │
     └─────┘
```

---

## Examples

```
mul(2, 3) = 6
mul(0, 100) = 0
mul(1, x) = x
mul(0.5, 0.5) = 0.25
mul(-1, 5) = -5
```

---

## Implementation

### Python

```python
def mul(a: float, b: float) -> float:
    """Binary multiplication."""
    return a * b
```

### C

```c
static inline float mul(float a, float b) {
    return a * b;
}
```

---

## Relationships

### Built From

Mul is elemental.

### Used In

- **[and](../logic/and.md)** — AND is multiplication for [0,1]
- **Matrix multiplication** — Repeated mul + add
- **Attention scaling** — `Q·K^T / √d`
- **Gating** — `gate × value`

### See Also

- **[and](../logic/and.md)** — Logical AND = multiplication
- **[add](add.md)** — Addition

---

## Use Cases

1. **Linear Layers**: `y = W × x` — the core of neural networks.

2. **Gating**: In LSTMs, GRUs, and attention: `output = gate × value`.

3. **Scaling**: Learning rates, normalization factors, temperature.

4. **Attention Scores**: `scores = Q × K^T`.

---

## Properties

- **Commutative**: `a × b = b × a`
- **Associative**: `(a × b) × c = a × (b × c)`
- **Identity**: `a × 1 = a`
- **Zero**: `a × 0 = 0`
- **Distributive**: `a × (b + c) = a×b + a×c`

---

## The AND Connection

In the Logic Kingdom, AND is defined as multiplication:

```
AND(a, b) = a × b
```

When a and b are in [0, 1]:
- `AND(1, 1) = 1 × 1 = 1`
- `AND(1, 0) = 1 × 0 = 0`
- `AND(0, 0) = 0 × 0 = 0`

Multiplication is the continuous extension of conjunction.

---

*"Multiplication is repeated addition. And gating. And scaling. And everything."*
