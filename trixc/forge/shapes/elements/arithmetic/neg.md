# Neg

*Unary Negation — Flipping the sign*

```
┌─────────────────────────────────────────────────────────────┐
│ NEG                                                         │
├─────────────────────────────────────────────────────────────┤
│ Kingdom: Arithmetic                                         │
│ Type: Elemental                                             │
│ Arity: Unary                                                │
│ Frozen: Yes                                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Definition

### Mathematical

```
neg(a) = -a
```

### Prose

Unary negation. Flips the sign of a number. Positive becomes negative, negative becomes positive.

---

## Visual

```
     ┌─────┐
a ───┤  -  ├─── -a
     └─────┘
```

---

## Examples

```
neg(5) = -5
neg(-3) = 3
neg(0) = 0
neg(0.5) = -0.5
```

---

## Implementation

### Python

```python
def neg(a: float) -> float:
    """Unary negation."""
    return -a
```

### C

```c
static inline float neg(float a) {
    return -a;
}
```

---

## Relationships

### Built From

Neg is elemental.

### Used In

- **Subtraction** — `a - b = a + neg(b)`
- **Gradient descent** — `θ - α∇L = θ + neg(α∇L)`
- **Reflection** — Negation reflects across zero

### See Also

- **[sub](sub.md)** — Subtraction uses negation
- **[not](../logic/not.md)** — Logical negation (different domain)

---

## Use Cases

1. **Gradient Descent**: We negate gradients to descend (minimize), not ascend.

2. **Subtraction**: Internally, `a - b` is often `a + (-b)`.

3. **Sign Flip**: Converting between conventions (e.g., rewards to costs).

4. **Reflection**: Negation reflects a value across zero.

---

## Properties

- **Involution**: `neg(neg(a)) = a`
- **Fixed point**: `neg(0) = 0`
- **Distributive over add**: `neg(a + b) = neg(a) + neg(b)`

---

## The Arithmetic NOT

Negation in arithmetic is analogous to NOT in logic:

| Logic (Boolean) | Arithmetic (Real) |
|-----------------|-------------------|
| NOT(0) = 1      | neg(0) = 0        |
| NOT(1) = 0      | neg(1) = -1       |

They're different operations on different domains, but both represent "inversion" in their respective worlds.

---

*"To descend, negate the gradient."*
