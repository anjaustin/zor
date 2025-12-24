# Sub

*Binary Subtraction — The difference between things*

```
┌─────────────────────────────────────────────────────────────┐
│ SUB                                                         │
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
sub(a, b) = a - b
```

### Prose

Binary subtraction. Returns the difference between two numbers. Order matters — subtraction is not commutative.

---

## Visual

```
     ┌─────┐
a ───┤     │
     │  -  ├─── a - b
b ───┤     │
     └─────┘
```

---

## Examples

```
sub(5, 3) = 2
sub(3, 5) = -2
sub(0, 0) = 0
sub(1.0, 0.5) = 0.5
```

---

## Implementation

### Python

```python
def sub(a: float, b: float) -> float:
    """Binary subtraction."""
    return a - b
```

### C

```c
static inline float sub(float a, float b) {
    return a - b;
}
```

---

## Relationships

### Built From

Sub is elemental.

### Used In

- **Normalization** — `(x - μ)` centers data
- **Loss computation** — `(pred - target)`
- **Gradient computation** — Finite differences

### See Also

- **[add](add.md)** — Addition
- **[neg](neg.md)** — Negation

---

## Use Cases

1. **Mean Subtraction**: Centering data by subtracting the mean.

2. **Error Computation**: `error = prediction - target`

3. **Relative Position**: In attention, relative positions are computed via subtraction.

4. **Differentiation**: Numerical derivatives use subtraction.

---

## Properties

- **Not commutative**: `a - b ≠ b - a` (unless a = b)
- **Not associative**: `(a - b) - c ≠ a - (b - c)`
- **Identity**: `a - 0 = a`
- **Self-inverse**: `a - a = 0`

---

*"The space between is measured by subtraction."*
