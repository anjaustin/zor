# Add

*Binary Addition — The foundation of arithmetic*

```
┌─────────────────────────────────────────────────────────────┐
│ ADD                                                         │
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
add(a, b) = a + b
```

### Prose

Binary addition. The most fundamental arithmetic operation. Takes two numbers, returns their sum.

---

## Visual

```
     ┌─────┐
a ───┤     │
     │  +  ├─── a + b
b ───┤     │
     └─────┘
```

---

## Examples

```
add(0, 0) = 0
add(1, 2) = 3
add(-1, 1) = 0
add(0.5, 0.5) = 1.0
add(3.14, 2.86) = 6.0
```

---

## Implementation

### Python

```python
def add(a: float, b: float) -> float:
    """Binary addition."""
    return a + b
```

### C

```c
static inline float add(float a, float b) {
    return a + b;
}
```

---

## Relationships

### Built From

Add is elemental — the primitive of arithmetic.

### Used In

- **[half_adder](../../compounds/arithmetic/half_adder.md)** — Binary bit addition
- **Residual connections** — x + F(x)
- **Bias addition** — Wx + b
- **Everything** — Addition is universal

### See Also

- **[sub](sub.md)** — Subtraction
- **[mul](mul.md)** — Multiplication

---

## Use Cases

1. **Residual Networks**: The skip connection `x + F(x)` enables very deep networks.

2. **Bias Terms**: Every linear layer adds bias: `y = Wx + b`.

3. **Accumulation**: Summing over sequences, batches, features.

4. **Gradient Updates**: `θ = θ + α∇L` — learning itself is addition.

---

## Properties

- **Commutative**: `a + b = b + a`
- **Associative**: `(a + b) + c = a + (b + c)`
- **Identity**: `a + 0 = a`
- **Inverse**: `a + (-a) = 0`

---

*"Sooner or later, everything comes down to addition."*
