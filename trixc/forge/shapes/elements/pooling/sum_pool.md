# Sum Pool

*Sum Pooling — Total accumulation*

```
┌─────────────────────────────────────────────────────────────┐
│ SUM POOL                                                    │
├─────────────────────────────────────────────────────────────┤
│ Kingdom: Pooling                                            │
│ Type: Elemental                                             │
│ Arity: N-ary (vector → scalar)                              │
│ Frozen: Yes                                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Definition

### Mathematical

```
sum_pool(x) = Σxᵢ = x₁ + x₂ + ... + xₙ
```

### Prose

Sum pooling adds all values together. Unlike average pooling, it doesn't divide — the magnitude grows with the number of elements. Useful when total activation matters more than average.

---

## Visual

```
Input:  [0.2, 0.8, 0.5, 0.3]
           +   +   +   +
        ─────────────────
              sum
        ─────────────────
              │
              ▼
Output: 1.8
```

---

## Examples

```python
sum_pool([1, 2, 3, 4]) = 10
sum_pool([0.5, 0.5]) = 1.0
sum_pool([-1, 1]) = 0
sum_pool([1]) = 1
```

---

## Implementation

### Python

```python
def sum_pool(x: list) -> float:
    """Sum pooling."""
    return sum(x)
```

### C

```c
static inline float sum_pool(const float* x, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum;
}
```

---

## Relationships

### Built From

Sum pool is elemental.

### Used In

- **Attention normalization** — Sum of attention weights should be 1
- **Loss accumulation** — Total loss over batch
- **Counting** — Sum of indicators = count

### See Also

- **[avg_pool](avg_pool.md)** — Sum divided by count
- **[add](../arithmetic/add.md)** — Binary addition

---

## Use Cases

1. **Attention Weights**: Check that softmax outputs sum to 1.

2. **Loss Computation**: Total loss = sum of individual losses.

3. **Population Counting**: Sum of binary indicators counts occurrences.

4. **Energy Computation**: Total "energy" of activations.

---

## Properties

- **Linear**: `sum_pool(αx) = α·sum_pool(x)`
- **Additive**: `sum_pool(x + y) = sum_pool(x) + sum_pool(y)`
- **Size-dependent**: Output scales with input size
- **Gradient**: All elements receive gradient of 1

---

## Sum vs Avg

```
avg_pool(x) = sum_pool(x) / len(x)
```

The difference matters when:
- **Sum**: Magnitude should grow with region size
- **Avg**: Magnitude should be independent of region size

Choose based on what the downstream computation expects.

---

*"The whole is the sum of its parts."*
