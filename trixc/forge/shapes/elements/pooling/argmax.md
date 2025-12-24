# Argmax

*Index of Maximum — Find the best*

```
┌─────────────────────────────────────────────────────────────┐
│ ARGMAX                                                      │
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
argmax(x) = i such that x[i] = max(x)
```

Returns the INDEX, not the value.

### Prose

Argmax finds the index of the maximum value in a list. Used when searching by similarity (higher = better) rather than distance (lower = better).

---

## Visual

```
similarities = [0.2, 0.8, 0.5, 0.9, 0.3]
                          ↑
                       index 3

argmax(similarities) = 3
```

---

## Examples

```python
argmax([1, 5, 3, 8, 2]) = 3   # Index of 8
argmax([10, 20, 30]) = 2      # Index of 30
argmax([100]) = 0             # Single element
argmax([3, 1, 4, 1, 5]) = 4   # Index of 5
```

---

## Implementation

### Python

```python
def argmax(x: list) -> int:
    """Index of maximum value."""
    max_idx = 0
    max_val = x[0]
    for i, val in enumerate(x):
        if val > max_val:
            max_val = val
            max_idx = i
    return max_idx

def argmax_k(x: list, k: int) -> list:
    """Indices of k largest values."""
    indexed = [(val, i) for i, val in enumerate(x)]
    indexed.sort(reverse=True)
    return [i for _, i in indexed[:k]]
```

### C

```c
static inline size_t argmax(const float* x, size_t n) {
    size_t max_idx = 0;
    float max_val = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_idx = i;
        }
    }
    return max_idx;
}
```

---

## Relationships

### Built From

Argmax is elemental.

### Used In

- **Classification** — Predicted class = argmax(logits)
- **Attention** — Hard attention uses argmax
- **Beam search** — Track best candidates

### See Also

- **[max_pool](max_pool.md)** — Returns value, not index
- **[argmin](argmin.md)** — Index of minimum
- **[softmax](../activation/softmax.md)** — Soft version of argmax

---

## Use Cases

1. **Classification**: `predicted_class = argmax(logits)`.

2. **Similarity Search**: When using cosine similarity (higher = more similar), use argmax instead of argmin.

3. **Winner-Take-All**: In competitive learning, the winning neuron is found by argmax.

---

## Argmax vs Softmax

```
logits = [2.0, 5.0, 1.0]

argmax(logits) = 1  # Hard selection: just the winner

softmax(logits) = [0.05, 0.93, 0.02]  # Soft: probability distribution
```

Argmax is discrete (picks one). Softmax is smooth (differentiable).

---

*"The winner takes it all — argmax tells you who."*
