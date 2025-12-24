# Argmin

*Index of Minimum — Find the winner*

```
┌─────────────────────────────────────────────────────────────┐
│ ARGMIN                                                      │
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
argmin(x) = i such that x[i] = min(x)
```

Returns the INDEX, not the value.

### Prose

Argmin finds the index of the minimum value in a list. Unlike min_pool which returns the minimum value, argmin returns WHERE that minimum is located.

**This is the final step of FrozenDB vector search.**

---

## Visual

```
distances = [12, 5, 8, 3, 15, 7]
                     ↑
                   index 3

argmin(distances) = 3
```

---

## Examples

```python
argmin([5, 3, 8, 1, 4]) = 3   # Index of 1
argmin([10, 20, 30]) = 0      # Index of 10
argmin([100]) = 0             # Single element
argmin([3, 1, 4, 1, 5]) = 1   # First occurrence of min
```

---

## Implementation

### Python

```python
def argmin(x: list) -> int:
    """Index of minimum value."""
    min_idx = 0
    min_val = x[0]
    for i, val in enumerate(x):
        if val < min_val:
            min_val = val
            min_idx = i
    return min_idx

def argmin_k(x: list, k: int) -> list:
    """Indices of k smallest values."""
    indexed = [(val, i) for i, val in enumerate(x)]
    indexed.sort()
    return [i for _, i in indexed[:k]]
```

### C

```c
static inline size_t argmin(const float* x, size_t n) {
    size_t min_idx = 0;
    float min_val = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] < min_val) {
            min_val = x[i];
            min_idx = i;
        }
    }
    return min_idx;
}
```

---

## Relationships

### Built From

Argmin is elemental.

### Used In

- **FrozenDB** — Find closest signature
- **Classification** — Minimum distance classifier
- **Optimization** — Find minimum loss

### See Also

- **[min_pool](min_pool.md)** — Returns value, not index
- **[argmax](argmax.md)** — Index of maximum

---

## Use Cases

1. **FrozenDB Query**: After computing Hamming distances to all signatures, argmin finds the closest match.

2. **Classification**: In nearest-neighbor classification, argmin over distances gives the predicted class.

3. **Beam Search**: Track the best candidates by their indices.

---

## FrozenDB Integration

```python
# The final step of vector search

# Signatures stored in database
signatures = [s0, s1, s2, ..., sn]
metadata = [m0, m1, m2, ..., mn]

# Query
query_sig = encode(query)
distances = [hamming(query_sig, s) for s in signatures]

# Find nearest
idx = argmin(distances)
result = metadata[idx]  # The matched item!
```

---

## Top-K Search

For finding K nearest neighbors instead of just one:

```python
def argmin_k(distances, k):
    """Return indices of k smallest distances."""
    indexed = [(d, i) for i, d in enumerate(distances)]
    indexed.sort()
    return [i for d, i in indexed[:k]]
```

---

*"Finding the winner is just finding the minimum — and knowing where it is."*
