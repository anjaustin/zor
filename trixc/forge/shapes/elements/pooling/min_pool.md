# Min Pool

*Minimum Pooling — Find the floor*

```
┌─────────────────────────────────────────────────────────────┐
│ MIN POOL                                                    │
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
min_pool(x) = min(x₁, x₂, ..., xₙ)
```

### Prose

Minimum pooling selects the smallest value from a region. The dual of max pooling — finds the "floor" rather than the "ceiling."

---

## Visual

```
Input:  [0.2, 0.8, 0.5, 0.3]
         ↑
     (minimum)
         │
         ▼
Output: 0.2
```

---

## Examples

```python
min_pool([1, 2, 3, 4]) = 1
min_pool([-5, -1, -3]) = -5
min_pool([7, 7, 7]) = 7
min_pool([0.9, 0.1, 0.5]) = 0.1
```

---

## Implementation

### Python

```python
def min_pool(x: list) -> float:
    """Minimum pooling."""
    return min(x)
```

### C

```c
static inline float min_pool(const float* x, size_t n) {
    float min_val = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] < min_val) min_val = x[i];
    }
    return min_val;
}
```

---

## Relationships

### Built From

Min pool is elemental.

### Used In

- **Erosion** — Morphological operation (shrinks features)
- **Conservative estimation** — Worst-case analysis
- **AND-like operations** — Min approximates AND in fuzzy logic

### See Also

- **[max_pool](max_pool.md)** — Maximum instead of minimum
- **[and](../logic/and.md)** — Logical AND (related in fuzzy logic)

---

## Use Cases

1. **Morphological Erosion**: Min pooling over a kernel erodes (shrinks) features.

2. **Worst-Case Analysis**: "What's the minimum confidence in this region?"

3. **Fuzzy AND**: In fuzzy logic, AND(a, b) = min(a, b).

4. **Conservative Bounds**: Lower bound on uncertain quantities.

---

## Properties

- **Idempotent**: `min_pool([min_pool(x)]) = min_pool(x)`
- **Monotonic**: If all values decrease, output decreases
- **Selective**: Only one value determines output
- **Dual to max**: `min(x) = -max(-x)`

---

## The Duality

Min and max are duals:

```
min(x₁, x₂, ..., xₙ) = -max(-x₁, -x₂, ..., -xₙ)
```

Anything you can do with max, you can do with min by negating.

---

## Fuzzy Logic Connection

In fuzzy logic (values in [0, 1]):

| Operation | Fuzzy Implementation |
|-----------|---------------------|
| AND(a, b) | min(a, b) |
| OR(a, b)  | max(a, b) |

This connects pooling operations to logic gates:
- Max pool ≈ OR (any feature present)
- Min pool ≈ AND (all features present)

---

*"The chain is only as strong as its weakest link."*
