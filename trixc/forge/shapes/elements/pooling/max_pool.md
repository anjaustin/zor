# Max Pool

*Maximum Pooling — Keep the strongest*

```
┌─────────────────────────────────────────────────────────────┐
│ MAX POOL                                                    │
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
max_pool(x) = max(x₁, x₂, ..., xₙ)
```

### Prose

Maximum pooling selects the largest value from a region. It's the "winner takes all" operation — only the strongest activation survives.

---

## Visual

```
Input:  [0.2, 0.8, 0.5, 0.3]
              ↑
         (maximum)
              │
              ▼
Output: 0.8
```

For 2D pooling over spatial regions:
```
┌─────┬─────┐
│ 1.0 │ 0.5 │
├─────┼─────┤     max
│ 0.3 │ 0.8 │  ───────→  1.0
└─────┴─────┘
```

---

## Examples

```python
max_pool([1, 2, 3, 4]) = 4
max_pool([-5, -1, -3]) = -1
max_pool([0.5]) = 0.5
max_pool([7, 7, 7]) = 7
```

---

## Implementation

### Python

```python
def max_pool(x: list) -> float:
    """Maximum pooling."""
    return max(x)
```

### C

```c
static inline float max_pool(const float* x, size_t n) {
    float max_val = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    return max_val;
}
```

---

## Relationships

### Built From

Max pool is elemental.

### Used In

- **CNNs** — Downsample feature maps
- **Global max pooling** — Pool entire spatial dimensions
- **OR-like operations** — Max approximates OR in some contexts

### See Also

- **[avg_pool](avg_pool.md)** — Mean instead of max
- **[min_pool](min_pool.md)** — Minimum instead of max
- **[or](../logic/or.md)** — Logical OR (related concept)

---

## Use Cases

1. **Spatial Downsampling**: In CNNs, max pool reduces spatial dimensions while keeping the strongest features.

2. **Translation Invariance**: Small shifts in input don't change the max much — provides robustness.

3. **Feature Detection**: "Is this feature present anywhere in the region?"

4. **Global Max Pool**: Pool entire feature map to get one value per channel.

---

## Properties

- **Idempotent**: `max_pool([max_pool(x)]) = max_pool(x)`
- **Monotonic**: If all values increase, output increases
- **Selective**: Only one value "wins"
- **Piecewise linear**: Gradient flows only to the maximum

---

## The Gradient Question

Max pooling has an interesting gradient property:

```
∂max(x)/∂xᵢ = { 1  if xᵢ = max(x)
              { 0  otherwise
```

Only the maximum element receives gradient. All others are "masked out." This creates sparse gradients.

---

## Max vs Avg Pooling

| Aspect | Max Pool | Avg Pool |
|--------|----------|----------|
| Output | Strongest | Average |
| Gradient | Sparse (1 element) | Dense (all elements) |
| Sensitivity | High contrast | Smooth |
| Use case | Edge detection | Smooth features |

Max pooling emphasizes strong features. Average pooling smooths everything together.

---

*"Only the strong survive."*
