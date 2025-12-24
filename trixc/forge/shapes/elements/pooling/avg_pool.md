# Avg Pool

*Average Pooling — The democratic summary*

```
┌─────────────────────────────────────────────────────────────┐
│ AVG POOL                                                    │
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
avg_pool(x) = (1/n) · Σxᵢ = mean(x)
```

### Prose

Average pooling computes the mean of all values in a region. Every element contributes equally — it's the democratic alternative to max pooling's "winner takes all."

---

## Visual

```
Input:  [0.2, 0.8, 0.5, 0.3]
           ↓   ↓   ↓   ↓
        ─────────────────
              mean
        ─────────────────
              │
              ▼
Output: 0.45
```

---

## Examples

```python
avg_pool([1, 2, 3, 4]) = 2.5
avg_pool([0, 0, 0, 10]) = 2.5
avg_pool([5, 5, 5, 5]) = 5.0
avg_pool([-2, 2]) = 0.0
```

---

## Implementation

### Python

```python
def avg_pool(x: list) -> float:
    """Average pooling."""
    return sum(x) / len(x)
```

### C

```c
static inline float avg_pool(const float* x, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum / (float)n;
}
```

---

## Relationships

### Built From

Avg pool is elemental (sum + divide, but conceptually atomic).

### Used In

- **Global Average Pooling** — Replace FC layers in CNNs
- **Feature aggregation** — Summarize spatial information
- **Embedding pooling** — Combine token embeddings

### See Also

- **[max_pool](max_pool.md)** — Maximum instead of mean
- **[sum_pool](sum_pool.md)** — Sum without division

---

## Use Cases

1. **Global Average Pooling**: In modern CNNs, GAP replaces fully-connected layers before classification.

2. **Sentence Embeddings**: Average word embeddings to get sentence embedding.

3. **Smooth Downsampling**: Reduce resolution while maintaining overall intensity.

4. **Anti-aliasing**: Average pooling smooths out high-frequency noise.

---

## Properties

- **Linear**: `avg_pool(αx) = α·avg_pool(x)`
- **Shift equivariant**: `avg_pool(x + c) = avg_pool(x) + c`
- **Bounded**: `min(x) ≤ avg_pool(x) ≤ max(x)`
- **Smooth gradient**: All elements receive equal gradient (1/n)

---

## The Gradient

Average pooling distributes gradient equally:

```
∂avg(x)/∂xᵢ = 1/n  for all i
```

Every element contributes to the output, so every element receives gradient. This is "dense" compared to max pooling's sparse gradient.

---

## Global Average Pooling (GAP)

A powerful pattern in CNNs:

```
Feature maps: [batch, channels, height, width]
                          ↓
           Global Average Pooling (per channel)
                          ↓
Output:       [batch, channels]
                          ↓
               Classification head
```

GAP reduces spatial dimensions entirely, giving one value per channel. This:
- Removes the need for large FC layers
- Reduces parameters dramatically
- Provides spatial invariance

---

*"Everyone gets a vote."*
