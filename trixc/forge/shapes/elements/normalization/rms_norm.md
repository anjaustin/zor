# RMS Norm

*Root Mean Square Normalization — The simpler stabilizer*

```
┌─────────────────────────────────────────────────────────────┐
│ RMS NORM                                                    │
├─────────────────────────────────────────────────────────────┤
│ Kingdom: Normalization                                      │
│ Type: Elemental                                             │
│ Arity: N-ary (vector → vector)                              │
│ Frozen: Yes                                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Definition

### Mathematical

```
RMSNorm(x) = x / √(mean(x²) + ε)

where:
  RMS = √(mean(x²)) = √((Σxᵢ²)/n)
  ε = small constant for stability
```

### Prose

RMS normalization scales a vector by its root mean square magnitude. Unlike Layer Norm, it doesn't subtract the mean — just rescales. Simpler and often just as effective.

---

## Visual

```
Input:  [3.0, 4.0]
         │
         ▼
    ┌─────────────┐
    │ Compute RMS │ → √((9+16)/2) = √12.5 ≈ 3.54
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │   x / RMS   │ → [3/3.54, 4/3.54]
    └─────────────┘
         │
         ▼
Output: [0.85, 1.13]
```

---

## Examples

```python
rms_norm([3, 4]) ≈ [0.85, 1.13]
rms_norm([1, 1, 1, 1]) = [1, 1, 1, 1]  # Already unit RMS
rms_norm([2, 0, 0, 0]) ≈ [2, 0, 0, 0]  # Sparse case
```

---

## Implementation

### Python

```python
import math

def rms_norm(x: list, eps: float = 1e-5) -> list:
    """RMS normalization."""
    n = len(x)
    rms = math.sqrt(sum(xi * xi for xi in x) / n + eps)
    return [xi / rms for xi in x]
```

### C

```c
static inline void rms_norm(float* x, size_t n, float eps) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / (float)n + eps);
    for (size_t i = 0; i < n; i++) {
        x[i] /= rms;
    }
}
```

---

## Relationships

### Built From

RMS Norm is elemental.

### Used In

- **LLaMA** — Uses RMSNorm instead of LayerNorm
- **T5** — Also uses RMSNorm
- **Modern transformers** — Increasingly popular

### See Also

- **[layer_norm](layer_norm.md)** — Full normalization with mean subtraction

---

## Use Cases

1. **LLaMA Architecture**: Meta's LLaMA uses RMSNorm for efficiency.

2. **Efficient Transformers**: One less operation (no mean subtraction).

3. **Pre-Norm Position**: Applied before attention and FFN blocks.

---

## Properties

- **Output RMS**: 1 (unit magnitude)
- **Mean**: Not necessarily 0 (unlike LayerNorm)
- **Scale invariant**: `rms_norm(α·x) = sign(α)·rms_norm(x)`
- **Simpler gradient**: Fewer operations = simpler backprop

---

## RMS Norm vs Layer Norm

| Aspect | RMS Norm | Layer Norm |
|--------|----------|------------|
| Mean subtraction | No | Yes |
| Output mean | Unchanged | 0 |
| Complexity | Simpler | More complex |
| Used in | LLaMA, T5 | GPT, BERT |
| Performance | Often equivalent | Standard |

The key insight: mean subtraction often doesn't matter much. RMSNorm removes it and works fine.

---

## Why It Works

RMSNorm controls the *magnitude* of activations without shifting their *center*. For many tasks, this is sufficient:

- Prevents explosion (large values get scaled down)
- Prevents vanishing (small values get scaled up)
- Preserves relative relationships

The mean subtraction in LayerNorm isn't always necessary.

---

*"Simpler is often better. RMSNorm proves it."*
