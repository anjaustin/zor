# Layer Norm

*Normalize across features — The stabilizer*

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER NORM                                                  │
├─────────────────────────────────────────────────────────────┤
│ Kingdom: Normalization                                      │
│ Type: Elemental                                             │
│ Arity: N-ary (vector → vector)                              │
│ Frozen: Yes (without affine parameters)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Definition

### Mathematical

```
LayerNorm(x) = (x - μ) / √(σ² + ε)

where:
  μ = mean(x)
  σ² = var(x)
  ε = small constant for stability
```

### Prose

Layer normalization centers and scales a vector to have zero mean and unit variance. It operates across the feature dimension, making each sample independent. Essential for transformer stability.

---

## Visual

```
Input:  [2.0, 4.0, 6.0, 8.0]
         │
         ▼
    ┌─────────────┐
    │ Compute μ   │ → μ = 5.0
    │ Compute σ²  │ → σ² = 5.0
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │  (x - μ)    │ → [-3, -1, 1, 3]
    │  ─────────  │
    │  √(σ² + ε)  │ → / 2.236
    └─────────────┘
         │
         ▼
Output: [-1.34, -0.45, 0.45, 1.34]
```

---

## Examples

```python
layer_norm([1, 2, 3, 4]) ≈ [-1.34, -0.45, 0.45, 1.34]
layer_norm([0, 0, 0, 0]) = [0, 0, 0, 0]  # Zero variance case
layer_norm([1, 1, 1, 1]) = [0, 0, 0, 0]  # Constant input
```

---

## Implementation

### Python

```python
import math

def layer_norm(x: list, eps: float = 1e-5) -> list:
    """Layer normalization (frozen, no affine)."""
    n = len(x)
    mean = sum(x) / n
    var = sum((xi - mean) ** 2 for xi in x) / n
    inv_std = 1.0 / math.sqrt(var + eps)
    return [(xi - mean) * inv_std for xi in x]
```

### C

```c
static inline void layer_norm(float* x, size_t n, float eps) {
    float mean = 0.0f;
    for (size_t i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;

    float var = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= (float)n;

    float inv_std = 1.0f / sqrtf(var + eps);
    for (size_t i = 0; i < n; i++) {
        x[i] = (x[i] - mean) * inv_std;
    }
}
```

---

## Relationships

### Built From

Layer norm is elemental (composed of mean, variance, division, but treated as atomic).

### Used In

- **Transformers** — Before or after attention/FFN blocks
- **RNNs** — Per-timestep normalization
- **Any deep network** — Training stability

### See Also

- **[rms_norm](rms_norm.md)** — Simpler variant without mean subtraction
- **Batch Norm** — Normalizes across batch dimension (not in Geocadesia)

---

## Use Cases

1. **Transformer Pre-Norm**: Modern transformers apply LayerNorm before attention and FFN.

2. **Post-Norm**: Original transformer applied LayerNorm after residual addition.

3. **Training Stability**: Without normalization, activations can explode or vanish.

4. **Gradient Flow**: Normalization helps maintain healthy gradients through depth.

---

## Properties

- **Output mean**: 0 (centered)
- **Output variance**: 1 (unit scale)
- **Invariant to shift**: `layer_norm(x + c) = layer_norm(x)`
- **Invariant to scale**: `layer_norm(α·x) = layer_norm(x)` (for α > 0)

---

## Layer Norm vs Batch Norm

| Aspect | Layer Norm | Batch Norm |
|--------|------------|------------|
| Normalizes across | Features | Batch |
| Batch size dependency | No | Yes |
| Inference mode | Same as training | Requires running stats |
| Typical use | Transformers, RNNs | CNNs |

Layer Norm works with batch size 1. Batch Norm doesn't.

---

## The Affine Extension

The frozen version (documented here) has no learnable parameters. In practice, an affine transform is often added:

```
LayerNorm_affine(x) = γ · LayerNorm(x) + β
```

Where γ (scale) and β (shift) are learned. This is "partial" frozen — the normalization is frozen, but the affine is parameterized.

---

*"Normalization exists because raw activations tend toward pathology."*
