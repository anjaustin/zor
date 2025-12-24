# SDPMX Pipeline

> **Vi's Synthesis: Hilbert Space Operators for Active Inference**
>
> *"The geometry dictates the operator order. AB ≠ BA."*

Date: 2024-12-20

---

## Overview

The SDPMX Pipeline implements the complete operator sequence for Hilbert space projection:

```
HALO = X ∘ M ∘ P ∘ D ∘ S
```

Read right-to-left:
1. **S**: Smooth the input (discrete → continuous)
2. **D**: Differentiate (surface → gradients)
3. **P**: Project onto healthy subspace
4. **M**: Generate XOR mask
5. **X**: Apply intervention

---

## The Five Operators

### S: Smoothing Operator

```
S: ℤ^n → C^∞(ℝ^n)
```

Maps discrete input to continuous spline surface using 1D Kolmogorov-Arnold splines per dimension.

```python
from trix.nn import SmoothingOperator

S = SmoothingOperator(dim=128, grid_size=16)
smoothed = S(x)  # [batch, dim] → [batch, dim]
```

**Key insight**: "768^16 = Heat Death. 768×16 = Doable."

### D: Differentiation Operator

```
D: C^∞(ℝ^n) → T(ℝ^n)
```

Computes gradients (tangent vectors) from the smooth surface.

```python
from trix.nn import DifferentiationOperator

D = DifferentiationOperator(S)  # D depends on S
gradients = D(x)  # Returns slopes at each point
```

**Note**: D requires S's splines to compute derivatives. Order matters.

### P: Projection Operator

```
P: T(ℝ^n) → V
```

Projects gradients onto learned "healthy" subspace using signature-based routing.

```python
from trix.nn import ProjectionOperator

P = ProjectionOperator(dim=128, num_subspaces=16)
projected, indices = P(gradients)
```

**Hilbert space interpretation**: `proj_V(x) = argmin_{v ∈ V} ||x - v||²`

### M: Mask Operator

```
M: V → {0,1}^n
```

Generates binary XOR mask from projected gradients.

```python
from trix.nn import MaskOperator

M = MaskOperator(dim=128, threshold=0.0)
mask = M(projected)  # Binary {0, 1} mask
```

### X: XOR Operator

```
X: {0,1}^n × ℤ^n → ℤ^n
```

Applies masked correction to the original state.

```python
from trix.nn import XOROperator

X = XOROperator(dim=128)
x_new = X(x, mask, projected)  # x + mask * correction
```

---

## Complete Pipeline

```python
from trix.nn import SDPMXPipeline

# Create pipeline
pipeline = SDPMXPipeline(
    dim=128,
    grid_size=16,
    num_subspaces=16,
    mask_threshold=0.0,
)

# Forward pass
x = torch.randn(batch_size, 128)
x_new, info = pipeline(x)

# With intermediates
x_new, info = pipeline(x, return_intermediates=True)
# info contains: smoothed, gradients, projected, mask, subspace_idx

# With loss
x_new, loss, info = pipeline.forward_with_loss(x, target)
```

---

## Why Order Matters

### Wrong Orders (Don't Work)

**D before S**: Can't differentiate discrete memory
```
X ∘ M ∘ P ∘ S ∘ D  ← WRONG
```

**P before D**: Can't differentiate a projection (loses information)
```
X ∘ M ∘ D ∘ P ∘ S  ← WRONG
```

**X before M**: Causality violation
```
M ∘ X ∘ P ∘ D ∘ S  ← WRONG
```

### Correct Order

```
S → D → P → M → X
```

Each operator transforms the output of the previous one. The geometry dictates this sequence.

---

## Theoretical Foundation

### Robert's Insight: Hilbert Space Geometry

- Every state is a point in n-dimensional Hilbert space
- "Healthy" states form a subspace V
- "Stuck" states are in the orthogonal complement V⊥
- HALO projects stuck states onto V: minimize ||x - proj_V(x)||²

### V'Gem's Architecture: Splines + XOR

- S uses Kolmogorov-Arnold 1D splines (avoid curse of dimensionality)
- XOR for reversible intervention
- Operators compose in the correct order

### The Vision: Ghost in the Machine

- Self-repairing computation
- Active inference (optimize the world, not just the weights)
- Observable learning

---

## Performance

### 6502 Benchmark

| Model | Accuracy | Parameters |
|-------|----------|------------|
| KAN (S only) | 99.55% | 19,800 |
| SDPMX (S→D→P→M→X) | 90.32% | 283,849 |

The gap indicates P needs refinement. The pipeline structure is validated, but the projection operator should learn the actual healthy subspace rather than just routing by signatures.

### Per-Operator Parameters

```
S (Smoothing):  ~2K params
P (Projection): ~33K params (dominant)
M (Mask):       ~100 params
X (XOR):        ~1 param
```

---

## Future Work

1. **Add E and C operators**: Entropy detection and healthy/stuck classification
2. **Improve P**: Learn true Hilbert projection, not just routing
3. **Connect to HALO**: Integrate with Guardian system for meta-learning
4. **Pattern storage**: Agitator library for successful interventions

---

## References

- Vi's Synthesis Document (2024-12-19)
- Robert Sitton: Hilbert Space Geometry
- Kolmogorov-Arnold Networks (KAN)
- Fields-Levin: Active Inference

---

*"Same computation. Four coordinate systems. The hat goes all the way down."*
