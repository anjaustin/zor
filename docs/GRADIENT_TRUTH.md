# Gradient Truth: Training Beyond STE

**Gradients flow only where there is genuine uncertainty. Structure stands apart as truth.**

---

## The Problem with STE

The Straight-Through Estimator (STE) is the standard technique for training discrete neural networks. It works by:

1. **Forward pass**: Apply hard quantization (e.g., `sign()` for ternary)
2. **Backward pass**: Pretend the quantization didn't happen (pass gradients through unchanged)

```python
# STE: The gradient is literally wrong
class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.sign()  # Discrete
    
    @staticmethod
    def backward(ctx, grad):
        return grad  # Pretend it was identity!
```

The gradient of a step function is zero almost everywhere. STE "lies" to the optimizer by pretending it's continuous. This works empirically, but:

- The gradients are mathematically incorrect
- You're optimizing a different function than you're running
- The discrete/continuous boundary is violated

---

## The Insight

STE conflates two ontologically distinct categories:

| Category | Nature | Method |
|----------|--------|--------|
| **Structure** | Exists, is discovered | Derivation, evolution, distillation |
| **Navigation** | Uncertain, is learned | Gradient descent |

**Structure** includes:
- Mathematical truths (XOR = a + b - 2ab)
- Computational topology (ripple adder structure)
- Converged weight patterns

**Navigation** includes:
- Which structure to use for which input
- How much to scale outputs
- Routing decisions

The elegant solution: **Never use STE.** Discover structure without gradients, then learn only continuous things with real gradients.

---

## The Architecture

Gradient Truth decomposes computation into three layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                          INPUT                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 1: ROUTING                              │
│                                                                  │
│   Continuous, learned, receives gradients.                      │
│   Maps input → shape selection via soft attention.              │
│                                                                  │
│   Parameters: O(d_model × num_shapes)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 2: SHAPE BANK                           │
│                                                                  │
│   Discrete, frozen, NO gradients.                               │
│   Library of computational primitives.                          │
│                                                                  │
│   Parameters: 0 (frozen truth)                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER 3: MAGNITUDE                            │
│                                                                  │
│   Continuous, learned, receives gradients.                      │
│   Scales shape outputs to match required magnitude.             │
│                                                                  │
│   Parameters: O(num_shapes)                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                          OUTPUT                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Shape Genesis

Shapes must exist before routing can select them. Three methods:

### 1. Mathematical Derivation

For domains with known structure (logic, arithmetic):

```python
# These are theorems, not learned patterns
xor = lambda a, b: a + b - 2*a*b
and_op = lambda a, b: a * b
or_op = lambda a, b: a + b - a*b
```

### 2. Evolutionary Search

For searchable structure spaces (circuits, LFSR tap patterns):

```python
# No gradients - just mutation and selection
def evolve_shapes(fitness_fn, generations=100):
    population = random_init()
    for gen in range(generations):
        scores = [fitness_fn(ind) for ind in population]
        population = select_and_mutate(population, scores)
    return best(population)
```

### 3. Distillation

For data-rich domains without known theory:

```python
# Use STE once for discovery, then freeze forever
model = train_with_ste(data)
shapes = extract_converged_patterns(model)
shapes.freeze()  # Never use STE again
```

---

## Three Implementations

Gradient Truth has three complementary implementations:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GRADIENT TRUTH: THREE PATHS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PATH A: Shape Banks              PATH B: Ternary MatMul                    │
│  ────────────────────             ─────────────────────                     │
│  GradientTruthFFN                 HierarchicalTriXFFN                       │
│                                                                             │
│  Route → ShapeBank → Execute      Route → Tile → y = W @ x                  │
│                                                                             │
│  FROZEN:                          FROZEN:                                   │
│    • Polynomial shapes              • Ternary weights (buffers)             │
│    • XOR/AND primitives                                                     │
│                                   LEARNED:                                  │
│  LEARNED:                           • up_scale, down_scale                  │
│    • Routing attention              • output_scale                          │
│    • Magnitude scales                                                       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PATH C: MatMul-Free                                                        │
│  ───────────────────                                                        │
│  SparseLookupFFN                                                            │
│                                                                             │
│  Route → Direction → Spline                                                 │
│                                                                             │
│  FROZEN:                          LEARNED:                                  │
│    • Ternary directions             • direction_scales                      │
│    • Ternary spline coefficients    • spline scales                         │
│                                     • compression network                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Usage

### Path A: Shape Banks (GradientTruthFFN)

The original Gradient Truth implementation with polynomial shape primitives:

```python
from trix.nn import GradientTruthFFN, PolynomialShapeBank

# Create shape bank (frozen mathematical shapes)
shapes = PolynomialShapeBank.from_primitives(d_model=512, num_shapes=16)

# Create FFN - routing and scales are learned, shapes are frozen
ffn = GradientTruthFFN(d_model=512, shape_bank=shapes)

# Train with standard optimizer
output, routing_info = ffn(x)
loss = criterion(output, target)
loss.backward()  # Gradients are mathematically correct!
```

### Path B: Ternary MatMul (HierarchicalTriXFFN)

```python
from trix import HierarchicalTriXFFN

# Gradient Truth is default (use_gradient_truth=True)
ffn = HierarchicalTriXFFN(
    d_model=512,
    num_tiles=64,
    tiles_per_cluster=8,
)

# Train with standard optimizer
output, routing_info, aux_losses = ffn(x)
loss = criterion(output, target) + aux_losses['total_aux']
loss.backward()  # Gradients are mathematically correct!

# What's frozen vs learned:
# - ffn.tiles[i].up_weight   → frozen ternary (buffer)
# - ffn.tiles[i].down_weight → frozen ternary (buffer)
# - ffn.tiles[i].up_scale    → learned (parameter)
# - ffn.tiles[i].down_scale  → learned (parameter)
```

### Path C: MatMul-Free (SparseLookupFFN)

```python
from trix import SparseLookupFFN

# Gradient Truth is default (use_gradient_truth=True)
ffn = SparseLookupFFN(
    d_model=512,
    num_tiles=64,
    tiles_per_cluster=8,
    grid_size=16,  # Spline resolution
)

output, routing_info, aux_losses = ffn(x)
loss = criterion(output, target) + aux_losses['total_aux']
loss.backward()

# What's frozen vs learned:
# - ffn.directions           → frozen ternary (buffer)
# - ffn.direction_scales     → learned (parameter)
# - ffn.splines[i].coeffs    → frozen ternary (buffer)
# - ffn.splines[i].scale     → learned (parameter)
# - ffn.compress             → learned (always)
```

### Legacy STE Mode (Deprecated)

For backward compatibility, STE mode is still available:

```python
# Not recommended - use only for comparison experiments
ffn = HierarchicalTriXFFN(d_model=512, num_tiles=64, use_gradient_truth=False)
ffn = SparseLookupFFN(d_model=512, num_tiles=64, use_gradient_truth=False)
```

---

## How Gradients Flow

```
Forward:
  input → routing (soft attention) → shape selection → 
  shape execution (frozen) → magnitude scaling → output

Backward:
  ∂L/∂output → ∂L/∂scales (direct) → 
  ∂L/∂routing_weights (through weighted sum) →
  ∂L/∂input (through router)
```

The key: routing uses soft attention (differentiable), shapes are frozen (no params to update), scales are continuous. Gradients flow through the attention weights, not through the shapes themselves.

---

## Comparison: STE vs Gradient Truth

| Aspect | STE Approach | Gradient Truth |
|--------|--------------|----------------|
| Discrete weights | Learned with fake gradients | Discovered, then frozen |
| Gradient correctness | Wrong (identity through step) | Correct (only through continuous) |
| Training stability | Can be unstable | Stable (standard optimization) |
| Interpretability | Opaque | Clear (which shape was selected?) |
| Deployment | Quantize, hope it works | Already exact |

---

## Connection to TriX

Gradient Truth unifies existing TriX components:

| Gradient Truth | TriX Equivalent |
|----------------|-----------------|
| FrozenShapeBank | FrozenShapeBank, Frozen6502Net |
| Routing layer | HierarchicalTriXFFN routing |
| Magnitude scales | VGem's output_scale |
| Distillation | XOR superposition compression |
| Shape composition | Atoms → Molecules → Proteins |

**TriX already had the pieces. Gradient Truth names the principle.**

---

## The Principle

> **Gradients should flow where there is genuine uncertainty.**
> **Structure should be fixed where there is mathematical certainty.**

This is ontological hygiene:
- Discrete things are discovered (or derived, or evolved)
- Continuous things are learned
- They interface through routing

---

## Development History

Gradient Truth emerged from applying the [Lincoln Manifold Method](LINCOLN_MANIFOLD_METHOD.md) to the question: *"Is there a more elegant way to train discrete networks than STE?"*

The four phases revealed:
1. **RAW**: STE is a "lie" - what if we stopped lying?
2. **NODES**: Separation of concerns - structure vs navigation
3. **REFLECT**: Ontological hygiene - respect what things ARE
4. **SYNTHESIZE**: Three-layer decomposition with frozen shapes

See `docs/lincoln/gradient_truth/` for the full exploration artifacts.

---

## API Reference

### Classes

- `Shape` - Single computational shape with signature
- `ShapeBank` - Collection of frozen shapes
- `PolynomialShapeBank` - Shapes from mathematical primitives
- `DistilledShapeBank` - Shapes extracted from trained networks
- `GradientTruthFFN` - FFN with three-layer decomposition
- `GradientTruthBlock` - Transformer block with GradientTruthFFN
- `ShapeGenesis` - Utilities for shape discovery

### Functions

- `create_gradient_truth_ffn(d_model, num_shapes, ...)` - Factory for FFN
- `create_gradient_truth_block(d_model, n_heads, num_shapes, ...)` - Factory for block

---

## References

- [THE_WAY.md](THE_WAY.md) - The unified philosophy
- [THEORY.md](THEORY.md) - Mathematical foundations
- [FROZEN_SHAPES.md](FROZEN_SHAPES.md) - Computation as geometry
- [MESA15_NG6502_VISION.md](MESA15_NG6502_VISION.md) - Learning IS Routing
- [LINCOLN_MANIFOLD_METHOD.md](LINCOLN_MANIFOLD_METHOD.md) - The discovery process

---

*"Don't learn what you can read. Don't approximate what you can compute exactly."*
