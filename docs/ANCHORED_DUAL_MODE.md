# Anchored Dual-Mode FFN

**The anchor doesn't predict the answer. It partitions the question.**

---

## The Core Insight

Speedup comes from NOT searching, not from searching faster.

When you can partition the input space with frozen geometry (anchors), you reduce the search space before any probabilistic routing happens. The anchor is the shoreline. The routing is the voyage. You can't sail through land, and that's the point.

---

## Two Deployment Modes from One Substrate

The architecture supports two distinct deployment modes:

| Mode | Components Used | Use Case |
|------|-----------------|----------|
| **Chips** | Shapes only | Deterministic, synthesizable to silicon |
| **Modal-models** | Shapes + routing | Shapes constrain probabilistic search |

The key insight: frozen shapes can run *slightly ahead* of probabilistic routing, constraining the possibility space in real-time.

---

## Architecture

```
Input ──┬──→ [Anchor Shapes] ──→ partition (fast, frozen)
        │         │
        │         ↓
        └──→ [Router(x, anchor)] ──→ tile selection (learned)
                          │
                          ↓
                [Tile Execution] ──→ output (frozen + scale)
```

### Path 1: Anchor (Frozen, Fast)

Anchors are ternary signatures in {-1, +1}^d. Input is quantized to ternary, then matched against anchors via Hamming similarity:

```python
# Ternary quantize input
x_ternary = x.sign()

# Hamming similarity = dot product for ternary
similarity = torch.einsum('bsd,ad->bsa', x_ternary, anchor_signatures)

# Soft partition membership
anchor_probs = softmax(similarity / temperature)
```

This is frozen - no gradients flow through the anchor signatures.

### Path 2: Router (Learned, Informed)

The router sees BOTH the input AND the anchor partition:

```python
routing_input = torch.cat([x, anchor_features], dim=-1)
tile_logits = router(routing_input)
```

The anchor constrains what tiles make sense. The router learns which tile works best within that constraint.

### Path 3: Execution (Frozen Directions, Learned Scales)

Each tile has a frozen ternary direction. Only the scale is learned:

```python
# Frozen: tile_directions in {-1, +1}^d
# Learned: tile_scales (per-tile magnitude)
delta = scales * directions
output = norm(x + delta)
```

---

## Temperature Annealing

Temperature controls partition sharpness:

| Temperature | Effect |
|-------------|--------|
| High (>1) | Soft partitions, exploration |
| Low (<1) | Hard partitions, commitment |

Annealing schedule for training:
- Start warm (soft partitions, router explores)
- End cold (hard partitions, router commits)

```python
temp = get_temperature_schedule(
    step=current_step,
    total_steps=total_steps,
    start_temp=2.0,
    end_temp=0.1,
    schedule='cosine'
)
model.set_temperature(temp)
```

---

## Training vs Inference

### Training Mode

Uses soft weighted combination for gradient flow:

```python
# Soft combination - gradients flow through tile_probs
directions = einsum('bst,td->bsd', tile_probs, tile_directions)
scales = einsum('bst,t->bs', tile_probs, tile_scales)
```

### Inference Mode

Uses hard selection for speed:

```python
# Hard selection - deterministic, fast
tile_idx = tile_logits.argmax(dim=-1)
directions = tile_directions[tile_idx]
scales = tile_scales[tile_idx]
```

---

## What This Architecture Can Learn

The architecture excels at:
- **Partitioning problems**: XOR, cluster-specific behavior
- **Routing decisions**: which transform for which input region
- **Scaled adjustments**: magnitude modulation of fixed directions

The architecture cannot learn:
- **Arbitrary linear transforms**: directions are frozen ternary
- **Fine-grained continuous mappings**: limited by ternary quantization

This is by design. The constraint IS the feature.

---

## Comparison to Other Architectures

| Architecture | Routing | Directions | Scales |
|--------------|---------|------------|--------|
| Dense FFN | N/A | Learned | Learned |
| MoE | Learned | Learned | Learned |
| Gradient Truth | Frozen | Frozen | Learned |
| **Anchored Dual-Mode** | Anchor-informed + Learned | Frozen | Learned |

Anchored Dual-Mode adds anchor-informed routing to Gradient Truth: the router doesn't just see input, it sees input + partition.

---

## The Shoreline Metaphor

Think of computation as navigation:
- **Shoreline**: The frozen shapes - where you CAN'T go (land)
- **Ocean**: The possibility space - where you CAN go (water)
- **Navigation**: The routing - choosing WHERE to go within the water

The shoreline doesn't move. It partitions reality. The navigator uses it as a constraint to find the right path.

For chips: deploy only the shoreline (deterministic shapes).
For modal-models: deploy both (shapes constrain probabilistic search).

---

## Usage

```python
from trix.nn import AnchoredDualModeFFN, get_temperature_schedule

# Create model
ffn = AnchoredDualModeFFN(
    d_model=512,
    num_anchors=16,      # Partition count
    num_tiles=64,        # Execution variants
    temperature=1.0,     # Initial temperature
    dropout=0.1,
)

# Training loop with annealing
for step in range(total_steps):
    temp = get_temperature_schedule(step, total_steps, 2.0, 0.1)
    ffn.set_temperature(temp)

    output, info = ffn(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# Inference (automatic hard selection)
ffn.eval()
output, info = ffn(x)
```

---

## Born from the Lincoln Manifold Method

This architecture emerged through the Lincoln Manifold exploration process, December 2025. The key insight crystallized from the question:

> "What if the deterministic shapes run slightly ahead, constraining the possibility space before probabilistic search begins?"

The answer: anchor-informed routing. The partition is the constraint.

---

## Testing

The AnchoredDualModeFFN architecture is verified by comprehensive validation tests:

**Unit Validation:** `tests/test_anchored_validation.py` (15 tests)
- Pattern learning (XOR, scaling, anchor-aware)
- Generalization to novel inputs
- Routing behavior and diversity
- Comparison to baselines

**Rigorous Foundation:** `tests/test_rigorous.py` (70 tests)
- Edge cases, numerical stability, invariants
- Gradient flow, determinism, serialization

**Integration:** `tests/test_integration_validation.py` (23 tests)
- Stacked AnchoredBlocks work together
- Gradient flow through deep stacks

**Task-Level:** `tests/test_task_validation.py` (16 tests)
- Language modeling, classification
- Comparison to standard transformers

```bash
# Run anchored validation
PYTHONPATH=src pytest tests/test_anchored_validation.py -v

# Run all validation tests
PYTHONPATH=src pytest tests/test_*_validation.py tests/test_rigorous.py -v
```

See [TESTING.md](TESTING.md) for the complete testing guide.

---

## See Also

- [GRADIENT_TRUTH.md](GRADIENT_TRUTH.md) - The frozen/learned separation principle
- [THE_WAY.md](THE_WAY.md) - Neural-geometric equivalence
- [TRUE_OCTAVE.md](TRUE_OCTAVE.md) - Hierarchical frozen architecture
- [TESTING.md](TESTING.md) - Comprehensive testing guide
- [API.md](API.md) - Complete API reference
