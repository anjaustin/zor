# Synthesis: True Octave TriX Architecture

## Architecture

### Core Principle

**Octaves are derived views, not independent banks.**

Coarse octave signatures = pooled summaries of fine octave signatures.
The derivation is frozen (deterministic pooling + sign).
Only the blend network is learned.

### Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TrueOctaveFFN                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LAYER 1: FINE OCTAVE                                                       │
│  ───────────────────                                                        │
│  • 64 tiles, each with frozen ternary up/down weights                      │
│  • Signatures: S_fine[i] = sign(up_weight[i].sum(dim=0))                   │
│  • Learned: up_scale, down_scale per tile                                  │
│                                                                             │
│  LAYER 2: MEDIUM OCTAVE (derived)                                          │
│  ────────────────────────────────                                          │
│  • 16 meta-tiles, signatures derived from fine                             │
│  • S_medium[j] = sign(mean(S_fine[4j:4j+4]))                               │
│  • Weights: mean of constituent fine tile weights, then sign              │
│  • Learned: meta_scale per meta-tile                                       │
│                                                                             │
│  LAYER 3: COARSE OCTAVE (derived)                                          │
│  ─────────────────────────────────                                         │
│  • 4 macro-tiles, signatures derived from medium                           │
│  • S_coarse[k] = sign(mean(S_medium[4k:4k+4]))                             │
│  • Weights: mean of constituent medium weights, then sign                  │
│  • Learned: macro_scale per macro-tile                                     │
│                                                                             │
│  BLEND NETWORK (learned)                                                    │
│  ───────────────────────                                                   │
│  • Input: x (normalized)                                                    │
│  • Output: [w_fine, w_medium, w_coarse] softmax weights                    │
│  • Architecture: Linear(d_model, d_model//4) → ReLU → Linear(→3)          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Derivation Formulas

```python
# Fine signatures (base truth)
S_fine = [sign(tile.up_weight.sum(dim=0)) for tile in fine_tiles]

# Medium signatures (derived from fine)
S_medium = [sign(mean(S_fine[4*j : 4*j+4])) for j in range(16)]

# Coarse signatures (derived from medium)  
S_coarse = [sign(mean(S_medium[4*k : 4*k+4])) for k in range(4)]

# Medium weights (derived from fine)
W_medium_up[j] = sign(mean(fine_tiles[4*j : 4*j+4].up_weight))
W_medium_down[j] = sign(mean(fine_tiles[4*j : 4*j+4].down_weight))

# Coarse weights (derived from medium)
W_coarse_up[k] = sign(mean(W_medium_up[4*k : 4*k+4]))
W_coarse_down[k] = sign(mean(W_medium_down[4*k : 4*k+4]))
```

### Routing

```python
def route(x, signatures, hard=False):
    scores = x @ signatures.T  # [B, T, num_tiles]
    if hard:
        idx = scores.argmax(dim=-1)
        weights = F.one_hot(idx, num_tiles).float()
    else:
        weights = F.softmax(scores, dim=-1)
    return weights

# Route at each octave
w_fine = route(x, S_fine, hard=hard)
w_medium = route(x, S_medium, hard=hard)
w_coarse = route(x, S_coarse, hard=hard)
```

### Forward Pass

```python
def forward(x, mode="generative"):
    hard = (mode == "deterministic")
    
    # Route at each octave
    out_fine = apply_octave(x, fine_tiles, S_fine, hard)
    out_medium = apply_octave(x, medium_tiles, S_medium, hard)
    out_coarse = apply_octave(x, coarse_tiles, S_coarse, hard)
    
    # Blend
    if hard:
        # Pick most confident octave
        blend_scores = blend_net(x)
        octave_idx = blend_scores.argmax(dim=-1)
        output = select_by_index([out_fine, out_medium, out_coarse], octave_idx)
    else:
        # Soft blend
        blend_weights = F.softmax(blend_net(x), dim=-1)
        output = (blend_weights[..., 0:1] * out_fine + 
                  blend_weights[..., 1:2] * out_medium +
                  blend_weights[..., 2:3] * out_coarse)
    
    return x + output * output_scale
```

---

## Key Decisions

1. **Derived octaves, not independent** (from Reflection insight)
   - Coarse = pooled(fine), not random
   - This is true to the octave concept (bit-shift = pool)

2. **Hard blend in deterministic mode** (from Reflection bug fix)
   - Don't force fine-only
   - Hard routing at all octaves + hard blend
   - Structure exists at all scales

3. **Signatures derived same as weights** (consistency)
   - Both use pool → sign
   - Signature = weight pattern summary

4. **Learned scales at each octave** (Gradient Truth)
   - Frozen: ternary weights, derivation structure
   - Learned: scales, blend network

---

## Implementation Spec

### Class: TrueOctaveFFN

```python
class TrueOctaveFFN(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_fine_tiles: int = 64,
        pool_factor: int = 4,  # 64 → 16 → 4
        d_hidden: int = None,  # defaults to d_model
        dropout: float = 0.1,
    ):
        # Fine tiles: random init, frozen ternary
        # Medium tiles: derived from fine
        # Coarse tiles: derived from medium
        # Blend network: learned
```

### Methods

```python
def derive_octaves(self):
    """Derive medium and coarse from fine. Call after init and after any fine tile changes."""

def set_mode(self, mode: Literal["generative", "deterministic"]):
    """Set routing mode."""

def forward(self, x) -> Tuple[Tensor, OctaveRoutingInfo]:
    """Forward pass with current mode."""

def get_octave_usage(self) -> Dict[str, float]:
    """Return average blend weights per octave."""
```

### OctaveRoutingInfo

```python
@dataclass
class OctaveRoutingInfo:
    blend_weights: Tensor      # [B, T, 3]
    fine_routing: Tensor       # [B, T, 64]
    medium_routing: Tensor     # [B, T, 16]
    coarse_routing: Tensor     # [B, T, 4]
    entropy: Tensor            # [B, T]
    mode: str
```

---

## Success Criteria

- [ ] Octaves are derived, not independent
- [ ] Medium signatures = pool(fine signatures)
- [ ] Coarse signatures = pool(medium signatures)
- [ ] Same for weights
- [ ] Deterministic mode uses hard routing at ALL octaves
- [ ] Deterministic mode uses hard blend (argmax), not soft
- [ ] Generative mode uses soft routing and soft blend
- [ ] Tests verify derivation relationships
- [ ] Tests verify mode behaviors

---

## Migration from MultiScaleTriXFFN

The current `MultiScaleTriXFFN` is a scaffold. `TrueOctaveFFN` replaces it with:

1. Derived octaves instead of independent
2. Fixed deterministic mode (hard everywhere, not fine-only)
3. Proper octave naming (fine/medium/coarse, not levels 0/1/2)

Keep `MultiScaleTriXFFN` for backward compatibility but mark deprecated.

---

## Future Extensions

1. **Temporal octaves**: Different history scopes
2. **Dynamic derivation**: Re-derive during training as fine tiles evolve
3. **Providence integration**: Use Hamming lookup instead of dot product (equivalent, maybe faster)
4. **Hierarchical freezing**: Freeze coarse first, then medium, then fine (curriculum)

---

*The wood cuts itself when you understand the grain.*
*Octaves are views of the same grain at different resolutions.*
