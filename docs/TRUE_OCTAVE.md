# True Octave: Derived Multi-Resolution Frozen Geometry

> *"Octaves are VIEWS of the same structure, not independent banks."*
> — Lincoln Manifold reflection, December 2025

---

## The Insight

The original MultiScale prototype had independent random octaves at each level. It worked (tests passed), but it wasn't true to the Octave concept.

**The Lincoln Manifold revealed:** Octaves should be DERIVED, not independent. Coarse is a compressed view of Fine. This is what bit-shifting means in the original Sparse Octave design.

```
Sparse Octave (Providence):     key >> 4  = coarser address
True Octave (TriX):             sign(pool(fine)) = coarser signature
```

Same principle. Different domain.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TrueOctaveFFN                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FINE OCTAVE (64 tiles)                                                     │
│  ──────────────────────                                                     │
│  • Random init, frozen ternary weights                                      │
│  • Signatures: S_fine[i] = sign(up_weight[i].mean(dim=0))                  │
│  • Learned: scale per tile                                                  │
│       │                                                                     │
│       │ derive: sign(mean(S_fine[4i : 4i+4]))                              │
│       ▼                                                                     │
│  MEDIUM OCTAVE (16 tiles)                                                   │
│  ───────────────────────                                                    │
│  • Derived from fine, frozen                                                │
│  • Signatures pool 4 fine signatures each                                   │
│  • Learned: scale per tile                                                  │
│       │                                                                     │
│       │ derive: sign(mean(S_medium[4j : 4j+4]))                            │
│       ▼                                                                     │
│  COARSE OCTAVE (4 tiles)                                                    │
│  ──────────────────────                                                     │
│  • Derived from medium, frozen                                              │
│  • Signatures pool 4 medium signatures each                                 │
│  • Learned: scale per tile                                                  │
│       │                                                                     │
│       └──────────────┬──────────────┘                                       │
│                      ▼                                                      │
│              BLEND NETWORK (learned)                                        │
│              ───────────────────────                                        │
│              • Input: normalized x                                          │
│              • Output: [w_fine, w_medium, w_coarse]                        │
│              • The only learned routing                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Derivation Formula

The key insight formalized:

```python
# Fine signatures (base truth, from random init)
S_fine = [sign(tile.up_weight.mean(dim=0)) for tile in fine_tiles]

# Medium signatures (derived)
S_medium[i] = sign(mean(S_fine[4*i : 4*i + 4]))

# Coarse signatures (derived from derived)
S_coarse[j] = sign(mean(S_medium[4*j : 4*j + 4]))
```

This is analogous to bit-shifting:
- Fine: all 64 "bits" of tile space
- Medium: 16 "bits" (4x compression)
- Coarse: 4 "bits" (16x compression)

The coarse signature doesn't lose fine information—it SUMMARIZES it.

---

## Two Modes, Same Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  MODE            ROUTING         BLEND           ENTROPY      USE CASE     │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Generative      Soft (softmax)  Soft (softmax)  > 0          LLM, fuzzy   │
│  Deterministic   Hard (argmax)   Hard (argmax)   = 0          6502, exact  │
│                                                                             │
│  THE FROZEN STRUCTURE IS IDENTICAL. ONLY THE ROUTING MODE CHANGES.         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

This is the key to unifying exact and probabilistic computation:
- **Deterministic systems** (6502, physics, logic): Use hard mode
- **Generative systems** (LLM, creativity): Use soft mode
- **Hybrid**: Switch modes based on confidence or domain

---

## Usage

### Basic Usage

```python
from trix import TrueOctaveFFN

# Create with default 64/16/4 tile structure
ffn = TrueOctaveFFN(d_model=512, num_fine_tiles=64, pool_factor=4)

# Generative mode (default)
ffn.set_mode("generative")
output, info = ffn(x)
# info.entropy > 0 (uncertainty exists)
# info.blend_weights are soft (learned distribution)

# Deterministic mode
ffn.set_mode("deterministic")
output, info = ffn(x)
# info.entropy == 0 (no uncertainty)
# info.blend_weights are one-hot (hard selection)
```

### Verify Derivation

```python
# Check that octaves are properly derived
checks = ffn.get_derivation_check()
assert all(checks.values()), "Derivation should hold"

# See which checks exist
for name, passed in checks.items():
    print(f"{name}: {'✓' if passed else '✗'}")
```

### Inspect Routing

```python
output, info = ffn(x)

# Which octave was used?
print(info.selected_octave)  # [B, T] tensor of 0/1/2

# Blend weights per position
print(info.blend_weights)  # [B, T, 3] soft or one-hot

# Which tile within each octave?
print(info.fine_tile_idx)    # [B, T]
print(info.medium_tile_idx)  # [B, T]
print(info.coarse_tile_idx)  # [B, T]

# Uncertainty measure
print(info.entropy.mean())  # 0 for deterministic, >0 for generative
```

### Transformer Block

```python
from trix import TrueOctaveBlock

block = TrueOctaveBlock(
    d_model=512,
    n_heads=8,
    num_fine_tiles=64,
    pool_factor=4,
)

# Set mode for the whole block
block.set_mode("deterministic")

output, info = block(x, is_causal=True)
```

### Stack of Blocks

```python
import torch.nn as nn
from trix import TrueOctaveBlock

class TrueOctaveTransformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, num_fine_tiles):
        super().__init__()
        self.blocks = nn.ModuleList([
            TrueOctaveBlock(d_model, n_heads, num_fine_tiles)
            for _ in range(n_layers)
        ])
    
    def set_mode(self, mode):
        for block in self.blocks:
            block.set_mode(mode)
    
    def forward(self, x):
        infos = []
        for block in self.blocks:
            x, info = block(x)
            infos.append(info)
        return x, infos
```

---

## What's Frozen vs Learned

| Component | Frozen | Learned |
|-----------|--------|---------|
| Fine tile weights (up, down) | ✓ | |
| Medium tile weights | ✓ (derived) | |
| Coarse tile weights | ✓ (derived) | |
| Derivation structure | ✓ | |
| Tile scales | | ✓ |
| Blend network | | ✓ |
| Output scale | | ✓ |

**Gradient Truth principle:** Structure is frozen. Navigation is learned.

---

## The Philosophy

### Why Derived Octaves?

Independent random octaves WORK (tests pass), but they're not TRUE to the Octave concept. The blend network can learn to compensate for arbitrary structure, but this is inefficient:
- More for blend to learn
- Less interpretable (octaves have no inherent meaning)
- Misses the hierarchical structure of reality

Derived octaves are EFFICIENT:
- Blend only learns which scale matters, not what the scales mean
- Octaves have inherent meaning (fine = precise, coarse = semantic)
- Mirrors the hierarchical structure of the world

### Why Same Architecture for Exact and Fuzzy?

The world has both:
- **Exact dynamics**: 2 + 2 = 4, always
- **Uncertain outcomes**: Which word comes next?

Current LLMs are "fuzzy all the way down"—they approximate even exact things.

TrueOctave separates them:
- **Frozen patterns** = the exact truths (ternary structure)
- **Soft routing** = the uncertainty (blend network)

Same architecture, different mode. The 6502 and GPT can share a substrate.

---

## Comparison

| Aspect | MultiScaleTriXFFN (scaffold) | TrueOctaveFFN (correct) |
|--------|------------------------------|-------------------------|
| Octave init | Random, independent | Derived: pool → sign |
| Deterministic mode | Fine-only, soft blend | Hard routing everywhere |
| Octave relationship | None | Hierarchical (coarse ⊃ fine) |
| Tests | Pass | Pass |
| Conceptually correct | No | Yes |

---

## The Lincoln Manifold Journey

This architecture was born from applying the [Lincoln Manifold Method](LINCOLN_MANIFOLD_METHOD.md) to the MultiScale prototype.

### Phase 1: RAW
> "Is MultiScale actually implementing Octave, or just borrowing the name?"

### Phase 2: NODES
- Node 2: Derived vs Independent (the key tension)
- Node 5: Bit-shift = Pooling on ternary
- Node 6: Providence and signature routing are isomorphic

### Phase 3: REFLECT
> "Octaves are VIEWS of the same structure at different resolutions, not independent banks."

### Phase 4: SYNTHESIZE
The derivation formula. The mode fix. TrueOctaveFFN.

The full exploration is preserved in `docs/lincoln/octave/`.

---

## API Reference

### TrueOctaveFFN

```python
class TrueOctaveFFN(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_fine_tiles: int = 64,
        pool_factor: int = 4,
        d_hidden: int = None,  # defaults to d_model
        dropout: float = 0.1,
    ):
        """
        True Octave FFN with derived multi-resolution structure.
        
        Args:
            d_model: Model dimension
            num_fine_tiles: Number of tiles in fine octave
            pool_factor: How many tiles pool into one (4 → 64/16/4)
            d_hidden: Hidden dimension per tile
            dropout: Dropout rate
        """
```

**Methods:**

```python
def set_mode(self, mode: Literal["generative", "deterministic"]):
    """Set routing mode."""

def forward(self, x: Tensor) -> Tuple[Tensor, OctaveRoutingInfo]:
    """Forward pass."""

def get_derivation_check(self) -> Dict[str, bool]:
    """Verify octave derivation relationships."""

def rederive(self):
    """Re-derive medium and coarse from fine. Call if fine tiles changed."""
```

### OctaveRoutingInfo

```python
@dataclass
class OctaveRoutingInfo:
    blend_weights: Tensor      # [B, T, 3] - weight per octave
    fine_tile_idx: Tensor      # [B, T] - selected fine tile
    medium_tile_idx: Tensor    # [B, T] - selected medium tile
    coarse_tile_idx: Tensor    # [B, T] - selected coarse tile
    selected_octave: Tensor    # [B, T] - which octave dominated
    entropy: Tensor            # [B, T] - routing uncertainty
    mode: str                  # "generative" or "deterministic"
```

### TrueOctaveBlock

```python
class TrueOctaveBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        num_fine_tiles: int = 64,
        pool_factor: int = 4,
        dropout: float = 0.1,
    ):
        """Full transformer block with TrueOctaveFFN."""
```

---

## Testing

The TrueOctaveFFN architecture is verified by 76 tests across two files:

### Basic Tests (`tests/test_octave.py`)

31 tests covering functionality, integration, and conceptual properties.

```bash
PYTHONPATH=src pytest tests/test_octave.py -v
```

### Rigorous Invariant Tests (`tests/test_octave_rigorous.py`)

45 tests across 10 categories of mathematical invariants:

| Category | Tests | What It Verifies |
|----------|-------|------------------|
| Derivation | 4 | coarse = sign(pool(fine)) |
| Ternary | 4 | weights ∈ {-1, 0, +1} |
| Frozen | 3 | structure unchanged by training |
| Mode | 6 | deterministic=exact, generative=soft |
| Gradient | 4 | flow only to learned params |
| Stability | 6 | no NaN, no Inf, bounded |
| Edge Cases | 8 | batch=1, zeros, large inputs |
| Reproducibility | 3 | same seed → same output |
| Training | 3 | loss decreases, params update |
| Integration | 4 | stacking, gradient flow |

```bash
# Run all rigorous tests
PYTHONPATH=src pytest tests/test_octave_rigorous.py -v

# Run specific category
PYTHONPATH=src pytest tests/test_octave_rigorous.py::TestDerivationInvariants -v
```

See [TESTING.md](TESTING.md) for the complete testing guide.

---

## References

- [LINCOLN_MANIFOLD_METHOD.md](LINCOLN_MANIFOLD_METHOD.md) — The discovery process
- [GRADIENT_TRUTH.md](GRADIENT_TRUTH.md) — Frozen structure, learned navigation
- [TESTING.md](TESTING.md) — Complete testing guide
- [docs/lincoln/octave/](lincoln/octave/) — The manifold exploration artifacts
- [SPARSE_OCTAVE.md](../trixc/docs/SPARSE_OCTAVE.md) — Original Octave concept (Providence)

---

*"The wood cuts itself when you understand the grain."*

*Octaves are views of the same grain at different resolutions.*
