# Providence: The Unified Architecture

*FFN as Content-Addressable Memory with Frozen Computation*

**Mesa 15 - Providence Emergence**

---

## The Core Insight

```
TILE = signature + shape + state
     = address   + transform + memory
     = key       + computation + value

Attention(Q, K, V) = softmax(QK^T) V
Providence(query, sigs, shapes) = shapes[argmin(hamming(query, sigs))](query)
```

**Attention is soft Providence. Providence is hard attention.**

---

## What is Providence?

Providence is the unified architecture that synthesizes four discoveries:

| Component | Source | Contribution |
|-----------|--------|--------------|
| **XOR Routing** | Gap 1 | Hamming distance matching instead of dot product |
| **Frozen Shapes** | Gap 2 | Mathematical shapes with 0 learnable compute params |
| **Hierarchical Routing** | Gap 3 | O(√n) cluster→tile scaling |
| **Temporal State** | Gap 3 | Per-tile memory that persists across forward passes |

The result is an FFN where:
- Routing is content-addressable (XOR matching)
- Computation can be pure geometry (frozen shapes)
- State accumulates across time (temporal persistence)
- Scaling is efficient (hierarchical O(√n))

---

## The Mathematical Foundation

### Routing via Hamming Distance

Traditional MoE routing:
```python
winner = argmax(x @ signatures.T)  # O(d) per comparison
```

Providence routing:
```python
winner = argmin(hamming(binarize(x), signatures))  # O(1) XOR + popcount
```

Benefits:
- O(1) bitwise operations vs O(d) multiply-accumulate
- Natural content-addressable semantics
- Direct connection to associative memory

### Frozen Shapes

Computation via fixed mathematical polynomials:

```python
XOR(a, b) = a + b - 2ab        # Exact on {0, 1}
AND(a, b) = ab                  # Exact on {0, 1}
OR(a, b)  = a + b - ab          # Exact on {0, 1}
NOT(a)    = 1 - a               # Exact on {0, 1}
```

These shapes:
- Have 0 learnable parameters in the computation
- Are 100% accurate on binary inputs
- Provide smooth gradients for training (via state updates)
- Can compose into complex operations (adders, comparators, etc.)

### State Persistence

Each tile maintains state that evolves:

```python
output, new_state = tile(input, old_state)
```

This enables:
- Temporal pattern recognition
- Regime-aware computation
- Memory across sequence positions

---

## Architecture

### ProvidenceTile

The atomic unit of Providence:

```python
class ProvidenceTile(nn.Module):
    """
    A tile with: signature (address) + shape (computation) + state (memory).
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        d_state: int,
        frozen_shape: Optional[Callable] = None,  # If provided, 0 compute params
    ):
        ...

    def get_signature(self) -> torch.Tensor:
        """Get ternary routing signature."""
        return self.signature_proj.sign()

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        """(input, state) → (output, new_state)"""
        ...
```

### ProvidenceFFN

The unified FFN:

```python
class ProvidenceFFN(nn.Module):
    """
    Providence = Temporal + Geometric + Sparse + Lookup + FFN

    Routing via XOR (Hamming distance)
    Computation via frozen or learned shapes
    State persists across forward passes
    Hierarchical for O(√n) scaling
    """

    def __init__(
        self,
        d_model: int,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        d_state: int = 16,
        use_frozen_shapes: bool = False,
        use_soft_routing: bool = True,
        mode: str = 'hierarchical',  # or 'flat'
    ):
        ...

    def forward(self, x, state=None):
        """
        Returns:
            output: Transformed input
            new_state: Updated state dict
            routing_info: Dict with tile_idx, cluster_idx, gate
            aux_losses: Dict with balance losses
        """
        ...
```

### ProvidenceBlock

For transformer integration:

```python
class ProvidenceBlock(nn.Module):
    """
    Transformer block: LayerNorm → Attention → LayerNorm → ProvidenceFFN
    """

    def forward(self, x, state=None, attn_mask=None):
        # Attention
        x = x + self.attn(self.norm1(x))
        # Providence FFN
        x = x + self.ffn(self.norm2(x), state)
        return x, new_state, routing_info, aux
```

---

## Usage

### Basic Usage

```python
from trix.nn import create_providence_ffn

# Create FFN with learned shapes
ffn = create_providence_ffn(d_model=128, num_tiles=64)

# Initialize state
state = ffn.init_state(batch_size=8)

# Forward pass
x = torch.randn(8, 128)
output, state, routing_info, aux = ffn(x, state)

# State persists - pass to next forward
output2, state, routing_info, aux = ffn(x, state)
```

### With Frozen Shapes (0 Compute Params)

```python
from trix.nn import create_frozen_providence_ffn

# All computation is fixed geometry
ffn = create_frozen_providence_ffn(d_model=128, num_tiles=16)

# Only routing and state updates are learned
# Shape execution has 0 learnable parameters
```

### With Specific Shapes

```python
from trix.nn import ProvidenceFFN

# Choose which frozen shapes to use
shape_names = ['xor', 'and', 'or', 'nand'] * 4  # 16 tiles

ffn = ProvidenceFFN(
    d_model=128,
    num_tiles=16,
    use_frozen_shapes=True,
    frozen_shape_names=shape_names,
)
```

### Sequence Processing

```python
# Process sequence with persistent state
ffn = create_providence_ffn(d_model=128, num_tiles=64)
state = ffn.init_state(batch_size=4)

# Token by token (state carries forward)
for t in range(seq_len):
    output, state, _, _ = ffn(x[:, t], state)

# Or process whole sequence at once
output, state, routing_info, aux = ffn(x, state)  # x: [batch, seq, d_model]
```

### Transformer Integration

```python
from trix.nn import ProvidenceBlock

block = ProvidenceBlock(
    d_model=128,
    n_heads=8,
    num_tiles=64,
    use_frozen_shapes=False,
)

x = torch.randn(4, 16, 128)
output, state, routing_info, aux = block(x)
```

---

## Routing Modes

### Hierarchical (Default)

O(√n) routing via cluster→tile:

```python
ffn = ProvidenceFFN(
    d_model=128,
    num_tiles=64,
    tiles_per_cluster=8,  # 8 clusters of 8 tiles
    mode='hierarchical',
)
# Comparisons: 8 + 8 = 16 (vs 64 for flat)
```

### Flat

O(n) routing, simpler but more comparisons:

```python
ffn = ProvidenceFFN(
    d_model=128,
    num_tiles=64,
    mode='flat',
)
```

---

## Training vs Inference

### Soft Routing (Training)

During training, use soft routing for gradient flow:

```python
ffn = ProvidenceFFN(
    d_model=128,
    num_tiles=64,
    use_soft_routing=True,   # Soft weights via softmax(-distance/temp)
    temperature=1.0,
)
ffn.train()

# All tiles contribute (weighted by soft gate)
# Gradients flow to all signatures
```

### Hard Routing (Inference)

During inference, only winner computes:

```python
ffn.eval()

# Only winning tile executes
# Sparse computation: 1/num_tiles
```

---

## Analysis Tools

### Routing Statistics

```python
stats = ffn.get_routing_stats()
# {
#     'num_tiles': 64,
#     'num_clusters': 8,
#     'active_tiles': 58,
#     'active_clusters': 8,
#     'tile_usage_std': 0.023,
#     'frozen_tiles': 0,
#     'learned_tiles': 64,
# }
```

### Transition Matrix

Track which tiles follow which:

```python
# After multiple forward passes
trans = ffn.get_transition_matrix(normalize=True)
# trans[i, j] = P(tile_j | tile_i was previous)
```

### Regime Analysis

```python
analysis = ffn.get_regime_analysis()
# {
#     'transition_matrix': ...,
#     'self_transition_rates': ...,
#     'stable_tiles': [3, 7, 12],      # High self-transition
#     'hub_tiles': [0, 5],             # High transition entropy
#     'frozen_tiles': [],
#     'learned_tiles': [0, 1, ..., 63],
# }
```

### Parameter Counting

```python
counts = ffn.count_parameters()
# {
#     'total': 234567,
#     'routing': 12345,    # Signatures, cluster routing
#     'compute': 200000,   # Shape parameters (0 if frozen)
#     'state': 22222,      # State update networks
# }
```

---

## When to Use What

| Scenario | Configuration |
|----------|---------------|
| General purpose | `create_providence_ffn(d_model, num_tiles)` |
| Maximum compression | `create_frozen_providence_ffn(d_model, num_tiles)` |
| Long sequences | High `d_state`, `use_state_routing=True` |
| Many tiles (>64) | `mode='hierarchical'` with `tiles_per_cluster=sqrt(n)` |
| Few tiles (<16) | `mode='flat'` |
| Training | `use_soft_routing=True`, `temperature=1.0` |
| Inference | `ffn.eval()` (automatic hard routing) |

---

## The Journey

Providence emerged from the convergence of six FFN variants:

1. **TriXFFN**: Ternary signatures, emergent routing
2. **SparseTriXFFN**: Explicit sparsity training
3. **HierarchicalTriXFFN**: O(√n) cluster→tile routing
4. **TemporalTileLayer**: State persistence
5. **SparseLookupFFN**: Routing IS computation
6. **Frozen6502**: 0 learnable compute params

Each was a projection of the same underlying structure. Providence is the unification.

---

## Relationship to Attention

```
Attention: softmax(QK^T) V
           ↓
           Soft matching (all keys contribute)
           ↓
           Weighted sum of values

Providence: shapes[argmin(hamming(Q, sigs))](Q)
            ↓
            Hard matching (one signature wins)
            ↓
            Single shape executes
```

They are the same mechanism viewed through different lenses:
- Attention is **soft** Providence (all participate, weighted)
- Providence is **hard** attention (winner takes all)

The soft routing mode during training bridges them.

---

## Files

| File | Description |
|------|-------------|
| `src/trix/nn/providence.py` | Main implementation |
| `src/trix/nn/xor_ffn.py` | XOR routing primitives |
| `src/trix/nn/frozen_shapes.py` | Frozen shape library |
| `src/trix/nn/hierarchical_temporal.py` | Temporal state primitives |
| `tests/test_providence_ffn.py` | 44 comprehensive tests |

---

*"The architecture that provides itself."*
