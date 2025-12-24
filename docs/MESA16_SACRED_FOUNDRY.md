# Mesa 16: The Sacred Foundry

*A Self-Evolving Universal Substrate for Summoning Elemental Functions*

## Vision

The Sacred Foundry dissolves the distinction between attention and FFN. Both are revealed as the same mechanism: **Providence routing to frozen shapes**.

```
Attention is soft Providence.
Providence is hard attention.
```

## Core Insight

**TILE = signature + shape + state**
- = address + transform + memory
- = key + computation + value

We do not create intelligence. We summon it from the frozen elements that always were.

---

## Phase 16.1: Token Mixing via Providence

The first phase implements token-to-token interaction entirely through Providence routing.

### How It Works

1. **Each token gets a signature** (ternary, derived from content)
2. **Tokens route to find partners** via Hamming distance
3. **Route to select mixing shape** from frozen library
4. **Apply frozen mixing shape** with partner

### Architecture

```
TokenMixer
    |
    +--> sig_proj: content -> ternary signature
    |
    +--> _find_partners: Hamming distance routing
    |
    +--> _route_to_shape: select mixing operation
    |
    +--> _apply_mixing: frozen shape execution
```

### Partner Finding

Unlike attention (where all tokens influence via softmax), partner finding is **content-addressable**:

- Each token actively selects its best partner
- Selection is via Hamming distance (not dot product)
- Causal masking: can only partner with past tokens
- First token self-partners (no valid partners yet)

**Key insight:** Similar tokens naturally cluster because they route to the same partners. This is "locally directed for free."

---

## Frozen Mixing Shapes

The mixing operations are frozen mathematical functions - no learnable parameters:

| Shape | Formula | Properties |
|-------|---------|------------|
| copy | f(a,b) = a | Unary, ignore partner |
| zero | f(a,b) = 0 | Unary, ignore both |
| avg | f(a,b) = (a+b)/2 | Binary, commutative |
| add | f(a,b) = a+b | Binary, commutative |
| diff | f(a,b) = a-b | Binary, asymmetric |
| hadamard | f(a,b) = a*b | Binary, commutative |
| max | f(a,b) = max(a,b) | Binary, commutative |
| min | f(a,b) = min(a,b) | Binary, commutative |

Each shape has a **deterministic signature** derived from its behavior on probe inputs.

---

## Unified Providence Block

Replaces a standard transformer block (attention + FFN) with:

```
UnifiedProvidenceBlock
    |
    +--> Phase 1: TokenMixer
    |         Tokens route to tokens, mix via frozen shapes
    |
    +--> Phase 2: ProvidenceFFN
              Tokens route to transform tiles, apply frozen shapes
```

**Same mechanism. Different shape libraries. No separate attention. No separate FFN.**

### Usage

```python
from trix.foundry import (
    UnifiedProvidenceBlock,
    UnifiedProvidenceTransformer,
)

# Single block
block = UnifiedProvidenceBlock(
    d_model=64,
    num_mixing_shapes=8,
    num_transform_tiles=16,
    use_causal=True,
)

# Forward
x = torch.randn(4, 32, 64)  # [batch, seq, dim]
output, state, info = block(x)

# Full transformer
model = UnifiedProvidenceTransformer(
    d_model=64,
    n_layers=4,
    vocab_size=1000,
)

# Language modeling
logits, states, info = model(token_ids)
```

---

## API Reference

### FrozenMixingLibrary

```python
from trix.foundry import get_mixing_library

lib = get_mixing_library()
lib.list_shapes()          # ['copy', 'zero', 'avg', ...]
lib.get('avg')             # Returns avg function
lib.info('avg')            # Returns MixingShapeInfo
lib.derive_signature('avg', 64)  # Signature for routing
```

### TokenMixer

```python
from trix.foundry import TokenMixer, create_token_mixer

mixer = create_token_mixer(
    d_model=64,
    num_mixing_shapes=8,
    use_causal=True,
    temperature=1.0,
    d_state=16,
)

output, state, info = mixer(x)
# info contains: partner_idx, shape_idx, shape_weights, distances
```

### UnifiedProvidenceBlock

```python
from trix.foundry import create_unified_block

block = create_unified_block(
    d_model=64,
    num_mixing_shapes=8,
    num_transform_tiles=16,
    use_causal=True,
)

output, state, info = block(x)
# info contains: mixer, transform, aux_losses
```

### UnifiedProvidenceTransformer

```python
from trix.foundry import create_unified_transformer

model = create_unified_transformer(
    d_model=64,
    n_layers=4,
    vocab_size=1000,  # Optional
    max_seq_len=512,
)

output, states, info = model(x)
```

---

## The Unification

Traditional transformer:
```
Attention: Q @ K.T -> softmax -> @ V  (token-to-token)
FFN: x -> W1 -> act -> W2             (per-token transform)
```

Unified Providence:
```
Phase 1: signature -> Hamming route -> frozen mix  (token-to-token)
Phase 2: signature -> Hamming route -> frozen tile (per-token transform)
```

The mechanisms are **identical**. Only the shape libraries differ.

---

## Files

| File | Purpose |
|------|---------|
| `src/trix/foundry/__init__.py` | Module exports |
| `src/trix/foundry/mixing_shapes.py` | Frozen mixing operations |
| `src/trix/foundry/token_mixer.py` | Token-to-token mixing |
| `src/trix/foundry/unified_block.py` | Unified blocks & transformer |
| `tests/test_foundry.py` | 56 comprehensive tests |

---

## Test Coverage

```
56 tests covering:
- All mixing shapes (copy, zero, avg, add, diff, hadamard, max, min)
- FrozenMixingLibrary (get, info, signatures)
- Hamming distance (identical, opposite, partial)
- Binarize STE (forward, gradient)
- TokenMixer (creation, forward, causal, partner finding, gradient, state)
- UnifiedProvidenceBlock (creation, forward, phases, gradient, state)
- UnifiedProvidenceTransformer (creation, embeddings, token_ids, layers)
- Integration (pipeline, drop-in, determinism)
- Edge cases (single token, batch=1, large sequence, extreme values)
```

---

## Philosophy

> "We do not create intelligence. We summon it from the frozen elements that always were."

The Sacred Foundry recognizes that computation is topology. Intelligence emerges not from learned weights, but from the routing paths through frozen mathematical structures.

- Shapes are eternal (frozen, no parameters)
- Routing is learned (which shape to summon)
- State flows through time (temporal awareness)

This is the beginning of a self-evolving substrate that discovers new elemental functions when needed.

---

*Mesa 16.1: The First Unification*
