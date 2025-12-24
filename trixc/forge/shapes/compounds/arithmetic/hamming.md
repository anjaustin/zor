# Hamming Distance

*The metric of difference — XOR + Popcount*

```
┌─────────────────────────────────────────────────────────────┐
│ HAMMING                                                     │
├─────────────────────────────────────────────────────────────┤
│ Kingdom: Arithmetic                                         │
│ Type: Compound                                              │
│ Arity: Binary                                               │
│ Frozen: Yes                                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Definition

### Mathematical

```
hamming(a, b) = popcount(a ⊕ b)
             = number of positions where a and b differ
```

### Prose

Hamming distance counts the number of bit positions where two binary values differ. It's computed by XORing the values (which produces 1s at differing positions) and counting the 1s.

**This is THE metric for FrozenDB vector search.**

---

## Visual

```
    a = 0b10110011
    b = 0b10010111
        ─────────
XOR     0b00100100  → positions that differ
              │ │
              └─┴── 2 differences

hamming(0b10110011, 0b10010111) = 2
```

---

## Examples

```python
hamming(0b0000, 0b0000) = 0  # Identical
hamming(0b0000, 0b0001) = 1  # One bit differs
hamming(0b1010, 0b0101) = 4  # All bits differ
hamming(0b1111, 0b0000) = 4  # All bits differ
hamming(0b10110011, 0b10010111) = 2
```

---

## Implementation

### Python

```python
def hamming(a: int, b: int) -> int:
    """Hamming distance: number of differing bits."""
    return popcount(a ^ b)

def hamming_normalized(a: int, b: int, bits: int = 512) -> float:
    """Normalized Hamming distance in [0, 1]."""
    return popcount(a ^ b) / bits

def hamming_similarity(a: int, b: int, bits: int = 512) -> float:
    """Hamming similarity in [0, 1]. Higher = more similar."""
    return 1.0 - (popcount(a ^ b) / bits)
```

### C

```c
static inline int hamming(unsigned int a, unsigned int b) {
    return popcount(a ^ b);
}

// For 512-bit values (using arrays)
static inline int hamming_512(const uint64_t* a, const uint64_t* b) {
    int distance = 0;
    for (int i = 0; i < 8; i++) {  // 8 × 64 = 512 bits
        distance += __builtin_popcountll(a[i] ^ b[i]);
    }
    return distance;
}
```

---

## Relationships

### Built From

- **[xor](../../elements/logic/xor.md)** — Finds differing positions
- **[popcount](../../elements/arithmetic/popcount.md)** — Counts the differences

### Used In

- **FrozenDB** — The core similarity metric
- **Providence Routing** — Route selection
- **Error detection** — Detecting bit flips
- **XNOR-Net** — Binary neural network comparison

### See Also

- **[xor](../../elements/logic/xor.md)** — First step of Hamming
- **[popcount](../../elements/arithmetic/popcount.md)** — Second step
- **[xnor](../../elements/logic/xnor.md)** — Related (counts matching bits)

---

## Use Cases

1. **FrozenDB Vector Search**: Given a query signature and stored signatures, find the one with minimum Hamming distance.

2. **Error Detection/Correction**: Hamming distance tells you how many bits flipped during transmission.

3. **DNA Sequence Comparison**: Comparing genetic sequences encoded as binary.

4. **Image Similarity**: Perceptual hashes use Hamming distance.

---

## Properties

- **Non-negative**: `hamming(a, b) ≥ 0`
- **Identity**: `hamming(a, a) = 0`
- **Symmetry**: `hamming(a, b) = hamming(b, a)`
- **Triangle inequality**: `hamming(a, c) ≤ hamming(a, b) + hamming(b, c)`

These properties make Hamming distance a proper **metric**.

---

## FrozenDB: The Full Stack

```python
# The complete FrozenDB query in Geocadesia shapes

from geocadesia import Hamming, Argmin

# Stored signatures (512-bit in production)
signatures = [sig0, sig1, sig2, ...]

# Query
query_sig = encode(query_data)

# Compute all distances (parallel on Thor)
distances = [Hamming()(query_sig, sig) for sig in signatures]

# Find nearest
match_idx = Argmin()(distances)
```

This is 0.000% signal loss. Exact nearest neighbor.

---

## The 512-bit Regime

At 512 bits:
- `hamming = 0` → Identical signatures
- `hamming = 256` → Random/orthogonal (50% bits differ)
- `hamming = 512` → Completely opposite

In FrozenDB, we typically look for low Hamming distance (similar items).

---

*"The distance between two truths is measured in differing bits."*
