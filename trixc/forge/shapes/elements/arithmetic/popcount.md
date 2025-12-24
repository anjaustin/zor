# Popcount

*Population Count — Count the 1-bits*

```
┌─────────────────────────────────────────────────────────────┐
│ POPCOUNT                                                    │
├─────────────────────────────────────────────────────────────┤
│ Kingdom: Arithmetic                                         │
│ Type: Elemental                                             │
│ Arity: Unary                                                │
│ Frozen: Yes                                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Definition

### Mathematical

```
popcount(x) = Σ bits(x) = number of 1-bits in x
```

### Prose

Population count (popcount) returns the number of bits set to 1 in a binary value. It's the foundation of Hamming distance and a key operation in FrozenDB vector search.

---

## Visual

```
    x = 0b10110101
        │ │││ │ │
        1 011 1 1 → count = 5

    popcount(0b10110101) = 5
```

---

## Examples

```python
popcount(0b0000) = 0    # No bits set
popcount(0b0001) = 1    # One bit set
popcount(0b0101) = 2    # Two bits set
popcount(0b1111) = 4    # Four bits set
popcount(0b11111111) = 8  # Eight bits set
popcount(255) = 8       # Same as above
```

---

## Implementation

### Python

```python
def popcount(x: int) -> int:
    """Population count: number of 1-bits."""
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count

# Brian Kernighan's algorithm (faster for sparse bits)
def popcount_fast(x: int) -> int:
    count = 0
    while x:
        x &= x - 1  # Clear lowest set bit
        count += 1
    return count
```

### C

```c
// Naive implementation
static inline int popcount(unsigned int x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

// Use hardware instruction if available
#ifdef __POPCNT__
#include <immintrin.h>
static inline int popcount_hw(unsigned int x) {
    return _mm_popcnt_u32(x);
}
#endif
```

---

## Relationships

### Built From

Popcount is elemental.

### Used In

- **[hamming](hamming.md)** — `hamming(a, b) = popcount(XOR(a, b))`
- **XNOR-Net** — Binary neural networks count matching bits
- **Error detection** — Parity checking

### See Also

- **[hamming](hamming.md)** — Hamming distance uses popcount
- **[xor](../logic/xor.md)** — XOR + popcount = Hamming

---

## Use Cases

1. **Hamming Distance**: The core of FrozenDB. `hamming(a, b) = popcount(a XOR b)`.

2. **Cardinality**: Count elements in a bit-set representation.

3. **Parity**: `popcount(x) % 2` gives parity (odd/even 1-bits).

4. **XNOR-Net**: Binary neural networks use popcount for dot products.

---

## Hardware Acceleration

Modern CPUs have dedicated popcount instructions:

| Architecture | Instruction |
|--------------|-------------|
| x86/x64 | `POPCNT` |
| ARM | `CNT` / `VCNT` |
| RISC-V | `cpop` (Zbb extension) |

On Thor (SiFive X288), popcount is expected to be a single-cycle operation.

---

## The Bit-Counting Universe

Popcount appears everywhere in computer science:

- **Cryptography**: Hamming weight in differential analysis
- **Compression**: Run-length encoding
- **Databases**: Bitmap indices
- **AI**: Binary neural networks
- **Error correction**: Hamming codes

It's one of the most fundamental operations on binary data.

---

*"Count your blessings. Count your bits."*
