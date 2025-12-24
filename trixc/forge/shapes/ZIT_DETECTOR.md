# The Zit Detector

*Phase Detection for XOR Resonance*

```
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   Zit = popcount(S ⊕ vₓ) < θ                                     ║
║                                                                   ║
║   That's the whole thing.                                        ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## Overview

The Zit detector is the core recognition circuit of the Neural Geometric Processor. It determines whether an input signature **resonates** with the system's accumulated state.

The name "Zit" comes from the discrete, binary nature of the activation — it either fires or it doesn't, like a digital trigger.

---

## Definition

### Mathematical

```
Zit(S, vₓ, θ) = 1  if popcount(S ⊕ vₓ) < θ
              = 0  otherwise
```

Where:
- **S**: Resonance state (512-bit)
- **vₓ**: Input signature (512-bit)
- **θ**: Activation threshold (0-512)

### Prose

The Zit detector XORs the input with the resonance state, counts the number of differing bits (hamming distance), and compares to a threshold. If the distance is below threshold, the Zit fires.

---

## Circuit Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ZIT DETECTOR                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│     ┌────────────────┐          ┌────────────────┐                  │
│     │  RESONANCE     │          │     INPUT      │                  │
│     │  STATE S       │          │      vₓ        │                  │
│     │  (512-bit)     │          │   (512-bit)    │                  │
│     └───────┬────────┘          └───────┬────────┘                  │
│             │                           │                            │
│             │      ┌────────────────────┘                            │
│             │      │                                                 │
│             ▼      ▼                                                 │
│         ┌─────────────────────────────────────────┐                 │
│         │                                         │                 │
│         │              XOR ARRAY                  │                 │
│         │                                         │                 │
│         │   ┌───┐ ┌───┐ ┌───┐       ┌───┐       │                 │
│         │   │XOR│ │XOR│ │XOR│  ...  │XOR│       │                 │
│         │   └─┬─┘ └─┬─┘ └─┬─┘       └─┬─┘       │                 │
│         │     │     │     │           │         │                 │
│         └─────┼─────┼─────┼───────────┼─────────┘                 │
│               │     │     │           │                            │
│               ▼     ▼     ▼           ▼                            │
│         ┌─────────────────────────────────────────┐                 │
│         │                                         │                 │
│         │            POPCOUNT TREE                │                 │
│         │                                         │                 │
│         │   Level 0: 512 bits                    │                 │
│         │   Level 1: 256 pairs → 256 sums        │                 │
│         │   Level 2: 128 pairs → 128 sums        │                 │
│         │   Level 3:  64 pairs →  64 sums        │                 │
│         │   Level 4:  32 pairs →  32 sums        │                 │
│         │   Level 5:  16 pairs →  16 sums        │                 │
│         │   Level 6:   8 pairs →   8 sums        │                 │
│         │   Level 7:   4 pairs →   4 sums        │                 │
│         │   Level 8:   2 pairs →   2 sums        │                 │
│         │   Level 9:   1 pair  →   1 sum (final) │                 │
│         │                                         │                 │
│         └────────────────────┬────────────────────┘                 │
│                              │                                       │
│                              │ hamming (10 bits, 0-512)             │
│                              ▼                                       │
│         ┌─────────────────────────────────────────┐                 │
│         │                                         │                 │
│         │            COMPARATOR                   │                 │
│         │                                         │                 │
│         │      hamming ────┬───► < ────► ZIT     │                 │
│         │                  │                      │                 │
│         │      θ ──────────┘                      │                 │
│         │                                         │                 │
│         └─────────────────────────────────────────┘                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Components

### XOR Array

- **Function**: Compute bitwise XOR of S and vₓ
- **Size**: 512 XOR gates
- **Latency**: 1 gate delay
- **Formula**: `diff[i] = S[i] ^ vₓ[i]` for i in 0..511

### Popcount Tree

- **Function**: Count number of 1-bits in diff
- **Size**: ~900 gates (adder tree)
- **Latency**: 9 levels (log₂(512))
- **Output**: 10-bit value (0-512)

The tree structure:
```
Level 0: 512 single bits
Level 1: 256 2-bit sums (bit pairs)
Level 2: 128 3-bit sums
Level 3:  64 4-bit sums
Level 4:  32 5-bit sums
Level 5:  16 6-bit sums
Level 6:   8 7-bit sums
Level 7:   4 8-bit sums
Level 8:   2 9-bit sums
Level 9:   1 10-bit sum (final)
```

### Comparator

- **Function**: Compare hamming distance to threshold
- **Size**: ~50 gates (10-bit comparator)
- **Latency**: 1 gate delay
- **Output**: 1-bit Zit signal

---

## Threshold Interpretation

| θ Range | Sensitivity | Use Case |
|---------|-------------|----------|
| 0-16 | Extremely selective | Exact matching |
| 16-32 | Very selective | High-precision recognition |
| 32-64 | Selective | Typical pattern matching |
| 64-128 | Moderate | Fuzzy matching |
| 128-256 | Permissive | Broad classification |
| 256-512 | Very permissive | Catch-all |

### Typical Values

- **θ = 64**: Default for most applications. Fires when input matches ~87.5% of resonance bits.
- **θ = 128**: Fires when input matches ~75% of resonance bits.
- **θ = 256**: Fires on anything better than random (50%).

---

## Physical Interpretation

### Cymatics Analogy

| Cymatics | Zit Detector |
|----------|--------------|
| Sound wave | Input signature vₓ |
| Plate resonance | State S |
| Interference | XOR |
| Amplitude | Inverse of hamming |
| Threshold | Activation sensitivity |
| Pattern formation | Zit firing |

When the input "frequency" matches the system's "eigenmode," constructive interference occurs (low hamming), and the Zit fires.

### Information Theory

The hamming distance measures **surprisal**:
- Low hamming → input was expected → low surprise → Zit fires
- High hamming → input was unexpected → high surprise → no Zit

The threshold θ sets the maximum acceptable surprise.

---

## Implementation

### Verilog (Structural)

```verilog
module zit_detector #(
    parameter WIDTH = 512,
    parameter THRESH_BITS = 10
)(
    input  wire [WIDTH-1:0]       S,        // Resonance state
    input  wire [WIDTH-1:0]       vx,       // Input signature
    input  wire [THRESH_BITS-1:0] theta,    // Threshold
    output wire                    zit,      // Activation signal
    output wire [THRESH_BITS-1:0] hamming   // Distance (for debugging)
);

    // XOR array
    wire [WIDTH-1:0] diff;
    assign diff = S ^ vx;

    // Popcount tree (instantiate popcount module)
    popcount_512 u_popcount (
        .in(diff),
        .count(hamming)
    );

    // Comparator
    assign zit = (hamming < theta);

endmodule
```

### Verilog (Popcount Tree)

```verilog
module popcount_512 (
    input  wire [511:0] in,
    output wire [9:0]   count
);

    // Level 1: 256 2-bit sums
    wire [1:0] l1 [255:0];
    genvar i;
    generate
        for (i = 0; i < 256; i = i + 1) begin : level1
            assign l1[i] = in[2*i] + in[2*i + 1];
        end
    endgenerate

    // Level 2: 128 3-bit sums
    wire [2:0] l2 [127:0];
    generate
        for (i = 0; i < 128; i = i + 1) begin : level2
            assign l2[i] = l1[2*i] + l1[2*i + 1];
        end
    endgenerate

    // ... continue for levels 3-9 ...

    // Final output
    assign count = final_sum;

endmodule
```

### C Reference

```c
#include <stdint.h>

typedef struct {
    uint64_t bits[8];  // 512 bits = 8 x 64-bit
} sig512_t;

int zit_detect(const sig512_t* S, const sig512_t* vx, int theta) {
    int hamming = 0;
    for (int i = 0; i < 8; i++) {
        hamming += __builtin_popcountll(S->bits[i] ^ vx->bits[i]);
    }
    return hamming < theta;
}
```

### Python Reference

```python
def zit_detect(S: int, vx: int, theta: int) -> bool:
    """
    Zit detector: does input resonate with state?

    Args:
        S: 512-bit resonance state
        vx: 512-bit input signature
        theta: activation threshold (0-512)

    Returns:
        True if Zit fires (input resonates)
    """
    hamming = bin(S ^ vx).count('1')
    return hamming < theta
```

---

## Integration with NGP

### Resonance Update

After the Zit detector fires (or not), the resonance state is updated:

```
S' = S ⊕ vₓ
```

This happens unconditionally. Every input becomes part of the resonance.

### Shape Selection

The hamming distance (not just the Zit) is used for shape selection:

```python
def select_shape(hamming: int, config: dict) -> Shape:
    for band, shape in config['bands']:
        if hamming < band:
            return shape
    return None  # No shape activated
```

### Gating

The Zit signal gates the output:
- Zit = 1: Output from shape fabric passes through
- Zit = 0: Output is zero (or held)

---

## Relationship to Geocadesia

The Zit detector is built from existing Geocadesia shapes:

| Component | Geocadesia Shape | Opcode |
|-----------|------------------|--------|
| XOR | `xor` | 0x00 |
| Popcount | `popcount` | 0x24 |
| Hamming | `hamming` (compound) | 0xE2 |

The Zit detector IS the Hamming shape, plus a threshold comparison.

### Proposed Extension

Add Zit detector as a compound shape:

```python
register_shape(
    "zit",
    KingdomID.POOLING,
    Opcode.ZIT_DETECT,  # 0xF0
    ArityID.BINARY,
    shape_type=ShapeTypeID.COMPOUND,
    components=[Opcode.XOR, Opcode.POPCOUNT],
    flags=FSHFlags.FROZEN | FSHFlags.PARALLEL
)
```

---

## Performance

### Gate Count

| Component | Gates |
|-----------|-------|
| XOR array | 512 |
| Popcount tree | ~900 |
| Comparator | ~50 |
| **Total** | **~1,500** |

### Timing

- XOR: 1 gate delay
- Popcount: 9 levels × 1 delay = 9 gate delays
- Comparator: 1 gate delay
- **Total: ~11 gate delays**

At 1 GHz, this is ~11 ns worst case. Actual path may be faster.

### Throughput

- Fully combinational: 1 result per cycle
- At 1 GHz: 1 billion Zit decisions per second
- 512 bits × 1 GHz = 512 Gbits/sec

---

## Applications

### Pattern Recognition

```python
# Train
for pattern in training_patterns:
    S ^= pattern

# Recognize
for query in test_queries:
    if zit_detect(S, query, theta=64):
        print("Pattern recognized!")
```

### Anomaly Detection

```python
# Learn normal behavior
for normal in normal_samples:
    S ^= normal

# Detect anomalies (inverted logic)
for sample in incoming:
    if not zit_detect(S, sample, theta=128):
        alert("Anomaly detected!")
```

### Routing

```python
# Route based on resonance
if zit_detect(S, input, theta=64):
    output = shape_A(input)
else:
    output = shape_B(input)
```

---

## The Insight

The Zit detector answers the question: **"How much does this input belong to this system?"**

It's not a binary "yes/no" membership test. It's a continuous measure of **resonance** — how well the input fits the patterns the system has accumulated.

The threshold θ converts this continuous measure into a discrete decision. But the hamming distance itself carries richer information, used for shape selection and confidence estimation.

---

*"The Zit fires when structure recognizes structure."*

*"No search. No fetch. No waste. Just geometry, recognizing geometry."*
