# Frozen Shape Binary Format (.fsh)

*The shapes describe themselves. Thor executes them.*

```
┌─────────────────────────────────────────────────────────────────────┐
│                   FROZEN SHAPE EXECUTABLE                            │
│                                                                      │
│   "Hardware speaks binary. So do shapes."                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Overview

The Frozen Shape format (`.fsh`) is a compact binary representation of computational shapes that can be loaded and executed directly by Thor hardware. Each shape file is self-describing, containing everything needed to identify and execute the shape.

**Three layers of shape storage:**
1. **Python** (`geocadesia/*.py`) — Reference implementation, human-readable
2. **Documentation** (`*.md`) — Mathematical definitions, examples, relationships
3. **Binary** (`bin/*.fsh`) — Hardware-executable, Thor-loadable

---

## File Structure

### Header (32 bytes, fixed)

```
Offset  Size  Field           Description
──────────────────────────────────────────────────────────
0x00    4     magic           Magic bytes: "FSHP"
0x04    1     version_major   Format version major
0x05    1     version_minor   Format version minor
0x06    1     kingdom         Kingdom ID (see table)
0x07    1     shape_type      0=elemental, 1=compound
0x08    1     arity           1=unary, 2=binary, 3=ternary, 0xFF=n-ary
0x09    1     opcode          Hardware instruction code
0x0A    2     flags           Bit flags (see table)
0x0C    4     component_count Number of component opcodes (compounds only)
0x10    16    name            Shape name (null-padded)
──────────────────────────────────────────────────────────
Total: 32 bytes
```

### Body (variable, compounds only)

```
Offset  Size  Field           Description
──────────────────────────────────────────────────────────
0x20    N     components      Array of component opcodes (1 byte each)
──────────────────────────────────────────────────────────
```

---

## Kingdom IDs

| ID   | Kingdom       | Description                        |
|------|---------------|-------------------------------------|
| 0x01 | LOGIC         | Boolean operations (XOR, AND, OR)   |
| 0x02 | ARITHMETIC    | Numeric operations (add, mul, popcount) |
| 0x03 | ACTIVATION    | Nonlinearities (ReLU, GELU, sigmoid) |
| 0x04 | NORMALIZATION | Normalization (LayerNorm, RMSNorm)  |
| 0x05 | LINEAR        | Matrix operations (matmul)          |
| 0x06 | ATTENTION     | Attention mechanisms                |
| 0x07 | POOLING       | Reduction operations (max, argmin)  |

---

## Opcode Map

### Logic Kingdom (0x00-0x1F)

| Opcode | Name  | Formula                    |
|--------|-------|----------------------------|
| 0x00   | XOR   | a + b - 2ab                |
| 0x01   | AND   | a × b                      |
| 0x02   | OR    | a + b - ab                 |
| 0x03   | NOT   | 1 - a                      |
| 0x04   | NAND  | 1 - ab                     |
| 0x05   | NOR   | 1 - (a + b - ab)           |
| 0x06   | XNOR  | 1 - (a + b - 2ab)          |

### Arithmetic Kingdom (0x20-0x3F)

| Opcode | Name     | Formula                    |
|--------|----------|----------------------------|
| 0x20   | ADD      | a + b                      |
| 0x21   | SUB      | a - b                      |
| 0x22   | MUL      | a × b                      |
| 0x23   | NEG      | -a                         |
| 0x24   | POPCOUNT | count of 1-bits            |

### Activation Kingdom (0x40-0x5F)

| Opcode | Name       | Formula                    |
|--------|------------|----------------------------|
| 0x40   | RELU       | max(0, x)                  |
| 0x41   | SIGMOID    | 1 / (1 + e^-x)             |
| 0x42   | TANH       | tanh(x)                    |
| 0x43   | GELU       | 0.5x(1 + tanh(√(2/π)(x + 0.044715x³))) |
| 0x44   | SWISH      | x · sigmoid(x)             |
| 0x45   | SOFTMAX    | e^xi / Σe^xj               |
| 0x46   | LEAKY_RELU | max(αx, x)                 |

### Normalization Kingdom (0x60-0x7F)

| Opcode | Name       | Formula                    |
|--------|------------|----------------------------|
| 0x60   | LAYER_NORM | (x - μ) / √(σ² + ε)        |
| 0x61   | RMS_NORM   | x / √(mean(x²) + ε)        |

### Pooling Kingdom (0x80-0x9F)

| Opcode | Name     | Formula                    |
|--------|----------|----------------------------|
| 0x80   | MAX_POOL | max(x)                     |
| 0x81   | AVG_POOL | mean(x)                    |
| 0x82   | SUM_POOL | Σx                         |
| 0x83   | MIN_POOL | min(x)                     |
| 0x84   | ARGMIN   | index of min(x)            |
| 0x85   | ARGMAX   | index of max(x)            |

### Compound Shapes (0xE0-0xFF)

| Opcode | Name       | Built From          | Purpose          |
|--------|------------|---------------------|------------------|
| 0xE0   | HALF_ADDER | XOR, AND            | Sum + Carry      |
| 0xE1   | FULL_ADDER | XOR×2, AND×2, OR    | Sum + Carry + Cin|
| 0xE2   | HAMMING    | XOR, POPCOUNT       | FrozenDB metric  |

---

## Flags

| Bit  | Name          | Description                          |
|------|---------------|--------------------------------------|
| 0x01 | DIFFERENTIABLE| Smooth/continuous implementation     |
| 0x02 | FROZEN        | No learnable parameters              |
| 0x04 | PARALLEL      | Can execute in parallel              |
| 0x08 | VECTORIZED    | Has SIMD implementation              |
| 0x10 | INPLACE       | Can operate in-place                 |

---

## Example: Hamming Distance

The Hamming shape is a compound shape used in FrozenDB for vector search.

### Hex Dump

```
0000: 46 53 48 50 01 00 02 01 02 e2 06 00 02 00 00 00  |FSHP............|
0010: 68 61 6d 6d 69 6e 67 00 00 00 00 00 00 00 00 00  |hamming.........|
0020: 00 24                                            |.$|
```

### Decoded

```
magic:           FSHP
version:         1.0
kingdom:         ARITHMETIC (0x02)
shape_type:      COMPOUND (0x01)
arity:           BINARY (0x02)
opcode:          0xE2 (HAMMING)
flags:           0x0006 (FROZEN | PARALLEL)
component_count: 2
name:            "hamming"
components:      [0x00, 0x24] = [XOR, POPCOUNT]
```

---

## Usage

### Python API

```python
from geocadesia import (
    FrozenShape,
    get_binary_shape,
    save_all_binary,
    print_opcode_table,
)

# Get a shape
shape = get_binary_shape("hamming")
print(shape.info())

# Save to file
shape.save("hamming.fsh")

# Load from file
loaded = FrozenShape.load("hamming.fsh")

# Save all shapes
save_all_binary("./shapes")

# Print opcode table
print_opcode_table()
```

### C API (Thor Integration)

```c
#include "trix_shapes.h"

// FrozenDB query using Hamming distance
size_t idx = trix_frozendb_query(signatures, query, n);

// Individual shapes
int dist = trix_hamming(a, b);
int bits = trix_popcount(x);
size_t min_idx = trix_argmin(values, n);
```

---

## FrozenDB Integration

The binary format enables FrozenDB to load shape definitions at runtime:

```
Query Pipeline:
┌────────┐     ┌────────────┐     ┌────────────┐
│ Input  │ ──→ │ XOR (0x00) │ ──→ │ POPCOUNT   │ ──→ distances
└────────┘     └────────────┘     │   (0x24)   │
                                  └────────────┘

                       ┌────────────┐
distances ───────────→ │ ARGMIN     │ ──→ match_idx
                       │   (0x84)   │
                       └────────────┘
```

The Thor hardware can load `xor.fsh`, `popcount.fsh`, and `argmin.fsh` and execute the complete query pipeline at 35 Tbits/sec.

---

## Directory Structure

```
shapes/
├── geocadesia/
│   ├── __init__.py      # Python package
│   ├── logic.py         # Logic kingdom implementations
│   ├── arithmetic.py    # Arithmetic kingdom implementations
│   ├── activation.py    # Activation functions
│   ├── normalization.py # Normalization layers
│   ├── pooling.py       # Pooling operations
│   ├── catalog.py       # Shape registry
│   └── binary.py        # Binary format support
├── bin/
│   ├── xor.fsh          # Binary shapes
│   ├── popcount.fsh
│   ├── hamming.fsh
│   └── ...
├── elements/
│   └── *.md             # Element documentation
├── compounds/
│   └── *.md             # Compound documentation
├── FROZENDB.md          # FrozenDB spec
├── BINARY_FORMAT.md     # This file
└── README.md            # Overview
```

---

## Future Extensions

1. **Bytecode Section**: Add implementation bytecode for portable execution
2. **Thor VCIX Binding**: Direct memory-mapped hardware execution
3. **Shape Pipelines**: Binary format for connected shape graphs
4. **Versioned Evolution**: Maintain backward compatibility as shapes evolve

---

*"Three formats: Python for development, Markdown for understanding, Binary for execution."*

*"It's all in the reflexes."*
