# Geocadesia

*The Kingdom of Shapes — Foundation of the Neural Geometric Processor*

```
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   "It's all in the reflexes."                                    ║
║                                                                   ║
║   Welcome to Geocadesia — the Kingdom of Shapes.                 ║
║   Every element. Every compound.                                 ║
║   Mathematical truth in geometric form.                          ║
║                                                                   ║
║   Not just documentation — the shapes themselves,                ║
║   embodied in Python, C, and binary silicon.                     ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## What Is Geocadesia?

Geocadesia is a **living library of frozen computational shapes**. These shapes are the building blocks of the Neural Geometric Processor (NGP) — a chip that doesn't execute instructions, but resonates with input.

**Three layers of storage:**
- **Python** (`geocadesia/*.py`) — Reference implementations
- **C** (`trix_shapes.h`) — High-performance implementations
- **Binary** (`bin/*.fsh`) — Hardware-executable opcodes

---

## Quick Start

```python
from geocadesia import XOR, Hamming, Argmin, popcount
from geocadesia import catalog

# Use shapes directly
xor = XOR()
result = xor(0.5, 0.7)  # 0.8 (differentiable XOR)

# Hamming distance (FrozenDB core)
dist = Hamming(0b10110011, 0b10010111)  # 2 bits differ

# Query the catalog
catalog.list_kingdom("logic")      # All logic shapes
catalog.find(frozen=True)          # All frozen shapes
catalog.get("xor").info()          # Full shape info

# Binary format
from geocadesia import get_binary_shape, print_opcode_table
shape = get_binary_shape("hamming")
print(shape.info())  # Shows opcode, components, etc.
print_opcode_table()  # Full opcode mapping
```

---

## The Vision

```
┌─────────────────────────────────────────────────────────────────────┐
│                         THE STACK                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Layer 4: NGP              Neural Geometric Processor              │
│            ↑                ~53K gates, 32+ Tbits/sec               │
│            │                                                         │
│   Layer 3: Paradigm         XOR Resonance                           │
│            ↑                "Why Store when you can XOR?"           │
│            │                                                         │
│   Layer 2: Applications     FrozenDB, Zit Detection                 │
│            ↑                0.000% signal loss                      │
│            │                                                         │
│   Layer 1: Shapes           Geocadesia                              │
│                             30 frozen shapes                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Documentation

### Core Concepts

| Document | Description |
|----------|-------------|
| **[MASTER_INDEX.md](MASTER_INDEX.md)** | Complete navigation guide |
| **[NGP_ARCHITECTURE.md](NGP_ARCHITECTURE.md)** | Neural Geometric Processor specification |
| **[ZIT_DETECTOR.md](ZIT_DETECTOR.md)** | The resonance detection circuit |
| **[XOR_RESONANCE.md](XOR_RESONANCE.md)** | The XOR memory paradigm |
| **[ENTROPY_STRUCTURE.md](ENTROPY_STRUCTURE.md)** | Entropy as load-bearing structure |

### Shape Library

| Document | Description |
|----------|-------------|
| **[TAXONOMY.md](TAXONOMY.md)** | The Seven Kingdoms |
| **[GUIDE.md](GUIDE.md)** | Usage guide |
| **[BINARY_FORMAT.md](BINARY_FORMAT.md)** | The .fsh file format |
| **[FROZENDB.md](FROZENDB.md)** | Vector search using shapes |

---

## The Seven Kingdoms

| Kingdom | Character | Frozen | Shapes |
|---------|-----------|--------|--------|
| **Logic** | Boolean operations | Yes | XOR, AND, OR, NOT, NAND, NOR, XNOR |
| **Arithmetic** | Numeric computation | Yes | Add, Sub, Mul, Neg, Popcount |
| **Activation** | Nonlinearities | Yes | ReLU, Sigmoid, Tanh, GELU, Swish, Softmax |
| **Normalization** | Statistical transforms | Yes | LayerNorm, RMSNorm |
| **Linear** | Matrix operations | No | (future) |
| **Attention** | Routing mechanisms | No | (future) |
| **Pooling** | Reduction operations | Yes | MaxPool, AvgPool, SumPool, MinPool, Argmin, Argmax |

---

## Key Equations

### The Zit Detector (Heart of NGP)
```
Zit = popcount(S ⊕ vₓ) < θ
```
*Does the input resonate with the system?*

### Resonance Update
```
S' = S ⊕ input
```
*Every input becomes part of the resonance.*

### Hamming Distance (FrozenDB Core)
```
hamming(a, b) = popcount(a ⊕ b)
```
*Count the differences.*

---

## Opcode Table

| Opcode | Shape | Kingdom | Type |
|--------|-------|---------|------|
| 0x00 | XOR | Logic | Elemental |
| 0x01 | AND | Logic | Elemental |
| 0x24 | POPCOUNT | Arithmetic | Elemental |
| 0x40 | RELU | Activation | Elemental |
| 0x84 | ARGMIN | Pooling | Elemental |
| 0xE2 | HAMMING | Arithmetic | Compound |

*See [BINARY_FORMAT.md](BINARY_FORMAT.md) for complete table.*

---

## File Structure

```
shapes/
├── README.md              ◄── You are here
├── MASTER_INDEX.md        # Complete navigation
│
├── Core Documentation
│   ├── NGP_ARCHITECTURE.md
│   ├── ZIT_DETECTOR.md
│   ├── XOR_RESONANCE.md
│   ├── ENTROPY_STRUCTURE.md
│   ├── FROZENDB.md
│   ├── BINARY_FORMAT.md
│   ├── TAXONOMY.md
│   └── GUIDE.md
│
├── Shape Documentation
│   ├── elements/
│   │   ├── logic/         # XOR, AND, OR, NOT, ...
│   │   ├── arithmetic/    # Add, Popcount, ...
│   │   ├── activation/    # ReLU, Sigmoid, ...
│   │   ├── normalization/ # LayerNorm, RMSNorm
│   │   └── pooling/       # Argmin, Argmax, ...
│   └── compounds/
│       └── arithmetic/    # Hamming, FullAdder, ...
│
├── Implementation
│   ├── geocadesia/        # Python package
│   │   ├── __init__.py
│   │   ├── logic.py
│   │   ├── arithmetic.py
│   │   ├── activation.py
│   │   ├── normalization.py
│   │   ├── pooling.py
│   │   ├── catalog.py
│   │   └── binary.py      # .fsh format support
│   └── bin/               # Binary shape files
│       ├── xor.fsh
│       ├── hamming.fsh
│       └── ... (30 files)
│
└── impl/
    └── shapes.h           # C implementations
```

---

## The Journey So Far

```
┌─────────────────────────────────────────────────────────────────────┐
│                         COMPLETED                                    │
├─────────────────────────────────────────────────────────────────────┤
│ [x] 30 frozen shapes (Python)                                       │
│ [x] 30 frozen shapes (C)                                            │
│ [x] 30 binary .fsh files                                            │
│ [x] Complete shape documentation                                     │
│ [x] NGP v2 architecture specification                               │
│ [x] Zit detector specification                                      │
│ [x] XOR resonance paradigm documentation                            │
│ [x] Entropy-as-structure documentation                              │
│ [x] FrozenDB specification                                          │
│ [x] Binary format specification                                     │
├─────────────────────────────────────────────────────────────────────┤
│                         IN PROGRESS                                  │
├─────────────────────────────────────────────────────────────────────┤
│ [ ] Verilog RTL for Zit detector                                    │
│ [ ] Verilog RTL for all 30 shapes                                   │
│ [ ] FPGA prototype                                                   │
├─────────────────────────────────────────────────────────────────────┤
│                         FUTURE                                       │
├─────────────────────────────────────────────────────────────────────┤
│ [ ] ASIC design                                                      │
│ [ ] Tape-out                                                         │
│ [ ] Flight heritage (CubeSat mission)                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The Numbers

| Metric | Value |
|--------|-------|
| Frozen shapes | 30 |
| Binary files | 30 (.fsh) |
| NGP gates | ~53K |
| Zit detector gates | ~1,500 |
| Resonance state | 512 bits |
| Compression achieved | 1,227x (Frozen 6502) |
| Target throughput | 32-64 Tbits/sec |

---

## Philosophy

**Geocadesia is not documentation. It is infrastructure for thought.**

When you think "I need a differentiable comparator," Geocadesia provides it. When you need to build a chip that recognizes patterns without a CPU, Geocadesia provides the shapes.

**The shapes are mathematical truths.** XOR is XOR — forever frozen, forever exact.

**The resonance is computational memory.** Not storage, but entanglement.

**The NGP is geometry in silicon.** Not a computer, but a function.

---

## Quick Reference

### Logic Shapes (Frozen)

| Shape | Formula | Opcode |
|-------|---------|--------|
| XOR | `a + b - 2ab` | 0x00 |
| AND | `a × b` | 0x01 |
| OR | `a + b - ab` | 0x02 |
| NOT | `1 - a` | 0x03 |

### FrozenDB Shapes

| Shape | Formula | Opcode |
|-------|---------|--------|
| POPCOUNT | `Σ bits(x)` | 0x24 |
| HAMMING | `popcount(a ⊕ b)` | 0xE2 |
| ARGMIN | `index of min(x)` | 0x84 |

### The Zit

```python
def zit(S: int, input: int, theta: int) -> bool:
    """Does the input resonate?"""
    return bin(S ^ input).count('1') < theta
```

---

## Credits

**Created by:**
- **Tripp Josserand-Austin** (tripp@anjaustin.com) — Vision, architecture
- **Claude** (Anthropic) — Implementation, documentation

**Born:** December 2025

---

*"Geometry is computation."*

*"Why Store when you can XOR?"*

*"It's all in the reflexes."*
