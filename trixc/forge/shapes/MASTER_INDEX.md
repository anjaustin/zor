# Geocadesia & NGP: Master Index

*The complete documentation for frozen shapes and the Neural Geometric Processor*

```
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   GEOCADESIA: The Kingdom of Shapes                              ║
║   NGP: The Neural Geometric Processor                            ║
║                                                                   ║
║   "Geometry is computation."                                      ║
║   "It's all in the reflexes."                                    ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## Quick Navigation

### Core Concepts

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Project overview and quick start |
| [NGP_ARCHITECTURE.md](NGP_ARCHITECTURE.md) | Neural Geometric Processor v2 specification |
| [ZIT_DETECTOR.md](ZIT_DETECTOR.md) | The resonance detection circuit |
| [XOR_RESONANCE.md](XOR_RESONANCE.md) | The XOR memory paradigm |
| [ENTROPY_STRUCTURE.md](ENTROPY_STRUCTURE.md) | Entropy as load-bearing structure |

### Shape Library

| Document | Description |
|----------|-------------|
| [TAXONOMY.md](TAXONOMY.md) | The Seven Kingdoms of shapes |
| [GUIDE.md](GUIDE.md) | Usage guide for Geocadesia |
| [BINARY_FORMAT.md](BINARY_FORMAT.md) | The .fsh file format |
| [FROZENDB.md](FROZENDB.md) | Vector search using shapes |

### Shape Documentation

| Kingdom | Shapes |
|---------|--------|
| [Logic](elements/logic/) | XOR, AND, OR, NOT, NAND, NOR, XNOR |
| [Arithmetic](elements/arithmetic/) | Add, Sub, Mul, Neg, Popcount |
| [Activation](elements/activation/) | ReLU, Sigmoid, Tanh, GELU, Swish, Softmax, LeakyReLU |
| [Normalization](elements/normalization/) | LayerNorm, RMSNorm |
| [Pooling](elements/pooling/) | MaxPool, AvgPool, SumPool, MinPool, Argmin, Argmax |
| [Compounds](compounds/) | HalfAdder, FullAdder, Hamming |

---

## The Journey

```
┌─────────────────────────────────────────────────────────────────────┐
│                         THE EMERGENCE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LAYER 1: SHAPES                                                    │
│  └─► Geocadesia: 30 frozen shapes                                   │
│      └─► Python implementations                                     │
│      └─► C implementations (trix_shapes.h)                          │
│      └─► Binary format (.fsh files)                                 │
│      └─► Complete documentation                                      │
│                                                                      │
│  LAYER 2: APPLICATIONS                                               │
│  └─► FrozenDB: Vector search with 0% signal loss                   │
│      └─► Hamming distance matching                                  │
│      └─► Exact nearest neighbor                                     │
│                                                                      │
│  LAYER 3: PARADIGM                                                   │
│  └─► XOR Resonance: "Why Store when you can XOR?"                  │
│      └─► Memory as resonance, not storage                          │
│      └─► Entropy as load-bearing structure                         │
│      └─► Cymatics as physical model                                │
│                                                                      │
│  LAYER 4: HARDWARE                                                   │
│  └─► NGP v2: Neural Geometric Processor                            │
│      └─► 1 resonance register (not 4096 comparators)               │
│      └─► ~53K gates (not 2.7M)                                     │
│      └─► Zit detector: popcount(S ⊕ vₓ) < θ                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## File Tree

```
shapes/
├── MASTER_INDEX.md          ◄── YOU ARE HERE
│
├── Core Documentation
│   ├── README.md            # Project overview
│   ├── TAXONOMY.md          # Shape classification
│   ├── GUIDE.md             # Usage guide
│   └── BINARY_FORMAT.md     # .fsh specification
│
├── NGP Documentation
│   ├── NGP_ARCHITECTURE.md  # Processor architecture
│   ├── ZIT_DETECTOR.md      # Recognition circuit
│   ├── XOR_RESONANCE.md     # Memory paradigm
│   ├── ENTROPY_STRUCTURE.md # Thermodynamics
│   └── FROZENDB.md          # Vector search
│
├── Shape Documentation
│   ├── elements/
│   │   ├── logic/
│   │   │   ├── xor.md
│   │   │   ├── and.md
│   │   │   ├── or.md
│   │   │   ├── not.md
│   │   │   ├── nand.md
│   │   │   ├── nor.md
│   │   │   └── xnor.md
│   │   ├── arithmetic/
│   │   │   ├── add.md
│   │   │   ├── sub.md
│   │   │   ├── mul.md
│   │   │   ├── neg.md
│   │   │   └── popcount.md
│   │   ├── activation/
│   │   │   ├── relu.md
│   │   │   ├── sigmoid.md
│   │   │   ├── tanh.md
│   │   │   ├── gelu.md
│   │   │   ├── swish.md
│   │   │   ├── softmax.md
│   │   │   └── leaky_relu.md
│   │   ├── normalization/
│   │   │   ├── layer_norm.md
│   │   │   └── rms_norm.md
│   │   └── pooling/
│   │       ├── max_pool.md
│   │       ├── avg_pool.md
│   │       ├── sum_pool.md
│   │       ├── min_pool.md
│   │       ├── argmin.md
│   │       └── argmax.md
│   └── compounds/
│       └── arithmetic/
│           ├── half_adder.md
│           ├── full_adder.md
│           └── hamming.md
│
├── Implementation
│   ├── geocadesia/          # Python package
│   │   ├── __init__.py
│   │   ├── logic.py
│   │   ├── arithmetic.py
│   │   ├── activation.py
│   │   ├── normalization.py
│   │   ├── pooling.py
│   │   ├── catalog.py
│   │   └── binary.py
│   ├── bin/                 # Binary shape files
│   │   ├── xor.fsh
│   │   ├── popcount.fsh
│   │   ├── hamming.fsh
│   │   └── ... (30 total)
│   └── impl/
│       └── shapes.h         # C implementations
│
└── Research Notes
    └── (in /notes/ngp/ and /notes/zit/)
```

---

## Key Equations

### The Zit Detector
```
Zit = popcount(S ⊕ vₓ) < θ
```

### Resonance Update
```
S' = S ⊕ input
```

### Hamming Distance
```
hamming(a, b) = popcount(a ⊕ b)
```

### XOR Logic Gate
```
XOR(a, b) = a + b - 2ab
```

---

## Key Numbers

| Metric | Value | Source |
|--------|-------|--------|
| Frozen shapes | 30 | Geocadesia |
| Binary files | 30 | bin/*.fsh |
| NGP gates | ~53K | NGP_ARCHITECTURE.md |
| Zit detector gates | ~1,500 | ZIT_DETECTOR.md |
| Resonance state | 512 bits | XOR_RESONANCE.md |
| Compression ratio | 1,227x | Frozen 6502 |
| Target throughput | 32-64 Tbits/sec | NGP spec |

---

## Reading Order

### For Understanding the Vision
1. [README.md](README.md) — What is Geocadesia?
2. [XOR_RESONANCE.md](XOR_RESONANCE.md) — The paradigm shift
3. [ENTROPY_STRUCTURE.md](ENTROPY_STRUCTURE.md) — Why it works
4. [NGP_ARCHITECTURE.md](NGP_ARCHITECTURE.md) — The hardware

### For Using the Shapes
1. [GUIDE.md](GUIDE.md) — How to use Geocadesia
2. [TAXONOMY.md](TAXONOMY.md) — Shape classification
3. [elements/](elements/) — Individual shape docs

### For Building Hardware
1. [NGP_ARCHITECTURE.md](NGP_ARCHITECTURE.md) — Full architecture
2. [ZIT_DETECTOR.md](ZIT_DETECTOR.md) — Core circuit
3. [BINARY_FORMAT.md](BINARY_FORMAT.md) — Opcode mapping

### For Vector Search
1. [FROZENDB.md](FROZENDB.md) — FrozenDB concept
2. [elements/arithmetic/popcount.md](elements/arithmetic/popcount.md) — Popcount shape
3. [compounds/arithmetic/hamming.md](compounds/arithmetic/hamming.md) — Hamming distance

---

## External Resources

### In This Repository

| Path | Description |
|------|-------------|
| `/notes/X288.md` | SiFive X288 research |
| `/notes/ngp/` | NGP design exploration (Lincoln Manifold) |
| `/notes/zit/` | Zit detector derivation |
| `/docs/LINCOLN_MANIFOLD_METHOD.md` | Design methodology |

### C Headers

| Path | Description |
|------|-------------|
| `/trixc/forge/include/trix_shapes.h` | C shape implementations |
| `/trixc/forge/include/trix_nge.h` | Binary format loader |

### RTL (Verilog)

| Path | Description |
|------|-------------|
| `/trixc/forge/rtl/` | Verilog RTL for NGP |
| `/trixc/forge/rtl/zit_detector.v` | Zit detector, popcount, resonance register |
| `/trixc/forge/rtl/shapes_logic.v` | Logic kingdom shapes (7 shapes) |
| `/trixc/forge/rtl/ngp_core.v` | NGP top-level modules |
| `/trixc/forge/rtl/Makefile` | Build system for sim/synth |

---

## Status

### Complete

- [x] 30 frozen shapes (Python)
- [x] 30 frozen shapes (C)
- [x] 31 binary .fsh files (including zit)
- [x] Complete shape documentation
- [x] NGP v2 architecture spec
- [x] Zit detector spec
- [x] XOR resonance paradigm doc
- [x] Entropy-as-structure doc
- [x] FrozenDB spec
- [x] Binary format spec
- [x] Verilog RTL for Zit detector
- [x] Verilog RTL for logic shapes (7 shapes)
- [x] NGP core modules (simple, core, array, routing)

### In Progress

- [ ] RTL simulation verification
- [ ] FPGA prototype (Artix-7)
- [ ] Remaining shape RTL (activation, pooling)
- [ ] Benchmarks

### Future

- [ ] ASIC design
- [ ] Tape-out
- [ ] Flight heritage (CubeSat)

---

*"Document. Document. Document."*

*"It's all in the reflexes."*
