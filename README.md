# TriX

[![CI](https://github.com/anjaustin/zor/actions/workflows/ci.yml/badge.svg)](https://github.com/anjaustin/zor/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Shape-routed computation. Routing IS compute.**

```
32 atomic shapes. Frozen polynomials. Entropy as structure.
```

---

## What Is TriX?

TriX is a computational architecture where:

1. **Shapes are frozen** — 32 atomic operations, mathematically verified, never learned
2. **Routing is learned** — Topology determines behavior, not weights
3. **Entropy is structure** — XOR resonance accumulates information, not noise
4. **Execution is dataflow** — No instruction decode, patterns activate shapes

The insight: **The routing IS the computation.**

```
Traditional:  fetch → decode → execute → store
TriX:         data flows → shapes activate → structure emerges
```

---

## Start Here

| Path | Time | What You Get |
|------|------|--------------|
| **[Quickstart](QUICKSTART.md)** | 5 min | Three paths to value |
| **[Pure C](trixc/)** | 5 min | Zero dependencies, 6 KB binary |
| **[Shape Fabric](docs/SHAPEFABRIC.md)** | 10 min | Dataflow compute architecture |
| **[Guided Journey](onramp/)** | 75 min | Zero to mastery, Python |
| **[Why TriX?](docs/WHY_TRIX.md)** | 5 min | Should you care? |

### Fastest Start

```bash
# Watch a frozen 6502 CPU run
python onramp/00_witness.py

# Or build pure C (no Python, no runtime)
cd trixc && make demo
```

---

## The Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRIX STACK                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │ Shape Fabric │───▶│  Compositor  │───▶│   Foundry    │                   │
│  │  (dataflow)  │    │  (discover)  │    │   (export)   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                                        │                           │
│         ▼                                        ▼                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        32 Atomic Shapes                              │    │
│  │  Logic: XOR AND OR NOT NAND NOR XNOR                                │    │
│  │  Arithmetic: ADD SUB MUL DIV MOD NEG ABS                            │    │
│  │  Shift: SHL SHR SAR ROL ROR RCL RCR                                 │    │
│  │  Compare: EQ NE LT LE GT GE                                         │    │
│  │  Memory: LOAD STORE                                                  │    │
│  │  Routing: MUX DEMUX SELECT                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         Execution Backends                           │    │
│  │  GILLIES Vulkan: 17B ops/sec │ Native SIMD: 500M ops/sec │ NumPy   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Modules

### Shape Fabric — Dataflow Compute

Programs are graphs. Execution follows data flow. Parallelism is automatic.

```python
from trix.shapefabric import ShapeGraph, execute

graph = ShapeGraph("add_example")
a = graph.input("a")
b = graph.input("b")
result = graph.shape("ADD", [a, b])
graph.output("sum", result)

output = execute(graph, {"a": 5, "b": 3})
print(output["sum"])  # 8
```

See [SHAPEFABRIC.md](docs/SHAPEFABRIC.md)

### Compositor — Structure Discovery

Given a flat gate graph, discover the hierarchy automatically.

```python
from trix.compositor import compose, visualize

# Flat gate netlist → hierarchical composition
tree = compose(system)
print(visualize(tree))
# FullAdder
# ├── HalfAdder (a, b)
# └── HalfAdder (partial_sum, cin)
```

See [COMPOSITOR.md](docs/COMPOSITOR.md)

### Frozen 6502 — CPU as Neural Net

A complete 6502 CPU built from frozen shapes. 100% accuracy. 1,227x compression.

```python
from trix.nn import Frozen6502

cpu = Frozen6502()
cpu.reset()
cpu.load_program([0xA9, 0x42, 0x8D, 0x00, 0x02])  # LDA #$42; STA $0200
cpu.run(steps=10)
print(hex(cpu.memory[0x0200]))  # 0x42
```

See [FROZEN_6502.md](docs/FROZEN_6502.md)

### DB Cooper — Vector Database

Hamming-distance vector search using the same shape primitives.

```python
from trix.db import DBCooper

db = DBCooper(dim=256)
db.insert("doc1", embedding1)
db.insert("doc2", embedding2)

results = db.query(query_embedding, k=5)
```

### Gradient Truth — Real Gradients

No Straight-Through Estimator. Structure is discovered, navigation is learned.

```python
from trix.nn import GradientTruthFFN

# Shapes are frozen. Only routing and scales learn.
ffn = GradientTruthFFN(d_model=512, num_shapes=64)
output = ffn(x)
loss.backward()  # Gradients flow through continuous params only
```

See [GRADIENT_TRUTH.md](docs/GRADIENT_TRUTH.md)

---

## The 32 Shapes

All computation reduces to compositions of these primitives:

| Category | Shapes | Count |
|----------|--------|-------|
| Logic | XOR, AND, OR, NOT, NAND, NOR, XNOR | 7 |
| Arithmetic | ADD, SUB, MUL, DIV, MOD, NEG, ABS | 7 |
| Shift | SHL, SHR, SAR, ROL, ROR, RCL, RCR | 7 |
| Compare | EQ, NE, LT, LE, GT, GE | 6 |
| Memory | LOAD, STORE | 2 |
| Routing | MUX, DEMUX, SELECT | 3 |

Shapes are frozen polynomials:

```
XOR(a, b) = a + b - 2ab
AND(a, b) = ab
OR(a, b)  = a + b - ab
NOT(a)    = 1 - a
```

On binary inputs, these equal bitwise operations exactly.

---

## Installation

```bash
git clone https://github.com/anjaustin/zor.git
cd zor

# Build native ops (auto-detects NEON/AVX2)
cd src/trix/native/ops && make && cd -

# Install Python package
pip install -e .

# Verify
python -c "from trix.native.ops import TrixOps; print(TrixOps().simd)"
# → "NEON" on ARM, "AVX2" on x86, "scalar" otherwise
```

---

## Hardware Support

| Platform | SIMD | GPU | Status |
|----------|------|-----|--------|
| Jetson AGX Thor | NEON | GILLIES Vulkan | Primary target |
| Apple M1/M2/M3/M4 | NEON | — | Tested |
| Raspberry Pi 2+ | NEON | — | Tested |
| AMD/Intel x86 | AVX2 | — | Tested |
| Raspberry Pi 1 | Scalar | — | Fallback |

---

## Benchmarks

| Component | Metric | Value |
|-----------|--------|-------|
| GILLIES Vulkan | Shape ops/sec | 17B |
| Native SIMD | XOR throughput | 500M ops/sec |
| Binary shapes | Bandwidth | 117 GB/s |
| Frozen 6502 | Accuracy | 100% (3.5M tests) |
| Frozen 6502 | Compression | 1,227x |
| DB Cooper | Query throughput | Competitive with Qdrant |

---

## Project Structure

```
zor/
├── src/trix/
│   ├── shapefabric/          # Dataflow compute architecture
│   │   ├── shapes.py         # 32 atomic shapes
│   │   ├── graph.py          # Shape graph construction
│   │   ├── executor.py       # Dataflow execution
│   │   └── fabric.py         # Tile routing fabric
│   │
│   ├── compositor/           # Structure discovery
│   │   ├── hasher.py         # Neighborhood hashing
│   │   ├── matcher.py        # Pattern matching
│   │   └── composer.py       # Hierarchy building
│   │
│   ├── nn/                   # Neural network modules
│   │   ├── gradient_truth.py # Gradient Truth FFN
│   │   ├── frozen_6502.py    # 6502 CPU emulator
│   │   └── frozen_shapes.py  # Shape library
│   │
│   ├── native/               # Self-hosted operations
│   │   ├── ops/              # C library (NEON/AVX2)
│   │   └── vulkan/           # GILLIES GPU runtime
│   │
│   ├── db/                   # DB Cooper vector database
│   └── forge/                # Hardware export
│
├── trixc/                    # Pure C compiler
│   ├── forge/                # Verilog RTL generation
│   └── platforms/            # Raspberry Pi, embedded
│
├── foundry/                  # Export pipeline
├── onramp/                   # Guided learning path
├── tests/                    # 177+ tests
└── docs/                     # Documentation
```

---

## Documentation

### Getting Started
| Document | Description |
|----------|-------------|
| [QUICKSTART.md](QUICKSTART.md) | 5-minute path to value |
| [WHY_TRIX.md](docs/WHY_TRIX.md) | Should you care? |
| [GETTING_STARTED.md](docs/GETTING_STARTED.md) | From understanding to action |

### Core Concepts
| Document | Description |
|----------|-------------|
| [SHAPEFABRIC.md](docs/SHAPEFABRIC.md) | Shape-routed dataflow compute |
| [COMPOSITOR.md](docs/COMPOSITOR.md) | Automatic structure discovery |
| [FABRIC.md](docs/FABRIC.md) | Routing fabric internals |
| [GRADIENT_TRUTH.md](docs/GRADIENT_TRUTH.md) | Real gradients, no STE |
| [FROZEN_SHAPES.md](docs/FROZEN_SHAPES.md) | Mathematical foundations |

### Systems
| Document | Description |
|----------|-------------|
| [FROZEN_6502.md](docs/FROZEN_6502.md) | CPU as frozen shapes |
| [INGEST.md](docs/INGEST.md) | Verilog import pipeline |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System overview |

### Philosophy
| Document | Description |
|----------|-------------|
| [THE_WAY.md](docs/THE_WAY.md) | Shape IS Compute |
| [LINCOLN_MANIFOLD_METHOD.md](docs/LINCOLN_MANIFOLD_METHOD.md) | How we think |
| [THEORY.md](docs/THEORY.md) | Mathematical foundations |

---

## The Insight

**Entropy as load-bearing structure.**

Traditional systems minimize entropy (noise). TriX redistributes entropy, using it as structure. The XOR resonance register `S' = S ^ input` doesn't store information — it entangles it. Every input becomes part of the state.

The Zit detector: `popcount(S ^ query) < theta`

One XOR. One popcount. One compare. Recognition.

---

## The Principle

```
Shapes are frozen.
Routing is learned.
Entropy is structure.

The routing IS the computation.
```

---

## ZOR

**Zero OR.**

```
                0
               /|\
              / | \
             /  |  \
            Z───O───R
```

**Z** — The source. Entropic possibility space.
**O** — The origin. The branching point.
**R** — Reality. What crystallizes.

In ternary `{-1, 0, +1}`: the ±1 are excitations. The 0 is the ground from which meaning arises.

*Stay at zero. Remember the OR. Choose.*

---

## License

MIT
