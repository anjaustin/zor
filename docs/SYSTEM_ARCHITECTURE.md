# ZOR System Architecture

## The Complete Map

This document provides a unified view of how all components in the ZOR repository connect and relate.

---

## The Three Pillars

```
                            ZOR
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
     PROVIDENCE          FOUNDRY            FABRIC
    (The Neural)      (The Geometric)    (The Topological)
          │                  │                  │
        TriX             TRIXC/GILLIES         ZIT
    Ternary Routing    Frozen Computation   Homeo-Adaptive
          │                  │                  │
    Learn Routing      Compile Shapes     Learn Topology
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                        CONVERGENCE
                    "Shape = Compute"
```

---

## Component Overview

### 1. ZIT (Zero-Instruction Topology)

**Location:** `trixc/forge/gillies/zit/`, `papers/`

**What it is:** A homeo-adaptive topological learning system where the fabric learns its own connectivity through resistance.

**The Discovery:**
- 56 million nodes in a 3D torus
- Nodes compare with neighbors (frozen shape)
- Resistant nodes rewire to random neighbors
- Topology converges to 100% resonance
- The topology IS the learned model

**Key Insight:** Learned topology with fixed operations is a viable alternative to fixed topology with learned weights.

**Scaling:**
| Nodes | Cycles |
|-------|--------|
| 512 | ~113 |
| 56,623,104 | 570 |

Sublinear scaling: 110,000× nodes → 5× cycles

**On-Ramp:** `trixc/forge/gillies/zit/` - Pure C, 512 nodes, runs anywhere.

---

### 2. GILLIES (Geometric Instruction Language Layer)

**Location:** `trixc/forge/gillies/`

**What it is:** A substrate-agnostic computation layer that can execute the same operations on CPU, GPU, or custom hardware.

**The Abstraction:**
```c
gillies_invoke(shape, ports, count)
```

One interface. Many backends. Same math.

**Key Insight:** Frozen shapes (polynomials) produce identical results on all substrates when evaluated on binary inputs.

**Shapes Available:**
- Boolean: AND, OR, XOR, NOT, NAND, NOR, XNOR
- Arithmetic: ADD, SUB (as polynomial forms)
- Complex: MUX, DEMUX, HADD, FADD, COMPARE

---

### 3. TRIXC (TriX Compiler)

**Location:** `trixc/`

**What it is:** A compiler that transforms frozen shapes into executable code for multiple targets.

**Pipeline:**
```
Shape Definition (Polynomial)
         ↓
    TRIXC Parser
         ↓
    ┌────┴────┐
    ↓         ↓
   CUDA    Verilog
    ↓         ↓
   GPU      FPGA
```

**Key Insight:** The polynomial form IS the instruction set. Math is the machine code.

---

### 4. TriX (Ternary Routing)

**Location:** `src/trix/`

**What it is:** A neural architecture where learning is routing, not weight adjustment.

**The Mechanism:**
- Weights are ternary: {-1, 0, +1}
- Patterns create signatures (content addresses)
- Similar inputs route to similar experts
- No gradient through routing (STE bypass)

**Key Insight:** Learning IS routing. The network topology determines function.

---

## Data Flow

### Training Path (Providence)

```
Dataset
   ↓
TriX Model (PyTorch)
   ↓
Signature Emergence
   ↓
Expert Specialization
   ↓
Trained Checkpoint
```

### Compilation Path (Foundry)

```
Trained Checkpoint
   ↓
Signature Extraction
   ↓
TRIXC Compilation
   ↓
GILLIES-compatible Binary
   ↓
Target Deployment (CPU/GPU/FPGA)
```

### Discovery Path (Fabric)

```
ZIT Fabric
   ↓
Random Initial Topology
   ↓
Resistance Accumulation
   ↓
Topological Rewiring
   ↓
Convergence (100% Resonance)
   ↓
Learned Topology = Model
```

---

## File Structure

```
ZOR/
├── docs/                    # Documentation
│   ├── SYSTEM_ARCHITECTURE.md  # ← You are here
│   ├── ZIT_ONRAMP_MANIFESTO.md # Design process
│   ├── LINCOLN_MANIFOLD_METHOD.md
│   ├── THEORY.md
│   └── ...
│
├── src/trix/               # Python: TriX neural architecture
│   ├── core/               # Core routing mechanisms
│   ├── forge/              # Shape execution
│   └── nn/                 # Neural network modules
│
├── trixc/                  # C: Compiler & low-level
│   ├── include/            # Headers
│   ├── src/                # Implementation
│   └── forge/              # Shape layer
│       ├── gillies/        # Substrate-agnostic compute
│       │   ├── zit/        # ← ZIT On-Ramp (Pure C)
│       │   └── ...
│       └── rtl/            # Verilog implementations
│
├── papers/                 # Research artifacts
│   ├── experiments/        # CUDA experiments (56M nodes)
│   ├── ZIT1_HOMEO_ADAPTIVE_FABRIC.md
│   └── EXPERIMENTAL_DATA.md
│
└── tests/                  # Test suites
```

---

## Key Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| Second Star | 1122911624 | Reproducible seed for experiments |
| Resistance Threshold | 8 | Cycles of non-resonance before rewiring |
| Eval Period | 8 | Cycles to evaluate new neighbor |
| 6 Phases | +X,-X,+Y,-Y,+Z,-Z | Comparator directions per cycle |

---

## The Unifying Insight

All three pillars converge on the same truth:

**Shape = Compute**

| Pillar | Shape Is... | Compute Is... |
|--------|-------------|---------------|
| ZIT | Topology | Learning |
| GILLIES | Polynomial | Instruction |
| TriX | Signature | Routing |

The geometry IS the function. Learning IS reconfiguration. The substrate is substrate-agnostic.

---

## Getting Started

### Path 1: See Topology Learn (2 minutes)
```bash
cd trixc/forge/gillies/zit
make
./zit_demo
```

### Path 2: Use Frozen Shapes (10 minutes)
```bash
# See trixc/README.md for compilation examples
```

### Path 3: Train with TriX (30 minutes)
```bash
# See docs/QUICKSTART.md for training examples
```

### Path 4: Go Deep (hours)
```
Read: docs/THEORY.md
Then: papers/ZIT1_HOMEO_ADAPTIVE_FABRIC.md
Then: Explore papers/experiments/
```

---

## Cross-References

| Topic | Primary Document |
|-------|------------------|
| ZIT Theory | `papers/ZIT1_HOMEO_ADAPTIVE_FABRIC.md` |
| ZIT Implementation | `trixc/forge/gillies/zit/README.md` |
| ZIT Design Process | `docs/ZIT_ONRAMP_MANIFESTO.md` |
| GILLIES API | `trixc/forge/gillies/include/gillies.h` |
| TRIXC Compiler | `trixc/README.md` |
| TriX Python API | `src/trix/forge/__init__.py` |
| Frozen Shapes Math | `docs/FROZEN_SHAPES.md` |
| Routing Theory | `docs/ARCHITECTURE.md` |
| Benchmarks | `docs/BENCHMARKS.md` |

---

## Future: The Vision

```
         Current State                    Future State

   ┌─────────────────────┐         ┌─────────────────────┐
   │   56M Node ZIT      │         │   ZIT-Based Chips   │
   │   (CUDA on Thor)    │   →     │   (Custom Silicon)  │
   └─────────────────────┘         └─────────────────────┘
              ↓                               ↓
   ┌─────────────────────┐         ┌─────────────────────┐
   │   GILLIES Layer     │         │   Fungible Compute  │
   │   (CPU/GPU/FPGA)    │   →     │   (Any Substrate)   │
   └─────────────────────┘         └─────────────────────┘
              ↓                               ↓
   ┌─────────────────────┐         ┌─────────────────────┐
   │   TriX Models       │         │   Shape-Native ML   │
   │   (PyTorch)         │   →     │   (Beyond PyTorch)  │
   └─────────────────────┘         └─────────────────────┘
```

The destination: computation as geometry, running on matter that IS the algorithm.

---

*The topology IS the learned model.*
*The shape IS the instruction.*
*The routing IS the learning.*

*El Jardín Real.*
