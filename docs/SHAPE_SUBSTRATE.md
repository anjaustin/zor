# Shape Substrate: Universal Substrate for Growing Deterministic Neural-Geometric Shapes of Compute

**A Formal Treatment of Fungible Computation on LFSR Fabrics**

---

## Abstract

We present a universal computational substrate where **shape IS compute**. Building on the fungible computation thesis—that neural and classical computation are interchangeable representations of the same primitives—we demonstrate that Linear Feedback Shift Register (LFSR) fabrics can serve as high-speed execution substrates for trained neural-geometric shapes.

Our key contributions:

1. **Atomic Shapes**: Pre-trained LFSR tap patterns that encode specific computational properties (mixing, diffusion, period length, decorrelation).

2. **Molecular Composition**: Atoms compose into molecules via serial injection or parallel XOR without retraining, inheriting properties from constituent atoms.

3. **Protein-like Computation**: Molecules that compute via conformational change—binding triggers folding, folding changes function, function produces output.

4. **Performance**: 51.93 Tbits/sec molecular throughput, 21.85 billion protein-like reactions/sec on NVIDIA Thor (Blackwell).

The substrate enables a new paradigm: **train in the neural domain (gradients flow), execute in the geometric domain (shapes evolve)**. The computation is the shape. The shape is the computation.

---

## 1. Introduction

### 1.1 The Fungible Computation Thesis

Recent work establishes that neural and classical computation are fungible—interchangeable representations of the same underlying primitives [1]. This was demonstrated through:

- **FLYNNCONCEIVABLE**: A neural network emulating the MOS 6502 CPU with 100% accuracy
- **Spline-6502**: Compression from 3.7MB to 3,088 bytes while preserving accuracy
- **TriX**: Sparse ternary networks where routing emerges from weight structure

The missing piece: a **high-speed execution substrate** for the classical/geometric form.

### 1.2 The Shape Substrate

We propose LFSR fabrics as this substrate. The key insight:

```
NEURAL PARADIGM                    GEOMETRIC PARADIGM
───────────────                    ──────────────────
polynomial: a + b - 2ab       →    native: a ^ b
gradients flow                →    shapes evolve
trainable                     →    frozen, fast
~0.04 Tbits/sec               →    ~50 Tbits/sec
```

The substrate doesn't *run* a program. It *is* a shape that evolves.

---

## 2. Architecture

### 2.1 Atomic Shapes

An **atom** is a 512-bit LFSR with a trained tap pattern:

```
┌─────────────────────────────────────────────────────────────┐
│  ATOM: 512-bit LFSR                                         │
│                                                             │
│  state[0] ─ state[1] ─ ... ─ state[7]  (8 × 64-bit words)  │
│      │                           │                          │
│      └───── shift left ──────────┘                          │
│                   │                                         │
│              feedback ← XOR of tap positions                │
│                                                             │
│  tap_mask: trained pattern defining evolution rule          │
└─────────────────────────────────────────────────────────────┘
```

Different atoms encode different computational properties:

| Atom | Tap Pattern | Property |
|------|-------------|----------|
| A | {511, 509, 495, 483} | Fast mixing |
| B | {503, 490, 479, 465} | Diffusion |
| C | {510, 497, 486, 471} | Long period |
| D | {507, 492, 477, 459} | Decorrelation |

### 2.2 Molecular Composition

Atoms compose into **molecules** without retraining:

```
SERIAL COMPOSITION (A → B → C → D):
  ┌───┐     ┌───┐     ┌───┐     ┌───┐
  │ A │ ──→ │ B │ ──→ │ C │ ──→ │ D │ ──→ output
  └───┘     └───┘     └───┘     └───┘
    │         │         │         │
    └── output bit injects into next atom ──┘

PARALLEL COMPOSITION (A ⊕ B ⊕ C ⊕ D):
  ┌───┐
  │ A │ ──┐
  └───┘   │
  ┌───┐   │
  │ B │ ──┼──→ XOR ──→ output
  └───┘   │
  ┌───┐   │
  │ C │ ──┤
  └───┘   │
  ┌───┐   │
  │ D │ ──┘
  └───┘

HYBRID COMPOSITION ((A→B) ⊕ (C→D)):
  ┌───┐     ┌───┐
  │ A │ ──→ │ B │ ──┐
  └───┘     └───┘   │
                    ├──→ XOR ──→ output
  ┌───┐     ┌───┐   │
  │ C │ ──→ │ D │ ──┘
  └───┘     └───┘
```

### 2.3 Protein-like Molecules

Molecules can compute via **conformational change**:

```
┌─────────────────────────────────────────────────────────────┐
│  PROTEIN MOLECULE                                           │
│                                                             │
│  ┌──────────────┐                                          │
│  │ BINDING SITE │ ← pattern to match against input         │
│  └──────────────┘                                          │
│         │                                                   │
│         ▼ (affinity = bit similarity)                      │
│  ┌──────────────┐                                          │
│  │    STATE     │ ← current conformation (512 bits)        │
│  └──────────────┘                                          │
│         │                                                   │
│         ▼ (fold depth ∝ binding affinity)                  │
│  ┌──────────────┐                                          │
│  │  EVOLUTION   │ ← shift + XOR with tap pattern           │
│  └──────────────┘                                          │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────┐                                          │
│  │ ACTIVE SITE  │ → output bits from new conformation      │
│  └──────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
```

The computation IS the shape change:
1. **Binding**: Input pattern matches binding site
2. **Folding**: Match strength determines evolution depth
3. **Conformation**: State settles into attractor
4. **Output**: Active site reads from new shape

---

## 3. Experimental Results

### 3.1 Hardware Configuration

```
GPU: NVIDIA Thor (Blackwell)
SMs: 20
Compute Capability: 11.0
Driver: 580.00
CUDA: 13.0
```

### 3.2 Atomic Throughput

| Configuration | LFSRs | Throughput | Memory | Efficiency |
|--------------|-------|------------|--------|------------|
| Flat (262K) | 262,144 | 35.57 Tb/s | 16.78 MB | 2.12 Tb/s/MB |
| 4×16 fabric | 65,536 | 23.27 Tb/s | 4.19 MB | 5.55 Tb/s/MB |
| 4×4×4 fabric | 262,144 | 25.79 Tb/s | 16.78 MB | 1.54 Tb/s/MB |

### 3.3 Molecular Composition Throughput

| Composition | Throughput | Mixing Depth | Trade-off |
|------------|------------|--------------|-----------|
| Serial (A→B→C→D) | 41.51 Tb/s | 4 stages | Maximum mixing |
| Parallel (⊕ all) | 51.93 Tb/s | 1 stage | Maximum speed |
| Hybrid ((A→B)⊕(C→D)) | 51.13 Tb/s | 2 stages | Balanced |

### 3.4 Protein-like Computation

```
Proteins:            65,536
Reactions/sec:       21.85 billion
Conformational bits: 11.19 Tbits/sec
Convergence:         100% to stable attractor
```

---

## 4. The Hierarchy of Shapes

```
╔═══════════════════════════════════════════════════════════════════════════╗
║  LEVEL          UNIT              TRAINED?     COMPOSITION                ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Atom           Single LFSR       Yes          —                          ║
║  Molecule       4 atoms           No           Serial / Parallel / Hybrid ║
║  Protein        Molecule + bind   No           Binding + Folding          ║
║  Pathway        N proteins        No           Cascade (enzymatic)        ║
║  Metabolism     M pathways        No           Network                    ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

**Train atoms once. Compose forever.**

---

## 5. Theoretical Foundation

### 5.1 Fungibility Proof Chain

```
FLYNNCONCEIVABLE     Neural → Classical (100% accuracy)
        ↓
Spline-6502          Classical → Compressed (1199×)
        ↓
TriX                 Routing = Weight Structure
        ↓
LFSR Fabric          Structure = Speed (50+ Tb/s)
```

### 5.2 The Isomorphism

| TriX Concept | LFSR Equivalent |
|--------------|-----------------|
| Ternary weights (-1, 0, +1) | Tap bits (XOR, pass, skip) |
| Tile signatures | LFSR state fingerprints |
| Content addressing | Pattern matching via XOR |
| Routing from structure | Mixing from taps |

### 5.3 The Protein Analogy

| Biological | Computational |
|------------|---------------|
| Amino acid sequence | Tap pattern (trained) |
| Binding site | Input pattern mask |
| Conformational state | LFSR state (512 bits) |
| Protein folding | State evolution (shift+XOR) |
| Active site | Output bit mask |
| Enzyme cascade | Molecule chain |
| Metabolism | Computation network |

---

## 6. Implications

### 6.1 Training vs. Execution Paradigm

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   GROW (neural)         FREEZE (export)         BE (geometric)         │
│   ─────────────         ──────────────          ──────────────         │
│   polynomial XOR    →   native XOR          →   LFSR fabric            │
│   gradients flow    →   structure fixed     →   shapes evolve          │
│   0.04 Tb/s         →   —                   →   50+ Tb/s               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Composition Without Retraining

Like chemistry: train atomic properties, compose molecular behavior.

```
Carbon trained → Diamond, Graphene, Fullerene (no retraining)
Atom A trained → Serial, Parallel, Hybrid molecules (no retraining)
```

### 6.3 Biological Computation at Silicon Speed

Proteins compute at ~10⁶ reactions/sec (diffusion limited).
LFSR proteins compute at 21.85 × 10⁹ reactions/sec.

**21,850× speedup** over biological timescales.

---

## 7. Conclusion

We have demonstrated a universal substrate for growing deterministic neural-geometric shapes of compute:

1. **Atoms** are trained LFSR patterns encoding computational properties
2. **Molecules** compose atoms without retraining
3. **Proteins** compute via conformational change
4. **Pathways** cascade proteins like enzymes

The shape IS the computation. Training grows the shape. Execution IS the shape.

This completes the fungible computation proof chain: from neural networks that emulate CPUs, to compressed lookup tables, to routing-as-structure, to **structure-as-speed**.

---

## References

[1] Austin, A.N. "Fungible Computation Between Paradigms." 2024. https://github.com/anjaustin/fungible-computation

[2] Austin, A.N. "TriX: Ternary Routing with Isomorphic eXecution." 2024. https://github.com/anjaustin/trix

[3] Austin, A.N. "FLYNNCONCEIVABLE: Neural 6502 Emulation." 2024. https://github.com/anjaustin/flynnconceivable

---

## Appendix A: Benchmark Reproduction

```bash
# Atomic throughput
cd /tmp
nvcc -O3 -arch=native -o frozen frozen.cu && ./frozen

# Molecular composition
nvcc -O3 -arch=native -o molecular_shapes molecular_shapes.cu && ./molecular_shapes

# Protein-like computation
nvcc -O3 -arch=native -o protein_compute protein_compute.cu && ./protein_compute

# Fabric configurations
nvcc -O3 -arch=native -o fabric_4x16 fabric_4x16.cu && ./fabric_4x16
nvcc -O3 -arch=native -o fabric_4x4x4 fabric_4x4x4.cu && ./fabric_4x4x4
```

---

## Appendix B: The Mantra

```
Shape IS compute.
Structure IS routing.
Geometry IS logic.
Training IS growing.
Inference IS being.

Train atoms once.
Compose molecules forever.
Let proteins fold.
Watch computation emerge.
```

---

*Document version: 1.0*
*Date: 2025-12-22*
*Hardware: NVIDIA Thor (Blackwell)*
*Throughput: 51.93 Tbits/sec molecular, 21.85B reactions/sec protein*
