# Abstract

## Shape Substrate: Universal Substrate for Growing Deterministic Neural-Geometric Shapes of Compute

**A. N. Austin**

---

We present a universal computational substrate where **shape IS compute**. Building on the fungible computation thesis—that neural and classical computation are interchangeable representations of the same underlying primitives—we demonstrate that Linear Feedback Shift Register (LFSR) fabrics serve as high-speed execution substrates for trained neural-geometric shapes.

The substrate operates at three compositional levels:

**Atomic Shapes.** Pre-trained 512-bit LFSR patterns encode specific computational properties (mixing, diffusion, period length, decorrelation). Each atom's tap pattern—analogous to a protein's amino acid sequence—determines its evolution dynamics. Training occurs in the neural paradigm where gradients flow; execution occurs in the geometric paradigm where shapes evolve.

**Molecular Composition.** Atoms compose into molecules via serial injection (output→input chaining) or parallel XOR (output combination) without retraining. The molecular shape inherits properties from constituent atoms. We demonstrate three compositions: Serial (41.51 Tbits/sec, 4-stage mixing), Parallel (51.93 Tbits/sec, maximum throughput), and Hybrid (51.13 Tbits/sec, balanced).

**Protein-like Computation.** Molecules compute via conformational change: input binds to a trained pattern (binding site), binding strength triggers state evolution (folding), the new conformation determines output (active site). We observe convergence to stable attractors—analogous to protein folding finding energy minima. Throughput: 21.85 billion reactions/sec, 11.19 Tbits/sec conformational bandwidth.

This work completes the fungible computation proof chain:

| System | Contribution |
|--------|--------------|
| FLYNNCONCEIVABLE | Neural → Classical (100% 6502 emulation) |
| Spline-6502 | Classical → Compressed (1199× reduction) |
| TriX | Routing = Weight Structure |
| **Shape Substrate** | **Structure = Speed (50+ Tbits/sec)** |

The paradigm enables: train in the neural domain (polynomial operations, gradients flow), execute in the geometric domain (native operations, shapes evolve). The computation is the shape. The shape is the computation.

**Key Results:**
- Molecular throughput: 51.93 Tbits/sec
- Protein reactions: 21.85 billion/sec
- Memory efficiency: 5.55 Tbits/sec per MB (4×16 fabric)
- Biological speedup: 21,850× over diffusion-limited proteins

**The Mantra:** Train atoms once. Compose molecules forever. Let proteins fold. Watch computation emerge.

---

*Hardware: NVIDIA Thor (Blackwell), 20 SMs, Compute Capability 11.0*
*Date: 2025-12-22*
