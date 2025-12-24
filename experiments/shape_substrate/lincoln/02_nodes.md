# Nodes of Interest: Shape Substrate

Extracted from RAW phase. These are the key tensions, insights, and decision points.

---

## Node 1: The Polynomial Identity

XOR(a,b) = a + b - 2ab is mathematically exact for binary inputs.
This is not an approximation. It's a theorem.

**Why it matters:** This is the foundation. Everything else depends on this being true. It IS true. Verifiably.

---

## Node 2: The Gradient Gap

Polynomial XOR has gradients. Native XOR does not.
This creates two paradigms: training (slow, differentiable) and inference (fast, discrete).

**Why it matters:** This is why the approach works. You can't train native XOR, but you can run it fast. You can train polynomial XOR, but it's slow. Use each where it fits.

**Tension with Node 1:** The identity is exact, but the performance isn't. Same math, 1000x speed difference.

---

## Node 3: The 1000x Gap

Polynomial: ~0.04 Tbits/sec
Native LFSR: ~35 Tbits/sec
Ratio: ~875x

**Why it matters:** This gap defines the value proposition. Train slow, run fast. If the gap were 2x, who cares? At 1000x, it's transformative.

**Question:** Is this gap fundamental or implementation-specific?

---

## Node 4: Composition Without Retraining

Atoms (trained tap patterns) compose into molecules (serial/parallel/hybrid) without additional training.

**Why it matters:** This is the scaling story. Train atoms once. Compose molecules forever. Like chemistry.

**Tension:** We haven't proven this for TRAINED atoms yet. Only for hand-designed tap patterns.

---

## Node 5: The Protein Analogy

Binding site → Signature matching
Conformational change → State evolution
Active site → Output extraction
Enzyme cascade → Molecule chain

**Why it matters:** This isn't metaphor. It's structural isomorphism. Same mechanism, different substrate.

**Question:** How deep does the analogy go? Are there biological insights we should import?

---

## Node 6: The Fungible Computation Chain

FLYNNCONCEIVABLE → Spline-6502 → TriX → LFSR Fabric

Each link proven independently. The chain proves neural ↔ classical fungibility.

**Why it matters:** This is the theoretical foundation. We're not making a claim without backing.

**Tension:** The chain is proven, but Shape Substrate is the newest link. Needs more validation.

---

## Node 7: The Hierarchy Emergence

Atom → Molecule → Protein → Pathway

This emerged from the work, not from planning. We discovered it while optimizing.

**Why it matters:** Emergent structure suggests we found something real, not just engineered something clever.

**Question:** Is this the natural hierarchy, or are there other valid decompositions?

---

## Node 8: The Onboarding Problem

Skeptical freshman won't read theory. They need to RUN something and see it work.

**Why it matters:** Adoption depends on accessibility. The 10-file tutorial is an attempt to solve this.

**Tension:** Each file must be < 100 lines AND teach something real. Hard constraint.

---

## Node 9: TriX Connection

TriX tiles ARE the protein mechanism:
- Tile weights = tap patterns
- Tile signatures = binding sites
- Winner-take-all = conformational selection

**Why it matters:** This connects Shape Substrate to existing TriX work. It's not a new system; it's a reframing.

**Question:** Should we refactor TriX to use the protein vocabulary? Or keep them separate?

---

## Node 10: Benchmark Validity

35 Tbits/sec is register operations, not memory bandwidth.
The state stays in registers, shifts 1000x, then writes back.

**Why it matters:** The number is real but needs context. Memory bandwidth is ~12 GB/s, not 35 Tb/s.

**Tension:** Different metrics for different purposes. Must be clear about what's being measured.

---

## Node 11: Training vs Execution Substrate

The LFSR fabric is an EXECUTION substrate, not a training substrate.
Training happens elsewhere (GPU with gradients).

**Why it matters:** Clear separation of concerns. Don't try to train on the fabric.

**Question:** Could we train routing on the fabric? Or is that a bad idea?

---

## Node 12: The Missing Training Demo

We showed composition works. We showed protein-like behavior works.
We didn't show training the tap patterns from scratch.

**Why it matters:** This is the gap between "proof of concept" and "real system."

**Tension:** Training tap patterns is a different problem (gradient descent on binary feedback). Need to explore.

---

## Tensions Summary

| Tension | Node A | Node B | Resolution Needed |
|---------|--------|--------|-------------------|
| Speed vs Trainability | 2 | 3 | Separate paradigms |
| Proven vs Demonstrated | 4 | 12 | Need training demo |
| Theory vs Practice | 6 | 10 | Benchmark clarity |
| Existing vs New | 9 | 5 | Vocabulary choice |

---

## Key Questions for Reflection

1. Is the 1000x gap a feature or a limitation?
2. How do we train tap patterns with gradients?
3. Should TriX adopt the protein vocabulary?
4. What's the minimum viable production use case?
5. How does this connect to reservoir computing?
