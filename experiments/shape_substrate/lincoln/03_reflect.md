# Reflections: Shape Substrate

Deep thinking on the nodes. Looking for the structure beneath the content.

---

## Core Insight

**The 1000x gap is not a bug. It's the entire point.**

If polynomial XOR ran at the same speed as native XOR, there would be no reason to have two forms. The gap creates the value:

- TRAIN in the slow form (gradients work)
- EXECUTE in the fast form (speed works)

This is not a limitation. This is the design.

---

## Resolved Tension: Speed vs Trainability

The tension between Node 2 (gradients) and Node 3 (1000x gap) dissolves when you realize they're meant for different phases:

```
TRAINING PHASE:
  - Use polynomial shapes
  - Gradients flow
  - Optimize tap patterns, routing signatures, binding sites
  - Speed doesn't matter (you do this once)

EXECUTION PHASE:
  - Export to native shapes
  - Run on LFSR fabric
  - 35 Tbits/sec throughput
  - No gradients needed (the shape is frozen)
```

The gap is a FEATURE. It forces you to separate concerns.

---

## Resolved Tension: Proven vs Demonstrated

Node 4 (composition without retraining) and Node 12 (missing training demo) seem to conflict. But they're actually independent:

1. **Composition is proven.** Given trained atoms, they compose. The math guarantees it.

2. **Training is separate.** How you GET the trained atoms is orthogonal to how you COMPOSE them.

The demo we need isn't "train and compose." It's:
- Train atom A for mixing
- Train atom B for diffusion
- Compose A→B and verify the molecule has both properties

This is doable. The question is: what's the training objective for a single atom?

---

## Pattern Recognition: The Two Types of Learning

Looking at the nodes, there are two distinct things being "learned":

1. **Structure learning** (tap patterns, binding sites)
   - What computation each shape performs
   - What inputs each shape should handle
   - This happens during training

2. **State evolution** (conformational change)
   - How the shape processes its input
   - The actual computation
   - This happens during execution

TriX already separates these:
- Tile weights = structure (learned)
- Tile computation = fixed (polynomial shapes)

The insight: we're not learning the computation. We're learning which computation to apply.

---

## The Biological Insight We Should Import

Real proteins don't just have one binding site. They have:
- **Allosteric sites** - secondary binding that modulates function
- **Feedback loops** - output affects future binding
- **Cooperative binding** - multiple substrates must bind together

Our current model has one binding site per shape. Extending to multiple sites could enable:
- Gating (only compute if A AND B are present)
- Modulation (B changes how A is processed)
- Memory (previous state affects current binding)

This is where reservoir computing connects. The LFSR state IS the memory.

---

## What I Now Understand

### 1. The System Has Three Layers

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: ROUTING                                           │
│  What: Which shape handles this input?                      │
│  How: Signature matching (dot product)                      │
│  Learned: Yes (signatures/binding sites)                    │
├─────────────────────────────────────────────────────────────┤
│  LAYER 2: COMPUTATION                                       │
│  What: Transform input to output                            │
│  How: Shape evaluation (polynomial or native)               │
│  Learned: Maybe (tap patterns can be trained)               │
├─────────────────────────────────────────────────────────────┤
│  LAYER 1: SUBSTRATE                                         │
│  What: Execute the shapes at speed                          │
│  How: LFSR fabric, CUDA, FPGA                              │
│  Learned: No (fixed hardware)                               │
└─────────────────────────────────────────────────────────────┘
```

### 2. Training Happens at Layer 2 & 3

- Layer 3 (routing): Train signatures to match inputs to shapes
- Layer 2 (computation): Optionally train tap patterns for specific properties
- Layer 1 (substrate): Fixed. No training.

### 3. The Hierarchy Maps to This

| Level | Layer 3 | Layer 2 | Layer 1 |
|-------|---------|---------|---------|
| Atom | One signature | One tap pattern | One LFSR |
| Molecule | Composition rule | Combined taps | Chained LFSRs |
| Protein | Binding affinity | Folding dynamics | State evolution |
| Pathway | Cascade routing | Multi-stage | Full fabric |

### 4. The Freshman Path Is Right

The 10-file tutorial maps the conceptual journey:
1. The polynomial identity exists (01)
2. Gradients matter (02)
3. It builds real things (03)
4. Speed gap is real (04)
5. Composition works (05, 06, 07)
6. Routing is pattern matching (08)
7. You can train routing (09)
8. It's the same as biology (10)

This IS the learning progression. Keep it.

---

## Remaining Questions

1. **How to train tap patterns?**
   - Can use genetic algorithms (discrete optimization)
   - Can use straight-through estimator (gradient approximation)
   - Can use evolutionary strategies

2. **What's the killer app?**
   - Reservoir computing is the obvious one
   - Cryptographic hash functions (post-export)
   - Random number generation
   - Maybe: binary neural network inference

3. **Should TriX adopt protein vocabulary?**
   - Probably not. Keep them as parallel framings.
   - "Tiles" for the ML audience, "Proteins" for the bio-curious
   - Same math, different metaphors

4. **How does this scale?**
   - Current: 262K LFSRs on one GPU
   - Next: Multi-GPU fabrics
   - Future: Custom silicon (ASICs for LFSRs)

---

## The Clean Cut Emerges

The Shape Substrate is:

1. **A substrate** (Layer 1): LFSR fabric that executes shapes at silicon speed
2. **A shape library** (Layer 2): Polynomial primitives that compose
3. **A routing mechanism** (Layer 3): Signature matching that learns

The proof-of-concept demonstrated all three. The next step is integration:
- Connect to TriX training pipeline
- Show end-to-end: train → export → execute
- Benchmark on a real task

---

## What Would Make This Undeniable

1. **End-to-end demo**: Train a TriX model, export to LFSR fabric, run inference, show speedup
2. **Real task benchmark**: Something people care about (e.g., embedding lookup, hash computation)
3. **Comparison**: Same model on GPU vs LFSR fabric, show 100x+ speedup
4. **Reproducibility**: Docker container, one command, see results
