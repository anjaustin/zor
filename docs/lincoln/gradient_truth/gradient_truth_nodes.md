# Nodes of Interest: Gradient Truth — Beyond the STE Lie

## Node 1: The Framing Inversion
The standard framing asks: "How do we get gradients through discrete things?"
The inverted framing asks: "Why are we trying to learn discrete things with gradients?"

Why it matters: If the framing is wrong, every solution within that frame is a workaround. STE is an answer to the wrong question.

## Node 2: Mathematical Truths vs Learned Approximations
XOR(a,b) = a + b - 2ab is not learned. It's discovered. It's a theorem, not a parameter.
The 16 shapes of the 6502 are geometry, not weights.

Why it matters: Some "weights" shouldn't be weights at all. They're encoding facts, not preferences. You don't train π.

Tension with Node 1: If some things are truths, how do we identify them? Not everything is XOR.

## Node 3: The Separation — WHAT vs WHICH vs HOW MUCH
- WHAT to compute: discrete structure (shapes, frozen)
- WHICH one to use: continuous routing (learned)
- HOW MUCH to scale: continuous magnitude (learned)

Why it matters: This decomposition places all learning in naturally continuous domains. No discretization needed for gradient flow.

Tension: Is this decomposition always possible? Can any computation be factored this way?

## Node 4: The Shape Library Problem
Shapes must exist before routing can select them. Where do shapes come from?

Options identified:
1. Mathematical derivation (theorems)
2. Enumeration (brute force for small spaces)
3. Evolution (search without gradients)
4. Observation (train with STE once, freeze forever)

Why it matters: This is the bootstrap problem. The elegance of the solution depends on having a principled shape genesis.

## Node 5: STE as Shape Discovery Tool
What if STE isn't the runtime training method but a one-time shape discovery phase?

Train with STE → observe converged structures → freeze as shapes → retrain with pure routing

Why it matters: This reframes STE from "necessary evil for training" to "archaeological tool for discovering structure." Use it once, not forever.

Tension with Node 2: If we use STE to discover shapes, are those shapes still "truths" or are they just learned patterns?

## Node 6: Factorization Interpretation
W = S * T where S is continuous scale, T is ternary structure.

Gradients flow through S. T is found separately (evolution, derivation, observation).

Why it matters: This is mathematically clean. Matrix factorization is well-understood. Opens up all the tools from that literature.

Similar to: BatchNorm (scale/shift modulates fixed features), LoRA (low-rank adaptation of frozen weights)

## Node 7: The Hierarchy — Atoms, Molecules, Proteins
TriX already has this structure:
- Atoms: single LFSRs (tap patterns evolved)
- Molecules: compositions of atoms (serial/parallel)
- Proteins: molecules + binding sites (conformational change)

Why it matters: Shapes compose. You don't need infinite shapes — you need composable primitives.

Connection to Node 4: The shape library might be small if composition is powerful.

## Node 8: Manifold Geometry Connection
Mesa 11 views signatures as points on a manifold. Training warps the manifold.

If shapes are fixed points, routing learns trajectories between them. The discrete structure defines the space; learning navigates it.

Why it matters: Provides geometric intuition. Learning is movement, not construction.

Tension: Is this more than metaphor? Is there a concrete manifold we can work with?

## Node 9: Voronoi Cells and Decision Boundaries
Each shape "owns" a region of input space. Routing is deciding which cell the input falls in.

The cells have continuous boundaries that can be learned. The cell contents (shapes) are discrete and fixed.

Why it matters: Clean separation — learn boundaries (continuous), freeze interiors (discrete).

## Node 10: The Lookup Table Anxiety
Am I just describing fancy lookup tables?

Counter: Lookup tables are O(2^n) in input size. Learned routing is O(n) or O(log n). The routing IS the compression. The routing IS the generalization.

Why it matters: Need to distinguish this from "just enumerate all cases."

## Node 11: Universality Question
Is there a finite set of primitive shapes that can represent any computation?

Boolean functions: AND, OR, NOT suffice.
Polynomials: Addition, multiplication suffice.
The polynomial representation of shapes: Already universal for Boolean logic.

Why it matters: If primitives are universal, the shape library is finite and small.

Tension: Boolean/polynomial universality works for exact computation. What about approximate learned features?

## Node 12: The Sparsity Connection
Ternary weights are sparse (many zeros). Sparsity is a form of discreteness.

But sparsity patterns can be learned (lottery ticket hypothesis, pruning).

What if: Learn continuous → identify sparse structure → freeze structure → learn routing over frozen structure

Why it matters: Connects to established ML literature on sparsity and pruning.

## Node 13: Progressive Commitment
You don't have to commit to shapes all at once.

Early training: soft, continuous, exploring
Late training: shapes crystallize, commit
Deployment: frozen, routed

Why it matters: This mirrors the progressive quantization schedule in QAT. Maybe the schedule leads to discrete structure, then we freeze what emerged.

## Node 14: The Elegance Objection
"STE works. Why fix what isn't broken?"

Counter: STE works for training. But it means your gradients are wrong. You're optimizing a different function than you're running. This MUST have consequences, even if they're subtle.

Why it matters: Need to articulate the actual cost of STE beyond aesthetics.
