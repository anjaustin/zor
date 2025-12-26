# Reflections: Gradient Truth — Beyond the STE Lie

## The First Pattern: Separation of Concerns

Looking across the nodes, one theme dominates: **separation**.

- Node 3: WHAT vs WHICH vs HOW MUCH
- Node 6: Structure vs Scale (W = S * T)
- Node 9: Cell interiors vs Cell boundaries
- Node 7: Atoms vs Composition

The pattern is consistent: there are two fundamentally different types of "thing" being conflated in traditional neural networks:

**Type A: Structural (discrete, static)**
- What shapes exist
- Which connections are present
- The topology of computation

**Type B: Parametric (continuous, learned)**
- How much to scale
- Which structure to use when
- The navigation through structure

STE is an attempt to learn Type A things with Type B methods. That's the category error.

## The Second Pattern: Existence vs Selection

Nodes 2, 4, and 11 circle around the same insight:

**Some things exist. They are not created by training.**

XOR is not invented by gradient descent. It exists as a mathematical fact. The network discovers it, not creates it. Similarly, the computational structure of a 6502 isn't learned — it's the geometry of how addition and logic gates compose.

This means: **shapes pre-exist training. Training selects among them.**

The shape library isn't arbitrary — it's constrained by mathematical reality. We might not know all the shapes, but they're there to be discovered.

## The Third Pattern: Hierarchy of Commitment

Nodes 5, 12, and 13 suggest a temporal structure:

1. **Exploration**: soft, continuous, gradients everywhere
2. **Crystallization**: structure emerges, patterns stabilize
3. **Commitment**: freeze structure, continue learning routing/scale
4. **Deployment**: fully frozen structure, routing only

This isn't a new training procedure — it's a *description of what already happens* in well-trained networks. Weights stabilize. Patterns emerge. The network "finds" its structure.

The insight: **make this process explicit. Name the phases. Separate the learning objectives.**

## Resolving Tension: The Bootstrap Problem (Nodes 4, 5)

Where do shapes come from?

The tension: we need shapes before we can route to them, but how do we know what shapes to have?

Resolution: **The shape space is discoverable, not arbitrary.**

Three valid approaches, not mutually exclusive:

1. **Derive from first principles** (Node 2)
   - Works for: Boolean logic, arithmetic, known algorithms
   - The shapes ARE the math. Just write them down.

2. **Evolve via search** (Node 4, option 3)
   - Works for: domains where validation is cheap but derivation is hard
   - LFSR tap patterns, circuit topologies
   - No gradients needed — just enumerate/evolve/validate

3. **Distill from gradient training** (Node 5)
   - Works for: domains where we have data but no theory
   - Train with STE → observe converged structure → freeze
   - STE becomes archaeology, not engineering

The key realization: **these three methods can chain.**

Derive primitives → Evolve compositions → Distill refinements

## Resolving Tension: Universality vs Practicality (Node 11)

Can a finite set of primitives represent any computation?

For exact Boolean logic: Yes. AND, OR, NOT suffice. Or just NAND. Or just NOR.

For polynomials: Yes. {+, *, constants} generate all polynomials.

For approximate learned features: **The question may be wrong.**

Reflection: Maybe not everything needs to be discrete shapes. The claim isn't "all computation is discrete" — it's "the discrete parts can be separated from the continuous parts."

A learned feature extractor might use continuous weights. But the downstream computation on those features might be discrete routing over frozen shapes.

Resolution: **Hybrid architecture.** Continuous where needed (early features?), discrete shapes where possible (operations, transformations).

## Resolving Tension: The Lookup Table Anxiety (Node 10)

Am I just describing lookup tables?

Deep reflection: What IS a lookup table? A mapping from input to output with no generalization.

What makes learned routing different?
- Routing generalizes: similar inputs route similarly
- Routing compresses: you don't store 2^n entries
- Routing is learned: the mapping emerges from data

A lookup table over n bits is 2^n entries.
Learned routing over n dimensions is O(1) comparisons with O(n) signatures.

**Routing is compressed, generalized, learned lookup. That's exactly what we want.**

The anxiety dissolves: lookup tables are what you get when routing is perfect and exhaustive. Learned routing is an approximation that trades exactness for compression and generalization.

## The Core Insight

After sitting with all 14 nodes, the structure beneath emerges:

**Gradients should flow where there is genuine uncertainty.**
**Structure should be fixed where there is mathematical certainty.**

STE violates this by pushing gradients through structure (pretending structure is uncertain) and produces "learned" structure (pretending mathematics is trainable).

The elegant solution:

1. **Identify structural truths** (derivation, evolution, distillation)
2. **Freeze structural truths** (no gradients, no STE)
3. **Learn routing among structures** (continuous, full gradients)
4. **Learn magnitude scaling** (continuous, full gradients)

This isn't three tricks — it's one principle: **respect the ontology of what you're learning.**

## What I Now Understand

The STE isn't wrong because it's a hack. It's wrong because it conflates two different types of things:
- Things that exist (shapes, structure, mathematical truths)
- Things that are learned (which truth to apply, how much to scale)

The elegant solution is ontological hygiene:
- Discrete things are discovered (or derived, or evolved)
- Continuous things are learned
- They interface through routing

TriX already embodies this in pieces:
- Frozen shapes (Mesa 14) — discrete, derived
- Routing pipelines (Mesa 15) — continuous, learned
- Learnable scales (VGem's fix) — continuous, learned
- Signature manifold (Mesa 11) — geometric interpretation

The synthesis will unify these pieces into a coherent training paradigm.

## The Name

This approach needs a name. Looking at what it does:

- Respects the truth of mathematical shapes
- Allows gradients where genuinely needed
- Separates existence from selection

**Gradient Truth**: Gradients only where there is genuine uncertainty. Truth (structure) stands apart.

Or perhaps: **Ontological Gradient Descent** — gradients respect the ontology of what they're updating.

Or simpler: **Shape-First Learning** — discover shapes, then learn to use them.
