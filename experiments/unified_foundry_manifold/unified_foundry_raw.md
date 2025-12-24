# RAW: Unified Neural-Geometric Deterministic Systems Foundry

## Stream of Consciousness

Three pillars that want to be one thing:

1. **Ternary routing** - Weights are addresses. {-1, 0, +1} means "want this", "don't care", "want opposite". The signature IS the tile's identity. No separate routing network needed - the weights themselves encode where inputs should go.

2. **Polynomial computation** - XOR = a + b - 2ab. This isn't approximation, it's exact. On binary inputs {0,1}, these polynomials compute logic perfectly. Gradients flow through because it's continuous math, but the result is discrete truth.

3. **Routing-only learning** - The shapes are frozen. They're discovered geometry, not learned approximations. Learning only figures out WHICH shape to use, not HOW to compute. This is why 78× fewer parameters works.

So what's the unified thing?

A foundry where you:
- Define what you want to compute (truth tables, specs)
- The system discovers the polynomial shapes automatically
- Ternary signatures emerge from shape behavior
- Routing learns to dispatch inputs to shapes
- Export to CUDA/Verilog/silicon

Wait. We already have pieces of this:
- `trix.forge` - shape generation, term representation
- `trix.routing` - RoutingPipeline, frozen shapes
- `trix.foundry` - FrozenFoundry, CPU implementations
- Backend layer - CUDA/CPU execution

But they're separate. What would UNIFIED look like?

The user said "5 by 5" - small composable units. Not monolithic.

What if the foundry is:
- A registry of shapes (the atoms)
- A composition algebra (how atoms combine)
- A signature derivation system (shapes → ternary addresses)
- A routing fabric (inputs → shapes via signatures)
- An export pipeline (fabric → hardware)

The key insight from Mesa 15: you don't learn the computation, you learn the routing. The computation is geometry - discovered, not trained.

So the foundry should:
1. Let you define truth tables
2. Automatically generate polynomial shapes
3. Derive ternary signatures from shape behavior
4. Build routing fabric
5. Export to target (CUDA, Verilog, ASIC)

What's the interface?

```python
foundry = Foundry()
foundry.define("xor", lambda a, b: a ^ b)
foundry.define("add", lambda a, b: (a + b) & 0xFF)
foundry.compose("alu", ["xor", "and", "or", "add"])
foundry.build()  # Discovers shapes, derives signatures, builds fabric
foundry.export("cuda", "output/")
foundry.export("verilog", "output/")
```

But wait - we already have this in pieces. What's actually NEW?

The UNIFICATION is new. Right now:
- Chip DSL is separate from forge
- Backend is separate from routing
- Foundry is separate from XORPU
- Signatures are derived differently in different places

A unified foundry would be:
- ONE place to define computations
- ONE representation for shapes (ShapeTerms)
- ONE signature derivation method
- ONE routing mechanism
- MULTIPLE export targets

The THREE PILLARS merge:
- Ternary routing → how we ADDRESS shapes
- Polynomial computation → what shapes ARE
- Routing-only learning → the ONLY thing that trains

## Questions Arising

- How do signatures relate to ShapeTerms? Can we derive one from the other?
- What's the composition algebra? How do shapes combine into larger shapes?
- Is there a canonical form for signatures? (Right now they're derived from behavior sampling)
- How does the routing fabric scale? O(n) shapes means O(n) signature comparisons?
- What's the minimal interface? What does a user actually need to specify?

## First Instincts

- The ShapeTerms representation is the ground truth. Everything else derives from it.
- Signatures should be COMPUTED from terms, not sampled from behavior.
- Composition is just term concatenation with index remapping.
- The routing fabric is a single matrix multiply: input @ signatures.T → winner.
- Export is already mostly done (cuda.py, verilog.py).

## What Scares Me

- Feature creep. We have working pieces. Unification could break them.
- Over-abstraction. The current code is concrete and testable.
- Losing the simplicity. XOR = a + b - 2ab is beautiful. Don't bury it.

## What Excites Me

- A single `Foundry` class that does everything
- Truth table → silicon in one pipeline
- The three pillars actually being three views of ONE thing
- This could be genuinely useful for deterministic systems

## Half-Formed Ideas

- Maybe signatures ARE just compressed ShapeTerms?
- Maybe routing IS just polynomial evaluation on a meta-level?
- Maybe the foundry is a compiler: spec → intermediate repr → target?
- What if shapes are first-class objects that know their own terms, signature, and export formats?
