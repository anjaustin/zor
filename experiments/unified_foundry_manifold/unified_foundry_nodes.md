# NODES: Unified Neural-Geometric Deterministic Systems Foundry

Extracted from RAW phase. These are observation points, not solutions.

---

## Node 1: The Three Pillars Are One Thing

Ternary routing, polynomial computation, and routing-only learning aren't three separate ideas - they're three perspectives on the same underlying structure.

- Ternary routing: HOW we address
- Polynomial computation: WHAT we address
- Routing-only learning: WHAT we learn (just addressing)

**Why it matters:** Unification isn't gluing pieces together. It's recognizing they were always connected.

---

## Node 2: ShapeTerms as Ground Truth

The `ShapeTerms` representation (from term.py) is the canonical form. Everything else should derive from it:

- Signatures derive from terms
- Verilog derives from terms
- CUDA derives from terms
- Hardware estimates derive from terms

**Why it matters:** Single source of truth prevents divergence.

**Tension:** Current signatures are derived from behavior sampling, not term structure.

---

## Node 3: Signature Derivation Gap

Current approach: Run shape on random inputs, project outputs to ternary signature.

Potential approach: Compute signature directly from polynomial terms.

Questions:
- What's the relationship between term structure and signature?
- Can we define signature algebraically from terms?
- Does the sampled signature actually capture term structure?

**Why it matters:** If signatures can be computed from terms, no sampling needed.

---

## Node 4: Composition Algebra

How do shapes combine into larger shapes?

Examples:
- Full adder = XOR(XOR(a,b), c) for sum, OR(AND(a,b), AND(c, XOR(a,b))) for carry
- Ripple adder = 8 chained full adders
- ALU = MUX over {XOR, AND, OR, ADD, ...}

What's the algebra?
- Sequential composition: output of one → input of another
- Parallel composition: same inputs → multiple outputs
- Selection composition: opcode selects which shape

**Why it matters:** Composition is how 5×5 units become systems.

---

## Node 5: The Existing Pieces

What we have:
- `trix.forge.term` - ShapeTerms, generators, validation
- `trix.forge.verilog` - Verilog export
- `trix.forge.cuda` - CUDA export
- `trix.forge.hardware` - Resource estimation
- `trix.forge.backend` - Execution layer (CPU/CUDA)
- `trix.forge.chip` - Chip DSL with execute()
- `trix.routing` - RoutingPipeline, frozen shapes
- `trix.foundry` - FrozenFoundry (separate implementation)

**Tension:** Two foundries exist (forge and foundry). Which is canonical?

**Why it matters:** Unification means ONE foundry, not two.

---

## Node 6: The Minimal Interface

What does a user actually need?

Input:
- Truth table OR lambda function
- Bit width
- Target (CUDA, Verilog, validation)

Output:
- Working implementation
- Validation proof
- Export artifacts

**Observation:** The current Chip DSL is close but uses routing module shapes, not forge shapes.

---

## Node 7: Routing Fabric Structure

Current routing: `scores = input @ signatures.T; winner = argmax(scores)`

This is O(n) in number of shapes. For small n (≤64), this is fine.

For larger systems, hierarchical routing exists (HierarchicalTriXFFN).

**Question:** Does the foundry need hierarchical routing, or is flat sufficient for deterministic systems?

**Observation:** Deterministic systems are typically small (ALU = 16 ops). Flat is probably fine.

---

## Node 8: Training vs No Training

Two modes:
1. **No training needed:** Shapes match operations exactly. Router is a lookup table.
2. **Training needed:** Shapes approximate operations. Router learns best match.

For deterministic systems, mode 1 is the goal. 100/100 or don't ship.

**Implication:** The foundry should support direct initialization (no training) when shapes are exact.

---

## Node 9: Export Pipeline

Current exports:
- CUDA: `shape_to_cuda()`, `export_cuda()`
- Verilog: `shape_to_verilog()`, `export_verilog()`
- JSON: `XORPU.export_terms()`

Missing:
- Unified export interface
- Composition export (export a composed system, not just individual shapes)
- Testbench generation that covers the full composition

---

## Node 10: What "Foundry" Means

Metallurgical foundry: raw metal → molten → cast into shape → finished product

Neural-Geometric foundry:
- Raw: Truth tables / lambda functions
- Molten: Polynomial term representation
- Cast: Shape with signature
- Finished: Exported to target (CUDA/Verilog/silicon)

**Insight:** The foundry is a compiler. Spec → IR (terms) → target.

---

## Node 11: The Signature Question

What IS a signature?

Current: A ternary vector derived from shape behavior on sample inputs.

Alternative: A hash of the term structure.

Another alternative: The term coefficients themselves, compressed.

**Tension:** Signatures are used for routing (content-addressable dispatch). What properties must they have?
- Different shapes → different signatures
- Similar shapes → similar signatures (for soft routing)
- Computable without executing the shape

---

## Node 12: "5 by 5" Philosophy

Small, composable, verifiable units.

Implications:
- Each shape is small enough to verify exhaustively (8-bit = 65K cases)
- Composition is explicit, not emergent
- The system is understandable by a human

**Contrast:** LLMs are billion-parameter black boxes. This is the opposite.

---

## Node 13: Gradients Through Exact Computation

The polynomial representation (XOR = a + b - 2ab) is differentiable.

But for deterministic systems, do we need gradients?

Use cases:
- Training a router to select shapes (yes, gradients help)
- Executing a fixed pipeline (no gradients needed)

**Observation:** Gradients are for learning the routing, not the computation.

---

## Node 14: The Forge vs Foundry Consolidation

`trix.forge` has:
- ShapeTerms (explicit polynomial representation)
- Backend (execution layer)
- Export (Verilog, CUDA)
- Chip DSL

`trix.foundry` has:
- FrozenFoundry (truth table → shape)
- CPU implementations (6502, etc.)
- Training pipelines

**Decision point:** Merge into forge? Keep separate? Create a third unified layer?

---

## Node 15: The Composition Compiler

If foundry is a compiler:
- Frontend: Truth tables, specs, composition operators
- Middle: ShapeTerms IR
- Backend: CUDA, Verilog, validation

Composition operators in the frontend:
- `seq(a, b)` - sequential
- `par(a, b)` - parallel
- `sel(op, [shapes])` - selection by opcode
- `rep(shape, n)` - repetition (like ripple carry)

**Insight:** The composition algebra IS the frontend language.
