# REFLECT: Unified Neural-Geometric Deterministic Systems Foundry

Taking the nodes seriously. Looking for the structure beneath the content.

---

## Core Insight: The Foundry IS a Compiler

The nodes keep pointing to compilation:
- Spec → IR → Target
- Frontend → Middle → Backend
- Truth table → Terms → Hardware

This isn't a metaphor. It's the actual structure.

```
┌─────────────────────────────────────────────────────────────────┐
│                    FOUNDRY = COMPILER                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   FRONTEND                                                       │
│   ├── Truth tables (lambda a,b: a ^ b)                          │
│   ├── Composition operators (seq, par, sel, rep)                │
│   └── Bit width, target specification                           │
│                                                                  │
│   MIDDLE (IR)                                                    │
│   ├── ShapeTerms (canonical polynomial representation)          │
│   ├── Derived signatures (for routing)                          │
│   └── Composition graph (how shapes connect)                    │
│                                                                  │
│   BACKEND                                                        │
│   ├── CUDA (shape_to_cuda, batched execution)                   │
│   ├── Verilog (shape_to_verilog, synthesis)                     │
│   ├── Validation (exhaustive, statistical)                      │
│   └── Hardware estimation (LUTs, cycles, power)                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Resolved Tension: Forge vs Foundry

The answer is clear now:

- `trix.forge` has the RIGHT representations (ShapeTerms, backends, exports)
- `trix.foundry` has the RIGHT abstractions (truth table → shape, CPU patterns)

**Resolution:** The unified foundry lives in `trix.forge` and USES the FrozenFoundry pattern.

Merge direction: Foundry concepts → Forge infrastructure.

Not a third thing. An evolution of forge.

---

## Resolved Tension: Signatures from Terms

The question was: sample behavior or compute from terms?

**Resolution:** Both are valid, for different purposes.

- **From terms:** Deterministic, algebraic, fast. Use when shapes are known.
- **From behavior:** General, works for any function. Use during discovery.

For a deterministic foundry, compute from terms:

```python
def signature_from_terms(shape: ShapeTerms) -> Tensor:
    """Derive ternary signature from term structure."""
    # Each term contributes to signature based on:
    # - Which output bits it affects
    # - Which input bits it uses
    # - The coefficient pattern

    # Intuition: The signature encodes "what this shape cares about"
    # +1 = wants this input pattern
    # -1 = wants opposite pattern
    #  0 = doesn't care
```

The signature is a HASH of the shape's structure, not a sample of its behavior.

---

## Resolved Tension: When to Train

**Deterministic systems:** No training. Initialize router directly from spec.

```python
# Example: ALU with 4 operations
foundry = Foundry(bits=8)
foundry.shape("xor", lambda a,b: a ^ b)
foundry.shape("and", lambda a,b: a & b)
foundry.shape("or",  lambda a,b: a | b)
foundry.shape("add", lambda a,b: (a + b) & 0xFF)

# No training - direct routing
alu = foundry.build()
result = alu.execute(a, b, "xor")  # Direct dispatch
```

**Learning systems:** Train router only when shapes are approximate.

But for "100/100 or don't ship" systems, training is verification, not discovery.

---

## The Composition Algebra

Four operators cover all deterministic composition:

### 1. SEQ (Sequential)
Output of shape A feeds input of shape B.
```
seq(A, B)(x) = B(A(x))
```
Terms: Substitute A's output variables into B's input variables.

### 2. PAR (Parallel)
Same input, multiple outputs.
```
par(A, B)(x) = (A(x), B(x))
```
Terms: Concatenate output bit terms.

### 3. SEL (Selection)
Opcode selects which shape executes.
```
sel(op, [A, B, C])(x) = {A(x) if op=0, B(x) if op=1, C(x) if op=2}
```
This is the routing fabric.

### 4. REP (Repetition)
Chain shape N times (ripple carry pattern).
```
rep(A, N)(x) = A(A(A(...A(x)...)))  # N times
```
Terms: N sequential compositions.

**Insight:** These four operators are sufficient for any deterministic composition. The frontend language is just these operators over atomic shapes.

---

## The Minimal Interface

After reflection, the interface crystallizes:

```python
from trix.forge import Foundry

# Create foundry
foundry = Foundry(bits=8)

# Register atomic shapes (from truth tables)
foundry.atom("xor", lambda a, b: a ^ b)
foundry.atom("and", lambda a, b: a & b)
foundry.atom("full_adder", lambda a, b, c: ((a ^ b ^ c), ((a & b) | (c & (a ^ b)))))

# Compose
foundry.compose("ripple_add", rep("full_adder", 8))
foundry.compose("alu", sel("op", ["xor", "and", "or", "ripple_add"]))

# Build (generates ShapeTerms, signatures, routing fabric)
system = foundry.build()

# Validate
result = system.validate(exhaustive=True)  # 100% or fail

# Execute
output = system.execute(a=42, b=13, op="xor")

# Export
system.export_cuda("output/")
system.export_verilog("output/")
```

---

## What I Now Understand

### The Three Pillars Unified

1. **Polynomial computation** is the SUBSTRATE (what shapes are)
2. **Ternary signatures** are the ADDRESSING (how we find shapes)
3. **Routing-only learning** is the PRINCIPLE (what we learn, if anything)

They're not three features to combine. They're three aspects of ONE architecture.

### The Foundry Structure

```
Foundry
├── AtomRegistry          # Atomic shapes (truth tables → ShapeTerms)
├── CompositionGraph      # How atoms combine (seq/par/sel/rep)
├── SignatureTable        # Derived from ShapeTerms
├── RoutingFabric         # input × signatures → winner
└── ExportPipeline        # ShapeTerms → CUDA/Verilog/etc
```

### The Data Flow

```
User Spec (truth table + composition)
         ↓
    ShapeTerms (polynomial IR)
         ↓
    ┌────┴────┐
    ↓         ↓
Signatures  Routing
    ↓         ↓
    └────┬────┘
         ↓
   Export Target (CUDA/Verilog)
```

### The Key Invariant

**If validation passes, the export is correct by construction.**

- Validation proves ShapeTerms match truth table
- Export is deterministic transform of ShapeTerms
- Therefore export matches truth table

This is the "100/100 or don't ship" guarantee.

---

## Remaining Questions

1. **Term-based signature derivation:** What's the exact algorithm?
2. **Composition term rewriting:** How do terms combine algebraically?
3. **Carry handling:** Multi-bit compositions need carry propagation.
4. **Error messages:** What fails when composition is invalid?

These are implementation details, not architectural questions. The structure is clear.

---

## What Surprised Me

The foundry isn't a new thing. It's what forge was always trying to be.

The pieces exist:
- ShapeTerms ✓
- Verilog export ✓
- CUDA export ✓
- Backend execution ✓
- Chip DSL ✓

What's missing is:
- Composition algebra (seq/par/sel/rep)
- Term-based signature derivation
- The unified `Foundry` interface that ties it together

That's maybe 300 lines of code on top of existing infrastructure.

---

## The Name

"Unified Composite Neural-Geometric Deterministic Systems Foundry"

Shorter: **Forge** (it's already named this)

The `trix.forge` module IS the foundry. We just need to complete it.
