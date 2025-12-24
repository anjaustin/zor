# REFLECT: Resolving Tensions

> Phase 3 of the Lincoln Manifold Method
> Date: 2025-12-22

---

## Tension 1: DSL Complexity vs Power

**Problem:** Simple DSL = limited chips. Complex DSL = hard to use.

**Resolution:** Layered API.

```python
# Level 1: One-liner for common patterns
chip = Chip.alu(ops=["add", "sub", "xor", "and"], bits=8)

# Level 2: Declarative for custom chips
chip = Chip("custom")
chip.input("a", 8).input("b", 8).input("op", 2)
chip.when(0).apply("add")
chip.when(1).apply("sub")
chip.output("result", 8)

# Level 3: Full IR for advanced users
ir = IRGraph()
ir.add_node(...)
```

Start with Level 1 and 2. Level 3 is escape hatch.

---

## Tension 2: Soft vs Hard Selection

**Problem:** Neural nets want soft selection (gradients). Hardware wants hard selection (deterministic).

**Resolution:** Mode switch.

```python
chip.compile(mode="soft")   # Differentiable, for training routers
chip.compile(mode="hard")   # Deterministic, for inference/export
```

Soft mode: weighted sum of all shape outputs
Hard mode: argmax selection, single shape executes

Default to soft (more flexible). Export to ONNX uses hard.

---

## Tension 3: Variable Bit Widths

**Problem:** Different chips have different bit widths. Shapes are width-specific.

**Resolution:** Parameterized shapes.

```python
# Shape factory
def make_add(bits):
    def add_nbits(a, b):
        # N-bit ripple adder
        ...
    return add_nbits

# Chip requests width, factory provides
chip = Chip("wide_alu", bits=16)  # Gets 16-bit shapes
```

Library provides common widths (8, 16, 32). Factory builds others on demand.

---

## Tension 4: Composition Overhead

**Problem:** Composing many shapes = many intermediate tensors.

**Resolution:** Lazy evaluation + fusion.

```python
# IR captures intent, doesn't execute
ir.add("sum", add(a, b))
ir.add("masked", and(sum, mask))

# Compiler fuses where possible
# Single pass: masked = and(add(a, b), mask)
```

For V1: accept overhead. Optimize in V2.

---

## Tension 5: State (Registers)

**Problem:** Real chips have registers. Frozen shapes are stateless.

**Resolution:** Out of scope for V1.

Combinational logic only. Sequential circuits require:
- Clock concept
- Register primitives
- Feedback handling

This is a separate project. Mark it as future work.

---

## Tension 6: Validation Scalability

**Problem:** Exhaustive validation doesn't scale (2^32 for 32-bit).

**Resolution:** Tiered validation.

```python
# Auto-selects based on input space size
chip.validate()

# < 1M cases: exhaustive
# >= 1M cases: statistical (random sample + edge cases)
```

Report confidence interval for statistical validation.

---

## The Minimal Viable Forge

For the first implementation:

**In scope:**
- 8-bit operations (exhaustively validatable)
- Combinational logic only (no state)
- Built-in shapes (add, sub, xor, and, or, not, shifts)
- Simple DSL (ops + routing)
- Soft/hard mode
- PyTorch export
- ONNX export

**Out of scope:**
- Variable bit widths (use 8-bit)
- Sequential circuits (no registers)
- Custom shape definition (use built-ins)
- Advanced optimization (accept overhead)

---

## The Cleanest Design

```python
from trix.forge import Chip

# Define chip
chip = Chip("alu_4op")
chip.input("a", 8)
chip.input("b", 8)
chip.input("op", 2)

chip.operation(0, "add")
chip.operation(1, "sub")
chip.operation(2, "xor")
chip.operation(3, "and")

chip.output("result", 8)

# Compile
model = chip.compile()

# Validate
chip.validate()  # 100% on 262,144 cases

# Export
chip.save("alu.pt")
chip.to_onnx("alu.onnx")

# Use
result = chip.compute(a=17, b=38, op=0)  # 55
```

This is remarkably similar to RoutingPipeline. The main addition is the declarative chip structure.

---

*End of Phase 3 - REFLECT*
