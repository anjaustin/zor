# RAW: Chip → Neural-Geometric Compiler

> Phase 1 of the Lincoln Manifold Method
> Date: 2025-12-22

---

## The Vision

```
"Build me a 6502 ALU"
        ↓
   [Compiler]
        ↓
Frozen Neural-Geometric Shape
```

Input: High-level chip specification
Output: Deployable frozen model (PyTorch, ONNX)

---

## Stream of Consciousness

What does "build this chipset" mean?

Options:
1. **Truth table** - exhaustive input→output mapping
2. **Gate netlist** - AND/OR/XOR gates wired together
3. **RTL-like** - register-transfer level description
4. **Functional** - compose operations declaratively

Truth table is complete but doesn't scale (2^32 entries for 32-bit).
Gate netlist is flexible but verbose.
RTL requires registers/state - more complex.
Functional feels right for frozen shapes.

---

## The Functional Approach

```python
# Define a simple ALU
alu = Chip("simple_alu")

# Declare ports
alu.input("a", bits=8)
alu.input("b", bits=8)
alu.input("op", bits=2)
alu.output("result", bits=8)

# Define operations
alu.op(0, "add", lambda a, b: add_8bit(a, b))
alu.op(1, "sub", lambda a, b: sub_8bit(a, b))
alu.op(2, "xor", lambda a, b: xor_8bit(a, b))
alu.op(3, "and", lambda a, b: and_8bit(a, b))

# Compile to frozen shape
model = alu.compile()
model.export("simple_alu.onnx")
```

This is basically what RoutingPipeline already does!

---

## What's Missing?

RoutingPipeline handles: input → router → shape → output

But real chips have:
1. **Internal wiring** - outputs of one op feed inputs of another
2. **Bit manipulation** - extract bits, concatenate, reorder
3. **Conditionals** - if carry, then...
4. **State** - registers that persist across cycles

---

## Layered Complexity

**Level 1: Combinational (stateless)**
- Pure function: inputs → outputs
- No memory, no feedback
- This is what we have now

**Level 2: Multi-stage combinational**
- Pipeline of operations
- Output of stage N → input of stage N+1
- Still stateless

**Level 3: Sequential (stateful)**
- Registers, flip-flops
- Feedback loops
- Requires clock/cycle concept

Let's nail Level 1 and 2 first. Level 3 is a bigger project.

---

## The Composition Problem

How do you wire shapes together?

```
a[8] ─┬─→ [ADD] ─→ sum[8] ─→ [XOR] ─→ result[8]
      │              ↑
b[8] ─┴──────────────┘
```

Need:
1. Named wires (or anonymous with fan-out)
2. Bit slicing: `a[0:4]` (low nibble)
3. Concatenation: `{carry, sum}` (9 bits from 1 + 8)
4. Constants: `0x00`, `0xFF`

---

## Proposed IR (Intermediate Representation)

```python
ir = [
    ("input", "a", 8),
    ("input", "b", 8),
    ("shape", "sum", "add_8bit", ["a", "b"]),
    ("shape", "diff", "sub_8bit", ["a", "b"]),
    ("const", "zero", 8, 0x00),
    ("mux", "result", "op", ["sum", "diff", "xor_ab", "and_ab"]),
    ("output", "result", 8),
]
```

Each node is: (type, name, ...args)

Types:
- `input`: declare input port
- `output`: declare output port
- `const`: constant value
- `shape`: apply frozen shape
- `slice`: extract bits
- `concat`: join bits
- `mux`: select based on control

---

## Compilation Stages

```
Specification (DSL)
       ↓
   [Parse]
       ↓
IR (DAG of nodes)
       ↓
   [Resolve Shapes]
       ↓
IR with frozen shapes attached
       ↓
   [Compose]
       ↓
Single composite forward()
       ↓
   [Optimize]
       ↓
Simplified composite
       ↓
   [Export]
       ↓
PyTorch / ONNX
```

---

## What Can We Reuse?

From `trix.routing`:
- Primitives (xor, and_, or_, not_)
- Shapes (add_8bit, sub_8bit, etc.)
- Router concept (for mux)

From `trix.compiler`:
- Already exists! Need to check what's there.

---

## First Test Case

Build a 4-function ALU:
- ADD, SUB, XOR, AND
- 8-bit operands
- 2-bit opcode

This is exactly the Calculator Test. We've already done it!

The question is: can we express it in a cleaner DSL and compile it automatically?

---

## Stretch Goal

Build a 6502 ALU from specification:
- All 8 arithmetic/logic operations
- Status flags (N, Z, C, V)
- BCD mode

If the compiler can do this, it can do anything combinational.

---

*End of Phase 1 - RAW*
