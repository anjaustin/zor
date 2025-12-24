# NODES: Pipeline Stages

> Phase 2 of the Lincoln Manifold Method
> Date: 2025-12-22

---

## NODE 1: The Three Atoms

Everything builds from:
```python
AND(a, b) = a * b
XOR(a, b) = a + b - 2*a*b
NOT(a)    = 1 - a
```

These are the quarks. Everything else is hadrons.

---

## NODE 2: Derived Shapes (Hadrons)

| Shape | Composition | Bits |
|-------|-------------|------|
| OR | `a + b - ab` | 1 |
| NAND | `1 - ab` | 1 |
| NOR | `1 - (a + b - ab)` | 1 |
| XNOR | `1 - (a + b - 2ab)` | 1 |
| MUX | `a + sel*(b - a)` | 1 |
| Full Adder | XOR + AND + OR | 1→2 |
| ADD | Chain of full adders | N→N |
| SUB | ADD with NOT + 1 | N→N |
| Shift | Bit rewiring | N→N |

---

## NODE 3: The DSL

```python
from trix.forge import Chip

# Declare chip
chip = Chip("alu_4op", bits=8)

# Inputs
chip.input("a", 8)
chip.input("b", 8)
chip.input("op", 2)

# Operations (indexed by opcode)
chip.when(0).do("add", a, b)
chip.when(1).do("sub", a, b)
chip.when(2).do("xor", a, b)
chip.when(3).do("and", a, b)

# Output
chip.output("result", 8)

# Compile
model = chip.compile()
```

---

## NODE 4: IR Nodes

```python
@dataclass
class IRNode:
    type: str      # "input", "output", "shape", "mux", "const", "slice", "concat"
    name: str      # unique identifier
    args: list     # type-specific arguments

# Examples
IRNode("input", "a", [8])                    # 8-bit input
IRNode("shape", "sum", ["add", "a", "b"])    # apply add shape
IRNode("mux", "result", ["op", "sum", "diff", "xor_ab", "and_ab"])
IRNode("output", "result", [8])
```

---

## NODE 5: Compilation Pipeline

```
┌─────────────┐
│ Chip DSL    │  Human-friendly specification
└─────┬───────┘
      ↓
┌─────────────┐
│ Parse       │  Extract structure
└─────┬───────┘
      ↓
┌─────────────┐
│ IR Graph    │  DAG of IRNodes
└─────┬───────┘
      ↓
┌─────────────┐
│ Resolve     │  Attach frozen shapes
└─────┬───────┘
      ↓
┌─────────────┐
│ Compose     │  Build composite forward()
└─────┬───────┘
      ↓
┌─────────────┐
│ Optimize    │  Simplify, fuse
└─────┬───────┘
      ↓
┌─────────────┐
│ Export      │  PyTorch, ONNX
└─────────────┘
```

---

## NODE 6: Shape Library

Pre-built shapes available to compiler:

**Atoms:**
- `and_1bit`, `xor_1bit`, `not_1bit`

**Arithmetic:**
- `add_Nbit`, `sub_Nbit`, `inc_Nbit`, `dec_Nbit`

**Logic:**
- `and_Nbit`, `or_Nbit`, `xor_Nbit`, `not_Nbit`

**Shifts:**
- `shl_Nbit`, `shr_Nbit`, `rol_Nbit`, `ror_Nbit`

**Control:**
- `mux_Nbit` (2-way), `mux4_Nbit` (4-way), `mux8_Nbit` (8-way)

---

## NODE 7: Bit Manipulation

```python
# Slice
chip.slice("low_nibble", "a", 0, 4)   # a[0:4]
chip.slice("high_nibble", "a", 4, 8)  # a[4:8]

# Concat
chip.concat("word", ["high", "low"])  # {high, low}

# Constant
chip.const("zero", 8, 0x00)
chip.const("one", 8, 0x01)
```

---

## NODE 8: The Forward Function

Compiler generates:

```python
class CompiledChip(nn.Module):
    def forward(self, a_bits, b_bits, op_bits):
        # Compute all shapes
        sum_out = frozen_add(a_bits, b_bits)
        diff_out = frozen_sub(a_bits, b_bits)
        xor_out = frozen_xor(a_bits, b_bits)
        and_out = frozen_and(a_bits, b_bits)

        # Stack for mux
        all_outputs = torch.stack([sum_out, diff_out, xor_out, and_out], dim=1)

        # Decode opcode
        op_idx = bits_to_int(op_bits)

        # Select (or soft-select for differentiability)
        result = all_outputs[:, op_idx]

        return result
```

---

## NODE 9: Validation Strategy

For each compiled chip:
1. Generate exhaustive test cases (if feasible)
2. Compare against reference implementation
3. Report accuracy (should be 100%)

```python
chip.validate(reference_fn, exhaustive=True)
# Output: 262144/262144 correct (100.0000%)
```

---

## NODE 10: Module Name

**`trix.forge`** - The chip forge

"Forge" because:
- We're forging chips from atomic shapes
- It's active, creative, constructive
- Complements "foundry" (shapes) and "routing" (learning)

---

*End of Phase 2 - NODES*
