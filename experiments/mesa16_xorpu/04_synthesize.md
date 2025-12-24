# SYNTHESIZE: trix.forge Specification

> Phase 4 of the Lincoln Manifold Method
> Date: 2025-12-22

---

## Module: `trix.forge`

A compiler that transforms chip specifications into frozen neural-geometric models.

---

## File Structure

```
src/trix/forge/
    __init__.py      # Public API
    chip.py          # Chip class (DSL)
    compiler.py      # Compilation pipeline
    shapes.py        # Shape library (reuse from routing)
    validate.py      # Validation utilities
```

---

## Core API

### Chip Class

```python
class Chip:
    def __init__(self, name: str, bits: int = 8):
        """Create a new chip specification."""

    def input(self, name: str, bits: int = None) -> "Chip":
        """Declare an input port."""

    def output(self, name: str, bits: int = None) -> "Chip":
        """Declare an output port."""

    def operation(self, opcode: int, op_name: str) -> "Chip":
        """Map opcode to operation."""

    def compile(self, mode: str = "soft") -> nn.Module:
        """Compile to PyTorch module."""

    def validate(self, exhaustive: bool = True) -> float:
        """Validate against truth functions."""

    def compute(self, **kwargs) -> int:
        """Compute single operation."""

    def save(self, path: str) -> None:
        """Save to PyTorch format."""

    def to_onnx(self, path: str) -> None:
        """Export to ONNX."""
```

---

## Built-in Operations

Reuse from `trix.routing.shapes`:

| Name | Operation |
|------|-----------|
| `add` | a + b (mod 2^N) |
| `sub` | a - b (mod 2^N) |
| `inc` | a + 1 |
| `dec` | a - 1 |
| `xor` | a ^ b |
| `and` | a & b |
| `or` | a \| b |
| `not` | ~a |
| `shl` | a << 1 |
| `shr` | a >> 1 |
| `rol` | rotate left |
| `ror` | rotate right |

---

## Usage Examples

### Example 1: Simple 4-op ALU

```python
from trix.forge import Chip

chip = Chip("alu_4op")
chip.input("a", 8)
chip.input("b", 8)
chip.input("op", 2)

chip.operation(0, "add")
chip.operation(1, "sub")
chip.operation(2, "xor")
chip.operation(3, "and")

chip.output("result", 8)

# Compile and validate
model = chip.compile()
accuracy = chip.validate()
print(f"Accuracy: {accuracy * 100}%")  # 100%

# Use it
print(chip.compute(a=17, b=38, op=0))  # 55
```

### Example 2: Shorthand

```python
from trix.forge import Chip

# One-liner for common ALU
chip = Chip.alu(["add", "sub", "xor", "and"])
chip.validate()  # 100%
```

### Example 3: Full 6502 ALU

```python
from trix.forge import Chip

chip = Chip("6502_alu")
chip.input("a", 8)
chip.input("b", 8)
chip.input("op", 4)  # 16 possible operations

# 6502 operations
chip.operation(0x00, "add")   # ADC
chip.operation(0x01, "sub")   # SBC
chip.operation(0x02, "and")   # AND
chip.operation(0x03, "or")    # ORA
chip.operation(0x04, "xor")   # EOR
chip.operation(0x05, "shl")   # ASL
chip.operation(0x06, "shr")   # LSR
chip.operation(0x07, "rol")   # ROL
chip.operation(0x08, "ror")   # ROR
chip.operation(0x09, "inc")   # INC
chip.operation(0x0A, "dec")   # DEC
# ... etc

chip.output("result", 8)

model = chip.compile()
chip.to_onnx("6502_alu.onnx")
```

---

## Implementation

### Chip.compile()

```python
def compile(self, mode="soft"):
    # 1. Collect all operations
    ops = self._operations  # {opcode: op_name}

    # 2. Get frozen shapes
    shapes = [get_shape(name) for name in ops.values()]
    truths = [get_truth(name) for name in ops.values()]

    # 3. Build router (from trix.routing)
    router = Router(self.input_bits, len(ops))

    # 4. Create compiled module
    return CompiledChip(router, shapes, truths, mode)
```

### CompiledChip.forward()

```python
def forward(self, x):
    a_bits = x[:, :self.bits]
    b_bits = x[:, self.bits:self.bits*2]

    # Compute all shapes
    outputs = [shape(a_bits, b_bits) for shape in self.shapes]
    stacked = torch.stack(outputs, dim=1)

    # Route
    if self.mode == "soft":
        weights = torch.softmax(self.router(x), dim=1)
        result = torch.einsum('bn,bno->bo', weights, stacked)
    else:
        idx = self.router(x).argmax(dim=1)
        result = stacked[torch.arange(len(x)), idx]

    return result
```

---

## Relationship to trix.routing

`trix.forge` builds on `trix.routing`:

```
trix.routing.primitives  → Atomic polynomials (xor, and_, not_)
trix.routing.shapes      → Pre-built shapes (add_8bit, etc.)
trix.routing.pipeline    → Training infrastructure
        ↓
trix.forge.Chip          → Declarative chip specification
trix.forge.compiler      → Spec → Model compilation
```

Forge is a higher-level API for chip specification. It uses routing's shapes and training internally.

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Lines to define 4-op ALU | < 10 |
| Compilation time | < 1 second |
| Validation (8-bit, 4 ops) | 100% |
| ONNX export | Working |

---

## Future Extensions (Out of Scope)

1. **Variable bit widths** (16, 32, 64)
2. **Sequential circuits** (registers, clocks)
3. **Custom shape definitions**
4. **Optimization passes** (fusion, simplification)
5. **Status flags** (carry, zero, negative)

---

*End of Phase 4 - SYNTHESIZE*

*Now: Build it.*
