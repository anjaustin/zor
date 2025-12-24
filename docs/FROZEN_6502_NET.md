# Frozen6502Net: The Neural Network That IS a 6502

> *"The geometry computes. No approximation. No learning."*

A complete 6502 ALU implemented as a PyTorch `nn.Module` with **0 learnable parameters**. The computation flows through 16 frozen mathematical shapes - pure geometry that achieves 100% accuracy by construction.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [The 16 Shapes](#the-16-shapes)
4. [Usage](#usage)
5. [ONNX Export](#onnx-export)
6. [Geometry Size Analysis](#geometry-size-analysis)
7. [API Reference](#api-reference)
8. [Testing](#testing)

---

## Overview

### What It Is

`Frozen6502Net` is a neural network where:
- **Computation** is frozen geometry (polynomials like `XOR = a + b - 2ab`)
- **Routing** is fixed tables (opcode → shape mapping from 6502 spec)
- **Parameters** = 0 (no learning, just math)
- **Accuracy** = 100% (by construction)

### What It Proves

When this network executes correctly, it proves:
1. **Computation is geometry** - The shapes ARE the ALU
2. **The geometry is portable** - Export to ONNX, run anywhere
3. **The 6502's structure is inevitable** - Determined by mathematics

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Frozen6502Net                               │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │   Opcode    │───▶│   Routing    │───▶│   Shape Bank      │  │
│  │   (input)   │    │   (fixed)    │    │   (16 shapes)     │  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
│         │                  │                      │            │
│         ▼                  ▼                      ▼            │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │  Registers  │───▶│ Input Select │───▶│  Shape Execution  │  │
│  │   (A,X,Y)   │    │   (fixed)    │    │   (pure math)     │  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
│                                                   │            │
│                                                   ▼            │
│                                        ┌───────────────────┐  │
│                                        │  Output + Flags   │  │
│                                        └───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Properties

| Property | Value |
|----------|-------|
| Learnable parameters | 0 |
| Shapes | 16 |
| Opcodes supported | 33 (expandable to 56) |
| Accuracy | 100% |
| Differentiable | Yes |
| Batch execution | Yes |
| ONNX exportable | Yes |

---

## The 16 Shapes

Each shape is a pure mathematical function with 0 parameters.

| ID | Name | Formula | ONNX Size |
|----|------|---------|-----------|
| 0 | RIPPLE_ADD | 8× FullAdder chain | 12 KB |
| 1 | RIPPLE_SUB | 8× FullAdder (inverted) | 12 KB |
| 2 | PARALLEL_AND | `ab` per bit | 215 B |
| 3 | PARALLEL_OR | `a + b - ab` per bit | 307 B |
| 4 | PARALLEL_XOR | `a + b - 2ab` per bit | 443 B |
| 5 | SHIFT_LEFT | Bit rearrangement | 821 B |
| 6 | SHIFT_RIGHT | Bit rearrangement | 821 B |
| 7 | ROTATE_LEFT | Shift + carry insertion | 918 B |
| 8 | ROTATE_RIGHT | Shift + carry insertion | 918 B |
| 9 | INCREMENT | RIPPLE_ADD with b=1 | 15 KB |
| 10 | DECREMENT | RIPPLE_SUB with b=1 | 13 KB |
| 11 | TRANSFER | Pass-through | 202 B |
| 12 | LOAD | Pass-through | 202 B |
| 13 | STORE | Pass-through | 202 B |
| 14 | BIT_TEST | AND + flag extraction | ~500 B |
| 15 | IDENTITY | No-op | 202 B |

### Why Adders Are Large

A single **full adder** computes:
```python
sum  = XOR(XOR(a, b), cin)
cout = OR(AND(a, b), AND(XOR(a, b), cin))
```

Expanded with `XOR = a + b - 2ab`:
- 2 XORs = 6 Mul + 4 Add + 4 Sub
- 2 ANDs = 2 Mul
- 1 OR = 1 Mul + 1 Add + 1 Sub

**~16 arithmetic operations per bit × 8 bits = ~128 operations per adder.**

The 12KB ONNX size for RIPPLE_ADD is the true size of an 8-bit adder expressed as continuous polynomials.

---

## Usage

### Basic Execution

```python
from trix.nn import Frozen6502Net, CPUState, OPCODE_TABLE, bits_to_int
import torch

# Create network
net = Frozen6502Net()
net.eval()

# Create state: A=42, Memory=13, Carry=1
state = CPUState.from_ints(a=42, m=13, c=1)

# Execute ADC (opcode 0)
opcode = torch.tensor([0])  # ADC
result = net(opcode, state)

# Get result
a_value = bits_to_int(result.a)  # tensor([56]) = 42 + 13 + 1
```

### Batch Execution

```python
# Execute 4 different operations in parallel
opcodes = torch.tensor([0, 1, 5, 6])  # ADC, SBC, AND, ORA

# Same state for all (or different states)
state = CPUState.from_ints(a=0xFF, m=0x0F, c=0)

results = net(opcodes, state)
```

### Available Opcodes

```python
from trix.nn import OPCODE_TABLE, get_opcode_name

for opcode_id, spec in OPCODE_TABLE.items():
    print(f"{opcode_id:2d}: {spec.name}")
```

Current opcodes:
```
 0: ADC    8: BIT    16: ROR_M   24: TXA
 1: SBC    9: ASL_A  17: INX     25: TAY
 2: CMP   10: LSR_A  18: DEX     26: TYA
 3: CPX   11: ROL_A  19: INY     27: LDA
 4: CPY   12: ROR_A  20: DEY     28: LDX
 5: AND   13: ASL_M  21: INC     29: LDY
 6: ORA   14: LSR_M  22: DEC     30: STA
 7: EOR   15: ROL_M  23: TAX     31: STX
                                 32: STY
```

---

## ONNX Export

The frozen geometry can be exported to ONNX for portable execution.

### Export

```python
from trix.nn import Frozen6502Net
import torch

net = Frozen6502Net()
net.eval()

# Create ONNX wrapper
class Frozen6502NetONNX(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Frozen6502Net()

    def forward(self, opcode, a, x, y, memory, carry):
        from trix.nn import CPUState
        state = CPUState(a=a, x=x, y=y, memory=memory, carry=carry)
        result = self.net(opcode, state)
        return result.a, result.x, result.y, result.memory, result.carry

# Export
wrapper = Frozen6502NetONNX()
wrapper.eval()

dummy_inputs = (
    torch.zeros(1, dtype=torch.long),  # opcode
    torch.zeros(1, 8),                  # a
    torch.zeros(1, 8),                  # x
    torch.zeros(1, 8),                  # y
    torch.zeros(1, 8),                  # memory
    torch.zeros(1),                     # carry
)

torch.onnx.export(
    wrapper,
    dummy_inputs,
    "frozen_6502.onnx",
    input_names=['opcode', 'a', 'x', 'y', 'memory', 'carry'],
    output_names=['out_a', 'out_x', 'out_y', 'out_memory', 'out_carry'],
)
```

### Run with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("frozen_6502.onnx")

# EOR: A XOR Memory
opcode = np.array([7], dtype=np.int64)  # EOR
a = np.array([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=np.float32)  # 0x55
memory = np.array([[1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.float32)  # 0xFF

outputs = session.run(None, {
    'opcode': opcode,
    'a': a,
    'x': np.zeros((1, 8), dtype=np.float32),
    'y': np.zeros((1, 8), dtype=np.float32),
    'memory': memory,
    'carry': np.array([0], dtype=np.float32),
})

# Result: 0xAA (10101010)
```

### What's in the ONNX

The 73KB ONNX file contains 915 nodes:

| Node Type | Count | Purpose |
|-----------|-------|---------|
| Mul | 243 | `ab` (AND), `2ab` (XOR) |
| Add | 131 | `a + b` |
| Sub | 109 | `a + b - 2ab` (XOR), `a + b - ab` (OR) |
| Constant | 179 | Fixed values (0, 1, 2) |
| Gather | 64 | Input routing |
| Equal | 26 | Opcode selection |
| Where | 10 | Masked execution |

**The geometry IS in the ONNX.** The Mul/Add/Sub nodes literally compute the polynomials:
- `XOR(a, b) = a + b - 2ab`
- `AND(a, b) = ab`
- `OR(a, b) = a + b - ab`

---

## Geometry Size Analysis

### Why 73KB?

The monolithic ONNX computes **all 16 shapes** for every input (required for ONNX tracing), then masks by opcode selection.

### Individual Shape Sizes

```
ripple_add       : 11,996 bytes (12 KB)
ripple_sub       : 12,233 bytes (12 KB)
increment        : 14,890 bytes (15 KB)  ← Largest
decrement        : 13,381 bytes (13 KB)
rotate_left      :    918 bytes
rotate_right     :    918 bytes
shift_left       :    821 bytes
shift_right      :    821 bytes
parallel_xor     :    443 bytes
parallel_or      :    307 bytes
parallel_and     :    215 bytes
transfer         :    202 bytes
identity         :    202 bytes
─────────────────────────────────────
TOTAL            : 56,347 bytes (55 KB)
```

The carry-chain shapes (add, sub, inc, dec) dominate because they unroll 8 full-adders into ~128 arithmetic nodes each.

### Optimization

ONNX Runtime optimization reduces 73KB → 59KB:

```python
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.optimized_model_filepath = "frozen_6502_opt.onnx"

session = ort.InferenceSession("frozen_6502.onnx", sess_options)
# Optimized model saved to frozen_6502_opt.onnx
```

### The Geometry IS That Size

A 6502 ALU is ~4,000 transistors. Expressed as continuous polynomials with 8-bit precision, 73KB is the true size of the geometry. The shapes are not bloated - they ARE the computation.

---

## API Reference

### Frozen6502Net

```python
class Frozen6502Net(nn.Module):
    """A Deterministic Neural Network That IS a 6502."""

    def forward(
        self,
        opcode: torch.Tensor,    # [batch] opcode indices
        state: CPUState,         # Input state
    ) -> CPUStateWithFlags:      # Output state + flags
        """Execute one instruction per batch element."""
```

### CPUState

```python
class CPUState(NamedTuple):
    a: torch.Tensor       # [batch, 8] Accumulator
    x: torch.Tensor       # [batch, 8] X register
    y: torch.Tensor       # [batch, 8] Y register
    memory: torch.Tensor  # [batch, 8] Memory operand
    carry: torch.Tensor   # [batch] Carry flag

    @classmethod
    def from_ints(cls, a=0, x=0, y=0, m=0, c=0) -> 'CPUState':
        """Create from integer values."""

    @classmethod
    def zeros(cls, batch_size: int) -> 'CPUState':
        """Create zero-initialized state."""
```

### CPUStateWithFlags

```python
class CPUStateWithFlags(NamedTuple):
    a: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    memory: torch.Tensor
    carry: torch.Tensor
    zero: torch.Tensor     # [batch] Zero flag
    negative: torch.Tensor # [batch] Negative flag
```

### Utility Functions

```python
def int_to_bits(x: torch.Tensor) -> torch.Tensor:
    """Convert integers to bit tensors. [batch] -> [batch, 8]"""

def bits_to_int(bits: torch.Tensor) -> torch.Tensor:
    """Convert bit tensors to integers. [batch, 8] -> [batch]"""

def get_opcode_id(name: str) -> int:
    """Get opcode ID from name. 'ADC' -> 0"""

def get_opcode_name(opcode_id: int) -> str:
    """Get name from opcode ID. 0 -> 'ADC'"""
```

---

## Testing

### Run Tests

```bash
# Frozen6502Net tests
pytest tests/test_frozen_6502_net.py -v

# All frozen tests
pytest tests/test_frozen*.py -v
```

### Test Coverage

The test suite validates:
- All 33 opcodes execute correctly
- Carry propagation works
- Flag computation (Z, N, C) is accurate
- Batch execution produces correct results
- ONNX export matches PyTorch output

---

## Files

| File | Purpose |
|------|---------|
| `src/trix/nn/frozen_6502_net.py` | The neural network |
| `src/trix/nn/frozen_6502.py` | Shape implementations |
| `tests/test_frozen_6502_net.py` | Tests |
| `experiments/frozen_emulator/frozen_6502.onnx` | Exported ONNX |
| `docs/FROZEN_6502_NET.md` | This document |

---

## See Also

- [FROZEN_6502.md](FROZEN_6502.md) - Theory and shape architecture
- [FROZEN_SHAPES.md](FROZEN_SHAPES.md) - Frozen shapes overview
- [ATOMIC_FUNCTIONS.md](ATOMIC_FUNCTIONS.md) - Pure math foundations

---

*"The Mul/Add/Sub nodes ARE the polynomial. The geometry computes."*
