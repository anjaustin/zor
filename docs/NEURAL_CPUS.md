# Neural CPUs

**100% Accurate Neural Network Emulation of Classic Processors**

---

## Overview

FrozenFoundry enables the creation of neural network versions of classic CPUs that achieve **100% accuracy with 0 training steps**. This document covers the four CPUs currently implemented:

| CPU | Year | Bits | Operations | Shapes | Parameters | Status |
|-----|------|------|------------|--------|------------|--------|
| MOS 6502 | 1975 | 8 | 30 | 17 | 510 | 100% |
| WDC 65816 | 1983 | 16 | 24 | 18 | 432 | 100% |
| Intel 80286 | 1982 | 16 | 32 | 26 | 832 | 100% |
| Intel 80486 | 1989 | 32 | 34 | 30 | 1,020 | 100% |

---

## MOS 6502 (1975)

The MOS Technology 6502 was one of the most influential 8-bit microprocessors in computing history. It powered the Apple II, Commodore 64, NES, Atari 2600, and countless other systems.

### Specifications

- **Architecture**: 8-bit
- **Transistors**: 3,510
- **Clock speed**: 1-2 MHz (original)
- **Address bus**: 16-bit (64KB addressable)

### Implemented Operations

```
Arithmetic:    ADC, SBC
Logic:         AND, ORA, EOR
Shifts:        ASL, LSR, ROL, ROR
Inc/Dec:       INC, DEC, INX, DEX, INY, DEY
Transfer:      TAX, TXA, TAY, TYA, TSX, TXS
Load/Store:    LDA, LDX, LDY, STA, STX, STY
Compare:       CMP, CPX, CPY
```

### Usage

```python
from trix.foundry.mos6502 import build_6502

foundry, result = build_6502()
print(f"Accuracy: {result.accuracy * 100}%")  # 100.0%
print(f"Training steps: {result.training_steps}")  # 0

# Export
foundry.export_onnx("mos_6502.onnx")
```

---

## WDC 65816 (1983)

The Western Design Center 65816 is the 16-bit successor to the 6502. It powered the Apple IIGS and Super Nintendo Entertainment System (SNES).

### Specifications

- **Architecture**: 16-bit
- **Backward compatible**: With 6502 (emulation mode)
- **Address bus**: 24-bit (16MB addressable)

### New Operations (vs 6502)

```
XBA    - Exchange B and A (swap high/low bytes)
TCS    - Transfer Accumulator to Stack Pointer
TSC    - Transfer Stack Pointer to Accumulator
TCD    - Transfer Accumulator to Direct Page
TDC    - Transfer Direct Page to Accumulator
```

### Usage

```python
from trix.foundry.wdc65816 import build_65816

foundry, result = build_65816()
# Same 100% accuracy, scaled to 16-bit
```

---

## Intel 80286 (1982)

The Intel 80286 was the second generation x86 processor, introducing protected mode for multitasking operating systems.

### Specifications

- **Architecture**: x86 (16-bit)
- **Transistors**: 134,000
- **Clock speed**: 6-12.5 MHz
- **Protected mode**: Yes (but no virtual 8086 mode)

### x86 Operations

```
Arithmetic:    ADD, ADC, SUB, SBB, INC, DEC, NEG, CMP
Logic:         AND, OR, XOR, NOT, TEST
Shifts:        SHL, SHR, SAR
Rotates:       ROL, ROR, RCL, RCR
Data:          MOV, XCHG, CBW, CWD
Flags:         CLC, STC, CMC
Stack:         PUSH, POP
286-specific:  BOUND, ENTER, LEAVE
```

### Custom Shapes Required

The x86 architecture requires shapes not present in 6502:

| Shape | Description | Why Needed |
|-------|-------------|------------|
| `negate` | Two's complement negation | NEG instruction |
| `add_no_carry` | ADD without carry input | x86 ADD ignores CF |
| `shift_right_arith` | Preserves sign bit | SAR instruction |
| `rotate_left_no_carry` | Rotate without CF | ROL instruction |
| `rotate_right_no_carry` | Rotate without CF | ROR instruction |
| `sign_extend` | CBW: byte to word | Sign extension |
| `sign_to_word` | CWD: all 0s or all 1s | DX:AX doubleword |
| `return_second` | Return operand B | XCHG semantics |

### Usage

```python
from trix.foundry.intel286 import build_286

foundry, result = build_286()
```

---

## Intel 80486 (1989)

The Intel 80486 added an on-chip FPU, L1 cache, and pipelined execution. It was the last x86 chip before the Pentium.

### Specifications

- **Architecture**: x86 (32-bit)
- **Transistors**: 1,200,000
- **Clock speed**: 25-50 MHz (original)
- **On-chip FPU**: Yes (x87)
- **L1 Cache**: 8KB

### 32-bit Operations

All 286 operations extended to 32-bit, plus:

```
MOVSX8   - Sign extend byte to dword
MOVSX16  - Sign extend word to dword
MOVZX8   - Zero extend byte to dword
MOVZX16  - Zero extend word to dword
CDQ      - Sign extend EAX to EDX:EAX
BSWAP    - Byte swap (endianness)
BT       - Bit test
XADD     - Exchange and add (486+)
CMPXCHG  - Compare and exchange (486+)
```

### 32-bit Custom Shapes

| Shape | Description |
|-------|-------------|
| `sign_extend_8_32` | Sign extend 8-bit to 32-bit |
| `sign_extend_16_32` | Sign extend 16-bit to 32-bit |
| `zero_extend_8_32` | Zero extend 8-bit to 32-bit |
| `zero_extend_16_32` | Zero extend 16-bit to 32-bit |
| `sign_to_dword` | CDQ: all 0s or all 1s |
| `bswap` | Swap byte order |
| `conditional_select` | MUX: if c then b else a |

### Usage

```python
from trix.foundry.intel386 import build_486

foundry, result = build_486()
```

---

## How It Works

### Frozen Shapes

Each CPU operation maps to a **frozen shape** - a mathematical function with 0 learnable parameters:

```python
# XOR gate as continuous polynomial
XOR(a, b) = a + b - 2ab    # Saddle surface

# Full adder from gates
sum  = XOR(XOR(a, b), c)
cout = OR(AND(a, b), AND(c, XOR(a, b)))
```

### Routing (The Only Learned Part)

The router maps opcodes to shapes:

```
Opcode [ADC] --> Router --> Shape [ripple_add] --> Result
                   |
             (510 params for 6502)
```

When shapes match perfectly, **no training is required**.

### Bit-Width Scaling

The same shapes work at any bit width:

```python
# 8-bit adder (6502)
foundry_8 = FrozenFoundry(bit_width=8)

# 16-bit adder (65816)
foundry_16 = FrozenFoundry(bit_width=16)

# 32-bit adder (486)
foundry_32 = FrozenFoundry(bit_width=32)
```

The frozen shapes scale automatically.

---

## Export Formats

### ONNX

Portable format for deployment:

```python
foundry.export_onnx("cpu.onnx")
```

### PyTorch

Native format with full model state:

```python
foundry.export("cpu.pt")
```

### Loading

```python
import torch

model = torch.load("exports/mos_6502.pt")
```

---

## Performance

| CPU | Build Time | Validation Time | ONNX Size |
|-----|------------|-----------------|-----------|
| 6502 | ~400ms | ~500ms | ~50KB |
| 65816 | ~450ms | ~500ms | ~55KB |
| 286 | ~500ms | ~600ms | ~80KB |
| 486 | ~650ms | ~700ms | ~120KB |

---

## File Reference

```
src/trix/foundry/
  frozen_foundry.py     # Core framework
  mos6502.py            # MOS 6502 (8-bit)
  wdc65816.py           # WDC 65816 (16-bit)
  intel286.py           # Intel 80286 (16-bit)
  intel386.py           # Intel 80486 (32-bit)
  export_all_cpus.py    # Export script

exports/
  mos_6502.onnx         # 6502 ONNX model
  mos_6502.pt           # 6502 PyTorch model
  wdc_65816.onnx        # 65816 ONNX model
  wdc_65816.pt          # 65816 PyTorch model
  intel_80286.onnx      # 286 ONNX model
  intel_80286.pt        # 286 PyTorch model
  intel_80486.onnx      # 486 ONNX model
  intel_80486.pt        # 486 PyTorch model
```

---

## Adding New CPUs

To add a new CPU:

1. Create a new file in `src/trix/foundry/`
2. Register operations with ground truth functions
3. Add custom shapes if needed
4. Build and validate

```python
from trix.foundry.frozen_foundry import FrozenFoundry

def build_my_cpu():
    foundry = FrozenFoundry(bit_width=8)

    # Register operations
    foundry.register("ADD", lambda a, b, c: ((a + b) & 0xFF, int((a + b) > 255)))

    # Add custom shape if needed
    def my_shape(a, b, c):
        # ... pure math implementation ...
        return result, carry
    foundry.register_shape("my_shape", my_shape, n_inputs=2)

    # Build
    result = foundry.build()
    return foundry, result
```

---

## Theory

### Why 100% Accuracy?

Deterministic systems have exact truth tables. We don't approximate - we discover the frozen shape that matches exactly.

### Why 0 Training Steps?

If all operations match existing shapes, the router can be initialized perfectly. No gradient descent needed.

### Compression

Traditional emulators implement each instruction as code. Our approach:

- **6502**: 30 operations in 510 parameters
- **486**: 34 operations in 1,020 parameters

The frozen shapes are shared across operations, achieving massive compression.

---

*"Computation is topology. Learning is routing."*
