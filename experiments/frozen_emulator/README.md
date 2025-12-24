# Frozen 6502 Emulator

A complete MOS 6502 CPU emulator built on frozen geometric shapes.

---

## Overview

This directory contains two implementations of the 6502 processor:

| File | Type | Lines | Purpose |
|------|------|-------|---------|
| `frozen_6502.py` | Python | 1,278 | Traditional emulator with frozen shape duals |
| `frozen_6502.onnx` | ONNX | 73KB | Neural network export for portable execution |

Both implementations use the same underlying mathematics - the frozen shapes that compute exactly.

---

## Python Emulator

### Quick Start

```python
from frozen_6502 import CPU6502

# Create CPU
cpu = CPU6502()

# Load a simple program: LDA #$42, STA $00, BRK
program = bytes([
    0xA9, 0x42,  # LDA #$42
    0x85, 0x00,  # STA $00
    0x00,        # BRK
])
cpu.load_binary(program, 0x0600)

# Run from address
cpu.run(0x0600)

# Check result
print(f"A = ${cpu.a:02X}")       # A = $42
print(f"[$00] = ${cpu.memory[0]:02X}")  # [$00] = $42
```

### Features

- **151 valid opcodes** (all official 6502 instructions)
- **56 unique instructions** (ADC, SBC, AND, ORA, EOR, etc.)
- **13 addressing modes** (immediate, zero page, absolute, indexed, indirect, etc.)
- **Complete flag handling** (N, V, B, D, I, Z, C)
- **48 built-in self-tests**
- **Zero dependencies** (pure Python + standard library)

### Running the Emulator

```bash
# Run built-in tests
python frozen_6502.py

# Expected output:
# Testing frozen shapes...
# Testing CPU instructions...
# ...
# All 48 tests passed!
```

### The Frozen Shapes

The emulator uses integer duals of the frozen mathematical shapes:

```python
def frozen_add(a: int, b: int, c: int) -> Tuple[int, int]:
    """8-bit addition with carry."""
    total = a + b + c
    return total & 0xFF, (total >> 8) & 1

def frozen_xor(a: int, b: int) -> int:
    """Bitwise XOR."""
    return a ^ b
```

These are the same operations as the neural network's polynomial forms (`XOR = a + b - 2ab`), just expressed in integer arithmetic for efficiency.

---

## ONNX Model

### What's Inside

The `frozen_6502.onnx` file is a 73KB neural network with:
- **0 learnable parameters**
- **16 frozen shapes** as arithmetic operations
- **33 opcodes** with fixed routing

### Geometry Breakdown

| Node Type | Count | Purpose |
|-----------|-------|---------|
| Mul | 243 | AND (`ab`), XOR (`2ab` term) |
| Add | 131 | Sum terms |
| Sub | 109 | XOR (`a + b - 2ab`), OR (`a + b - ab`) |
| Constant | 179 | Fixed values |

### Using the ONNX

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("frozen_6502.onnx")

# Execute ADC: 42 + 13 + 1 = 56
def bits(val):
    return np.array([[float((val >> i) & 1) for i in range(8)]], dtype=np.float32)

outputs = session.run(None, {
    'opcode': np.array([0], dtype=np.int64),  # ADC
    'a': bits(42),
    'x': np.zeros((1, 8), dtype=np.float32),
    'y': np.zeros((1, 8), dtype=np.float32),
    'memory': bits(13),
    'carry': np.array([1], dtype=np.float32),
})

result = int(sum(outputs[0][0, i] * (2**i) for i in range(8)))
print(f"Result: {result}")  # 56
```

---

## Tests

### Python Tests

```bash
# Run pytest tests
cd /workspace/trix_latest/TriXO
pytest experiments/frozen_emulator/test_frozen_6502.py -v

# 73 tests covering:
# - All frozen shapes
# - All addressing modes
# - All ALU operations
# - Integration tests (multiply routine, etc.)
```

### Neural Network Tests

```bash
# Run frozen 6502 net tests
pytest tests/test_frozen_6502_net.py -v

# 27 tests covering:
# - All 33 opcodes
# - Carry propagation
# - Flag computation
# - ONNX verification
```

---

## Architecture

### CPU State

```python
class CPU6502:
    a: int       # Accumulator (8-bit)
    x: int       # X register (8-bit)
    y: int       # Y register (8-bit)
    sp: int      # Stack pointer (8-bit)
    pc: int      # Program counter (16-bit)
    p: int       # Status register (8-bit): NV-BDIZC
    memory: bytearray  # 64KB address space
```

### Instruction Execution

```
1. Fetch opcode from memory[PC]
2. Decode: opcode → (mnemonic, addressing_mode, length, cycles)
3. Calculate effective address (addressing mode)
4. Execute operation (frozen shape)
5. Update flags
6. Advance PC
```

### Addressing Modes

| Mode | Example | Effective Address |
|------|---------|-------------------|
| Immediate | `LDA #$42` | PC + 1 |
| Zero Page | `LDA $00` | memory[PC+1] |
| Zero Page,X | `LDA $00,X` | (memory[PC+1] + X) & 0xFF |
| Absolute | `LDA $1234` | memory[PC+1:PC+3] |
| Absolute,X | `LDA $1234,X` | memory[PC+1:PC+3] + X |
| Absolute,Y | `LDA $1234,Y` | memory[PC+1:PC+3] + Y |
| Indirect | `JMP ($1234)` | memory[memory[PC+1:PC+3]] |
| (Indirect,X) | `LDA ($00,X)` | memory[(memory[PC+1] + X) & 0xFF] |
| (Indirect),Y | `LDA ($00),Y` | memory[memory[PC+1]] + Y |
| Relative | `BEQ $FE` | PC + 2 + signed(memory[PC+1]) |

---

## Why Two Implementations?

| Implementation | Pros | Cons |
|----------------|------|------|
| Python | Fast, complete, easy to debug | Not portable to other runtimes |
| ONNX | Portable, embeddable, GPU-capable | Larger, all shapes computed |

The Python emulator is for development and testing. The ONNX export proves the geometry can be serialized and executed anywhere.

---

## Files

| File | Description |
|------|-------------|
| `frozen_6502.py` | Complete Python 6502 emulator |
| `test_frozen_6502.py` | Pytest tests (73 tests) |
| `frozen_6502.onnx` | Exported neural network (73KB) |
| `README.md` | This file |

---

## See Also

- [FROZEN_6502.md](../../docs/FROZEN_6502.md) - Theory and architecture
- [FROZEN_6502_NET.md](../../docs/FROZEN_6502_NET.md) - Neural network API
- [FROZEN_SHAPES.md](../../docs/FROZEN_SHAPES.md) - Frozen shapes overview

---

*"Mario jumps because the geometry computes his trajectory."*
