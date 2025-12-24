# FrozenFoundry

**Neural Models for Deterministic Systems**

Turn any deterministic system into a 100% accurate neural model in under a second.

---

## Quick Start

```python
from trix.foundry import FrozenFoundry

# Create foundry
foundry = FrozenFoundry(bit_width=8)

# Register operations with ground truth
foundry.register("add", lambda a, b, c: ((a + b + c) & 0xFF, int((a + b + c) > 255)))
foundry.register("sub", lambda a, b, c: ((a + (b ^ 0xFF) + c) & 0xFF, int((a + (b ^ 0xFF) + c) > 255)))
foundry.register("and", lambda a, b, c: (a & b, 0))
foundry.register("xor", lambda a, b, c: (a ^ b, 0))

# Build model
result = foundry.build()

# Validate
assert foundry.validate() == 1.0  # 100% accuracy

# Export
foundry.export("my_alu.pt")
foundry.export_onnx("my_alu.onnx")
```

---

## How It Works

### 1. Shape Library

FrozenFoundry includes 17 built-in frozen shapes:

| Shape | Operation | Description |
|-------|-----------|-------------|
| `ripple_add` | ADC | 8-bit addition with carry |
| `ripple_sub` | SBC | 8-bit subtraction with borrow |
| `parallel_and` | AND | Bitwise AND |
| `parallel_or` | ORA | Bitwise OR |
| `parallel_xor` | EOR | Bitwise XOR |
| `parallel_nand` | NAND | Bitwise NAND |
| `parallel_nor` | NOR | Bitwise NOR |
| `parallel_xnor` | XNOR | Bitwise XNOR |
| `shift_left` | ASL | Shift left |
| `shift_right` | LSR | Shift right |
| `rotate_left` | ROL | Rotate left through carry |
| `rotate_right` | ROR | Rotate right through carry |
| `increment` | INC | Add 1 |
| `decrement` | DEC | Subtract 1 |
| `identity` | NOP | Pass through |
| `complement` | NOT | Bitwise complement |
| `compare` | CMP | Subtract for flags |

### 2. Shape Matching

When you register an operation with its ground truth function:

```python
foundry.register("my_op", lambda a, b, c: (result, carry))
```

The foundry automatically tests your function against all shapes to find a 100% match.

### 3. Router Training

The only learned component is the router - mapping operations to shapes:

```
Operation → [Router] → Shape → Result
              ↑
         (learned)
```

If shapes match exactly, **no training is needed** (0 steps).

---

## API Reference

### FrozenFoundry

```python
foundry = FrozenFoundry(
    bit_width=8,      # Bits per operand (default: 8)
    device='cpu',     # PyTorch device
)
```

### register()

Register an operation with its ground truth:

```python
foundry.register(
    name="add",
    truth_fn=lambda a, b, c: (result, carry)
)
```

- `name`: Operation name
- `truth_fn`: Function `(a: int, b: int, c: int) -> (result: int, carry: int)`
  - `a`: First operand (0 to 2^bit_width - 1)
  - `b`: Second operand
  - `c`: Carry in (0 or 1)
  - Returns: `(result, carry_out)`

### register_shape()

Register a custom frozen shape:

```python
def my_shape(a: Tensor, b: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
    # a, b: [batch, bit_width] bit tensors
    # c: [batch] carry tensor
    # Returns: (result, carry_out)
    ...

foundry.register_shape("my_shape", my_shape, n_inputs=2, uses_carry=True)
```

### build()

Build the neural model:

```python
result = foundry.build(
    max_steps=100,        # Max training steps
    lr=0.1,               # Learning rate
    batch_size=1024,      # Batch size
    n_samples_per_op=1000,# Samples per operation
    verbose=True,         # Print progress
)

# result.accuracy       - Final accuracy
# result.training_steps - Steps taken (0 if no training needed)
# result.num_params     - Parameter count
```

### validate()

Validate accuracy on random samples:

```python
accuracy = foundry.validate(n_samples=10000)
assert accuracy == 1.0
```

### export()

Export the model:

```python
foundry.export("model.pt")      # PyTorch checkpoint
foundry.export_onnx("model.onnx")  # ONNX format
```

---

## Examples

### 8-bit ALU

```python
foundry = FrozenFoundry(bit_width=8)

# Arithmetic
foundry.register("ADD", lambda a, b, c: ((a + b + c) & 0xFF, int((a + b + c) > 255)))
foundry.register("SUB", lambda a, b, c: ((a + (b ^ 0xFF) + c) & 0xFF, int((a + (b ^ 0xFF) + c) > 255)))

# Logic
foundry.register("AND", lambda a, b, c: (a & b, 0))
foundry.register("OR",  lambda a, b, c: (a | b, 0))
foundry.register("XOR", lambda a, b, c: (a ^ b, 0))

# Shifts
foundry.register("SHL", lambda a, b, c: ((a << 1) & 0xFF, (a >> 7) & 1))
foundry.register("SHR", lambda a, b, c: (a >> 1, a & 1))

# Build
result = foundry.build()
print(f"Accuracy: {result.accuracy * 100}%")  # 100%
print(f"Training: {result.training_steps} steps")  # 0
```

### 16-bit ALU

```python
foundry = FrozenFoundry(bit_width=16)

foundry.register("ADD16", lambda a, b, c: (
    (a + b + c) & 0xFFFF,
    int((a + b + c) > 65535)
))

result = foundry.build()
```

### Custom Shape

```python
def barrel_shift_left(a, b, c):
    # b specifies shift amount (0-7)
    result = []
    carry = c.squeeze(-1) if c.dim() > 1 else c
    shift = b[:, :3]  # Use low 3 bits of b as shift amount

    # ... implement barrel shifter ...

    return result, carry

foundry.register_shape("barrel_shift", barrel_shift_left, n_inputs=2)
```

---

## Performance

| Metric | Value |
|--------|-------|
| Shape matching | < 100ms |
| Training (if needed) | < 1 second |
| Typical accuracy | **100%** |
| Parameters (8-bit, 9 ops) | 153 |

---

## Theory

### Why It Works

Deterministic systems have exact truth tables. Instead of learning an approximation:

1. **Discover** the frozen shape that matches the truth table
2. **Route** operations to shapes (the only learned part)
3. **Execute** frozen geometry (0 learnable parameters)

### The Math

Frozen shapes use continuous polynomial forms:

```python
XOR(a, b) = a + b - 2ab      # Saddle surface
AND(a, b) = ab               # Product
OR(a, b)  = a + b - ab       # Union
NOT(a)    = 1 - a            # Reflection
```

These are:
- **Differentiable** (gradients flow)
- **Exact on {0, 1}** (100% accuracy on binary inputs)
- **Composable** (build complex shapes from primitives)

---

## Implemented CPUs

FrozenFoundry has been used to create 100% accurate neural network versions of classic CPUs:

| CPU | Year | Bits | Operations | Parameters | Time |
|-----|------|------|------------|------------|------|
| **MOS 6502** | 1975 | 8 | 30 | 510 | 400ms |
| **WDC 65816** | 1983 | 16 | 24 | 432 | 450ms |
| **Intel 80286** | 1982 | 16 | 32 | 832 | 500ms |
| **Intel 80486** | 1989 | 32 | 34 | 1,020 | 650ms |

All achieve **100% accuracy with 0 training steps**.

See [NEURAL_CPUS.md](NEURAL_CPUS.md) for detailed documentation.
See [SHAPE_LIBRARY.md](SHAPE_LIBRARY.md) for the complete shape reference.

### Running the CPUs

```python
# MOS 6502
from trix.foundry.mos6502 import build_6502
foundry, result = build_6502()

# WDC 65816
from trix.foundry.wdc65816 import build_65816
foundry, result = build_65816()

# Intel 80286
from trix.foundry.intel286 import build_286
foundry, result = build_286()

# Intel 80486
from trix.foundry.intel386 import build_486
foundry, result = build_486()
```

### Export All CPUs

```bash
python -m trix.foundry.export_all_cpus
```

This exports all CPU models to `exports/` in both ONNX and PyTorch formats.

---

## Use Cases

- **CPU Emulation**: 6502, 65816, x86, Z80, ARM instruction sets
- **Hardware Simulation**: FPGA designs, ASIC verification
- **Cryptography**: Hash functions, block ciphers
- **Protocol Parsing**: CRC, checksums, encoding
- **Game Logic**: Deterministic game rules

---

## Related Documentation

- [NEURAL_CPUS.md](NEURAL_CPUS.md) - Detailed CPU documentation
- [SHAPE_LIBRARY.md](SHAPE_LIBRARY.md) - Complete shape reference

---

*"Computation is topology. Learning is routing."*
