# trix.native - Framework-Free Execution

**Zero PyTorch. Zero TensorFlow. Pure computation.**

`trix.native` provides the execution layer for TriX - frozen shapes that compute without frameworks, trained routing that learns without gradients, and CUDA kernels that run at hardware speed.

## When to Use

| Use Case | Module |
|----------|--------|
| **Inference/Execution** | `trix.native` (this module) |
| **Pure Python computation** | `trix.shapes` |
| **Gradient-based training** | `trix.nn.frozen_shapes` |

## Quick Start

### Frozen Shapes (Pure Math)

```python
from trix.native import FrozenShapes, FrozenALU

# Core primitives
result = FrozenShapes.xor(a, b)      # XOR: a + b - 2ab (on binary: a ^ b)
result = FrozenShapes.and_op(a, b)   # AND: ab
result = FrozenShapes.or_op(a, b)    # OR: a + b - ab
result = FrozenShapes.not_op(a)      # NOT: 1 - a

# 6502 ALU operations (16 frozen shapes)
alu = FrozenALU()
result = alu.execute('RIPPLE_ADD', a_bits, b_bits, carry)
result = alu.execute('PARALLEL_XOR', a_bits, b_bits)
result = alu.execute('SHIFT_LEFT', a_bits)
```

### Frozen Hybrid (Learned Routing + Frozen Execution)

```python
from trix.native import NativeFrozenHybrid

# Create model: 0 params in shapes, ~176 in routing
model = NativeFrozenHybrid()

# Train: learns which shape to use for each opcode
model.train_supervised()

# Result: 100% accuracy on 6502 ALU operations
# The shapes ARE the computation. Learning IS routing.
```

### Sparse Octave Lookup (Transformer FFN Replacement)

```python
from trix.native import SparseOctaveLookupFFN

# Multi-scale sparse memory instead of dense MatMul
ffn = SparseOctaveLookupFFN(d_model=768, n_octaves=3)

# Each octave operates at different scale:
# - Octave 0: Fine-grained patterns
# - Octave 1: Medium-scale features
# - Octave 2: Coarse abstractions
```

## Components

### FrozenShapes

Mathematical primitives as continuous polynomials:

```
XOR(a, b) = a + b - 2ab    # Saddle surface
AND(a, b) = ab              # Product
OR(a, b)  = a + b - ab      # Union
NOT(a)    = 1 - a           # Reflection
```

On binary inputs {0, 1}, these are **exact**. The polynomial form provides smooth gradients for training; the binary form provides exact execution.

### FrozenALU

The 16 frozen shapes needed for 6502 CPU emulation:

| Shape | Operation | Description |
|-------|-----------|-------------|
| RIPPLE_ADD | a + b + c | 8-bit ripple-carry adder |
| RIPPLE_SUB | a - b - c | 8-bit subtractor |
| PARALLEL_AND | a & b | Bitwise AND |
| PARALLEL_OR | a \| b | Bitwise OR |
| PARALLEL_XOR | a ^ b | Bitwise XOR |
| SHIFT_LEFT | a << 1 | Arithmetic shift left |
| SHIFT_RIGHT | a >> 1 | Logical shift right |
| ROTATE_LEFT | rol(a, c) | Rotate left through carry |
| ROTATE_RIGHT | ror(a, c) | Rotate right through carry |
| INCREMENT | a + 1 | Increment |
| DECREMENT | a - 1 | Decrement |
| TRANSFER | a | Pass-through |
| LOAD | mem | Load from memory |
| STORE | a | Store to memory |
| BIT_TEST | a & m | Test bits |
| IDENTITY | a | No operation |

### NativeRouter

Learnable routing layer that maps inputs to shapes:

```python
from trix.native import NativeRouter

router = NativeRouter(num_shapes=16)

# Training: supervised learning of opcode->shape mapping
router.supervised_update(opcode, correct_shape)

# Inference: sample the most likely shape
shape_id = router.sample_shape(opcode)
```

### NativeFrozenHybrid

Complete model combining frozen shapes with learned routing:

```python
from trix.native import NativeFrozenHybrid

model = NativeFrozenHybrid()
model.train_supervised()

# Get the frozen dispatch table
table = model.get_routing_table()
# {0: 'RIPPLE_ADD', 1: 'RIPPLE_SUB', ...}
```

### Providence (Content-Addressed Memory)

Hamming distance-based routing for sparse memory access:

```python
from trix.native import Providence

# Binary signatures for content addressing
providence = Providence(num_entries=1024, signature_dim=64)

# Route to nearest signature (minimum Hamming distance)
indices = providence.route(query_signatures)
```

## Backend

`trix.native` uses:
- **CuPy** for GPU training (when available)
- **NumPy** for CPU inference (always works)

No PyTorch, no TensorFlow, no ONNX runtime. The frozen shapes are mathematical truths that need no framework.

## Relationship to Other Modules

```
┌─────────────────────────────────────────────────────────────┐
│                      trix.shapes                            │
│                   (Pure Python, Zero deps)                  │
│                      add(), xor(), inc()                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      trix.native                            │
│                   (CuPy/NumPy execution)                    │
│         FrozenShapes, FrozenALU, NativeFrozenHybrid        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      trix.nn.*                              │
│                   (PyTorch training)                        │
│     frozen_shapes (gradients), frozen_6502 (nn.Module)     │
│                                                             │
│  ⚠️ DEPRECATED for execution - use trix.native instead     │
│     Still valid for gradient-based training                 │
└─────────────────────────────────────────────────────────────┘
```

## Export to C

The frozen shapes can be exported to pure C for embedded deployment:

```python
from foundry.export import DispatchExporter

exporter = DispatchExporter(
    dispatch_table=model.get_routing_table(),
    shapes=['RIPPLE_ADD', 'PARALLEL_XOR', ...],
    name='my_alu'
)
exporter.export('./output/')
# Generates: my_alu.h, my_alu.c
```

## Performance

| Backend | Throughput | Use Case |
|---------|------------|----------|
| HSOS Python | ~3M ops/sec | Development, testing |
| NumPy (CPU) | ~50M ops/sec | Batch inference |
| CuPy (GPU) | ~100M ops/sec | Training |
| GILLIES (C) | ~900M ops/sec | Production CPU |
| **GILLIES Vulkan** | **~19B ops/sec** | **GPU compute** |

The shapes are the same. Only the execution substrate changes.

## GILLIES Vulkan

For GPU-accelerated shape execution, see [`vulkan/`](./vulkan/):

```bash
cd vulkan
make
./gillies_vulkan_bench
```

Results on NVIDIA Thor GPU:

```
Elements        GPU (M/s)    CPU (M/s)  GPU Speedup
16777216         19440         1245        15.6x
```

**19.4 billion XOR operations per second.**

See [GILLIES_VULKAN.md](/workspace/ZOR/docs/GILLIES_VULKAN.md) for full documentation.
