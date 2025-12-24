# XORPU: Neural-Geometric Distributed Processing Fabric

> Complete specification for exact computation via polynomial frozen shapes.

**Version:** 1.0.0
**Date:** 2025-12-22
**Status:** Production Ready

---

## Executive Summary

XORPU is a coprocessor architecture that achieves **100% accuracy** on logical and arithmetic operations by expressing computation as polynomial evaluation over binary inputs. Instead of approximation, XORPU computes exactly.

**The Core Insight:**
```
XOR(a, b) = a + b - 2ab
AND(a, b) = ab
NOT(a)    = 1 - a

Everything else is composition.
```

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [The Three Atoms](#the-three-atoms)
3. [Explicit Terms](#explicit-terms)
4. [Supported Shapes](#supported-shapes)
5. [Backend Layer](#backend-layer)
6. [Foundry (Unified Interface)](#foundry-unified-interface)
7. [Chip DSL (Legacy)](#chip-dsl-legacy)
8. [Composition Operators](#composition-operators)
9. [Signature System](#signature-system)
10. [API Reference](#api-reference)
11. [Export Formats](#export-formats)
12. [Hardware Validation](#hardware-validation)
13. [Performance](#performance)
14. [Quick Start](#quick-start)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         XORPU                                    │
├─────────────────────────────────────────────────────────────────┤
│  Inputs: a[bits], b[bits]                                       │
│  Output: result[bits]                                           │
│  Select: shape_id[4]                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ... ┌─────┐           │
│  │ XOR │ │ AND │ │ OR  │ │ NOT │ │ ADD │     │SLTU │           │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘     └──┬──┘           │
│     │       │       │       │       │           │               │
│     └───────┴───────┴───────┴───────┴───────────┘               │
│                         │                                        │
│                    ┌────┴────┐                                   │
│                    │   MUX   │ ← shape_id                        │
│                    └────┬────┘                                   │
│                         │                                        │
│                    result[bits]                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Properties

| Property | Value |
|----------|-------|
| Accuracy | 100% (exact computation) |
| Shapes | 15 operations |
| Bit widths | 8, 16, 32, 64 |
| Total terms (32-bit) | 930 |
| Execution | 1 cycle (combinational) |

---

## The Three Atoms

All computation in XORPU derives from three polynomial primitives:

### XOR (Exclusive Or)
```
XOR(a, b) = a + b - 2ab

Truth table verification:
  a=0, b=0: 0 + 0 - 0 = 0 ✓
  a=0, b=1: 0 + 1 - 0 = 1 ✓
  a=1, b=0: 1 + 0 - 0 = 1 ✓
  a=1, b=1: 1 + 1 - 2 = 0 ✓
```

### AND (Conjunction)
```
AND(a, b) = ab

Truth table verification:
  a=0, b=0: 0×0 = 0 ✓
  a=0, b=1: 0×1 = 0 ✓
  a=1, b=0: 1×0 = 0 ✓
  a=1, b=1: 1×1 = 1 ✓
```

### NOT (Negation)
```
NOT(a) = 1 - a

Truth table verification:
  a=0: 1 - 0 = 1 ✓
  a=1: 1 - 1 = 0 ✓
```

### Derived Operations

| Operation | Derivation |
|-----------|------------|
| OR(a,b) | a + b - ab |
| NAND(a,b) | 1 - ab |
| NOR(a,b) | 1 - a - b + ab |
| XNOR(a,b) | 1 - a - b + 2ab |

---

## Explicit Terms

XORPU represents all computations as explicit polynomial terms.

### Data Structures

```python
@dataclass(frozen=True)
class Term:
    """A single polynomial term: coefficient × product of variables."""
    coefficient: int       # -2, -1, 1, 2
    variables: Tuple[int, ...]  # Input bit indices, sorted

@dataclass
class BitTerms:
    """Terms for a single output bit."""
    bit_index: int
    terms: List[Term]

@dataclass
class ShapeTerms:
    """Complete polynomial specification for a shape."""
    name: str
    input_bits: int
    output_bits: int
    bit_terms: List[BitTerms]
```

### Example: XOR Bit 0

```python
# XOR for bit 0: result[0] = a[0] + b[0] - 2*a[0]*b[0]
BitTerms(
    bit_index=0,
    terms=[
        Term(1, (0,)),      # +1 × a[0]
        Term(1, (32,)),     # +1 × b[0]  (b starts at index 32)
        Term(-2, (0, 32)),  # -2 × a[0] × b[0]
    ]
)
```

### Term Counts by Shape (32-bit)

| Shape | Terms per bit | Total terms |
|-------|---------------|-------------|
| XOR | 3 | 96 |
| AND | 1 | 32 |
| OR | 3 | 96 |
| NOT | 2 | 64 |
| NAND | 2 | 64 |
| NOR | 4 | 128 |
| XNOR | 4 | 128 |
| NOP | 1 | 32 |
| ADD | 3 | 96 |
| SUB | 3 | 96 |

---

## Supported Shapes

### Logical Operations (Fast Path)

| ID | Shape | Operation | Terms/bit |
|----|-------|-----------|-----------|
| 0 | XOR | a ⊕ b | 3 |
| 1 | AND | a ∧ b | 1 |
| 2 | OR | a ∨ b | 3 |
| 3 | NOT | ¬a | 2 |
| 4 | NAND | ¬(a ∧ b) | 2 |
| 5 | NOR | ¬(a ∨ b) | 4 |
| 6 | XNOR | ¬(a ⊕ b) | 4 |
| 7 | NOP | a | 1 |

### Arithmetic Operations

| ID | Shape | Operation | Notes |
|----|-------|-----------|-------|
| 8 | ADD | a + b | Ripple carry (structural) |
| 9 | SUB | a - b | Two's complement |

### Shift Operations

| ID | Shape | Operation |
|----|-------|-----------|
| 10 | SLL | a << b |
| 11 | SRL | a >> b (logical) |
| 12 | SRA | a >> b (arithmetic) |

### Comparison Operations

| ID | Shape | Operation |
|----|-------|-----------|
| 13 | SLT | a < b (signed) |
| 14 | SLTU | a < b (unsigned) |

---

## Backend Layer

CUDA is the hardware layer until custom silicon exists.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Application                                                 │
│  chip.execute(a, b, "xor")                                   │
├─────────────────────────────────────────────────────────────┤
│  Backend Interface                                           │
│  execute(), execute_batch(), compile_shapes()                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┬─────────────────────┐              │
│  │ CUDABackend         │ CPUBackend          │              │
│  │ (NVIDIA Thor)       │ (reference)         │              │
│  │ - Auto-detected     │ - Always available  │              │
│  │ - GPU parallel      │ - Polynomial eval   │              │
│  └─────────────────────┴─────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### Usage

```python
from trix.forge import get_backend, list_backends

# List available backends
backends = list_backends()
# {'cpu': BackendInfo(available=True), 'cuda': BackendInfo(available=True)}

# Auto-detect best backend
backend = get_backend()
print(backend.info())  # BackendInfo(name='CUDA', device='NVIDIA Thor')

# Execute single operation
result = backend.execute("xor", 0xAAAA, 0x5555, bits=16)
# result = 0xFFFF

# Execute batch
results = backend.execute_batch("and", [1,2,3], [4,5,6], bits=8)
# results = [0, 0, 2]

# Check available shapes
shapes = backend.available_shapes()
# ['xor', 'and', 'or', 'not', 'nand', 'nor', 'xnor', 'nop', ...]
```

### Explicit Backend Selection

```python
from trix.forge import CPUBackend, CUDABackend

# Force CPU
cpu = CPUBackend(bits=32)
result = cpu.execute("xor", 5, 3)

# Force CUDA
cuda = CUDABackend(bits=32)
result = cuda.execute("xor", 5, 3)
```

### Backend Interface

```python
class Backend(ABC):
    def info(self) -> BackendInfo: ...
    def available_shapes(self) -> List[str]: ...
    def execute(self, shape: str, a: int, b: int, bits: int) -> int: ...
    def execute_batch(self, shape: str, a_batch, b_batch, bits: int) -> List[int]: ...
    def compile_shapes(self, shapes: List[str], bits: int) -> bool: ...
```

---

## Foundry (Unified Interface)

The Foundry is a compiler: **Spec → IR (ShapeTerms) → Target (CUDA/Verilog)**.

This is the recommended interface for building deterministic systems.

### The Three Pillars

1. **Polynomial computation** - ShapeTerms is the IR
2. **Ternary signatures** - Derived from term structure, not sampled
3. **Routing-only learning** - Shapes are frozen geometry

### Quick Example

```python
from trix.forge import Foundry

# Create foundry
foundry = Foundry(bits=8)

# Register atomic shapes from truth functions
foundry.atom("xor", lambda a, b: a ^ b)
foundry.atom("and", lambda a, b: a & b)
foundry.atom("or",  lambda a, b: a | b)

# Build executable system
system = foundry.build()

# Validate (100% or fail)
result = system.validate(exhaustive=True)
assert result.all_passed()

# Execute
print(system.execute(42, 13, "xor"))  # 39

# Export
system.export_cuda("output/cuda/")
system.export_verilog("output/verilog/")
```

### Foundry API

```python
from trix.forge import Foundry

# Create with bit width
foundry = Foundry(bits=8)

# Register atoms (fluent API)
foundry.atom("xor", lambda a, b: a ^ b)
       .atom("and", lambda a, b: a & b)
       .atom("or",  lambda a, b: a | b)

# Register composites (see Composition Operators)
from trix.forge import par, sel
foundry.compose("xor_and", par("xor", "and"))
foundry.compose("alu", sel("xor", "and", "or"))

# Introspection
print(foundry.list_atoms())      # ['xor', 'and', 'or']
print(foundry.list_composites()) # ['xor_and', 'alu']
print(foundry.summary())

# Build executable system
system = foundry.build()
```

### System API

```python
# Execute single operation
result = system.execute(a, b, "xor")

# Execute batch
results = system.execute_batch([1,2,3], [4,5,6], "xor")

# Validate
validation = system.validate(exhaustive=True)
print(validation.all_passed())  # True
print(validation.summary())

# Export
system.export_cuda("cuda/")
system.export_verilog("verilog/")

# Introspection
print(system.list_operations())
print(system.summary())
sig = system.get_signature("xor")
```

### 6502 ALU Example

```python
foundry = Foundry(bits=8)

# 6502 operations
foundry.atom("AND", lambda a, b: a & b)
foundry.atom("ORA", lambda a, b: a | b)
foundry.atom("EOR", lambda a, b: a ^ b)

system = foundry.build()
result = system.validate(exhaustive=True)
print(f"6502 ALU: {len(system.shapes)} ops, validation: {result.all_passed()}")
```

---

## Composition Operators

Four operators for composing shapes into larger systems.

### seq(a, b) - Sequential

Output of A feeds input of B: `seq(A, B)(x) = B(A(x))`

```python
from trix.forge import Foundry, seq

foundry = Foundry(bits=8)
foundry.atom("double", lambda a, b: (a * 2) & 0xFF)
foundry.atom("inc", lambda a, b: (a + 1) & 0xFF)
foundry.compose("double_then_inc", seq("double", "inc"))
```

### par(a, b, ...) - Parallel

Same input, concatenated outputs: `par(A, B)(x) = [A(x), B(x)]`

```python
from trix.forge import Foundry, par

foundry = Foundry(bits=8)
foundry.atom("xor", lambda a, b: a ^ b)
foundry.atom("and", lambda a, b: a & b)
foundry.compose("xor_and", par("xor", "and"))  # 16-bit output
```

### sel(*shapes) - Selection

Opcode selects which shape executes (MUX):

```python
from trix.forge import Foundry, sel

foundry = Foundry(bits=8)
foundry.atom("xor", lambda a, b: a ^ b)
foundry.atom("and", lambda a, b: a & b)
foundry.atom("or",  lambda a, b: a | b)
foundry.compose("alu", sel("xor", "and", "or"))  # 3-way selection
```

### rep(shape, n) - Repetition

Chain shape N times: `rep(A, N)(x) = A(A(...A(x)...))`

```python
from trix.forge import Foundry, rep

foundry = Foundry(bits=1)
foundry.atom("half_adder", lambda a, b: (a ^ b, a & b))
foundry.compose("ripple_8", rep("half_adder", 8))
```

---

## Signature System

Ternary signatures derived from polynomial structure for content-addressable dispatch.

### Key Insight

Signatures are computed from **term structure**, not sampled from behavior.
The address IS the geometry.

### Signature Properties

- **Ternary values**: {-1, 0, +1}
- **Deterministic**: Same shape always produces same signature
- **Discriminative**: Different shapes produce different signatures
- **Dimension**: 64 by default (configurable)

### Usage

```python
from trix.forge import (
    signature_from_terms,
    SignatureTable,
    build_signature_table,
    generate_xor_shape,
)

# Derive signature from shape
shape = generate_xor_shape(bits=8)
sig = signature_from_terms(shape)
print(sig)  # tensor([-1, 0, 1, 1, ...])

# Build lookup table
from trix.forge import generate_all_shapes
shapes = generate_all_shapes(bits=8)
table = build_signature_table(shapes)

# Lookup
name, score = table.lookup(query_signature)
```

### SignatureTable

Content-addressable dispatch:

```python
from trix.forge import SignatureTable

table = SignatureTable(sig_dim=64)
table.add("xor", xor_shape)
table.add("and", and_shape)

# Single lookup
name, score = table.lookup(query_sig)

# Batch lookup
names, scores = table.lookup_batch(query_sigs)
```

---

## Chip DSL (Legacy)

Declarative chip specification with direct backend execution.

### Two Execution Modes

| Mode | Path | Use Case |
|------|------|----------|
| Neural Router | `compile()` → `train_router()` → `compute()` | Learning, differentiable |
| Direct Backend | `execute()` / `execute_batch()` | Production, exact |

### Quick Example

```python
from trix.forge import Chip

# Create 32-bit ALU
chip = Chip.alu(["xor", "and", "or", "not", "add", "sub"], bits=32)

# Direct execution - no training needed!
result = chip.execute(0xAAAAAAAA, 0x55555555, "xor")
# result = 0xFFFFFFFF

# Batch execution
results = chip.execute_batch([1, 2, 3], [4, 5, 6], "xor")
# results = [5, 7, 5]

# Benchmark throughput
stats = chip.benchmark("xor", n_samples=100000)
print(f"{stats['ops_per_sec']/1e6:.1f} M ops/sec on {stats['backend']}")
# 6.0 M ops/sec on CUDA
```

### Full API

```python
# Create chip manually
chip = Chip("my_alu", bits=8)
chip.input("a", 8).input("b", 8).input("op", 2)
chip.operation(0, "xor")
chip.operation(1, "and")
chip.operation(2, "or")
chip.output("result", 8)

# Or use factory
chip = Chip.alu(["xor", "and", "or"], bits=8)

# Set backend explicitly
chip.with_backend("cuda")  # or "cpu" or auto

# Execute by name
result = chip.execute(0xAA, 0x55, "xor")

# Execute by opcode
result = chip.execute(0xAA, 0x55, 0)  # opcode 0 = xor

# Batch
results = chip.execute_batch([...], [...], "and")

# Benchmark
stats = chip.benchmark("xor", n_samples=100000)
# {'operation': 'xor', 'samples': 100000, 'ops_per_sec': 6000000, 'backend': 'CUDA'}

# Summary
print(chip.summary())
# Chip: my_alu
#   Bits: 8
#   Operations: 3
#   Opcodes: ['xor', 'and', 'or']
#   Backend: CUDA
```

### Neural Router Mode (Alternative)

For differentiable routing (when you need gradients):

```python
chip = Chip.alu(["xor", "and", "or"], bits=8)
chip.compile()
chip.train_router(max_epochs=100)
accuracy = chip.validate()  # 100%
result = chip.compute(17, 38, "xor")  # Uses trained router
```

---

## API Reference

### Shape Generation

```python
from trix.forge import (
    generate_xor_shape,
    generate_and_shape,
    generate_all_shapes,
    SHAPE_GENERATORS,
)

# Generate single shape
xor = generate_xor_shape(bits=32)
print(xor.total_terms())  # 96

# Generate all shapes
shapes = generate_all_shapes(bits=32)
print(len(shapes))  # 15

# Available generators
print(SHAPE_GENERATORS.keys())
# {'xor', 'and', 'or', 'not', 'nand', 'nor', 'xnor', 'nop',
#  'add', 'sub', 'sll', 'srl', 'sra', 'slt', 'sltu'}
```

### Evaluation

```python
from trix.forge import compute_fast, compute_auto, evaluate_batch

# Fast path (direct computation)
result = compute_fast("xor", 0xAAAA, 0x5555, bits=16)
# result = 0xFFFF

# Auto selection (fast path if available)
result = compute_auto("add", 100, 200, bits=32)
# result = 300

# Batch evaluation
results = evaluate_batch_fast("xor", [1, 2, 3], [4, 5, 6], bits=8)
# results = [5, 7, 5]
```

### Validation

```python
from trix.forge.term import (
    validate_quick,
    validate_edge,
    validate_exhaustive_8bit,
    validate_multiwidth,
)

# Quick validation (1000 random samples)
result = validate_quick(shape, truth_fn, bits=32)
print(result.passed, result.accuracy)

# Edge case validation (119 cases)
result = validate_edge(shape, truth_fn, bits=32)

# Exhaustive 8-bit (65,536 cases)
result = validate_exhaustive_8bit(shape_8bit, truth_fn)

# Multi-width validation
results = validate_multiwidth("xor", lambda a,b: a^b)
# {8: ValidationResult, 16: ..., 32: ..., 64: ...}
```

### Hardware Estimation

```python
from trix.forge import (
    estimate_shape,
    estimate_xorpu,
    estimate_summary,
    HardwareEstimate,
)

# Single shape
est = estimate_shape(xor_shape)
print(est.luts, est.cycles, est.power_mw)
# 32, 1, 3.2

# Full XORPU
total = estimate_xorpu(shapes)
print(total.summary())
# "448 LUTs, 0 FFs, 1 cyc, 44.8 mW"

# Summary table
print(estimate_summary(shapes))
```

---

## Export Formats

### Verilog RTL

```python
from trix.forge import shape_to_verilog, export_verilog

# Single shape
verilog = shape_to_verilog(xor_shape)
print(verilog)

# Export all shapes
files = export_verilog(shapes, "rtl/")
# Creates: rtl/xorpu_xor.v, rtl/xorpu_and.v, ..., rtl/xorpu_top.v
```

**Output Example:**
```verilog
module xorpu_xor (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] result
);
    assign result[0] = a[0] ^ b[0];
    assign result[1] = a[1] ^ b[1];
    // ...
endmodule
```

### CUDA Kernels

```python
from trix.forge import shape_to_cuda, export_cuda

# Single shape
cuda = shape_to_cuda(xor_shape)

# Export all shapes
files = export_cuda(shapes, "cuda/")
# Creates: cuda/xorpu_kernels.cu, cuda/Makefile
```

**Output Example:**
```cuda
__global__ void xorpu_xor(uint32_t* a, uint32_t* b, uint32_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    result[idx] = a[idx] ^ b[idx];
}
```

### JSON Specification

```python
xorpu = XORPU()
spec = xorpu.export_terms("xorpu_spec.json")
```

**Output Format:**
```json
{
  "version": "1.0.0",
  "format_version": "1.0",
  "timestamp": "2025-12-22T00:00:00",
  "bits": 32,
  "shapes": {
    "xor": {
      "input_bits": 64,
      "output_bits": 32,
      "total_terms": 96,
      "bits": [
        {
          "bit": 0,
          "terms": [
            {"coeff": 1, "vars": [0]},
            {"coeff": 1, "vars": [32]},
            {"coeff": -2, "vars": [0, 32]}
          ]
        }
      ]
    }
  }
}
```

---

## Hardware Validation

### CUDA (NVIDIA Thor)

Validated on NVIDIA Thor GPU with CUDA 13.0.

```bash
cd /tmp/xorpu_cuda
make
./xorpu_test
```

**Results:**
```
XORPU CUDA Validation
=====================

Testing XOR:
  PASS: xor(0xAAAAAAAA, 0x55555555) = 0xFFFFFFFF [PASS]
  PASS: xor(0xFFFFFFFF, 0xFFFFFFFF) = 0x0 [PASS]
  PASS: xor(0x12345678, 0x0) = 0x12345678 [PASS]

Testing AND:
  PASS: and(0xAAAAAAAA, 0x55555555) = 0x0 [PASS]
  PASS: and(0xFFFFFFFF, 0xF0F0F0F) = 0xF0F0F0F [PASS]

Testing OR:
  PASS: or(0xAAAAAAAA, 0x55555555) = 0xFFFFFFFF [PASS]
  PASS: or(0x0, 0x12345678) = 0x12345678 [PASS]

Validation complete.
```

### FPGA (Pending)

Verilog ready for synthesis on:
- Lattice iCE40 (iCESugar-nano)
- Xilinx UltraScale+ (AWS F2)

```bash
# Generate Verilog
python -c "
from trix.forge import generate_all_shapes, export_verilog
shapes = generate_all_shapes(bits=32)
export_verilog(shapes, 'rtl/')
"

# Synthesize (iCE40)
cd rtl/
yosys -p "synth_ice40 -top xorpu_top -json xorpu.json" xorpu_*.v
nextpnr-ice40 --lp1k --json xorpu.json --asc xorpu.asc
icepack xorpu.asc xorpu.bin
```

---

## Performance

### Throughput (CUDA)

From `experiments/qupid/frozen.cu` on Thor:

| Benchmark | Throughput |
|-----------|------------|
| 512-bit cipher | 58.62 GB/s |
| 512-bit LFSR | 35.56 Tbits/sec |
| 512-bit permutation | 14.23 GB/s |

### Hardware Estimates (FPGA)

| Configuration | LUTs | Power |
|---------------|------|-------|
| XOR only (32-bit) | 32 | 3.2 mW |
| Full XORPU (32-bit) | 448 | 44.8 mW |
| Full XORPU (8-bit) | 112 | 11.2 mW |

### Scaling

| Width | Shapes | Total Terms | Est. LUTs |
|-------|--------|-------------|-----------|
| 8 | 15 | 234 | ~56 |
| 16 | 15 | 466 | ~112 |
| 32 | 15 | 930 | ~224 |
| 64 | 15 | 1,858 | ~448 |

Terms scale **linearly** with bit width.

---

## Quick Start

### Installation

```python
# From the trix package
from trix.forge import (
    # Shape generation
    generate_xor_shape,
    generate_all_shapes,

    # Computation
    compute_fast,
    evaluate_batch_fast,

    # Export
    shape_to_verilog,
    export_verilog,
    shape_to_cuda,
    export_cuda,

    # Hardware estimation
    estimate_shape,
    estimate_summary,
)
```

### Generate and Validate

```python
# Generate all shapes
shapes = generate_all_shapes(bits=32)

# Quick validation
from trix.forge.term import validate_quick
for name, shape in shapes.items():
    truth = {
        'xor': lambda a,b: a^b,
        'and': lambda a,b: a&b,
        'or': lambda a,b: a|b,
    }.get(name)
    if truth:
        result = validate_quick(shape, truth, bits=32)
        print(f"{name}: {result.accuracy*100:.0f}%")
```

### Export for Hardware

```python
# Verilog for FPGA
export_verilog(shapes, "rtl/")

# CUDA for GPU
export_cuda(shapes, "cuda/")

# Check estimates
print(estimate_summary(shapes))
```

---

## File Structure

```
trix/forge/
├── __init__.py      # All exports
├── term.py          # Explicit term representation
├── backend.py       # Hardware backend layer (CUDA/CPU)
├── foundry.py       # Unified Foundry interface (NEW)
├── system.py        # Executable System class (NEW)
├── composition.py   # seq/par/sel/rep operators (NEW)
├── signature.py     # Term-based signature derivation (NEW)
├── chip.py          # Chip DSL with execute()
├── verilog.py       # Verilog RTL generation
├── cuda.py          # CUDA kernel generation
├── hardware.py      # Resource estimation
└── xorpu_spec.py    # XORPU class and shapes

tests/
├── test_forge_term.py         # 143 tests
├── test_forge_verilog.py      # 43 tests
├── test_forge_hardware.py     # 39 tests
├── test_forge_backend.py      # 31 tests
├── test_forge_chip_backend.py # 26 tests
├── test_forge_foundry.py      # 43 tests (NEW)
├── test_forge_composition.py  # 24 tests (NEW)
├── test_forge_signature.py    # 23 tests (NEW)
└── ...

docs/
├── XORPU_COMPLETE.md       # This file
└── XORPU_PRODUCTION_ROADMAP.md

tmp/xorpu_production/
├── SESSION_LOG.md          # Development log
└── phase_*/                # Phase working dirs
```

---

## References

- Mesa 14: Frozen Shapes - Computation IS geometry
- Mesa 15: Learning IS Routing - 78× fewer parameters
- Mesa 16: XORPU - Geometry in silicon
- [Fungible Computation](https://github.com/anjaustin/fungible-computation)

---

## Appendix: Test Coverage

```
tests/test_forge_term.py         143 tests
tests/test_forge_verilog.py       43 tests
tests/test_forge_hardware.py      39 tests
tests/test_forge_backend.py       31 tests
tests/test_forge_chip_backend.py  26 tests
tests/test_forge_foundry.py       43 tests (NEW)
tests/test_forge_composition.py   24 tests (NEW)
tests/test_forge_signature.py     23 tests (NEW)
────────────────────────────────────────
Total:                           393 tests (all passing)
```

### Test Categories

- **Term tests**: Creation, evaluation, serialization
- **Shape tests**: All 15 shapes, all bit widths
- **Validation tests**: Quick, edge, exhaustive tiers
- **Cross-validation**: Terms vs tensor functions
- **Verilog tests**: Generation, export, testbench
- **CUDA tests**: Kernel generation, compilation
- **Hardware tests**: LUT/FF/power estimation
- **Backend tests**: CPU/CUDA execution, batch, registry
- **Chip backend tests**: execute(), execute_batch(), benchmark()
- **Foundry tests**: atom(), build(), execute(), validate(), export (NEW)
- **Composition tests**: seq, par, sel, rep operators (NEW)
- **Signature tests**: derivation, table, lookup, batch (NEW)

---

*Geometry in Motion.*

*The blade is sharpened. Ready for silicon.*
