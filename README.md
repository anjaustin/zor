# TriX

**Ternary neural computation. Self-hosted. No dependencies.**

```
Weights are {-1, 0, +1}. Shapes are frozen polynomials. Gradients are real.
```

---

## What Is TriX?

TriX is a neural architecture where:

1. **Weights are ternary** — `{-1, 0, +1}` only
2. **Shapes are frozen** — Pure polynomial math, no learning
3. **Routing is geometric** — Hamming distance, not learned gates
4. **Execution is self-hosted** — No PyTorch, no BLAS, no external math

The insight: **Ternary weights eliminate multiplication.**

```
y = W @ x  where W ∈ {-1, 0, +1}

becomes:

y = Σ(x where w=+1) - Σ(x where w=-1)
```

Just addition. Just routing.

---

## The Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRIX STACK                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LAYER              WHAT IT DOES                      SPEED                 │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  trix.nn            Training (PyTorch)                Gradient flow         │
│       │                                                                     │
│       ▼                                                                     │
│  trix.native        Inference (Pure C)                1.5 GOP/s             │
│       │             NEON (ARM) / AVX2 (x86)           2x SIMD speedup       │
│       │                                                                     │
│       ▼                                                                     │
│  Binary Shapes      Frozen inference                  117 GB/s              │
│       │             8 elements per byte               32x memory reduction  │
│       │                                                                     │
│       ▼                                                                     │
│  GILLIES Vulkan     GPU compute                       19 B ops/sec          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/your-org/trix.git
cd trix

# Build native ops (auto-detects NEON/AVX2)
cd src/trix/native/ops && make && cd -

# Install Python package
pip install -e .

# Verify
python -c "from trix.native.ops import TrixOps; print(TrixOps().simd)"
# → "NEON" on ARM, "AVX2" on x86, "scalar" otherwise
```

---

## Quick Start

### Training (Gradient Truth)

```python
from trix.nn import GradientTruthFFN

# Shapes are frozen. Only routing and scales learn.
ffn = GradientTruthFFN(d_model=512, num_shapes=64)

# Real gradients. No STE. No lies.
output = ffn(x)
loss = criterion(output, target)
loss.backward()  # Gradients flow through continuous params only
```

### Inference (Native Ops)

```python
from trix.native.ops import TrixOps

ops = TrixOps()
print(f"SIMD: {ops.simd}")  # NEON / AVX2 / scalar

# Ternary matmul - no multiplication
y = ops.ternary_matmul(W, x, scale)

# Hamming routing - XOR + popcount
winner = ops.route_hamming(query, signatures)
```

### Binary Frozen Shapes

```python
# After training, freeze to binary for maximum speed
from trix.native.ops import TrixOps

ops = TrixOps()

# Polynomial: a XOR b = a + b - 2ab (float32)
# Binary: a XOR b = a ^ b (bitwise, 32x faster)

a_bin = ops.binarize(a)
b_bin = ops.binarize(b)
result = ops.apply_binary_xor(a_bin, b_bin)  # 117 GB/s
```

---

## Core Concepts

### Gradient Truth

Traditional ternary networks use STE (Straight-Through Estimator):
- Forward: quantize to {-1, 0, +1}
- Backward: pretend quantization didn't happen

**STE is a lie.** The gradient is mathematically wrong.

Gradient Truth separates what learns from what doesn't:

| Component | Learns? | Gradients |
|-----------|---------|-----------|
| Shapes | NO | Frozen polynomials |
| Routing | YES | Real dot products |
| Scales | YES | Real multiplication |

**Only continuous things get gradients. Discrete things are discovered, then frozen.**

### Frozen Shapes

Shapes are pure polynomials:

```
XOR(a, b) = a + b - 2ab
AND(a, b) = ab
OR(a, b)  = a + b - ab
NOT(a)    = 1 - a
```

On binary inputs {0, 1}, these polynomials equal bitwise operations exactly.

After training:
1. Binarize inputs (threshold at 0.5)
2. Replace polynomial with bitwise op
3. 32x memory reduction, 2.8x speedup

### Ternary Matmul

```c
// Traditional: y = W @ x
for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++)
        y[i] += W[i][j] * x[j];  // Multiplication

// Ternary: y = route(W, x)
for (i = 0; i < rows; i++)
    y[i] = sum(x where W[i]=+1) - sum(x where W[i]=-1);  // No multiplication
```

Weights are stored as bitmasks: 2 bits per weight (pos/neg flags).

---

## Hardware Support

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PLATFORM SUPPORT                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PLATFORM             SIMD        GPU              STATUS                   │
│  ─────────────────────────────────────────────────────────────────────────  │
│  Jetson AGX Thor      NEON        GILLIES Vulkan   Primary target           │
│  Apple M1/M2/M3/M4    NEON        —                Tested                   │
│  Raspberry Pi 2+      NEON        —                Tested                   │
│  AMD Ryzen            AVX2        —                Tested                   │
│  Intel Core           AVX2        —                Supported                │
│  Raspberry Pi 1       Scalar      —                Fallback                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Benchmarks

### Native Ops (CPU)

| Operation | Size | Scalar | SIMD | Speedup |
|-----------|------|--------|------|---------|
| Ternary MatVec | 512² | 0.70 GOP/s | 1.48 GOP/s | 2.1x |
| Ternary MatMul | 32×512² | 0.70 GOP/s | 1.48 GOP/s | 2.1x |
| Binary XOR | 1M elements | 42 GB/s | 117 GB/s | 2.8x |

### GILLIES Vulkan (GPU)

| Metric | Value |
|--------|-------|
| Shape ops/sec | 19 billion |
| Bandwidth | 228 GB/s |
| Correctness | 0 errors / 33M tests |

---

## Project Structure

```
trix/
├── src/trix/
│   ├── nn/                    # Neural network modules
│   │   ├── gradient_truth.py  # Gradient Truth FFN
│   │   ├── hierarchical.py    # Hierarchical routing
│   │   └── ...
│   │
│   ├── native/                # Self-hosted operations
│   │   ├── ops/               # C library + Python bindings
│   │   │   ├── trix_ops.c     # NEON + AVX2 implementations
│   │   │   ├── trix_ops.h     # API header
│   │   │   └── __init__.py    # ctypes bindings
│   │   │
│   │   ├── vulkan/            # GILLIES GPU runtime
│   │   │   └── gillies.py     # Vulkan compute interface
│   │   │
│   │   └── programmable_tile.py  # Native training observer
│   │
│   └── kernel/                # Legacy NEON kernel (superseded)
│
├── tests/                     # 177+ tests
├── docs/                      # Documentation
└── examples/                  # Usage examples
```

---

## Self-Sufficiency

TriX computes itself. The only external dependencies are:

| Dependency | Purpose | Replaceable? |
|------------|---------|--------------|
| Python | Orchestration | Yes (C API exists) |
| NumPy/CuPy | Array containers | Yes (just memory) |
| GCC | Compile native ops | One-time |

**No BLAS. No cuDNN. No external math.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  BEFORE                           AFTER                                     │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│  PyTorch                          trix.native                               │
│    └── cuDNN                        └── libtrix_ops.so (50 KB)              │
│    └── cuBLAS                                                               │
│    └── NCCL                                                                 │
│    └── MKL                                                                  │
│    └── ... (hundreds of MB)                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Testing

### Quick Start

```bash
# Run all Python tests
PYTHONPATH=src pytest tests/ -v

# Run C tests
cd src/trix/native/ops && make test

# Run all tests
make test
```

### Test Categories

| Test File | Tests | What It Verifies |
|-----------|-------|------------------|
| `test_octave.py` | 31 | TrueOctaveFFN basic functionality |
| `test_octave_rigorous.py` | 45 | Mathematical invariants, edge cases |
| `test_multiscale.py` | 21 | MultiScaleTriXFFN (scaffold) |
| `test_hierarchical.py` | 32 | HierarchicalTriXFFN with Gradient Truth |
| `test_sparse_lookup.py` | 24 | SparseLookupFFN MatMul-free path |

### Rigorous Test Suite

The `test_octave_rigorous.py` suite verifies 10 categories of invariants:

```
1. Derivation Invariants     coarse = sign(pool(fine))
2. Ternary Invariants        all weights ∈ {-1, 0, +1}
3. Frozen Invariants         structure unchanged by training
4. Mode Invariants           deterministic=exact, generative=soft
5. Gradient Invariants       flow only to learned parameters
6. Numerical Stability       no NaN, no Inf, bounded outputs
7. Edge Cases                batch=1, seq=1, zeros, large inputs
8. Reproducibility           same seed → same output
9. Training Correctness      loss decreases, params update
10. Block Integration        stacking, gradient flow
```

Run the rigorous tests:

```bash
PYTHONPATH=src pytest tests/test_octave_rigorous.py -v
```

See [TESTING.md](docs/TESTING.md) for detailed documentation.

---

## Documentation

| Document | Description |
|----------|-------------|
| [THE_WAY.md](docs/THE_WAY.md) | Philosophy: Shape IS Compute |
| [GRADIENT_TRUTH.md](docs/GRADIENT_TRUTH.md) | No STE. Real gradients. |
| [TRUE_OCTAVE.md](docs/TRUE_OCTAVE.md) | Multi-resolution: exact + fuzzy |
| [TESTING.md](docs/TESTING.md) | Test suite guide |
| [LINCOLN_MANIFOLD_METHOD.md](docs/LINCOLN_MANIFOLD_METHOD.md) | How we think |
| [EXECUTION_STACK.md](docs/EXECUTION_STACK.md) | HSOS → GILLIES → Vulkan |
| [THEORY.md](docs/THEORY.md) | Mathematical foundations |

---

## The Principle

```
Train shapes with gradients.
Freeze shapes as polynomials.
Execute shapes as geometry.

Shape IS compute.
```

---

## License

MIT
