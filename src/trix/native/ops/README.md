# TriX Native Ops

**Self-hosted computation primitives. No external math libraries.**

## Overview

TriX Native Ops provides C implementations of core operations that don't depend on NumPy, BLAS, or any external math library. The key insight: **ternary weights eliminate multiplication**.

```
Traditional matmul:  y = W @ x       → N² multiplications
Ternary matmul:      y = Σ±x         → 0 multiplications (just routing)
```

## Building

```bash
cd src/trix/native/ops
make          # Build shared library
make test     # Run tests (30 tests)
make bench    # Run benchmarks
```

## Components

### 1. Frozen Shapes (Polynomial Form)

For training with gradients:

```c
float shape_xor(float a, float b);     // a + b - 2ab
float shape_and(float a, float b);     // ab
float shape_or(float a, float b);      // a + b - ab
float shape_not(float a);              // 1 - a
float shape_relu(float x);             // max(0, x)

void shape_half_adder(float a, float b, float* sum, float* carry);
void shape_full_adder(float a, float b, float cin, float* sum, float* carry);
```

### 2. Binary Frozen Shapes

For inference (maximum speed):

```c
uint8_t shape_xor_binary(uint8_t a, uint8_t b);   // a ^ b
uint8_t shape_and_binary(uint8_t a, uint8_t b);   // a & b
uint8_t shape_or_binary(uint8_t a, uint8_t b);    // a | b
uint8_t shape_not_binary(uint8_t a);              // ~a

// Vectorized (process 8 elements per byte)
void trix_apply_binary_xor(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t bytes);
void trix_apply_binary_and(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t bytes);
```

**Performance:**
- Binary XOR: 117 GB/s (2.8x faster than polynomial)
- Memory: 32x smaller (1 bit vs 32 bits per element)

### 3. Ternary Matrix Operations

Matrix multiply without multiplication:

```c
// Create ternary matrix from float weights (quantizes to {-1, 0, +1})
TernaryMatrix trix_ternary_from_float(const float* weights, size_t rows, size_t cols);

// Matrix-vector multiply (no actual multiplication!)
void trix_ternary_matvec(
    const TernaryMatrix* mat,
    const float* x,
    float* y,
    const float* scale
);

// Batched matrix multiply
void trix_ternary_matmul(
    const TernaryMatrix* mat,
    const float* X,
    float* Y,
    const float* scale,
    size_t batch
);
```

**How it works:**
```
y[i] = Σⱼ w[i,j] * x[j]

For ternary weights {-1, 0, +1}:
y[i] = Σ_{w=+1} x[j] - Σ_{w=-1} x[j]

Just additions and subtractions. Zero multiplies.
```

### 4. Tile Operations

Complete forward/backward for a TriX tile:

```c
TrixTile* trix_tile_create(size_t d_model, size_t d_hidden);
void trix_tile_forward(TrixTile* tile, const float* x, float* y, size_t batch);
void trix_tile_backward(TrixTile* tile, const float* d_y, float* d_x, size_t batch);
void trix_tile_free(TrixTile* tile);
```

### 5. Hamming Routing

Content-addressable routing via XOR + popcount:

```c
// Hamming distance between packed binary vectors
size_t trix_hamming_distance(const uint8_t* a, const uint8_t* b, size_t bytes);

// Route to closest signature
size_t trix_route_hamming(
    const uint8_t* input,
    const uint8_t* signatures,
    size_t num_sigs,
    size_t packed_dim
);

// Binarize float vector
void trix_binarize(const float* x, uint8_t* out, size_t dim);
```

## Python Bindings

```python
from trix.native.ops import TrixOps, relu, xor, and_, or_

ops = TrixOps()

# Frozen shapes
y = relu(x)              # max(0, x)
z = xor(a, b)            # a + b - 2ab

# Reductions
total = ops.sum(x)       # via adder tree
length = ops.norm(x)     # L2 norm

# Routing
binary = ops.binarize(x)
best = ops.route_hamming(binary, signatures)
```

## Benchmarks

```
Ternary MatVec [512×512]:     0.71 GOP/s
Ternary MatMul [32×512×512]:  0.74 GOP/s
Tile Forward [32×512×1024]:   0.71 GOP/s
Hamming Routing [256 sigs]:   0.84 M routes/sec

Binary vs Polynomial XOR (1M elements):
  Binary:     0.003 ms (117 GB/s)
  Polynomial: 0.009 ms
  Speedup:    2.8x
  Memory:     122 KB vs 3,906 KB (32x smaller)
```

## The Key Insight

```
┌─────────────────────────────────────────────────────────────┐
│  Polynomial form and binary form are THE SAME FUNCTION      │
│  on binary inputs {0, 1}.                                   │
│                                                             │
│  Training:   Use polynomial (gradients flow)                │
│  Inference:  Use binary (maximum speed)                     │
│                                                             │
│  Same truth. Different representations. Fungible.           │
└─────────────────────────────────────────────────────────────┘
```

## Files

```
trix_ops.h      # API header
trix_ops.c      # Implementation
test_ops.c      # Test suite (30 tests)
bench_ops.c     # Benchmarks
__init__.py     # Python bindings
Makefile        # Build system
libtrix_ops.so  # Compiled library
```
