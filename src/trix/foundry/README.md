# Hollywood Squares Foundry

**Native GPU-accelerated neural networks. No PyTorch. No TensorFlow. Pure CUDA.**

[![Tests](https://img.shields.io/badge/tests-12%20passed-brightgreen)]()
[![A/B Tests](https://img.shields.io/badge/A%2FB%20tests-4%2F4%20Native%20wins-brightgreen)]()
[![Speedup](https://img.shields.io/badge/training%20speedup-85x-blue)]()

## Overview

The Hollywood Squares Foundry is a complete machine learning system built from first principles using only CuPy and raw CUDA kernels. It implements the Hollywood Squares architecture: **position IS computation**.

### Key Results

| Metric | Native | PyTorch | Improvement |
|--------|--------|---------|-------------|
| Training Speed | 368K samples/sec | 4.3K samples/sec | **85×** |
| Inference Speed | 44.5M tokens/sec | 5.7M tokens/sec | **7.8×** |
| Final Loss | 0.000000 | 0.048459 | Native converges better |
| Memory Usage | 17 MB | 19 MB | 11% less |

## Quick Start

```python
from trix.foundry.native_training import NativeHollywoodSquares, Trainer
import cupy as cp

# Create model
model = NativeHollywoodSquares(
    d_model=128,
    num_tiles=16,
    lr=0.01,
)

# Generate data
train_x = cp.random.randn(10000, 128).astype(cp.float32)
train_y = train_x * 0.9  # Simple regression target

# Train
trainer = Trainer(model, loss_fn='mse')
for epoch in range(100):
    loss = trainer.train_step(train_x, train_y)
    print(f"Epoch {epoch}: loss = {loss:.6f}")

# Save
model.save("model.npz")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Native Training System                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Forward   │  │  Backward   │  │   Adam Optimizer    │  │
│  │   Kernel    │──│   Kernel    │──│   (Pure CuPy)       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      CuPy GPU Arrays                         │
├─────────────────────────────────────────────────────────────┤
│                      CUDA Runtime                            │
├─────────────────────────────────────────────────────────────┤
│                      NVIDIA GPU                              │
└─────────────────────────────────────────────────────────────┘
```

### Core Principle: Position IS Computation

Traditional neural networks compute `output = f(input)` through matrix multiplications.

Hollywood Squares computes `output = lookup[position(input)]` where:
- The position in memory determines which tile handles the input
- Each tile has frozen geometric structure (spline coefficients, directions)
- Values flow through the geometry like electrons through transistors

### Weight Compression

| Component | Original | Compressed | Ratio |
|-----------|----------|------------|-------|
| Signatures | float32 | 2-bit ternary | **16×** |
| Directions | float32 | INT8 | **4×** |
| Total routing | 16 KB | 2.5 KB | **6.4×** |

## Files

### Core Implementation

| File | Description | Lines |
|------|-------------|-------|
| `native_training.py` | Complete training system | ~400 |
| `hollywood_squares_emergence.py` | Optimized inference kernel | ~300 |
| `hollywood_squares_ffn_fast.py` | Shared memory kernel | ~250 |
| `hollywood_squares_ffn.py` | Basic GPU kernel | ~200 |

### Testing & Benchmarking

| File | Description |
|------|-------------|
| `test_hollywood_squares.py` | 12 rigorous tests |
| `ab_test_harness.py` | Native vs PyTorch A/B tests |
| `benchmark_harness.py` | Reproducible benchmarks |
| `benchmark_training.py` | Training speed comparison |

### Results & Logs

| File | Description |
|------|-------------|
| `ab_test_results.json` | Full A/B test data |
| `benchmark_results.json` | Inference benchmarks |
| `training_benchmark.json` | Training benchmarks |
| `test_results.json` | Test outputs |

## Components

### 1. Forward Kernel

The forward kernel processes input through:
1. **Routing**: Compute content scores (dot product with ternary signatures)
2. **Spatial weighting**: Apply B-spline kernel based on position
3. **Tile selection**: Pick winning tile (argmax)
4. **Spline lookup**: Interpolate output scale from coefficients
5. **Output**: Apply scaled direction with residual connection

```cuda
// Simplified forward logic
float content_score = dot(x, signature[tile]);
float spatial_score = cubic_bspline(position - tile_center);
float combined = content_score * spatial_score;
int best_tile = argmax(combined);
float scale = spline_lookup(x, best_tile);
output = x + scale * direction[best_tile];  // Residual
```

### 2. Backward Kernel

The backward kernel computes gradients for:
- **Directions**: `d_directions = d_output * scale`
- **Spline coefficients**: `d_coeffs = d_output * basis_values`
- **Input**: `d_input = d_output` (residual pass-through)

Uses atomic operations for gradient accumulation across batch.

### 3. Adam Optimizer

Pure CuPy implementation:

```python
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad**2
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)
param -= lr * m_hat / (sqrt(v_hat) + eps)
```

### 4. Loss Functions

- **MSE**: `loss = mean((pred - target)^2)`
- **Cross-Entropy**: `loss = -mean(log(softmax(logits)[targets]))`

Both return loss value and gradient.

## Benchmarks

### Training Speed (samples/sec)

```
Config                         Native          PyTorch         Speedup
----------------------------------------------------------------------
1000×10@64                     95,564          1,099           87×
10000×10@256                   312,328         4,387           71×
10000×50@256                   357,644         4,418           81×
```

### Inference Speed (tokens/sec)

```
Batch Size    Native (M/s)    PyTorch (M/s)    Speedup
------------------------------------------------------
1,000         5.66            0.08             74×
10,000        67.91           0.73             93×
100,000       47.94           4.06             12×
500,000       54.71           8.14             7×
```

### A/B Test Results

```
Test                 Native          PyTorch         Winner     Ratio
----------------------------------------------------------------------
Convergence          0.000000        0.048459        Native     ∞
Training Speed       367,767/s       4,333/s         Native     85×
Inference Speed      44.5M/s         5.7M/s          Native     7.8×
Memory Usage         17 MB           19 MB           Native     1.1×
----------------------------------------------------------------------
Native wins: 4/4
```

## Running Tests

```bash
# Run test suite
PYTHONPATH=src python -m pytest src/trix/foundry/test_hollywood_squares.py -v

# Run A/B tests
PYTHONPATH=src python src/trix/foundry/ab_test_harness.py

# Run benchmarks
PYTHONPATH=src python src/trix/foundry/benchmark_harness.py

# Run training demo
PYTHONPATH=src python src/trix/foundry/native_training.py
```

## Hardware Requirements

- **GPU**: NVIDIA with CUDA support (tested on Thor with 132 GB)
- **Compute Capability**: 7.0+ recommended
- **CUDA**: 11.0+ (tested on 13.0)
- **Dependencies**: CuPy, NumPy

## Known Limitations

See [GAPS.md](GAPS.md) for full details.

### Current Gaps

1. **Fixed dimensions**: d_model must be multiple of 16, max 128
2. **No state routing**: Temporal routing from v4 not ported
3. **Simplified compression**: Uses first two dims instead of learned network
4. **Single GPU only**: No distributed training

### Addressed

- ✅ Forward pass
- ✅ Backward pass
- ✅ Adam optimizer
- ✅ Save/Load
- ✅ Loss functions

## Theory

### Why Position-Based Computation Works

SparseLookupFFN routes each input to a specific tile based on content and position. This routing is:
- **Deterministic**: Same input → same tile
- **Parallel**: All routing decisions independent
- **Cache-friendly**: Tiles loaded once, reused many times

Unlike SHA-256's carry propagation (which is value-dependent), neural network routing has no such dependencies. This makes it ideal for position-based acceleration.

### Comparison to PyTorch

PyTorch provides general-purpose autograd through tape-based differentiation. This generality has overhead:
- Graph construction
- Memory allocation per operation
- Dynamic dispatch

For a **fixed architecture** like Hollywood Squares, we know the exact computation graph. We can:
- Fuse operations into single kernels
- Pre-allocate all memory
- Eliminate dispatch overhead

Result: 85× faster training.

## Citation

```
@software{hollywood_squares_foundry,
  title = {Hollywood Squares Foundry: Native GPU Neural Networks},
  author = {TriX Team},
  year = {2024},
  note = {No PyTorch. No TensorFlow. Pure CUDA.}
}
```

## License

See repository LICENSE file.
