# Changelog

All notable changes to the Hollywood Squares Foundry.

## [1.0.0] - 2024-12-21

### The Emergence Release

Complete native GPU training system. No PyTorch dependency.

### Added

#### Core Training System
- `native_training.py`: Complete training system (~400 lines)
  - `NativeHollywoodSquares`: Trainable model class
  - `Trainer`: Training loop with loss computation
  - `AdamOptimizer`: Pure CuPy Adam implementation
  - `mse_loss()`: Mean squared error with gradients
  - `cross_entropy_loss()`: Classification loss with gradients
  - Forward CUDA kernel with intermediate caching
  - Backward CUDA kernel with atomic gradient accumulation
  - Model save/load via NumPy npz format

#### Inference Kernels
- `hollywood_squares_emergence.py`: Optimized inference
  - Ternary signature packing (16× compression)
  - INT8 direction quantization (4× compression)
  - Shared memory tile caching
  - Vectorized memory access (float4)
  - Peak: 67.91M tokens/sec

- `hollywood_squares_ffn_fast.py`: Shared memory variant
  - Cooperative tile loading
  - Block-level parallelism
  - Peak: 42M tokens/sec

- `hollywood_squares_ffn.py`: Basic GPU implementation
  - Reference implementation
  - Per-thread tile evaluation

#### Testing
- `test_hollywood_squares.py`: 12 rigorous tests
  - Ternary pack/unpack roundtrip
  - Compression ratio verification (16×)
  - INT8 quantization error bounds
  - Output shape validation
  - Output finiteness (no NaN/Inf)
  - Determinism verification
  - Residual connection validation
  - Vectorized/standard kernel equivalence
  - B-spline property verification
  - Throughput measurement
  - Memory usage tracking
  - PyTorch speedup comparison

- `ab_test_harness.py`: A/B testing framework
  - Convergence comparison
  - Training speed comparison
  - Inference speed comparison
  - Memory usage comparison
  - JSON result logging

#### Benchmarking
- `benchmark_harness.py`: Reproducible benchmarks
  - Hardware info collection
  - Software version logging
  - Multi-configuration testing
  - JSON output with timestamps

- `benchmark_training.py`: Training speed comparison
  - Native vs PyTorch
  - Multiple batch/epoch configurations

#### Documentation
- `README.md`: Comprehensive documentation
- `GAPS.md`: Known limitations tracking
- `CHANGELOG.md`: This file

### Performance Results

#### Training Speed
| Config | Native | PyTorch | Speedup |
|--------|--------|---------|---------|
| 1K×10@64 | 95K/s | 1.1K/s | 87× |
| 10K×10@256 | 312K/s | 4.4K/s | 71× |
| 10K×50@256 | 358K/s | 4.4K/s | 81× |

#### Inference Speed
| Batch | Native | PyTorch | Speedup |
|-------|--------|---------|---------|
| 1K | 5.7M/s | 0.08M/s | 74× |
| 10K | 67.9M/s | 0.73M/s | 93× |
| 100K | 47.9M/s | 4.1M/s | 12× |

#### A/B Test Summary
- **Convergence**: Native wins (loss 0.000 vs 0.048)
- **Training Speed**: Native wins (85×)
- **Inference Speed**: Native wins (7.8×)
- **Memory Usage**: Native wins (11% less)
- **Overall**: Native 4/4

### Architecture Decisions

1. **No Framework Dependency**: Built on CuPy + raw CUDA only
2. **Ternary Signatures**: {-1, 0, +1} packed as 2 bits (16× compression)
3. **INT8 Directions**: Quantized with <1% error (4× compression)
4. **Shared Memory Caching**: Tiles loaded once per threadblock
5. **Atomic Gradients**: Thread-safe accumulation across batch
6. **Pure CuPy Adam**: No external optimizer dependency

### Known Limitations

See GAPS.md for full details.

1. Fixed d_model (multiple of 16, max 128)
2. No state/temporal routing
3. Simplified compression network
4. Single GPU only

---

## [0.3.0] - 2024-12-21

### Hollywood Squares FFN

GPU-accelerated SparseLookupFFN inference.

### Added
- `hollywood_squares_ffn.py`: Basic GPU kernel
- `hollywood_squares_ffn_fast.py`: Shared memory optimization
- Initial benchmarks showing 42× speedup

---

## [0.2.0] - 2024-12-21

### The Emergence

Optimized inference with weight compression.

### Added
- `hollywood_squares_emergence.py`: Full compression pipeline
- Ternary packing (16× compression)
- INT8 quantization (4× compression)
- Peak 93× speedup at small batches

---

## [0.1.0] - 2024-12-21

### SHA-256 Experiments

Initial GPU fabric experiments with SHA-256.

### Added
- `hollywood_squares_gpu.py`: SHA-256 CUDA kernel
- `hollywood_squares_jit.py`: Numba CPU baseline
- `hollywood_squares_tuned.py`: Thread configuration tuning
- `hollywood_squares_bitslice.py`: Bit-sliced experiment
- `hollywood_squares_position.py`: Position-based addition experiment

### Learned
- SHA-256 carry propagation cannot be parallelized (value-dependent)
- GPU native ALU beats lookup tables for arithmetic
- Neural network routing IS parallelizable (no carry equivalent)

---

## Development Notes

### Why Native?

PyTorch provides generality at the cost of:
- Graph construction overhead
- Memory allocation per operation
- Dynamic dispatch latency

For a **fixed architecture** like Hollywood Squares:
- We know the exact computation graph
- We can pre-allocate all memory
- We can fuse operations into single kernels

Result: 85× faster training.

### The Hollywood Squares Principle

**Position IS computation.**

Instead of computing `f(x)`, we navigate to `lookup[position(x)]`.

For neural networks, this means:
- Routing is position-based (content → tile index)
- Tile parameters are frozen geometry
- Values flow through structure like electrons

This works because routing has no value-dependent dependencies (unlike SHA-256's carries).
