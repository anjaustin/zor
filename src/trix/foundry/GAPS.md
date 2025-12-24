# Hollywood Squares Foundry - Known Gaps and Limitations

This document tracks all known gaps, limitations, and areas for future work.
Last updated: 2024-12-21 (Post A/B Testing)

## A/B Test Results Summary

Native vs PyTorch comparison (all tests passed):

| Test | Native | PyTorch | Winner | Ratio |
|------|--------|---------|--------|-------|
| Convergence | 0.000000 | 0.048459 | Native | Converges better |
| Training Speed | 367,767/s | 4,333/s | Native | 85x |
| Inference Speed | 44.5M/s | 5.7M/s | Native | 7.8x |
| Memory Usage | 17 MB | 19 MB | Native | 11% less |
| **Overall** | | | **Native 4/4** | |

---

## Addressed Gaps (v1.0.0)

### 1. No Backward Pass / Gradient Support

**Status**: RESOLVED

**Solution**: Implemented in `native_training.py`:
- CUDA backward kernel with atomic gradient accumulation
- Gradients for `directions` and `spline_coeffs`
- Residual pass-through for input gradients

**Verification**: A/B tests confirm convergence (loss → 0.000000).

---

### 2. No API Documentation

**Status**: RESOLVED

**Solution**: Created comprehensive documentation:
- `API.md`: Complete API reference
- `QUICKSTART.md`: Getting started guide
- `ARCHITECTURE.md`: System design with diagrams
- `README.md`: Overview with benchmarks
- `CHANGELOG.md`: Version history

---

### 3. No PyTorch Comparison

**Status**: RESOLVED

**Solution**: Created A/B testing framework:
- `ab_test_harness.py`: 4 comprehensive tests
- `benchmark_training.py`: Training speed comparison
- `benchmark_harness.py`: Inference benchmarks

---

## Open Gaps

### 4. Fixed Dimension Constraints

**Status**: Architectural limitation

**Constraints**:
- `d_model` must be multiple of 16 (for ternary packing)
- `d_model` max 128 (shared memory limit)
- `num_tiles` max 32 (shared memory limit)

**Impact**: Cannot use with larger models without modification.

**Future work**:
- Multi-pass kernel for larger d_model
- Dynamic shared memory allocation

---

### 5. No State/Temporal Routing

**Status**: Not ported from SparseLookupFFNv4

**Impact**: Missing state-based routing feature from v4 (e.g., carry flag routing for 6502).

**Details**: SparseLookupFFNv4 has:
- `num_states` parameter
- `state_encoder` embedding
- `state_modulation` per-tile
- `temporal_scores` in routing

None of these are in the native kernel.

**Future work**: Port state routing to CUDA kernel.

---

### 6. Speedup Decreases at Large Batch Sizes

**Observed**:
- 93x speedup at 10K batch
- 7x speedup at 500K batch

**Cause**: Both implementations become memory-bandwidth limited at large batches.

**Impact**: Less relative benefit for large-batch inference.

**Potential fixes**:
- Tensor core utilization (not currently used)
- Half-precision (FP16) support
- Better memory coalescing

---

### 7. Spline Coefficients Not Quantized

**Status**: Still float32

**Impact**: Spline coefficients could be compressed further.

**Current compression**:
- Signatures: 16x compressed (ternary → 2-bit)
- Directions: 4x compressed (float32 → INT8)
- Splines: No compression

**Future work**: INT8 or INT16 spline coefficients.

---

### 8. No Multi-GPU Support

**Status**: Single GPU only

**Impact**: Cannot scale beyond single device.

**Future work**:
- Data parallelism across GPUs
- Model parallelism for larger tile counts

---

### 9. Simplified Compression Network

**Status**: Placeholder implementation

**Details**: Real SparseLookupFFNv4 uses:
```python
self.compress = nn.Sequential(
    nn.Linear(d_model, d_model // 4),
    nn.GELU(),
    nn.Linear(d_model // 4, 2),
    nn.Tanh(),
)
```

Native uses:
```cuda
float a = tanhf(x_local[0] * 0.1f);
float b = tanhf(x_local[1] * 0.1f);
```

This is NOT equivalent. It just uses first two input dimensions.

**Impact**: Spline lookups not properly routed based on learned compression.

**Future work**: Implement full compression network in CUDA.

---

### 10. No Numerical Equivalence Test vs PyTorch

**Status**: Not implemented

**Impact**: Cannot verify bit-exact equivalence with PyTorch.

**Current tests verify**:
- Output shape correct
- Output finite (no NaN/Inf)
- Deterministic
- Residual connection works

**Not verified**:
- Numerically equivalent to PyTorch with same weights

**Future work**:
- Export PyTorch weights to native format
- Compare outputs within FP tolerance

---

### 11. No Stress Tests

**Status**: Not implemented

**Future work**:
- Long-running stability tests
- Memory leak detection
- Edge case testing (empty batch, single element, etc.)

---

### 12. No Cross-Platform Testing

**Status**: Only tested on NVIDIA Thor (Jetson)

**Future work**:
- Test on consumer GPUs (RTX 3090, 4090)
- Test on data center GPUs (A100, H100)
- Document compute capability requirements

---

### 13. No ONNX/TensorRT Export

**Status**: Not implemented

**Impact**: Cannot deploy to standard production inference servers.

**Future work**: ONNX export or TensorRT plugin.

---

## Gap Tracking

| ID | Gap | Severity | Status |
|----|-----|----------|--------|
| 1 | No backward pass | Critical | **RESOLVED** |
| 2 | No API docs | Medium | **RESOLVED** |
| 3 | No PyTorch comparison | High | **RESOLVED** |
| 4 | Fixed dimensions | High | Open |
| 5 | No state routing | Medium | Open |
| 6 | Speedup scaling | Low | Known |
| 7 | Spline quantization | Low | Open |
| 8 | No multi-GPU | Medium | Open |
| 9 | Simplified compress | High | Open |
| 10 | No equivalence test | Medium | Open |
| 11 | No stress tests | Medium | Open |
| 12 | No cross-platform | Medium | Open |
| 13 | No ONNX export | Low | Open |

**Summary**: 3 gaps resolved, 10 gaps open

---

## Priority Roadmap

### Phase 1: Core Completeness
1. **Gap #9**: Implement full compression network in CUDA
2. **Gap #10**: Add numerical equivalence tests with weight export

### Phase 2: Production Readiness
3. **Gap #11**: Add stress tests and edge case handling
4. **Gap #12**: Cross-platform GPU testing

### Phase 3: Feature Parity
5. **Gap #5**: Port state/temporal routing from v4
6. **Gap #4**: Support larger d_model with multi-pass kernel

### Phase 4: Deployment
7. **Gap #8**: Multi-GPU support
8. **Gap #13**: ONNX/TensorRT export

---

## Test Coverage

Current test suite (`test_hollywood_squares.py`):

| Test | Coverage |
|------|----------|
| Ternary pack/unpack roundtrip | Compression correctness |
| Compression ratio verification | 16x compression |
| INT8 quantization error bounds | <1% error |
| Output shape validation | API correctness |
| Output finiteness | Numerical stability |
| Determinism verification | Reproducibility |
| Residual connection validation | Architecture correctness |
| Vectorized/standard equivalence | Kernel correctness |
| B-spline property verification | Math correctness |
| Throughput measurement | Performance |
| Memory usage tracking | Resource usage |
| PyTorch speedup comparison | Relative performance |

**All 12 tests passing.**

---

*This document should be updated whenever gaps are identified or fixed.*
