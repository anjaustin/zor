# Supported ONNX Operations

**Complete compatibility matrix for TRIXC C code generation.**

---

## Summary

| Category | Supported | Partial | Planned |
|----------|-----------|---------|---------|
| Activations | 7 | 1 | 0 |
| Arithmetic | 10 | 0 | 1 |
| Matrix | 2 | 0 | 0 |
| Normalization | 2 | 1 | 0 |
| Reduction | 4 | 1 | 0 |
| Shape | 6 | 2 | 2 |
| Comparison | 4 | 0 | 0 |
| **Total** | **35** | **5** | **3** |

---

## Activations

| ONNX Op | TRIXC Shape | Status | Notes |
|---------|-------------|--------|-------|
| `Relu` | `trix_onnx_relu` | ✅ Full | `max(0, x)` |
| `Gelu` | `trix_onnx_gelu` | ✅ Full | `x * sigmoid(1.702x)` approximation |
| `Sigmoid` | `trix_onnx_sigmoid` | ✅ Full | `1 / (1 + exp(-x))` |
| `Tanh` | `trix_onnx_tanh` | ✅ Full | Standard tanh |
| `Softmax` | `trix_onnx_softmax` | ✅ Full | Last axis, numerically stable |
| `Silu` | `trix_onnx_silu` | ✅ Full | `x * sigmoid(x)` (Swish) |
| `LeakyRelu` | `trix_onnx_leaky_relu` | ✅ Full | `max(alpha*x, x)` |
| `Elu` | — | ⚠️ Planned | — |

### Generated Code Examples

```c
// ReLU
for (int i = 0; i < n; i++) out[i] = trix_onnx_relu(in[i]);

// GELU
for (int i = 0; i < n; i++) out[i] = trix_onnx_gelu(in[i]);

// Softmax
trix_onnx_softmax(in, out, n);
```

---

## Arithmetic (Element-wise)

| ONNX Op | TRIXC Shape | Status | Notes |
|---------|-------------|--------|-------|
| `Add` | `trix_onnx_add` | ✅ Full | Element-wise addition |
| `Sub` | `trix_onnx_sub` | ✅ Full | Element-wise subtraction |
| `Mul` | `trix_onnx_mul` | ✅ Full | Element-wise multiplication |
| `Div` | `trix_onnx_div` | ✅ Full | Element-wise division |
| `Neg` | `trix_onnx_neg` | ✅ Full | Negation |
| `Abs` | `fabsf` | ✅ Full | Absolute value |
| `Sqrt` | `sqrtf` | ✅ Full | Square root |
| `Exp` | `expf` | ✅ Full | Exponential |
| `Log` | `logf` | ✅ Full | Natural logarithm |
| `Pow` | `trix_onnx_pow` | ✅ Full | Element-wise power |
| `Mod` | — | ⚠️ Planned | — |

### Broadcasting

**Not supported in v1.** Both operands must have identical shapes.

```
[1, 768] + [768]     ❌ Broadcasting required
[1, 768] + [1, 768]  ✅ Shapes match
```

Workaround: Use explicit `Expand` op in your ONNX model.

---

## Matrix Operations

| ONNX Op | TRIXC Shape | Status | Notes |
|---------|-------------|--------|-------|
| `MatMul` | `trix_onnx_matmul` | ✅ Full | `C = A @ B` |
| `Gemm` | `trix_onnx_gemm` | ✅ Full | `C = alpha*A@B + beta*bias` |

### Dimension Handling

```c
// MatMul: [M, K] @ [K, N] -> [M, N]
trix_onnx_matmul(A, B, C, M, N, K);

// Gemm with bias
trix_onnx_gemm(A, B, bias, C, M, N, K, alpha, beta);
```

### Transpose Attributes

For `Gemm` with `transA` or `transB`, use `trix_onnx_matmul_t`:

```c
trix_onnx_matmul_t(A, B, C, M, N, K, trans_a, trans_b);
```

---

## Normalization

| ONNX Op | TRIXC Shape | Status | Notes |
|---------|-------------|--------|-------|
| `LayerNormalization` | `trix_onnx_layer_norm` | ✅ Full | Affine transform |
| `BatchNormalization` | — | ⚠️ Partial | Inference only, fused |
| `RMSNorm` | `trix_onnx_rms_norm` | ✅ Full | Root mean square norm |

### LayerNorm

```c
// LayerNorm with gamma and beta
trix_onnx_layer_norm(input, gamma, beta, output, n, epsilon);
```

### BatchNorm

BatchNorm is decomposed at conversion time:
```
BatchNorm → (x - mean) / sqrt(var + eps) * gamma + beta
```

---

## Reduction

| ONNX Op | TRIXC Shape | Status | Notes |
|---------|-------------|--------|-------|
| `ReduceSum` | `trix_onnx_reduce_sum` | ✅ Full | Sum reduction |
| `ReduceMean` | `trix_onnx_reduce_mean` | ✅ Full | Mean reduction |
| `ReduceMax` | `trix_onnx_reduce_max` | ✅ Full | Max reduction |
| `ReduceMin` | `trix_onnx_reduce_min` | ✅ Full | Min reduction |
| `ReduceProd` | — | ⚠️ Partial | — |

### Axis Handling

Currently reduces over **all elements**. Axis-specific reduction is planned.

```c
float sum = trix_onnx_reduce_sum(input, n);
float mean = trix_onnx_reduce_mean(input, n);
```

---

## Shape Operations

| ONNX Op | TRIXC Shape | Status | Notes |
|---------|-------------|--------|-------|
| `Reshape` | `memcpy` | ✅ Full | Logical reshape (copy) |
| `Transpose` | `trix_onnx_transpose_2d` | ⚠️ 2D only | Higher dims planned |
| `Identity` | `memcpy` | ✅ Full | Passthrough |
| `Dropout` | `memcpy` | ✅ Full | No-op at inference |
| `Flatten` | `memcpy` | ✅ Full | Logical flatten |
| `Squeeze` | `memcpy` | ✅ Full | Remove size-1 dims |
| `Unsqueeze` | `memcpy` | ✅ Full | Add size-1 dims |
| `Concat` | — | ⚠️ Planned | — |
| `Split` | — | ⚠️ Planned | — |
| `Gather` | — | ⚠️ Partial | — |

### Reshape/Flatten

These are logical operations - the data layout doesn't change:

```c
// Reshape is just a copy (same memory layout)
memcpy(output, input, size * sizeof(float));
```

---

## Comparison

| ONNX Op | TRIXC Shape | Status | Notes |
|---------|-------------|--------|-------|
| `Equal` | Element-wise | ✅ Full | Returns 0.0 or 1.0 |
| `Greater` | Element-wise | ✅ Full | Returns 0.0 or 1.0 |
| `Less` | Element-wise | ✅ Full | Returns 0.0 or 1.0 |
| `Where` | Element-wise | ✅ Full | Conditional select |
| `Clip` | Element-wise | ✅ Full | Clamp to range |

---

## Attention

| ONNX Op | TRIXC Shape | Status | Notes |
|---------|-------------|--------|-------|
| `Attention` | `trix_onnx_attention` | ✅ Full | Full self-attention |

```c
// Self-attention
trix_onnx_attention(Q, K, V, output, seq_len, d_k, scale);
```

---

## Convolution

| ONNX Op | TRIXC Shape | Status | Notes |
|---------|-------------|--------|-------|
| `Conv` | — | ⚠️ Partial | Decomposed to im2col + matmul |
| `ConvTranspose` | — | ⚠️ Planned | — |

Conv is a composed operation:
```
Conv → im2col → MatMul → Add (bias)
```

---

## Pooling

| ONNX Op | TRIXC Shape | Status | Notes |
|---------|-------------|--------|-------|
| `GlobalAveragePool` | `trix_onnx_global_avg_pool` | ✅ Full | Global average |
| `GlobalMaxPool` | `trix_onnx_global_max_pool` | ✅ Full | Global max |
| `MaxPool` | — | ⚠️ Planned | — |
| `AveragePool` | — | ⚠️ Planned | — |

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ Full | Fully implemented and tested |
| ⚠️ Partial | Works but with limitations |
| ⚠️ Planned | On the roadmap |
| ❌ | Not supported |

---

## Adding New Operations

To add a new operation:

1. **Add frozen shape to `onnx_shapes.h`**:
```c
static inline void trix_onnx_myop(const float* in, float* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = /* your math here */;
    }
}
```

2. **Add mapping in `onnx2trix.py`**:
```python
SHAPE_MAP = {
    ...
    "MyOp": {"kind": "MYOP", "composed": False},
}
```

3. **Add emission template**:
```python
if kind == "MYOP":
    return f"    trix_onnx_myop({in_c}, {out_c}, {size});"
```

4. **Add tests**.

---

## Model Compatibility

| Model Type | Compatible | Notes |
|------------|------------|-------|
| MLP | ✅ Yes | Fully supported |
| CNN | ⚠️ Partial | Conv decomposition needed |
| Transformer (encoder) | ✅ Yes | All ops supported |
| Transformer (decoder) | ⚠️ Partial | Attention mask handling |
| ResNet | ⚠️ Partial | Residual connections work |
| BERT | ✅ Yes | Full support |
| GPT-2 | ⚠️ Partial | Large models need external weights |

---

*Shapes are frozen. Operations are exact. Math doesn't approximate.*
