# ONNX Shapes Reference

*Frozen shapes for the entire ONNX zoo*

> *"You hand me an ONNX model, I hand you back a 50 KB binary. That's the deal."*

---

## What's This About?

ONNX has ~150 operators. Neural networks use maybe 20 of them regularly. TRIXC provides frozen shape implementations for all the important ones.

**The key insight:** Every ONNX operator is just math. Math doesn't change. We freeze it.

---

## The Shape Map

### Tier 1: The Heavy Hitters

These shapes do 90% of the work in any neural network.

| ONNX Op | TRIXC Function | Formula | Notes |
|---------|----------------|---------|-------|
| MatMul | `trix_onnx_matmul` | C = A @ B | The workhorse |
| Gemm | `trix_onnx_gemm` | C = αAB + βbias | Linear layers |
| Add | `trix_onnx_add` | c = a + b | Element-wise |
| Mul | `trix_onnx_mul` | c = a * b | Element-wise |
| Relu | `trix_onnx_relu` | max(0, x) | Activation |
| Gelu | `trix_onnx_gelu` | x * σ(1.702x) | Transformer activation |
| Softmax | `trix_onnx_softmax` | exp(x)/Σexp | Attention weights |

### Tier 2: The Supporting Cast

| ONNX Op | TRIXC Function | Formula |
|---------|----------------|---------|
| Sub | `trix_onnx_sub` | c = a - b |
| Div | `trix_onnx_div` | c = a / b |
| Neg | `trix_onnx_neg` | c = -a |
| Abs | `trix_onnx_abs` | c = \|a\| |
| Sqrt | `trix_onnx_sqrt` | c = √a |
| Exp | `trix_onnx_exp` | c = eᵃ |
| Log | `trix_onnx_log` | c = ln(a) |
| Pow | `trix_onnx_pow` | c = aᵇ |

### Tier 3: Activations

| ONNX Op | TRIXC Function | Formula | Usage |
|---------|----------------|---------|-------|
| Relu | `trix_onnx_relu` | max(0, x) | Classic |
| Sigmoid | `trix_onnx_sigmoid` | 1/(1+e⁻ˣ) | Gates, binary |
| Tanh | `trix_onnx_tanh` | tanh(x) | RNNs |
| Gelu | `trix_onnx_gelu` | x·Φ(x) | Transformers |
| Gelu (exact) | `trix_onnx_gelu_exact` | x·0.5(1+erf(x/√2)) | When you need precision |
| SiLU/Swish | `trix_onnx_silu` | x·σ(x) | Modern nets |

### Tier 4: Normalization

| ONNX Op | TRIXC Function | Composition |
|---------|----------------|-------------|
| LayerNorm | `trix_onnx_layer_norm` | mean → sub → var → div → scale → shift |
| BatchNorm | `trix_onnx_batch_norm` | Uses running stats (inference) |
| RMSNorm | `trix_onnx_rms_norm` | rms → div → scale (no mean subtraction) |

### Tier 5: Reductions

| ONNX Op | TRIXC Function | Output |
|---------|----------------|--------|
| ReduceSum | `trix_onnx_reduce_sum` | Sum of elements |
| ReduceMean | `trix_onnx_reduce_mean` | Average |
| ReduceMax | `trix_onnx_reduce_max` | Maximum |
| ReduceMin | `trix_onnx_reduce_min` | Minimum |
| ReduceProd | `trix_onnx_reduce_prod` | Product |

### Tier 6: Shape Operations

| ONNX Op | TRIXC Function | What It Does |
|---------|----------------|--------------|
| Transpose | `trix_onnx_transpose_2d` | Swap axes |
| Concat (axis 0) | `trix_onnx_concat_axis0` | Vertical stack |
| Concat (axis 1) | `trix_onnx_concat_axis1` | Horizontal stack |
| Gather | `trix_onnx_gather` | Embedding lookup |

### Tier 7: Comparisons and Logic

| ONNX Op | TRIXC Function | Output |
|---------|----------------|--------|
| Equal | `trix_onnx_equal` | 1.0 if equal, else 0.0 |
| Greater | `trix_onnx_greater` | 1.0 if a > b |
| Less | `trix_onnx_less` | 1.0 if a < b |
| Where | `trix_onnx_where` | cond ? a : b |
| Clip | `trix_onnx_clip` | clamp(x, min, max) |

### Tier 8: The Crown Jewel

```c
/**
 * Scaled Dot-Product Attention
 *
 * attn = softmax(Q @ K^T / √d_k) @ V
 *
 * This is the operation that powers every transformer.
 * And it's a frozen shape.
 */
trix_onnx_attention(Q, K, V, output, seq_len, d_k, scale);
```

---

## Detailed API

### Matrix Operations

```c
/**
 * Matrix Multiply: C[M,N] = A[M,K] @ B[K,N]
 */
static inline void trix_onnx_matmul(
    const float* a,    // [M, K]
    const float* b,    // [K, N]
    float* c,          // [M, N]
    int M, int N, int K
);

/**
 * General Matrix Multiply: C = α·A@B + β·bias
 */
static inline void trix_onnx_gemm(
    const float* a,    // [M, K]
    const float* b,    // [K, N]
    const float* bias, // [N] - broadcast across rows
    float* c,          // [M, N]
    int M, int N, int K,
    float alpha,       // Scale for A@B
    float beta         // Scale for bias
);

/**
 * MatMul with transpose options
 */
static inline void trix_onnx_matmul_t(
    const float* a, const float* b, float* c,
    int M, int N, int K,
    int trans_a,       // Transpose A?
    int trans_b        // Transpose B?
);
```

### Activation Functions

```c
/**
 * ReLU: max(0, x)
 *
 * The simplest non-linearity. If it's negative, kill it.
 */
static inline float trix_onnx_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

/**
 * GELU: x * Φ(x) ≈ x * σ(1.702 * x)
 *
 * The activation that launched a thousand transformers.
 * This is the fast approximation used in BERT, GPT, etc.
 */
static inline float trix_onnx_gelu(float x) {
    return x * trix_onnx_sigmoid(1.702f * x);
}

/**
 * GELU (exact): x * 0.5 * (1 + erf(x / √2))
 *
 * The mathematically exact version. Slower but precise.
 */
static inline float trix_onnx_gelu_exact(float x) {
    return x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
}

/**
 * Sigmoid: 1 / (1 + exp(-x))
 *
 * Squashes anything to (0, 1). Classic.
 */
static inline float trix_onnx_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * Softmax: exp(x_i - max) / Σ exp(x_j - max)
 *
 * Turns logits into probabilities. Numerically stable version.
 */
static inline void trix_onnx_softmax(
    const float* x,    // Input logits
    float* out,        // Output probabilities
    int n              // Length
);
```

### Normalization

```c
/**
 * Layer Normalization: (x - μ) / σ * γ + β
 *
 * Per-sample normalization. The heart of every transformer layer.
 *
 * Composed from frozen shapes:
 * 1. mean = reduce_mean(x)
 * 2. centered = x - mean
 * 3. var = reduce_mean(centered²)
 * 4. normalized = centered / sqrt(var + eps)
 * 5. output = normalized * gamma + beta
 */
static inline void trix_onnx_layer_norm(
    const float* x,        // Input
    const float* gamma,    // Scale (learned, then frozen)
    const float* beta,     // Shift (learned, then frozen)
    float* out,            // Output
    int n,                 // Dimension
    float eps              // Numerical stability (typically 1e-5)
);

/**
 * RMS Normalization: x / RMS(x) * γ
 *
 * Simpler than LayerNorm. Used in LLaMA, PaLM, etc.
 * No mean subtraction, just scale by inverse RMS.
 */
static inline void trix_onnx_rms_norm(
    const float* x,
    const float* gamma,
    float* out,
    int n,
    float eps
);

/**
 * Batch Normalization (inference mode)
 *
 * Uses running statistics, not batch statistics.
 * The running_mean and running_var are frozen from training.
 */
static inline void trix_onnx_batch_norm(
    const float* x,
    const float* gamma,
    const float* beta,
    const float* running_mean,  // Frozen from training
    const float* running_var,   // Frozen from training
    float* out,
    int n,
    float eps
);
```

### Attention

```c
/**
 * Scaled Dot-Product Attention (single head)
 *
 * attn = softmax(Q @ K^T / scale) @ V
 *
 * This is the operation that changed everything.
 * And it's just three matrix multiplies and a softmax.
 * All frozen shapes.
 *
 * Memory: Uses alloca for attention scores (seq_len × seq_len)
 */
static inline void trix_onnx_attention(
    const float* Q,        // [seq_len, d_k] Queries
    const float* K,        // [seq_len, d_k] Keys
    const float* V,        // [seq_len, d_k] Values
    float* out,            // [seq_len, d_k] Output
    int seq_len,           // Sequence length
    int d_k,               // Key dimension
    float scale            // Usually 1/√d_k
);
```

---

## ONNX to TRIXC Mapping

The `onnx2trix.py` tool maps ONNX operators to TRIXC shapes:

```python
SHAPE_MAP = {
    # Arithmetic
    "Add": {"kind": "ADD"},
    "Sub": {"kind": "SUB"},
    "Mul": {"kind": "MUL"},
    "Div": {"kind": "DIV"},
    "MatMul": {"kind": "MATMUL"},
    "Gemm": {"kind": "GEMM"},

    # Activations
    "Relu": {"kind": "RELU"},
    "Sigmoid": {"kind": "SIGMOID"},
    "Tanh": {"kind": "TANH"},
    "Softmax": {"kind": "SOFTMAX"},
    "Gelu": {"kind": "GELU"},

    # Normalization (composed)
    "LayerNormalization": {
        "kind": "LAYER_NORM",
        "composed": True,
        "decomposition": ["REDUCE_MEAN", "SUB", "MUL",
                          "REDUCE_MEAN", "SQRT", "DIV", "MUL", "ADD"]
    },

    # ... etc
}
```

**Key insight:** Composed operations are expanded into their primitive shapes during compilation. LayerNorm isn't a black box - it's 8 frozen shapes chained together.

---

## Usage Example

```c
#include <trixc/onnx_shapes.h>

// A simple MLP forward pass
void mlp_forward(
    const float* x,      // [batch, 768]
    const float* W1,     // [768, 3072] - frozen from training
    const float* b1,     // [3072]
    const float* W2,     // [3072, 768]
    const float* b2,     // [768]
    float* output,       // [batch, 768]
    int batch
) {
    float hidden[3072];

    for (int b = 0; b < batch; b++) {
        // Layer 1: Linear + GELU
        trix_onnx_gemm(&x[b * 768], W1, b1, hidden,
                       1, 3072, 768, 1.0f, 1.0f);
        for (int i = 0; i < 3072; i++) {
            hidden[i] = trix_onnx_gelu(hidden[i]);
        }

        // Layer 2: Linear
        trix_onnx_gemm(hidden, W2, b2, &output[b * 768],
                       1, 768, 3072, 1.0f, 1.0f);
    }
}
```

---

## Why Frozen?

Every function in this file is:
1. **Deterministic** - Same input → same output, always
2. **Exact** - No approximation errors (within floating point)
3. **Portable** - Just C, runs anywhere
4. **Small** - Each function is ~10-50 lines

The shapes don't learn. They don't change. They just compute.

**What learns?** The weights. During training, you learn W1, W2, gamma, beta, etc. Then you freeze them and compile.

**What's frozen from the start?** The shapes. MatMul is always MatMul. GELU is always GELU. These are mathematical facts, not parameters.

---

## Supported ONNX Opset

Tested against ONNX opset version 13+. Most operations are backwards compatible.

### Not Yet Supported

| Op | Why |
|----|-----|
| Conv | Needs im2col + matmul lowering |
| MaxPool, AvgPool | Need specialized implementation |
| LSTM, GRU | Complex, need dedicated shapes |
| Resize | Interpolation is tricky |

These are on the roadmap. For now, if your model uses them, you'll get a helpful error from `onnx2trix.py`.

---

## The Principle

> *"ONNX defines what to compute. TRIXC defines how to compute it. The shapes are the bridge."*

Every ONNX model is a graph of operations. Every operation is a shape. Every shape compiles to C. The magic is in recognizing that neural network "learning" is just finding good weights for frozen shapes.

Once you have the weights, you have everything. Freeze them. Compile them. Run them in 50 KB instead of 500 MB.

---

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│   ONNX Runtime: "I'll interpret your model at runtime"      │
│                                                              │
│   TRIXC: "I'll compile your model at build time.            │
│           Then there's no model. Just math."                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```
