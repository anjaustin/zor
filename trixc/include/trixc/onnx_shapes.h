/*
 * TRIXC ONNX Shapes
 *
 * Frozen shapes for ONNX-compatible operations.
 * These map directly to ONNX operator semantics.
 *
 * "ONNX is notation. Shapes are truth. C is execution."
 *
 * Created by: Tripp + Claude
 * Date: December 2025
 */

#ifndef TRIXC_ONNX_SHAPES_H
#define TRIXC_ONNX_SHAPES_H

#include "apu.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <alloca.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Activation Shapes (0 parameters - frozen)
 *
 * These are mathematical truths, not learned functions.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * ReLU: max(0, x)
 * The simplest non-linearity. Frozen.
 */
static inline float trix_onnx_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

/**
 * Sigmoid: 1 / (1 + exp(-x))
 * Squashes to (0, 1). Frozen.
 */
static inline float trix_onnx_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 * Squashes to (-1, 1). Frozen.
 */
static inline float trix_onnx_tanh(float x) {
    return tanhf(x);
}

/**
 * GELU: x * Φ(x) ≈ x * sigmoid(1.702 * x)
 * Gaussian Error Linear Unit. Used in BERT, GPT. Frozen.
 */
static inline float trix_onnx_gelu(float x) {
    return x * trix_onnx_sigmoid(1.702f * x);
}

/**
 * GELU (exact): x * 0.5 * (1 + erf(x / sqrt(2)))
 * More accurate but slower. Frozen.
 */
static inline float trix_onnx_gelu_exact(float x) {
    return x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
}

/**
 * SiLU / Swish: x * sigmoid(x)
 * Self-gated activation. Frozen.
 */
static inline float trix_onnx_silu(float x) {
    return x * trix_onnx_sigmoid(x);
}

/**
 * Softmax: exp(x_i) / sum(exp(x_j))
 * Converts logits to probabilities. Frozen.
 */
static inline void trix_onnx_softmax(const float* x, float* out, int n) {
    /* Numerical stability: subtract max */
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Compute exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        out[i] = expf(x[i] - max_val);
        sum += out[i];
    }

    /* Normalize */
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++) {
        out[i] *= inv_sum;
    }
}

/**
 * Log-Softmax: log(softmax(x))
 * More numerically stable for cross-entropy. Frozen.
 */
static inline void trix_onnx_log_softmax(const float* x, float* out, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += expf(x[i] - max_val);
    }

    float log_sum = logf(sum) + max_val;
    for (int i = 0; i < n; i++) {
        out[i] = x[i] - log_sum;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Element-wise Arithmetic Shapes
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline void trix_onnx_add(
    const float* a, const float* b, float* out, int n
) {
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

static inline void trix_onnx_sub(
    const float* a, const float* b, float* out, int n
) {
    for (int i = 0; i < n; i++) out[i] = a[i] - b[i];
}

static inline void trix_onnx_mul(
    const float* a, const float* b, float* out, int n
) {
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
}

static inline void trix_onnx_div(
    const float* a, const float* b, float* out, int n
) {
    for (int i = 0; i < n; i++) out[i] = a[i] / b[i];
}

static inline void trix_onnx_neg(const float* a, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = -a[i];
}

static inline void trix_onnx_abs(const float* a, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = fabsf(a[i]);
}

static inline void trix_onnx_sqrt(const float* a, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = sqrtf(a[i]);
}

static inline void trix_onnx_exp(const float* a, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = expf(a[i]);
}

static inline void trix_onnx_log(const float* a, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = logf(a[i]);
}

static inline void trix_onnx_pow(
    const float* a, const float* b, float* out, int n
) {
    for (int i = 0; i < n; i++) out[i] = powf(a[i], b[i]);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Matrix Shapes
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * MatMul: C[M,N] = A[M,K] @ B[K,N]
 * Dense matrix multiplication. The workhorse of neural networks.
 */
static inline void trix_onnx_matmul(
    const float* a, const float* b, float* c,
    int M, int N, int K
) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

/**
 * Gemm: C = alpha * A @ B + beta * bias
 * General matrix multiply with bias. Common in linear layers.
 */
static inline void trix_onnx_gemm(
    const float* a, const float* b, const float* bias, float* c,
    int M, int N, int K,
    float alpha, float beta
) {
    trix_onnx_matmul(a, b, c, M, N, K);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            c[i * N + j] = alpha * c[i * N + j] + beta * bias[j];
        }
    }
}

/**
 * MatMul with transpose options
 */
static inline void trix_onnx_matmul_t(
    const float* a, const float* b, float* c,
    int M, int N, int K,
    int trans_a, int trans_b
) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_val = trans_a ? a[k * M + i] : a[i * K + k];
                float b_val = trans_b ? b[j * K + k] : b[k * N + j];
                sum += a_val * b_val;
            }
            c[i * N + j] = sum;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Reduction Shapes
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline float trix_onnx_reduce_sum(const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i];
    return sum;
}

static inline float trix_onnx_reduce_mean(const float* x, int n) {
    return trix_onnx_reduce_sum(x, n) / (float)n;
}

static inline float trix_onnx_reduce_max(const float* x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > m) m = x[i];
    }
    return m;
}

static inline float trix_onnx_reduce_min(const float* x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < m) m = x[i];
    }
    return m;
}

static inline float trix_onnx_reduce_prod(const float* x, int n) {
    float prod = 1.0f;
    for (int i = 0; i < n; i++) prod *= x[i];
    return prod;
}

/* Reduce along axis (for 2D tensors) */
static inline void trix_onnx_reduce_sum_axis(
    const float* x, float* out,
    int rows, int cols, int axis
) {
    if (axis == 0) {
        /* Sum across rows, output shape: [cols] */
        for (int j = 0; j < cols; j++) {
            float sum = 0.0f;
            for (int i = 0; i < rows; i++) {
                sum += x[i * cols + j];
            }
            out[j] = sum;
        }
    } else {
        /* Sum across cols, output shape: [rows] */
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) {
                sum += x[i * cols + j];
            }
            out[i] = sum;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Normalization Shapes (Composed from primitives)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
 * Per-sample normalization. Common in transformers.
 */
static inline void trix_onnx_layer_norm(
    const float* x, const float* gamma, const float* beta,
    float* out, int n, float eps
) {
    /* Compute mean */
    float mean = trix_onnx_reduce_mean(x, n);

    /* Compute variance */
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)n;

    /* Normalize and scale */
    float inv_std = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < n; i++) {
        out[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
    }
}

/**
 * BatchNorm (inference mode): (x - mean) / sqrt(var + eps) * gamma + beta
 * Uses running statistics, not batch statistics.
 */
static inline void trix_onnx_batch_norm(
    const float* x,
    const float* gamma, const float* beta,
    const float* running_mean, const float* running_var,
    float* out, int n, float eps
) {
    for (int i = 0; i < n; i++) {
        float inv_std = 1.0f / sqrtf(running_var[i] + eps);
        out[i] = gamma[i] * (x[i] - running_mean[i]) * inv_std + beta[i];
    }
}

/**
 * RMSNorm: x / sqrt(mean(x^2) + eps) * gamma
 * Simpler normalization used in LLaMA, etc.
 */
static inline void trix_onnx_rms_norm(
    const float* x, const float* gamma,
    float* out, int n, float eps
) {
    /* Compute RMS */
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += x[i] * x[i];
    }
    float rms = sqrtf(sum_sq / (float)n + eps);
    float inv_rms = 1.0f / rms;

    /* Scale */
    for (int i = 0; i < n; i++) {
        out[i] = gamma[i] * x[i] * inv_rms;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Shape Operations
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Transpose 2D: swap rows and columns
 */
static inline void trix_onnx_transpose_2d(
    const float* x, float* out,
    int rows, int cols
) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out[j * rows + i] = x[i * cols + j];
        }
    }
}

/**
 * Concat 2D along axis 0 (vertical stack)
 */
static inline void trix_onnx_concat_axis0(
    const float* a, const float* b, float* out,
    int rows_a, int rows_b, int cols
) {
    memcpy(out, a, rows_a * cols * sizeof(float));
    memcpy(out + rows_a * cols, b, rows_b * cols * sizeof(float));
}

/**
 * Concat 2D along axis 1 (horizontal stack)
 */
static inline void trix_onnx_concat_axis1(
    const float* a, const float* b, float* out,
    int rows, int cols_a, int cols_b
) {
    int cols_out = cols_a + cols_b;
    for (int i = 0; i < rows; i++) {
        memcpy(out + i * cols_out, a + i * cols_a, cols_a * sizeof(float));
        memcpy(out + i * cols_out + cols_a, b + i * cols_b, cols_b * sizeof(float));
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Attention Components (for Transformer support)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Scaled Dot-Product Attention (single head)
 * attn = softmax(Q @ K^T / sqrt(d_k)) @ V
 */
static inline void trix_onnx_attention(
    const float* Q, const float* K, const float* V,
    float* out,
    int seq_len, int d_k,
    float scale
) {
    /* Allocate temporary for attention scores */
    float* scores = (float*)alloca(seq_len * seq_len * sizeof(float));

    /* Q @ K^T */
    trix_onnx_matmul_t(Q, K, scores, seq_len, seq_len, d_k, 0, 1);

    /* Scale */
    for (int i = 0; i < seq_len * seq_len; i++) {
        scores[i] *= scale;
    }

    /* Softmax per row */
    for (int i = 0; i < seq_len; i++) {
        trix_onnx_softmax(scores + i * seq_len, scores + i * seq_len, seq_len);
    }

    /* @ V */
    trix_onnx_matmul(scores, V, out, seq_len, d_k, seq_len);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Comparison and Logic (for conditional operations)
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline void trix_onnx_equal(
    const float* a, const float* b, float* out, int n, float tol
) {
    for (int i = 0; i < n; i++) {
        out[i] = fabsf(a[i] - b[i]) < tol ? 1.0f : 0.0f;
    }
}

static inline void trix_onnx_greater(
    const float* a, const float* b, float* out, int n
) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] > b[i] ? 1.0f : 0.0f;
    }
}

static inline void trix_onnx_less(
    const float* a, const float* b, float* out, int n
) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] < b[i] ? 1.0f : 0.0f;
    }
}

static inline void trix_onnx_where(
    const float* cond, const float* a, const float* b, float* out, int n
) {
    for (int i = 0; i < n; i++) {
        out[i] = cond[i] > 0.5f ? a[i] : b[i];
    }
}

static inline void trix_onnx_clip(
    const float* x, float* out, int n, float min_val, float max_val
) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        if (v < min_val) v = min_val;
        if (v > max_val) v = max_val;
        out[i] = v;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Embedding (for NLP models)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Gather: lookup embeddings by indices
 */
static inline void trix_onnx_gather(
    const float* embeddings,  /* [vocab_size, d_model] */
    const int* indices,       /* [seq_len] */
    float* out,               /* [seq_len, d_model] */
    int seq_len, int d_model
) {
    for (int i = 0; i < seq_len; i++) {
        int idx = indices[i];
        memcpy(out + i * d_model, embeddings + idx * d_model, d_model * sizeof(float));
    }
}

#endif /* TRIXC_ONNX_SHAPES_H */
