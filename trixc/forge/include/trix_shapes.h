/**
 * TRIX Shapes - Frozen Computational Shapes
 *
 * Pure geometric implementations of neural operations.
 * Zero learnable parameters. Exact computation.
 *
 * "Geometry is computation."
 */

#ifndef TRIX_SHAPES_H
#define TRIX_SHAPES_H

#include <math.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Frozen Logic Gates
 * ============================================================================
 *
 * These implement boolean logic as differentiable operations.
 * Exact at binary inputs (0 and 1), smooth everywhere else.
 */

/**
 * XOR Gate: a + b - 2ab
 *
 * Truth table (exact at binary):
 *   XOR(0, 0) = 0
 *   XOR(0, 1) = 1
 *   XOR(1, 0) = 1
 *   XOR(1, 1) = 0
 */
static inline float trix_xor(float a, float b) {
    return a + b - 2.0f * a * b;
}

/**
 * AND Gate: a * b
 *
 * Truth table:
 *   AND(0, 0) = 0
 *   AND(0, 1) = 0
 *   AND(1, 0) = 0
 *   AND(1, 1) = 1
 */
static inline float trix_and(float a, float b) {
    return a * b;
}

/**
 * OR Gate: a + b - ab
 *
 * Truth table:
 *   OR(0, 0) = 0
 *   OR(0, 1) = 1
 *   OR(1, 0) = 1
 *   OR(1, 1) = 1
 */
static inline float trix_or(float a, float b) {
    return a + b - a * b;
}

/**
 * NOT Gate: 1 - a
 *
 * Truth table:
 *   NOT(0) = 1
 *   NOT(1) = 0
 */
static inline float trix_not(float a) {
    return 1.0f - a;
}

/**
 * NAND Gate: NOT(AND(a, b)) = 1 - ab
 */
static inline float trix_nand(float a, float b) {
    return 1.0f - a * b;
}

/**
 * NOR Gate: NOT(OR(a, b)) = 1 - (a + b - ab)
 */
static inline float trix_nor(float a, float b) {
    return 1.0f - (a + b - a * b);
}

/**
 * XNOR Gate: NOT(XOR(a, b)) = 1 - (a + b - 2ab)
 */
static inline float trix_xnor(float a, float b) {
    return 1.0f - (a + b - 2.0f * a * b);
}

/* ============================================================================
 * De Morgan's Laws (Verification)
 * ============================================================================
 *
 * These verify that our frozen shapes satisfy De Morgan's laws:
 *   NOT(AND(a,b)) = OR(NOT(a), NOT(b))
 *   NOT(OR(a,b)) = AND(NOT(a), NOT(b))
 */

static inline float trix_demorgan_1(float a, float b) {
    /* NOT(AND(a,b)) = OR(NOT(a), NOT(b)) */
    float left = trix_not(trix_and(a, b));
    float right = trix_or(trix_not(a), trix_not(b));
    return left - right;  /* Should be ~0 */
}

static inline float trix_demorgan_2(float a, float b) {
    /* NOT(OR(a,b)) = AND(NOT(a), NOT(b)) */
    float left = trix_not(trix_or(a, b));
    float right = trix_and(trix_not(a), trix_not(b));
    return left - right;  /* Should be ~0 */
}

/* ============================================================================
 * Activation Functions
 * ============================================================================ */

/**
 * ReLU: max(0, x)
 */
static inline float trix_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

/**
 * Leaky ReLU: x if x > 0 else alpha * x
 */
static inline float trix_leaky_relu(float x, float alpha) {
    return x > 0.0f ? x : alpha * x;
}

/**
 * Sigmoid: 1 / (1 + exp(-x))
 */
static inline float trix_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 */
static inline float trix_tanh(float x) {
    return tanhf(x);
}

/**
 * GELU (approximate): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
static inline float trix_gelu(float x) {
    const float sqrt_2_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_pi * (x + coeff * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

/**
 * Swish: x * sigmoid(x)
 */
static inline float trix_swish(float x) {
    return x * trix_sigmoid(x);
}

/* ============================================================================
 * Vector Operations
 * ============================================================================ */

/**
 * Softmax (in-place)
 */
static inline void trix_softmax(float* x, size_t n) {
    /* Find max for numerical stability */
    float max_val = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    /* Compute exp and sum */
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    /* Normalize */
    for (size_t i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

/**
 * Layer Normalization (in-place)
 */
static inline void trix_layer_norm(float* x, size_t n, float eps) {
    /* Compute mean */
    float mean = 0.0f;
    for (size_t i = 0; i < n; i++) {
        mean += x[i];
    }
    mean /= (float)n;

    /* Compute variance */
    float var = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= (float)n;

    /* Normalize */
    float inv_std = 1.0f / sqrtf(var + eps);
    for (size_t i = 0; i < n; i++) {
        x[i] = (x[i] - mean) * inv_std;
    }
}

/**
 * Argmax: find index of maximum value
 */
static inline size_t trix_argmax(const float* x, size_t n) {
    size_t max_idx = 0;
    float max_val = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_idx = i;
        }
    }
    return max_idx;
}

/**
 * Argmin: find index of minimum value
 * Used in FrozenDB for nearest neighbor search.
 */
static inline size_t trix_argmin(const float* x, size_t n) {
    size_t min_idx = 0;
    float min_val = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] < min_val) {
            min_val = x[i];
            min_idx = i;
        }
    }
    return min_idx;
}

/* ============================================================================
 * Bit Operations (FrozenDB Support)
 * ============================================================================ */

/**
 * Popcount: count the number of 1-bits
 * Foundation of Hamming distance for FrozenDB.
 */
static inline int trix_popcount(unsigned int x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(x);
#else
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
#endif
}

/**
 * Popcount for 64-bit values
 */
static inline int trix_popcount64(unsigned long long x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
#endif
}

/**
 * Hamming distance: count differing bits between two values
 * hamming(a, b) = popcount(a XOR b)
 * THE metric for FrozenDB vector search.
 */
static inline int trix_hamming(unsigned int a, unsigned int b) {
    return trix_popcount(a ^ b);
}

/**
 * Hamming distance for 64-bit values
 */
static inline int trix_hamming64(unsigned long long a, unsigned long long b) {
    return trix_popcount64(a ^ b);
}

/**
 * Hamming distance for 512-bit values (stored as array of 8 x uint64_t)
 * This is the core operation for FrozenDB vector search on Thor.
 */
static inline int trix_hamming_512(const unsigned long long* a,
                                   const unsigned long long* b) {
    int distance = 0;
    for (int i = 0; i < 8; i++) {  /* 8 × 64 = 512 bits */
        distance += trix_popcount64(a[i] ^ b[i]);
    }
    return distance;
}

/**
 * FrozenDB query: find nearest neighbor in signature array
 * Returns the index of the closest match.
 *
 * signatures: array of n signatures, each 512-bit (8 x uint64_t)
 * query: the query signature (512-bit)
 * n: number of signatures
 */
static inline size_t trix_frozendb_query(
    const unsigned long long* signatures,
    const unsigned long long* query,
    size_t n
) {
    size_t best_idx = 0;
    int best_dist = trix_hamming_512(signatures, query);

    for (size_t i = 1; i < n; i++) {
        int dist = trix_hamming_512(signatures + i * 8, query);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }

    return best_idx;
}

/* ============================================================================
 * Matrix Operations
 * ============================================================================ */

/**
 * Matrix-Vector Multiply: y = Wx + b
 *
 * W: [out_dim x in_dim] (row-major)
 * x: [in_dim]
 * b: [out_dim] (can be NULL)
 * y: [out_dim]
 */
static inline void trix_matvec(const float* W, const float* x, const float* b,
                               float* y, size_t out_dim, size_t in_dim) {
    for (size_t i = 0; i < out_dim; i++) {
        float sum = (b != NULL) ? b[i] : 0.0f;
        for (size_t j = 0; j < in_dim; j++) {
            sum += W[i * in_dim + j] * x[j];
        }
        y[i] = sum;
    }
}

/**
 * Matrix-Matrix Multiply: C = AB
 *
 * A: [M x K]
 * B: [K x N]
 * C: [M x N]
 */
static inline void trix_matmul(const float* A, const float* B, float* C,
                               size_t M, size_t K, size_t N) {
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/* ============================================================================
 * Full Adder (Composite Frozen Shape)
 * ============================================================================
 *
 * Implements: Sum = A XOR B XOR Cin
 *            Cout = (A AND B) OR (Cin AND (A XOR B))
 */

static inline void trix_full_adder(float a, float b, float cin,
                                   float* sum, float* cout) {
    float xor_ab = trix_xor(a, b);
    *sum = trix_xor(xor_ab, cin);
    *cout = trix_or(trix_and(a, b), trix_and(cin, xor_ab));
}

/* ============================================================================
 * Half Adder
 * ============================================================================ */

static inline void trix_half_adder(float a, float b,
                                   float* sum, float* carry) {
    *sum = trix_xor(a, b);
    *carry = trix_and(a, b);
}

#ifdef __cplusplus
}
#endif

#endif /* TRIX_SHAPES_H */
