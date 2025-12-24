/**
 * Geocadesia: Shape Implementations (C)
 *
 * Reference implementations of all shapes in Geocadesia.
 * Header-only library for maximum portability.
 *
 * "It's all in the reflexes."
 */

#ifndef GEOCADESIA_SHAPES_H
#define GEOCADESIA_SHAPES_H

#include <math.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ==========================================================================
 * LOGIC KINGDOM - Boolean operations (frozen, exact)
 * ========================================================================== */

/**
 * Differentiable XOR for continuous inputs in [0, 1].
 * XOR(a, b) = a + b - 2ab
 */
static inline float geo_xor(float a, float b) {
    return a + b - 2.0f * a * b;
}

/**
 * Discrete XOR for boolean inputs.
 */
static inline int geo_xor_discrete(int a, int b) {
    return a ^ b;
}

/**
 * Differentiable AND for continuous inputs in [0, 1].
 * AND(a, b) = a * b
 */
static inline float geo_and(float a, float b) {
    return a * b;
}

/**
 * Discrete AND for boolean inputs.
 */
static inline int geo_and_discrete(int a, int b) {
    return a & b;
}

/**
 * Differentiable OR for continuous inputs in [0, 1].
 * OR(a, b) = a + b - ab (probabilistic OR)
 */
static inline float geo_or(float a, float b) {
    return a + b - a * b;
}

/**
 * Discrete OR for boolean inputs.
 */
static inline int geo_or_discrete(int a, int b) {
    return a | b;
}

/**
 * Differentiable NOT for continuous inputs in [0, 1].
 * NOT(a) = 1 - a
 */
static inline float geo_not(float a) {
    return 1.0f - a;
}

/**
 * Discrete NOT for boolean inputs.
 */
static inline int geo_not_discrete(int a) {
    return !a;
}

/**
 * Differentiable NAND for continuous inputs in [0, 1].
 * NAND(a, b) = 1 - ab
 * Universal gate.
 */
static inline float geo_nand(float a, float b) {
    return 1.0f - a * b;
}

/**
 * Discrete NAND for boolean inputs.
 */
static inline int geo_nand_discrete(int a, int b) {
    return !(a & b);
}

/**
 * Differentiable NOR for continuous inputs in [0, 1].
 * NOR(a, b) = (1-a)(1-b)
 * Universal gate.
 */
static inline float geo_nor(float a, float b) {
    return (1.0f - a) * (1.0f - b);
}

/**
 * Discrete NOR for boolean inputs.
 */
static inline int geo_nor_discrete(int a, int b) {
    return !(a | b);
}

/**
 * Differentiable XNOR (equivalence) for continuous inputs in [0, 1].
 * XNOR(a, b) = 1 - a - b + 2ab
 */
static inline float geo_xnor(float a, float b) {
    return 1.0f - a - b + 2.0f * a * b;
}

/**
 * Discrete XNOR for boolean inputs.
 */
static inline int geo_xnor_discrete(int a, int b) {
    return !(a ^ b);
}

/* ==========================================================================
 * ARITHMETIC KINGDOM - Numeric computation (frozen)
 * ========================================================================== */

/**
 * Result struct for half adder.
 */
typedef struct {
    float sum;
    float carry;
} GeoHalfAdderResult;

/**
 * Differentiable half adder for continuous inputs in [0, 1].
 * sum   = XOR(a, b)
 * carry = AND(a, b)
 */
static inline GeoHalfAdderResult geo_half_adder(float a, float b) {
    GeoHalfAdderResult result;
    result.sum = a + b - 2.0f * a * b;   /* XOR */
    result.carry = a * b;                 /* AND */
    return result;
}

/**
 * Result struct for full adder.
 */
typedef struct {
    float sum;
    float carry_out;
} GeoFullAdderResult;

/**
 * Differentiable full adder for continuous inputs in [0, 1].
 * Adds a + b + c_in, produces sum and carry_out.
 */
static inline GeoFullAdderResult geo_full_adder(float a, float b, float c_in) {
    /* First half adder: a XOR b */
    float sum1 = a + b - 2.0f * a * b;
    float carry1 = a * b;

    /* Second half adder: sum1 XOR c_in */
    float sum_out = sum1 + c_in - 2.0f * sum1 * c_in;
    float carry2 = sum1 * c_in;

    /* OR the carries */
    float c_out = carry1 + carry2 - carry1 * carry2;

    GeoFullAdderResult result;
    result.sum = sum_out;
    result.carry_out = c_out;
    return result;
}

/* ==========================================================================
 * ACTIVATION KINGDOM - Nonlinearities (mostly frozen)
 * ========================================================================== */

/**
 * Rectified Linear Unit.
 * ReLU(x) = max(0, x)
 */
static inline float geo_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

/**
 * Branchless ReLU (may be faster on some architectures).
 */
static inline float geo_relu_branchless(float x) {
    return x * (x > 0.0f);
}

/**
 * Logistic sigmoid function.
 * sigmoid(x) = 1 / (1 + e^(-x))
 * Numerically stable implementation.
 */
static inline float geo_sigmoid(float x) {
    if (x >= 0.0f) {
        return 1.0f / (1.0f + expf(-x));
    } else {
        float exp_x = expf(x);
        return exp_x / (1.0f + exp_x);
    }
}

/**
 * Hyperbolic tangent.
 * tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 */
static inline float geo_tanh(float x) {
    return tanhf(x);
}

/**
 * Gaussian Error Linear Unit (exact).
 * GELU(x) = x * Phi(x)
 */
static inline float geo_gelu(float x) {
    return x * 0.5f * (1.0f + erff(x / sqrtf(2.0f)));
}

/**
 * GELU approximation using tanh.
 * Faster than exact GELU.
 */
static inline float geo_gelu_approx(float x) {
    const float sqrt_2_pi = 0.7978845608f;  /* sqrt(2/pi) */
    float inner = sqrt_2_pi * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(inner));
}

/**
 * Swish activation (SiLU).
 * swish(x) = x * sigmoid(x)
 */
static inline float geo_swish(float x) {
    return x * geo_sigmoid(x);
}

/**
 * Leaky ReLU.
 * leaky_relu(x) = x if x > 0 else alpha * x
 */
static inline float geo_leaky_relu(float x, float alpha) {
    return x > 0.0f ? x : alpha * x;
}

/**
 * Softmax function (in-place).
 * Numerically stable implementation.
 *
 * @param x     Input array (will be overwritten with output)
 * @param n     Length of array
 */
static inline void geo_softmax(float* x, size_t n) {
    /* Find max for numerical stability */
    float max_x = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max_x) max_x = x[i];
    }

    /* Compute exp and sum */
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_x);
        sum += x[i];
    }

    /* Normalize */
    for (size_t i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

/**
 * Softmax function (out-of-place).
 *
 * @param in    Input array
 * @param out   Output array (must be pre-allocated)
 * @param n     Length of arrays
 */
static inline void geo_softmax_out(const float* in, float* out, size_t n) {
    /* Find max for numerical stability */
    float max_x = in[0];
    for (size_t i = 1; i < n; i++) {
        if (in[i] > max_x) max_x = in[i];
    }

    /* Compute exp and sum */
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        out[i] = expf(in[i] - max_x);
        sum += out[i];
    }

    /* Normalize */
    for (size_t i = 0; i < n; i++) {
        out[i] /= sum;
    }
}

/* ==========================================================================
 * POOLING KINGDOM - Reduction operations (frozen)
 * ========================================================================== */

/**
 * Maximum pooling.
 */
static inline float geo_max_pool(const float* x, size_t n) {
    float max_val = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    return max_val;
}

/**
 * Average pooling.
 */
static inline float geo_avg_pool(const float* x, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum / (float)n;
}

/**
 * Sum pooling.
 */
static inline float geo_sum_pool(const float* x, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum;
}

/**
 * Minimum pooling.
 */
static inline float geo_min_pool(const float* x, size_t n) {
    float min_val = x[0];
    for (size_t i = 1; i < n; i++) {
        if (x[i] < min_val) min_val = x[i];
    }
    return min_val;
}

#ifdef __cplusplus
}
#endif

#endif /* GEOCADESIA_SHAPES_H */
