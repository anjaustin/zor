/*
 * TRIX APU SHAPES
 *
 * Frozen arithmetic shapes at multiple precisions.
 * Full adders, ripple adders, shifters, and more.
 *
 * All shapes are polynomials. All polynomials are frozen.
 */

#ifndef TRIX_APU_SHAPES_H
#define TRIX_APU_SHAPES_H

#include "apu.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * FULL ADDER - The atomic unit of arithmetic
 *
 * sum = a ⊕ b ⊕ c
 * carry = (a ∧ b) ∨ ((a ⊕ b) ∧ c)
 *
 * In polynomial form:
 * sum = a + b + c - 2ab - 2ac - 2bc + 4abc
 * carry = ab + ac + bc - 2abc
 * ============================================================================ */

static inline void trix_shape_full_adder(
    float a, float b, float c,
    float* sum, float* carry
) {
    /* XOR chain for sum */
    float ab_xor = a + b - 2.0f * a * b;
    *sum = ab_xor + c - 2.0f * ab_xor * c;

    /* AND-OR chain for carry */
    float ab_and = a * b;
    float ab_xor_c = ab_xor * c;
    *carry = ab_and + ab_xor_c - ab_and * ab_xor_c;
}

/* Full adder with precision control */
static inline void trix_shape_full_adder_p(
    float a, float b, float c,
    float* sum, float* carry,
    trix_precision_t prec
) {
    trix_shape_full_adder(a, b, c, sum, carry);

    /* Apply precision truncation */
    switch (prec) {
        case TRIX_FP4:
            *sum = trix_fp4_to_fp32(trix_fp32_to_fp4(*sum));
            *carry = trix_fp4_to_fp32(trix_fp32_to_fp4(*carry));
            break;
        case TRIX_FP8:
            *sum = trix_fp8_to_fp32(trix_fp32_to_fp8(*sum));
            *carry = trix_fp8_to_fp32(trix_fp32_to_fp8(*carry));
            break;
        case TRIX_FP16:
            *sum = trix_fp16_to_fp32(trix_fp32_to_fp16(*sum));
            *carry = trix_fp16_to_fp32(trix_fp32_to_fp16(*carry));
            break;
        default:
            break;
    }
}

/* ============================================================================
 * RIPPLE CARRY ADDER - 8-bit addition
 * ============================================================================ */

static inline void trix_shape_ripple_add(
    const float* a,    /* 8 bits, LSB first */
    const float* b,    /* 8 bits, LSB first */
    float c_in,
    float* result,     /* 8 bits, LSB first */
    float* c_out
) {
    float carry = c_in;
    for (int i = 0; i < 8; i++) {
        float sum;
        trix_shape_full_adder(a[i], b[i], carry, &sum, &carry);
        result[i] = sum;
    }
    *c_out = carry;
}

/* Ripple add with precision control */
static inline void trix_shape_ripple_add_p(
    const float* a,
    const float* b,
    float c_in,
    float* result,
    float* c_out,
    trix_precision_t prec
) {
    float carry = c_in;
    for (int i = 0; i < 8; i++) {
        float sum;
        trix_shape_full_adder_p(a[i], b[i], carry, &sum, &carry, prec);
        result[i] = sum;
    }
    *c_out = carry;
}

/* ============================================================================
 * RIPPLE CARRY SUBTRACTOR - 8-bit subtraction (a - b)
 * Uses: a - b = a + (~b) + 1
 * ============================================================================ */

static inline void trix_shape_ripple_sub(
    const float* a,
    const float* b,
    float c_in,   /* Borrow in (inverted) */
    float* result,
    float* c_out  /* Borrow out (inverted) */
) {
    float carry = 1.0f - c_in;  /* Invert for subtraction */
    for (int i = 0; i < 8; i++) {
        float b_inv = 1.0f - b[i];  /* NOT b */
        float sum;
        trix_shape_full_adder(a[i], b_inv, carry, &sum, &carry);
        result[i] = sum;
    }
    *c_out = 1.0f - carry;  /* Invert back */
}

/* ============================================================================
 * PARALLEL LOGIC OPERATIONS - 8-bit
 * ============================================================================ */

static inline void trix_shape_and_8bit(
    const float* a,
    const float* b,
    float* result
) {
    for (int i = 0; i < 8; i++) {
        result[i] = a[i] * b[i];
    }
}

static inline void trix_shape_or_8bit(
    const float* a,
    const float* b,
    float* result
) {
    for (int i = 0; i < 8; i++) {
        result[i] = a[i] + b[i] - a[i] * b[i];
    }
}

static inline void trix_shape_xor_8bit(
    const float* a,
    const float* b,
    float* result
) {
    for (int i = 0; i < 8; i++) {
        result[i] = a[i] + b[i] - 2.0f * a[i] * b[i];
    }
}

static inline void trix_shape_not_8bit(
    const float* a,
    float* result
) {
    for (int i = 0; i < 8; i++) {
        result[i] = 1.0f - a[i];
    }
}

/* ============================================================================
 * SHIFT OPERATIONS
 * ============================================================================ */

/* Arithmetic Shift Left */
static inline void trix_shape_asl(
    const float* a,
    float* result,
    float* c_out
) {
    *c_out = a[7];  /* MSB goes to carry */
    result[0] = 0.0f;  /* LSB becomes 0 */
    for (int i = 1; i < 8; i++) {
        result[i] = a[i - 1];
    }
}

/* Logical Shift Right */
static inline void trix_shape_lsr(
    const float* a,
    float* result,
    float* c_out
) {
    *c_out = a[0];  /* LSB goes to carry */
    result[7] = 0.0f;  /* MSB becomes 0 */
    for (int i = 0; i < 7; i++) {
        result[i] = a[i + 1];
    }
}

/* Rotate Left through Carry */
static inline void trix_shape_rol(
    const float* a,
    float c_in,
    float* result,
    float* c_out
) {
    *c_out = a[7];  /* MSB goes to carry */
    result[0] = c_in;  /* Carry goes to LSB */
    for (int i = 1; i < 8; i++) {
        result[i] = a[i - 1];
    }
}

/* Rotate Right through Carry */
static inline void trix_shape_ror(
    const float* a,
    float c_in,
    float* result,
    float* c_out
) {
    *c_out = a[0];  /* LSB goes to carry */
    result[7] = c_in;  /* Carry goes to MSB */
    for (int i = 0; i < 7; i++) {
        result[i] = a[i + 1];
    }
}

/* ============================================================================
 * INCREMENT / DECREMENT
 * ============================================================================ */

static inline void trix_shape_inc(
    const float* a,
    float* result,
    float* c_out
) {
    float one[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    trix_shape_ripple_add(a, one, 0.0f, result, c_out);
}

static inline void trix_shape_dec(
    const float* a,
    float* result,
    float* c_out
) {
    float one[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    trix_shape_ripple_sub(a, one, 0.0f, result, c_out);
}

/* ============================================================================
 * HAMMING DISTANCE - Providence core operation
 *
 * hamming(a, b) = popcount(a XOR b)
 * ============================================================================ */

static inline float trix_shape_hamming_8bit(
    const float* a,
    const float* b
) {
    float xor_result[8];
    trix_shape_xor_8bit(a, b, xor_result);

    /* Popcount: sum of all bits */
    float count = 0.0f;
    for (int i = 0; i < 8; i++) {
        count += xor_result[i];
    }
    return count;
}

/* Hamming distance for arbitrary length */
static inline float trix_shape_hamming(
    const float* a,
    const float* b,
    int len
) {
    float count = 0.0f;
    for (int i = 0; i < len; i++) {
        float xor_bit = a[i] + b[i] - 2.0f * a[i] * b[i];
        count += xor_bit;
    }
    return count;
}

/* ============================================================================
 * COMPARE - For routing decisions
 * ============================================================================ */

/* Compare two values, return routing signal */
static inline float trix_shape_compare(float a, float b) {
    /* Returns ~1 if a > b, ~0 if a <= b */
    /* Soft comparison for gradient flow */
    float diff = a - b;
    return 1.0f / (1.0f + expf(-10.0f * diff));  /* Steep sigmoid */
}

/* Argmin over array */
static inline int trix_shape_argmin(const float* values, int len) {
    int min_idx = 0;
    float min_val = values[0];
    for (int i = 1; i < len; i++) {
        if (values[i] < min_val) {
            min_val = values[i];
            min_idx = i;
        }
    }
    return min_idx;
}

/* ============================================================================
 * ALIASES - For clarity in tests and user code
 * ============================================================================ */

#define trix_shape_ripple_add_8bit trix_shape_ripple_add
#define trix_shape_ripple_sub_8bit trix_shape_ripple_sub

#ifdef __cplusplus
}
#endif

#endif /* TRIX_APU_SHAPES_H */
