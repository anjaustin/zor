/*
 * GILLIES Shape Implementations
 *
 * Frozen polynomials, CUDA-compatible.
 * These are the eternal truths that GILLIES executes.
 *
 * "Geometry is computation. Learning is routing."
 *
 * (c) 2025 TriX Project
 */

#ifndef GILLIES_SHAPES_DEVICE_H
#define GILLIES_SHAPES_DEVICE_H

#include <stdint.h>

#ifdef __CUDACC__
#define GILLIES_FUNC __device__ __host__ __forceinline__
#else
#define GILLIES_FUNC static inline
#endif

/* ============================================================================
 * SHAPE IDENTIFIERS
 * The frozen vocabulary. Each ID maps to one eternal polynomial.
 * ============================================================================ */

typedef enum {
    GILLIES_SHAPE_XOR = 0,      // a + b - 2ab
    GILLIES_SHAPE_AND = 1,      // ab
    GILLIES_SHAPE_OR  = 2,      // a + b - ab
    GILLIES_SHAPE_NOT = 3,      // 1 - a
    GILLIES_SHAPE_NAND = 4,     // 1 - ab
    GILLIES_SHAPE_NOR = 5,      // 1 - a - b + ab
    GILLIES_SHAPE_XNOR = 6,     // 1 - a - b + 2ab
    GILLIES_SHAPE_ADD = 7,      // a + b
    GILLIES_SHAPE_SUB = 8,      // a - b
    GILLIES_SHAPE_MUL = 9,      // a * b
    GILLIES_SHAPE_IDENTITY = 10, // a (passthrough)
    GILLIES_NUM_SHAPES
} gillies_shape_t;

/* Shape names for debugging */
#ifdef __cplusplus
extern "C" {
#endif

static const char* GILLIES_SHAPE_NAMES[] = {
    "XOR", "AND", "OR", "NOT", "NAND", "NOR", "XNOR",
    "ADD", "SUB", "MUL", "IDENTITY"
};

#ifdef __cplusplus
}
#endif

/* ============================================================================
 * BITWISE IMPLEMENTATIONS (Fast, for inference)
 * ============================================================================ */

GILLIES_FUNC uint8_t gillies_xor_u8(uint8_t a, uint8_t b) {
    return a ^ b;
}

GILLIES_FUNC uint8_t gillies_and_u8(uint8_t a, uint8_t b) {
    return a & b;
}

GILLIES_FUNC uint8_t gillies_or_u8(uint8_t a, uint8_t b) {
    return a | b;
}

GILLIES_FUNC uint8_t gillies_not_u8(uint8_t a) {
    return ~a;
}

GILLIES_FUNC uint8_t gillies_nand_u8(uint8_t a, uint8_t b) {
    return ~(a & b);
}

GILLIES_FUNC uint8_t gillies_nor_u8(uint8_t a, uint8_t b) {
    return ~(a | b);
}

GILLIES_FUNC uint8_t gillies_xnor_u8(uint8_t a, uint8_t b) {
    return ~(a ^ b);
}

GILLIES_FUNC uint8_t gillies_add_u8(uint8_t a, uint8_t b) {
    return a + b;  // Natural overflow
}

GILLIES_FUNC uint8_t gillies_sub_u8(uint8_t a, uint8_t b) {
    return a - b;  // Natural underflow
}

GILLIES_FUNC uint8_t gillies_mul_u8(uint8_t a, uint8_t b) {
    return a * b;  // Low byte of result
}

/* ============================================================================
 * POLYNOMIAL IMPLEMENTATIONS (Differentiable, for training)
 *
 * These are the Zhegalkin polynomials - Boolean algebra in polynomial form.
 * They produce exact results for binary inputs, smooth gradients for floats.
 * ============================================================================ */

GILLIES_FUNC float gillies_xor_f32(float a, float b) {
    // XOR: a ⊕ b = a + b - 2ab
    return a + b - 2.0f * a * b;
}

GILLIES_FUNC float gillies_and_f32(float a, float b) {
    // AND: a ∧ b = ab
    return a * b;
}

GILLIES_FUNC float gillies_or_f32(float a, float b) {
    // OR: a ∨ b = a + b - ab
    return a + b - a * b;
}

GILLIES_FUNC float gillies_not_f32(float a) {
    // NOT: ¬a = 1 - a
    return 1.0f - a;
}

GILLIES_FUNC float gillies_nand_f32(float a, float b) {
    // NAND: ¬(a ∧ b) = 1 - ab
    return 1.0f - a * b;
}

GILLIES_FUNC float gillies_nor_f32(float a, float b) {
    // NOR: ¬(a ∨ b) = 1 - a - b + ab
    return 1.0f - a - b + a * b;
}

GILLIES_FUNC float gillies_xnor_f32(float a, float b) {
    // XNOR: ¬(a ⊕ b) = 1 - a - b + 2ab
    return 1.0f - a - b + 2.0f * a * b;
}

GILLIES_FUNC float gillies_add_f32(float a, float b) {
    return a + b;
}

GILLIES_FUNC float gillies_sub_f32(float a, float b) {
    return a - b;
}

GILLIES_FUNC float gillies_mul_f32(float a, float b) {
    return a * b;
}

GILLIES_FUNC float gillies_identity_f32(float a) {
    return a;
}

/* ============================================================================
 * DISPATCH - Route shape_id to implementation
 * ============================================================================ */

GILLIES_FUNC float gillies_dispatch_f32(uint8_t shape_id, float a, float b) {
    switch (shape_id) {
        case GILLIES_SHAPE_XOR:      return gillies_xor_f32(a, b);
        case GILLIES_SHAPE_AND:      return gillies_and_f32(a, b);
        case GILLIES_SHAPE_OR:       return gillies_or_f32(a, b);
        case GILLIES_SHAPE_NOT:      return gillies_not_f32(a);
        case GILLIES_SHAPE_NAND:     return gillies_nand_f32(a, b);
        case GILLIES_SHAPE_NOR:      return gillies_nor_f32(a, b);
        case GILLIES_SHAPE_XNOR:     return gillies_xnor_f32(a, b);
        case GILLIES_SHAPE_ADD:      return gillies_add_f32(a, b);
        case GILLIES_SHAPE_SUB:      return gillies_sub_f32(a, b);
        case GILLIES_SHAPE_MUL:      return gillies_mul_f32(a, b);
        case GILLIES_SHAPE_IDENTITY: return gillies_identity_f32(a);
        default: return 0.0f;
    }
}

GILLIES_FUNC uint8_t gillies_dispatch_u8(uint8_t shape_id, uint8_t a, uint8_t b) {
    switch (shape_id) {
        case GILLIES_SHAPE_XOR:      return gillies_xor_u8(a, b);
        case GILLIES_SHAPE_AND:      return gillies_and_u8(a, b);
        case GILLIES_SHAPE_OR:       return gillies_or_u8(a, b);
        case GILLIES_SHAPE_NOT:      return gillies_not_u8(a);
        case GILLIES_SHAPE_NAND:     return gillies_nand_u8(a, b);
        case GILLIES_SHAPE_NOR:      return gillies_nor_u8(a, b);
        case GILLIES_SHAPE_XNOR:     return gillies_xnor_u8(a, b);
        case GILLIES_SHAPE_ADD:      return gillies_add_u8(a, b);
        case GILLIES_SHAPE_SUB:      return gillies_sub_u8(a, b);
        case GILLIES_SHAPE_MUL:      return gillies_mul_u8(a, b);
        case GILLIES_SHAPE_IDENTITY: return a;
        default: return 0;
    }
}

/* ============================================================================
 * FULL ADDER - The atomic unit of multi-bit arithmetic
 *
 * sum = a ⊕ b ⊕ c
 * carry = (a ∧ b) ∨ ((a ⊕ b) ∧ c)
 * ============================================================================ */

GILLIES_FUNC void gillies_full_adder_f32(
    float a, float b, float c,
    float* sum, float* carry
) {
    float ab_xor = gillies_xor_f32(a, b);
    *sum = gillies_xor_f32(ab_xor, c);

    float ab_and = gillies_and_f32(a, b);
    float ab_xor_c = gillies_and_f32(ab_xor, c);
    *carry = gillies_or_f32(ab_and, ab_xor_c);
}

GILLIES_FUNC void gillies_full_adder_u8(
    uint8_t a, uint8_t b, uint8_t c,
    uint8_t* sum, uint8_t* carry
) {
    *sum = (a ^ b ^ c) & 1;
    *carry = ((a & b) | ((a ^ b) & c)) & 1;
}

#endif /* GILLIES_SHAPES_DEVICE_H */
