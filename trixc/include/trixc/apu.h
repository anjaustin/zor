/*
 * TRIX ENDOGENOUS APU
 *
 * Arithmetic Processing Unit for Mixed-Precision Computation
 *
 * "Precision is a shape. The APU is frozen."
 *
 * Supports: FP4, FP8, FP16, FP32, FP64
 * All conversions and computations are frozen shapes.
 * No runtime libraries. No CUDA. Just math.
 *
 * (c) 2025 TriX Project
 */

#ifndef TRIX_APU_H
#define TRIX_APU_H

#include <stdint.h>
#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * PRECISION TYPES
 * ============================================================================ */

typedef enum {
    TRIX_FP4  = 0,   /* E2M1: 1 sign, 2 exp, 1 mantissa - extreme compression */
    TRIX_FP8  = 1,   /* E4M3: 1 sign, 4 exp, 3 mantissa - ML standard */
    TRIX_FP16 = 2,   /* IEEE 754 half precision */
    TRIX_FP32 = 3,   /* IEEE 754 single precision */
    TRIX_FP64 = 4,   /* IEEE 754 double precision */
    TRIX_NUM_PRECISIONS = 5
} trix_precision_t;

/* Precision bit widths */
static const uint8_t TRIX_PRECISION_BITS[TRIX_NUM_PRECISIONS] = {
    4, 8, 16, 32, 64
};

/* Precision names */
static const char* TRIX_PRECISION_NAMES[TRIX_NUM_PRECISIONS] = {
    "FP4", "FP8", "FP16", "FP32", "FP64"
};

/* ============================================================================
 * FP4 (E2M1) - Extreme Compression
 *
 * Format: [sign:1][exp:2][mant:1]
 * Values: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} (with subnormals)
 * Use: Routing decisions, coarse sketches
 * ============================================================================ */

typedef uint8_t trix_fp4_t;  /* Only lower 4 bits used */

/* FP4 special values */
#define TRIX_FP4_ZERO     0x0
#define TRIX_FP4_ONE      0x4   /* 0|10|0 = 1.0 */
#define TRIX_FP4_MAX      0x7   /* 0|11|1 = 6.0 */
#define TRIX_FP4_NEG_ONE  0xC   /* 1|10|0 = -1.0 */

/* ============================================================================
 * FP8 (E4M3) - ML Standard
 *
 * Format: [sign:1][exp:4][mant:3]
 * Range: ±448, precision: ~1%
 * Use: Weights, activations
 * ============================================================================ */

typedef uint8_t trix_fp8_t;

/* FP8 special values */
#define TRIX_FP8_ZERO     0x00
#define TRIX_FP8_ONE      0x38  /* 0|0111|000 = 1.0 (bias=7) */
#define TRIX_FP8_MAX      0x7F  /* 0|1111|111 = 448 */
#define TRIX_FP8_NEG_ONE  0xB8  /* 1|0111|000 = -1.0 */

/* ============================================================================
 * FP16 - IEEE 754 Half Precision
 *
 * Format: [sign:1][exp:5][mant:10]
 * Range: ±65504, precision: ~0.1%
 * Use: General computation
 * ============================================================================ */

typedef uint16_t trix_fp16_t;

/* FP16 special values */
#define TRIX_FP16_ZERO     0x0000
#define TRIX_FP16_ONE      0x3C00  /* 0|01111|0000000000 = 1.0 */
#define TRIX_FP16_INF      0x7C00
#define TRIX_FP16_NEG_ONE  0xBC00

/* ============================================================================
 * CONVERSION SHAPES - Frozen bit manipulation
 * ============================================================================ */

/* FP32 → FP16 */
static inline trix_fp16_t trix_fp32_to_fp16(float x) {
    uint32_t bits;
    __builtin_memcpy(&bits, &x, sizeof(bits));

    uint16_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x03FF;

    /* Handle special cases */
    if (exp <= 0) {
        /* Underflow to zero or subnormal */
        if (exp < -10) return sign;
        mant = (mant | 0x0400) >> (1 - exp);
        return sign | mant;
    }
    if (exp >= 31) {
        /* Overflow to infinity or NaN */
        if (exp == 31 && mant) return sign | 0x7C00 | mant;
        return sign | 0x7C00;
    }

    return sign | ((uint16_t)exp << 10) | mant;
}

/* FP16 → FP32 */
static inline float trix_fp16_to_fp32(trix_fp16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = (h & 0x03FF) << 13;

    if (exp == 0) {
        /* Zero or subnormal */
        if (mant == 0) {
            uint32_t bits = sign;
            float result;
            __builtin_memcpy(&result, &bits, sizeof(result));
            return result;
        }
        /* Normalize subnormal */
        while ((mant & 0x00800000) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x00800000;
    } else if (exp == 31) {
        exp = 255;  /* Inf or NaN */
    } else {
        exp = exp - 15 + 127;
    }

    uint32_t bits = sign | ((uint32_t)exp << 23) | mant;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}

/* FP32 → FP8 (E4M3) */
static inline trix_fp8_t trix_fp32_to_fp8(float x) {
    uint32_t bits;
    __builtin_memcpy(&bits, &x, sizeof(bits));

    uint8_t sign = (bits >> 24) & 0x80;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 7;  /* FP8 bias = 7 */
    uint32_t mant = (bits >> 20) & 0x07;

    /* Clamp to FP8 range */
    if (exp <= 0) return sign;  /* Underflow */
    if (exp >= 15) return sign | 0x7F;  /* Overflow to max */

    return sign | ((uint8_t)exp << 3) | mant;
}

/* FP8 → FP32 */
static inline float trix_fp8_to_fp32(trix_fp8_t f) {
    uint32_t sign = (f & 0x80) << 24;
    int32_t exp = (f >> 3) & 0x0F;
    uint32_t mant = (f & 0x07) << 20;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t bits = sign;
            float result;
            __builtin_memcpy(&result, &bits, sizeof(result));
            return result;
        }
        /* Subnormal - normalize */
        while ((mant & 0x00800000) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x00800000;
    }

    exp = exp - 7 + 127;  /* Convert from FP8 bias to FP32 bias */

    uint32_t bits = sign | ((uint32_t)exp << 23) | mant;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}

/* FP32 → FP4 (E2M1) */
static inline trix_fp4_t trix_fp32_to_fp4(float x) {
    uint32_t bits;
    __builtin_memcpy(&bits, &x, sizeof(bits));

    uint8_t sign = (bits >> 28) & 0x8;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 1;  /* FP4 bias = 1 */
    uint32_t mant = (bits >> 22) & 0x1;

    /* Clamp to FP4 range */
    if (exp <= 0) return sign;  /* Underflow */
    if (exp >= 3) return sign | 0x7;  /* Overflow to max (6.0) */

    return sign | ((uint8_t)exp << 1) | mant;
}

/* FP4 → FP32 */
static inline float trix_fp4_to_fp32(trix_fp4_t f) {
    uint32_t sign = (f & 0x8) << 28;
    int32_t exp = (f >> 1) & 0x3;
    uint32_t mant = (f & 0x1) << 22;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t bits = sign;
            float result;
            __builtin_memcpy(&result, &bits, sizeof(result));
            return result;
        }
        /* Subnormal */
        exp = -126 + 127;  /* Minimum normal exponent */
        mant <<= 1;
    } else {
        exp = exp - 1 + 127;
    }

    uint32_t bits = sign | ((uint32_t)exp << 23) | mant;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}

/* ============================================================================
 * FROZEN LOGIC SHAPES - Work at any precision via FP32
 * ============================================================================ */

/* XOR: a ⊕ b = a + b - 2ab (exact on binary, approximate on float) */
static inline float trix_shape_xor_f32(float a, float b) {
    return a + b - 2.0f * a * b;
}

/* AND: a ∧ b = ab */
static inline float trix_shape_and_f32(float a, float b) {
    return a * b;
}

/* OR: a ∨ b = a + b - ab */
static inline float trix_shape_or_f32(float a, float b) {
    return a + b - a * b;
}

/* NOT: ¬a = 1 - a */
static inline float trix_shape_not_f32(float a) {
    return 1.0f - a;
}

/* NAND: ¬(a ∧ b) = 1 - ab */
static inline float trix_shape_nand_f32(float a, float b) {
    return 1.0f - a * b;
}

/* NOR: ¬(a ∨ b) = 1 - a - b + ab */
static inline float trix_shape_nor_f32(float a, float b) {
    return 1.0f - a - b + a * b;
}

/* XNOR: ¬(a ⊕ b) = 1 - (a + b - 2ab) = 1 - a - b + 2ab */
static inline float trix_shape_xnor_f32(float a, float b) {
    return 1.0f - a - b + 2.0f * a * b;
}

/* ============================================================================
 * APU DISPATCH - Route operations to appropriate precision
 * ============================================================================ */

typedef enum {
    TRIX_OP_XOR = 0,
    TRIX_OP_AND,
    TRIX_OP_OR,
    TRIX_OP_NOT,
    TRIX_OP_NAND,
    TRIX_OP_NOR,
    TRIX_OP_ADD,
    TRIX_OP_SUB,
    TRIX_OP_MUL,
    TRIX_OP_ACCUMULATE,
    TRIX_NUM_OPS
} trix_op_t;

/* Default precision routing table - can be customized per model */
static const trix_precision_t TRIX_APU_DEFAULT_ROUTING[TRIX_NUM_OPS] = {
    TRIX_FP16,  /* XOR: medium precision */
    TRIX_FP16,  /* AND: medium precision */
    TRIX_FP16,  /* OR: medium precision */
    TRIX_FP16,  /* NOT: medium precision */
    TRIX_FP16,  /* NAND: medium precision */
    TRIX_FP16,  /* NOR: medium precision */
    TRIX_FP16,  /* ADD: medium precision */
    TRIX_FP16,  /* SUB: medium precision */
    TRIX_FP16,  /* MUL: medium precision */
    TRIX_FP32,  /* ACCUMULATE: high precision (headroom) */
};

/* ============================================================================
 * APU CONTEXT - Holds routing table and state
 * ============================================================================ */

typedef struct {
    trix_precision_t routing[TRIX_NUM_OPS];
    uint64_t op_counts[TRIX_NUM_OPS];
    uint64_t precision_counts[TRIX_NUM_PRECISIONS];
} trix_apu_t;

/* Initialize APU with default routing */
static inline void trix_apu_init(trix_apu_t* apu) {
    for (int i = 0; i < TRIX_NUM_OPS; i++) {
        apu->routing[i] = TRIX_APU_DEFAULT_ROUTING[i];
        apu->op_counts[i] = 0;
    }
    for (int i = 0; i < TRIX_NUM_PRECISIONS; i++) {
        apu->precision_counts[i] = 0;
    }
}

/* Set routing for a specific operation */
static inline void trix_apu_set_routing(trix_apu_t* apu, trix_op_t op, trix_precision_t prec) {
    apu->routing[op] = prec;
}

/* ============================================================================
 * APU EXECUTE - The main dispatch function
 * ============================================================================ */

/* Execute binary operation through APU */
static inline float trix_apu_execute(
    trix_apu_t* apu,
    trix_op_t op,
    float a,
    float b,
    trix_precision_t in_prec,
    trix_precision_t out_prec
) {
    /* Get compute precision from routing table */
    trix_precision_t compute_prec = apu->routing[op];

    /* Update counters */
    apu->op_counts[op]++;
    apu->precision_counts[compute_prec]++;

    float result;

    /* Execute at compute precision */
    /* For now, we compute in FP32 and convert at boundaries */
    /* Future: specialize for each precision level */

    switch (op) {
        case TRIX_OP_XOR: result = trix_shape_xor_f32(a, b); break;
        case TRIX_OP_AND: result = trix_shape_and_f32(a, b); break;
        case TRIX_OP_OR:  result = trix_shape_or_f32(a, b);  break;
        case TRIX_OP_NOT: result = trix_shape_not_f32(a);    break;
        case TRIX_OP_NAND: result = trix_shape_nand_f32(a, b); break;
        case TRIX_OP_NOR: result = trix_shape_nor_f32(a, b); break;
        case TRIX_OP_ADD: result = a + b; break;
        case TRIX_OP_SUB: result = a - b; break;
        case TRIX_OP_MUL: result = a * b; break;
        case TRIX_OP_ACCUMULATE: result = a + b; break;  /* Same as add but tracked separately */
        default: result = 0.0f; break;
    }

    /* Apply precision truncation based on compute precision */
    switch (compute_prec) {
        case TRIX_FP4:
            result = trix_fp4_to_fp32(trix_fp32_to_fp4(result));
            break;
        case TRIX_FP8:
            result = trix_fp8_to_fp32(trix_fp32_to_fp8(result));
            break;
        case TRIX_FP16:
            result = trix_fp16_to_fp32(trix_fp32_to_fp16(result));
            break;
        case TRIX_FP32:
        case TRIX_FP64:
        default:
            /* No truncation needed */
            break;
    }

    return result;
}

/* Execute unary operation */
static inline float trix_apu_execute_unary(
    trix_apu_t* apu,
    trix_op_t op,
    float a,
    trix_precision_t in_prec,
    trix_precision_t out_prec
) {
    return trix_apu_execute(apu, op, a, 0.0f, in_prec, out_prec);
}

/* ============================================================================
 * APU STATISTICS
 * ============================================================================ */

static inline void trix_apu_print_stats(const trix_apu_t* apu) {
    printf("APU Statistics:\n");
    printf("  Operations:\n");
    for (int i = 0; i < TRIX_NUM_OPS; i++) {
        if (apu->op_counts[i] > 0) {
            printf("    %d: %lu ops\n", i, (unsigned long)apu->op_counts[i]);
        }
    }
    printf("  Precisions:\n");
    for (int i = 0; i < TRIX_NUM_PRECISIONS; i++) {
        if (apu->precision_counts[i] > 0) {
            printf("    %s: %lu ops\n", TRIX_PRECISION_NAMES[i],
                   (unsigned long)apu->precision_counts[i]);
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif /* TRIX_APU_H */
