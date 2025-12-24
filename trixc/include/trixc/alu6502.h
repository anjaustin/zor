/*
 * TRIX APU 6502 ALU
 *
 * Complete 6502 ALU with APU precision management.
 * 16 frozen shapes, precision-controlled execution.
 *
 * "A 14KB executable that runs a CPU. With mixed precision."
 */

#ifndef TRIX_APU_ALU6502_H
#define TRIX_APU_ALU6502_H

#include "apu.h"
#include "shapes.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 6502 OPCODES
 * ============================================================================ */

typedef enum {
    ALU_ADC = 0,   /* Add with carry */
    ALU_SBC = 1,   /* Subtract with borrow */
    ALU_AND = 2,   /* Bitwise AND */
    ALU_ORA = 3,   /* Bitwise OR */
    ALU_EOR = 4,   /* Bitwise XOR (Exclusive OR) */
    ALU_ASL = 5,   /* Arithmetic shift left */
    ALU_LSR = 6,   /* Logical shift right */
    ALU_ROL = 7,   /* Rotate left */
    ALU_ROR = 8,   /* Rotate right */
    ALU_INC = 9,   /* Increment */
    ALU_DEC = 10,  /* Decrement */
    ALU_NUM_OPS = 11
} trix_alu_op_t;

static const char* ALU_OP_NAMES[ALU_NUM_OPS] = {
    "ADC", "SBC", "AND", "ORA", "EOR",
    "ASL", "LSR", "ROL", "ROR", "INC", "DEC"
};

/* ============================================================================
 * ALU PRECISION ROUTING TABLE
 *
 * Different operations can use different precisions.
 * This is the "frozen" routing learned during training.
 * ============================================================================ */

typedef struct {
    trix_apu_t apu;
    trix_precision_t op_precisions[ALU_NUM_OPS];
    uint64_t op_counts[ALU_NUM_OPS];
} trix_alu6502_t;

/* Initialize ALU with default precision routing */
static inline void trix_alu6502_init(trix_alu6502_t* alu) {
    trix_apu_init(&alu->apu);

    /* Default: all ops at FP16 except accumulation at FP32 */
    for (int i = 0; i < ALU_NUM_OPS; i++) {
        alu->op_precisions[i] = TRIX_FP16;
        alu->op_counts[i] = 0;
    }

    /* ADC and SBC need higher precision for carry chain */
    alu->op_precisions[ALU_ADC] = TRIX_FP32;
    alu->op_precisions[ALU_SBC] = TRIX_FP32;
}

/* Set precision for a specific operation */
static inline void trix_alu6502_set_precision(
    trix_alu6502_t* alu,
    trix_alu_op_t op,
    trix_precision_t prec
) {
    alu->op_precisions[op] = prec;
}

/* ============================================================================
 * ALU EXECUTION WITH PRECISION CONTROL
 * ============================================================================ */

/* Execute ALU operation */
static inline void trix_alu6502_execute(
    trix_alu6502_t* alu,
    trix_alu_op_t op,
    const float* a,      /* Input A: 8 bits */
    const float* b,      /* Input B: 8 bits (unused for some ops) */
    float c_in,          /* Carry in */
    float* result,       /* Result: 8 bits */
    float* c_out         /* Carry out */
) {
    trix_precision_t prec = alu->op_precisions[op];
    alu->op_counts[op]++;
    alu->apu.precision_counts[prec]++;

    /* Execute shape based on operation */
    switch (op) {
        case ALU_ADC:
            trix_shape_ripple_add_p(a, b, c_in, result, c_out, prec);
            break;

        case ALU_SBC:
            trix_shape_ripple_sub(a, b, c_in, result, c_out);
            /* Apply precision truncation */
            if (prec == TRIX_FP16) {
                for (int i = 0; i < 8; i++) {
                    result[i] = trix_fp16_to_fp32(trix_fp32_to_fp16(result[i]));
                }
            }
            break;

        case ALU_AND:
            trix_shape_and_8bit(a, b, result);
            *c_out = 0.0f;
            break;

        case ALU_ORA:
            trix_shape_or_8bit(a, b, result);
            *c_out = 0.0f;
            break;

        case ALU_EOR:
            trix_shape_xor_8bit(a, b, result);
            *c_out = 0.0f;
            break;

        case ALU_ASL:
            trix_shape_asl(a, result, c_out);
            break;

        case ALU_LSR:
            trix_shape_lsr(a, result, c_out);
            break;

        case ALU_ROL:
            trix_shape_rol(a, c_in, result, c_out);
            break;

        case ALU_ROR:
            trix_shape_ror(a, c_in, result, c_out);
            break;

        case ALU_INC:
            trix_shape_inc(a, result, c_out);
            break;

        case ALU_DEC:
            trix_shape_dec(a, result, c_out);
            break;

        default:
            /* Unknown op - zero result */
            for (int i = 0; i < 8; i++) result[i] = 0.0f;
            *c_out = 0.0f;
            break;
    }
}

/* ============================================================================
 * INTEGER CONVENIENCE INTERFACE
 * ============================================================================ */

/* Convert int to bits */
static inline void trix_int_to_bits(uint8_t val, float* bits) {
    for (int i = 0; i < 8; i++) {
        bits[i] = (float)((val >> i) & 1);
    }
}

/* Convert bits to int */
static inline uint8_t trix_bits_to_int(const float* bits) {
    uint8_t val = 0;
    for (int i = 0; i < 8; i++) {
        if (bits[i] > 0.5f) val |= (1 << i);
    }
    return val;
}

/* Execute ALU operation on integers */
static inline uint8_t trix_alu6502_execute_int(
    trix_alu6502_t* alu,
    trix_alu_op_t op,
    uint8_t a,
    uint8_t b,
    int c_in,
    int* c_out
) {
    float a_bits[8], b_bits[8], result_bits[8];
    float carry_out;

    trix_int_to_bits(a, a_bits);
    trix_int_to_bits(b, b_bits);

    trix_alu6502_execute(alu, op, a_bits, b_bits, (float)c_in, result_bits, &carry_out);

    *c_out = (carry_out > 0.5f) ? 1 : 0;
    return trix_bits_to_int(result_bits);
}

/* ============================================================================
 * ALU STATISTICS
 * ============================================================================ */

static inline void trix_alu6502_print_stats(const trix_alu6502_t* alu) {
    printf("6502 ALU Statistics:\n");
    printf("  Operations:\n");
    for (int i = 0; i < ALU_NUM_OPS; i++) {
        if (alu->op_counts[i] > 0) {
            printf("    %s: %lu ops @ %s\n",
                   ALU_OP_NAMES[i],
                   (unsigned long)alu->op_counts[i],
                   TRIX_PRECISION_NAMES[alu->op_precisions[i]]);
        }
    }
    printf("  Precision usage:\n");
    for (int i = 0; i < TRIX_NUM_PRECISIONS; i++) {
        if (alu->apu.precision_counts[i] > 0) {
            printf("    %s: %lu ops\n",
                   TRIX_PRECISION_NAMES[i],
                   (unsigned long)alu->apu.precision_counts[i]);
        }
    }
}

/* ============================================================================
 * STANDALONE MAIN (for --standalone compilation)
 * ============================================================================ */

#ifdef TRIX_ALU6502_STANDALONE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

static trix_alu_op_t parse_opcode(const char* name) {
    for (int i = 0; i < ALU_NUM_OPS; i++) {
        if (strcasecmp(name, ALU_OP_NAMES[i]) == 0) {
            return (trix_alu_op_t)i;
        }
    }
    return ALU_ADC;  /* Default */
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: %s <opcode> <a> <b> [carry]\n", argv[0]);
        printf("Opcodes: ADC SBC AND ORA EOR ASL LSR ROL ROR INC DEC\n");
        printf("\nExample: %s ADC 16 32\n", argv[0]);
        return 1;
    }

    trix_alu_op_t op = parse_opcode(argv[1]);
    uint8_t a = (uint8_t)atoi(argv[2]);
    uint8_t b = (uint8_t)atoi(argv[3]);
    int c_in = (argc > 4) ? atoi(argv[4]) : 0;
    int c_out;

    trix_alu6502_t alu;
    trix_alu6502_init(&alu);

    uint8_t result = trix_alu6502_execute_int(&alu, op, a, b, c_in, &c_out);

    printf("%s %d, %d", ALU_OP_NAMES[op], a, b);
    if (c_in) printf(" (C=1)");
    printf(" = %d", result);
    if (c_out) printf(" (C=1)");
    printf("\n");

    return 0;
}

#endif /* TRIX_ALU6502_STANDALONE */

#ifdef __cplusplus
}
#endif

#endif /* TRIX_APU_ALU6502_H */
