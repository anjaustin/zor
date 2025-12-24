/*
 * TRIX APU Test Suite
 *
 * Tests precision conversion, shapes, and ALU operations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "apu.h"
#include "shapes.h"
#include "providence.h"
#include "alu6502.h"

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s\n", msg); \
        failures++; \
    } else { \
        printf("PASS: %s\n", msg); \
        passes++; \
    } \
} while(0)

#define FLOAT_EQ(a, b, tol) (fabsf((a) - (b)) < (tol))

static int passes = 0;
static int failures = 0;

/* ============================================================================
 * PRECISION CONVERSION TESTS
 * ============================================================================ */

void test_fp16_conversion(void) {
    printf("\n=== FP16 Conversion Tests ===\n");

    /* Test 1.0 */
    trix_fp16_t one = trix_fp32_to_fp16(1.0f);
    float one_back = trix_fp16_to_fp32(one);
    TEST_ASSERT(FLOAT_EQ(one_back, 1.0f, 1e-5f), "FP16: 1.0 roundtrip");

    /* Test 0.0 */
    trix_fp16_t zero = trix_fp32_to_fp16(0.0f);
    float zero_back = trix_fp16_to_fp32(zero);
    TEST_ASSERT(FLOAT_EQ(zero_back, 0.0f, 1e-5f), "FP16: 0.0 roundtrip");

    /* Test -1.0 */
    trix_fp16_t neg_one = trix_fp32_to_fp16(-1.0f);
    float neg_one_back = trix_fp16_to_fp32(neg_one);
    TEST_ASSERT(FLOAT_EQ(neg_one_back, -1.0f, 1e-5f), "FP16: -1.0 roundtrip");

    /* Test 0.5 */
    trix_fp16_t half = trix_fp32_to_fp16(0.5f);
    float half_back = trix_fp16_to_fp32(half);
    TEST_ASSERT(FLOAT_EQ(half_back, 0.5f, 1e-5f), "FP16: 0.5 roundtrip");

    /* Test 100.0 */
    trix_fp16_t hundred = trix_fp32_to_fp16(100.0f);
    float hundred_back = trix_fp16_to_fp32(hundred);
    TEST_ASSERT(FLOAT_EQ(hundred_back, 100.0f, 0.1f), "FP16: 100.0 roundtrip");
}

void test_fp8_conversion(void) {
    printf("\n=== FP8 Conversion Tests ===\n");

    /* Test 1.0 */
    trix_fp8_t one = trix_fp32_to_fp8(1.0f);
    float one_back = trix_fp8_to_fp32(one);
    TEST_ASSERT(FLOAT_EQ(one_back, 1.0f, 0.1f), "FP8: 1.0 roundtrip");

    /* Test 0.0 */
    trix_fp8_t zero = trix_fp32_to_fp8(0.0f);
    float zero_back = trix_fp8_to_fp32(zero);
    TEST_ASSERT(FLOAT_EQ(zero_back, 0.0f, 0.1f), "FP8: 0.0 roundtrip");

    /* Test 2.0 */
    trix_fp8_t two = trix_fp32_to_fp8(2.0f);
    float two_back = trix_fp8_to_fp32(two);
    TEST_ASSERT(FLOAT_EQ(two_back, 2.0f, 0.2f), "FP8: 2.0 roundtrip");
}

void test_fp4_conversion(void) {
    printf("\n=== FP4 Conversion Tests ===\n");

    /* Test 0.0 */
    trix_fp4_t zero = trix_fp32_to_fp4(0.0f);
    float zero_back = trix_fp4_to_fp32(zero);
    TEST_ASSERT(FLOAT_EQ(zero_back, 0.0f, 0.1f), "FP4: 0.0 roundtrip");

    /* Test 1.0 */
    trix_fp4_t one = trix_fp32_to_fp4(1.0f);
    float one_back = trix_fp4_to_fp32(one);
    TEST_ASSERT(one_back >= 0.5f && one_back <= 2.0f, "FP4: 1.0 approximate roundtrip");
}

/* ============================================================================
 * FROZEN SHAPE TESTS
 * ============================================================================ */

void test_logic_shapes(void) {
    printf("\n=== Logic Shape Tests ===\n");

    /* XOR truth table */
    TEST_ASSERT(FLOAT_EQ(trix_shape_xor_f32(0.0f, 0.0f), 0.0f, 1e-5f), "XOR(0,0) = 0");
    TEST_ASSERT(FLOAT_EQ(trix_shape_xor_f32(0.0f, 1.0f), 1.0f, 1e-5f), "XOR(0,1) = 1");
    TEST_ASSERT(FLOAT_EQ(trix_shape_xor_f32(1.0f, 0.0f), 1.0f, 1e-5f), "XOR(1,0) = 1");
    TEST_ASSERT(FLOAT_EQ(trix_shape_xor_f32(1.0f, 1.0f), 0.0f, 1e-5f), "XOR(1,1) = 0");

    /* AND truth table */
    TEST_ASSERT(FLOAT_EQ(trix_shape_and_f32(0.0f, 0.0f), 0.0f, 1e-5f), "AND(0,0) = 0");
    TEST_ASSERT(FLOAT_EQ(trix_shape_and_f32(0.0f, 1.0f), 0.0f, 1e-5f), "AND(0,1) = 0");
    TEST_ASSERT(FLOAT_EQ(trix_shape_and_f32(1.0f, 0.0f), 0.0f, 1e-5f), "AND(1,0) = 0");
    TEST_ASSERT(FLOAT_EQ(trix_shape_and_f32(1.0f, 1.0f), 1.0f, 1e-5f), "AND(1,1) = 1");

    /* OR truth table */
    TEST_ASSERT(FLOAT_EQ(trix_shape_or_f32(0.0f, 0.0f), 0.0f, 1e-5f), "OR(0,0) = 0");
    TEST_ASSERT(FLOAT_EQ(trix_shape_or_f32(0.0f, 1.0f), 1.0f, 1e-5f), "OR(0,1) = 1");
    TEST_ASSERT(FLOAT_EQ(trix_shape_or_f32(1.0f, 0.0f), 1.0f, 1e-5f), "OR(1,0) = 1");
    TEST_ASSERT(FLOAT_EQ(trix_shape_or_f32(1.0f, 1.0f), 1.0f, 1e-5f), "OR(1,1) = 1");

    /* NOT truth table */
    TEST_ASSERT(FLOAT_EQ(trix_shape_not_f32(0.0f), 1.0f, 1e-5f), "NOT(0) = 1");
    TEST_ASSERT(FLOAT_EQ(trix_shape_not_f32(1.0f), 0.0f, 1e-5f), "NOT(1) = 0");
}

void test_full_adder(void) {
    printf("\n=== Full Adder Tests ===\n");

    float sum, carry;

    /* 0 + 0 + 0 = 0, carry 0 */
    trix_shape_full_adder(0.0f, 0.0f, 0.0f, &sum, &carry);
    TEST_ASSERT(FLOAT_EQ(sum, 0.0f, 1e-5f) && FLOAT_EQ(carry, 0.0f, 1e-5f),
                "FA(0,0,0) = 0,0");

    /* 1 + 0 + 0 = 1, carry 0 */
    trix_shape_full_adder(1.0f, 0.0f, 0.0f, &sum, &carry);
    TEST_ASSERT(FLOAT_EQ(sum, 1.0f, 1e-5f) && FLOAT_EQ(carry, 0.0f, 1e-5f),
                "FA(1,0,0) = 1,0");

    /* 1 + 1 + 0 = 0, carry 1 */
    trix_shape_full_adder(1.0f, 1.0f, 0.0f, &sum, &carry);
    TEST_ASSERT(FLOAT_EQ(sum, 0.0f, 1e-5f) && FLOAT_EQ(carry, 1.0f, 1e-5f),
                "FA(1,1,0) = 0,1");

    /* 1 + 1 + 1 = 1, carry 1 */
    trix_shape_full_adder(1.0f, 1.0f, 1.0f, &sum, &carry);
    TEST_ASSERT(FLOAT_EQ(sum, 1.0f, 1e-5f) && FLOAT_EQ(carry, 1.0f, 1e-5f),
                "FA(1,1,1) = 1,1");
}

/* ============================================================================
 * ALU TESTS
 * ============================================================================ */

void test_alu_operations(void) {
    printf("\n=== ALU Operation Tests ===\n");

    trix_alu6502_t alu;
    trix_alu6502_init(&alu);

    int c_out;
    uint8_t result;

    /* ADC: 16 + 32 = 48 */
    result = trix_alu6502_execute_int(&alu, ALU_ADC, 16, 32, 0, &c_out);
    TEST_ASSERT(result == 48 && c_out == 0, "ADC: 16 + 32 = 48");

    /* ADC: 255 + 1 = 0 with carry */
    result = trix_alu6502_execute_int(&alu, ALU_ADC, 255, 1, 0, &c_out);
    TEST_ASSERT(result == 0 && c_out == 1, "ADC: 255 + 1 = 0 (carry)");

    /* SBC: 50 - 30 = 20 */
    result = trix_alu6502_execute_int(&alu, ALU_SBC, 50, 30, 0, &c_out);
    TEST_ASSERT(result == 20, "SBC: 50 - 30 = 20");

    /* AND: 0xFF & 0xAA = 0xAA */
    result = trix_alu6502_execute_int(&alu, ALU_AND, 0xFF, 0xAA, 0, &c_out);
    TEST_ASSERT(result == 0xAA, "AND: 0xFF & 0xAA = 0xAA");

    /* ORA: 0xF0 | 0x0F = 0xFF */
    result = trix_alu6502_execute_int(&alu, ALU_ORA, 0xF0, 0x0F, 0, &c_out);
    TEST_ASSERT(result == 0xFF, "ORA: 0xF0 | 0x0F = 0xFF");

    /* EOR: 0xFF ^ 0xAA = 0x55 */
    result = trix_alu6502_execute_int(&alu, ALU_EOR, 0xFF, 0xAA, 0, &c_out);
    TEST_ASSERT(result == 0x55, "EOR: 0xFF ^ 0xAA = 0x55");

    /* ASL: 0x40 << 1 = 0x80 */
    result = trix_alu6502_execute_int(&alu, ALU_ASL, 0x40, 0, 0, &c_out);
    TEST_ASSERT(result == 0x80 && c_out == 0, "ASL: 0x40 << 1 = 0x80");

    /* ASL: 0x80 << 1 = 0x00 with carry */
    result = trix_alu6502_execute_int(&alu, ALU_ASL, 0x80, 0, 0, &c_out);
    TEST_ASSERT(result == 0x00 && c_out == 1, "ASL: 0x80 << 1 = 0x00 (carry)");

    /* LSR: 0x02 >> 1 = 0x01 */
    result = trix_alu6502_execute_int(&alu, ALU_LSR, 0x02, 0, 0, &c_out);
    TEST_ASSERT(result == 0x01 && c_out == 0, "LSR: 0x02 >> 1 = 0x01");

    /* INC: 0x41 + 1 = 0x42 */
    result = trix_alu6502_execute_int(&alu, ALU_INC, 0x41, 0, 0, &c_out);
    TEST_ASSERT(result == 0x42, "INC: 0x41 + 1 = 0x42");

    /* DEC: 0x42 - 1 = 0x41 */
    result = trix_alu6502_execute_int(&alu, ALU_DEC, 0x42, 0, 0, &c_out);
    TEST_ASSERT(result == 0x41, "DEC: 0x42 - 1 = 0x41");

    printf("\n");
    trix_alu6502_print_stats(&alu);
}

/* ============================================================================
 * HAMMING DISTANCE TESTS
 * ============================================================================ */

void test_hamming_distance(void) {
    printf("\n=== Hamming Distance Tests ===\n");

    float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};  /* 0x00 */
    float b[8] = {1, 1, 1, 1, 1, 1, 1, 1};  /* 0xFF */
    float c[8] = {1, 0, 1, 0, 1, 0, 1, 0};  /* 0x55 */

    float dist_ab = trix_shape_hamming_8bit(a, b);
    TEST_ASSERT(FLOAT_EQ(dist_ab, 8.0f, 1e-5f), "hamming(0x00, 0xFF) = 8");

    float dist_aa = trix_shape_hamming_8bit(a, a);
    TEST_ASSERT(FLOAT_EQ(dist_aa, 0.0f, 1e-5f), "hamming(0x00, 0x00) = 0");

    float dist_bc = trix_shape_hamming_8bit(b, c);
    TEST_ASSERT(FLOAT_EQ(dist_bc, 4.0f, 1e-5f), "hamming(0xFF, 0x55) = 4");
}

/* ============================================================================
 * APU CONTEXT TESTS
 * ============================================================================ */

void test_apu_context(void) {
    printf("\n=== APU Context Tests ===\n");

    trix_apu_t apu;
    trix_apu_init(&apu);

    /* Execute some operations */
    float result;
    result = trix_apu_execute(&apu, TRIX_OP_XOR, 1.0f, 0.0f, TRIX_FP32, TRIX_FP32);
    TEST_ASSERT(FLOAT_EQ(result, 1.0f, 1e-5f), "APU XOR(1,0) = 1");

    result = trix_apu_execute(&apu, TRIX_OP_AND, 1.0f, 1.0f, TRIX_FP32, TRIX_FP32);
    TEST_ASSERT(FLOAT_EQ(result, 1.0f, 1e-5f), "APU AND(1,1) = 1");

    result = trix_apu_execute(&apu, TRIX_OP_ADD, 2.0f, 3.0f, TRIX_FP32, TRIX_FP32);
    TEST_ASSERT(FLOAT_EQ(result, 5.0f, 1e-5f), "APU ADD(2,3) = 5");

    TEST_ASSERT(apu.op_counts[TRIX_OP_XOR] == 1, "APU tracked 1 XOR");
    TEST_ASSERT(apu.op_counts[TRIX_OP_AND] == 1, "APU tracked 1 AND");
    TEST_ASSERT(apu.op_counts[TRIX_OP_ADD] == 1, "APU tracked 1 ADD");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(int argc, char** argv) {
    printf("TRIX APU Test Suite\n");
    printf("===================\n");

    test_fp16_conversion();
    test_fp8_conversion();
    test_fp4_conversion();
    test_logic_shapes();
    test_full_adder();
    test_alu_operations();
    test_hamming_distance();
    test_apu_context();

    printf("\n===================\n");
    printf("Results: %d passed, %d failed\n", passes, failures);

    return failures > 0 ? 1 : 0;
}
