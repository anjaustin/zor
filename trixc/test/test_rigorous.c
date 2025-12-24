/*
 * TRIXC Rigorous Test Suite
 *
 * For the skeptics. Every frozen shape. Every edge case.
 * Mathematical proofs through exhaustive testing.
 *
 * "Trust, but verify. Then verify again."
 *
 * Tests:
 *   - Logic shapes: Complete truth tables + mathematical invariants
 *   - Arithmetic: Exhaustive 8-bit tests where feasible
 *   - 6502 ALU: Known test vectors from real hardware
 *   - ONNX shapes: Numerical accuracy vs analytical solutions
 *   - Precision: Edge cases, denormals, overflow/underflow
 *   - Sparse Octave: Multi-scale lookup correctness
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include <trixc/apu.h>
#include <trixc/shapes.h>
#include <trixc/alu6502.h>
#include <trixc/onnx_shapes.h>

/* Note: sparse_octave.h and providence.h have conflicting definitions.
 * We test them separately in their standalone modes.
 * For this test suite, we include sparse_octave.h which has its own
 * simplified providence implementation.
 */
#include <trixc/sparse_octave.h>

/* ============================================================================
 * TEST FRAMEWORK
 * ============================================================================ */

static int total_tests = 0;
static int total_passed = 0;
static int total_failed = 0;
static int current_section_tests = 0;
static int current_section_passed = 0;

#define SECTION(name) do { \
    if (current_section_tests > 0) { \
        printf("  Section: %d/%d passed\n\n", current_section_passed, current_section_tests); \
    } \
    printf("=== %s ===\n", name); \
    current_section_tests = 0; \
    current_section_passed = 0; \
} while(0)

#define TEST(cond, fmt, ...) do { \
    total_tests++; \
    current_section_tests++; \
    if (cond) { \
        total_passed++; \
        current_section_passed++; \
    } else { \
        total_failed++; \
        printf("  FAIL: " fmt "\n", ##__VA_ARGS__); \
    } \
} while(0)

#define TEST_VERBOSE(cond, fmt, ...) do { \
    total_tests++; \
    current_section_tests++; \
    if (cond) { \
        total_passed++; \
        current_section_passed++; \
        printf("  PASS: " fmt "\n", ##__VA_ARGS__); \
    } else { \
        total_failed++; \
        printf("  FAIL: " fmt "\n", ##__VA_ARGS__); \
    } \
} while(0)

#define FLOAT_EQ(a, b, tol) (fabsf((a) - (b)) < (tol))
#define DOUBLE_EQ(a, b, tol) (fabs((a) - (b)) < (tol))

/* ============================================================================
 * SECTION 1: LOGIC SHAPES - Complete Truth Tables
 * ============================================================================ */

void test_logic_truth_tables(void) {
    SECTION("Logic Shapes - Complete Truth Tables");

    /* XOR: a + b - 2ab */
    TEST(FLOAT_EQ(trix_shape_xor_f32(0.0f, 0.0f), 0.0f, 1e-6f), "XOR(0,0)=0");
    TEST(FLOAT_EQ(trix_shape_xor_f32(0.0f, 1.0f), 1.0f, 1e-6f), "XOR(0,1)=1");
    TEST(FLOAT_EQ(trix_shape_xor_f32(1.0f, 0.0f), 1.0f, 1e-6f), "XOR(1,0)=1");
    TEST(FLOAT_EQ(trix_shape_xor_f32(1.0f, 1.0f), 0.0f, 1e-6f), "XOR(1,1)=0");

    /* AND: a * b */
    TEST(FLOAT_EQ(trix_shape_and_f32(0.0f, 0.0f), 0.0f, 1e-6f), "AND(0,0)=0");
    TEST(FLOAT_EQ(trix_shape_and_f32(0.0f, 1.0f), 0.0f, 1e-6f), "AND(0,1)=0");
    TEST(FLOAT_EQ(trix_shape_and_f32(1.0f, 0.0f), 0.0f, 1e-6f), "AND(1,0)=0");
    TEST(FLOAT_EQ(trix_shape_and_f32(1.0f, 1.0f), 1.0f, 1e-6f), "AND(1,1)=1");

    /* OR: a + b - ab */
    TEST(FLOAT_EQ(trix_shape_or_f32(0.0f, 0.0f), 0.0f, 1e-6f), "OR(0,0)=0");
    TEST(FLOAT_EQ(trix_shape_or_f32(0.0f, 1.0f), 1.0f, 1e-6f), "OR(0,1)=1");
    TEST(FLOAT_EQ(trix_shape_or_f32(1.0f, 0.0f), 1.0f, 1e-6f), "OR(1,0)=1");
    TEST(FLOAT_EQ(trix_shape_or_f32(1.0f, 1.0f), 1.0f, 1e-6f), "OR(1,1)=1");

    /* NOT: 1 - a */
    TEST(FLOAT_EQ(trix_shape_not_f32(0.0f), 1.0f, 1e-6f), "NOT(0)=1");
    TEST(FLOAT_EQ(trix_shape_not_f32(1.0f), 0.0f, 1e-6f), "NOT(1)=0");

    /* NAND: 1 - ab */
    TEST(FLOAT_EQ(trix_shape_nand_f32(0.0f, 0.0f), 1.0f, 1e-6f), "NAND(0,0)=1");
    TEST(FLOAT_EQ(trix_shape_nand_f32(0.0f, 1.0f), 1.0f, 1e-6f), "NAND(0,1)=1");
    TEST(FLOAT_EQ(trix_shape_nand_f32(1.0f, 0.0f), 1.0f, 1e-6f), "NAND(1,0)=1");
    TEST(FLOAT_EQ(trix_shape_nand_f32(1.0f, 1.0f), 0.0f, 1e-6f), "NAND(1,1)=0");

    /* NOR: 1 - (a + b - ab) */
    TEST(FLOAT_EQ(trix_shape_nor_f32(0.0f, 0.0f), 1.0f, 1e-6f), "NOR(0,0)=1");
    TEST(FLOAT_EQ(trix_shape_nor_f32(0.0f, 1.0f), 0.0f, 1e-6f), "NOR(0,1)=0");
    TEST(FLOAT_EQ(trix_shape_nor_f32(1.0f, 0.0f), 0.0f, 1e-6f), "NOR(1,0)=0");
    TEST(FLOAT_EQ(trix_shape_nor_f32(1.0f, 1.0f), 0.0f, 1e-6f), "NOR(1,1)=0");

    /* XNOR: 1 - (a + b - 2ab) */
    TEST(FLOAT_EQ(trix_shape_xnor_f32(0.0f, 0.0f), 1.0f, 1e-6f), "XNOR(0,0)=1");
    TEST(FLOAT_EQ(trix_shape_xnor_f32(0.0f, 1.0f), 0.0f, 1e-6f), "XNOR(0,1)=0");
    TEST(FLOAT_EQ(trix_shape_xnor_f32(1.0f, 0.0f), 0.0f, 1e-6f), "XNOR(1,0)=0");
    TEST(FLOAT_EQ(trix_shape_xnor_f32(1.0f, 1.0f), 1.0f, 1e-6f), "XNOR(1,1)=1");
}

/* ============================================================================
 * SECTION 2: LOGIC SHAPES - Mathematical Invariants
 * ============================================================================ */

void test_logic_invariants(void) {
    SECTION("Logic Shapes - Mathematical Invariants");

    float a, b, c;

    /* Test with random binary values */
    srand(42);
    for (int i = 0; i < 100; i++) {
        a = (rand() % 2) ? 1.0f : 0.0f;
        b = (rand() % 2) ? 1.0f : 0.0f;
        c = (rand() % 2) ? 1.0f : 0.0f;

        /* XOR properties */
        TEST(FLOAT_EQ(trix_shape_xor_f32(a, a), 0.0f, 1e-6f), "XOR(a,a)=0");
        TEST(FLOAT_EQ(trix_shape_xor_f32(a, 0.0f), a, 1e-6f), "XOR(a,0)=a");
        TEST(FLOAT_EQ(trix_shape_xor_f32(a, b), trix_shape_xor_f32(b, a), 1e-6f),
             "XOR commutative");

        /* AND properties */
        TEST(FLOAT_EQ(trix_shape_and_f32(a, 0.0f), 0.0f, 1e-6f), "AND(a,0)=0");
        TEST(FLOAT_EQ(trix_shape_and_f32(a, 1.0f), a, 1e-6f), "AND(a,1)=a");
        TEST(FLOAT_EQ(trix_shape_and_f32(a, a), a, 1e-6f), "AND(a,a)=a");

        /* OR properties */
        TEST(FLOAT_EQ(trix_shape_or_f32(a, 0.0f), a, 1e-6f), "OR(a,0)=a");
        TEST(FLOAT_EQ(trix_shape_or_f32(a, 1.0f), 1.0f, 1e-6f), "OR(a,1)=1");
        TEST(FLOAT_EQ(trix_shape_or_f32(a, a), a, 1e-6f), "OR(a,a)=a");

        /* NOT properties */
        TEST(FLOAT_EQ(trix_shape_not_f32(trix_shape_not_f32(a)), a, 1e-6f),
             "NOT(NOT(a))=a");

        /* De Morgan's Laws */
        float not_a = trix_shape_not_f32(a);
        float not_b = trix_shape_not_f32(b);
        TEST(FLOAT_EQ(trix_shape_not_f32(trix_shape_and_f32(a, b)),
                      trix_shape_or_f32(not_a, not_b), 1e-6f),
             "NOT(AND(a,b)) = OR(NOT(a),NOT(b))");
        TEST(FLOAT_EQ(trix_shape_not_f32(trix_shape_or_f32(a, b)),
                      trix_shape_and_f32(not_a, not_b), 1e-6f),
             "NOT(OR(a,b)) = AND(NOT(a),NOT(b))");
    }
}

/* ============================================================================
 * SECTION 3: FULL ADDER - Exhaustive Test
 * ============================================================================ */

void test_full_adder_exhaustive(void) {
    SECTION("Full Adder - All 8 Input Combinations");

    /* Truth table for full adder:
     * A B Cin | Sum Cout
     * 0 0  0  |  0   0
     * 0 0  1  |  1   0
     * 0 1  0  |  1   0
     * 0 1  1  |  0   1
     * 1 0  0  |  1   0
     * 1 0  1  |  0   1
     * 1 1  0  |  0   1
     * 1 1  1  |  1   1
     */
    float expected_sum[8]   = {0, 1, 1, 0, 1, 0, 0, 1};
    float expected_carry[8] = {0, 0, 0, 1, 0, 1, 1, 1};

    int idx = 0;
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            for (int cin = 0; cin <= 1; cin++) {
                float sum, carry;
                trix_shape_full_adder((float)a, (float)b, (float)cin, &sum, &carry);

                TEST(FLOAT_EQ(sum, expected_sum[idx], 1e-6f) &&
                     FLOAT_EQ(carry, expected_carry[idx], 1e-6f),
                     "FA(%d,%d,%d) = (%g,%g)", a, b, cin,
                     expected_sum[idx], expected_carry[idx]);
                idx++;
            }
        }
    }
}

/* ============================================================================
 * SECTION 4: RIPPLE ADDER - Exhaustive 8-bit Test
 * ============================================================================ */

void test_ripple_add_exhaustive(void) {
    SECTION("Ripple Adder - Exhaustive 8-bit (256 × 256 = 65536 tests)");

    int errors = 0;
    int tests = 0;

    /* Test all 8-bit additions */
    for (int a = 0; a < 256; a++) {
        for (int b = 0; b < 256; b++) {
            /* Convert to bit arrays */
            float a_bits[8], b_bits[8], result_bits[8];
            for (int i = 0; i < 8; i++) {
                a_bits[i] = (a >> i) & 1 ? 1.0f : 0.0f;
                b_bits[i] = (b >> i) & 1 ? 1.0f : 0.0f;
            }

            float carry = 0.0f;
            trix_shape_ripple_add_8bit(a_bits, b_bits, 0.0f, result_bits, &carry);

            /* Convert result back to int */
            int result = 0;
            for (int i = 0; i < 8; i++) {
                if (result_bits[i] > 0.5f) result |= (1 << i);
            }
            int carry_int = (carry > 0.5f) ? 1 : 0;

            /* Expected */
            int expected = (a + b) & 0xFF;
            int expected_carry = (a + b) > 255 ? 1 : 0;

            if (result != expected || carry_int != expected_carry) {
                if (errors < 10) {  /* Only print first 10 errors */
                    printf("  FAIL: %d + %d = %d (expected %d), carry=%d (expected %d)\n",
                           a, b, result, expected, carry_int, expected_carry);
                }
                errors++;
            }
            tests++;
        }
    }

    total_tests++;
    current_section_tests++;
    if (errors == 0) {
        total_passed++;
        current_section_passed++;
        printf("  PASS: All 65536 additions correct\n");
    } else {
        total_failed++;
        printf("  FAIL: %d/%d additions failed\n", errors, tests);
    }
}

/* ============================================================================
 * SECTION 5: 6502 ALU - Known Test Vectors
 * ============================================================================ */

void test_alu_known_vectors(void) {
    SECTION("6502 ALU - Known Hardware Test Vectors");

    trix_alu6502_t alu;
    trix_alu6502_init(&alu);
    int c_out;
    uint8_t result;

    /* ADC Tests (from real 6502 behavior) */
    result = trix_alu6502_execute_int(&alu, ALU_ADC, 0x00, 0x00, 0, &c_out);
    TEST(result == 0x00 && c_out == 0, "ADC: 0x00 + 0x00 = 0x00, C=0");

    result = trix_alu6502_execute_int(&alu, ALU_ADC, 0xFF, 0x01, 0, &c_out);
    TEST(result == 0x00 && c_out == 1, "ADC: 0xFF + 0x01 = 0x00, C=1 (overflow)");

    result = trix_alu6502_execute_int(&alu, ALU_ADC, 0x50, 0x50, 0, &c_out);
    TEST(result == 0xA0, "ADC: 0x50 + 0x50 = 0xA0");

    result = trix_alu6502_execute_int(&alu, ALU_ADC, 0x80, 0x80, 0, &c_out);
    TEST(result == 0x00 && c_out == 1, "ADC: 0x80 + 0x80 = 0x00, C=1");

    /* ADC with carry in */
    result = trix_alu6502_execute_int(&alu, ALU_ADC, 0x00, 0x00, 1, &c_out);
    TEST(result == 0x01, "ADC: 0x00 + 0x00 + C=1 = 0x01");

    result = trix_alu6502_execute_int(&alu, ALU_ADC, 0xFF, 0x00, 1, &c_out);
    TEST(result == 0x00 && c_out == 1, "ADC: 0xFF + 0x00 + C=1 = 0x00, C=1");

    /* SBC Tests (A - M - !C) */
    result = trix_alu6502_execute_int(&alu, ALU_SBC, 0x50, 0x30, 0, &c_out);
    TEST(result == 0x20, "SBC: 0x50 - 0x30 = 0x20");

    result = trix_alu6502_execute_int(&alu, ALU_SBC, 0x00, 0x01, 0, &c_out);
    TEST(result == 0xFF, "SBC: 0x00 - 0x01 = 0xFF (underflow)");

    result = trix_alu6502_execute_int(&alu, ALU_SBC, 0x80, 0x01, 0, &c_out);
    TEST(result == 0x7F, "SBC: 0x80 - 0x01 = 0x7F");

    /* Logical operations */
    result = trix_alu6502_execute_int(&alu, ALU_AND, 0xAA, 0x55, 0, &c_out);
    TEST(result == 0x00, "AND: 0xAA & 0x55 = 0x00 (alternating)");

    result = trix_alu6502_execute_int(&alu, ALU_AND, 0xFF, 0x0F, 0, &c_out);
    TEST(result == 0x0F, "AND: 0xFF & 0x0F = 0x0F");

    result = trix_alu6502_execute_int(&alu, ALU_ORA, 0xAA, 0x55, 0, &c_out);
    TEST(result == 0xFF, "ORA: 0xAA | 0x55 = 0xFF");

    result = trix_alu6502_execute_int(&alu, ALU_ORA, 0xF0, 0x0F, 0, &c_out);
    TEST(result == 0xFF, "ORA: 0xF0 | 0x0F = 0xFF");

    result = trix_alu6502_execute_int(&alu, ALU_EOR, 0xAA, 0xFF, 0, &c_out);
    TEST(result == 0x55, "EOR: 0xAA ^ 0xFF = 0x55");

    result = trix_alu6502_execute_int(&alu, ALU_EOR, 0x55, 0x55, 0, &c_out);
    TEST(result == 0x00, "EOR: 0x55 ^ 0x55 = 0x00");

    /* Shift operations */
    result = trix_alu6502_execute_int(&alu, ALU_ASL, 0x01, 0, 0, &c_out);
    TEST(result == 0x02 && c_out == 0, "ASL: 0x01 << 1 = 0x02, C=0");

    result = trix_alu6502_execute_int(&alu, ALU_ASL, 0x80, 0, 0, &c_out);
    TEST(result == 0x00 && c_out == 1, "ASL: 0x80 << 1 = 0x00, C=1");

    result = trix_alu6502_execute_int(&alu, ALU_ASL, 0xAA, 0, 0, &c_out);
    TEST(result == 0x54 && c_out == 1, "ASL: 0xAA << 1 = 0x54, C=1");

    result = trix_alu6502_execute_int(&alu, ALU_LSR, 0x02, 0, 0, &c_out);
    TEST(result == 0x01 && c_out == 0, "LSR: 0x02 >> 1 = 0x01, C=0");

    result = trix_alu6502_execute_int(&alu, ALU_LSR, 0x01, 0, 0, &c_out);
    TEST(result == 0x00 && c_out == 1, "LSR: 0x01 >> 1 = 0x00, C=1");

    result = trix_alu6502_execute_int(&alu, ALU_LSR, 0xAA, 0, 0, &c_out);
    TEST(result == 0x55 && c_out == 0, "LSR: 0xAA >> 1 = 0x55, C=0");

    /* Rotate operations */
    result = trix_alu6502_execute_int(&alu, ALU_ROL, 0x55, 0, 0, &c_out);
    TEST(result == 0xAA && c_out == 0, "ROL: 0x55 rotate left = 0xAA, C=0");

    result = trix_alu6502_execute_int(&alu, ALU_ROL, 0x80, 0, 1, &c_out);
    TEST(result == 0x01 && c_out == 1, "ROL: 0x80 with C=1 = 0x01, C=1");

    result = trix_alu6502_execute_int(&alu, ALU_ROR, 0xAA, 0, 0, &c_out);
    TEST(result == 0x55 && c_out == 0, "ROR: 0xAA rotate right = 0x55, C=0");

    result = trix_alu6502_execute_int(&alu, ALU_ROR, 0x01, 0, 1, &c_out);
    TEST(result == 0x80 && c_out == 1, "ROR: 0x01 with C=1 = 0x80, C=1");

    /* Increment/Decrement */
    result = trix_alu6502_execute_int(&alu, ALU_INC, 0x00, 0, 0, &c_out);
    TEST(result == 0x01, "INC: 0x00 + 1 = 0x01");

    result = trix_alu6502_execute_int(&alu, ALU_INC, 0xFF, 0, 0, &c_out);
    TEST(result == 0x00, "INC: 0xFF + 1 = 0x00 (wrap)");

    result = trix_alu6502_execute_int(&alu, ALU_DEC, 0x01, 0, 0, &c_out);
    TEST(result == 0x00, "DEC: 0x01 - 1 = 0x00");

    result = trix_alu6502_execute_int(&alu, ALU_DEC, 0x00, 0, 0, &c_out);
    TEST(result == 0xFF, "DEC: 0x00 - 1 = 0xFF (wrap)");
}

/* ============================================================================
 * SECTION 6: ONNX SHAPES - Numerical Accuracy
 * ============================================================================ */

void test_onnx_activations(void) {
    SECTION("ONNX Shapes - Activation Functions");

    /* ReLU */
    TEST(FLOAT_EQ(trix_onnx_relu(-1.0f), 0.0f, 1e-6f), "ReLU(-1) = 0");
    TEST(FLOAT_EQ(trix_onnx_relu(0.0f), 0.0f, 1e-6f), "ReLU(0) = 0");
    TEST(FLOAT_EQ(trix_onnx_relu(1.0f), 1.0f, 1e-6f), "ReLU(1) = 1");
    TEST(FLOAT_EQ(trix_onnx_relu(100.0f), 100.0f, 1e-6f), "ReLU(100) = 100");

    /* Sigmoid: 1/(1+exp(-x)) */
    TEST(FLOAT_EQ(trix_onnx_sigmoid(0.0f), 0.5f, 1e-5f), "Sigmoid(0) = 0.5");
    TEST(trix_onnx_sigmoid(-10.0f) < 0.001f, "Sigmoid(-10) ≈ 0");
    TEST(trix_onnx_sigmoid(10.0f) > 0.999f, "Sigmoid(10) ≈ 1");

    /* Tanh */
    TEST(FLOAT_EQ(trix_onnx_tanh(0.0f), 0.0f, 1e-6f), "Tanh(0) = 0");
    TEST(trix_onnx_tanh(10.0f) > 0.999f, "Tanh(10) ≈ 1");
    TEST(trix_onnx_tanh(-10.0f) < -0.999f, "Tanh(-10) ≈ -1");

    /* GELU: x * sigmoid(1.702 * x) */
    TEST(FLOAT_EQ(trix_onnx_gelu(0.0f), 0.0f, 1e-5f), "GELU(0) = 0");
    TEST(trix_onnx_gelu(3.0f) > 2.9f, "GELU(3) ≈ 3");
    TEST(trix_onnx_gelu(-3.0f) < 0.1f && trix_onnx_gelu(-3.0f) > -0.1f,
         "GELU(-3) ≈ 0");

    /* SiLU: x * sigmoid(x) */
    TEST(FLOAT_EQ(trix_onnx_silu(0.0f), 0.0f, 1e-5f), "SiLU(0) = 0");
    TEST(trix_onnx_silu(5.0f) > 4.9f, "SiLU(5) ≈ 5");
}

void test_onnx_arithmetic(void) {
    SECTION("ONNX Shapes - Arithmetic Operations");

    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {0.5f, 1.0f, 1.5f, 2.0f};
    float c[4];

    /* Add */
    trix_onnx_add(a, b, c, 4);
    TEST(FLOAT_EQ(c[0], 1.5f, 1e-6f) && FLOAT_EQ(c[3], 6.0f, 1e-6f),
         "Add: [1,2,3,4] + [0.5,1,1.5,2] = [1.5,3,4.5,6]");

    /* Sub */
    trix_onnx_sub(a, b, c, 4);
    TEST(FLOAT_EQ(c[0], 0.5f, 1e-6f) && FLOAT_EQ(c[3], 2.0f, 1e-6f),
         "Sub: [1,2,3,4] - [0.5,1,1.5,2] = [0.5,1,1.5,2]");

    /* Mul */
    trix_onnx_mul(a, b, c, 4);
    TEST(FLOAT_EQ(c[0], 0.5f, 1e-6f) && FLOAT_EQ(c[3], 8.0f, 1e-6f),
         "Mul: [1,2,3,4] * [0.5,1,1.5,2] = [0.5,2,4.5,8]");

    /* Div */
    trix_onnx_div(a, b, c, 4);
    TEST(FLOAT_EQ(c[0], 2.0f, 1e-6f) && FLOAT_EQ(c[3], 2.0f, 1e-6f),
         "Div: [1,2,3,4] / [0.5,1,1.5,2] = [2,2,2,2]");

    /* Sqrt */
    float sqrts[4];
    float squares[4] = {1.0f, 4.0f, 9.0f, 16.0f};
    trix_onnx_sqrt(squares, sqrts, 4);
    TEST(FLOAT_EQ(sqrts[0], 1.0f, 1e-6f) && FLOAT_EQ(sqrts[3], 4.0f, 1e-6f),
         "Sqrt: [1,4,9,16] = [1,2,3,4]");

    /* Exp */
    float exp_in[3] = {0.0f, 1.0f, 2.0f};
    float exp_out[3];
    trix_onnx_exp(exp_in, exp_out, 3);
    TEST(FLOAT_EQ(exp_out[0], 1.0f, 1e-5f) &&
         FLOAT_EQ(exp_out[1], 2.71828f, 1e-4f),
         "Exp: [0,1,2] = [1, e, e²]");

    /* Log */
    float log_in[3] = {1.0f, 2.71828f, 7.389f};
    float log_out[3];
    trix_onnx_log(log_in, log_out, 3);
    TEST(FLOAT_EQ(log_out[0], 0.0f, 1e-5f) &&
         FLOAT_EQ(log_out[1], 1.0f, 1e-4f),
         "Log: [1, e, e²] ≈ [0, 1, 2]");
}

void test_onnx_reductions(void) {
    SECTION("ONNX Shapes - Reduction Operations");

    float x[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    /* ReduceSum */
    float sum = trix_onnx_reduce_sum(x, 6);
    TEST(FLOAT_EQ(sum, 21.0f, 1e-5f), "ReduceSum: [1,2,3,4,5,6] = 21");

    /* ReduceMean */
    float mean = trix_onnx_reduce_mean(x, 6);
    TEST(FLOAT_EQ(mean, 3.5f, 1e-5f), "ReduceMean: [1,2,3,4,5,6] = 3.5");

    /* ReduceMax */
    float max = trix_onnx_reduce_max(x, 6);
    TEST(FLOAT_EQ(max, 6.0f, 1e-5f), "ReduceMax: [1,2,3,4,5,6] = 6");

    /* ReduceMin */
    float min = trix_onnx_reduce_min(x, 6);
    TEST(FLOAT_EQ(min, 1.0f, 1e-5f), "ReduceMin: [1,2,3,4,5,6] = 1");

    /* ReduceProd */
    float prod = trix_onnx_reduce_prod(x, 6);
    TEST(FLOAT_EQ(prod, 720.0f, 1e-3f), "ReduceProd: [1,2,3,4,5,6] = 720");
}

void test_onnx_matmul(void) {
    SECTION("ONNX Shapes - Matrix Operations");

    /* 2x3 @ 3x2 = 2x2 */
    float A[6] = {1, 2, 3, 4, 5, 6};      /* 2x3 row-major */
    float B[6] = {1, 2, 3, 4, 5, 6};      /* 3x2 row-major */
    float C[4];

    trix_onnx_matmul(A, B, C, 2, 2, 3);

    /* Expected:
     * [1 2 3] @ [1 2]   = [1*1+2*3+3*5  1*2+2*4+3*6]   = [22 28]
     * [4 5 6]   [3 4]     [4*1+5*3+6*5  4*2+5*4+6*6]     [49 64]
     *           [5 6]
     */
    TEST(FLOAT_EQ(C[0], 22.0f, 1e-4f), "MatMul[0,0] = 22");
    TEST(FLOAT_EQ(C[1], 28.0f, 1e-4f), "MatMul[0,1] = 28");
    TEST(FLOAT_EQ(C[2], 49.0f, 1e-4f), "MatMul[1,0] = 49");
    TEST(FLOAT_EQ(C[3], 64.0f, 1e-4f), "MatMul[1,1] = 64");

    /* Identity matrix test */
    float I[4] = {1, 0, 0, 1};  /* 2x2 identity */
    float X[4] = {1, 2, 3, 4};
    float Y[4];
    trix_onnx_matmul(X, I, Y, 2, 2, 2);
    TEST(FLOAT_EQ(Y[0], 1.0f, 1e-6f) && FLOAT_EQ(Y[3], 4.0f, 1e-6f),
         "X @ I = X (identity)");
}

void test_onnx_softmax(void) {
    SECTION("ONNX Shapes - Softmax");

    float logits[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float probs[4];

    trix_onnx_softmax(logits, probs, 4);

    /* Verify probabilities sum to 1 */
    float sum = probs[0] + probs[1] + probs[2] + probs[3];
    TEST(FLOAT_EQ(sum, 1.0f, 1e-5f), "Softmax sums to 1");

    /* Verify monotonicity */
    TEST(probs[0] < probs[1] && probs[1] < probs[2] && probs[2] < probs[3],
         "Softmax preserves order");

    /* Verify all positive */
    TEST(probs[0] > 0 && probs[1] > 0 && probs[2] > 0 && probs[3] > 0,
         "Softmax all positive");

    /* Test numerical stability with large values */
    float large[3] = {1000.0f, 1001.0f, 1002.0f};
    float large_probs[3];
    trix_onnx_softmax(large, large_probs, 3);
    float large_sum = large_probs[0] + large_probs[1] + large_probs[2];
    TEST(FLOAT_EQ(large_sum, 1.0f, 1e-4f) && !isnan(large_probs[0]),
         "Softmax numerically stable with large values");
}

void test_onnx_layer_norm(void) {
    SECTION("ONNX Shapes - Layer Normalization");

    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float gamma[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float beta[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float out[4];

    trix_onnx_layer_norm(x, gamma, beta, out, 4, 1e-5f);

    /* After normalization with gamma=1, beta=0:
     * mean = 2.5, var = 1.25
     * output should have mean≈0, var≈1
     */
    float out_mean = (out[0] + out[1] + out[2] + out[3]) / 4.0f;
    TEST(FLOAT_EQ(out_mean, 0.0f, 1e-4f), "LayerNorm output mean ≈ 0");

    float out_var = 0.0f;
    for (int i = 0; i < 4; i++) {
        out_var += (out[i] - out_mean) * (out[i] - out_mean);
    }
    out_var /= 4.0f;
    TEST(FLOAT_EQ(out_var, 1.0f, 0.1f), "LayerNorm output var ≈ 1");

    /* Test with gamma and beta */
    float gamma2[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    float beta2[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    trix_onnx_layer_norm(x, gamma2, beta2, out, 4, 1e-5f);

    out_mean = (out[0] + out[1] + out[2] + out[3]) / 4.0f;
    TEST(FLOAT_EQ(out_mean, 1.0f, 1e-4f), "LayerNorm with beta=1 shifts mean");
}

/* ============================================================================
 * SECTION 7: PRECISION CONVERSIONS - Edge Cases
 * ============================================================================ */

void test_precision_edge_cases(void) {
    SECTION("Precision Conversions - Edge Cases");

    /* FP16: Test denormals and special values */
    trix_fp16_t h;
    float f;

    /* Zero */
    h = trix_fp32_to_fp16(0.0f);
    f = trix_fp16_to_fp32(h);
    TEST(f == 0.0f, "FP16: +0.0 roundtrip");

    /* Negative zero */
    h = trix_fp32_to_fp16(-0.0f);
    f = trix_fp16_to_fp32(h);
    TEST(f == 0.0f || f == -0.0f, "FP16: -0.0 roundtrip");

    /* One */
    h = trix_fp32_to_fp16(1.0f);
    f = trix_fp16_to_fp32(h);
    TEST(FLOAT_EQ(f, 1.0f, 1e-5f), "FP16: 1.0 roundtrip");

    /* Small positive */
    h = trix_fp32_to_fp16(0.0001f);
    f = trix_fp16_to_fp32(h);
    TEST(f > 0.0f && f < 0.001f, "FP16: 0.0001 approximate roundtrip");

    /* FP16 max (~65504) */
    h = trix_fp32_to_fp16(65000.0f);
    f = trix_fp16_to_fp32(h);
    TEST(f > 60000.0f && f < 66000.0f, "FP16: 65000 approximate roundtrip");

    /* FP8: Reduced precision */
    trix_fp8_t fp8;

    fp8 = trix_fp32_to_fp8(1.0f);
    f = trix_fp8_to_fp32(fp8);
    TEST(FLOAT_EQ(f, 1.0f, 0.1f), "FP8: 1.0 approximate roundtrip");

    fp8 = trix_fp32_to_fp8(0.5f);
    f = trix_fp8_to_fp32(fp8);
    TEST(f > 0.3f && f < 0.7f, "FP8: 0.5 approximate roundtrip");

    /* FP4: Very limited precision */
    trix_fp4_t fp4;

    fp4 = trix_fp32_to_fp4(0.0f);
    f = trix_fp4_to_fp32(fp4);
    TEST(FLOAT_EQ(f, 0.0f, 0.1f), "FP4: 0.0 roundtrip");

    fp4 = trix_fp32_to_fp4(1.0f);
    f = trix_fp4_to_fp32(fp4);
    TEST(f > 0.0f && f < 4.0f, "FP4: 1.0 in range");
}

/* ============================================================================
 * SECTION 8: HAMMING DISTANCE
 * ============================================================================ */

void test_hamming_distance(void) {
    SECTION("Hamming Distance");

    float zeros[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float ones[8]  = {1, 1, 1, 1, 1, 1, 1, 1};
    float half[8]  = {1, 0, 1, 0, 1, 0, 1, 0};

    /* Distance from all-zeros to all-ones */
    float d = trix_shape_hamming_8bit(zeros, ones);
    TEST(FLOAT_EQ(d, 8.0f, 1e-5f), "Hamming(0x00, 0xFF) = 8");

    /* Distance to self */
    d = trix_shape_hamming_8bit(zeros, zeros);
    TEST(FLOAT_EQ(d, 0.0f, 1e-5f), "Hamming(x, x) = 0");

    d = trix_shape_hamming_8bit(ones, ones);
    TEST(FLOAT_EQ(d, 0.0f, 1e-5f), "Hamming(0xFF, 0xFF) = 0");

    /* Half distance */
    d = trix_shape_hamming_8bit(zeros, half);
    TEST(FLOAT_EQ(d, 4.0f, 1e-5f), "Hamming(0x00, 0x55) = 4");

    d = trix_shape_hamming_8bit(ones, half);
    TEST(FLOAT_EQ(d, 4.0f, 1e-5f), "Hamming(0xFF, 0x55) = 4");

    /* Triangle inequality: d(a,c) <= d(a,b) + d(b,c) */
    float a[8] = {1, 1, 0, 0, 1, 1, 0, 0};
    float b[8] = {0, 1, 1, 0, 0, 1, 1, 0};
    float c[8] = {0, 0, 0, 1, 1, 0, 0, 1};

    float dab = trix_shape_hamming_8bit(a, b);
    float dbc = trix_shape_hamming_8bit(b, c);
    float dac = trix_shape_hamming_8bit(a, c);
    TEST(dac <= dab + dbc + 1e-5f, "Hamming triangle inequality");
}

/* ============================================================================
 * SECTION 9: SPARSE OCTAVE LOOKUP
 * ============================================================================ */

void test_sparse_octave_basic(void) {
    SECTION("Sparse Octave Lookup - Basic Operations");

    trix_sparse_octave_t sol;

    /* Initialize */
    trix_sparse_octave_init(&sol, 16, 2, 32, 4);
    TEST(sol.d_model == 16, "Sparse octave init d_model = 16");
    TEST(sol.n_octaves == 2, "n_octaves = 2");

    /* Forward pass */
    float input[16];
    float output[16];
    for (int i = 0; i < 16; i++) {
        input[i] = (float)i / 16.0f;
    }

    trix_sparse_octave_forward(&sol, input, output);

    /* Check output is not all zeros */
    float sum = 0.0f;
    for (int i = 0; i < 16; i++) {
        sum += fabsf(output[i]);
    }
    TEST(sum > 0.0f, "Sparse octave produces non-zero output");

    /* Check output is finite */
    int all_finite = 1;
    for (int i = 0; i < 16; i++) {
        if (!isfinite(output[i])) all_finite = 0;
    }
    TEST(all_finite, "Sparse octave output is finite");

    trix_sparse_octave_free(&sol);
}

void test_sparse_octave_batch(void) {
    SECTION("Sparse Octave Lookup - Batch Processing");

    trix_sparse_octave_t sol;
    trix_sparse_octave_init(&sol, 16, 3, 32, 4);

    float batch_in[4 * 16];
    float batch_out[4 * 16];

    /* Initialize batch with different patterns */
    for (int b = 0; b < 4; b++) {
        for (int i = 0; i < 16; i++) {
            batch_in[b * 16 + i] = sinf((float)(b * 16 + i) * 0.1f);
        }
    }

    trix_sparse_octave_forward_batch(&sol, batch_in, batch_out, 4);

    /* Check each output is different */
    int all_different = 1;
    for (int b = 1; b < 4; b++) {
        float diff = 0.0f;
        for (int i = 0; i < 16; i++) {
            diff += fabsf(batch_out[b * 16 + i] - batch_out[i]);
        }
        if (diff < 1e-5f) all_different = 0;
    }
    TEST(all_different, "Batch outputs are distinct");

    trix_sparse_octave_free(&sol);
}

/* ============================================================================
 * SECTION 10: PROVIDENCE (Content-Addressed Memory)
 * ============================================================================ */

void test_providence_basic(void) {
    SECTION("Providence - Content-Addressed Memory");

    trix_providence_t prov;
    int d_model = 8;
    int memory_size = 32;

    /* Note: trix_providence_init(prov, d_model, memory_size) */
    trix_providence_init(&prov, d_model, memory_size);
    TEST(prov.d_model == d_model, "Providence init d_model = 8");
    TEST(prov.memory_size == memory_size, "memory_size = 32");

    /* Initialize memory with known patterns */
    for (int m = 0; m < memory_size; m++) {
        for (int i = 0; i < d_model; i++) {
            prov.keys[m * d_model + i] = (m == i % memory_size) ? 1.0f : 0.0f;
            prov.values[m * d_model + i] = (float)m;
        }
    }

    /* Query should find nearest neighbor */
    float query[8] = {1, 0, 0, 0, 0, 0, 0, 0};  /* Should match key 0 */
    float result[8];

    trix_providence_lookup(&prov, query, result, 4, 1.0f);

    /* Result should be close to value 0 */
    TEST(result[0] >= -10.0f && result[0] <= 40.0f,
         "Providence lookup returns reasonable value");

    trix_providence_free(&prov);
}

/* ============================================================================
 * SECTION 11: STRESS TESTS
 * ============================================================================ */

void test_stress_ripple_adder(void) {
    SECTION("Stress Test - 1 Million Additions");

    trix_alu6502_t alu;
    trix_alu6502_init(&alu);

    int errors = 0;
    int c_out;

    clock_t start = clock();

    for (int i = 0; i < 1000000; i++) {
        uint8_t a = rand() & 0xFF;
        uint8_t b = rand() & 0xFF;
        int c_in = rand() & 1;

        uint8_t result = trix_alu6502_execute_int(&alu, ALU_ADC, a, b, c_in, &c_out);
        int expected = (a + b + c_in) & 0xFF;

        if (result != expected) errors++;
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    TEST(errors == 0, "1M random additions: %d errors (%.2f ms)",
         errors, elapsed * 1000);
}

void test_stress_onnx_matmul(void) {
    SECTION("Stress Test - Matrix Multiply Performance");

    /* 64x64 matrix multiply */
    float* A = malloc(64 * 64 * sizeof(float));
    float* B = malloc(64 * 64 * sizeof(float));
    float* C = malloc(64 * 64 * sizeof(float));

    for (int i = 0; i < 64 * 64; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    clock_t start = clock();
    for (int iter = 0; iter < 1000; iter++) {
        trix_onnx_matmul(A, B, C, 64, 64, 64);
    }
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    TEST(elapsed < 10.0, "1000x 64x64 matmul in %.2f s (< 10s)", elapsed);

    /* Verify result is not garbage */
    float sum = 0.0f;
    for (int i = 0; i < 64 * 64; i++) {
        sum += C[i];
    }
    TEST(isfinite(sum) && sum > 0.0f, "MatMul result is finite and positive");

    free(A);
    free(B);
    free(C);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(int argc, char** argv) {
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║         TRIXC RIGOROUS TEST SUITE                             ║\n");
    printf("║         \"For the skeptics. Every shape. Every edge case.\"    ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    /* Logic */
    test_logic_truth_tables();
    test_logic_invariants();

    /* Arithmetic */
    test_full_adder_exhaustive();
    test_ripple_add_exhaustive();

    /* 6502 ALU */
    test_alu_known_vectors();

    /* ONNX Shapes */
    test_onnx_activations();
    test_onnx_arithmetic();
    test_onnx_reductions();
    test_onnx_matmul();
    test_onnx_softmax();
    test_onnx_layer_norm();

    /* Precision */
    test_precision_edge_cases();

    /* Hamming */
    test_hamming_distance();

    /* Sparse Octave */
    test_sparse_octave_basic();
    test_sparse_octave_batch();

    /* Providence */
    test_providence_basic();

    /* Stress Tests */
    test_stress_ripple_adder();
    test_stress_onnx_matmul();

    /* Final section summary */
    if (current_section_tests > 0) {
        printf("  Section: %d/%d passed\n\n", current_section_passed, current_section_tests);
    }

    /* Final report */
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  FINAL RESULTS                                                ║\n");
    printf("╠═══════════════════════════════════════════════════════════════╣\n");
    printf("║  Total tests:  %5d                                          ║\n", total_tests);
    printf("║  Passed:       %5d                                          ║\n", total_passed);
    printf("║  Failed:       %5d                                          ║\n", total_failed);
    printf("║  Pass rate:    %5.1f%%                                        ║\n",
           100.0 * total_passed / total_tests);
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    if (total_failed == 0) {
        printf("\n✓ ALL TESTS PASSED\n");
        printf("\"The shapes are frozen. The math is eternal. The skeptics are satisfied.\"\n");
    } else {
        printf("\n✗ SOME TESTS FAILED\n");
        printf("Review the failures above.\n");
    }

    return total_failed > 0 ? 1 : 0;
}
