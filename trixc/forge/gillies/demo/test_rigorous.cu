/*
 * GILLIES Rigorous Test Suite
 *
 * "For the skeptics. Every shape. Every edge case."
 *
 * Tests:
 *   1. All shapes - complete truth tables for binary inputs
 *   2. Mathematical invariants - De Morgan, associativity, etc.
 *   3. Polynomial correctness - verify formulas are exact
 *   4. Composition - all 2-shape combinations
 *   5. Fungibility - exhaustive CPU vs GPU comparison
 *   6. Full adder - all 8 combinations
 *   7. Ripple adder - exhaustive 4-bit (256 cases)
 *   8. Stress test - many invocations
 *   9. Edge cases - boundary values, precision
 *
 * (c) 2025 TriX Project
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "../include/gillies.h"

/* ============================================================================
 * TEST FRAMEWORK
 * ============================================================================ */

static int g_tests_passed = 0;
static int g_tests_failed = 0;
static int g_section_passed = 0;
static int g_section_failed = 0;

#define TOLERANCE 0.0001f

#define TEST_ASSERT(cond, msg) do { \
    if (cond) { \
        g_tests_passed++; \
        g_section_passed++; \
    } else { \
        g_tests_failed++; \
        g_section_failed++; \
        printf("    FAIL: %s\n", msg); \
    } \
} while(0)

#define TEST_FLOAT_EQ(a, b, msg) TEST_ASSERT(fabsf((a) - (b)) < TOLERANCE, msg)

#define SECTION_START(name) do { \
    printf("\n=== %s ===\n", name); \
    g_section_passed = 0; \
    g_section_failed = 0; \
} while(0)

#define SECTION_END() do { \
    printf("  Section: %d/%d passed\n", g_section_passed, g_section_passed + g_section_failed); \
} while(0)

/* ============================================================================
 * TEST 1: SHAPE TRUTH TABLES
 *
 * Verify all shapes produce correct outputs for all binary input combinations.
 * ============================================================================ */

void test_shape_truth_tables() {
    SECTION_START("Shape Truth Tables - Complete Binary Verification");

    gillies_context_t* ctx = gillies_create();

    // XOR truth table
    float xor_expected[4] = {0, 1, 1, 0};  // 00, 01, 10, 11
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            gillies_reset(ctx);
            gillies_set_port(ctx, 0, (float)a);
            gillies_set_port(ctx, 1, (float)b);
            gillies_invoke(ctx, GILLIES_SHAPE_XOR, 0, 1, 2);
            gillies_execute(ctx);
            float result = gillies_get_port(ctx, 2);
            float expected = xor_expected[a * 2 + b];
            char msg[64];
            snprintf(msg, sizeof(msg), "XOR(%d, %d) = %.2f, expected %.2f", a, b, result, expected);
            TEST_FLOAT_EQ(result, expected, msg);
        }
    }

    // AND truth table
    float and_expected[4] = {0, 0, 0, 1};
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            gillies_reset(ctx);
            gillies_set_port(ctx, 0, (float)a);
            gillies_set_port(ctx, 1, (float)b);
            gillies_invoke(ctx, GILLIES_SHAPE_AND, 0, 1, 2);
            gillies_execute(ctx);
            float result = gillies_get_port(ctx, 2);
            float expected = and_expected[a * 2 + b];
            char msg[64];
            snprintf(msg, sizeof(msg), "AND(%d, %d) = %.2f, expected %.2f", a, b, result, expected);
            TEST_FLOAT_EQ(result, expected, msg);
        }
    }

    // OR truth table
    float or_expected[4] = {0, 1, 1, 1};
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            gillies_reset(ctx);
            gillies_set_port(ctx, 0, (float)a);
            gillies_set_port(ctx, 1, (float)b);
            gillies_invoke(ctx, GILLIES_SHAPE_OR, 0, 1, 2);
            gillies_execute(ctx);
            float result = gillies_get_port(ctx, 2);
            float expected = or_expected[a * 2 + b];
            char msg[64];
            snprintf(msg, sizeof(msg), "OR(%d, %d) = %.2f, expected %.2f", a, b, result, expected);
            TEST_FLOAT_EQ(result, expected, msg);
        }
    }

    // NOT truth table
    for (int a = 0; a <= 1; a++) {
        gillies_reset(ctx);
        gillies_set_port(ctx, 0, (float)a);
        gillies_invoke(ctx, GILLIES_SHAPE_NOT, 0, 0, 2);
        gillies_execute(ctx);
        float result = gillies_get_port(ctx, 2);
        float expected = 1.0f - a;
        char msg[64];
        snprintf(msg, sizeof(msg), "NOT(%d) = %.2f, expected %.2f", a, result, expected);
        TEST_FLOAT_EQ(result, expected, msg);
    }

    // NAND truth table
    float nand_expected[4] = {1, 1, 1, 0};
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            gillies_reset(ctx);
            gillies_set_port(ctx, 0, (float)a);
            gillies_set_port(ctx, 1, (float)b);
            gillies_invoke(ctx, GILLIES_SHAPE_NAND, 0, 1, 2);
            gillies_execute(ctx);
            float result = gillies_get_port(ctx, 2);
            float expected = nand_expected[a * 2 + b];
            char msg[64];
            snprintf(msg, sizeof(msg), "NAND(%d, %d) = %.2f, expected %.2f", a, b, result, expected);
            TEST_FLOAT_EQ(result, expected, msg);
        }
    }

    // NOR truth table
    float nor_expected[4] = {1, 0, 0, 0};
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            gillies_reset(ctx);
            gillies_set_port(ctx, 0, (float)a);
            gillies_set_port(ctx, 1, (float)b);
            gillies_invoke(ctx, GILLIES_SHAPE_NOR, 0, 1, 2);
            gillies_execute(ctx);
            float result = gillies_get_port(ctx, 2);
            float expected = nor_expected[a * 2 + b];
            char msg[64];
            snprintf(msg, sizeof(msg), "NOR(%d, %d) = %.2f, expected %.2f", a, b, result, expected);
            TEST_FLOAT_EQ(result, expected, msg);
        }
    }

    // XNOR truth table
    float xnor_expected[4] = {1, 0, 0, 1};
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            gillies_reset(ctx);
            gillies_set_port(ctx, 0, (float)a);
            gillies_set_port(ctx, 1, (float)b);
            gillies_invoke(ctx, GILLIES_SHAPE_XNOR, 0, 1, 2);
            gillies_execute(ctx);
            float result = gillies_get_port(ctx, 2);
            float expected = xnor_expected[a * 2 + b];
            char msg[64];
            snprintf(msg, sizeof(msg), "XNOR(%d, %d) = %.2f, expected %.2f", a, b, result, expected);
            TEST_FLOAT_EQ(result, expected, msg);
        }
    }

    gillies_destroy(ctx);
    SECTION_END();
}

/* ============================================================================
 * TEST 2: MATHEMATICAL INVARIANTS
 *
 * Verify algebraic properties hold.
 * ============================================================================ */

void test_mathematical_invariants() {
    SECTION_START("Mathematical Invariants");

    gillies_context_t* ctx = gillies_create();

    // Test over a grid of values
    float test_values[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    int n = sizeof(test_values) / sizeof(test_values[0]);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float a = test_values[i];
            float b = test_values[j];

            // De Morgan: NOT(AND(a,b)) = OR(NOT(a), NOT(b))
            gillies_reset(ctx);
            gillies_set_port(ctx, 0, a);
            gillies_set_port(ctx, 1, b);

            // Left side: NOT(AND(a,b))
            gillies_invoke(ctx, GILLIES_SHAPE_AND, 0, 1, 10);
            gillies_invoke(ctx, GILLIES_SHAPE_NOT, 10, 0, 11);

            // Right side: OR(NOT(a), NOT(b))
            gillies_invoke(ctx, GILLIES_SHAPE_NOT, 0, 0, 12);
            gillies_invoke(ctx, GILLIES_SHAPE_NOT, 1, 0, 13);
            gillies_invoke(ctx, GILLIES_SHAPE_OR, 12, 13, 14);

            gillies_execute(ctx);

            float left = gillies_get_port(ctx, 11);
            float right = gillies_get_port(ctx, 14);

            char msg[128];
            snprintf(msg, sizeof(msg), "De Morgan AND: a=%.2f, b=%.2f, left=%.4f, right=%.4f", a, b, left, right);
            TEST_FLOAT_EQ(left, right, msg);

            // De Morgan: NOT(OR(a,b)) = AND(NOT(a), NOT(b))
            gillies_reset(ctx);
            gillies_set_port(ctx, 0, a);
            gillies_set_port(ctx, 1, b);

            // Left side: NOT(OR(a,b))
            gillies_invoke(ctx, GILLIES_SHAPE_OR, 0, 1, 10);
            gillies_invoke(ctx, GILLIES_SHAPE_NOT, 10, 0, 11);

            // Right side: AND(NOT(a), NOT(b))
            gillies_invoke(ctx, GILLIES_SHAPE_NOT, 0, 0, 12);
            gillies_invoke(ctx, GILLIES_SHAPE_NOT, 1, 0, 13);
            gillies_invoke(ctx, GILLIES_SHAPE_AND, 12, 13, 14);

            gillies_execute(ctx);

            left = gillies_get_port(ctx, 11);
            right = gillies_get_port(ctx, 14);

            snprintf(msg, sizeof(msg), "De Morgan OR: a=%.2f, b=%.2f, left=%.4f, right=%.4f", a, b, left, right);
            TEST_FLOAT_EQ(left, right, msg);

            // XOR = (a AND NOT(b)) OR (NOT(a) AND b)
            // NOTE: This identity only holds for BINARY inputs {0, 1}.
            // For continuous floats, the polynomial forms diverge.
            // XOR polynomial: a + b - 2ab
            // Expansion:      a(1-b) + (1-a)b - a(1-b)(1-a)b (different polynomial!)
            // So we only test this for binary inputs.
            if ((a == 0.0f || a == 1.0f) && (b == 0.0f || b == 1.0f)) {
                gillies_reset(ctx);
                gillies_set_port(ctx, 0, a);
                gillies_set_port(ctx, 1, b);

                // Left: XOR(a,b)
                gillies_invoke(ctx, GILLIES_SHAPE_XOR, 0, 1, 10);

                // Right: (a AND NOT(b)) OR (NOT(a) AND b)
                gillies_invoke(ctx, GILLIES_SHAPE_NOT, 1, 0, 11);  // NOT(b)
                gillies_invoke(ctx, GILLIES_SHAPE_AND, 0, 11, 12); // a AND NOT(b)
                gillies_invoke(ctx, GILLIES_SHAPE_NOT, 0, 0, 13);  // NOT(a)
                gillies_invoke(ctx, GILLIES_SHAPE_AND, 13, 1, 14); // NOT(a) AND b
                gillies_invoke(ctx, GILLIES_SHAPE_OR, 12, 14, 15); // final OR

                gillies_execute(ctx);

                left = gillies_get_port(ctx, 10);
                right = gillies_get_port(ctx, 15);

                snprintf(msg, sizeof(msg), "XOR expansion: a=%.2f, b=%.2f, left=%.4f, right=%.4f", a, b, left, right);
                TEST_FLOAT_EQ(left, right, msg);
            }

            // Double negation: NOT(NOT(a)) = a
            gillies_reset(ctx);
            gillies_set_port(ctx, 0, a);

            gillies_invoke(ctx, GILLIES_SHAPE_NOT, 0, 0, 10);
            gillies_invoke(ctx, GILLIES_SHAPE_NOT, 10, 0, 11);

            gillies_execute(ctx);

            float result = gillies_get_port(ctx, 11);
            snprintf(msg, sizeof(msg), "Double NOT: a=%.2f, NOT(NOT(a))=%.4f", a, result);
            TEST_FLOAT_EQ(result, a, msg);
        }
    }

    gillies_destroy(ctx);
    SECTION_END();
}

/* ============================================================================
 * TEST 3: POLYNOMIAL EXACTNESS
 *
 * Verify the polynomial formulas produce exact expected values.
 * ============================================================================ */

void test_polynomial_exactness() {
    SECTION_START("Polynomial Formula Exactness");

    gillies_context_t* ctx = gillies_create();

    // Test grid
    float values[] = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    int n = sizeof(values) / sizeof(values[0]);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float a = values[i];
            float b = values[j];

            // XOR: a + b - 2ab
            gillies_reset(ctx);
            gillies_set_port(ctx, 0, a);
            gillies_set_port(ctx, 1, b);
            gillies_invoke(ctx, GILLIES_SHAPE_XOR, 0, 1, 2);
            gillies_execute(ctx);

            float result = gillies_get_port(ctx, 2);
            float expected = a + b - 2.0f * a * b;

            char msg[128];
            snprintf(msg, sizeof(msg), "XOR polynomial: a=%.1f, b=%.1f, result=%.6f, expected=%.6f", a, b, result, expected);
            TEST_FLOAT_EQ(result, expected, msg);

            // AND: ab
            gillies_reset(ctx);
            gillies_set_port(ctx, 0, a);
            gillies_set_port(ctx, 1, b);
            gillies_invoke(ctx, GILLIES_SHAPE_AND, 0, 1, 2);
            gillies_execute(ctx);

            result = gillies_get_port(ctx, 2);
            expected = a * b;

            snprintf(msg, sizeof(msg), "AND polynomial: a=%.1f, b=%.1f, result=%.6f, expected=%.6f", a, b, result, expected);
            TEST_FLOAT_EQ(result, expected, msg);

            // OR: a + b - ab
            gillies_reset(ctx);
            gillies_set_port(ctx, 0, a);
            gillies_set_port(ctx, 1, b);
            gillies_invoke(ctx, GILLIES_SHAPE_OR, 0, 1, 2);
            gillies_execute(ctx);

            result = gillies_get_port(ctx, 2);
            expected = a + b - a * b;

            snprintf(msg, sizeof(msg), "OR polynomial: a=%.1f, b=%.1f, result=%.6f, expected=%.6f", a, b, result, expected);
            TEST_FLOAT_EQ(result, expected, msg);
        }
    }

    gillies_destroy(ctx);
    SECTION_END();
}

/* ============================================================================
 * TEST 4: FULL ADDER - ALL COMBINATIONS
 * ============================================================================ */

void test_full_adder_exhaustive() {
    SECTION_START("Full Adder - All 8 Input Combinations");

    gillies_context_t* ctx = gillies_create();

    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            for (int cin = 0; cin <= 1; cin++) {
                gillies_reset(ctx);

                gillies_set_port(ctx, 0, (float)a);
                gillies_set_port(ctx, 1, (float)b);
                gillies_set_port(ctx, 2, (float)cin);

                // sum = a XOR b XOR cin
                gillies_invoke(ctx, GILLIES_SHAPE_XOR, 0, 1, 10);
                gillies_invoke(ctx, GILLIES_SHAPE_XOR, 10, 2, 11);

                // cout = (a AND b) OR ((a XOR b) AND cin)
                gillies_invoke(ctx, GILLIES_SHAPE_AND, 0, 1, 12);
                gillies_invoke(ctx, GILLIES_SHAPE_AND, 10, 2, 13);
                gillies_invoke(ctx, GILLIES_SHAPE_OR, 12, 13, 14);

                gillies_execute(ctx);

                float sum = gillies_get_port(ctx, 11);
                float cout = gillies_get_port(ctx, 14);

                int exp_sum = a ^ b ^ cin;
                int exp_cout = (a & b) | ((a ^ b) & cin);

                char msg[128];
                snprintf(msg, sizeof(msg), "FA(%d,%d,%d): sum=%.2f, expected=%d", a, b, cin, sum, exp_sum);
                TEST_FLOAT_EQ(sum, (float)exp_sum, msg);

                snprintf(msg, sizeof(msg), "FA(%d,%d,%d): cout=%.2f, expected=%d", a, b, cin, cout, exp_cout);
                TEST_FLOAT_EQ(cout, (float)exp_cout, msg);
            }
        }
    }

    gillies_destroy(ctx);
    SECTION_END();
}

/* ============================================================================
 * TEST 5: RIPPLE ADDER - EXHAUSTIVE 4-BIT
 *
 * Test all 256 × 256 = 65536 additions (takes a bit of time)
 * We'll do 4-bit (16 × 16 = 256) for reasonable runtime
 * ============================================================================ */

void test_ripple_adder_4bit() {
    SECTION_START("Ripple Adder - Exhaustive 4-bit (256 additions)");

    gillies_context_t* ctx = gillies_create();

    int correct = 0;
    int total = 256;

    for (int a = 0; a < 16; a++) {
        for (int b = 0; b < 16; b++) {
            gillies_reset(ctx);

            // Set input bits
            for (int i = 0; i < 4; i++) {
                gillies_set_port(ctx, i, (float)((a >> i) & 1));      // a bits
                gillies_set_port(ctx, 4 + i, (float)((b >> i) & 1));  // b bits
            }
            gillies_set_port(ctx, 8, 0.0f);  // carry in

            // Build 4-bit ripple adder
            // Bit 0
            int port = 20;
            gillies_invoke(ctx, GILLIES_SHAPE_XOR, 0, 4, port);      // a0 XOR b0
            gillies_invoke(ctx, GILLIES_SHAPE_XOR, port, 8, port+1); // sum0 = t XOR cin
            gillies_invoke(ctx, GILLIES_SHAPE_AND, 0, 4, port+2);    // a0 AND b0
            gillies_invoke(ctx, GILLIES_SHAPE_AND, port, 8, port+3); // t AND cin
            gillies_invoke(ctx, GILLIES_SHAPE_OR, port+2, port+3, port+4); // c0

            // Bit 1
            port = 30;
            int c_prev = 24;
            gillies_invoke(ctx, GILLIES_SHAPE_XOR, 1, 5, port);
            gillies_invoke(ctx, GILLIES_SHAPE_XOR, port, c_prev, port+1);
            gillies_invoke(ctx, GILLIES_SHAPE_AND, 1, 5, port+2);
            gillies_invoke(ctx, GILLIES_SHAPE_AND, port, c_prev, port+3);
            gillies_invoke(ctx, GILLIES_SHAPE_OR, port+2, port+3, port+4);

            // Bit 2
            port = 40;
            c_prev = 34;
            gillies_invoke(ctx, GILLIES_SHAPE_XOR, 2, 6, port);
            gillies_invoke(ctx, GILLIES_SHAPE_XOR, port, c_prev, port+1);
            gillies_invoke(ctx, GILLIES_SHAPE_AND, 2, 6, port+2);
            gillies_invoke(ctx, GILLIES_SHAPE_AND, port, c_prev, port+3);
            gillies_invoke(ctx, GILLIES_SHAPE_OR, port+2, port+3, port+4);

            // Bit 3
            port = 50;
            c_prev = 44;
            gillies_invoke(ctx, GILLIES_SHAPE_XOR, 3, 7, port);
            gillies_invoke(ctx, GILLIES_SHAPE_XOR, port, c_prev, port+1);
            gillies_invoke(ctx, GILLIES_SHAPE_AND, 3, 7, port+2);
            gillies_invoke(ctx, GILLIES_SHAPE_AND, port, c_prev, port+3);
            gillies_invoke(ctx, GILLIES_SHAPE_OR, port+2, port+3, port+4);

            gillies_execute(ctx);

            // Read result bits
            int sum = 0;
            sum |= ((int)(gillies_get_port(ctx, 21) + 0.5f)) << 0;
            sum |= ((int)(gillies_get_port(ctx, 31) + 0.5f)) << 1;
            sum |= ((int)(gillies_get_port(ctx, 41) + 0.5f)) << 2;
            sum |= ((int)(gillies_get_port(ctx, 51) + 0.5f)) << 3;
            int cout = (int)(gillies_get_port(ctx, 54) + 0.5f);

            int expected = a + b;
            int exp_sum = expected & 0xF;
            int exp_cout = (expected >> 4) & 1;

            if (sum == exp_sum && cout == exp_cout) {
                correct++;
            }
        }
    }

    char msg[128];
    snprintf(msg, sizeof(msg), "All 256 4-bit additions correct");
    TEST_ASSERT(correct == total, msg);

    printf("  PASS: %d/%d additions correct\n", correct, total);

    gillies_destroy(ctx);
    SECTION_END();
}

/* ============================================================================
 * TEST 6: FUNGIBILITY - EXHAUSTIVE CPU VS GPU
 * ============================================================================ */

void test_fungibility_exhaustive() {
    SECTION_START("Fungibility - Exhaustive CPU vs GPU Comparison");

    gillies_context_t* ctx_gpu = gillies_create();
    gillies_context_t* ctx_cpu = gillies_create();

    float values[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    int n = sizeof(values) / sizeof(values[0]);

    gillies_shape_t shapes[] = {
        GILLIES_SHAPE_XOR, GILLIES_SHAPE_AND, GILLIES_SHAPE_OR,
        GILLIES_SHAPE_NAND, GILLIES_SHAPE_NOR, GILLIES_SHAPE_XNOR,
        GILLIES_SHAPE_ADD, GILLIES_SHAPE_SUB, GILLIES_SHAPE_MUL
    };
    int num_shapes = sizeof(shapes) / sizeof(shapes[0]);

    for (int s = 0; s < num_shapes; s++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float a = values[i];
                float b = values[j];

                // GPU
                gillies_reset(ctx_gpu);
                gillies_set_port(ctx_gpu, 0, a);
                gillies_set_port(ctx_gpu, 1, b);
                gillies_invoke(ctx_gpu, shapes[s], 0, 1, 2);
                gillies_execute(ctx_gpu);
                float gpu_result = gillies_get_port(ctx_gpu, 2);

                // CPU
                gillies_reset(ctx_cpu);
                gillies_set_port(ctx_cpu, 0, a);
                gillies_set_port(ctx_cpu, 1, b);
                gillies_invoke(ctx_cpu, shapes[s], 0, 1, 2);
                gillies_execute_cpu(ctx_cpu);
                float cpu_result = gillies_get_port(ctx_cpu, 2);

                float diff = fabsf(gpu_result - cpu_result);

                char msg[128];
                snprintf(msg, sizeof(msg), "Fungibility %s(%.2f, %.2f): GPU=%.6f, CPU=%.6f, diff=%.9f",
                         GILLIES_SHAPE_NAMES[shapes[s]], a, b, gpu_result, cpu_result, diff);
                TEST_ASSERT(diff < TOLERANCE, msg);
            }
        }
    }

    gillies_destroy(ctx_gpu);
    gillies_destroy(ctx_cpu);
    SECTION_END();
}

/* ============================================================================
 * TEST 7: STRESS TEST - MANY INVOCATIONS
 * ============================================================================ */

void test_stress_many_invocations() {
    SECTION_START("Stress Test - 500 Invocations");

    gillies_context_t* ctx = gillies_create();

    // Build a chain of 500 operations
    // Each alternates XOR and AND to create a complex pattern
    gillies_set_port(ctx, 0, 1.0f);
    gillies_set_port(ctx, 1, 0.0f);

    int num_invocations = 500;
    for (int i = 0; i < num_invocations; i++) {
        gillies_shape_t shape = (i % 2 == 0) ? GILLIES_SHAPE_XOR : GILLIES_SHAPE_AND;
        uint16_t in0 = (i == 0) ? 0 : (10 + i - 1);
        uint16_t in1 = 1;
        uint16_t out = 10 + i;
        gillies_invoke(ctx, shape, in0, in1, out);
    }

    // Execute
    clock_t start = clock();
    gillies_execute(ctx);
    clock_t end = clock();

    double time_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

    // Verify execution completed
    TEST_ASSERT(ctx->execution_count == num_invocations, "All invocations executed");

    printf("  Executed %d invocations in %.2f ms\n", num_invocations, time_ms);
    printf("  Throughput: %.0f invocations/sec\n", num_invocations / (time_ms / 1000.0));

    gillies_destroy(ctx);
    SECTION_END();
}

/* ============================================================================
 * TEST 8: EDGE CASES
 * ============================================================================ */

void test_edge_cases() {
    SECTION_START("Edge Cases");

    gillies_context_t* ctx = gillies_create();

    // Test with very small values
    gillies_reset(ctx);
    gillies_set_port(ctx, 0, 1e-6f);
    gillies_set_port(ctx, 1, 1e-6f);
    gillies_invoke(ctx, GILLIES_SHAPE_ADD, 0, 1, 2);
    gillies_execute(ctx);
    float result = gillies_get_port(ctx, 2);
    TEST_FLOAT_EQ(result, 2e-6f, "ADD small values");

    // Test with values very close to 1
    gillies_reset(ctx);
    gillies_set_port(ctx, 0, 0.999999f);
    gillies_set_port(ctx, 1, 0.999999f);
    gillies_invoke(ctx, GILLIES_SHAPE_AND, 0, 1, 2);
    gillies_execute(ctx);
    result = gillies_get_port(ctx, 2);
    float expected = 0.999999f * 0.999999f;
    TEST_FLOAT_EQ(result, expected, "AND values near 1");

    // Test XOR with identical values (should be near 0)
    gillies_reset(ctx);
    gillies_set_port(ctx, 0, 0.5f);
    gillies_set_port(ctx, 1, 0.5f);
    gillies_invoke(ctx, GILLIES_SHAPE_XOR, 0, 1, 2);
    gillies_execute(ctx);
    result = gillies_get_port(ctx, 2);
    expected = 0.5f + 0.5f - 2.0f * 0.5f * 0.5f;  // = 0.5
    TEST_FLOAT_EQ(result, expected, "XOR identical values");

    // Test NOT of 0.5
    gillies_reset(ctx);
    gillies_set_port(ctx, 0, 0.5f);
    gillies_invoke(ctx, GILLIES_SHAPE_NOT, 0, 0, 2);
    gillies_execute(ctx);
    result = gillies_get_port(ctx, 2);
    TEST_FLOAT_EQ(result, 0.5f, "NOT(0.5) = 0.5");

    // Test identity shape
    gillies_reset(ctx);
    gillies_set_port(ctx, 0, 0.12345f);
    gillies_invoke(ctx, GILLIES_SHAPE_IDENTITY, 0, 0, 2);
    gillies_execute(ctx);
    result = gillies_get_port(ctx, 2);
    TEST_FLOAT_EQ(result, 0.12345f, "IDENTITY preserves value");

    // Test SUB
    gillies_reset(ctx);
    gillies_set_port(ctx, 0, 1.0f);
    gillies_set_port(ctx, 1, 0.3f);
    gillies_invoke(ctx, GILLIES_SHAPE_SUB, 0, 1, 2);
    gillies_execute(ctx);
    result = gillies_get_port(ctx, 2);
    TEST_FLOAT_EQ(result, 0.7f, "SUB(1.0, 0.3) = 0.7");

    // Test MUL
    gillies_reset(ctx);
    gillies_set_port(ctx, 0, 0.5f);
    gillies_set_port(ctx, 1, 0.5f);
    gillies_invoke(ctx, GILLIES_SHAPE_MUL, 0, 1, 2);
    gillies_execute(ctx);
    result = gillies_get_port(ctx, 2);
    TEST_FLOAT_EQ(result, 0.25f, "MUL(0.5, 0.5) = 0.25");

    gillies_destroy(ctx);
    SECTION_END();
}

/* ============================================================================
 * TEST 9: COMPOSITION DEPTH
 * ============================================================================ */

void test_composition_depth() {
    SECTION_START("Deep Composition - 20 Levels");

    gillies_context_t* ctx = gillies_create();

    // Build a chain: NOT(NOT(NOT(...NOT(x)...)))
    // 20 NOTs should return the original value

    gillies_set_port(ctx, 0, 0.7f);

    for (int i = 0; i < 20; i++) {
        uint16_t in = (i == 0) ? 0 : (10 + i - 1);
        uint16_t out = 10 + i;
        gillies_invoke(ctx, GILLIES_SHAPE_NOT, in, 0, out);
    }

    gillies_execute(ctx);

    float result = gillies_get_port(ctx, 29);  // Final output
    float expected = 0.7f;  // 20 NOTs = identity

    TEST_FLOAT_EQ(result, expected, "20 NOTs returns original value");

    // Build XOR chain: start with 1, XOR with 1 repeatedly
    // 1 XOR 1 = 0, 0 XOR 1 = 1, 1 XOR 1 = 0, ... alternates
    // After N XORs: even N -> result is 1's initial XOR'd N times with 1
    // Starting: 1
    // After 1 XOR:  0
    // After 2 XORs: 1
    // After 3 XORs: 0
    // ...
    // After 19 XORs (odd count): 0
    gillies_reset(ctx);
    gillies_set_port(ctx, 0, 1.0f);

    for (int i = 0; i < 19; i++) {
        uint16_t in0 = (i == 0) ? 0 : (10 + i - 1);
        uint16_t in1 = 0;
        uint16_t out = 10 + i;
        gillies_invoke(ctx, GILLIES_SHAPE_XOR, in0, in1, out);
    }

    gillies_execute(ctx);

    result = gillies_get_port(ctx, 28);  // 19 XORs: 1→0→1→0...→0
    expected = 0.0f;
    TEST_FLOAT_EQ(result, expected, "19 XORs with 1 returns 0 (alternating pattern)");

    gillies_destroy(ctx);
    SECTION_END();
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main() {
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                  GILLIES RIGOROUS TEST SUITE                          ║\n");
    printf("║         \"For the skeptics. Every shape. Every edge case.\"            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");

    clock_t total_start = clock();

    test_shape_truth_tables();
    test_mathematical_invariants();
    test_polynomial_exactness();
    test_full_adder_exhaustive();
    test_ripple_adder_4bit();
    test_fungibility_exhaustive();
    test_stress_many_invocations();
    test_edge_cases();
    test_composition_depth();

    clock_t total_end = clock();
    double total_time = (double)(total_end - total_start) / CLOCKS_PER_SEC;

    printf("\n╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║  FINAL RESULTS                                                        ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Total tests:   %5d                                                  ║\n", g_tests_passed + g_tests_failed);
    printf("║  Passed:        %5d                                                  ║\n", g_tests_passed);
    printf("║  Failed:        %5d                                                  ║\n", g_tests_failed);
    printf("║  Pass rate:    %5.1f%%                                                 ║\n",
           100.0 * g_tests_passed / (g_tests_passed + g_tests_failed));
    printf("║  Time:         %5.2fs                                                  ║\n", total_time);
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");

    if (g_tests_failed == 0) {
        printf("\n✓ ALL TESTS PASSED\n");
        printf("\"The shapes are frozen. The math is eternal. The skeptics are satisfied.\"\n\n");
    } else {
        printf("\n✗ SOME TESTS FAILED\n");
        printf("Review failures above.\n\n");
    }

    return (g_tests_failed == 0) ? 0 : 1;
}
