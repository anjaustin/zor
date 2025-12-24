/*
 * GILLIES Minimal Proof
 *
 * Demonstrates:
 *   1. Shape invocation
 *   2. Data routing between invocations
 *   3. Correct computation
 *   4. GPU and CPU execution
 *   5. Fungibility (same results on different substrates)
 *
 * "The routing IS the program. The shapes ARE the instruction set."
 *
 * (c) 2025 TriX Project
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "../include/gillies.h"

/* ============================================================================
 * TEST 1: Basic Shape Invocations
 * ============================================================================ */

bool test_basic_shapes() {
    printf("Test 1: Basic Shape Invocations\n");
    printf("================================\n\n");

    gillies_context_t* ctx = gillies_create();

    // Set inputs
    // Port 0: a = 0x55 (binary: 01010101)
    // Port 1: b = 0xAA (binary: 10101010)
    gillies_set_port(ctx, 0, 85.0f);   // 0x55
    gillies_set_port(ctx, 1, 170.0f);  // 0xAA

    printf("Inputs:\n");
    printf("  Port 0 (a) = 0x55 = %d\n", 0x55);
    printf("  Port 1 (b) = 0xAA = %d\n", 0xAA);
    printf("\n");

    // Test XOR: 0x55 ^ 0xAA = 0xFF
    gillies_invoke(ctx, GILLIES_SHAPE_XOR, 0, 1, 2);

    // Test AND: 0x55 & 0xAA = 0x00
    gillies_invoke(ctx, GILLIES_SHAPE_AND, 0, 1, 3);

    // Test OR: 0x55 | 0xAA = 0xFF
    gillies_invoke(ctx, GILLIES_SHAPE_OR, 0, 1, 4);

    printf("Invocations:\n");
    gillies_print_graph(ctx);
    printf("\n");

    // Execute on GPU
    gillies_execute(ctx);

    // Check results
    float xor_result = gillies_get_port(ctx, 2);
    float and_result = gillies_get_port(ctx, 3);
    float or_result = gillies_get_port(ctx, 4);

    printf("Results (GPU):\n");
    printf("  XOR(0x55, 0xAA) = %.0f (expected 255)\n", xor_result);
    printf("  AND(0x55, 0xAA) = %.0f (expected 0)\n", and_result);
    printf("  OR(0x55, 0xAA)  = %.0f (expected 255)\n", or_result);
    printf("\n");

    // Using polynomial forms: these work on float representations
    // XOR: 85 + 170 - 2*85*170 = 255 - 28900 = not what we want for floats
    // For integer-like behavior, we use the values as-is

    // For the polynomial forms to work correctly, inputs should be 0 or 1
    // Let's verify with binary inputs
    gillies_reset(ctx);
    gillies_set_port(ctx, 0, 1.0f);  // a = 1
    gillies_set_port(ctx, 1, 0.0f);  // b = 0

    gillies_invoke(ctx, GILLIES_SHAPE_XOR, 0, 1, 2);
    gillies_invoke(ctx, GILLIES_SHAPE_AND, 0, 1, 3);
    gillies_invoke(ctx, GILLIES_SHAPE_OR, 0, 1, 4);

    gillies_execute(ctx);

    float xor_bin = gillies_get_port(ctx, 2);
    float and_bin = gillies_get_port(ctx, 3);
    float or_bin = gillies_get_port(ctx, 4);

    printf("Binary verification (a=1, b=0):\n");
    printf("  XOR(1, 0) = %.2f (expected 1.00)\n", xor_bin);
    printf("  AND(1, 0) = %.2f (expected 0.00)\n", and_bin);
    printf("  OR(1, 0)  = %.2f (expected 1.00)\n", or_bin);
    printf("\n");

    bool pass = (fabsf(xor_bin - 1.0f) < 0.01f) &&
                (fabsf(and_bin - 0.0f) < 0.01f) &&
                (fabsf(or_bin - 1.0f) < 0.01f);

    printf("Test 1: %s\n\n", pass ? "PASSED" : "FAILED");

    gillies_destroy(ctx);
    return pass;
}

/* ============================================================================
 * TEST 2: Composition - Shapes Feeding Shapes
 * ============================================================================ */

bool test_composition() {
    printf("Test 2: Shape Composition\n");
    printf("=========================\n\n");

    gillies_context_t* ctx = gillies_create();

    // Compute: ((a XOR b) AND c) OR d
    // With a=1, b=1, c=1, d=0
    // Expected: ((1 XOR 1) AND 1) OR 0 = (0 AND 1) OR 0 = 0 OR 0 = 0

    gillies_set_port(ctx, 0, 1.0f);  // a
    gillies_set_port(ctx, 1, 1.0f);  // b
    gillies_set_port(ctx, 2, 1.0f);  // c
    gillies_set_port(ctx, 3, 0.0f);  // d

    printf("Inputs: a=1, b=1, c=1, d=0\n");
    printf("Computing: ((a XOR b) AND c) OR d\n\n");

    // Build graph
    gillies_invoke(ctx, GILLIES_SHAPE_XOR, 0, 1, 10);  // t1 = a XOR b
    gillies_invoke(ctx, GILLIES_SHAPE_AND, 10, 2, 11); // t2 = t1 AND c
    gillies_invoke(ctx, GILLIES_SHAPE_OR, 11, 3, 12);  // result = t2 OR d

    printf("Invocations:\n");
    gillies_print_graph(ctx);
    printf("\n");

    // Execute
    gillies_execute(ctx);

    float t1 = gillies_get_port(ctx, 10);
    float t2 = gillies_get_port(ctx, 11);
    float result = gillies_get_port(ctx, 12);

    printf("Results:\n");
    printf("  t1 = XOR(1, 1) = %.2f (expected 0.00)\n", t1);
    printf("  t2 = AND(t1, 1) = %.2f (expected 0.00)\n", t2);
    printf("  result = OR(t2, 0) = %.2f (expected 0.00)\n", result);
    printf("\n");

    bool pass = (fabsf(result - 0.0f) < 0.01f);

    // Try another case: a=1, b=0, c=1, d=0
    // Expected: ((1 XOR 0) AND 1) OR 0 = (1 AND 1) OR 0 = 1 OR 0 = 1
    gillies_reset(ctx);
    gillies_set_port(ctx, 0, 1.0f);
    gillies_set_port(ctx, 1, 0.0f);
    gillies_set_port(ctx, 2, 1.0f);
    gillies_set_port(ctx, 3, 0.0f);

    gillies_invoke(ctx, GILLIES_SHAPE_XOR, 0, 1, 10);
    gillies_invoke(ctx, GILLIES_SHAPE_AND, 10, 2, 11);
    gillies_invoke(ctx, GILLIES_SHAPE_OR, 11, 3, 12);

    gillies_execute(ctx);

    result = gillies_get_port(ctx, 12);
    printf("Second case (a=1, b=0, c=1, d=0):\n");
    printf("  result = %.2f (expected 1.00)\n", result);
    printf("\n");

    pass = pass && (fabsf(result - 1.0f) < 0.01f);

    printf("Test 2: %s\n\n", pass ? "PASSED" : "FAILED");

    gillies_destroy(ctx);
    return pass;
}

/* ============================================================================
 * TEST 3: Fungibility - Same Graph, CPU vs GPU
 * ============================================================================ */

bool test_fungibility() {
    printf("Test 3: Fungibility (CPU vs GPU)\n");
    printf("================================\n\n");

    // Create two contexts
    gillies_context_t* ctx_gpu = gillies_create();
    gillies_context_t* ctx_cpu = gillies_create();

    // Same inputs
    float a = 0.7f, b = 0.3f, c = 0.5f;

    gillies_set_port(ctx_gpu, 0, a);
    gillies_set_port(ctx_gpu, 1, b);
    gillies_set_port(ctx_gpu, 2, c);

    gillies_set_port(ctx_cpu, 0, a);
    gillies_set_port(ctx_cpu, 1, b);
    gillies_set_port(ctx_cpu, 2, c);

    printf("Inputs: a=%.2f, b=%.2f, c=%.2f\n\n", a, b, c);

    // Same graph: (a XOR b) + c
    gillies_invoke(ctx_gpu, GILLIES_SHAPE_XOR, 0, 1, 10);
    gillies_invoke(ctx_gpu, GILLIES_SHAPE_ADD, 10, 2, 11);

    gillies_invoke(ctx_cpu, GILLIES_SHAPE_XOR, 0, 1, 10);
    gillies_invoke(ctx_cpu, GILLIES_SHAPE_ADD, 10, 2, 11);

    // Execute on different substrates
    gillies_execute(ctx_gpu);      // GPU
    gillies_execute_cpu(ctx_cpu);  // CPU

    float gpu_xor = gillies_get_port(ctx_gpu, 10);
    float gpu_add = gillies_get_port(ctx_gpu, 11);

    float cpu_xor = gillies_get_port(ctx_cpu, 10);
    float cpu_add = gillies_get_port(ctx_cpu, 11);

    printf("GPU Results:\n");
    printf("  XOR(%.2f, %.2f) = %.4f\n", a, b, gpu_xor);
    printf("  ADD(xor, %.2f) = %.4f\n", c, gpu_add);

    printf("\nCPU Results:\n");
    printf("  XOR(%.2f, %.2f) = %.4f\n", a, b, cpu_xor);
    printf("  ADD(xor, %.2f) = %.4f\n", c, cpu_add);

    // Manual verification
    // XOR(0.7, 0.3) = 0.7 + 0.3 - 2*0.7*0.3 = 1.0 - 0.42 = 0.58
    // ADD(0.58, 0.5) = 1.08
    float expected_xor = a + b - 2.0f * a * b;
    float expected_add = expected_xor + c;

    printf("\nExpected:\n");
    printf("  XOR = %.4f\n", expected_xor);
    printf("  ADD = %.4f\n", expected_add);

    // Check fungibility
    float diff_xor = fabsf(gpu_xor - cpu_xor);
    float diff_add = fabsf(gpu_add - cpu_add);

    printf("\nDifference (GPU - CPU):\n");
    printf("  XOR: %.6f\n", diff_xor);
    printf("  ADD: %.6f\n", diff_add);
    printf("\n");

    bool pass = (diff_xor < 0.0001f) && (diff_add < 0.0001f);

    printf("Test 3: %s\n", pass ? "PASSED" : "FAILED");
    printf("Fungibility: %s\n\n",
           pass ? "CONFIRMED - Same graph, same results on different substrates"
                : "FAILED - Results differ between substrates");

    gillies_destroy(ctx_gpu);
    gillies_destroy(ctx_cpu);
    return pass;
}

/* ============================================================================
 * TEST 4: Full Adder via Shape Composition
 * ============================================================================ */

bool test_full_adder() {
    printf("Test 4: Full Adder via Composition\n");
    printf("==================================\n\n");

    gillies_context_t* ctx = gillies_create();

    // Full adder: sum = a XOR b XOR c_in
    //             c_out = (a AND b) OR ((a XOR b) AND c_in)

    // Test all 8 input combinations
    int pass_count = 0;
    int total = 8;

    printf("Testing all 8 input combinations:\n\n");
    printf("  a  b  cin | sum cout | expected\n");
    printf("  ---------+----------+----------\n");

    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            for (int cin = 0; cin <= 1; cin++) {
                gillies_reset(ctx);

                gillies_set_port(ctx, 0, (float)a);
                gillies_set_port(ctx, 1, (float)b);
                gillies_set_port(ctx, 2, (float)cin);

                // sum = a XOR b XOR cin
                gillies_invoke(ctx, GILLIES_SHAPE_XOR, 0, 1, 10);   // t1 = a XOR b
                gillies_invoke(ctx, GILLIES_SHAPE_XOR, 10, 2, 11);  // sum = t1 XOR cin

                // cout = (a AND b) OR ((a XOR b) AND cin)
                gillies_invoke(ctx, GILLIES_SHAPE_AND, 0, 1, 12);   // t2 = a AND b
                gillies_invoke(ctx, GILLIES_SHAPE_AND, 10, 2, 13);  // t3 = t1 AND cin
                gillies_invoke(ctx, GILLIES_SHAPE_OR, 12, 13, 14);  // cout = t2 OR t3

                gillies_execute(ctx);

                float sum = gillies_get_port(ctx, 11);
                float cout = gillies_get_port(ctx, 14);

                // Expected values
                int exp_sum = a ^ b ^ cin;
                int exp_cout = (a & b) | ((a ^ b) & cin);

                // Round to nearest integer for comparison
                int got_sum = (int)(sum + 0.5f);
                int got_cout = (int)(cout + 0.5f);

                bool correct = (got_sum == exp_sum) && (got_cout == exp_cout);
                if (correct) pass_count++;

                printf("  %d  %d  %d   |  %d    %d   |   %d    %d    %s\n",
                       a, b, cin, got_sum, got_cout, exp_sum, exp_cout,
                       correct ? "[OK]" : "[FAIL]");
            }
        }
    }

    printf("\n");
    bool pass = (pass_count == total);
    printf("Test 4: %s (%d/%d combinations correct)\n\n",
           pass ? "PASSED" : "FAILED", pass_count, total);

    gillies_destroy(ctx);
    return pass;
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main() {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                                                                       ║\n");
    printf("║                         G I L L I E S                                 ║\n");
    printf("║                                                                       ║\n");
    printf("║       Geometric Instruction Language Layer In Every System           ║\n");
    printf("║                                                                       ║\n");
    printf("║                      THE MINIMAL PROOF                                ║\n");
    printf("║                                                                       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    int passed = 0;
    int total = 4;

    if (test_basic_shapes()) passed++;
    if (test_composition()) passed++;
    if (test_fungibility()) passed++;
    if (test_full_adder()) passed++;

    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("                           RESULTS\n");
    printf("═══════════════════════════════════════════════════════════════════════\n\n");

    printf("Tests passed: %d / %d\n\n", passed, total);

    if (passed == total) {
        printf("╔═══════════════════════════════════════════════════════════════════════╗\n");
        printf("║                                                                       ║\n");
        printf("║                      G I L L I E S   I S   R E A L                    ║\n");
        printf("║                                                                       ║\n");
        printf("║   - Shapes executed (frozen polynomials)                              ║\n");
        printf("║   - Data routed between invocations                                   ║\n");
        printf("║   - Composition verified (shapes feeding shapes)                      ║\n");
        printf("║   - Fungibility proven (CPU = GPU)                                    ║\n");
        printf("║   - Full adder built from primitives                                  ║\n");
        printf("║                                                                       ║\n");
        printf("║   \"The routing IS the program.\"                                       ║\n");
        printf("║   \"The shapes ARE the instruction set.\"                               ║\n");
        printf("║   \"Mathematics as the universal bus.\"                                 ║\n");
        printf("║   \"Geometry as the protocol.\"                                         ║\n");
        printf("║                                                                       ║\n");
        printf("║                   THE FOUNDATION IS LAID.                             ║\n");
        printf("║                                                                       ║\n");
        printf("╚═══════════════════════════════════════════════════════════════════════╝\n");
    } else {
        printf("Some tests failed. Debug and retry.\n");
    }

    printf("\n");
    return (passed == total) ? 0 : 1;
}
