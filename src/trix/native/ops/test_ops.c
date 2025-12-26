/**
 * TriX Native Ops - Test Suite
 */

#include "trix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define TEST(name) static int test_##name(void)
#define RUN_TEST(name) do { \
    printf("  %-40s", #name); \
    if (test_##name()) { \
        printf("[PASS]\n"); \
        passed++; \
    } else { \
        printf("[FAIL]\n"); \
        failed++; \
    } \
} while(0)

#define ASSERT(cond) do { if (!(cond)) return 0; } while(0)
#define ASSERT_NEAR(a, b, eps) ASSERT(fabs((a) - (b)) < (eps))

/* ============================================================================
 * FROZEN SHAPES TESTS
 * ============================================================================ */

TEST(relu_positive) {
    ASSERT_NEAR(shape_relu(5.0f), 5.0f, 1e-6);
    ASSERT_NEAR(shape_relu(0.001f), 0.001f, 1e-6);
    return 1;
}

TEST(relu_negative) {
    ASSERT_NEAR(shape_relu(-5.0f), 0.0f, 1e-6);
    ASSERT_NEAR(shape_relu(-0.001f), 0.0f, 1e-6);
    return 1;
}

TEST(relu_zero) {
    ASSERT_NEAR(shape_relu(0.0f), 0.0f, 1e-6);
    return 1;
}

TEST(xor_binary) {
    /* On binary inputs, polynomial XOR equals hardware XOR */
    ASSERT_NEAR(shape_xor(0.0f, 0.0f), 0.0f, 1e-6);  /* 0 ^ 0 = 0 */
    ASSERT_NEAR(shape_xor(0.0f, 1.0f), 1.0f, 1e-6);  /* 0 ^ 1 = 1 */
    ASSERT_NEAR(shape_xor(1.0f, 0.0f), 1.0f, 1e-6);  /* 1 ^ 0 = 1 */
    ASSERT_NEAR(shape_xor(1.0f, 1.0f), 0.0f, 1e-6);  /* 1 ^ 1 = 0 */
    return 1;
}

TEST(and_binary) {
    ASSERT_NEAR(shape_and(0.0f, 0.0f), 0.0f, 1e-6);
    ASSERT_NEAR(shape_and(0.0f, 1.0f), 0.0f, 1e-6);
    ASSERT_NEAR(shape_and(1.0f, 0.0f), 0.0f, 1e-6);
    ASSERT_NEAR(shape_and(1.0f, 1.0f), 1.0f, 1e-6);
    return 1;
}

TEST(or_binary) {
    ASSERT_NEAR(shape_or(0.0f, 0.0f), 0.0f, 1e-6);
    ASSERT_NEAR(shape_or(0.0f, 1.0f), 1.0f, 1e-6);
    ASSERT_NEAR(shape_or(1.0f, 0.0f), 1.0f, 1e-6);
    ASSERT_NEAR(shape_or(1.0f, 1.0f), 1.0f, 1e-6);
    return 1;
}

TEST(half_adder) {
    float sum, carry;
    
    shape_half_adder(0.0f, 0.0f, &sum, &carry);
    ASSERT_NEAR(sum, 0.0f, 1e-6);
    ASSERT_NEAR(carry, 0.0f, 1e-6);
    
    shape_half_adder(1.0f, 0.0f, &sum, &carry);
    ASSERT_NEAR(sum, 1.0f, 1e-6);
    ASSERT_NEAR(carry, 0.0f, 1e-6);
    
    shape_half_adder(0.0f, 1.0f, &sum, &carry);
    ASSERT_NEAR(sum, 1.0f, 1e-6);
    ASSERT_NEAR(carry, 0.0f, 1e-6);
    
    shape_half_adder(1.0f, 1.0f, &sum, &carry);
    ASSERT_NEAR(sum, 0.0f, 1e-6);  /* 1 + 1 = 10 binary, sum = 0 */
    ASSERT_NEAR(carry, 1.0f, 1e-6); /* carry = 1 */
    
    return 1;
}

TEST(full_adder) {
    float sum, carry;
    
    /* 0 + 0 + 0 = 00 */
    shape_full_adder(0.0f, 0.0f, 0.0f, &sum, &carry);
    ASSERT_NEAR(sum, 0.0f, 1e-6);
    ASSERT_NEAR(carry, 0.0f, 1e-6);
    
    /* 1 + 1 + 1 = 11 (binary) = 3 */
    shape_full_adder(1.0f, 1.0f, 1.0f, &sum, &carry);
    ASSERT_NEAR(sum, 1.0f, 1e-6);
    ASSERT_NEAR(carry, 1.0f, 1e-6);
    
    /* 1 + 1 + 0 = 10 */
    shape_full_adder(1.0f, 1.0f, 0.0f, &sum, &carry);
    ASSERT_NEAR(sum, 0.0f, 1e-6);
    ASSERT_NEAR(carry, 1.0f, 1e-6);
    
    return 1;
}

/* ============================================================================
 * TERNARY MATRIX TESTS
 * ============================================================================ */

TEST(ternary_from_float) {
    float weights[] = {1.0f, -1.0f, 0.0f, 0.5f, -0.5f, 0.0f};
    TernaryMatrix mat = trix_ternary_from_float(weights, 2, 3);
    
    ASSERT(mat.pos != NULL);
    ASSERT(mat.neg != NULL);
    ASSERT(mat.rows == 2);
    ASSERT(mat.cols == 3);
    
    trix_ternary_free(&mat);
    return 1;
}

TEST(ternary_matvec_identity) {
    /* Weight = [1, 0, 0; 0, 1, 0; 0, 0, 1] should give identity */
    float weights[] = {1.0f, 0.0f, 0.0f, 
                       0.0f, 1.0f, 0.0f, 
                       0.0f, 0.0f, 1.0f};
    float scale[] = {1.0f, 1.0f, 1.0f};
    float x[] = {1.0f, 2.0f, 3.0f};
    float y[3];
    
    TernaryMatrix mat = trix_ternary_from_float(weights, 3, 3);
    trix_ternary_matvec(&mat, x, y, scale);
    
    ASSERT_NEAR(y[0], 1.0f, 1e-6);
    ASSERT_NEAR(y[1], 2.0f, 1e-6);
    ASSERT_NEAR(y[2], 3.0f, 1e-6);
    
    trix_ternary_free(&mat);
    return 1;
}

TEST(ternary_matvec_signs) {
    /* Weight = [1, -1; -1, 1] */
    float weights[] = {1.0f, -1.0f, 
                       -1.0f, 1.0f};
    float scale[] = {1.0f, 1.0f};
    float x[] = {3.0f, 2.0f};
    float y[2];
    
    TernaryMatrix mat = trix_ternary_from_float(weights, 2, 2);
    trix_ternary_matvec(&mat, x, y, scale);
    
    /* y[0] = 1*3 + (-1)*2 = 1 */
    /* y[1] = (-1)*3 + 1*2 = -1 */
    ASSERT_NEAR(y[0], 1.0f, 1e-6);
    ASSERT_NEAR(y[1], -1.0f, 1e-6);
    
    trix_ternary_free(&mat);
    return 1;
}

TEST(ternary_matvec_scaled) {
    float weights[] = {1.0f, 1.0f};
    float scale[] = {2.0f};  /* Scale output by 2 */
    float x[] = {3.0f, 4.0f};
    float y[1];
    
    TernaryMatrix mat = trix_ternary_from_float(weights, 1, 2);
    trix_ternary_matvec(&mat, x, y, scale);
    
    /* y[0] = 2.0 * (3 + 4) = 14 */
    ASSERT_NEAR(y[0], 14.0f, 1e-6);
    
    trix_ternary_free(&mat);
    return 1;
}

/* ============================================================================
 * REDUCTION TESTS
 * ============================================================================ */

TEST(sum_small) {
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_NEAR(trix_sum(x, 4), 10.0f, 1e-6);
    return 1;
}

TEST(sum_empty) {
    ASSERT_NEAR(trix_sum(NULL, 0), 0.0f, 1e-6);
    return 1;
}

TEST(norm_unit) {
    float x[] = {3.0f, 4.0f};
    ASSERT_NEAR(trix_norm(x, 2), 5.0f, 1e-6);  /* 3-4-5 triangle */
    return 1;
}

/* ============================================================================
 * TILE TESTS
 * ============================================================================ */

TEST(tile_create_free) {
    TrixTile* tile = trix_tile_create(64, 128);
    ASSERT(tile != NULL);
    ASSERT(tile->d_model == 64);
    ASSERT(tile->d_hidden == 128);
    trix_tile_free(tile);
    return 1;
}

TEST(tile_forward) {
    TrixTile* tile = trix_tile_create(4, 8);
    
    /* Initialize with known weights */
    float up[8 * 4], down[4 * 8], bias[8];
    float scales_up[8], scales_down[4];
    
    for (int i = 0; i < 8 * 4; i++) up[i] = (i % 3 == 0) ? 1.0f : 0.0f;
    for (int i = 0; i < 4 * 8; i++) down[i] = (i % 3 == 0) ? 1.0f : 0.0f;
    for (int i = 0; i < 8; i++) { bias[i] = 0.0f; scales_up[i] = 1.0f; }
    for (int i = 0; i < 4; i++) scales_down[i] = 1.0f;
    
    trix_tile_init_weights(tile, up, down, bias, scales_up, scales_down);
    
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[4];
    
    trix_tile_forward(tile, x, y, 1);
    
    /* Just verify no crash and output is finite */
    for (int i = 0; i < 4; i++) {
        ASSERT(isfinite(y[i]));
    }
    
    trix_tile_free(tile);
    return 1;
}

TEST(tile_backward) {
    TrixTile* tile = trix_tile_create(4, 8);
    
    float up[8 * 4], down[4 * 8], bias[8];
    float scales_up[8], scales_down[4];
    
    for (int i = 0; i < 8 * 4; i++) up[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    for (int i = 0; i < 4 * 8; i++) down[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    for (int i = 0; i < 8; i++) { bias[i] = 0.1f; scales_up[i] = 1.0f; }
    for (int i = 0; i < 4; i++) scales_down[i] = 1.0f;
    
    trix_tile_init_weights(tile, up, down, bias, scales_up, scales_down);
    
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[4];
    trix_tile_forward(tile, x, y, 1);
    
    float d_y[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float d_x[4];
    trix_tile_backward(tile, d_y, d_x, 1);
    
    /* Verify gradients are finite */
    for (int i = 0; i < 4; i++) {
        ASSERT(isfinite(d_x[i]));
    }
    
    trix_tile_free(tile);
    return 1;
}

/* ============================================================================
 * ROUTING TESTS
 * ============================================================================ */

TEST(hamming_identical) {
    uint8_t a[] = {0xFF, 0x00, 0xAA};
    uint8_t b[] = {0xFF, 0x00, 0xAA};
    ASSERT(trix_hamming_distance(a, b, 3) == 0);
    return 1;
}

TEST(hamming_opposite) {
    uint8_t a[] = {0xFF};
    uint8_t b[] = {0x00};
    ASSERT(trix_hamming_distance(a, b, 1) == 8);  /* All 8 bits differ */
    return 1;
}

TEST(route_hamming) {
    uint8_t input[] = {0xFF};
    uint8_t sigs[] = {0x00, 0xFF, 0x0F};  /* 3 signatures */
    
    size_t best = trix_route_hamming(input, sigs, 3, 1);
    ASSERT(best == 1);  /* 0xFF matches signature 1 */
    return 1;
}

TEST(binarize) {
    float x[] = {1.0f, -1.0f, 0.5f, -0.5f, 0.0f, 1.0f, -1.0f, 1.0f};
    uint8_t out[1];
    trix_binarize(x, out, 8);
    
    /* Expected: 1,0,1,0,1,1,0,1 = 0b10110101 = 0xB5 (LSB first) */
    /* Actually: bit 0 = x[0] >= 0 = 1
     *           bit 1 = x[1] >= 0 = 0
     *           bit 2 = x[2] >= 0 = 1
     *           bit 3 = x[3] >= 0 = 0
     *           bit 4 = x[4] >= 0 = 1
     *           bit 5 = x[5] >= 0 = 1
     *           bit 6 = x[6] >= 0 = 0
     *           bit 7 = x[7] >= 0 = 1
     * = 0b10110101 = 0xB5
     */
    ASSERT(out[0] == 0xB5);
    return 1;
}

/* ============================================================================
 * BINARY FROZEN SHAPES TESTS
 * ============================================================================ */

TEST(binary_xor) {
    /* Verify binary XOR matches polynomial XOR on all inputs */
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            float poly = shape_xor((float)a, (float)b);
            uint8_t bits = shape_xor_binary((uint8_t)a, (uint8_t)b);
            ASSERT((int)poly == (int)bits);
        }
    }
    return 1;
}

TEST(binary_and) {
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            float poly = shape_and((float)a, (float)b);
            uint8_t bits = shape_and_binary((uint8_t)a, (uint8_t)b);
            ASSERT((int)poly == (int)bits);
        }
    }
    return 1;
}

TEST(binary_or) {
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            float poly = shape_or((float)a, (float)b);
            uint8_t bits = shape_or_binary((uint8_t)a, (uint8_t)b);
            ASSERT((int)poly == (int)bits);
        }
    }
    return 1;
}

TEST(binary_not) {
    for (int a = 0; a <= 1; a++) {
        float poly = shape_not((float)a);
        uint8_t bits = shape_not_binary((uint8_t)a) & 1;
        ASSERT((int)poly == (int)bits);
    }
    return 1;
}

TEST(binary_half_adder) {
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            float poly_sum, poly_carry;
            uint8_t bin_sum, bin_carry;
            
            shape_half_adder((float)a, (float)b, &poly_sum, &poly_carry);
            shape_half_adder_binary((uint8_t)a, (uint8_t)b, &bin_sum, &bin_carry);
            
            ASSERT((int)poly_sum == (int)(bin_sum & 1));
            ASSERT((int)poly_carry == (int)(bin_carry & 1));
        }
    }
    return 1;
}

TEST(binary_full_adder) {
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            for (int cin = 0; cin <= 1; cin++) {
                float poly_sum, poly_carry;
                uint8_t bin_sum, bin_carry;
                
                shape_full_adder((float)a, (float)b, (float)cin, &poly_sum, &poly_carry);
                shape_full_adder_binary((uint8_t)a, (uint8_t)b, (uint8_t)cin, &bin_sum, &bin_carry);
                
                ASSERT((int)poly_sum == (int)(bin_sum & 1));
                ASSERT((int)poly_carry == (int)(bin_carry & 1));
            }
        }
    }
    return 1;
}

TEST(apply_binary_xor) {
    uint8_t a[] = {0xFF, 0x00, 0xAA};
    uint8_t b[] = {0x0F, 0xF0, 0x55};
    uint8_t out[3];
    
    trix_apply_binary_xor(a, b, out, 3);
    
    ASSERT(out[0] == (0xFF ^ 0x0F));  /* 0xF0 */
    ASSERT(out[1] == (0x00 ^ 0xF0));  /* 0xF0 */
    ASSERT(out[2] == (0xAA ^ 0x55));  /* 0xFF */
    return 1;
}

TEST(apply_binary_and) {
    uint8_t a[] = {0xFF, 0x00, 0xAA};
    uint8_t b[] = {0x0F, 0xF0, 0x55};
    uint8_t out[3];
    
    trix_apply_binary_and(a, b, out, 3);
    
    ASSERT(out[0] == (0xFF & 0x0F));  /* 0x0F */
    ASSERT(out[1] == (0x00 & 0xF0));  /* 0x00 */
    ASSERT(out[2] == (0xAA & 0x55));  /* 0x00 */
    return 1;
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    int passed = 0, failed = 0;
    
    printf("TriX Native Ops Test Suite\n");
    printf("==========================\n\n");
    
    trix_ops_init();
    printf("Version: %s\n", trix_ops_version());
    printf("NEON:    %s\n", trix_has_neon() ? "YES" : "NO");
    printf("AVX2:    %s\n\n", trix_has_avx2() ? "YES" : "NO");
    
    printf("Frozen Shapes:\n");
    RUN_TEST(relu_positive);
    RUN_TEST(relu_negative);
    RUN_TEST(relu_zero);
    RUN_TEST(xor_binary);
    RUN_TEST(and_binary);
    RUN_TEST(or_binary);
    RUN_TEST(half_adder);
    RUN_TEST(full_adder);
    
    printf("\nTernary Matrix:\n");
    RUN_TEST(ternary_from_float);
    RUN_TEST(ternary_matvec_identity);
    RUN_TEST(ternary_matvec_signs);
    RUN_TEST(ternary_matvec_scaled);
    
    printf("\nReductions:\n");
    RUN_TEST(sum_small);
    RUN_TEST(sum_empty);
    RUN_TEST(norm_unit);
    
    printf("\nTile Operations:\n");
    RUN_TEST(tile_create_free);
    RUN_TEST(tile_forward);
    RUN_TEST(tile_backward);
    
    printf("\nRouting:\n");
    RUN_TEST(hamming_identical);
    RUN_TEST(hamming_opposite);
    RUN_TEST(route_hamming);
    RUN_TEST(binarize);
    
    printf("\nBinary Frozen Shapes:\n");
    RUN_TEST(binary_xor);
    RUN_TEST(binary_and);
    RUN_TEST(binary_or);
    RUN_TEST(binary_not);
    RUN_TEST(binary_half_adder);
    RUN_TEST(binary_full_adder);
    RUN_TEST(apply_binary_xor);
    RUN_TEST(apply_binary_and);
    
    printf("\n==========================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    
    return failed > 0 ? 1 : 0;
}
