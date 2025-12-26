/**
 * DB Cooper Native Operations - Test Suite
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "db_cooper_ops.h"

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s\n", msg); \
        failures++; \
    } else { \
        printf("PASS: %s\n", msg); \
        passes++; \
    } \
} while(0)

static int passes = 0;
static int failures = 0;

void test_version() {
    const char* ver = cooper_ops_version();
    ASSERT(ver != NULL && strlen(ver) > 0, "version returns string");
}

void test_simd_detection() {
    printf("\nSIMD Detection:\n");
    printf("  NEON: %s\n", cooper_has_neon() ? "YES" : "NO");
    printf("  AVX2: %s\n", cooper_has_avx2() ? "YES" : "NO");
    ASSERT(1, "simd detection runs");
}

void test_pack_unpack() {
    int8_t ternary[8] = {1, -1, 0, 1, -1, -1, 0, 1};
    uint8_t pos[1], neg[1];
    int8_t recovered[8];
    
    cooper_pack_ternary(ternary, pos, neg, 8);
    cooper_unpack_ternary(pos, neg, recovered, 8);
    
    int match = 1;
    for (int i = 0; i < 8; i++) {
        if (ternary[i] != recovered[i]) match = 0;
    }
    
    ASSERT(match, "pack/unpack roundtrip");
}

void test_similarity_identical() {
    /* Two identical vectors should have max similarity */
    uint8_t pos[4] = {0xFF, 0x00, 0xAA, 0x55};
    uint8_t neg[4] = {0x00, 0xFF, 0x55, 0xAA};
    
    int32_t score = cooper_ternary_similarity(pos, neg, pos, neg, 4);
    
    /* Should be 32 (all 32 bits agree) */
    ASSERT(score == 32, "identical vectors have max similarity");
}

void test_similarity_opposite() {
    /* Opposite vectors should have negative similarity */
    uint8_t q_pos[2] = {0xFF, 0x00};
    uint8_t q_neg[2] = {0x00, 0xFF};
    uint8_t d_pos[2] = {0x00, 0xFF};  /* Opposite of query */
    uint8_t d_neg[2] = {0xFF, 0x00};
    
    int32_t score = cooper_ternary_similarity(q_pos, q_neg, d_pos, d_neg, 2);
    
    /* Should be -16 (all 16 bits conflict) */
    ASSERT(score == -16, "opposite vectors have negative similarity");
}

void test_similarity_orthogonal() {
    /* Non-overlapping vectors should have zero similarity */
    uint8_t q_pos[2] = {0xF0, 0x00};
    uint8_t q_neg[2] = {0x00, 0x00};
    uint8_t d_pos[2] = {0x00, 0x0F};
    uint8_t d_neg[2] = {0x00, 0x00};
    
    int32_t score = cooper_ternary_similarity(q_pos, q_neg, d_pos, d_neg, 2);
    
    ASSERT(score == 0, "orthogonal vectors have zero similarity");
}

void test_batch_similarity() {
    /* Query */
    uint8_t q_pos[4] = {0xFF, 0xFF, 0x00, 0x00};
    uint8_t q_neg[4] = {0x00, 0x00, 0xFF, 0xFF};
    
    /* Documents (3 docs, 4 bytes each) */
    uint8_t d_pos[12] = {
        0xFF, 0xFF, 0x00, 0x00,  /* doc0: identical to query */
        0x00, 0x00, 0xFF, 0xFF,  /* doc1: opposite */
        0x0F, 0x0F, 0x00, 0x00   /* doc2: partial match */
    };
    uint8_t d_neg[12] = {
        0x00, 0x00, 0xFF, 0xFF,
        0xFF, 0xFF, 0x00, 0x00,
        0x00, 0x00, 0x0F, 0x0F
    };
    
    int32_t scores[3];
    cooper_batch_similarity(q_pos, q_neg, d_pos, d_neg, scores, 3, 4);
    
    ASSERT(scores[0] == 32, "batch: identical doc");
    ASSERT(scores[1] == -32, "batch: opposite doc");
    ASSERT(scores[2] > 0 && scores[2] < 32, "batch: partial match doc");
}

void test_derive_coarse() {
    /* Fine: 16 dims, pool_factor 4 -> coarse: 4 dims */
    uint8_t fine_pos[2] = {0xFF, 0x00};  /* dims 0-7 positive, 8-15 zero */
    uint8_t fine_neg[2] = {0x00, 0xFF};  /* dims 0-7 zero, 8-15 negative */
    uint8_t coarse_pos[1], coarse_neg[1];
    
    cooper_derive_coarse(fine_pos, fine_neg, coarse_pos, coarse_neg, 16, 4);
    
    /* Coarse dim 0: pool of dims 0-3 (all +1) -> +1 */
    /* Coarse dim 1: pool of dims 4-7 (all +1) -> +1 */
    /* Coarse dim 2: pool of dims 8-11 (all -1) -> -1 */
    /* Coarse dim 3: pool of dims 12-15 (all -1) -> -1 */
    
    ASSERT((coarse_pos[0] & 0x03) == 0x03, "coarse derivation: positive dims");
    ASSERT((coarse_neg[0] & 0x0C) == 0x0C, "coarse derivation: negative dims");
}

void test_popcount() {
    uint8_t bits[4] = {0xFF, 0x0F, 0xAA, 0x00};  /* 8 + 4 + 4 + 0 = 16 */
    int32_t count = cooper_popcount(bits, 4);
    ASSERT(count == 16, "popcount");
}

void test_hamming_distance() {
    uint8_t a[2] = {0xFF, 0x00};
    uint8_t b[2] = {0x0F, 0xF0};  /* 4 + 4 = 8 bits different */
    
    int32_t dist = cooper_hamming_distance(a, b, 2);
    ASSERT(dist == 8, "hamming distance");
}

void bench_similarity() {
    printf("\n--- Benchmark ---\n");
    
    size_t dims = 384;
    size_t packed_bytes = (dims + 7) / 8;
    size_t n_docs = 10000;
    size_t n_queries = 100;
    
    /* Allocate */
    uint8_t* q_pos = (uint8_t*)malloc(packed_bytes);
    uint8_t* q_neg = (uint8_t*)malloc(packed_bytes);
    uint8_t* d_pos = (uint8_t*)malloc(n_docs * packed_bytes);
    uint8_t* d_neg = (uint8_t*)malloc(n_docs * packed_bytes);
    int32_t* scores = (int32_t*)malloc(n_docs * sizeof(int32_t));
    
    /* Fill with random data */
    srand(42);
    for (size_t i = 0; i < packed_bytes; i++) {
        q_pos[i] = rand() & 0xFF;
        q_neg[i] = rand() & 0xFF;
    }
    for (size_t i = 0; i < n_docs * packed_bytes; i++) {
        d_pos[i] = rand() & 0xFF;
        d_neg[i] = rand() & 0xFF;
    }
    
    /* Benchmark batch similarity */
    clock_t start = clock();
    for (size_t q = 0; q < n_queries; q++) {
        cooper_batch_similarity(q_pos, q_neg, d_pos, d_neg, scores, n_docs, packed_bytes);
    }
    clock_t end = clock();
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double qps = n_queries / elapsed;
    double comparisons = (double)n_queries * n_docs;
    double mops = comparisons / elapsed / 1e6;
    
    printf("  Documents: %zu\n", n_docs);
    printf("  Dimensions: %zu\n", dims);
    printf("  Queries: %zu\n", n_queries);
    printf("  Time: %.3f sec\n", elapsed);
    printf("  Queries/sec: %.0f\n", qps);
    printf("  M comparisons/sec: %.1f\n", mops);
    
    free(q_pos);
    free(q_neg);
    free(d_pos);
    free(d_neg);
    free(scores);
}

int main() {
    printf("DB Cooper Native Operations Test\n");
    printf("================================\n\n");
    
    cooper_ops_init();
    printf("Version: %s\n", cooper_ops_version());
    
    test_version();
    test_simd_detection();
    test_pack_unpack();
    test_similarity_identical();
    test_similarity_opposite();
    test_similarity_orthogonal();
    test_batch_similarity();
    test_derive_coarse();
    test_popcount();
    test_hamming_distance();
    
    bench_similarity();
    
    printf("\n================================\n");
    printf("Results: %d passed, %d failed\n", passes, failures);
    
    return failures > 0 ? 1 : 0;
}
