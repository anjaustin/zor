/**
 * DB Cooper Native Operations
 * 
 * NEON-optimized ternary similarity for Octave DB.
 * Pure bit operations. No floats at query time.
 * 
 * The 5 Optimizations:
 *   1. NEON popcount (vcntq_u8) - count agreement/conflict bits
 *   2. NEON bitwise (vandq, veorq) - parallel AND/XOR on packed ternary
 *   3. NEON horizontal sum (vaddvq) - reduce to scalar
 *   4. NEON batch processing - multiple documents in parallel
 *   5. NEON pooling - vectorized coarse derivation
 */

#ifndef DB_COOPER_OPS_H
#define DB_COOPER_OPS_H

#include <stdint.h>
#include <stddef.h>

/* SIMD detection */
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define COOPER_USE_NEON 1
#else
#define COOPER_USE_NEON 0
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#define COOPER_USE_AVX2 1
#else
#define COOPER_USE_AVX2 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Library version.
 */
const char* cooper_ops_version(void);

/**
 * Initialize the library (detect SIMD, etc).
 */
void cooper_ops_init(void);

/**
 * Check SIMD availability.
 */
int cooper_has_neon(void);
int cooper_has_avx2(void);

/**
 * Ternary similarity between two packed vectors.
 * 
 * score = popcount(q_pos & d_pos) + popcount(q_neg & d_neg)
 *       - popcount(q_pos & d_neg) - popcount(q_neg & d_pos)
 * 
 * @param q_pos  Query positive bitmask (packed uint8)
 * @param q_neg  Query negative bitmask (packed uint8)
 * @param d_pos  Document positive bitmask (packed uint8)
 * @param d_neg  Document negative bitmask (packed uint8)
 * @param packed_bytes  Number of bytes in each bitmask
 * @return  Integer similarity score
 */
int32_t cooper_ternary_similarity(
    const uint8_t* q_pos,
    const uint8_t* q_neg,
    const uint8_t* d_pos,
    const uint8_t* d_neg,
    size_t packed_bytes
);

/**
 * Batch similarity: one query against N documents.
 * 
 * @param q_pos  Query positive bitmask
 * @param q_neg  Query negative bitmask
 * @param d_pos  Document positive bitmasks [N x packed_bytes]
 * @param d_neg  Document negative bitmasks [N x packed_bytes]
 * @param scores Output scores [N]
 * @param n_docs Number of documents
 * @param packed_bytes Bytes per vector
 */
void cooper_batch_similarity(
    const uint8_t* q_pos,
    const uint8_t* q_neg,
    const uint8_t* d_pos,
    const uint8_t* d_neg,
    int32_t* scores,
    size_t n_docs,
    size_t packed_bytes
);

/**
 * Derive coarse from fine via sign(mean(chunks)).
 * 
 * @param fine_pos  Fine positive bitmask [fine_bytes]
 * @param fine_neg  Fine negative bitmask [fine_bytes]
 * @param coarse_pos  Output coarse positive [coarse_bytes]
 * @param coarse_neg  Output coarse negative [coarse_bytes]
 * @param fine_dims  Number of fine dimensions
 * @param pool_factor  How many fine dims per coarse dim
 */
void cooper_derive_coarse(
    const uint8_t* fine_pos,
    const uint8_t* fine_neg,
    uint8_t* coarse_pos,
    uint8_t* coarse_neg,
    size_t fine_dims,
    size_t pool_factor
);

/**
 * Pack ternary int8 array to bitmasks.
 * 
 * @param ternary  Input array of {-1, 0, +1} as int8
 * @param pos_out  Output positive bitmask
 * @param neg_out  Output negative bitmask
 * @param n_dims   Number of dimensions
 */
void cooper_pack_ternary(
    const int8_t* ternary,
    uint8_t* pos_out,
    uint8_t* neg_out,
    size_t n_dims
);

/**
 * Unpack bitmasks to ternary int8 array.
 */
void cooper_unpack_ternary(
    const uint8_t* pos,
    const uint8_t* neg,
    int8_t* ternary_out,
    size_t n_dims
);

/**
 * Hamming distance between two packed vectors.
 * (Number of differing bits)
 */
int32_t cooper_hamming_distance(
    const uint8_t* a,
    const uint8_t* b,
    size_t packed_bytes
);

/**
 * Population count of packed vector.
 * (Number of set bits)
 */
int32_t cooper_popcount(
    const uint8_t* bits,
    size_t packed_bytes
);

#ifdef __cplusplus
}
#endif

#endif /* DB_COOPER_OPS_H */
