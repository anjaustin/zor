/**
 * DB Cooper Native Operations
 * 
 * NEON-optimized ternary similarity for Octave DB.
 * Pure bit operations. No floats at query time.
 */

#include "db_cooper_ops.h"
#include <string.h>

#define COOPER_OPS_VERSION "1.0.0"

/* ============================================================================
 * SCALAR IMPLEMENTATIONS (fallback)
 * ============================================================================ */

/* Lookup table for byte popcount */
static const uint8_t POPCOUNT_TABLE[256] = {
    0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8
};

static inline int popcount_byte(uint8_t b) {
    return POPCOUNT_TABLE[b];
}

static int32_t popcount_scalar(const uint8_t* bits, size_t n_bytes) {
    int32_t count = 0;
    for (size_t i = 0; i < n_bytes; i++) {
        count += popcount_byte(bits[i]);
    }
    return count;
}

static int32_t similarity_scalar(
    const uint8_t* q_pos, const uint8_t* q_neg,
    const uint8_t* d_pos, const uint8_t* d_neg,
    size_t n_bytes
) {
    int32_t agree_pos = 0, agree_neg = 0;
    int32_t conflict_pn = 0, conflict_np = 0;
    
    for (size_t i = 0; i < n_bytes; i++) {
        agree_pos += popcount_byte(q_pos[i] & d_pos[i]);
        agree_neg += popcount_byte(q_neg[i] & d_neg[i]);
        conflict_pn += popcount_byte(q_pos[i] & d_neg[i]);
        conflict_np += popcount_byte(q_neg[i] & d_pos[i]);
    }
    
    return (agree_pos + agree_neg) - (conflict_pn + conflict_np);
}

/* ============================================================================
 * NEON IMPLEMENTATIONS (ARM)
 * ============================================================================ */

#if COOPER_USE_NEON

/* NEON popcount: count bits in 16 bytes at once */
static inline int32_t neon_popcount_16(const uint8_t* data) {
    uint8x16_t v = vld1q_u8(data);
    uint8x16_t cnt = vcntq_u8(v);
    
    /* Sum all bytes: 16 -> 8 -> 4 -> 2 -> 1 */
    #if defined(__aarch64__)
    return vaddvq_u8(cnt);
    #else
    uint8x8_t sum8 = vadd_u8(vget_low_u8(cnt), vget_high_u8(cnt));
    uint16x4_t sum4 = vpaddl_u8(sum8);
    uint32x2_t sum2 = vpaddl_u16(sum4);
    uint64x1_t sum1 = vpaddl_u32(sum2);
    return (int32_t)vget_lane_u64(sum1, 0);
    #endif
}

static int32_t popcount_neon(const uint8_t* bits, size_t n_bytes) {
    int32_t count = 0;
    size_t i = 0;
    
    /* Process 16 bytes at a time */
    for (; i + 16 <= n_bytes; i += 16) {
        count += neon_popcount_16(bits + i);
    }
    
    /* Remainder with scalar */
    for (; i < n_bytes; i++) {
        count += popcount_byte(bits[i]);
    }
    
    return count;
}

/* NEON ternary similarity: process 16 bytes at a time */
static int32_t similarity_neon(
    const uint8_t* q_pos, const uint8_t* q_neg,
    const uint8_t* d_pos, const uint8_t* d_neg,
    size_t n_bytes
) {
    int32_t agree = 0, conflict = 0;
    size_t i = 0;
    
    /* Process 16 bytes at a time with NEON */
    for (; i + 16 <= n_bytes; i += 16) {
        uint8x16_t qp = vld1q_u8(q_pos + i);
        uint8x16_t qn = vld1q_u8(q_neg + i);
        uint8x16_t dp = vld1q_u8(d_pos + i);
        uint8x16_t dn = vld1q_u8(d_neg + i);
        
        /* Agreement: (q_pos & d_pos) | (q_neg & d_neg) */
        uint8x16_t agree_pp = vandq_u8(qp, dp);
        uint8x16_t agree_nn = vandq_u8(qn, dn);
        
        /* Conflict: (q_pos & d_neg) | (q_neg & d_pos) */
        uint8x16_t conf_pn = vandq_u8(qp, dn);
        uint8x16_t conf_np = vandq_u8(qn, dp);
        
        /* Count bits */
        uint8x16_t agree_cnt = vaddq_u8(vcntq_u8(agree_pp), vcntq_u8(agree_nn));
        uint8x16_t conf_cnt = vaddq_u8(vcntq_u8(conf_pn), vcntq_u8(conf_np));
        
        /* Horizontal sum */
        #if defined(__aarch64__)
        agree += vaddvq_u8(agree_cnt);
        conflict += vaddvq_u8(conf_cnt);
        #else
        uint8x8_t a8 = vadd_u8(vget_low_u8(agree_cnt), vget_high_u8(agree_cnt));
        uint8x8_t c8 = vadd_u8(vget_low_u8(conf_cnt), vget_high_u8(conf_cnt));
        uint16x4_t a4 = vpaddl_u8(a8);
        uint16x4_t c4 = vpaddl_u8(c8);
        uint32x2_t a2 = vpaddl_u16(a4);
        uint32x2_t c2 = vpaddl_u16(c4);
        agree += vget_lane_u32(a2, 0) + vget_lane_u32(a2, 1);
        conflict += vget_lane_u32(c2, 0) + vget_lane_u32(c2, 1);
        #endif
    }
    
    /* Remainder with scalar */
    for (; i < n_bytes; i++) {
        agree += popcount_byte(q_pos[i] & d_pos[i]);
        agree += popcount_byte(q_neg[i] & d_neg[i]);
        conflict += popcount_byte(q_pos[i] & d_neg[i]);
        conflict += popcount_byte(q_neg[i] & d_pos[i]);
    }
    
    return agree - conflict;
}

/* NEON batch similarity: process multiple documents */
static void batch_similarity_neon(
    const uint8_t* q_pos, const uint8_t* q_neg,
    const uint8_t* d_pos, const uint8_t* d_neg,
    int32_t* scores,
    size_t n_docs, size_t packed_bytes
) {
    /* For each document, compute similarity */
    for (size_t doc = 0; doc < n_docs; doc++) {
        const uint8_t* dp = d_pos + doc * packed_bytes;
        const uint8_t* dn = d_neg + doc * packed_bytes;
        scores[doc] = similarity_neon(q_pos, q_neg, dp, dn, packed_bytes);
    }
}

#endif /* COOPER_USE_NEON */

/* ============================================================================
 * AVX2 IMPLEMENTATIONS (x86)
 * ============================================================================ */

#if COOPER_USE_AVX2

static int32_t popcount_avx2(const uint8_t* bits, size_t n_bytes) {
    int32_t count = 0;
    size_t i = 0;
    
    /* Process 32 bytes at a time */
    for (; i + 32 <= n_bytes; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(bits + i));
        /* AVX2 doesn't have direct popcount, use lookup table method */
        const __m256i lookup = _mm256_setr_epi8(
            0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
            0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4
        );
        const __m256i low_mask = _mm256_set1_epi8(0x0f);
        
        __m256i lo = _mm256_and_si256(v, low_mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
        __m256i cnt_lo = _mm256_shuffle_epi8(lookup, lo);
        __m256i cnt_hi = _mm256_shuffle_epi8(lookup, hi);
        __m256i cnt = _mm256_add_epi8(cnt_lo, cnt_hi);
        
        /* Horizontal sum */
        __m256i sum16 = _mm256_sad_epu8(cnt, _mm256_setzero_si256());
        count += _mm256_extract_epi64(sum16, 0) + _mm256_extract_epi64(sum16, 1) +
                 _mm256_extract_epi64(sum16, 2) + _mm256_extract_epi64(sum16, 3);
    }
    
    /* Remainder with scalar */
    for (; i < n_bytes; i++) {
        count += popcount_byte(bits[i]);
    }
    
    return count;
}

static int32_t similarity_avx2(
    const uint8_t* q_pos, const uint8_t* q_neg,
    const uint8_t* d_pos, const uint8_t* d_neg,
    size_t n_bytes
) {
    int32_t agree = 0, conflict = 0;
    size_t i = 0;
    
    const __m256i lookup = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4
    );
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    
    /* Process 32 bytes at a time */
    for (; i + 32 <= n_bytes; i += 32) {
        __m256i qp = _mm256_loadu_si256((const __m256i*)(q_pos + i));
        __m256i qn = _mm256_loadu_si256((const __m256i*)(q_neg + i));
        __m256i dp = _mm256_loadu_si256((const __m256i*)(d_pos + i));
        __m256i dn = _mm256_loadu_si256((const __m256i*)(d_neg + i));
        
        /* Agreement and conflict */
        __m256i agree_bits = _mm256_or_si256(
            _mm256_and_si256(qp, dp),
            _mm256_and_si256(qn, dn)
        );
        __m256i conf_bits = _mm256_or_si256(
            _mm256_and_si256(qp, dn),
            _mm256_and_si256(qn, dp)
        );
        
        /* Popcount via lookup */
        __m256i a_lo = _mm256_shuffle_epi8(lookup, _mm256_and_si256(agree_bits, low_mask));
        __m256i a_hi = _mm256_shuffle_epi8(lookup, _mm256_and_si256(_mm256_srli_epi16(agree_bits, 4), low_mask));
        __m256i c_lo = _mm256_shuffle_epi8(lookup, _mm256_and_si256(conf_bits, low_mask));
        __m256i c_hi = _mm256_shuffle_epi8(lookup, _mm256_and_si256(_mm256_srli_epi16(conf_bits, 4), low_mask));
        
        __m256i a_cnt = _mm256_add_epi8(a_lo, a_hi);
        __m256i c_cnt = _mm256_add_epi8(c_lo, c_hi);
        
        /* Horizontal sum */
        __m256i a_sum = _mm256_sad_epu8(a_cnt, _mm256_setzero_si256());
        __m256i c_sum = _mm256_sad_epu8(c_cnt, _mm256_setzero_si256());
        
        agree += _mm256_extract_epi64(a_sum, 0) + _mm256_extract_epi64(a_sum, 1) +
                 _mm256_extract_epi64(a_sum, 2) + _mm256_extract_epi64(a_sum, 3);
        conflict += _mm256_extract_epi64(c_sum, 0) + _mm256_extract_epi64(c_sum, 1) +
                    _mm256_extract_epi64(c_sum, 2) + _mm256_extract_epi64(c_sum, 3);
    }
    
    /* Remainder with scalar */
    for (; i < n_bytes; i++) {
        agree += popcount_byte(q_pos[i] & d_pos[i]);
        agree += popcount_byte(q_neg[i] & d_neg[i]);
        conflict += popcount_byte(q_pos[i] & d_neg[i]);
        conflict += popcount_byte(q_neg[i] & d_pos[i]);
    }
    
    return agree - conflict;
}

#endif /* COOPER_USE_AVX2 */

/* ============================================================================
 * PUBLIC API
 * ============================================================================ */

const char* cooper_ops_version(void) {
    return COOPER_OPS_VERSION;
}

void cooper_ops_init(void) {
    /* Nothing to do - SIMD detected at compile time */
}

int cooper_has_neon(void) {
    return COOPER_USE_NEON;
}

int cooper_has_avx2(void) {
    return COOPER_USE_AVX2;
}

int32_t cooper_ternary_similarity(
    const uint8_t* q_pos, const uint8_t* q_neg,
    const uint8_t* d_pos, const uint8_t* d_neg,
    size_t packed_bytes
) {
    #if COOPER_USE_NEON
    return similarity_neon(q_pos, q_neg, d_pos, d_neg, packed_bytes);
    #elif COOPER_USE_AVX2
    return similarity_avx2(q_pos, q_neg, d_pos, d_neg, packed_bytes);
    #else
    return similarity_scalar(q_pos, q_neg, d_pos, d_neg, packed_bytes);
    #endif
}

void cooper_batch_similarity(
    const uint8_t* q_pos, const uint8_t* q_neg,
    const uint8_t* d_pos, const uint8_t* d_neg,
    int32_t* scores,
    size_t n_docs, size_t packed_bytes
) {
    #if COOPER_USE_NEON
    batch_similarity_neon(q_pos, q_neg, d_pos, d_neg, scores, n_docs, packed_bytes);
    #else
    /* Scalar fallback */
    for (size_t doc = 0; doc < n_docs; doc++) {
        const uint8_t* dp = d_pos + doc * packed_bytes;
        const uint8_t* dn = d_neg + doc * packed_bytes;
        scores[doc] = cooper_ternary_similarity(q_pos, q_neg, dp, dn, packed_bytes);
    }
    #endif
}

void cooper_derive_coarse(
    const uint8_t* fine_pos, const uint8_t* fine_neg,
    uint8_t* coarse_pos, uint8_t* coarse_neg,
    size_t fine_dims, size_t pool_factor
) {
    size_t coarse_dims = fine_dims / pool_factor;
    size_t coarse_bytes = (coarse_dims + 7) / 8;
    
    memset(coarse_pos, 0, coarse_bytes);
    memset(coarse_neg, 0, coarse_bytes);
    
    for (size_t c = 0; c < coarse_dims; c++) {
        int sum = 0;
        
        /* Sum pool_factor fine values */
        for (size_t p = 0; p < pool_factor; p++) {
            size_t f = c * pool_factor + p;
            size_t byte_idx = f / 8;
            size_t bit_idx = f % 8;
            
            int pos_bit = (fine_pos[byte_idx] >> bit_idx) & 1;
            int neg_bit = (fine_neg[byte_idx] >> bit_idx) & 1;
            
            sum += pos_bit - neg_bit;
        }
        
        /* Sign of sum determines coarse value */
        size_t c_byte = c / 8;
        size_t c_bit = c % 8;
        
        if (sum > 0) {
            coarse_pos[c_byte] |= (1 << c_bit);
        } else if (sum < 0) {
            coarse_neg[c_byte] |= (1 << c_bit);
        }
        /* sum == 0: leave both as 0 */
    }
}

void cooper_pack_ternary(
    const int8_t* ternary,
    uint8_t* pos_out,
    uint8_t* neg_out,
    size_t n_dims
) {
    size_t n_bytes = (n_dims + 7) / 8;
    memset(pos_out, 0, n_bytes);
    memset(neg_out, 0, n_bytes);
    
    for (size_t i = 0; i < n_dims; i++) {
        size_t byte_idx = i / 8;
        size_t bit_idx = i % 8;
        
        if (ternary[i] > 0) {
            pos_out[byte_idx] |= (1 << bit_idx);
        } else if (ternary[i] < 0) {
            neg_out[byte_idx] |= (1 << bit_idx);
        }
    }
}

void cooper_unpack_ternary(
    const uint8_t* pos,
    const uint8_t* neg,
    int8_t* ternary_out,
    size_t n_dims
) {
    for (size_t i = 0; i < n_dims; i++) {
        size_t byte_idx = i / 8;
        size_t bit_idx = i % 8;
        
        int pos_bit = (pos[byte_idx] >> bit_idx) & 1;
        int neg_bit = (neg[byte_idx] >> bit_idx) & 1;
        
        ternary_out[i] = (int8_t)(pos_bit - neg_bit);
    }
}

int32_t cooper_hamming_distance(
    const uint8_t* a,
    const uint8_t* b,
    size_t packed_bytes
) {
    int32_t dist = 0;
    
    #if COOPER_USE_NEON
    size_t i = 0;
    for (; i + 16 <= packed_bytes; i += 16) {
        uint8x16_t va = vld1q_u8(a + i);
        uint8x16_t vb = vld1q_u8(b + i);
        uint8x16_t diff = veorq_u8(va, vb);
        uint8x16_t cnt = vcntq_u8(diff);
        #if defined(__aarch64__)
        dist += vaddvq_u8(cnt);
        #else
        uint8x8_t sum8 = vadd_u8(vget_low_u8(cnt), vget_high_u8(cnt));
        uint16x4_t sum4 = vpaddl_u8(sum8);
        uint32x2_t sum2 = vpaddl_u16(sum4);
        dist += vget_lane_u32(sum2, 0) + vget_lane_u32(sum2, 1);
        #endif
    }
    for (; i < packed_bytes; i++) {
        dist += popcount_byte(a[i] ^ b[i]);
    }
    #else
    for (size_t i = 0; i < packed_bytes; i++) {
        dist += popcount_byte(a[i] ^ b[i]);
    }
    #endif
    
    return dist;
}

int32_t cooper_popcount(const uint8_t* bits, size_t packed_bytes) {
    #if COOPER_USE_NEON
    return popcount_neon(bits, packed_bytes);
    #elif COOPER_USE_AVX2
    return popcount_avx2(bits, packed_bytes);
    #else
    return popcount_scalar(bits, packed_bytes);
    #endif
}
