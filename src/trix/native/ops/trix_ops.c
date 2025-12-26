/**
 * TriX Native Operations - Implementation
 * 
 * Self-hosted computation. No BLAS. No external math.
 * Ternary weights = routing, not multiplication.
 */

#include "trix_ops.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TRIX_OPS_VERSION "1.2.0"  /* Added NEON + AVX2 support */

/* ============================================================================
 * AVX2 OPTIMIZATION HELPERS (x86)
 * ============================================================================ */

#if TRIX_USE_AVX2

/* AVX2 horizontal sum of __m256 (8 floats) */
static inline float avx2_hsum_f32(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    lo = _mm_add_ss(lo, shuf);
    return _mm_cvtss_f32(lo);
}

/* AVX2 ternary dot product: processes 8 floats at a time */
static inline float avx2_ternary_dot(
    const float* x,
    const uint8_t* pos_bits,
    const uint8_t* neg_bits,
    size_t dim
) {
    __m256 pos_acc = _mm256_setzero_ps();
    __m256 neg_acc = _mm256_setzero_ps();
    
    size_t i = 0;
    
    /* Process 8 elements at a time (1 byte of bitmask) */
    for (; i + 8 <= dim; i += 8) {
        size_t byte_idx = i / 8;
        uint8_t pos_byte = pos_bits[byte_idx];
        uint8_t neg_byte = neg_bits[byte_idx];
        
        __m256 x_vec = _mm256_loadu_ps(&x[i]);
        
        /* Build mask vectors from bits */
        float pos_mask[8], neg_mask[8];
        for (int b = 0; b < 8; b++) {
            pos_mask[b] = (pos_byte & (1 << b)) ? 1.0f : 0.0f;
            neg_mask[b] = (neg_byte & (1 << b)) ? 1.0f : 0.0f;
        }
        __m256 pos_m = _mm256_loadu_ps(pos_mask);
        __m256 neg_m = _mm256_loadu_ps(neg_mask);
        
        pos_acc = _mm256_fmadd_ps(x_vec, pos_m, pos_acc);
        neg_acc = _mm256_fmadd_ps(x_vec, neg_m, neg_acc);
    }
    
    float pos_sum = avx2_hsum_f32(pos_acc);
    float neg_sum = avx2_hsum_f32(neg_acc);
    
    /* Handle remainder with scalar code */
    for (; i < dim; i++) {
        size_t byte_idx = i / 8;
        size_t bit_idx = i % 8;
        
        if (pos_bits[byte_idx] & (1 << bit_idx)) {
            pos_sum += x[i];
        }
        if (neg_bits[byte_idx] & (1 << bit_idx)) {
            neg_sum += x[i];
        }
    }
    
    return pos_sum - neg_sum;
}

#endif /* TRIX_USE_AVX2 */

/* ============================================================================
 * NEON OPTIMIZATION HELPERS (ARM)
 * ============================================================================ */

#if TRIX_USE_NEON

/* NEON horizontal sum of float32x4 */
static inline float neon_hsum_f32(float32x4_t v) {
    float32x2_t sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    sum = vpadd_f32(sum, sum);
    return vget_lane_f32(sum, 0);
}

/* NEON ternary dot product: sum of x[i] where w=+1, minus sum where w=-1 */
static inline float neon_ternary_dot(
    const float* x,
    const uint8_t* pos_bits,
    const uint8_t* neg_bits,
    size_t dim
) {
    float32x4_t pos_acc = vdupq_n_f32(0.0f);
    float32x4_t neg_acc = vdupq_n_f32(0.0f);
    
    size_t i = 0;
    
    /* Process 32 elements at a time (4 bytes of bitmask = 32 bits) */
    for (; i + 32 <= dim; i += 32) {
        for (int byte = 0; byte < 4; byte++) {
            uint8_t pos_byte = pos_bits[(i / 8) + byte];
            uint8_t neg_byte = neg_bits[(i / 8) + byte];
            
            /* Process 8 elements per byte, 4 at a time with NEON */
            size_t base = i + byte * 8;
            
            /* First 4 elements */
            float32x4_t x_vec = vld1q_f32(&x[base]);
            
            /* Build mask vectors from bits */
            float pos_mask[4], neg_mask[4];
            for (int b = 0; b < 4; b++) {
                pos_mask[b] = (pos_byte & (1 << b)) ? 1.0f : 0.0f;
                neg_mask[b] = (neg_byte & (1 << b)) ? 1.0f : 0.0f;
            }
            float32x4_t pos_m = vld1q_f32(pos_mask);
            float32x4_t neg_m = vld1q_f32(neg_mask);
            
            pos_acc = vmlaq_f32(pos_acc, x_vec, pos_m);
            neg_acc = vmlaq_f32(neg_acc, x_vec, neg_m);
            
            /* Second 4 elements */
            x_vec = vld1q_f32(&x[base + 4]);
            for (int b = 0; b < 4; b++) {
                pos_mask[b] = (pos_byte & (1 << (b + 4))) ? 1.0f : 0.0f;
                neg_mask[b] = (neg_byte & (1 << (b + 4))) ? 1.0f : 0.0f;
            }
            pos_m = vld1q_f32(pos_mask);
            neg_m = vld1q_f32(neg_mask);
            
            pos_acc = vmlaq_f32(pos_acc, x_vec, pos_m);
            neg_acc = vmlaq_f32(neg_acc, x_vec, neg_m);
        }
    }
    
    float pos_sum = neon_hsum_f32(pos_acc);
    float neg_sum = neon_hsum_f32(neg_acc);
    
    /* Handle remainder with scalar code */
    for (; i < dim; i++) {
        size_t byte_idx = i / 8;
        size_t bit_idx = i % 8;
        
        if (pos_bits[byte_idx] & (1 << bit_idx)) {
            pos_sum += x[i];
        }
        if (neg_bits[byte_idx] & (1 << bit_idx)) {
            neg_sum += x[i];
        }
    }
    
    return pos_sum - neg_sum;
}

#endif /* TRIX_USE_NEON */

/* ============================================================================
 * INITIALIZATION
 * ============================================================================ */

void trix_ops_init(void) {
    /* NEON is compile-time detected via TRIX_USE_NEON */
}

const char* trix_ops_version(void) {
    return TRIX_OPS_VERSION;
}

int trix_has_neon(void) {
#if TRIX_USE_NEON
    return 1;
#else
    return 0;
#endif
}

int trix_has_avx2(void) {
#if TRIX_USE_AVX2
    return 1;
#else
    return 0;
#endif
}

/* ============================================================================
 * TERNARY MATRIX OPERATIONS
 * ============================================================================ */

TernaryMatrix trix_ternary_from_float(const float* weights, size_t rows, size_t cols) {
    TernaryMatrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.packed_cols = (cols + 7) / 8;
    
    size_t total_bytes = rows * mat.packed_cols;
    mat.pos = (uint8_t*)calloc(total_bytes, 1);
    mat.neg = (uint8_t*)calloc(total_bytes, 1);
    
    if (!mat.pos || !mat.neg) {
        free(mat.pos);
        free(mat.neg);
        mat.pos = mat.neg = NULL;
        return mat;
    }
    
    /* Quantize: positive -> pos bit, negative -> neg bit, zero -> neither */
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            float w = weights[i * cols + j];
            size_t byte_idx = i * mat.packed_cols + (j / 8);
            size_t bit_idx = j % 8;
            
            if (w > 0.0f) {
                mat.pos[byte_idx] |= (1 << bit_idx);
            } else if (w < 0.0f) {
                mat.neg[byte_idx] |= (1 << bit_idx);
            }
            /* w == 0: neither bit set */
        }
    }
    
    return mat;
}

void trix_ternary_free(TernaryMatrix* mat) {
    if (mat) {
        free(mat->pos);
        free(mat->neg);
        mat->pos = mat->neg = NULL;
    }
}

/* Scalar implementation of ternary matvec */
static void trix_ternary_matvec_scalar(
    const TernaryMatrix* mat,
    const float* x,
    float* y,
    const float* scale
) {
    for (size_t i = 0; i < mat->rows; i++) {
        float pos_sum = 0.0f;
        float neg_sum = 0.0f;
        
        size_t row_offset = i * mat->packed_cols;
        
        for (size_t byte_idx = 0; byte_idx < mat->packed_cols; byte_idx++) {
            uint8_t pos_bits = mat->pos[row_offset + byte_idx];
            uint8_t neg_bits = mat->neg[row_offset + byte_idx];
            
            size_t col_base = byte_idx * 8;
            
            for (int bit = 0; bit < 8 && col_base + bit < mat->cols; bit++) {
                size_t col = col_base + bit;
                float val = x[col];
                
                if (pos_bits & (1 << bit)) {
                    pos_sum += val;
                }
                if (neg_bits & (1 << bit)) {
                    neg_sum += val;
                }
            }
        }
        
        y[i] = scale[i] * (pos_sum - neg_sum);
    }
}

#if TRIX_USE_NEON
/* NEON-optimized implementation (ARM) */
static void trix_ternary_matvec_neon(
    const TernaryMatrix* mat,
    const float* x,
    float* y,
    const float* scale
) {
    for (size_t i = 0; i < mat->rows; i++) {
        size_t row_offset = i * mat->packed_cols;
        float result = neon_ternary_dot(
            x,
            mat->pos + row_offset,
            mat->neg + row_offset,
            mat->cols
        );
        y[i] = scale[i] * result;
    }
}
#endif

#if TRIX_USE_AVX2
/* AVX2-optimized implementation (x86) */
static void trix_ternary_matvec_avx2(
    const TernaryMatrix* mat,
    const float* x,
    float* y,
    const float* scale
) {
    for (size_t i = 0; i < mat->rows; i++) {
        size_t row_offset = i * mat->packed_cols;
        float result = avx2_ternary_dot(
            x,
            mat->pos + row_offset,
            mat->neg + row_offset,
            mat->cols
        );
        y[i] = scale[i] * result;
    }
}
#endif

void trix_ternary_matvec(
    const TernaryMatrix* mat,
    const float* x,
    float* y,
    const float* scale
) {
    /*
     * Ternary matmul without multiplication:
     * y[i] = scale[i] * (sum_{w=+1}(x[j]) - sum_{w=-1}(x[j]))
     * 
     * Dispatches to best available SIMD implementation.
     */
#if TRIX_USE_NEON
    trix_ternary_matvec_neon(mat, x, y, scale);
#elif TRIX_USE_AVX2
    trix_ternary_matvec_avx2(mat, x, y, scale);
#else
    trix_ternary_matvec_scalar(mat, x, y, scale);
#endif
}

void trix_ternary_matmul(
    const TernaryMatrix* mat,
    const float* X,
    float* Y,
    const float* scale,
    size_t batch
) {
    /* Batched version: apply matvec to each row of X */
    for (size_t b = 0; b < batch; b++) {
        trix_ternary_matvec(
            mat,
            X + b * mat->cols,
            Y + b * mat->rows,
            scale
        );
    }
}

void trix_ternary_matvec_t(
    const TernaryMatrix* mat,
    const float* x,
    float* y,
    const float* scale
) {
    /*
     * Transposed matmul: y = W^T @ x
     * 
     * y[j] = sum_i(w[i,j] * x[i])
     *      = sum_i(scale[i] * sign[i,j] * x[i])
     * 
     * We accumulate contributions from each row.
     */
    
    /* Zero output */
    memset(y, 0, mat->cols * sizeof(float));
    
    for (size_t i = 0; i < mat->rows; i++) {
        float scaled_x = scale[i] * x[i];
        size_t row_offset = i * mat->packed_cols;
        
        for (size_t byte_idx = 0; byte_idx < mat->packed_cols; byte_idx++) {
            uint8_t pos_bits = mat->pos[row_offset + byte_idx];
            uint8_t neg_bits = mat->neg[row_offset + byte_idx];
            
            size_t col_base = byte_idx * 8;
            
            for (int bit = 0; bit < 8 && col_base + bit < mat->cols; bit++) {
                size_t col = col_base + bit;
                
                if (pos_bits & (1 << bit)) {
                    y[col] += scaled_x;
                }
                if (neg_bits & (1 << bit)) {
                    y[col] -= scaled_x;
                }
            }
        }
    }
}

/* ============================================================================
 * REDUCTION OPERATIONS
 * ============================================================================ */

float trix_sum(const float* x, size_t n) {
    /*
     * Pairwise summation for numerical stability.
     * Conceptually: tree of full_adders.
     * 
     * For large n, this is O(n) work, O(log n) depth.
     */
    if (n == 0) return 0.0f;
    if (n == 1) return x[0];
    if (n == 2) return x[0] + x[1];
    
    /* Recursive pairwise sum */
    size_t mid = n / 2;
    float left = trix_sum(x, mid);
    float right = trix_sum(x + mid, n - mid);
    return left + right;
}

float trix_dot(const float* a, const float* b, size_t n) {
    /*
     * Dot product via pairwise summation.
     * 
     * For ternary b in {-1, 0, +1}, this becomes routing:
     *   sum(a[i] * b[i]) = sum_{b=+1}(a[i]) - sum_{b=-1}(a[i])
     * 
     * Here we handle general float for flexibility.
     */
    if (n == 0) return 0.0f;
    
    /* Allocate temp for products (could optimize with streaming) */
    float* prod = (float*)malloc(n * sizeof(float));
    if (!prod) return 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        prod[i] = a[i] * b[i];
    }
    
    float result = trix_sum(prod, n);
    free(prod);
    return result;
}

float trix_norm(const float* x, size_t n) {
    /*
     * L2 norm = sqrt(sum(x^2))
     */
    if (n == 0) return 0.0f;
    
    float* sq = (float*)malloc(n * sizeof(float));
    if (!sq) return 0.0f;
    
    for (size_t i = 0; i < n; i++) {
        sq[i] = x[i] * x[i];
    }
    
    float sum_sq = trix_sum(sq, n);
    free(sq);
    return sqrtf(sum_sq);
}

/* ============================================================================
 * TILE OPERATIONS
 * ============================================================================ */

TrixTile* trix_tile_create(size_t d_model, size_t d_hidden) {
    TrixTile* tile = (TrixTile*)calloc(1, sizeof(TrixTile));
    if (!tile) return NULL;
    
    tile->d_model = d_model;
    tile->d_hidden = d_hidden;
    
    /* Allocate bias and scales */
    tile->bias = (float*)calloc(d_hidden, sizeof(float));
    tile->scales_up = (float*)malloc(d_hidden * sizeof(float));
    tile->scales_down = (float*)malloc(d_model * sizeof(float));
    
    if (!tile->bias || !tile->scales_up || !tile->scales_down) {
        trix_tile_free(tile);
        return NULL;
    }
    
    /* Default scales to 1.0 */
    for (size_t i = 0; i < d_hidden; i++) tile->scales_up[i] = 1.0f;
    for (size_t i = 0; i < d_model; i++) tile->scales_down[i] = 1.0f;
    
    return tile;
}

void trix_tile_free(TrixTile* tile) {
    if (tile) {
        trix_ternary_free(&tile->weights_up);
        trix_ternary_free(&tile->weights_down);
        free(tile->bias);
        free(tile->scales_up);
        free(tile->scales_down);
        free(tile->cached_input);
        free(tile->cached_hidden);
        free(tile);
    }
}

void trix_tile_init_weights(
    TrixTile* tile,
    const float* weights_up,
    const float* weights_down,
    const float* bias,
    const float* scales_up,
    const float* scales_down
) {
    /* Free existing ternary matrices */
    trix_ternary_free(&tile->weights_up);
    trix_ternary_free(&tile->weights_down);
    
    /* Quantize weights to ternary */
    tile->weights_up = trix_ternary_from_float(weights_up, tile->d_hidden, tile->d_model);
    tile->weights_down = trix_ternary_from_float(weights_down, tile->d_model, tile->d_hidden);
    
    /* Copy bias and scales */
    if (bias) memcpy(tile->bias, bias, tile->d_hidden * sizeof(float));
    if (scales_up) memcpy(tile->scales_up, scales_up, tile->d_hidden * sizeof(float));
    if (scales_down) memcpy(tile->scales_down, scales_down, tile->d_model * sizeof(float));
}

void trix_tile_forward(
    TrixTile* tile,
    const float* x,
    float* y,
    size_t batch
) {
    /*
     * Forward pass:
     *   hidden = ReLU(W_up @ x + bias)
     *   output = W_down @ hidden
     * 
     * Using ternary matmul (no actual multiplication in W @ x).
     */
    
    /* Allocate/resize cache */
    size_t input_size = batch * tile->d_model;
    size_t hidden_size = batch * tile->d_hidden;
    
    tile->cached_input = (float*)realloc(tile->cached_input, input_size * sizeof(float));
    tile->cached_hidden = (float*)realloc(tile->cached_hidden, hidden_size * sizeof(float));
    tile->cached_batch = batch;
    
    /* Cache input for backward */
    memcpy(tile->cached_input, x, input_size * sizeof(float));
    
    /* Up projection: hidden = W_up @ x */
    float* pre_act = (float*)malloc(hidden_size * sizeof(float));
    trix_ternary_matmul(&tile->weights_up, x, pre_act, tile->scales_up, batch);
    
    /* Add bias and apply ReLU */
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < tile->d_hidden; h++) {
            size_t idx = b * tile->d_hidden + h;
            float val = pre_act[idx] + tile->bias[h];
            tile->cached_hidden[idx] = shape_relu(val);
        }
    }
    free(pre_act);
    
    /* Down projection: output = W_down @ hidden */
    trix_ternary_matmul(&tile->weights_down, tile->cached_hidden, y, tile->scales_down, batch);
}

void trix_tile_backward(
    TrixTile* tile,
    const float* d_y,
    float* d_x,
    size_t batch
) {
    /*
     * Backward pass:
     *   d_hidden = W_down^T @ d_y
     *   d_hidden *= ReLU'(pre_act)  [using cached hidden > 0]
     *   d_x = W_up^T @ d_hidden
     * 
     * Note: This computes d_x only. Weight gradients would need
     * additional accumulators (for training, use with optimizer).
     */
    
    float* d_hidden = (float*)malloc(batch * tile->d_hidden * sizeof(float));
    
    /* d_hidden = W_down^T @ d_y */
    for (size_t b = 0; b < batch; b++) {
        trix_ternary_matvec_t(
            &tile->weights_down,
            d_y + b * tile->d_model,
            d_hidden + b * tile->d_hidden,
            tile->scales_down
        );
    }
    
    /* Apply ReLU gradient: d_hidden *= (cached_hidden > 0) */
    for (size_t i = 0; i < batch * tile->d_hidden; i++) {
        d_hidden[i] *= shape_relu_grad(tile->cached_hidden[i]);
    }
    
    /* d_x = W_up^T @ d_hidden */
    for (size_t b = 0; b < batch; b++) {
        trix_ternary_matvec_t(
            &tile->weights_up,
            d_hidden + b * tile->d_hidden,
            d_x + b * tile->d_model,
            tile->scales_up
        );
    }
    
    free(d_hidden);
}

/* ============================================================================
 * ROUTING
 * ============================================================================ */

size_t trix_hamming_distance(const uint8_t* a, const uint8_t* b, size_t bytes) {
    size_t dist = 0;
    for (size_t i = 0; i < bytes; i++) {
        dist += popcount8(a[i] ^ b[i]);
    }
    return dist;
}

size_t trix_route_hamming(
    const uint8_t* input,
    const uint8_t* signatures,
    size_t num_sigs,
    size_t packed_dim
) {
    size_t best_idx = 0;
    size_t best_dist = (size_t)-1;
    
    for (size_t i = 0; i < num_sigs; i++) {
        size_t dist = trix_hamming_distance(input, signatures + i * packed_dim, packed_dim);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    
    return best_idx;
}

void trix_binarize(const float* x, uint8_t* out, size_t dim) {
    size_t packed = (dim + 7) / 8;
    memset(out, 0, packed);
    
    for (size_t i = 0; i < dim; i++) {
        if (x[i] >= 0.0f) {
            out[i / 8] |= (1 << (i % 8));
        }
    }
}

/* ============================================================================
 * BINARY FROZEN SHAPES - Vectorized Application
 * ============================================================================ */

void trix_apply_binary_xor(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t bytes) {
    for (size_t i = 0; i < bytes; i++) {
        out[i] = a[i] ^ b[i];
    }
}

void trix_apply_binary_and(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t bytes) {
    for (size_t i = 0; i < bytes; i++) {
        out[i] = a[i] & b[i];
    }
}

void trix_apply_binary_or(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t bytes) {
    for (size_t i = 0; i < bytes; i++) {
        out[i] = a[i] | b[i];
    }
}

void trix_apply_binary_not(const uint8_t* a, uint8_t* out, size_t bytes) {
    for (size_t i = 0; i < bytes; i++) {
        out[i] = ~a[i];
    }
}
