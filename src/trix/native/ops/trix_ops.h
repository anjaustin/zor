/**
 * TriX Native Operations
 * 
 * Self-hosted computation primitives. No external math libraries.
 * Matrix multiply via ternary routing. Activation via frozen shapes.
 * 
 * "The shape IS the computation."
 */

#ifndef TRIX_OPS_H
#define TRIX_OPS_H

#include <stdint.h>
#include <stddef.h>

/* SIMD detection */

/* ARM NEON */
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define TRIX_USE_NEON 1
#else
#define TRIX_USE_NEON 0
#endif

/* x86 AVX2 */
#if defined(__AVX2__)
#include <immintrin.h>
#define TRIX_USE_AVX2 1
#else
#define TRIX_USE_AVX2 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * TYPES
 * ============================================================================ */

/* Ternary weight: -1, 0, +1 encoded as two bits */
typedef struct {
    uint8_t* pos;   /* Bitmask: 1 where weight = +1 */
    uint8_t* neg;   /* Bitmask: 1 where weight = -1 */
    size_t rows;
    size_t cols;
    size_t packed_cols;  /* cols / 8, rounded up */
} TernaryMatrix;

/* Dense float array */
typedef struct {
    float* data;
    size_t rows;
    size_t cols;
} FloatMatrix;

/* Tile state for forward/backward */
typedef struct {
    TernaryMatrix weights_up;
    TernaryMatrix weights_down;
    float* bias;
    float* scales_up;
    float* scales_down;
    size_t d_model;
    size_t d_hidden;
    
    /* Cached for backward */
    float* cached_input;
    float* cached_hidden;
    size_t cached_batch;
} TrixTile;

/* ============================================================================
 * FROZEN SHAPES - Mathematical Truths
 * ============================================================================ */

/**
 * ReLU: max(0, x)
 * The simplest frozen shape.
 */
static inline float shape_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

/**
 * ReLU derivative: 1 if x > 0, else 0
 */
static inline float shape_relu_grad(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

/**
 * XOR polynomial: a + b - 2ab
 * On binary inputs {0,1}, equals hardware XOR.
 */
static inline float shape_xor(float a, float b) {
    return a + b - 2.0f * a * b;
}

/**
 * AND polynomial: ab
 */
static inline float shape_and(float a, float b) {
    return a * b;
}

/**
 * OR polynomial: a + b - ab
 */
static inline float shape_or(float a, float b) {
    return a + b - a * b;
}

/**
 * NOT polynomial: 1 - a
 */
static inline float shape_not(float a) {
    return 1.0f - a;
}

/**
 * Half adder: (sum, carry) from 2 bits
 * sum = XOR(a, b)
 * carry = AND(a, b)
 */
static inline void shape_half_adder(float a, float b, float* sum, float* carry) {
    *sum = shape_xor(a, b);
    *carry = shape_and(a, b);
}

/**
 * Full adder: (sum, carry) from 3 bits
 * sum = XOR(XOR(a, b), cin)
 * carry = OR(AND(a, b), AND(XOR(a, b), cin))
 */
static inline void shape_full_adder(float a, float b, float cin, float* sum, float* carry) {
    float ab_xor = shape_xor(a, b);
    *sum = shape_xor(ab_xor, cin);
    *carry = shape_or(shape_and(a, b), shape_and(ab_xor, cin));
}

/* ============================================================================
 * TERNARY MATRIX OPERATIONS
 * ============================================================================ */

/**
 * Create a ternary matrix from dense float weights.
 * Quantizes to {-1, 0, +1} based on sign.
 */
TernaryMatrix trix_ternary_from_float(const float* weights, size_t rows, size_t cols);

/**
 * Free ternary matrix memory.
 */
void trix_ternary_free(TernaryMatrix* mat);

/**
 * Ternary matrix-vector multiply.
 * 
 * For ternary weights {-1, 0, +1}, no actual multiplication needed:
 *   y[i] = sum_j(x[j] * w[i,j])
 *        = sum_{w=+1}(x[j]) - sum_{w=-1}(x[j])
 * 
 * Just routing and sign flips. O(nnz) adds, zero multiplies.
 * 
 * @param mat     Ternary weight matrix [out_dim, in_dim]
 * @param x       Input vector [in_dim]
 * @param y       Output vector [out_dim] (must be pre-allocated)
 * @param scale   Per-output scaling factor [out_dim]
 */
void trix_ternary_matvec(
    const TernaryMatrix* mat,
    const float* x,
    float* y,
    const float* scale
);

/**
 * Batched ternary matrix multiply.
 * 
 * @param mat     Ternary weight matrix [out_dim, in_dim]
 * @param X       Input matrix [batch, in_dim]
 * @param Y       Output matrix [batch, out_dim]
 * @param scale   Per-output scaling [out_dim]
 * @param batch   Batch size
 */
void trix_ternary_matmul(
    const TernaryMatrix* mat,
    const float* X,
    float* Y,
    const float* scale,
    size_t batch
);

/**
 * Ternary matrix-vector multiply, transposed.
 * Used in backward pass: d_input = W^T @ d_output
 */
void trix_ternary_matvec_t(
    const TernaryMatrix* mat,
    const float* x,
    float* y,
    const float* scale
);

/* ============================================================================
 * REDUCTION OPERATIONS (via adder trees)
 * ============================================================================ */

/**
 * Sum reduction using adder tree structure.
 * Conceptually: tree of full_adders.
 * In practice: pairwise summation for numerical stability.
 */
float trix_sum(const float* x, size_t n);

/**
 * Dot product via adder tree.
 * For ternary: uses sign routing instead of multiply.
 */
float trix_dot(const float* a, const float* b, size_t n);

/**
 * L2 norm via adder tree.
 * sqrt(sum(x^2))
 */
float trix_norm(const float* x, size_t n);

/* ============================================================================
 * TILE OPERATIONS
 * ============================================================================ */

/**
 * Create a tile with given dimensions.
 */
TrixTile* trix_tile_create(size_t d_model, size_t d_hidden);

/**
 * Free tile memory.
 */
void trix_tile_free(TrixTile* tile);

/**
 * Initialize tile weights from float arrays.
 */
void trix_tile_init_weights(
    TrixTile* tile,
    const float* weights_up,
    const float* weights_down,
    const float* bias,
    const float* scales_up,
    const float* scales_down
);

/**
 * Forward pass through tile.
 * 
 * hidden = ReLU(ternary_matmul(x, W_up) + bias)
 * output = ternary_matmul(hidden, W_down)
 * 
 * @param tile    The tile
 * @param x       Input [batch, d_model]
 * @param y       Output [batch, d_model]
 * @param batch   Batch size
 */
void trix_tile_forward(
    TrixTile* tile,
    const float* x,
    float* y,
    size_t batch
);

/**
 * Backward pass through tile.
 * Computes gradients w.r.t. input.
 * 
 * @param tile    The tile (uses cached activations)
 * @param d_y     Gradient of loss w.r.t. output [batch, d_model]
 * @param d_x     Gradient of loss w.r.t. input [batch, d_model]
 * @param batch   Batch size
 */
void trix_tile_backward(
    TrixTile* tile,
    const float* d_y,
    float* d_x,
    size_t batch
);

/* ============================================================================
 * ROUTING (Hamming distance)
 * ============================================================================ */

/**
 * Compute Hamming distance between binary vectors.
 * Uses XOR + popcount.
 */
size_t trix_hamming_distance(const uint8_t* a, const uint8_t* b, size_t bytes);

/**
 * Route input to best-matching signature via Hamming distance.
 * 
 * @param input         Binarized input [packed_dim]
 * @param signatures    Array of signatures [num_sigs][packed_dim]
 * @param num_sigs      Number of signatures
 * @param packed_dim    Dimension in bytes
 * @return              Index of closest signature
 */
size_t trix_route_hamming(
    const uint8_t* input,
    const uint8_t* signatures,
    size_t num_sigs,
    size_t packed_dim
);

/**
 * Binarize float vector to packed bits.
 * x >= 0 -> 1, x < 0 -> 0
 */
void trix_binarize(const float* x, uint8_t* out, size_t dim);

/* ============================================================================
 * UTILITY
 * ============================================================================ */

/**
 * Fast popcount for byte.
 */
static inline int popcount8(uint8_t x) {
    x = (x & 0x55) + ((x >> 1) & 0x55);
    x = (x & 0x33) + ((x >> 2) & 0x33);
    x = (x & 0x0F) + ((x >> 4) & 0x0F);
    return x;
}

/* ============================================================================
 * BINARY FROZEN SHAPES - Pure Bitwise Operations
 * ============================================================================
 * 
 * On binary inputs {0, 1}, polynomial shapes equal bitwise operations.
 * These are the FROZEN forms - no gradients, maximum speed.
 * 
 * Use for inference after training with polynomial forms.
 */

/**
 * Binary XOR: a ^ b
 * Polynomial equivalent: a + b - 2ab
 */
static inline uint8_t shape_xor_binary(uint8_t a, uint8_t b) {
    return a ^ b;
}

/**
 * Binary AND: a & b
 * Polynomial equivalent: ab
 */
static inline uint8_t shape_and_binary(uint8_t a, uint8_t b) {
    return a & b;
}

/**
 * Binary OR: a | b
 * Polynomial equivalent: a + b - ab
 */
static inline uint8_t shape_or_binary(uint8_t a, uint8_t b) {
    return a | b;
}

/**
 * Binary NOT: ~a
 * Polynomial equivalent: 1 - a
 */
static inline uint8_t shape_not_binary(uint8_t a) {
    return ~a;
}

/**
 * Binary NAND: ~(a & b)
 * Polynomial equivalent: 1 - ab
 */
static inline uint8_t shape_nand_binary(uint8_t a, uint8_t b) {
    return ~(a & b);
}

/**
 * Binary NOR: ~(a | b)
 * Polynomial equivalent: 1 - a - b + ab
 */
static inline uint8_t shape_nor_binary(uint8_t a, uint8_t b) {
    return ~(a | b);
}

/**
 * Binary XNOR: ~(a ^ b)
 * Polynomial equivalent: 1 - a - b + 2ab
 */
static inline uint8_t shape_xnor_binary(uint8_t a, uint8_t b) {
    return ~(a ^ b);
}

/**
 * Binary half adder: (sum, carry)
 * sum = a ^ b
 * carry = a & b
 */
static inline void shape_half_adder_binary(uint8_t a, uint8_t b, uint8_t* sum, uint8_t* carry) {
    *sum = a ^ b;
    *carry = a & b;
}

/**
 * Binary full adder: (sum, carry)
 * sum = a ^ b ^ cin
 * carry = (a & b) | ((a ^ b) & cin)
 */
static inline void shape_full_adder_binary(uint8_t a, uint8_t b, uint8_t cin, uint8_t* sum, uint8_t* carry) {
    uint8_t ab_xor = a ^ b;
    *sum = ab_xor ^ cin;
    *carry = (a & b) | (ab_xor & cin);
}

/**
 * Apply binary shape to packed byte arrays.
 * Processes 8 bits per operation.
 */
void trix_apply_binary_xor(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t bytes);
void trix_apply_binary_and(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t bytes);
void trix_apply_binary_or(const uint8_t* a, const uint8_t* b, uint8_t* out, size_t bytes);
void trix_apply_binary_not(const uint8_t* a, uint8_t* out, size_t bytes);

/**
 * Initialize library (call once at startup).
 */
void trix_ops_init(void);

/**
 * Get version string.
 */
const char* trix_ops_version(void);

/**
 * Check if NEON SIMD is available (ARM).
 */
int trix_has_neon(void);

/**
 * Check if AVX2 SIMD is available (x86).
 */
int trix_has_avx2(void);

#ifdef __cplusplus
}
#endif

#endif /* TRIX_OPS_H */
