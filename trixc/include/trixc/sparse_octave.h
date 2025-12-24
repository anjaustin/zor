/*
 * Sparse Octave Lookup - Pure Frozen Shapes Implementation
 *
 * Multi-scale sparse memory lookup using only frozen shapes.
 * No CUDA. No frameworks. Just math.
 *
 * "Information lives at different scales. Capture it where it lives."
 *
 * All operations are frozen shapes:
 * - Hamming distance: XOR + popcount (frozen)
 * - Softmax: exp + div (frozen)
 * - Top-k: comparison (frozen)
 * - Blend: mul + add (frozen)
 *
 * Created by: Tripp + Claude
 * Date: December 2025
 */

#ifndef TRIXC_SPARSE_OCTAVE_H
#define TRIXC_SPARSE_OCTAVE_H

#include "apu.h"
#include "shapes.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <alloca.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Configuration
 * ═══════════════════════════════════════════════════════════════════════════ */

#define TRIX_MAX_OCTAVES 4
#define TRIX_DEFAULT_TOP_K 16

/* ═══════════════════════════════════════════════════════════════════════════
 * Providence: Content-Addressed Memory
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int d_model;
    int memory_size;
    float* keys;      /* [memory_size, d_model] */
    float* values;    /* [memory_size, d_model] */
} trix_providence_t;

/**
 * Initialize Providence memory
 */
static inline void trix_providence_init(
    trix_providence_t* prov,
    int d_model,
    int memory_size
) {
    prov->d_model = d_model;
    prov->memory_size = memory_size;
    prov->keys = (float*)calloc(memory_size * d_model, sizeof(float));
    prov->values = (float*)calloc(memory_size * d_model, sizeof(float));

    /* Random initialization */
    for (int i = 0; i < memory_size * d_model; i++) {
        prov->keys[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.04f;
        prov->values[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.04f;
    }
}

/**
 * Free Providence memory
 */
static inline void trix_providence_free(trix_providence_t* prov) {
    free(prov->keys);
    free(prov->values);
}

/**
 * Hamming-like distance (L1 as differentiable proxy)
 * This is a frozen shape: d(a,b) = Σ |a_i - b_i|
 */
static inline float trix_hamming_distance(
    const float* a,
    const float* b,
    int len
) {
    float dist = 0.0f;
    for (int i = 0; i < len; i++) {
        dist += fabsf(a[i] - b[i]);
    }
    return dist;
}

/**
 * Soft lookup with attention blending
 * All operations are frozen shapes.
 */
static inline void trix_providence_lookup(
    const trix_providence_t* prov,
    const float* query,     /* [d_model] */
    float* output,          /* [d_model] */
    int top_k,
    float temperature
) {
    int d = prov->d_model;
    int m = prov->memory_size;
    top_k = top_k < m ? top_k : m;

    /* Compute distances to all keys (frozen Hamming shape) */
    float* distances = (float*)alloca(m * sizeof(float));
    for (int i = 0; i < m; i++) {
        distances[i] = trix_hamming_distance(query, &prov->keys[i * d], d);
    }

    /* Find top-k smallest (frozen comparison shapes) */
    int* top_k_idx = (int*)alloca(top_k * sizeof(int));
    float* top_k_dist = (float*)alloca(top_k * sizeof(float));

    /* Simple selection (could be optimized with partial sort) */
    for (int k = 0; k < top_k; k++) {
        float min_dist = INFINITY;
        int min_idx = 0;
        for (int i = 0; i < m; i++) {
            if (distances[i] < min_dist) {
                min_dist = distances[i];
                min_idx = i;
            }
        }
        top_k_idx[k] = min_idx;
        top_k_dist[k] = min_dist;
        distances[min_idx] = INFINITY;  /* Mark as used */
    }

    /* Compute attention weights (frozen softmax shape) */
    float* weights = (float*)alloca(top_k * sizeof(float));
    float max_neg_dist = -top_k_dist[0] / temperature;
    for (int k = 1; k < top_k; k++) {
        float nd = -top_k_dist[k] / temperature;
        if (nd > max_neg_dist) max_neg_dist = nd;
    }

    float sum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        weights[k] = expf(-top_k_dist[k] / temperature - max_neg_dist);
        sum += weights[k];
    }
    for (int k = 0; k < top_k; k++) {
        weights[k] /= sum;
    }

    /* Weighted sum of values (frozen mul + add shapes) */
    memset(output, 0, d * sizeof(float));
    for (int k = 0; k < top_k; k++) {
        int idx = top_k_idx[k];
        for (int i = 0; i < d; i++) {
            output[i] += weights[k] * prov->values[idx * d + i];
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Sparse Octave Lookup FFN
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    int d_model;
    int n_octaves;
    int memory_size;
    int top_k;
    trix_providence_t octaves[TRIX_MAX_OCTAVES];
    int shifts[TRIX_MAX_OCTAVES];
    float blend_weights[TRIX_MAX_OCTAVES];
    float* blend_bias;    /* [d_model] */
    float* output_scale;  /* [d_model] */
} trix_sparse_octave_t;

/**
 * Initialize Sparse Octave Lookup
 */
static inline void trix_sparse_octave_init(
    trix_sparse_octave_t* sol,
    int d_model,
    int n_octaves,
    int memory_size,
    int top_k
) {
    sol->d_model = d_model;
    sol->n_octaves = n_octaves < TRIX_MAX_OCTAVES ? n_octaves : TRIX_MAX_OCTAVES;
    sol->memory_size = memory_size;
    sol->top_k = top_k;

    /* Initialize each octave */
    int default_shifts[] = {0, 4, 8, 12};
    for (int i = 0; i < sol->n_octaves; i++) {
        trix_providence_init(&sol->octaves[i], d_model, memory_size);
        sol->shifts[i] = default_shifts[i];
        sol->blend_weights[i] = 1.0f / sol->n_octaves;
    }

    /* Initialize output parameters */
    sol->blend_bias = (float*)calloc(d_model, sizeof(float));
    sol->output_scale = (float*)malloc(d_model * sizeof(float));
    for (int i = 0; i < d_model; i++) {
        sol->output_scale[i] = 1.0f;
    }
}

/**
 * Free Sparse Octave Lookup
 */
static inline void trix_sparse_octave_free(trix_sparse_octave_t* sol) {
    for (int i = 0; i < sol->n_octaves; i++) {
        trix_providence_free(&sol->octaves[i]);
    }
    free(sol->blend_bias);
    free(sol->output_scale);
}

/**
 * Extract octave key via bit shift (frozen shape)
 * Simulates bit shifting via quantization.
 */
static inline void trix_extract_octave_key(
    const float* input,
    float* output,
    int d_model,
    int shift
) {
    if (shift == 0) {
        memcpy(output, input, d_model * sizeof(float));
        return;
    }

    /* Frozen quantization shape */
    float scale = (float)(1 << shift);
    for (int i = 0; i < d_model; i++) {
        output[i] = floorf(input[i] * 256.0f / scale) * scale / 256.0f;
    }
}

/**
 * Forward pass through Sparse Octave Lookup
 * All operations are frozen shapes.
 */
static inline void trix_sparse_octave_forward(
    const trix_sparse_octave_t* sol,
    const float* input,    /* [d_model] */
    float* output          /* [d_model] */
) {
    int d = sol->d_model;

    /* Temporary storage for octave outputs */
    float* octave_outputs = (float*)alloca(sol->n_octaves * d * sizeof(float));
    float* key = (float*)alloca(d * sizeof(float));

    /* Process each octave */
    for (int o = 0; o < sol->n_octaves; o++) {
        /* Extract octave-specific key (frozen shape) */
        trix_extract_octave_key(input, key, d, sol->shifts[o]);

        /* Providence lookup (frozen shapes) */
        trix_providence_lookup(
            &sol->octaves[o],
            key,
            &octave_outputs[o * d],
            sol->top_k,
            1.0f  /* temperature */
        );
    }

    /* Compute softmax blend weights (frozen shape) */
    float weights[TRIX_MAX_OCTAVES];
    float max_w = sol->blend_weights[0];
    for (int o = 1; o < sol->n_octaves; o++) {
        if (sol->blend_weights[o] > max_w) max_w = sol->blend_weights[o];
    }

    float sum = 0.0f;
    for (int o = 0; o < sol->n_octaves; o++) {
        weights[o] = expf(sol->blend_weights[o] - max_w);
        sum += weights[o];
    }
    for (int o = 0; o < sol->n_octaves; o++) {
        weights[o] /= sum;
    }

    /* Blend octave outputs (frozen mul + add shapes) */
    memset(output, 0, d * sizeof(float));
    for (int o = 0; o < sol->n_octaves; o++) {
        for (int i = 0; i < d; i++) {
            output[i] += weights[o] * octave_outputs[o * d + i];
        }
    }

    /* Apply output scale and bias (frozen shapes) */
    for (int i = 0; i < d; i++) {
        output[i] = output[i] * sol->output_scale[i] + sol->blend_bias[i];
    }
}

/**
 * Batch forward pass
 */
static inline void trix_sparse_octave_forward_batch(
    const trix_sparse_octave_t* sol,
    const float* input,    /* [batch, d_model] */
    float* output,         /* [batch, d_model] */
    int batch_size
) {
    int d = sol->d_model;
    for (int b = 0; b < batch_size; b++) {
        trix_sparse_octave_forward(sol, &input[b * d], &output[b * d]);
    }
}

/**
 * Parameter count
 */
static inline int trix_sparse_octave_param_count(const trix_sparse_octave_t* sol) {
    /* Providence memories (keys + values) */
    int memory_params = sol->n_octaves * 2 * sol->memory_size * sol->d_model;
    /* Blend weights + bias + output scale */
    int blend_params = sol->n_octaves + sol->d_model + sol->d_model;
    return memory_params + blend_params;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Test
 * ═══════════════════════════════════════════════════════════════════════════ */

#ifdef TRIX_SPARSE_OCTAVE_STANDALONE

#include <stdio.h>
#include <time.h>

int main(void) {
    printf("Sparse Octave Lookup - Pure Frozen Shapes\n");
    printf("==========================================\n\n");

    srand((unsigned)time(NULL));

    int d_model = 64;
    int n_octaves = 3;
    int memory_size = 128;
    int top_k = 8;
    int batch_size = 4;

    /* Initialize */
    trix_sparse_octave_t sol;
    trix_sparse_octave_init(&sol, d_model, n_octaves, memory_size, top_k);

    printf("Configuration:\n");
    printf("  d_model: %d\n", d_model);
    printf("  n_octaves: %d\n", n_octaves);
    printf("  memory_size: %d\n", memory_size);
    printf("  top_k: %d\n", top_k);
    printf("  parameters: %d\n\n", trix_sparse_octave_param_count(&sol));

    /* Test forward pass */
    float* input = (float*)malloc(batch_size * d_model * sizeof(float));
    float* output = (float*)malloc(batch_size * d_model * sizeof(float));

    for (int i = 0; i < batch_size * d_model; i++) {
        input[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    printf("Running forward pass (batch=%d)...\n", batch_size);
    clock_t start = clock();

    trix_sparse_octave_forward_batch(&sol, input, output, batch_size);

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;

    printf("  Time: %.3f ms\n", elapsed);
    printf("  Input[0:4]: %.4f, %.4f, %.4f, %.4f\n",
           input[0], input[1], input[2], input[3]);
    printf("  Output[0:4]: %.4f, %.4f, %.4f, %.4f\n",
           output[0], output[1], output[2], output[3]);

    /* Verify output is not zero */
    float sum = 0.0f;
    for (int i = 0; i < batch_size * d_model; i++) {
        sum += fabsf(output[i]);
    }
    printf("  Output L1 norm: %.4f\n", sum);

    if (sum > 0.0f) {
        printf("\nPASS: Forward pass produces non-zero output\n");
    } else {
        printf("\nFAIL: Output is all zeros\n");
    }

    /* Cleanup */
    free(input);
    free(output);
    trix_sparse_octave_free(&sol);

    printf("\n\"Information lives at different scales.\"\n");

    return 0;
}

#endif /* TRIX_SPARSE_OCTAVE_STANDALONE */

#endif /* TRIX_SPARSE_OCTAVE_H */
