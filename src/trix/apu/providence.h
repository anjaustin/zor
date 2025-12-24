/*
 * TRIX APU PROVIDENCE
 *
 * Content-addressed routing with precision awareness.
 * Providence routes based on content similarity (Hamming distance).
 * The APU manages precision for routing vs compute.
 *
 * "Providence routes shapes. APU routes precision."
 */

#ifndef TRIX_APU_PROVIDENCE_H
#define TRIX_APU_PROVIDENCE_H

#include "apu.h"
#include "shapes.h"
#include <stdlib.h>
#include <alloca.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * PROVIDENCE MEMORY - Content-addressable memory bank
 * ============================================================================ */

typedef struct {
    float* keys;        /* Binary patterns [num_entries * key_bits] */
    float* values;      /* Associated values [num_entries * value_dim] */
    int num_entries;
    int key_bits;
    int value_dim;
    trix_precision_t key_precision;    /* Precision for key comparison */
    trix_precision_t value_precision;  /* Precision for value retrieval */
} trix_providence_t;

/* Initialize Providence memory */
static inline void trix_providence_init(
    trix_providence_t* prov,
    int num_entries,
    int key_bits,
    int value_dim,
    trix_precision_t key_prec,
    trix_precision_t value_prec
) {
    prov->num_entries = num_entries;
    prov->key_bits = key_bits;
    prov->value_dim = value_dim;
    prov->key_precision = key_prec;
    prov->value_precision = value_prec;

    prov->keys = (float*)calloc(num_entries * key_bits, sizeof(float));
    prov->values = (float*)calloc(num_entries * value_dim, sizeof(float));
}

/* Free Providence memory */
static inline void trix_providence_free(trix_providence_t* prov) {
    free(prov->keys);
    free(prov->values);
    prov->keys = NULL;
    prov->values = NULL;
}

/* Set a key-value pair */
static inline void trix_providence_set(
    trix_providence_t* prov,
    int idx,
    const float* key,
    const float* value
) {
    for (int i = 0; i < prov->key_bits; i++) {
        prov->keys[idx * prov->key_bits + i] = key[i];
    }
    for (int i = 0; i < prov->value_dim; i++) {
        prov->values[idx * prov->value_dim + i] = value[i];
    }
}

/* ============================================================================
 * PROVIDENCE LOOKUP - Content-addressed retrieval
 * ============================================================================ */

/* Find nearest neighbor by Hamming distance */
static inline int trix_providence_nearest(
    const trix_providence_t* prov,
    const float* query
) {
    float min_dist = 1e9f;
    int min_idx = 0;

    for (int i = 0; i < prov->num_entries; i++) {
        float dist = trix_shape_hamming(
            query,
            &prov->keys[i * prov->key_bits],
            prov->key_bits
        );
        if (dist < min_dist) {
            min_dist = dist;
            min_idx = i;
        }
    }

    return min_idx;
}

/* Lookup value by nearest key */
static inline void trix_providence_lookup(
    const trix_providence_t* prov,
    const float* query,
    float* result
) {
    int idx = trix_providence_nearest(prov, query);
    for (int i = 0; i < prov->value_dim; i++) {
        result[i] = prov->values[idx * prov->value_dim + i];
    }
}

/* Soft lookup with attention weights */
static inline void trix_providence_soft_lookup(
    const trix_providence_t* prov,
    const float* query,
    float* result,
    float temperature
) {
    /* Compute distances to all keys */
    float* distances = (float*)alloca(prov->num_entries * sizeof(float));
    float* weights = (float*)alloca(prov->num_entries * sizeof(float));

    float max_neg_dist = -1e9f;
    for (int i = 0; i < prov->num_entries; i++) {
        distances[i] = trix_shape_hamming(
            query,
            &prov->keys[i * prov->key_bits],
            prov->key_bits
        );
        float neg_dist = -distances[i] / temperature;
        if (neg_dist > max_neg_dist) max_neg_dist = neg_dist;
    }

    /* Softmax over negative distances */
    float sum = 0.0f;
    for (int i = 0; i < prov->num_entries; i++) {
        weights[i] = expf(-distances[i] / temperature - max_neg_dist);
        sum += weights[i];
    }
    for (int i = 0; i < prov->num_entries; i++) {
        weights[i] /= sum;
    }

    /* Weighted sum of values */
    for (int j = 0; j < prov->value_dim; j++) {
        result[j] = 0.0f;
        for (int i = 0; i < prov->num_entries; i++) {
            result[j] += weights[i] * prov->values[i * prov->value_dim + j];
        }
    }
}

/* ============================================================================
 * PRECISION-AWARE PROVIDENCE
 *
 * Key comparison at low precision (FP4/FP8) - fast, approximate
 * Value retrieval at high precision (FP16/FP32) - accurate
 * ============================================================================ */

/* Providence lookup with APU precision management */
static inline void trix_providence_lookup_apu(
    trix_apu_t* apu,
    const trix_providence_t* prov,
    const float* query,
    float* result
) {
    /* Keys are compared at key_precision (typically low) */
    /* Values are retrieved at value_precision (typically higher) */

    /* Quantize query to key precision for comparison */
    float* quantized_query = (float*)alloca(prov->key_bits * sizeof(float));

    switch (prov->key_precision) {
        case TRIX_FP4:
            for (int i = 0; i < prov->key_bits; i++) {
                quantized_query[i] = trix_fp4_to_fp32(trix_fp32_to_fp4(query[i]));
            }
            break;
        case TRIX_FP8:
            for (int i = 0; i < prov->key_bits; i++) {
                quantized_query[i] = trix_fp8_to_fp32(trix_fp32_to_fp8(query[i]));
            }
            break;
        default:
            for (int i = 0; i < prov->key_bits; i++) {
                quantized_query[i] = query[i];
            }
            break;
    }

    /* Find nearest at low precision */
    int idx = trix_providence_nearest(prov, quantized_query);

    /* Retrieve value at value precision */
    for (int i = 0; i < prov->value_dim; i++) {
        float val = prov->values[idx * prov->value_dim + i];

        switch (prov->value_precision) {
            case TRIX_FP16:
                result[i] = trix_fp16_to_fp32(trix_fp32_to_fp16(val));
                break;
            case TRIX_FP8:
                result[i] = trix_fp8_to_fp32(trix_fp32_to_fp8(val));
                break;
            default:
                result[i] = val;
                break;
        }
    }

    /* Update APU counters */
    apu->precision_counts[prov->key_precision] += prov->num_entries;
    apu->precision_counts[prov->value_precision] += prov->value_dim;
}

/* ============================================================================
 * HIERARCHICAL PROVIDENCE - Multi-scale lookup
 *
 * Octave 0: Full precision comparison (slow, exact)
 * Octave 1: Reduced precision (faster, approximate)
 * Octave 2: Sketch comparison (fastest, coarse)
 * ============================================================================ */

typedef struct {
    trix_providence_t octaves[4];  /* Up to 4 octave levels */
    int num_octaves;
    trix_precision_t octave_precisions[4];
} trix_hierarchical_providence_t;

/* Initialize hierarchical Providence */
static inline void trix_hierarchical_providence_init(
    trix_hierarchical_providence_t* hprov,
    int num_entries,
    int base_key_bits,
    int value_dim
) {
    hprov->num_octaves = 3;

    /* Octave 0: Full precision */
    hprov->octave_precisions[0] = TRIX_FP16;
    trix_providence_init(&hprov->octaves[0], num_entries, base_key_bits,
                          value_dim, TRIX_FP16, TRIX_FP32);

    /* Octave 1: Reduced precision, subsampled keys */
    hprov->octave_precisions[1] = TRIX_FP8;
    trix_providence_init(&hprov->octaves[1], num_entries, base_key_bits / 4,
                          value_dim, TRIX_FP8, TRIX_FP16);

    /* Octave 2: Sketch, heavily compressed */
    hprov->octave_precisions[2] = TRIX_FP4;
    trix_providence_init(&hprov->octaves[2], num_entries, base_key_bits / 16,
                          value_dim, TRIX_FP4, TRIX_FP8);
}

/* Coarse-to-fine lookup */
static inline void trix_hierarchical_providence_lookup(
    trix_apu_t* apu,
    const trix_hierarchical_providence_t* hprov,
    const float* query,
    float* result
) {
    /* Start at coarsest octave, refine down */
    int candidate_idx = 0;

    /* Octave 2: Quick sketch match */
    /* (In practice, would use this to prune candidates) */

    /* Octave 1: Medium precision refinement */

    /* Octave 0: Final high-precision lookup */
    trix_providence_lookup_apu(apu, &hprov->octaves[0], query, result);
}

/* Free hierarchical Providence */
static inline void trix_hierarchical_providence_free(
    trix_hierarchical_providence_t* hprov
) {
    for (int i = 0; i < hprov->num_octaves; i++) {
        trix_providence_free(&hprov->octaves[i]);
    }
}

#ifdef __cplusplus
}
#endif

#endif /* TRIX_APU_PROVIDENCE_H */
