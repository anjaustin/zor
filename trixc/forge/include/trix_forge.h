/**
 * TRIX Forge - Neural Compilation Framework
 *
 * Main header file for TRIX Forge C API.
 *
 * "Watch neural networks think."
 */

#ifndef TRIX_FORGE_H
#define TRIX_FORGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/* Version */
#define TRIX_FORGE_VERSION_MAJOR 1
#define TRIX_FORGE_VERSION_MINOR 0
#define TRIX_FORGE_VERSION_PATCH 0
#define TRIX_FORGE_VERSION_STRING "1.0.0"

/* ============================================================================
 * Core Types
 * ============================================================================ */

/** Opaque handle to a TRIX model */
typedef struct trix_model_t trix_model_t;

/** Opaque handle to an NGE executable */
typedef struct trix_nge_t trix_nge_t;

/** Opaque handle to the event spine */
typedef struct trix_spine_t trix_spine_t;

/** Status codes */
typedef enum {
    TRIX_OK = 0,
    TRIX_ERROR_INVALID_ARGUMENT = -1,
    TRIX_ERROR_FILE_NOT_FOUND = -2,
    TRIX_ERROR_INVALID_FORMAT = -3,
    TRIX_ERROR_OUT_OF_MEMORY = -4,
    TRIX_ERROR_VALIDATION_FAILED = -5,
    TRIX_ERROR_COMPILATION_FAILED = -6,
    TRIX_ERROR_TARGET_NOT_SUPPORTED = -7,
} trix_status_t;

/** Data types */
typedef enum {
    TRIX_DTYPE_FLOAT32 = 0,
    TRIX_DTYPE_FLOAT16 = 1,
    TRIX_DTYPE_INT8 = 2,
    TRIX_DTYPE_UINT8 = 3,
} trix_dtype_t;

/* ============================================================================
 * Event System (Glassbox)
 * ============================================================================ */

/** Event structure */
typedef struct {
    uint32_t id;
    const char* type;
    const char* source;
    const char* message;
    uint32_t frame;
    double timestamp;
} trix_event_t;

/** Event callback function type */
typedef void (*trix_event_callback_t)(const trix_event_t* event, void* user_data);

/** Create a new event spine */
trix_spine_t* trix_spine_create(void);

/** Destroy an event spine */
void trix_spine_destroy(trix_spine_t* spine);

/** Subscribe to events */
void trix_spine_subscribe(trix_spine_t* spine,
                          trix_event_callback_t callback,
                          void* user_data);

/** Emit an event */
void trix_spine_emit(trix_spine_t* spine,
                     const char* type,
                     const char* source,
                     const char* message,
                     uint32_t frame);

/** Get event count */
size_t trix_spine_count(const trix_spine_t* spine);

/** Clear all events */
void trix_spine_clear(trix_spine_t* spine);

/* ============================================================================
 * Model API
 * ============================================================================ */

/** Load a model from .trix file */
trix_model_t* trix_model_load(const char* path);

/** Create an empty model */
trix_model_t* trix_model_create(const char* name);

/** Destroy a model */
void trix_model_destroy(trix_model_t* model);

/** Get model name */
const char* trix_model_name(const trix_model_t* model);

/** Get number of shapes */
size_t trix_model_shape_count(const trix_model_t* model);

/** Validate a model */
trix_status_t trix_model_validate(const trix_model_t* model);

/* ============================================================================
 * Compilation API
 * ============================================================================ */

/** Compilation options */
typedef struct {
    const char* target;           /* Target platform ID */
    int include_metadata;         /* Include metadata in output */
    int include_debug;            /* Include debug symbols */
    int enable_glassbox;          /* Enable Glassbox events */
    const char* precision;        /* "fp32", "fp16", "fp8" */
    const char* optimize;         /* "none", "basic", "aggressive" */
} trix_compile_options_t;

/** Default compilation options */
#define TRIX_COMPILE_OPTIONS_DEFAULT { \
    .target = "c", \
    .include_metadata = 1, \
    .include_debug = 0, \
    .enable_glassbox = 1, \
    .precision = "fp16", \
    .optimize = "aggressive" \
}

/** Compile a model to NGE */
trix_status_t trix_compile(const trix_model_t* model,
                           const trix_compile_options_t* options,
                           trix_nge_t** out_nge);

/** Get list of available targets */
const char** trix_list_targets(size_t* count);

/* ============================================================================
 * NGE API
 * ============================================================================ */

/** Load an NGE from file */
trix_nge_t* trix_nge_load(const char* path);

/** Save an NGE to file */
trix_status_t trix_nge_save(const trix_nge_t* nge, const char* path);

/** Destroy an NGE */
void trix_nge_destroy(trix_nge_t* nge);

/** Get NGE model name */
const char* trix_nge_model_name(const trix_nge_t* nge);

/** Get NGE target */
const char* trix_nge_target(const trix_nge_t* nge);

/** Get NGE size in bytes */
size_t trix_nge_size(const trix_nge_t* nge);

/** Check if NGE has Glassbox enabled */
int trix_nge_has_glassbox(const trix_nge_t* nge);

/* ============================================================================
 * Inference API
 * ============================================================================ */

/** Opaque handle to inference context */
typedef struct trix_context_t trix_context_t;

/** Create inference context from NGE */
trix_context_t* trix_context_create(const trix_nge_t* nge);

/** Destroy inference context */
void trix_context_destroy(trix_context_t* ctx);

/** Run forward pass */
trix_status_t trix_forward(trix_context_t* ctx,
                           const float* input,
                           size_t input_size,
                           float* output,
                           size_t output_size);

/** Attach event spine for Glassbox */
void trix_context_attach_spine(trix_context_t* ctx, trix_spine_t* spine);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/** Get error message for status code */
const char* trix_status_message(trix_status_t status);

/** Get library version string */
const char* trix_version(void);

#ifdef __cplusplus
}
#endif

#endif /* TRIX_FORGE_H */
