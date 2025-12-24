/**
 * TRIX NGE - Neural-Geometric Executable Runtime
 *
 * Header for loading and executing NGE files.
 *
 * "The executable describes itself."
 */

#ifndef TRIX_NGE_H
#define TRIX_NGE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * NGE Format Constants
 * ============================================================================ */

/** NGE Magic bytes: "TRIX" */
#define TRIX_NGE_MAGIC 0x58495254  /* "TRIX" in little-endian */

/** Current NGE version */
#define TRIX_NGE_VERSION_MAJOR 1
#define TRIX_NGE_VERSION_MINOR 0

/** NGE Flags */
#define TRIX_NGE_FLAG_GLASSBOX   0x0001  /* Glassbox events enabled */
#define TRIX_NGE_FLAG_METADATA   0x0002  /* Metadata section present */
#define TRIX_NGE_FLAG_DEBUG      0x0004  /* Debug symbols included */
#define TRIX_NGE_FLAG_COMPRESSED 0x0008  /* Data section compressed */
#define TRIX_NGE_FLAG_ENCRYPTED  0x0010  /* Data section encrypted */

/** Maximum sizes */
#define TRIX_NGE_MAX_TARGET_LEN  16
#define TRIX_NGE_HEADER_SIZE     64

/* ============================================================================
 * NGE Header Structure
 * ============================================================================ */

/**
 * NGE File Header (64 bytes, fixed size)
 *
 * All multi-byte values are little-endian.
 */
typedef struct {
    uint32_t magic;           /* 0-3:   "TRIX" (0x58495254) */
    uint16_t version_major;   /* 4-5:   Version major */
    uint16_t version_minor;   /* 6-7:   Version minor */
    uint16_t flags;           /* 8-9:   Bit flags */
    uint16_t reserved1;       /* 10-11: Reserved */
    uint32_t metadata_offset; /* 12-15: Offset to metadata */
    uint32_t metadata_size;   /* 16-19: Size of metadata */
    uint32_t code_offset;     /* 20-23: Offset to code */
    uint32_t code_size;       /* 24-27: Size of code */
    uint32_t data_offset;     /* 28-31: Offset to data */
    uint32_t data_size;       /* 32-35: Size of data */
    char target[16];          /* 36-51: Target platform */
    uint8_t reserved2[12];    /* 52-63: Reserved */
} trix_nge_header_t;

/* ============================================================================
 * NGE Runtime Types
 * ============================================================================ */

/** NGE handle (opaque in API, defined here for implementation) */
typedef struct {
    trix_nge_header_t header;
    char* metadata_json;
    uint8_t* code;
    uint8_t* data;

    /* Parsed metadata */
    char* model_name;
    char* model_version;
    size_t input_size;
    size_t output_size;
} trix_nge_impl_t;

/* ============================================================================
 * NGE API
 * ============================================================================ */

/**
 * Load an NGE file from disk.
 *
 * @param path Path to the NGE file
 * @return Handle to loaded NGE, or NULL on error
 */
trix_nge_impl_t* trix_nge_load_file(const char* path);

/**
 * Load an NGE from memory.
 *
 * @param data Pointer to NGE data
 * @param size Size of data in bytes
 * @return Handle to loaded NGE, or NULL on error
 */
trix_nge_impl_t* trix_nge_load_memory(const void* data, size_t size);

/**
 * Free an NGE handle.
 *
 * @param nge Handle to free
 */
void trix_nge_free(trix_nge_impl_t* nge);

/**
 * Check if NGE header is valid.
 *
 * @param header Header to validate
 * @return 1 if valid, 0 if invalid
 */
int trix_nge_header_valid(const trix_nge_header_t* header);

/**
 * Get NGE model name.
 */
const char* trix_nge_get_model_name(const trix_nge_impl_t* nge);

/**
 * Get NGE target platform.
 */
const char* trix_nge_get_target(const trix_nge_impl_t* nge);

/**
 * Get total NGE size in bytes.
 */
size_t trix_nge_get_size(const trix_nge_impl_t* nge);

/**
 * Check if NGE has Glassbox enabled.
 */
int trix_nge_has_glassbox(const trix_nge_impl_t* nge);

/**
 * Get expected input size.
 */
size_t trix_nge_input_size(const trix_nge_impl_t* nge);

/**
 * Get expected output size.
 */
size_t trix_nge_output_size(const trix_nge_impl_t* nge);

/* ============================================================================
 * NGE Execution
 * ============================================================================ */

/**
 * Execute NGE forward pass.
 *
 * @param nge NGE handle
 * @param input Input tensor data
 * @param input_size Size of input in floats
 * @param output Output tensor data (pre-allocated)
 * @param output_size Size of output buffer in floats
 * @return 0 on success, error code on failure
 */
int trix_nge_forward(trix_nge_impl_t* nge,
                     const float* input, size_t input_size,
                     float* output, size_t output_size);

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/**
 * Print NGE info to stdout.
 */
void trix_nge_print_info(const trix_nge_impl_t* nge);

/**
 * Dump NGE header in hex format.
 */
void trix_nge_dump_header(const trix_nge_header_t* header);

#ifdef __cplusplus
}
#endif

#endif /* TRIX_NGE_H */
