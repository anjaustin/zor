/**
 * CUDA Driver-Level Frozen Shapes
 *
 * Direct GPU execution of 6502 ALU shapes.
 * No Python. No CuPy. Just CUDA.
 *
 * Compile: nvcc -ptx -o cuda_shapes.ptx cuda_shapes.cu
 * Or:      nvcc -shared -o libcuda_shapes.so cuda_shapes.cu -Xcompiler -fPIC
 */

#include <cuda_runtime.h>
#include <stdint.h>

// =============================================================================
// SHAPE IDs (matches Python enum)
// =============================================================================

#define SHAPE_RIPPLE_ADD    0
#define SHAPE_RIPPLE_SUB    1
#define SHAPE_PARALLEL_AND  2
#define SHAPE_PARALLEL_OR   3
#define SHAPE_PARALLEL_XOR  4
#define SHAPE_SHIFT_LEFT    5
#define SHAPE_SHIFT_RIGHT   6
#define SHAPE_ROTATE_LEFT   7
#define SHAPE_ROTATE_RIGHT  8
#define SHAPE_INCREMENT     9
#define SHAPE_DECREMENT     10
#define SHAPE_TRANSFER      11
#define SHAPE_LOAD          12
#define SHAPE_STORE         13
#define SHAPE_BIT_TEST      14
#define SHAPE_IDENTITY      15

// =============================================================================
// PURE MATH PRIMITIVES (device functions, inlined)
// =============================================================================

__device__ __forceinline__ uint8_t d_xor(uint8_t a, uint8_t b) {
    return a ^ b;
}

__device__ __forceinline__ uint8_t d_and(uint8_t a, uint8_t b) {
    return a & b;
}

__device__ __forceinline__ uint8_t d_or(uint8_t a, uint8_t b) {
    return a | b;
}

__device__ __forceinline__ uint8_t d_not(uint8_t a) {
    return ~a;
}

// Full adder: returns (sum, carry_out) packed as uint16_t
__device__ __forceinline__ uint16_t d_full_adder_8bit(uint8_t a, uint8_t b, uint8_t c_in) {
    uint16_t sum = (uint16_t)a + (uint16_t)b + (uint16_t)c_in;
    return sum;  // Low 8 bits = result, bit 8 = carry
}

// =============================================================================
// FROZEN SHAPES (device functions)
// =============================================================================

// ADC: A + B + C
__device__ __forceinline__ uint16_t shape_ripple_add(uint8_t a, uint8_t b, uint8_t c) {
    uint16_t result = (uint16_t)a + (uint16_t)b + (uint16_t)c;
    return result;  // Low 8 bits = result, bit 8 = carry
}

// SBC: A - B - (1-C) = A + ~B + C
__device__ __forceinline__ uint16_t shape_ripple_sub(uint8_t a, uint8_t b, uint8_t c) {
    // ~b must stay in 8-bit range (0xFF - b), not integer promoted
    uint8_t b_inv = (uint8_t)(0xFF - b);  // Equivalent to ~b for uint8
    uint16_t result = (uint16_t)a + (uint16_t)b_inv + (uint16_t)c;
    return result;  // Low 8 bits = result, bit 8 = carry (inverted borrow)
}

// AND: A & B
__device__ __forceinline__ uint16_t shape_parallel_and(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)(a & b);  // No carry
}

// OR: A | B
__device__ __forceinline__ uint16_t shape_parallel_or(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)(a | b);  // No carry
}

// XOR: A ^ B
__device__ __forceinline__ uint16_t shape_parallel_xor(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)(a ^ b);  // No carry
}

// ASL: Shift left, MSB to carry
__device__ __forceinline__ uint16_t shape_shift_left(uint8_t a, uint8_t b, uint8_t c) {
    uint8_t carry = (a >> 7) & 1;
    uint8_t result = a << 1;
    return (uint16_t)result | ((uint16_t)carry << 8);
}

// LSR: Shift right, LSB to carry
__device__ __forceinline__ uint16_t shape_shift_right(uint8_t a, uint8_t b, uint8_t c) {
    uint8_t carry = a & 1;
    uint8_t result = a >> 1;
    return (uint16_t)result | ((uint16_t)carry << 8);
}

// ROL: Rotate left through carry
__device__ __forceinline__ uint16_t shape_rotate_left(uint8_t a, uint8_t b, uint8_t c) {
    uint8_t carry_out = (a >> 7) & 1;
    uint8_t result = (a << 1) | (c & 1);
    return (uint16_t)result | ((uint16_t)carry_out << 8);
}

// ROR: Rotate right through carry
__device__ __forceinline__ uint16_t shape_rotate_right(uint8_t a, uint8_t b, uint8_t c) {
    uint8_t carry_out = a & 1;
    uint8_t result = (a >> 1) | ((c & 1) << 7);
    return (uint16_t)result | ((uint16_t)carry_out << 8);
}

// INC: A + 1
__device__ __forceinline__ uint16_t shape_increment(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)((a + 1) & 0xFF);  // Wrap, no carry output
}

// DEC: A - 1
__device__ __forceinline__ uint16_t shape_decrement(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)((a - 1) & 0xFF);  // Wrap, no carry output
}

// Transfer/Load/Store: Pass through
__device__ __forceinline__ uint16_t shape_transfer(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)a;
}

// BIT: A & B (for flag computation)
__device__ __forceinline__ uint16_t shape_bit_test(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)(a & b);
}

// Identity: Pass through
__device__ __forceinline__ uint16_t shape_identity(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)a;
}

// =============================================================================
// SHAPE DISPATCHER (single function, switch on shape_id)
// =============================================================================

__device__ __forceinline__ uint16_t execute_shape(
    uint8_t shape_id,
    uint8_t a,
    uint8_t b,
    uint8_t c
) {
    switch (shape_id) {
        case SHAPE_RIPPLE_ADD:   return shape_ripple_add(a, b, c);
        case SHAPE_RIPPLE_SUB:   return shape_ripple_sub(a, b, c);
        case SHAPE_PARALLEL_AND: return shape_parallel_and(a, b, c);
        case SHAPE_PARALLEL_OR:  return shape_parallel_or(a, b, c);
        case SHAPE_PARALLEL_XOR: return shape_parallel_xor(a, b, c);
        case SHAPE_SHIFT_LEFT:   return shape_shift_left(a, b, c);
        case SHAPE_SHIFT_RIGHT:  return shape_shift_right(a, b, c);
        case SHAPE_ROTATE_LEFT:  return shape_rotate_left(a, b, c);
        case SHAPE_ROTATE_RIGHT: return shape_rotate_right(a, b, c);
        case SHAPE_INCREMENT:    return shape_increment(a, b, c);
        case SHAPE_DECREMENT:    return shape_decrement(a, b, c);
        case SHAPE_TRANSFER:     return shape_transfer(a, b, c);
        case SHAPE_LOAD:         return shape_transfer(a, b, c);
        case SHAPE_STORE:        return shape_transfer(a, b, c);
        case SHAPE_BIT_TEST:     return shape_bit_test(a, b, c);
        case SHAPE_IDENTITY:     return shape_identity(a, b, c);
        default:                 return shape_identity(a, b, c);
    }
}

// =============================================================================
// ROUTING TABLE (in constant memory - cached, broadcast to all threads)
// =============================================================================

__constant__ uint8_t d_routing_table[256];  // opcode -> shape_id

// =============================================================================
// MAIN KERNEL: Execute batch of ALU operations
// =============================================================================

extern "C" __global__ void execute_6502_alu(
    const uint8_t* __restrict__ opcodes,    // [N] opcode for each operation
    const uint8_t* __restrict__ operand_a,  // [N] first operand
    const uint8_t* __restrict__ operand_b,  // [N] second operand
    const uint8_t* __restrict__ carry_in,   // [N] carry input
    uint8_t* __restrict__ result,           // [N] result output
    uint8_t* __restrict__ carry_out,        // [N] carry output
    uint32_t n                              // number of operations
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Lookup shape from routing table (constant memory - cached)
        uint8_t opcode = opcodes[idx];
        uint8_t shape_id = d_routing_table[opcode];

        // Get operands
        uint8_t a = operand_a[idx];
        uint8_t b = operand_b[idx];
        uint8_t c = carry_in[idx];

        // Execute frozen shape
        uint16_t out = execute_shape(shape_id, a, b, c);

        // Store results
        result[idx] = (uint8_t)(out & 0xFF);
        carry_out[idx] = (uint8_t)((out >> 8) & 1);
    }
}

// =============================================================================
// DIRECT SHAPE KERNEL (no routing, shape_id provided per-thread)
// =============================================================================

extern "C" __global__ void execute_shape_direct(
    const uint8_t* __restrict__ shape_ids,  // [N] shape for each operation
    const uint8_t* __restrict__ operand_a,  // [N] first operand
    const uint8_t* __restrict__ operand_b,  // [N] second operand
    const uint8_t* __restrict__ carry_in,   // [N] carry input
    uint8_t* __restrict__ result,           // [N] result output
    uint8_t* __restrict__ carry_out,        // [N] carry output
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        uint8_t shape_id = shape_ids[idx];
        uint8_t a = operand_a[idx];
        uint8_t b = operand_b[idx];
        uint8_t c = carry_in[idx];

        uint16_t out = execute_shape(shape_id, a, b, c);

        result[idx] = (uint8_t)(out & 0xFF);
        carry_out[idx] = (uint8_t)((out >> 8) & 1);
    }
}

// =============================================================================
// BATCH KERNEL: Same opcode for entire batch (common case)
// =============================================================================

extern "C" __global__ void execute_shape_batch(
    uint8_t shape_id,                       // Single shape for all
    const uint8_t* __restrict__ operand_a,  // [N] first operand
    const uint8_t* __restrict__ operand_b,  // [N] second operand
    const uint8_t* __restrict__ carry_in,   // [N] carry input
    uint8_t* __restrict__ result,           // [N] result output
    uint8_t* __restrict__ carry_out,        // [N] carry output
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        uint8_t a = operand_a[idx];
        uint8_t b = operand_b[idx];
        uint8_t c = carry_in[idx];

        uint16_t out = execute_shape(shape_id, a, b, c);

        result[idx] = (uint8_t)(out & 0xFF);
        carry_out[idx] = (uint8_t)((out >> 8) & 1);
    }
}

// =============================================================================
// HOST-SIDE WRAPPER (for loading routing table)
// =============================================================================

extern "C" void set_routing_table(const uint8_t* host_table) {
    cudaMemcpyToSymbol(d_routing_table, host_table, 256 * sizeof(uint8_t));
}

// =============================================================================
// VERIFICATION KERNEL (compute ground truth for testing)
// =============================================================================

extern "C" __global__ void verify_shapes(
    uint8_t* results,  // [16 * 256 * 256 * 2] = 2MB output
    uint32_t n_per_shape
) {
    // Each thread handles one (shape, a, b, c) combination
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = 16 * 256 * 256 * 2;  // 16 shapes, 256 a, 256 b, 2 carry

    if (idx < total) {
        uint8_t shape_id = (idx / (256 * 256 * 2)) % 16;
        uint8_t a = (idx / (256 * 2)) % 256;
        uint8_t b = (idx / 2) % 256;
        uint8_t c = idx % 2;

        uint16_t out = execute_shape(shape_id, a, b, c);

        // Pack result + carry into output
        results[idx * 2] = (uint8_t)(out & 0xFF);
        results[idx * 2 + 1] = (uint8_t)((out >> 8) & 1);
    }
}
