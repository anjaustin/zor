/**
 * CUDA Executor Library
 *
 * Minimal C interface for Python to call.
 * Python sets up once, then calls execute() repeatedly.
 * No Python overhead in the hot path.
 *
 * Build: nvcc -shared -o libcuda_executor.so cuda_executor_lib.cu -Xcompiler -fPIC
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// =============================================================================
// FROZEN SHAPES
// =============================================================================

__device__ __forceinline__ uint16_t shape_ripple_add(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)a + (uint16_t)b + (uint16_t)c;
}

__device__ __forceinline__ uint16_t shape_ripple_sub(uint8_t a, uint8_t b, uint8_t c) {
    uint8_t b_inv = (uint8_t)(0xFF - b);
    return (uint16_t)a + (uint16_t)b_inv + (uint16_t)c;
}

__device__ __forceinline__ uint16_t shape_parallel_and(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)(a & b);
}

__device__ __forceinline__ uint16_t shape_parallel_or(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)(a | b);
}

__device__ __forceinline__ uint16_t shape_parallel_xor(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)(a ^ b);
}

__device__ __forceinline__ uint16_t shape_shift_left(uint8_t a, uint8_t b, uint8_t c) {
    uint8_t carry = (a >> 7) & 1;
    uint8_t result = a << 1;
    return (uint16_t)result | ((uint16_t)carry << 8);
}

__device__ __forceinline__ uint16_t shape_shift_right(uint8_t a, uint8_t b, uint8_t c) {
    uint8_t carry = a & 1;
    uint8_t result = a >> 1;
    return (uint16_t)result | ((uint16_t)carry << 8);
}

__device__ __forceinline__ uint16_t shape_rotate_left(uint8_t a, uint8_t b, uint8_t c) {
    uint8_t carry_out = (a >> 7) & 1;
    uint8_t result = (a << 1) | (c & 1);
    return (uint16_t)result | ((uint16_t)carry_out << 8);
}

__device__ __forceinline__ uint16_t shape_rotate_right(uint8_t a, uint8_t b, uint8_t c) {
    uint8_t carry_out = a & 1;
    uint8_t result = (a >> 1) | ((c & 1) << 7);
    return (uint16_t)result | ((uint16_t)carry_out << 8);
}

__device__ __forceinline__ uint16_t shape_increment(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)((a + 1) & 0xFF);
}

__device__ __forceinline__ uint16_t shape_decrement(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)((a - 1) & 0xFF);
}

__constant__ uint8_t d_routing_table[256];

__device__ __forceinline__ uint16_t execute_shape(uint8_t shape_id, uint8_t a, uint8_t b, uint8_t c) {
    switch (shape_id) {
        case 0:  return shape_ripple_add(a, b, c);
        case 1:  return shape_ripple_sub(a, b, c);
        case 2:  return shape_parallel_and(a, b, c);
        case 3:  return shape_parallel_or(a, b, c);
        case 4:  return shape_parallel_xor(a, b, c);
        case 5:  return shape_shift_left(a, b, c);
        case 6:  return shape_shift_right(a, b, c);
        case 7:  return shape_rotate_left(a, b, c);
        case 8:  return shape_rotate_right(a, b, c);
        case 9:  return shape_increment(a, b, c);
        case 10: return shape_decrement(a, b, c);
        default: return (uint16_t)a;
    }
}

__global__ void execute_6502_alu(
    const uint8_t* __restrict__ opcodes,
    const uint8_t* __restrict__ operand_a,
    const uint8_t* __restrict__ operand_b,
    const uint8_t* __restrict__ carry_in,
    uint8_t* __restrict__ result,
    uint8_t* __restrict__ carry_out,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint8_t opcode = opcodes[idx];
        uint8_t shape_id = d_routing_table[opcode];
        uint16_t out = execute_shape(shape_id, operand_a[idx], operand_b[idx], carry_in[idx]);
        result[idx] = (uint8_t)(out & 0xFF);
        carry_out[idx] = (uint8_t)((out >> 8) & 1);
    }
}

// =============================================================================
// EXECUTOR STATE
// =============================================================================

struct Executor {
    // Unified memory buffers
    uint8_t* opcodes;
    uint8_t* operand_a;
    uint8_t* operand_b;
    uint8_t* carry_in;
    uint8_t* result;
    uint8_t* carry_out;

    // Config
    uint32_t max_n;
    int block_size;
    int grid_size;

    // CUDA graph
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStream_t stream;
    bool graph_valid;
};

static Executor* g_executor = nullptr;

// =============================================================================
// C API (exported)
// =============================================================================

extern "C" {

// Initialize executor with max batch size
int executor_init(uint32_t max_n) {
    if (g_executor) return -1;  // Already initialized

    g_executor = new Executor();
    g_executor->max_n = max_n;
    g_executor->block_size = 256;
    g_executor->grid_size = (max_n + 255) / 256;
    g_executor->graph_valid = false;

    // Allocate unified memory
    cudaMallocManaged(&g_executor->opcodes, max_n);
    cudaMallocManaged(&g_executor->operand_a, max_n);
    cudaMallocManaged(&g_executor->operand_b, max_n);
    cudaMallocManaged(&g_executor->carry_in, max_n);
    cudaMallocManaged(&g_executor->result, max_n);
    cudaMallocManaged(&g_executor->carry_out, max_n);

    // Create stream
    cudaStreamCreate(&g_executor->stream);

    // Set default routing (identity)
    uint8_t routing[256];
    for (int i = 0; i < 11; i++) routing[i] = i;
    for (int i = 11; i < 256; i++) routing[i] = 15;
    cudaMemcpyToSymbol(d_routing_table, routing, 256);

    return 0;
}

// Get buffer pointers (Python writes directly to these)
uint8_t* executor_get_opcodes() { return g_executor ? g_executor->opcodes : nullptr; }
uint8_t* executor_get_operand_a() { return g_executor ? g_executor->operand_a : nullptr; }
uint8_t* executor_get_operand_b() { return g_executor ? g_executor->operand_b : nullptr; }
uint8_t* executor_get_carry_in() { return g_executor ? g_executor->carry_in : nullptr; }
uint8_t* executor_get_result() { return g_executor ? g_executor->result : nullptr; }
uint8_t* executor_get_carry_out() { return g_executor ? g_executor->carry_out : nullptr; }

// Set routing table
void executor_set_routing(const uint8_t* routing) {
    cudaMemcpyToSymbol(d_routing_table, routing, 256);
}

// Create CUDA graph for fixed batch size (call once per batch size)
int executor_create_graph(uint32_t n) {
    if (!g_executor || n > g_executor->max_n) return -1;

    // Destroy old graph if exists
    if (g_executor->graph_valid) {
        cudaGraphExecDestroy(g_executor->graph_exec);
        cudaGraphDestroy(g_executor->graph);
    }

    // Capture graph
    int grid = (n + 255) / 256;
    cudaStreamBeginCapture(g_executor->stream, cudaStreamCaptureModeGlobal);
    execute_6502_alu<<<grid, 256, 0, g_executor->stream>>>(
        g_executor->opcodes,
        g_executor->operand_a,
        g_executor->operand_b,
        g_executor->carry_in,
        g_executor->result,
        g_executor->carry_out,
        n
    );
    cudaStreamEndCapture(g_executor->stream, &g_executor->graph);
    cudaGraphInstantiate(&g_executor->graph_exec, g_executor->graph, NULL, NULL, 0);

    g_executor->graph_valid = true;
    return 0;
}

// Execute graph (minimal overhead - just graph launch + sync)
int executor_run_graph() {
    if (!g_executor || !g_executor->graph_valid) return -1;

    cudaGraphLaunch(g_executor->graph_exec, g_executor->stream);
    cudaStreamSynchronize(g_executor->stream);

    return 0;
}

// Execute N operations (raw kernel, no graph)
int executor_run(uint32_t n) {
    if (!g_executor || n > g_executor->max_n) return -1;

    int grid = (n + 255) / 256;
    execute_6502_alu<<<grid, 256, 0, g_executor->stream>>>(
        g_executor->opcodes,
        g_executor->operand_a,
        g_executor->operand_b,
        g_executor->carry_in,
        g_executor->result,
        g_executor->carry_out,
        n
    );
    cudaStreamSynchronize(g_executor->stream);

    return 0;
}

// Cleanup
void executor_destroy() {
    if (!g_executor) return;

    if (g_executor->graph_valid) {
        cudaGraphExecDestroy(g_executor->graph_exec);
        cudaGraphDestroy(g_executor->graph);
    }

    cudaStreamDestroy(g_executor->stream);
    cudaFree(g_executor->opcodes);
    cudaFree(g_executor->operand_a);
    cudaFree(g_executor->operand_b);
    cudaFree(g_executor->carry_in);
    cudaFree(g_executor->result);
    cudaFree(g_executor->carry_out);

    delete g_executor;
    g_executor = nullptr;
}

}  // extern "C"
