/**
 * Frozen Shapes - C++ Header-Only Library
 *
 * Pure C++ implementation of frozen 6502 ALU shapes.
 * No Python. No runtime dependencies. Just math.
 *
 * Usage:
 *   #include "frozen_shapes.hpp"
 *   using namespace frozen;
 *
 *   ShapeExecutor exec(1024 * 1024);  // Max batch size
 *   exec.execute(opcodes, a, b, c, n);
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace frozen {

// =============================================================================
// CUDA Error Checking
// =============================================================================

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                    cudaGetErrorString(err));                  \
        }                                                                      \
    } while (0)

// =============================================================================
// Shape IDs
// =============================================================================

enum class Shape : uint8_t {
    RIPPLE_ADD = 0,
    RIPPLE_SUB = 1,
    PARALLEL_AND = 2,
    PARALLEL_OR = 3,
    PARALLEL_XOR = 4,
    SHIFT_LEFT = 5,
    SHIFT_RIGHT = 6,
    ROTATE_LEFT = 7,
    ROTATE_RIGHT = 8,
    INCREMENT = 9,
    DECREMENT = 10,
    TRANSFER = 11,
    LOAD = 12,
    STORE = 13,
    BIT_TEST = 14,
    IDENTITY = 15
};

// =============================================================================
// Device Functions - Frozen Shapes
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
    return (uint16_t)(a << 1) | ((uint16_t)((a >> 7) & 1) << 8);
}

__device__ __forceinline__ uint16_t shape_shift_right(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)(a >> 1) | ((uint16_t)(a & 1) << 8);
}

__device__ __forceinline__ uint16_t shape_rotate_left(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)((a << 1) | (c & 1)) | ((uint16_t)((a >> 7) & 1) << 8);
}

__device__ __forceinline__ uint16_t shape_rotate_right(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)((a >> 1) | ((c & 1) << 7)) | ((uint16_t)(a & 1) << 8);
}

__device__ __forceinline__ uint16_t shape_increment(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)((a + 1) & 0xFF);
}

__device__ __forceinline__ uint16_t shape_decrement(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)((a - 1) & 0xFF);
}

__device__ __forceinline__ uint16_t shape_transfer(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)a;
}

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
        case 11: return shape_transfer(a, b, c);
        case 12: return shape_transfer(a, b, c);
        case 13: return shape_transfer(a, b, c);
        case 14: return shape_parallel_and(a, b, c);
        default: return (uint16_t)a;
    }
}

// =============================================================================
// Routing Table (constant memory)
// =============================================================================

__constant__ uint8_t d_routing_table[256];

// =============================================================================
// Kernel
// =============================================================================

__global__ void frozen_alu_kernel(
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
// Shape Executor Class
// =============================================================================

class ShapeExecutor {
public:
    // Buffers (unified memory)
    uint8_t* opcodes;
    uint8_t* operand_a;
    uint8_t* operand_b;
    uint8_t* carry_in;
    uint8_t* result;
    uint8_t* carry_out;

    // Configuration
    uint32_t max_n;
    int block_size = 256;

    // CUDA graph for fixed-size execution
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaStream_t stream = nullptr;
    uint32_t graph_n = 0;

    ShapeExecutor(uint32_t max_batch_size) : max_n(max_batch_size) {
        // Allocate unified memory
        CUDA_CHECK(cudaMallocManaged(&opcodes, max_n));
        CUDA_CHECK(cudaMallocManaged(&operand_a, max_n));
        CUDA_CHECK(cudaMallocManaged(&operand_b, max_n));
        CUDA_CHECK(cudaMallocManaged(&carry_in, max_n));
        CUDA_CHECK(cudaMallocManaged(&result, max_n));
        CUDA_CHECK(cudaMallocManaged(&carry_out, max_n));

        // Create stream
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Set default routing (identity)
        uint8_t routing[256];
        for (int i = 0; i < 11; i++) routing[i] = i;
        for (int i = 11; i < 256; i++) routing[i] = 15;
        set_routing(routing);
    }

    ~ShapeExecutor() {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
        cudaFree(opcodes);
        cudaFree(operand_a);
        cudaFree(operand_b);
        cudaFree(carry_in);
        cudaFree(result);
        cudaFree(carry_out);
    }

    // Disable copy
    ShapeExecutor(const ShapeExecutor&) = delete;
    ShapeExecutor& operator=(const ShapeExecutor&) = delete;

    void set_routing(const uint8_t* routing) {
        CUDA_CHECK(cudaMemcpyToSymbol(d_routing_table, routing, 256));
    }

    void create_graph(uint32_t n) {
        if (n > max_n) {
            throw std::runtime_error("Batch size exceeds maximum");
        }

        // Destroy old graph
        if (graph_exec) {
            cudaGraphExecDestroy(graph_exec);
            graph_exec = nullptr;
        }
        if (graph) {
            cudaGraphDestroy(graph);
            graph = nullptr;
        }

        // Capture new graph
        int grid = (n + block_size - 1) / block_size;

        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        frozen_alu_kernel<<<grid, block_size, 0, stream>>>(
            opcodes, operand_a, operand_b, carry_in, result, carry_out, n
        );
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

        graph_n = n;
    }

    void execute_graph() {
        if (!graph_exec) {
            throw std::runtime_error("Graph not created");
        }
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void execute(uint32_t n) {
        if (n > max_n) {
            throw std::runtime_error("Batch size exceeds maximum");
        }

        int grid = (n + block_size - 1) / block_size;
        frozen_alu_kernel<<<grid, block_size, 0, stream>>>(
            opcodes, operand_a, operand_b, carry_in, result, carry_out, n
        );
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

}  // namespace frozen
