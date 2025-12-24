/**
 * CUDA Persistent Kernel - GPU-Resident Executor
 *
 * The kernel stays resident on the GPU, polling for work.
 * No launch overhead. Just memory writes to submit work.
 *
 * This is the next level: the ORCHESTRATION becomes a frozen shape.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// =============================================================================
// WORK QUEUE STRUCTURE (in unified memory)
// =============================================================================

struct WorkQueue {
    // Control flags (atomics)
    volatile uint32_t work_ready;      // CPU sets to 1 when work is ready
    volatile uint32_t work_done;       // GPU sets to 1 when work is done
    volatile uint32_t shutdown;        // CPU sets to 1 to terminate

    // Work descriptor
    uint32_t n_ops;                    // Number of operations

    // Data pointers (pre-allocated, reused)
    uint8_t* opcodes;
    uint8_t* operand_a;
    uint8_t* operand_b;
    uint8_t* carry_in;
    uint8_t* result;
    uint8_t* carry_out;
};

// =============================================================================
// FROZEN SHAPES (same as cuda_shapes.cu)
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

__device__ __forceinline__ uint16_t shape_transfer(uint8_t a, uint8_t b, uint8_t c) {
    return (uint16_t)a;
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
        case 11: return shape_transfer(a, b, c);
        case 12: return shape_transfer(a, b, c);
        case 13: return shape_transfer(a, b, c);
        case 14: return shape_parallel_and(a, b, c);
        default: return (uint16_t)a;
    }
}

// =============================================================================
// PERSISTENT KERNEL - The Orchestration Shape
// Uses atomic counters for grid-wide synchronization
// =============================================================================

__device__ uint32_t d_ready_count;
__device__ uint32_t d_done_count;
__device__ uint32_t d_generation;

extern "C" __global__ void persistent_executor(WorkQueue* queue, uint32_t num_blocks) {
    // Each block processes its portion of work
    const uint32_t block_stride = blockDim.x;
    const uint32_t thread_id = threadIdx.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t global_stride = blockDim.x * num_blocks;
    const uint32_t global_id = block_id * blockDim.x + thread_id;

    uint32_t my_generation = 0;

    while (true) {
        // All threads in block check for shutdown
        if (queue->shutdown) return;

        // Block 0, thread 0 waits for work
        if (block_id == 0 && thread_id == 0) {
            // Wait for work_ready
            while (!queue->work_ready && !queue->shutdown) {
                // Spin
            }

            if (queue->shutdown) return;

            // Increment generation
            atomicAdd(&d_generation, 1);
            __threadfence();
        }

        // All threads wait for new generation
        __syncthreads();

        // Check shutdown again
        if (queue->shutdown) return;

        // Wait for generation to advance (other blocks)
        if (thread_id == 0) {
            while (d_generation == my_generation && !queue->shutdown) {
                // Spin
            }
        }
        __syncthreads();

        if (queue->shutdown) return;

        my_generation++;

        // Process work with grid-stride
        uint32_t n = queue->n_ops;
        for (uint32_t idx = global_id; idx < n; idx += global_stride) {
            uint8_t opcode = queue->opcodes[idx];
            uint8_t shape_id = d_routing_table[opcode];
            uint8_t a = queue->operand_a[idx];
            uint8_t b = queue->operand_b[idx];
            uint8_t c = queue->carry_in[idx];

            uint16_t out = execute_shape(shape_id, a, b, c);

            queue->result[idx] = (uint8_t)(out & 0xFF);
            queue->carry_out[idx] = (uint8_t)((out >> 8) & 1);
        }

        __syncthreads();

        // Signal done - last block to finish signals host
        if (thread_id == 0) {
            uint32_t finished = atomicAdd(&d_done_count, 1) + 1;
            if (finished == num_blocks) {
                __threadfence_system();
                queue->work_ready = 0;
                queue->work_done = 1;
                __threadfence_system();
                d_done_count = 0;  // Reset for next iteration
            }
        }

        __syncthreads();
    }
}

// =============================================================================
// COOPERATIVE GROUPS VERSION (more efficient for large workloads)
// =============================================================================

extern "C" __global__ void persistent_executor_cg(WorkQueue* queue) {
    // This version uses cooperative groups for grid-wide sync
    // Requires cudaLaunchCooperativeKernel

    const uint32_t grid_stride = blockDim.x * gridDim.x;
    const uint32_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (true) {
        // Block 0, thread 0 waits for work
        if (global_idx == 0) {
            while (!queue->work_ready && !queue->shutdown) {
                __threadfence_system();
            }
        }

        // Grid sync (all blocks wait)
        asm volatile("bar.sync 0, %0;" :: "r"(gridDim.x * blockDim.x));

        if (queue->shutdown) return;

        // Process
        uint32_t n = queue->n_ops;
        for (uint32_t idx = global_idx; idx < n; idx += grid_stride) {
            uint8_t opcode = queue->opcodes[idx];
            uint8_t shape_id = d_routing_table[opcode];
            uint8_t a = queue->operand_a[idx];
            uint8_t b = queue->operand_b[idx];
            uint8_t c = queue->carry_in[idx];

            uint16_t out = execute_shape(shape_id, a, b, c);

            queue->result[idx] = (uint8_t)(out & 0xFF);
            queue->carry_out[idx] = (uint8_t)((out >> 8) & 1);
        }

        // Grid sync
        asm volatile("bar.sync 0, %0;" :: "r"(gridDim.x * blockDim.x));

        if (global_idx == 0) {
            __threadfence_system();
            queue->work_ready = 0;
            queue->work_done = 1;
            __threadfence_system();
        }

        asm volatile("bar.sync 0, %0;" :: "r"(gridDim.x * blockDim.x));
    }
}

// =============================================================================
// HOST SIDE - Work Submission
// =============================================================================

extern "C" void submit_work(WorkQueue* queue) {
    // Ensure data is visible to GPU
    __sync_synchronize();

    // Signal work ready
    queue->work_ready = 1;
    __sync_synchronize();
}

extern "C" void wait_for_completion(WorkQueue* queue) {
    // Spin until done
    while (!queue->work_done) {
        __sync_synchronize();
    }

    // Reset for next iteration
    queue->work_done = 0;
    __sync_synchronize();
}

extern "C" void shutdown_executor(WorkQueue* queue) {
    queue->shutdown = 1;
    __sync_synchronize();
}

// =============================================================================
// BENCHMARK HARNESS (standalone)
// =============================================================================

#ifdef STANDALONE_BENCHMARK

#include <chrono>
#include <random>

int main() {
    printf("CUDA Persistent Kernel Benchmark\n");
    printf("================================\n\n");

    // Allocate work queue in unified memory
    WorkQueue* queue;
    cudaMallocManaged(&queue, sizeof(WorkQueue));

    // Allocate data buffers in unified memory
    const uint32_t N = 1024 * 1024;  // 1M ops
    cudaMallocManaged(&queue->opcodes, N);
    cudaMallocManaged(&queue->operand_a, N);
    cudaMallocManaged(&queue->operand_b, N);
    cudaMallocManaged(&queue->carry_in, N);
    cudaMallocManaged(&queue->result, N);
    cudaMallocManaged(&queue->carry_out, N);

    // Initialize queue
    queue->work_ready = 0;
    queue->work_done = 0;
    queue->shutdown = 0;
    queue->n_ops = N;

    // Set up routing table
    uint8_t routing[256];
    for (int i = 0; i < 256; i++) routing[i] = i % 16;
    cudaMemcpyToSymbol(d_routing_table, routing, 256);

    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> op_dist(0, 10);
    std::uniform_int_distribution<> byte_dist(0, 255);
    std::uniform_int_distribution<> bit_dist(0, 1);

    for (uint32_t i = 0; i < N; i++) {
        queue->opcodes[i] = op_dist(gen);
        queue->operand_a[i] = byte_dist(gen);
        queue->operand_b[i] = byte_dist(gen);
        queue->carry_in[i] = bit_dist(gen);
    }

    // On Thor (unified memory), no prefetch needed
    // Data is accessible from both CPU and GPU
    cudaDeviceSynchronize();

    // Launch persistent kernel
    int block_size = 256;
    int grid_size = 128;  // Keep it reasonable for persistence

    // Initialize device globals
    uint32_t zero = 0;
    cudaMemcpyToSymbol(d_ready_count, &zero, sizeof(uint32_t));
    cudaMemcpyToSymbol(d_done_count, &zero, sizeof(uint32_t));
    cudaMemcpyToSymbol(d_generation, &zero, sizeof(uint32_t));

    printf("Launching persistent kernel (%d blocks x %d threads)...\n", grid_size, block_size);
    persistent_executor<<<grid_size, block_size>>>(queue, grid_size);

    // Wait for kernel to start
    cudaDeviceSynchronize();

    // Benchmark: submit work batches
    const int iterations = 100;

    printf("Submitting %d work batches of %d ops each...\n\n", iterations, N);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        submit_work(queue);
        wait_for_completion(queue);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Shutdown
    shutdown_executor(queue);
    cudaDeviceSynchronize();

    // Results
    double total_ops = (double)N * iterations;
    double ops_per_sec = total_ops / (elapsed / 1e6);
    double latency_us = (double)elapsed / iterations;

    printf("Results:\n");
    printf("  Total ops:    %.0f\n", total_ops);
    printf("  Total time:   %.2f ms\n", elapsed / 1000.0);
    printf("  Throughput:   %.2f B ops/sec\n", ops_per_sec / 1e9);
    printf("  Per-batch:    %.1f μs\n", latency_us);
    printf("\n");
    printf("Zero kernel launch overhead.\n");
    printf("The orchestration IS a frozen shape.\n");

    // Cleanup
    cudaFree(queue->opcodes);
    cudaFree(queue->operand_a);
    cudaFree(queue->operand_b);
    cudaFree(queue->carry_in);
    cudaFree(queue->result);
    cudaFree(queue->carry_out);
    cudaFree(queue);

    return 0;
}

#endif  // STANDALONE_BENCHMARK
