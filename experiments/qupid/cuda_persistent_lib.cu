/**
 * CUDA Persistent Executor Library
 *
 * A worker thread runs continuously, polling for work.
 * Python just writes to shared memory - no function call in hot path.
 *
 * Build: nvcc -shared -O3 -o libcuda_persistent.so cuda_persistent_lib.cu -Xcompiler -fPIC -lpthread
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <pthread.h>
#include <atomic>

// =============================================================================
// FROZEN SHAPES (same as before)
// =============================================================================

__device__ __forceinline__ uint16_t execute_shape(uint8_t shape_id, uint8_t a, uint8_t b, uint8_t c) {
    switch (shape_id) {
        case 0:  return (uint16_t)a + (uint16_t)b + (uint16_t)c;  // ADD
        case 1:  { uint8_t b_inv = (uint8_t)(0xFF - b); return (uint16_t)a + (uint16_t)b_inv + (uint16_t)c; }  // SUB
        case 2:  return (uint16_t)(a & b);  // AND
        case 3:  return (uint16_t)(a | b);  // OR
        case 4:  return (uint16_t)(a ^ b);  // XOR
        case 5:  return (uint16_t)(a << 1) | ((uint16_t)((a >> 7) & 1) << 8);  // ASL
        case 6:  return (uint16_t)(a >> 1) | ((uint16_t)(a & 1) << 8);  // LSR
        case 7:  return (uint16_t)((a << 1) | (c & 1)) | ((uint16_t)((a >> 7) & 1) << 8);  // ROL
        case 8:  return (uint16_t)((a >> 1) | ((c & 1) << 7)) | ((uint16_t)(a & 1) << 8);  // ROR
        case 9:  return (uint16_t)((a + 1) & 0xFF);  // INC
        case 10: return (uint16_t)((a - 1) & 0xFF);  // DEC
        default: return (uint16_t)a;  // Identity
    }
}

__constant__ uint8_t d_routing_table[256];

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
// SHARED CONTROL STRUCTURE (CPU-GPU coherent)
// =============================================================================

struct alignas(64) ControlBlock {
    // Work submission
    std::atomic<uint32_t> work_ready;     // Python sets to N (number of ops) when work is ready
    std::atomic<uint32_t> work_done;      // Worker sets to 1 when done
    std::atomic<uint32_t> shutdown;       // Python sets to 1 to terminate worker

    // Statistics
    std::atomic<uint64_t> total_executions;
    std::atomic<uint64_t> total_ops;

    // Padding to cache line
    char _pad[64 - 28];
};

struct Executor {
    // Control block in unified memory (CPU-GPU coherent)
    ControlBlock* control;

    // Data buffers in unified memory
    uint8_t* opcodes;
    uint8_t* operand_a;
    uint8_t* operand_b;
    uint8_t* carry_in;
    uint8_t* result;
    uint8_t* carry_out;

    // Config
    uint32_t max_n;

    // CUDA graph
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStream_t stream;
    uint32_t graph_n;

    // Worker thread
    pthread_t worker_thread;
    bool worker_running;
};

static Executor* g_executor = nullptr;

// =============================================================================
// WORKER THREAD (runs continuously, polls for work)
// =============================================================================

void* worker_loop(void* arg) {
    Executor* exec = (Executor*)arg;

    // Set CUDA device for this thread
    cudaSetDevice(0);

    printf("[Worker] Started, CUDA device set\n");

    while (!exec->control->shutdown.load(std::memory_order_acquire)) {
        // Check for work
        uint32_t n = exec->control->work_ready.load(std::memory_order_acquire);

        if (n > 0) {
            // Work available - use raw kernel (simpler, avoids graph issues)
            int grid = (n + 255) / 256;

            printf("[Worker] Launching kernel with n=%d, grid=%d\n", n, grid);

            execute_6502_alu<<<grid, 256>>>(
                exec->opcodes, exec->operand_a, exec->operand_b,
                exec->carry_in, exec->result, exec->carry_out, n
            );

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("[Worker] Kernel launch error: %s\n", cudaGetErrorString(err));
            }

            cudaDeviceSynchronize();

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("[Worker] Sync error: %s\n", cudaGetErrorString(err));
            }

            printf("[Worker] Kernel completed\n");

            // Update stats
            exec->control->total_executions.fetch_add(1, std::memory_order_relaxed);
            exec->control->total_ops.fetch_add(n, std::memory_order_relaxed);

            // Signal done
            exec->control->work_ready.store(0, std::memory_order_release);
            exec->control->work_done.store(1, std::memory_order_release);

            printf("[Worker] Signaled done\n");
        } else {
            // No work - yield
            #if defined(__aarch64__)
            asm volatile("yield" ::: "memory");
            #elif defined(__x86_64__)
            asm volatile("pause" ::: "memory");
            #else
            std::atomic_thread_fence(std::memory_order_seq_cst);
            #endif
        }
    }

    printf("[Worker] Shutting down\n");
    return nullptr;
}

// =============================================================================
// C API (exported)
// =============================================================================

extern "C" {

int persistent_init(uint32_t max_n) {
    if (g_executor) return -1;

    g_executor = new Executor();
    g_executor->max_n = max_n;
    g_executor->graph_n = 0;

    // Allocate control block in unified memory
    cudaMallocManaged(&g_executor->control, sizeof(ControlBlock));
    new (g_executor->control) ControlBlock();
    g_executor->control->work_ready.store(0);
    g_executor->control->work_done.store(0);
    g_executor->control->shutdown.store(0);
    g_executor->control->total_executions.store(0);
    g_executor->control->total_ops.store(0);

    // Allocate data buffers
    cudaMallocManaged(&g_executor->opcodes, max_n);
    cudaMallocManaged(&g_executor->operand_a, max_n);
    cudaMallocManaged(&g_executor->operand_b, max_n);
    cudaMallocManaged(&g_executor->carry_in, max_n);
    cudaMallocManaged(&g_executor->result, max_n);
    cudaMallocManaged(&g_executor->carry_out, max_n);

    // Create stream
    cudaStreamCreate(&g_executor->stream);

    // Set default routing
    uint8_t routing[256];
    for (int i = 0; i < 11; i++) routing[i] = i;
    for (int i = 11; i < 256; i++) routing[i] = 15;
    cudaMemcpyToSymbol(d_routing_table, routing, 256);

    // Start worker thread
    g_executor->worker_running = true;
    pthread_create(&g_executor->worker_thread, nullptr, worker_loop, g_executor);

    return 0;
}

// Get buffer pointers
uint8_t* persistent_get_opcodes() { return g_executor ? g_executor->opcodes : nullptr; }
uint8_t* persistent_get_operand_a() { return g_executor ? g_executor->operand_a : nullptr; }
uint8_t* persistent_get_operand_b() { return g_executor ? g_executor->operand_b : nullptr; }
uint8_t* persistent_get_carry_in() { return g_executor ? g_executor->carry_in : nullptr; }
uint8_t* persistent_get_result() { return g_executor ? g_executor->result : nullptr; }
uint8_t* persistent_get_carry_out() { return g_executor ? g_executor->carry_out : nullptr; }

// Get control block pointer (Python accesses directly)
ControlBlock* persistent_get_control() { return g_executor ? g_executor->control : nullptr; }

// Submit work (just sets the flag - no kernel launch)
void persistent_submit(uint32_t n) {
    if (!g_executor) return;
    g_executor->control->work_done.store(0, std::memory_order_release);
    g_executor->control->work_ready.store(n, std::memory_order_release);
}

// Wait for completion (polls the flag)
void persistent_wait() {
    if (!g_executor) return;
    while (!g_executor->control->work_done.load(std::memory_order_acquire)) {
        #if defined(__aarch64__)
        asm volatile("yield" ::: "memory");
        #elif defined(__x86_64__)
        asm volatile("pause" ::: "memory");
        #else
        std::atomic_thread_fence(std::memory_order_seq_cst);
        #endif
    }
}

// Get stats
uint64_t persistent_get_total_executions() {
    return g_executor ? g_executor->control->total_executions.load() : 0;
}

uint64_t persistent_get_total_ops() {
    return g_executor ? g_executor->control->total_ops.load() : 0;
}

// Shutdown
void persistent_destroy() {
    if (!g_executor) return;

    // Signal shutdown
    g_executor->control->shutdown.store(1, std::memory_order_release);

    // Wait for worker to finish
    pthread_join(g_executor->worker_thread, nullptr);

    // Cleanup
    if (g_executor->graph_n > 0) {
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
    cudaFree(g_executor->control);

    delete g_executor;
    g_executor = nullptr;
}

}  // extern "C"
