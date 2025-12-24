/**
 * 8x8 Pod Architecture - 64 Parallel 6502s
 *
 * Structure:
 *   - 1 Master Coordinator (dispatches work)
 *   - 8 Pods (like SMs)
 *   - 8 Workers per Pod (like warps)
 *   - 64 Total Workers
 *
 * Each worker has:
 *   - Its own routing table (zero page)
 *   - Access to pod-shared fabrics
 *   - Independent execution
 *
 * Build: nvcc -std=c++17 -O3 -o fabric_8x8 fabric_8x8.cu
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

// =============================================================================
// ARCHITECTURE CONSTANTS
// =============================================================================

#define NUM_PODS 8
#define WORKERS_PER_POD 8
#define TOTAL_WORKERS (NUM_PODS * WORKERS_PER_POD)

#define ZERO_PAGE_SIZE 64    // Routes per worker (smaller for demo)
#define DATA_SIZE 256        // Data bytes per worker
#define FABRIC_TYPES 4       // Fabrics per pod

// =============================================================================
// DATA STRUCTURES
// =============================================================================

// Route entry: source → transform → destination
struct Route {
    uint8_t src_worker : 4;    // Source worker in pod (0-7)
    uint8_t src_reg : 4;       // Source register (0-15)
    uint8_t transform : 4;     // Transform to apply (0-15)
    uint8_t dst_reg : 4;       // Destination register (0-15)
};

// Worker state (like a 6502's internal state)
struct WorkerState {
    Route routes[ZERO_PAGE_SIZE];  // Routing table (zero page)
    uint8_t regs[16];              // Working registers
    uint8_t data[DATA_SIZE];       // Local data
    uint8_t active;                // Is this worker active?
    uint8_t pod_id;                // Which pod am I in?
    uint8_t worker_id;             // My ID within the pod
    uint8_t flags;                 // Status flags
};

// Pod state (shared among 8 workers)
struct PodState {
    uint8_t shared_mem[1024];      // Pod-shared memory
    uint8_t fabric_config[FABRIC_TYPES][16];  // Fabric configurations
    uint8_t active_workers;        // Bitmask of active workers
};

// Master state (coordinates all pods)
struct MasterState {
    uint8_t task_queue[256];       // Pending tasks
    uint8_t pod_status[NUM_PODS];  // Status of each pod
    uint8_t dispatch_table[64];    // Which worker gets which task
};

// Complete system state
struct SystemState {
    MasterState master;
    PodState pods[NUM_PODS];
    WorkerState workers[TOTAL_WORKERS];
};

// =============================================================================
// TRANSFORMS (the "frozen shapes" each worker can apply)
// =============================================================================

__device__ __forceinline__ uint8_t apply_transform(uint8_t op, uint8_t a, uint8_t b) {
    switch (op) {
        case 0:  return a;            // PASS
        case 1:  return a + b;        // ADD
        case 2:  return a - b;        // SUB
        case 3:  return a & b;        // AND
        case 4:  return a | b;        // OR
        case 5:  return a ^ b;        // XOR
        case 6:  return a << (b & 7); // SHL
        case 7:  return a >> (b & 7); // SHR
        case 8:  return a < b ? a : b; // MIN
        case 9:  return a > b ? a : b; // MAX
        case 10: return a + 1;        // INC
        case 11: return a - 1;        // DEC
        case 12: return ~a;           // NOT
        case 13: return (a + b) >> 1; // AVG
        case 14: return a * b;        // MUL
        case 15: return 0;            // ZERO
        default: return a;
    }
}

// =============================================================================
// WORKER KERNEL - Each thread is a 6502 worker
// =============================================================================

__global__ void execute_workers(SystemState* sys) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id >= TOTAL_WORKERS) return;

    int pod_id = global_id / WORKERS_PER_POD;
    int local_id = global_id % WORKERS_PER_POD;

    WorkerState* me = &sys->workers[global_id];
    PodState* pod = &sys->pods[pod_id];

    // Check if I'm active
    if (!me->active) return;

    // Execute my routing table
    for (int r = 0; r < ZERO_PAGE_SIZE; r++) {
        Route route = me->routes[r];

        // Skip empty routes (src_worker == 0xF means inactive)
        if (route.src_worker == 0xF) continue;

        // Get source value
        uint8_t src_value;
        if (route.src_worker == local_id) {
            // From my own registers
            src_value = me->regs[route.src_reg];
        } else {
            // From another worker in my pod (via shared memory)
            int src_global = pod_id * WORKERS_PER_POD + route.src_worker;
            src_value = sys->workers[src_global].regs[route.src_reg];
        }

        // Apply transform
        uint8_t result = apply_transform(route.transform, src_value, me->regs[route.dst_reg]);

        // Store to destination register
        me->regs[route.dst_reg] = result;
    }

    // Sync within pod
    __syncthreads();
}

// =============================================================================
// MASTER KERNEL - Dispatches work to pods
// =============================================================================

__global__ void master_dispatch(SystemState* sys, uint8_t* tasks, int num_tasks) {
    // Single thread acts as master
    if (threadIdx.x != 0) return;

    // Simple round-robin dispatch
    for (int t = 0; t < num_tasks && t < TOTAL_WORKERS; t++) {
        int worker_id = t % TOTAL_WORKERS;
        sys->workers[worker_id].regs[0] = tasks[t];  // Load task into reg 0
        sys->workers[worker_id].active = 1;
    }
}

// =============================================================================
// PHASED EXECUTION - For chained dependencies within a pod
// =============================================================================

__global__ void execute_workers_phased(SystemState* sys, int phase) {
    int pod_id = blockIdx.x;
    int local_id = threadIdx.x;

    if (pod_id >= NUM_PODS || local_id >= WORKERS_PER_POD) return;

    // Only execute workers in this phase
    if (local_id != phase) return;

    int global_id = pod_id * WORKERS_PER_POD + local_id;
    WorkerState* me = &sys->workers[global_id];

    if (!me->active) return;

    // Execute my routing table
    for (int r = 0; r < ZERO_PAGE_SIZE; r++) {
        Route route = me->routes[r];
        if (route.src_worker == 0xF) continue;

        uint8_t src_value;
        if (route.src_worker == local_id) {
            src_value = me->regs[route.src_reg];
        } else {
            int src_global = pod_id * WORKERS_PER_POD + route.src_worker;
            src_value = sys->workers[src_global].regs[route.src_reg];
        }

        uint8_t result = apply_transform(route.transform, src_value, me->regs[route.dst_reg]);
        me->regs[route.dst_reg] = result;
    }
}

// =============================================================================
// COLLECTIVE OPERATIONS - Pod-level coordination
// =============================================================================

__global__ void pod_reduce(SystemState* sys, int reduce_op) {
    int pod_id = blockIdx.x;
    int local_id = threadIdx.x;

    if (pod_id >= NUM_PODS || local_id >= WORKERS_PER_POD) return;

    __shared__ uint8_t shared_vals[WORKERS_PER_POD];

    int global_id = pod_id * WORKERS_PER_POD + local_id;
    shared_vals[local_id] = sys->workers[global_id].regs[0];

    __syncthreads();

    // Tree reduction within pod
    for (int stride = WORKERS_PER_POD / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            uint8_t a = shared_vals[local_id];
            uint8_t b = shared_vals[local_id + stride];
            shared_vals[local_id] = apply_transform(reduce_op, a, b);
        }
        __syncthreads();
    }

    // Worker 0 of each pod writes result to pod shared memory
    if (local_id == 0) {
        sys->pods[pod_id].shared_mem[0] = shared_vals[0];
    }
}

// =============================================================================
// HOST-SIDE HELPERS
// =============================================================================

void init_system(SystemState* sys) {
    memset(sys, 0, sizeof(SystemState));

    // Initialize all routes as inactive
    for (int w = 0; w < TOTAL_WORKERS; w++) {
        for (int r = 0; r < ZERO_PAGE_SIZE; r++) {
            sys->workers[w].routes[r].src_worker = 0xF;  // Inactive marker
        }
        sys->workers[w].pod_id = w / WORKERS_PER_POD;
        sys->workers[w].worker_id = w % WORKERS_PER_POD;
    }
}

void set_route(SystemState* sys, int worker, int route_idx,
               int src_worker, int src_reg, int transform, int dst_reg) {
    Route r;
    r.src_worker = src_worker;
    r.src_reg = src_reg;
    r.transform = transform;
    r.dst_reg = dst_reg;
    sys->workers[worker].routes[route_idx] = r;
}

void activate_worker(SystemState* sys, int worker) {
    sys->workers[worker].active = 1;
}

// =============================================================================
// DEMOS
// =============================================================================

void demo_parallel_execution() {
    printf("=== DEMO: 64 Workers in Parallel ===\n\n");

    SystemState* h_sys = new SystemState();
    SystemState* d_sys;
    cudaMalloc(&d_sys, sizeof(SystemState));

    init_system(h_sys);

    // Setup: Each worker increments its input and passes to next
    // Worker 0: reg[0] → INC → reg[1]
    // Worker 1: reg[0] → INC → reg[1]
    // ...

    for (int w = 0; w < TOTAL_WORKERS; w++) {
        // Route 0: Take reg[0], increment, store in reg[1]
        set_route(h_sys, w, 0, w % WORKERS_PER_POD, 0, 10, 1);  // INC
        h_sys->workers[w].regs[0] = w;  // Initialize with worker ID
        activate_worker(h_sys, w);
    }

    cudaMemcpy(d_sys, h_sys, sizeof(SystemState), cudaMemcpyHostToDevice);

    // Execute all workers
    execute_workers<<<NUM_PODS, WORKERS_PER_POD>>>(d_sys);
    cudaDeviceSynchronize();

    cudaMemcpy(h_sys, d_sys, sizeof(SystemState), cudaMemcpyDeviceToHost);

    // Check results
    printf("All 64 workers executed in parallel.\n");
    printf("Each worker: reg[0] (ID) → INC → reg[1]\n\n");

    printf("Sample results:\n");
    for (int p = 0; p < NUM_PODS; p++) {
        int w = p * WORKERS_PER_POD;  // First worker of each pod
        printf("  Pod %d, Worker 0: reg[0]=%d → reg[1]=%d\n",
               p, h_sys->workers[w].regs[0], h_sys->workers[w].regs[1]);
    }
    printf("\n");

    delete h_sys;
    cudaFree(d_sys);
}

void demo_pod_communication() {
    printf("=== DEMO: Intra-Pod Communication ===\n\n");

    SystemState* h_sys = new SystemState();
    SystemState* d_sys;
    cudaMalloc(&d_sys, sizeof(SystemState));

    init_system(h_sys);

    // Setup: Workers within Pod 0 share data
    // Worker 0: Has value 10 in reg[0]
    // Worker 1: Reads from Worker 0's reg[0], adds to its reg[0]
    // Worker 2: Reads from Worker 1's reg[0], adds to its reg[0]
    // etc.

    int pod = 0;
    for (int local = 0; local < WORKERS_PER_POD; local++) {
        int w = pod * WORKERS_PER_POD + local;
        h_sys->workers[w].regs[0] = 10;  // Each starts with 10

        if (local > 0) {
            // Read from previous worker, add to self
            set_route(h_sys, w, 0, local - 1, 0, 1, 0);  // ADD
        }
        activate_worker(h_sys, w);
    }

    cudaMemcpy(d_sys, h_sys, sizeof(SystemState), cudaMemcpyHostToDevice);

    // Execute in phases - Worker 0 first, then 1, then 2, etc.
    // This ensures chained reads see previous writes
    for (int phase = 0; phase < WORKERS_PER_POD; phase++) {
        execute_workers_phased<<<NUM_PODS, WORKERS_PER_POD>>>(d_sys, phase);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_sys, d_sys, sizeof(SystemState), cudaMemcpyDeviceToHost);

    printf("Workers in Pod 0 form a chain:\n");
    printf("  Worker N reads Worker N-1's reg[0] and adds to own reg[0]\n");
    printf("  (Phased execution ensures dependencies are respected)\n\n");

    printf("Results (each worker started with 10):\n");
    for (int local = 0; local < WORKERS_PER_POD; local++) {
        int w = pod * WORKERS_PER_POD + local;
        printf("  Worker %d: reg[0] = %d\n", local, h_sys->workers[w].regs[0]);
    }
    printf("\n");

    delete h_sys;
    cudaFree(d_sys);
}

void demo_pod_reduce() {
    printf("=== DEMO: Pod-Level Reduction ===\n\n");

    SystemState* h_sys = new SystemState();
    SystemState* d_sys;
    cudaMalloc(&d_sys, sizeof(SystemState));

    init_system(h_sys);

    // Setup: Each worker in each pod has a value
    // We reduce within each pod using ADD

    for (int p = 0; p < NUM_PODS; p++) {
        for (int local = 0; local < WORKERS_PER_POD; local++) {
            int w = p * WORKERS_PER_POD + local;
            h_sys->workers[w].regs[0] = local + 1;  // Values 1-8 in each pod
            activate_worker(h_sys, w);
        }
    }

    cudaMemcpy(d_sys, h_sys, sizeof(SystemState), cudaMemcpyHostToDevice);

    // Reduce each pod (ADD = op 1)
    pod_reduce<<<NUM_PODS, WORKERS_PER_POD>>>(d_sys, 1);
    cudaDeviceSynchronize();

    cudaMemcpy(h_sys, d_sys, sizeof(SystemState), cudaMemcpyDeviceToHost);

    printf("Each pod has workers with values 1,2,3,4,5,6,7,8\n");
    printf("Reducing with ADD should give 1+2+3+4+5+6+7+8 = 36\n\n");

    printf("Pod reduction results (stored in shared_mem[0]):\n");
    for (int p = 0; p < NUM_PODS; p++) {
        printf("  Pod %d: %d\n", p, h_sys->pods[p].shared_mem[0]);
    }
    printf("\n");

    delete h_sys;
    cudaFree(d_sys);
}

void benchmark() {
    printf("=== BENCHMARK: 64 Workers ===\n\n");

    SystemState* d_sys;
    cudaMalloc(&d_sys, sizeof(SystemState));

    SystemState* h_sys = new SystemState();
    init_system(h_sys);

    // All workers active, each has 16 routes
    for (int w = 0; w < TOTAL_WORKERS; w++) {
        for (int r = 0; r < 16; r++) {
            set_route(h_sys, w, r, w % WORKERS_PER_POD, r % 16, r % 16, (r + 1) % 16);
        }
        activate_worker(h_sys, w);
    }

    cudaMemcpy(d_sys, h_sys, sizeof(SystemState), cudaMemcpyHostToDevice);

    // Warm up
    for (int i = 0; i < 100; i++) {
        execute_workers<<<NUM_PODS, WORKERS_PER_POD>>>(d_sys);
    }
    cudaDeviceSynchronize();

    // Benchmark
    int iterations = 100000;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        execute_workers<<<NUM_PODS, WORKERS_PER_POD>>>(d_sys);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double cycles_per_sec = iterations / (ms / 1000.0);
    double worker_ops = cycles_per_sec * TOTAL_WORKERS * 16;  // 16 routes per worker

    printf("Configuration:\n");
    printf("  Pods: %d\n", NUM_PODS);
    printf("  Workers per pod: %d\n", WORKERS_PER_POD);
    printf("  Total workers: %d\n", TOTAL_WORKERS);
    printf("  Routes per worker: 16\n\n");

    printf("Results:\n");
    printf("  Iterations: %d\n", iterations);
    printf("  Time: %.2f ms\n", ms);
    printf("  Cycles/sec: %.2f M\n", cycles_per_sec / 1e6);
    printf("  Worker-ops/sec: %.2f M\n", worker_ops / 1e6);
    printf("  Equivalent 6502s: %.0f (at 1 MHz each)\n", worker_ops / 1e6);
    printf("\n");

    delete h_sys;
    cudaFree(d_sys);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    printf("=======================================================================\n");
    printf("8x8 POD ARCHITECTURE - 64 Parallel 6502s\n");
    printf("=======================================================================\n\n");

    printf("Structure:\n");
    printf("  ┌─────────────────────────────────────────────────┐\n");
    printf("  │              MASTER COORDINATOR                  │\n");
    printf("  └───────────────────────┬─────────────────────────┘\n");
    printf("                          │\n");
    printf("    ┌──────┬──────┬──────┼──────┬──────┬──────┐\n");
    printf("    ▼      ▼      ▼      ▼      ▼      ▼      ▼\n");
    printf("  Pod0   Pod1   Pod2   Pod3   Pod4   Pod5   Pod6   Pod7\n");
    printf("  8×W    8×W    8×W    8×W    8×W    8×W    8×W    8×W\n");
    printf("\n");
    printf("  W = Worker (a 6502-like unit with routing table)\n\n");

    demo_parallel_execution();
    demo_pod_communication();
    demo_pod_reduce();
    benchmark();

    printf("=======================================================================\n");
    printf("THE ARCHITECTURE\n");
    printf("=======================================================================\n\n");

    printf("This is multi-threaded 6502:\n");
    printf("  - 64 independent workers (each has zero page, registers)\n");
    printf("  - 8 pods (workers in same pod share memory, can communicate)\n");
    printf("  - 1 master (dispatches work, collects results)\n");
    printf("\n");
    printf("Each worker executes its OWN routing table.\n");
    printf("Workers can read each other's registers within a pod.\n");
    printf("Pods can reduce/aggregate via collective operations.\n\n");

    printf("This is NOT SIMD. Workers can have DIFFERENT routes.\n");
    printf("This is coordination at scale.\n\n");

    return 0;
}
