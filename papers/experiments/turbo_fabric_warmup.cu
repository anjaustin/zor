// =============================================================================
// TURBO_FABRIC_WARMUP.cu - Pre-Launch Sequence + Maximum Performance
// =============================================================================
//
// Pre-launch warmup flattens:
// 1. GPU clock ramping (power states)
// 2. Memory allocation caching
// 3. Kernel compilation/caching
// 4. Context initialization
//
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define CUBE_SIZE 4
#define NUM_NODES 64
#define NUM_NEIGHBORS 6
#define REWIRE_THRESHOLD 8
#define EVAL_CYCLES 8
#define CYCLES_PER_LAUNCH 100
#define WARMUP_ITERATIONS 10

struct TurboNode {
    uint8_t state;
    uint8_t frustration;
    uint16_t neighbors[NUM_NEIGHBORS];
    uint16_t lfsr;
    uint8_t rewiring;
    uint8_t rewire_dir;
    uint8_t eval_counter;
    uint16_t old_neighbor;
    uint8_t pre_frust;
};

__device__ __forceinline__ int calc_neighbor(int id, int dir) {
    int x = id % CUBE_SIZE;
    int y = (id / CUBE_SIZE) % CUBE_SIZE;
    int z = id / (CUBE_SIZE * CUBE_SIZE);
    int nx = x, ny = y, nz = z;
    switch (dir) {
        case 0: nx = (x + 1) % CUBE_SIZE; break;
        case 1: nx = (x + CUBE_SIZE - 1) % CUBE_SIZE; break;
        case 2: ny = (y + 1) % CUBE_SIZE; break;
        case 3: ny = (y + CUBE_SIZE - 1) % CUBE_SIZE; break;
        case 4: nz = (z + 1) % CUBE_SIZE; break;
        case 5: nz = (z + CUBE_SIZE - 1) % CUBE_SIZE; break;
    }
    return nx + ny * CUBE_SIZE + nz * CUBE_SIZE * CUBE_SIZE;
}

__device__ __forceinline__ uint16_t lfsr_step(uint16_t s) {
    return (s >> 1) | (((s ^ (s >> 2) ^ (s >> 3) ^ (s >> 5)) & 1) << 15);
}

// Warmup kernel - exercises all code paths
__global__ void warmup_kernel(TurboNode* nodes, int iterations) {
    int idx = threadIdx.x;
    if (idx >= NUM_NODES) return;

    __shared__ uint8_t shared_states[NUM_NODES];
    __shared__ int shared_counter;

    uint8_t state = nodes[idx].state;
    uint16_t lfsr = nodes[idx].lfsr;

    for (int i = 0; i < iterations; i++) {
        // Exercise shared memory
        shared_states[idx] = state;
        __syncthreads();

        // Exercise computation
        int nb_idx = (idx + 1) % NUM_NODES;
        uint8_t nb_val = shared_states[nb_idx];
        if (state > nb_val) state = nb_val;

        // Exercise atomics
        if (idx == 0) shared_counter = 0;
        __syncthreads();
        atomicAdd(&shared_counter, 1);
        __syncthreads();

        lfsr = lfsr_step(lfsr);
    }

    nodes[idx].state = state;
    nodes[idx].lfsr = lfsr;
}

__global__ void turbo_fabric_kernel(
    TurboNode* nodes,
    int* cycle_resonant,
    int* convergence_cycle,
    int start_cycle,
    int num_cycles
) {
    int idx = threadIdx.x;
    if (idx >= NUM_NODES) return;

    uint8_t state = nodes[idx].state;
    uint8_t frustration = nodes[idx].frustration;
    uint16_t neighbors[NUM_NEIGHBORS];
    #pragma unroll
    for (int i = 0; i < NUM_NEIGHBORS; i++) {
        neighbors[i] = nodes[idx].neighbors[i];
    }
    uint16_t lfsr = nodes[idx].lfsr;
    uint8_t rewiring = nodes[idx].rewiring;
    uint8_t rewire_dir = nodes[idx].rewire_dir;
    uint8_t eval_counter = nodes[idx].eval_counter;
    uint16_t old_neighbor = nodes[idx].old_neighbor;
    uint8_t pre_frust = nodes[idx].pre_frust;

    __shared__ uint8_t shared_states[NUM_NODES];
    __shared__ int shared_resonant;

    for (int cycle = 0; cycle < num_cycles; cycle++) {
        uint8_t resonance = 1;

        #pragma unroll
        for (int phase = 0; phase < NUM_NEIGHBORS; phase++) {
            shared_states[idx] = state;
            __syncthreads();

            uint8_t nb_val = shared_states[neighbors[phase]];
            bool should_swap = (phase % 2 == 0) ? (state > nb_val) : (nb_val > state);
            if (should_swap) { state = nb_val; resonance = 0; }
            __syncthreads();
        }

        if (!rewiring) {
            if (!resonance) { if (frustration < 255) frustration++; }
            else { frustration >>= 1; }

            if (frustration >= REWIRE_THRESHOLD) {
                rewiring = 1;
                lfsr = lfsr_step(lfsr);
                old_neighbor = neighbors[rewire_dir];
                pre_frust = frustration;
                eval_counter = 0;
                int new_nb = lfsr % NUM_NODES;
                if (new_nb == idx) new_nb = (new_nb + 1) % NUM_NODES;
                neighbors[rewire_dir] = new_nb;
                frustration = 0;
            }
        } else {
            eval_counter++;
            if (!resonance && frustration < 255) frustration++;
            if (eval_counter >= EVAL_CYCLES) {
                if (frustration >= pre_frust) neighbors[rewire_dir] = old_neighbor;
                rewire_dir = (rewire_dir + 1) % NUM_NEIGHBORS;
                rewiring = 0;
                frustration = 0;
            }
        }
        lfsr = lfsr_step(lfsr);

        if (idx == 0) shared_resonant = 0;
        __syncthreads();
        if (resonance) atomicAdd(&shared_resonant, 1);
        __syncthreads();

        if (idx == 0) {
            cycle_resonant[cycle] = shared_resonant;
            if (shared_resonant == NUM_NODES && *convergence_cycle < 0)
                *convergence_cycle = start_cycle + cycle;
        }
        __syncthreads();

        if (shared_resonant == NUM_NODES) break;
    }

    nodes[idx].state = state;
    nodes[idx].frustration = frustration;
    #pragma unroll
    for (int i = 0; i < NUM_NEIGHBORS; i++) nodes[idx].neighbors[i] = neighbors[i];
    nodes[idx].lfsr = lfsr;
    nodes[idx].rewiring = rewiring;
    nodes[idx].rewire_dir = rewire_dir;
    nodes[idx].eval_counter = eval_counter;
    nodes[idx].old_neighbor = old_neighbor;
    nodes[idx].pre_frust = pre_frust;
}

__global__ void init_fabric(TurboNode* nodes) {
    int idx = threadIdx.x;
    if (idx >= NUM_NODES) return;
    nodes[idx].state = (idx * 3) & 0xFF;
    nodes[idx].frustration = 0;
    nodes[idx].rewiring = 0;
    nodes[idx].rewire_dir = 0;
    nodes[idx].eval_counter = 0;
    nodes[idx].lfsr = 0xBEEF ^ (idx * 0x1337);
    for (int d = 0; d < NUM_NEIGHBORS; d++)
        nodes[idx].neighbors[d] = calc_neighbor(idx, d);
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     TURBO FABRIC + PRE-LAUNCH WARMUP                             ║\n");
    printf("║     Flattening GPU startup overhead                              ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs)\n\n", prop.name, prop.multiProcessorCount);

    TurboNode* d_nodes;
    int* d_cycle_resonant;
    int* d_convergence;

    cudaMalloc(&d_nodes, NUM_NODES * sizeof(TurboNode));
    cudaMalloc(&d_cycle_resonant, CYCLES_PER_LAUNCH * sizeof(int));
    cudaMalloc(&d_convergence, sizeof(int));

    // =======================================================================
    // PRE-LAUNCH WARMUP SEQUENCE
    // =======================================================================
    printf(">>> PRE-LAUNCH WARMUP SEQUENCE <<<\n\n");

    printf("Phase 1: GPU Clock Ramp-up...\n");
    // Force GPU to max clock by running compute-intensive warmup
    for (int w = 0; w < WARMUP_ITERATIONS; w++) {
        init_fabric<<<1, NUM_NODES>>>(d_nodes);
        warmup_kernel<<<1, NUM_NODES>>>(d_nodes, 1000);
    }
    cudaDeviceSynchronize();
    printf("  Complete.\n");

    printf("Phase 2: Memory Warmup...\n");
    // Touch all memory regions
    cudaMemset(d_nodes, 0, NUM_NODES * sizeof(TurboNode));
    cudaMemset(d_cycle_resonant, 0, CYCLES_PER_LAUNCH * sizeof(int));
    cudaDeviceSynchronize();
    printf("  Complete.\n");

    printf("Phase 3: Kernel Cache Warmup...\n");
    // Run the actual kernel a few times to cache it
    int h_conv = -1;
    for (int w = 0; w < 3; w++) {
        init_fabric<<<1, NUM_NODES>>>(d_nodes);
        cudaMemcpy(d_convergence, &h_conv, sizeof(int), cudaMemcpyHostToDevice);
        turbo_fabric_kernel<<<1, NUM_NODES>>>(d_nodes, d_cycle_resonant, d_convergence, 0, 10);
    }
    cudaDeviceSynchronize();
    printf("  Complete.\n");

    printf("\n>>> WARMUP COMPLETE - GPU AT PEAK PERFORMANCE <<<\n\n");

    // =======================================================================
    // ACTUAL BENCHMARK
    // =======================================================================

    printf("Running benchmark...\n\n");

    // Fresh initialization
    init_fabric<<<1, NUM_NODES>>>(d_nodes);
    h_conv = -1;
    cudaMemcpy(d_convergence, &h_conv, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    int max_cycles = 500;
    int current_cycle = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Precise timing with synchronization before start
    cudaDeviceSynchronize();
    cudaEventRecord(start);

    while (current_cycle < max_cycles && h_conv < 0) {
        int cycles_this = min(CYCLES_PER_LAUNCH, max_cycles - current_cycle);
        turbo_fabric_kernel<<<1, NUM_NODES>>>(
            d_nodes, d_cycle_resonant, d_convergence, current_cycle, cycles_this);
        cudaMemcpy(&h_conv, d_convergence, sizeof(int), cudaMemcpyDeviceToHost);
        current_cycle += cycles_this;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("     RESULTS (POST-WARMUP)\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    if (h_conv >= 0) {
        printf("  Converged at cycle: %d\n", h_conv);
        printf("  Time: %.4f ms\n", ms);
        printf("  Cycles/sec: %.0f\n", h_conv / (ms / 1000.0));
        printf("  µs per cycle: %.3f\n\n", ms * 1000.0 / h_conv);

        float verilog_ms = 240000;  // 4 minutes
        float speedup = verilog_ms / ms;

        printf("═══════════════════════════════════════════════════════════════════\n");
        printf("     SPEEDUP\n");
        printf("═══════════════════════════════════════════════════════════════════\n\n");
        printf("  Verilog: 240,000 ms (~4 min)\n");
        printf("  CUDA:    %.4f ms\n", ms);
        printf("  SPEEDUP: %.0fx\n\n", speedup);

        if (speedup >= 1000000) {
            printf("  ╔═════════════════════════════════════════════════════╗\n");
            printf("  ║   OVER 1,000,000x FASTER THAN VERILOG SIMULATION   ║\n");
            printf("  ╚═════════════════════════════════════════════════════╝\n\n");
        }
    }

    cudaFree(d_nodes);
    cudaFree(d_cycle_resonant);
    cudaFree(d_convergence);

    return 0;
}
