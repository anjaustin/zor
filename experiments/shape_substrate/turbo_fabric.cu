// =============================================================================
// TURBO_FABRIC.cu - Maximum Performance Plastic Fabric
// =============================================================================
//
// Target: 20,000x speedup over Verilog
//
// Optimizations:
// 1. Fused 6-phase kernel with __syncthreads() (no kernel launch overhead)
// 2. Batch multiple cycles per kernel launch
// 3. All state in shared memory
// 4. Minimal host-device communication
//
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// Configuration
#define CUBE_SIZE 4
#define NUM_NODES 64
#define NUM_NEIGHBORS 6
#define REWIRE_THRESHOLD 8
#define EVAL_CYCLES 8
#define CYCLES_PER_LAUNCH 100  // Batch this many cycles per kernel

struct TurboNode {
    uint8_t state;
    uint8_t frustration;
    uint8_t resonance;
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

// =============================================================================
// TURBO KERNEL - All phases fused, multiple cycles batched
// =============================================================================

__global__ void turbo_fabric_kernel(
    TurboNode* nodes,
    int* cycle_resonant,      // Output: resonant count per cycle
    int* cycle_rewires,       // Output: rewires per cycle
    int* convergence_cycle,   // Output: which cycle converged (-1 if none)
    int start_cycle,
    int num_cycles
) {
    int idx = threadIdx.x;
    if (idx >= NUM_NODES) return;

    // Load node state into registers
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

    // Shared memory for state exchange
    __shared__ uint8_t shared_states[NUM_NODES];
    __shared__ int shared_resonant;
    __shared__ int shared_rewires;

    // Run multiple cycles
    for (int cycle = 0; cycle < num_cycles; cycle++) {
        uint8_t resonance = 1;  // Assume resonant until proven otherwise

        // === 6 SEQUENTIAL PHASES ===
        #pragma unroll
        for (int phase = 0; phase < NUM_NEIGHBORS; phase++) {
            // Publish current state
            shared_states[idx] = state;
            __syncthreads();

            // Get neighbor value
            int nb_idx = neighbors[phase];
            uint8_t nb_val = shared_states[nb_idx];

            // Comparator swap
            bool should_swap;
            if (phase % 2 == 0) {
                should_swap = (state > nb_val);
            } else {
                should_swap = (nb_val > state);
            }

            if (should_swap) {
                state = nb_val;
                resonance = 0;
            }

            __syncthreads();
        }

        // === PLASTICITY UPDATE ===
        if (!rewiring) {
            if (!resonance) {
                if (frustration < 255) frustration++;
            } else {
                frustration >>= 1;
            }

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

                // Count rewire (use atomic in shared, then reduce)
                atomicAdd(&shared_rewires, 1);
            }
        } else {
            eval_counter++;
            if (!resonance && frustration < 255) frustration++;

            if (eval_counter >= EVAL_CYCLES) {
                if (frustration >= pre_frust) {
                    neighbors[rewire_dir] = old_neighbor;
                }
                rewire_dir = (rewire_dir + 1) % NUM_NEIGHBORS;
                rewiring = 0;
                frustration = 0;
            }
        }

        lfsr = lfsr_step(lfsr);

        // === COUNT RESONANCE ===
        if (idx == 0) {
            shared_resonant = 0;
            shared_rewires = 0;
        }
        __syncthreads();

        if (resonance) atomicAdd(&shared_resonant, 1);
        __syncthreads();

        // Thread 0 writes results
        if (idx == 0) {
            cycle_resonant[cycle] = shared_resonant;
            cycle_rewires[cycle] = shared_rewires;

            if (shared_resonant == NUM_NODES && *convergence_cycle < 0) {
                *convergence_cycle = start_cycle + cycle;
            }
        }
        __syncthreads();

        // Early exit if converged
        if (shared_resonant == NUM_NODES) {
            break;
        }
    }

    // Store final state back
    nodes[idx].state = state;
    nodes[idx].frustration = frustration;
    #pragma unroll
    for (int i = 0; i < NUM_NEIGHBORS; i++) {
        nodes[idx].neighbors[i] = neighbors[i];
    }
    nodes[idx].lfsr = lfsr;
    nodes[idx].rewiring = rewiring;
    nodes[idx].rewire_dir = rewire_dir;
    nodes[idx].eval_counter = eval_counter;
    nodes[idx].old_neighbor = old_neighbor;
    nodes[idx].pre_frust = pre_frust;
}

__global__ void init_turbo_fabric(TurboNode* nodes) {
    int idx = threadIdx.x;
    if (idx >= NUM_NODES) return;

    nodes[idx].state = (idx * 3) & 0xFF;
    nodes[idx].frustration = 0;
    nodes[idx].rewiring = 0;
    nodes[idx].rewire_dir = 0;
    nodes[idx].eval_counter = 0;
    nodes[idx].old_neighbor = 0;
    nodes[idx].pre_frust = 0;
    nodes[idx].lfsr = 0xBEEF ^ (idx * 0x1337);

    for (int d = 0; d < NUM_NEIGHBORS; d++) {
        nodes[idx].neighbors[d] = calc_neighbor(idx, d);
    }
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     TURBO FABRIC: Maximum Performance 4x4x4                      ║\n");
    printf("║     Target: 20,000x speedup over Verilog                         ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    TurboNode* d_nodes;
    int* d_cycle_resonant;
    int* d_cycle_rewires;
    int* d_convergence_cycle;

    cudaMalloc(&d_nodes, NUM_NODES * sizeof(TurboNode));
    cudaMalloc(&d_cycle_resonant, CYCLES_PER_LAUNCH * sizeof(int));
    cudaMalloc(&d_cycle_rewires, CYCLES_PER_LAUNCH * sizeof(int));
    cudaMalloc(&d_convergence_cycle, sizeof(int));

    int h_convergence = -1;
    cudaMemcpy(d_convergence_cycle, &h_convergence, sizeof(int), cudaMemcpyHostToDevice);

    init_turbo_fabric<<<1, NUM_NODES>>>(d_nodes);
    cudaDeviceSynchronize();

    printf("Optimizations:\n");
    printf("  - Fused 6-phase kernel (no launch overhead)\n");
    printf("  - %d cycles batched per launch\n", CYCLES_PER_LAUNCH);
    printf("  - All state in shared memory\n");
    printf("  - Register-resident node state\n\n");

    int h_resonant[CYCLES_PER_LAUNCH];
    int total_rewires = 0;
    int max_cycles = 500;
    int current_cycle = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    while (current_cycle < max_cycles && h_convergence < 0) {
        int cycles_this_launch = min(CYCLES_PER_LAUNCH, max_cycles - current_cycle);

        turbo_fabric_kernel<<<1, NUM_NODES>>>(
            d_nodes,
            d_cycle_resonant,
            d_cycle_rewires,
            d_convergence_cycle,
            current_cycle,
            cycles_this_launch
        );

        cudaMemcpy(&h_convergence, d_convergence_cycle, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_resonant, d_cycle_resonant, cycles_this_launch * sizeof(int), cudaMemcpyDeviceToHost);

        // Print progress
        int last_res = h_resonant[cycles_this_launch - 1];
        if (h_convergence >= 0) {
            last_res = NUM_NODES;
        }
        printf("Cycles %d-%d: %d/%d resonant (%d%%)\n",
               current_cycle, current_cycle + cycles_this_launch - 1,
               last_res, NUM_NODES, (last_res * 100) / NUM_NODES);

        current_cycle += cycles_this_launch;

        if (h_convergence >= 0) break;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("     RESULTS\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    if (h_convergence >= 0) {
        printf("  CONVERGED at cycle %d\n", h_convergence);
        printf("  Time: %.3f ms\n", ms);
        printf("  Cycles/sec: %.0f\n\n", h_convergence / (ms / 1000.0));

        // Calculate speedup
        float verilog_time_ms = 4 * 60 * 1000;  // ~4 minutes
        float speedup = verilog_time_ms / ms;

        printf("═══════════════════════════════════════════════════════════════════\n");
        printf("     SPEEDUP ANALYSIS\n");
        printf("═══════════════════════════════════════════════════════════════════\n\n");
        printf("  Verilog simulation: ~4 minutes (240,000 ms)\n");
        printf("  CUDA Turbo:         %.3f ms\n", ms);
        printf("  SPEEDUP:            %.0fx\n\n", speedup);

        if (speedup >= 20000) {
            printf("  ╔═══════════════════════════════════════╗\n");
            printf("  ║   TARGET ACHIEVED: >20,000x SPEEDUP   ║\n");
            printf("  ╚═══════════════════════════════════════╝\n\n");
        } else {
            printf("  Target: 20,000x\n");
            printf("  Need:   %.1f ms to hit target\n", verilog_time_ms / 20000);
        }
    }

    cudaFree(d_nodes);
    cudaFree(d_cycle_resonant);
    cudaFree(d_cycle_rewires);
    cudaFree(d_convergence_cycle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
