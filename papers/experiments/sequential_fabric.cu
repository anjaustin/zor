// =============================================================================
// SEQUENTIAL_FABRIC.cu - True-to-Verilog Sequential Swap Implementation
// =============================================================================
//
// The Verilog implementation uses SEQUENTIAL phases:
// - Phase 0: Compare with +X neighbor, swap if needed
// - Phase 1: Compare with -X neighbor, swap if needed
// - ... through phase 5
//
// This is fundamentally different from parallel median computation!
// The sequential approach creates cascading corrections that converge faster.
//
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define CUBE_SIZE 4
#define NUM_NODES 64
#define NUM_NEIGHBORS 6
#define REWIRE_THRESHOLD 8   // Lower threshold for more aggressive learning
#define EVAL_CYCLES 8

struct SeqNode {
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
    uint8_t has_participated;
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
// SEQUENTIAL PHASE KERNEL - One phase at a time
// =============================================================================

__global__ void sequential_phase(
    SeqNode* nodes,
    int phase  // 0-5 for 6 neighbors
) {
    int idx = threadIdx.x;
    if (idx >= NUM_NODES) return;

    __shared__ uint8_t states[NUM_NODES];
    states[idx] = nodes[idx].state;
    __syncthreads();

    // Get neighbor value for THIS phase
    int nb_idx = nodes[idx].neighbors[phase];
    uint8_t nb_val = states[nb_idx];
    uint8_t my_val = nodes[idx].state;

    // Frozen shape: Comparator swap
    // Positive direction (even phase): swap if I'm greater
    // Negative direction (odd phase): swap if neighbor is greater
    bool should_swap;
    if (phase % 2 == 0) {
        // +X, +Y, +Z: sort descending (I should be <= neighbor)
        should_swap = (my_val > nb_val);
    } else {
        // -X, -Y, -Z: sort ascending (neighbor should be <= me)
        should_swap = (nb_val > my_val);
    }

    // Apply swap
    if (should_swap) {
        nodes[idx].state = nb_val;
        nodes[idx].resonance = 0;  // Not resonant if we swapped
    }
    // Note: We only clear resonance on swap, keep it if no swap
    // This accumulates across phases

    nodes[idx].has_participated = 1;
}

// =============================================================================
// RESET RESONANCE AT CYCLE START
// =============================================================================

__global__ void reset_resonance(SeqNode* nodes) {
    int idx = threadIdx.x;
    if (idx >= NUM_NODES) return;
    nodes[idx].resonance = 1;  // Assume resonant until proven otherwise
}

// =============================================================================
// PLASTICITY UPDATE KERNEL - After all 6 phases
// =============================================================================

__global__ void plasticity_update(
    SeqNode* nodes,
    int* resonant_count,
    int* rewire_count
) {
    int idx = threadIdx.x;
    if (idx >= NUM_NODES) return;

    uint8_t res = nodes[idx].resonance;
    uint8_t frust = nodes[idx].frustration;
    uint16_t rnd = nodes[idx].lfsr;
    uint8_t rewiring = nodes[idx].rewiring;
    uint8_t rewire_dir = nodes[idx].rewire_dir;
    uint8_t has_part = nodes[idx].has_participated;

    if (!rewiring) {
        // Normal mode
        if (!res && has_part) {
            // Frustrated - only count if participated
            if (frust < 255) frust++;
        } else if (res) {
            // Resonant - decay
            frust >>= 1;
        }

        // Check if should rewire
        if (frust >= REWIRE_THRESHOLD) {
            rewiring = 1;
            rnd = lfsr_step(rnd);

            // Save old neighbor
            nodes[idx].old_neighbor = nodes[idx].neighbors[rewire_dir];
            nodes[idx].pre_frust = frust;
            nodes[idx].eval_counter = 0;

            // Pick random new neighbor
            int new_nb = rnd % NUM_NODES;
            if (new_nb == idx) new_nb = (new_nb + 1) % NUM_NODES;
            nodes[idx].neighbors[rewire_dir] = new_nb;

            // Reset frustration for evaluation
            frust = 0;

            atomicAdd(rewire_count, 1);
        }
    } else {
        // Evaluation mode
        nodes[idx].eval_counter++;

        if (!res && frust < 255) frust++;

        if (nodes[idx].eval_counter >= EVAL_CYCLES) {
            // Evaluate: is new connection better?
            if (frust >= nodes[idx].pre_frust) {
                // New is worse - revert
                nodes[idx].neighbors[rewire_dir] = nodes[idx].old_neighbor;
            }

            // Move to next direction
            rewire_dir = (rewire_dir + 1) % NUM_NEIGHBORS;
            rewiring = 0;
            frust = 0;
        }
    }

    nodes[idx].frustration = frust;
    nodes[idx].rewiring = rewiring;
    nodes[idx].rewire_dir = rewire_dir;
    nodes[idx].lfsr = lfsr_step(rnd);

    if (res) atomicAdd(resonant_count, 1);
}

// =============================================================================
// INITIALIZATION
// =============================================================================

__global__ void init_seq_fabric(SeqNode* nodes) {
    int idx = threadIdx.x;
    if (idx >= NUM_NODES) return;

    nodes[idx].state = (idx * 3) & 0xFF;
    nodes[idx].frustration = 0;
    nodes[idx].resonance = 1;
    nodes[idx].rewiring = 0;
    nodes[idx].rewire_dir = 0;
    nodes[idx].eval_counter = 0;
    nodes[idx].old_neighbor = 0;
    nodes[idx].pre_frust = 0;
    nodes[idx].has_participated = 0;
    nodes[idx].lfsr = 0xBEEF ^ (idx * 0x1337);

    for (int d = 0; d < NUM_NEIGHBORS; d++) {
        nodes[idx].neighbors[d] = calc_neighbor(idx, d);
    }
}

// =============================================================================
// HOST
// =============================================================================

int main() {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     SEQUENTIAL FABRIC: True Verilog-style 4x4x4                  ║\n");
    printf("║     6 sequential phases per cycle (not parallel median)          ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    SeqNode* d_nodes;
    int* d_resonant;
    int* d_rewires;

    cudaMalloc(&d_nodes, NUM_NODES * sizeof(SeqNode));
    cudaMalloc(&d_resonant, sizeof(int));
    cudaMalloc(&d_rewires, sizeof(int));

    init_seq_fabric<<<1, NUM_NODES>>>(d_nodes);
    cudaDeviceSynchronize();

    printf("Architecture:\n");
    printf("  4x4x4 torus (64 nodes)\n");
    printf("  6 sequential phases per cycle\n");
    printf("  Comparator swap (not median)\n");
    printf("  Rewire threshold: %d\n", REWIRE_THRESHOLD);
    printf("  Evaluation period: %d cycles\n\n", EVAL_CYCLES);

    printf("CYCLE,RESONANT,PERCENT,REWIRES\n");

    int total_rewires = 0;
    int max_cycles = 500;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int cycle = 0; cycle < max_cycles; cycle++) {
        // Reset resonance at start of cycle
        reset_resonance<<<1, NUM_NODES>>>(d_nodes);

        // Run 6 sequential phases
        for (int phase = 0; phase < NUM_NEIGHBORS; phase++) {
            sequential_phase<<<1, NUM_NODES>>>(d_nodes, phase);
            cudaDeviceSynchronize();  // Must sync between phases!
        }

        // Plasticity update
        cudaMemset(d_resonant, 0, sizeof(int));
        cudaMemset(d_rewires, 0, sizeof(int));
        plasticity_update<<<1, NUM_NODES>>>(d_nodes, d_resonant, d_rewires);

        int h_res, h_rew;
        cudaMemcpy(&h_res, d_resonant, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_rew, d_rewires, sizeof(int), cudaMemcpyDeviceToHost);
        total_rewires += h_rew;

        if (cycle % 20 == 0 || h_res == NUM_NODES) {
            printf("%d,%d,%d%%,%d\n", cycle, h_res, (h_res * 100) / NUM_NODES, total_rewires);
        }

        if (h_res == NUM_NODES) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);

            printf("\n*** FULL CONVERGENCE at cycle %d ***\n", cycle);
            printf("    64/64 nodes resonant (100%%)\n");
            printf("    Total rewires: %d\n", total_rewires);
            printf("    Time: %.2f ms\n", ms);
            printf("\n");
            printf("═══════════════════════════════════════════════════════════════════\n");
            printf("FUNKY CONVERGENCE ACHIEVED!\n");
            printf("═══════════════════════════════════════════════════════════════════\n");
            printf("\n\"The topology learned.\"\n");
            printf("\"The self emerged.\"\n");
            printf("\"The wood cut itself.\"\n\n");
            break;
        }
    }

    cudaFree(d_nodes);
    cudaFree(d_resonant);
    cudaFree(d_rewires);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
