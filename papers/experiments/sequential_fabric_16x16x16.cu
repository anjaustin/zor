// =============================================================================
// SEQUENTIAL_FABRIC_16x16x16.cu - 4096 Node Scale Test
// =============================================================================
//
// 16x16x16 = 4096 nodes - massive scale test!
// If 8x8x8 converged in 113 cycles, can we maintain this at 64x nodes?
//
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define CUBE_SIZE 16
#define NUM_NODES 4096  // 16x16x16
#define NUM_NEIGHBORS 6
#define REWIRE_THRESHOLD 8
#define EVAL_CYCLES 8
#define THREADS_PER_BLOCK 256

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

__global__ void sequential_phase(SeqNode* nodes, uint8_t* all_states, int phase) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;

    int nb_idx = nodes[idx].neighbors[phase];
    uint8_t nb_val = all_states[nb_idx];
    uint8_t my_val = nodes[idx].state;

    bool should_swap;
    if (phase % 2 == 0) {
        should_swap = (my_val > nb_val);
    } else {
        should_swap = (nb_val > my_val);
    }

    if (should_swap) {
        nodes[idx].state = nb_val;
        nodes[idx].resonance = 0;
    }
    nodes[idx].has_participated = 1;
}

__global__ void sync_states(SeqNode* nodes, uint8_t* all_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;
    all_states[idx] = nodes[idx].state;
}

__global__ void reset_resonance(SeqNode* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;
    nodes[idx].resonance = 1;
}

__global__ void plasticity_update(SeqNode* nodes, int* resonant_count, int* rewire_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;

    uint8_t res = nodes[idx].resonance;
    uint8_t frust = nodes[idx].frustration;
    uint16_t rnd = nodes[idx].lfsr;
    uint8_t rewiring = nodes[idx].rewiring;
    uint8_t rewire_dir = nodes[idx].rewire_dir;
    uint8_t has_part = nodes[idx].has_participated;

    if (!rewiring) {
        if (!res && has_part) {
            if (frust < 255) frust++;
        } else if (res) {
            frust >>= 1;
        }

        if (frust >= REWIRE_THRESHOLD) {
            rewiring = 1;
            rnd = lfsr_step(rnd);
            nodes[idx].old_neighbor = nodes[idx].neighbors[rewire_dir];
            nodes[idx].pre_frust = frust;
            nodes[idx].eval_counter = 0;
            int new_nb = rnd % NUM_NODES;
            if (new_nb == idx) new_nb = (new_nb + 1) % NUM_NODES;
            nodes[idx].neighbors[rewire_dir] = new_nb;
            frust = 0;
            atomicAdd(rewire_count, 1);
        }
    } else {
        nodes[idx].eval_counter++;
        if (!res && frust < 255) frust++;
        if (nodes[idx].eval_counter >= EVAL_CYCLES) {
            if (frust >= nodes[idx].pre_frust) {
                nodes[idx].neighbors[rewire_dir] = nodes[idx].old_neighbor;
            }
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

__global__ void init_fabric(SeqNode* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
    nodes[idx].lfsr = 0xBEEF ^ (idx * 0x1337) ^ (idx << 8);

    for (int d = 0; d < NUM_NEIGHBORS; d++) {
        nodes[idx].neighbors[d] = calc_neighbor(idx, d);
    }
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     SEQUENTIAL FABRIC 16x16x16: 4096 Nodes                       ║\n");
    printf("║     Massive scale test - 64x the original 4x4x4!                 ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs, %zu MB)\n\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024 * 1024));

    SeqNode* d_nodes;
    uint8_t* d_states;
    int* d_resonant;
    int* d_rewires;

    cudaMalloc(&d_nodes, NUM_NODES * sizeof(SeqNode));
    cudaMalloc(&d_states, NUM_NODES * sizeof(uint8_t));
    cudaMalloc(&d_resonant, sizeof(int));
    cudaMalloc(&d_rewires, sizeof(int));

    int blocks = (NUM_NODES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    init_fabric<<<blocks, THREADS_PER_BLOCK>>>(d_nodes);
    sync_states<<<blocks, THREADS_PER_BLOCK>>>(d_nodes, d_states);
    cudaDeviceSynchronize();

    printf("Architecture:\n");
    printf("  16x16x16 torus (4096 nodes)\n");
    printf("  6 sequential phases per cycle\n");
    printf("  Rewire threshold: %d\n\n", REWIRE_THRESHOLD);

    printf("CYCLE,RESONANT,PERCENT,REWIRES\n");

    int total_rewires = 0;
    int max_cycles = 500;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int cycle = 0; cycle < max_cycles; cycle++) {
        reset_resonance<<<blocks, THREADS_PER_BLOCK>>>(d_nodes);

        for (int phase = 0; phase < NUM_NEIGHBORS; phase++) {
            sequential_phase<<<blocks, THREADS_PER_BLOCK>>>(d_nodes, d_states, phase);
            cudaDeviceSynchronize();
            sync_states<<<blocks, THREADS_PER_BLOCK>>>(d_nodes, d_states);
            cudaDeviceSynchronize();
        }

        cudaMemset(d_resonant, 0, sizeof(int));
        cudaMemset(d_rewires, 0, sizeof(int));
        plasticity_update<<<blocks, THREADS_PER_BLOCK>>>(d_nodes, d_resonant, d_rewires);

        int h_res, h_rew;
        cudaMemcpy(&h_res, d_resonant, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_rew, d_rewires, sizeof(int), cudaMemcpyDeviceToHost);
        total_rewires += h_rew;

        if (cycle % 25 == 0 || h_res == NUM_NODES) {
            printf("%d,%d,%d%%,%d\n", cycle, h_res, (h_res * 100) / NUM_NODES, total_rewires);
        }

        if (h_res == NUM_NODES) {
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);

            printf("\n*** FULL CONVERGENCE at cycle %d ***\n", cycle);
            printf("    4096/4096 nodes resonant (100%%)\n");
            printf("    Total rewires: %d\n", total_rewires);
            printf("    Time: %.2f ms (%.1f cycles/sec)\n", ms, cycle / (ms / 1000.0));
            printf("\n");
            printf("═══════════════════════════════════════════════════════════════════\n");
            printf("     FUNKY CONVERGENCE AT MASSIVE SCALE!\n");
            printf("═══════════════════════════════════════════════════════════════════\n");
            printf("\n");
            printf("Scaling comparison:\n");
            printf("  4x4x4:    64 nodes, ~158 cycles\n");
            printf("  8x8x8:   512 nodes,  113 cycles\n");
            printf("  16x16x16: 4096 nodes, %d cycles\n\n", cycle);
            break;
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    int final_res;
    cudaMemcpy(&final_res, d_resonant, sizeof(int), cudaMemcpyDeviceToHost);

    if (final_res < NUM_NODES) {
        printf("\nFinal status after %d cycles:\n", max_cycles);
        printf("  Resonant: %d/%d (%d%%)\n", final_res, NUM_NODES,
               (final_res * 100) / NUM_NODES);
        printf("  Rewires: %d\n", total_rewires);
        printf("  Time: %.2f ms\n", ms);
    }

    cudaFree(d_nodes);
    cudaFree(d_states);
    cudaFree(d_resonant);
    cudaFree(d_rewires);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
