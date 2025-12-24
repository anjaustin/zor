// =============================================================================
// SINGLE_OCTAVE.cu - Verify 4x4x4 convergence (baseline)
// =============================================================================
//
// This matches the original Verilog zit_plastic_fabric that achieved 100%
// convergence in ~65 cycles. Let's verify the CUDA implementation works.
//
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define CUBE_SIZE 4
#define NUM_NODES 64  // 4x4x4
#define NUM_NEIGHBORS 6
#define REWIRE_THRESHOLD 16
#define FRUSTRATION_BITS 8

struct Node {
    uint8_t state;
    uint8_t frustration;
    uint8_t resonance;
    uint16_t neighbors[NUM_NEIGHBORS];
    uint16_t lfsr;
};

__device__ __forceinline__ int calc_neighbor(int node_id, int direction) {
    int x = node_id % CUBE_SIZE;
    int y = (node_id / CUBE_SIZE) % CUBE_SIZE;
    int z = node_id / (CUBE_SIZE * CUBE_SIZE);
    int new_x = x, new_y = y, new_z = z;
    switch (direction) {
        case 0: new_x = (x + 1) % CUBE_SIZE; break;
        case 1: new_x = (x + CUBE_SIZE - 1) % CUBE_SIZE; break;
        case 2: new_y = (y + 1) % CUBE_SIZE; break;
        case 3: new_y = (y + CUBE_SIZE - 1) % CUBE_SIZE; break;
        case 4: new_z = (z + 1) % CUBE_SIZE; break;
        case 5: new_z = (z + CUBE_SIZE - 1) % CUBE_SIZE; break;
    }
    return new_x + new_y * CUBE_SIZE + new_z * CUBE_SIZE * CUBE_SIZE;
}

__device__ __forceinline__ uint16_t lfsr_step(uint16_t s) {
    return (s >> 1) | (((s ^ (s >> 2) ^ (s >> 3) ^ (s >> 5)) & 1) << 15);
}

__global__ void single_octave_cycle(Node* nodes, int* resonant, int* rewires) {
    int idx = threadIdx.x;
    if (idx >= NUM_NODES) return;

    __shared__ uint8_t states[NUM_NODES];
    states[idx] = nodes[idx].state;
    __syncthreads();

    // Get neighbor values
    uint8_t nb[NUM_NEIGHBORS];
    #pragma unroll
    for (int i = 0; i < NUM_NEIGHBORS; i++) {
        nb[i] = states[nodes[idx].neighbors[i]];
    }

    // Frozen median shape
    int below = 0, above = 0;
    uint8_t my = nodes[idx].state;
    #pragma unroll
    for (int i = 0; i < NUM_NEIGHBORS; i++) {
        if (nb[i] < my) below++;
        if (nb[i] > my) above++;
    }

    // Update state
    uint8_t new_state = my;
    if (below > above && my > 0) new_state = my - 1;
    else if (above > below && my < 255) new_state = my + 1;

    // Resonance check
    uint8_t res = (below == above) || (below <= 1 && above <= 1);

    nodes[idx].state = new_state;
    nodes[idx].resonance = res;

    // Frustration
    uint8_t frust = nodes[idx].frustration;
    if (!res) {
        if (frust < 255) frust++;
    } else {
        frust >>= 1;
    }

    // Rewire
    uint16_t rnd = nodes[idx].lfsr;
    uint8_t did_rewire = 0;

    if (frust >= REWIRE_THRESHOLD) {
        did_rewire = 1;
        rnd = lfsr_step(rnd);
        int dir = rnd % NUM_NEIGHBORS;
        rnd = lfsr_step(rnd);
        int new_nb = rnd % NUM_NODES;
        if (new_nb == idx) new_nb = (new_nb + 1) % NUM_NODES;
        nodes[idx].neighbors[dir] = new_nb;
        frust = 0;
    }

    nodes[idx].lfsr = lfsr_step(rnd);
    nodes[idx].frustration = frust;

    if (res) atomicAdd(resonant, 1);
    if (did_rewire) atomicAdd(rewires, 1);
}

__global__ void init_single_octave(Node* nodes) {
    int idx = threadIdx.x;
    if (idx >= NUM_NODES) return;

    nodes[idx].state = (idx * 3) & 0xFF;  // Gradient
    nodes[idx].frustration = 0;
    nodes[idx].resonance = 0;
    nodes[idx].lfsr = 0xBEEF ^ (idx * 0x1337);

    #pragma unroll
    for (int dir = 0; dir < NUM_NEIGHBORS; dir++) {
        nodes[idx].neighbors[dir] = calc_neighbor(idx, dir);
    }
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     SINGLE OCTAVE: 4x4x4 = 64 NODES (Baseline Verification)      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    Node* d_nodes;
    int* d_resonant;
    int* d_rewires;

    cudaMalloc(&d_nodes, NUM_NODES * sizeof(Node));
    cudaMalloc(&d_resonant, sizeof(int));
    cudaMalloc(&d_rewires, sizeof(int));

    init_single_octave<<<1, NUM_NODES>>>(d_nodes);
    cudaDeviceSynchronize();

    printf("CYCLE,RESONANT,PERCENT,REWIRES\n");

    int total_rewires = 0;
    int max_cycles = 500;

    for (int cycle = 0; cycle < max_cycles; cycle++) {
        cudaMemset(d_resonant, 0, sizeof(int));
        cudaMemset(d_rewires, 0, sizeof(int));

        single_octave_cycle<<<1, NUM_NODES>>>(d_nodes, d_resonant, d_rewires);

        int h_res, h_rew;
        cudaMemcpy(&h_res, d_resonant, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_rew, d_rewires, sizeof(int), cudaMemcpyDeviceToHost);
        total_rewires += h_rew;

        if (cycle % 10 == 0 || h_res == NUM_NODES) {
            printf("%d,%d,%d%%,%d\n", cycle, h_res, (h_res * 100) / NUM_NODES, total_rewires);
        }

        if (h_res == NUM_NODES) {
            printf("\n*** FULL CONVERGENCE at cycle %d ***\n", cycle);
            printf("    64/64 nodes resonant (100%%)\n");
            printf("    Total rewires: %d\n", total_rewires);
            break;
        }
    }

    cudaFree(d_nodes);
    cudaFree(d_resonant);
    cudaFree(d_rewires);

    return 0;
}
