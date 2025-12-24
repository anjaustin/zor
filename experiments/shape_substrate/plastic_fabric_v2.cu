// =============================================================================
// PLASTIC_FABRIC_V2.cu - Improved Self-Organizing Topological Plasticity
// =============================================================================
//
// Improvements over v1:
// - Lower rewire threshold (8 instead of 16)
// - Smarter rewiring with geometric locality preference
// - Slower frustration decay to maintain learning pressure
// - Exploration noise to escape local minima
//
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// =============================================================================
// CONFIGURATION - Tuned for convergence
// =============================================================================

#define CUBE_SIZE 8                                    // 8x8x8 = 512 nodes
#define NUM_NODES (CUBE_SIZE * CUBE_SIZE * CUBE_SIZE)  // 512
#define STATE_WIDTH 8
#define FRUSTRATION_BITS 8
#define REWIRE_THRESHOLD 8                             // Lower threshold!
#define DECAY_SHIFT 2                                  // Slower decay
#define NUM_NEIGHBORS 6

// =============================================================================
// NODE STRUCTURE
// =============================================================================

struct PlasticNode {
    uint8_t state;
    uint8_t frustration;
    uint8_t resonance;
    uint8_t rewiring;
    uint16_t neighbors[NUM_NEIGHBORS];
    uint16_t lfsr;
};

// =============================================================================
// DEVICE HELPER FUNCTIONS
// =============================================================================

__device__ __forceinline__ int calc_torus_neighbor(int node_id, int direction) {
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

// Manhattan distance on torus
__device__ __forceinline__ int torus_distance(int a, int b) {
    int ax = a % CUBE_SIZE, ay = (a / CUBE_SIZE) % CUBE_SIZE, az = a / (CUBE_SIZE * CUBE_SIZE);
    int bx = b % CUBE_SIZE, by = (b / CUBE_SIZE) % CUBE_SIZE, bz = b / (CUBE_SIZE * CUBE_SIZE);

    int dx = min(abs(ax - bx), CUBE_SIZE - abs(ax - bx));
    int dy = min(abs(ay - by), CUBE_SIZE - abs(ay - by));
    int dz = min(abs(az - bz), CUBE_SIZE - abs(az - bz));

    return dx + dy + dz;
}

__device__ __forceinline__ uint16_t lfsr_step(uint16_t state) {
    uint16_t bit = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1;
    return (state >> 1) | (bit << 15);
}

// =============================================================================
// FROZEN SHAPE: Median Comparator
// =============================================================================

__device__ void frozen_median_shape(
    uint8_t my_state,
    uint8_t* neighbor_values,
    uint8_t* new_state,
    uint8_t* resonance
) {
    int count_below = 0;
    int count_above = 0;

    #pragma unroll
    for (int i = 0; i < NUM_NEIGHBORS; i++) {
        if (neighbor_values[i] < my_state) count_below++;
        if (neighbor_values[i] > my_state) count_above++;
    }

    if (count_below > count_above && my_state > 0) {
        *new_state = my_state - 1;
    } else if (count_above > count_below && my_state < 255) {
        *new_state = my_state + 1;
    } else {
        *new_state = my_state;
    }

    *resonance = (count_below == count_above) ||
                 (count_below <= 1 && count_above <= 1);
}

// =============================================================================
// SMART REWIRING - Prefer nearby nodes
// =============================================================================

__device__ int smart_rewire_target(int my_id, uint16_t rnd) {
    // 70% chance: pick a neighbor of a neighbor (2-hop)
    // 30% chance: pick random (exploration)

    if ((rnd & 0x7) < 5) {  // ~62% geometric locality
        // Pick a direction, then offset in a different axis
        int x = my_id % CUBE_SIZE;
        int y = (my_id / CUBE_SIZE) % CUBE_SIZE;
        int z = my_id / (CUBE_SIZE * CUBE_SIZE);

        // Random 2-hop neighbor
        int dx = ((rnd >> 3) & 0x3) - 1;  // -1, 0, 1
        int dy = ((rnd >> 5) & 0x3) - 1;
        int dz = ((rnd >> 7) & 0x3) - 1;

        // At least one step
        if (dx == 0 && dy == 0 && dz == 0) {
            dx = 1;
        }

        int new_x = (x + dx + CUBE_SIZE) % CUBE_SIZE;
        int new_y = (y + dy + CUBE_SIZE) % CUBE_SIZE;
        int new_z = (z + dz + CUBE_SIZE) % CUBE_SIZE;

        return new_x + new_y * CUBE_SIZE + new_z * CUBE_SIZE * CUBE_SIZE;
    } else {
        // Random global (exploration)
        return rnd % NUM_NODES;
    }
}

// =============================================================================
// PLASTIC FABRIC KERNEL V2
// =============================================================================

__global__ void plastic_fabric_cycle_v2(
    PlasticNode* nodes,
    int* global_resonant,
    int* global_frustration,
    int* global_rewires,
    int cycle
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;

    __shared__ uint8_t shared_states[NUM_NODES];
    shared_states[idx] = nodes[idx].state;
    __syncthreads();

    // LISTEN
    uint8_t neighbor_values[NUM_NEIGHBORS];
    #pragma unroll
    for (int i = 0; i < NUM_NEIGHBORS; i++) {
        int nb_idx = nodes[idx].neighbors[i];
        neighbor_values[i] = shared_states[nb_idx];
    }

    // REACT
    uint8_t new_state;
    uint8_t resonance;
    frozen_median_shape(nodes[idx].state, neighbor_values, &new_state, &resonance);

    nodes[idx].state = new_state;
    nodes[idx].resonance = resonance;

    // Frustration update
    uint8_t frust = nodes[idx].frustration;
    if (!resonance) {
        if (frust < 255) frust++;
    } else {
        // Slower decay - only decay by 1 bit
        if (frust > 0) frust = frust - (frust >> DECAY_SHIFT);
    }

    // REWIRE with smart targeting
    uint8_t rewiring = 0;
    uint16_t rnd = nodes[idx].lfsr;

    if (frust >= REWIRE_THRESHOLD) {
        rewiring = 1;

        rnd = lfsr_step(rnd);
        int dir = rnd % NUM_NEIGHBORS;
        rnd = lfsr_step(rnd);

        // Smart rewire target (geometric locality + exploration)
        int new_neighbor = smart_rewire_target(idx, rnd);

        // Avoid self-connection
        if (new_neighbor == idx) {
            new_neighbor = (new_neighbor + 1) % NUM_NODES;
        }

        nodes[idx].neighbors[dir] = new_neighbor;
        rnd = lfsr_step(rnd);

        // Reset frustration but keep some memory (partial reset)
        frust = frust >> 2;
    } else {
        rnd = lfsr_step(rnd);
    }

    nodes[idx].lfsr = rnd;
    nodes[idx].frustration = frust;
    nodes[idx].rewiring = rewiring;

    // AGGREGATE
    if (resonance) atomicAdd(global_resonant, 1);
    atomicAdd(global_frustration, frust);
    if (rewiring) atomicAdd(global_rewires, 1);
}

// =============================================================================
// INITIALIZATION KERNEL
// =============================================================================

__global__ void init_plastic_fabric_v2(PlasticNode* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;

    // Gradient initialization
    nodes[idx].state = (idx >> 1) & 0xFF;
    nodes[idx].frustration = 0;
    nodes[idx].resonance = 0;
    nodes[idx].rewiring = 0;

    #pragma unroll
    for (int dir = 0; dir < NUM_NEIGHBORS; dir++) {
        nodes[idx].neighbors[dir] = calc_torus_neighbor(idx, dir);
    }

    nodes[idx].lfsr = 0xBEEF ^ (idx * 0x1337) ^ (idx << 8);
}

// =============================================================================
// HOST FUNCTIONS
// =============================================================================

void run_plastic_fabric_v2(int max_cycles, bool verbose) {
    printf("\n");
    printf("================================================================\n");
    printf("     PLASTIC FABRIC V2: %dx%dx%d = %d NODES\n",
           CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, NUM_NODES);
    printf("     Improved with geometric locality + exploration\n");
    printf("================================================================\n\n");

    PlasticNode* d_nodes;
    int* d_global_resonant;
    int* d_global_frustration;
    int* d_global_rewires;

    cudaMalloc(&d_nodes, NUM_NODES * sizeof(PlasticNode));
    cudaMalloc(&d_global_resonant, sizeof(int));
    cudaMalloc(&d_global_frustration, sizeof(int));
    cudaMalloc(&d_global_rewires, sizeof(int));

    int threads = 256;
    int blocks = (NUM_NODES + threads - 1) / threads;

    init_plastic_fabric_v2<<<blocks, threads>>>(d_nodes);
    cudaDeviceSynchronize();

    printf("Rewire threshold: %d (lower = more aggressive)\n", REWIRE_THRESHOLD);
    printf("Decay shift: %d (higher = slower decay)\n", DECAY_SHIFT);
    printf("Rewire strategy: 62%% geometric locality, 38%% exploration\n");
    printf("\n");

    if (verbose) {
        printf("CYCLE,RESONANT,PERCENT,FRUSTRATION,REWIRES\n");
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int convergence_cycle = 0;
    int total_rewires = 0;
    int stable_count = 0;
    int last_resonant = 0;

    for (int cycle = 0; cycle < max_cycles; cycle++) {
        cudaMemset(d_global_resonant, 0, sizeof(int));
        cudaMemset(d_global_frustration, 0, sizeof(int));
        cudaMemset(d_global_rewires, 0, sizeof(int));

        plastic_fabric_cycle_v2<<<blocks, threads>>>(
            d_nodes,
            d_global_resonant,
            d_global_frustration,
            d_global_rewires,
            cycle
        );

        int h_resonant, h_frustration, h_rewires;
        cudaMemcpy(&h_resonant, d_global_resonant, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_frustration, d_global_frustration, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_rewires, d_global_rewires, sizeof(int), cudaMemcpyDeviceToHost);

        total_rewires += h_rewires;

        if (verbose && (cycle % 50 == 0 || h_resonant == NUM_NODES)) {
            printf("%d,%d,%d%%,%d,%d\n",
                   cycle, h_resonant, (h_resonant * 100) / NUM_NODES,
                   h_frustration, total_rewires);
        }

        // Track stability
        if (h_resonant == last_resonant) {
            stable_count++;
        } else {
            stable_count = 0;
            last_resonant = h_resonant;
        }

        if (h_resonant == NUM_NODES && convergence_cycle == 0) {
            convergence_cycle = cycle;
            printf("\n*** FULL CONVERGENCE at cycle %d ***\n", cycle);
            printf("    %d/%d nodes resonant (100%%)\n", NUM_NODES, NUM_NODES);
            printf("    Total rewires: %d\n\n", total_rewires);
        }

        // Early exit if converged and stable
        if (convergence_cycle > 0 && cycle > convergence_cycle + 50) {
            break;
        }

        // Report if stuck at high resonance
        if (stable_count > 100 && h_resonant > NUM_NODES * 0.9) {
            printf("\nNear-convergence at %d%% (%d/%d) - may need more exploration\n\n",
                   (h_resonant * 100) / NUM_NODES, h_resonant, NUM_NODES);
            stable_count = 0;  // Reset to continue watching
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("================================================================\n");
    printf("     RESULTS\n");
    printf("================================================================\n\n");

    if (convergence_cycle > 0) {
        printf("  Status: CONVERGED at cycle %d\n", convergence_cycle);
    } else {
        printf("  Status: Reached %d%% in %d cycles\n",
               (last_resonant * 100) / NUM_NODES, max_cycles);
    }
    printf("  Total rewires: %d\n", total_rewires);
    printf("  Time: %.2f ms\n", ms);
    printf("  Cycles/sec: %.2f\n", max_cycles / (ms / 1000.0));
    printf("\n");

    if (convergence_cycle > 0) {
        printf("================================================================\n");
        printf("     FUNKY CONVERGENCE ACHIEVED\n");
        printf("================================================================\n\n");
    }

    cudaFree(d_nodes);
    cudaFree(d_global_resonant);
    cudaFree(d_global_frustration);
    cudaFree(d_global_rewires);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     PLASTIC FABRIC V2 - Improved Topological Self-Organization   ║\n");
    printf("║     Geometric locality + exploration for better convergence      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nGPU: %s (%d SMs, %zu MB unified memory)\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024 * 1024));

    int max_cycles = 2000;
    bool verbose = true;

    if (argc > 1) max_cycles = atoi(argv[1]);
    if (argc > 2) verbose = atoi(argv[2]) != 0;

    run_plastic_fabric_v2(max_cycles, verbose);

    return 0;
}
