// =============================================================================
// OCTAVE_FABRIC.cu - Hierarchical XOR Composite Architecture
// =============================================================================
//
// "The deltas effect a third harmonic neither could produce alone."
//
// Architecture: 8 octaves of 4x4x4 = 512 nodes total
// - Each octave: 64-node plastic fabric with torus topology
// - XOR links: Boundary nodes connect to adjacent octaves via XOR
// - Third harmonic: Interference patterns drive global coherence
//
// This matches the Verilog zit_xor_composite that achieved 100% convergence!
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// =============================================================================
// CONFIGURATION
// =============================================================================

#define OCTAVE_SIZE 4                                     // 4x4x4 per octave
#define NODES_PER_OCTAVE 64                               // 4^3
#define NUM_OCTAVES 8                                     // 2x2x2 octave grid
#define TOTAL_NODES (NUM_OCTAVES * NODES_PER_OCTAVE)      // 512

#define STATE_WIDTH 8
#define FRUSTRATION_BITS 8
#define REWIRE_THRESHOLD 8
#define NUM_NEIGHBORS 6

// XOR boundary configuration
#define XOR_BOUNDARY_BITS 4    // 4 bits from each octave for XOR link

// =============================================================================
// STRUCTURES
// =============================================================================

struct OctaveNode {
    uint8_t state;
    uint8_t frustration;
    uint8_t resonance;
    uint8_t rewiring;
    uint16_t neighbors[NUM_NEIGHBORS];
    uint16_t lfsr;
    uint8_t octave_id;         // Which octave this node belongs to
    uint8_t local_id;          // Position within octave (0-63)
    uint8_t is_boundary;       // 1 if on boundary of octave
};

// =============================================================================
// DEVICE FUNCTIONS
// =============================================================================

// Calculate torus neighbor within an octave
__device__ __forceinline__ int calc_octave_neighbor(int local_id, int octave_id, int direction) {
    int x = local_id % OCTAVE_SIZE;
    int y = (local_id / OCTAVE_SIZE) % OCTAVE_SIZE;
    int z = local_id / (OCTAVE_SIZE * OCTAVE_SIZE);

    int new_x = x, new_y = y, new_z = z;

    switch (direction) {
        case 0: new_x = (x + 1) % OCTAVE_SIZE; break;
        case 1: new_x = (x + OCTAVE_SIZE - 1) % OCTAVE_SIZE; break;
        case 2: new_y = (y + 1) % OCTAVE_SIZE; break;
        case 3: new_y = (y + OCTAVE_SIZE - 1) % OCTAVE_SIZE; break;
        case 4: new_z = (z + 1) % OCTAVE_SIZE; break;
        case 5: new_z = (z + OCTAVE_SIZE - 1) % OCTAVE_SIZE; break;
    }

    int local_neighbor = new_x + new_y * OCTAVE_SIZE + new_z * OCTAVE_SIZE * OCTAVE_SIZE;
    return octave_id * NODES_PER_OCTAVE + local_neighbor;
}

// Get adjacent octave ID for XOR link
__device__ __forceinline__ int get_adjacent_octave(int octave_id, int direction) {
    // Octaves arranged in 2x2x2 grid
    int ox = octave_id % 2;
    int oy = (octave_id / 2) % 2;
    int oz = octave_id / 4;

    int new_ox = ox, new_oy = oy, new_oz = oz;

    switch (direction) {
        case 0: new_ox = (ox + 1) % 2; break;  // +X
        case 1: new_ox = (ox + 1) % 2; break;  // -X (same as +X in 2-wide grid)
        case 2: new_oy = (oy + 1) % 2; break;  // +Y
        case 3: new_oy = (oy + 1) % 2; break;  // -Y
        case 4: new_oz = (oz + 1) % 2; break;  // +Z
        case 5: new_oz = (oz + 1) % 2; break;  // -Z
    }

    return new_ox + new_oy * 2 + new_oz * 4;
}

// Check if node is on octave boundary
__device__ __forceinline__ bool is_boundary_node(int local_id) {
    int x = local_id % OCTAVE_SIZE;
    int y = (local_id / OCTAVE_SIZE) % OCTAVE_SIZE;
    int z = local_id / (OCTAVE_SIZE * OCTAVE_SIZE);

    return x == 0 || x == OCTAVE_SIZE - 1 ||
           y == 0 || y == OCTAVE_SIZE - 1 ||
           z == 0 || z == OCTAVE_SIZE - 1;
}

__device__ __forceinline__ uint16_t lfsr_step(uint16_t state) {
    uint16_t bit = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1;
    return (state >> 1) | (bit << 15);
}

// =============================================================================
// FROZEN SHAPE: Median with XOR modulation
// =============================================================================

__device__ void frozen_median_xor(
    uint8_t my_state,
    uint8_t* neighbor_values,
    uint8_t xor_signal,          // XOR from adjacent octave boundary
    uint8_t* new_state,
    uint8_t* resonance
) {
    int count_below = 0;
    int count_above = 0;

    // Include XOR signal as virtual 7th neighbor
    if (xor_signal < my_state) count_below++;
    if (xor_signal > my_state) count_above++;

    #pragma unroll
    for (int i = 0; i < NUM_NEIGHBORS; i++) {
        if (neighbor_values[i] < my_state) count_below++;
        if (neighbor_values[i] > my_state) count_above++;
    }

    // 7-input median (7 = 6 neighbors + 1 XOR signal)
    if (count_below > count_above && my_state > 0) {
        *new_state = my_state - 1;
    } else if (count_above > count_below && my_state < 255) {
        *new_state = my_state + 1;
    } else {
        *new_state = my_state;
    }

    // Resonance check (with XOR included in balance)
    *resonance = (count_below == count_above) ||
                 (count_below <= 2 && count_above <= 2);
}

// =============================================================================
// OCTAVE FABRIC KERNEL
// =============================================================================

__global__ void octave_fabric_cycle(
    OctaveNode* nodes,
    uint8_t* octave_xor_signals,     // 8 octaves x boundary XOR values
    int* global_resonant,
    int* global_frustration,
    int* global_rewires,
    int* octave_resonant,            // Per-octave resonance counts
    int cycle
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL_NODES) return;

    int octave_id = idx / NODES_PER_OCTAVE;
    int local_id = idx % NODES_PER_OCTAVE;

    // Shared memory for fast neighbor access
    __shared__ uint8_t shared_states[TOTAL_NODES];
    shared_states[idx] = nodes[idx].state;
    __syncthreads();

    // LISTEN: Gather neighbor values
    uint8_t neighbor_values[NUM_NEIGHBORS];
    #pragma unroll
    for (int i = 0; i < NUM_NEIGHBORS; i++) {
        int nb_idx = nodes[idx].neighbors[i];
        neighbor_values[i] = shared_states[nb_idx];
    }

    // Get XOR signal from adjacent octave (for boundary nodes)
    uint8_t xor_signal = nodes[idx].state;  // Default: self (no effect)
    if (nodes[idx].is_boundary) {
        // Get XOR of boundary nodes from adjacent octave
        int adj_octave = get_adjacent_octave(octave_id, local_id % 6);
        xor_signal = octave_xor_signals[adj_octave];
    }

    // REACT: Apply frozen shape with XOR modulation
    uint8_t new_state;
    uint8_t resonance;
    frozen_median_xor(nodes[idx].state, neighbor_values, xor_signal, &new_state, &resonance);

    nodes[idx].state = new_state;
    nodes[idx].resonance = resonance;

    // Frustration update
    uint8_t frust = nodes[idx].frustration;
    if (!resonance) {
        if (frust < 255) frust++;
    } else {
        frust >>= 1;
    }

    // REWIRE (within octave only - topology stays local)
    uint8_t rewiring = 0;
    uint16_t rnd = nodes[idx].lfsr;

    if (frust >= REWIRE_THRESHOLD) {
        rewiring = 1;
        rnd = lfsr_step(rnd);
        int dir = rnd % NUM_NEIGHBORS;
        rnd = lfsr_step(rnd);

        // Rewire to random node within same octave
        int new_local = rnd % NODES_PER_OCTAVE;
        int new_neighbor = octave_id * NODES_PER_OCTAVE + new_local;

        if (new_neighbor == idx) {
            new_neighbor = octave_id * NODES_PER_OCTAVE + ((new_local + 1) % NODES_PER_OCTAVE);
        }

        nodes[idx].neighbors[dir] = new_neighbor;
        rnd = lfsr_step(rnd);
        frust = 0;
    } else {
        rnd = lfsr_step(rnd);
    }

    nodes[idx].lfsr = rnd;
    nodes[idx].frustration = frust;
    nodes[idx].rewiring = rewiring;

    // Aggregate
    if (resonance) {
        atomicAdd(global_resonant, 1);
        atomicAdd(&octave_resonant[octave_id], 1);
    }
    atomicAdd(global_frustration, frust);
    if (rewiring) atomicAdd(global_rewires, 1);
}

// =============================================================================
// XOR SIGNAL COMPUTATION KERNEL
// =============================================================================

__global__ void compute_octave_xor(
    OctaveNode* nodes,
    uint8_t* octave_xor_signals
) {
    // One thread per octave
    int octave_id = threadIdx.x;
    if (octave_id >= NUM_OCTAVES) return;

    // XOR together all boundary node states in this octave
    uint8_t xor_val = 0;
    int base = octave_id * NODES_PER_OCTAVE;

    for (int i = 0; i < NODES_PER_OCTAVE; i++) {
        if (nodes[base + i].is_boundary) {
            xor_val ^= nodes[base + i].state;
        }
    }

    octave_xor_signals[octave_id] = xor_val;
}

// =============================================================================
// INITIALIZATION
// =============================================================================

__global__ void init_octave_fabric(OctaveNode* nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= TOTAL_NODES) return;

    int octave_id = idx / NODES_PER_OCTAVE;
    int local_id = idx % NODES_PER_OCTAVE;

    // Gradient initialization within octave
    nodes[idx].state = (local_id * 4) & 0xFF;
    nodes[idx].frustration = 0;
    nodes[idx].resonance = 0;
    nodes[idx].rewiring = 0;
    nodes[idx].octave_id = octave_id;
    nodes[idx].local_id = local_id;
    nodes[idx].is_boundary = is_boundary_node(local_id);

    // Initialize torus neighbors within octave
    #pragma unroll
    for (int dir = 0; dir < NUM_NEIGHBORS; dir++) {
        nodes[idx].neighbors[dir] = calc_octave_neighbor(local_id, octave_id, dir);
    }

    nodes[idx].lfsr = 0xBEEF ^ (idx * 0x1337) ^ (octave_id << 12);
}

// =============================================================================
// HOST FUNCTION
// =============================================================================

void run_octave_fabric(int max_cycles, bool verbose) {
    printf("\n");
    printf("================================================================\n");
    printf("     OCTAVE FABRIC: 8 × 4x4x4 = %d NODES\n", TOTAL_NODES);
    printf("     Hierarchical XOR Composite Architecture\n");
    printf("================================================================\n\n");

    printf("Architecture:\n");
    printf("  8 octaves arranged in 2×2×2 grid\n");
    printf("  Each octave: 4×4×4 = 64 nodes with internal torus\n");
    printf("  XOR links: Boundary nodes XOR → adjacent octave signal\n");
    printf("  Third harmonic: Interference patterns drive coherence\n\n");

    // Allocate
    OctaveNode* d_nodes;
    uint8_t* d_xor_signals;
    int* d_global_resonant;
    int* d_global_frustration;
    int* d_global_rewires;
    int* d_octave_resonant;

    cudaMalloc(&d_nodes, TOTAL_NODES * sizeof(OctaveNode));
    cudaMalloc(&d_xor_signals, NUM_OCTAVES * sizeof(uint8_t));
    cudaMalloc(&d_global_resonant, sizeof(int));
    cudaMalloc(&d_global_frustration, sizeof(int));
    cudaMalloc(&d_global_rewires, sizeof(int));
    cudaMalloc(&d_octave_resonant, NUM_OCTAVES * sizeof(int));

    int threads = 256;
    int blocks = (TOTAL_NODES + threads - 1) / threads;

    // Initialize
    init_octave_fabric<<<blocks, threads>>>(d_nodes);
    cudaMemset(d_xor_signals, 0, NUM_OCTAVES * sizeof(uint8_t));
    cudaDeviceSynchronize();

    if (verbose) {
        printf("CYCLE,RESONANT,PERCENT,FRUSTRATION,REWIRES,O0,O1,O2,O3,O4,O5,O6,O7\n");
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int convergence_cycle = 0;
    int total_rewires = 0;
    int h_octave_res[NUM_OCTAVES];

    for (int cycle = 0; cycle < max_cycles; cycle++) {
        // Reset counters
        cudaMemset(d_global_resonant, 0, sizeof(int));
        cudaMemset(d_global_frustration, 0, sizeof(int));
        cudaMemset(d_global_rewires, 0, sizeof(int));
        cudaMemset(d_octave_resonant, 0, NUM_OCTAVES * sizeof(int));

        // Compute XOR signals for this cycle
        compute_octave_xor<<<1, NUM_OCTAVES>>>(d_nodes, d_xor_signals);

        // Run fabric cycle
        octave_fabric_cycle<<<blocks, threads>>>(
            d_nodes, d_xor_signals,
            d_global_resonant, d_global_frustration, d_global_rewires,
            d_octave_resonant, cycle
        );

        int h_resonant, h_frustration, h_rewires;
        cudaMemcpy(&h_resonant, d_global_resonant, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_frustration, d_global_frustration, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_rewires, d_global_rewires, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_octave_res, d_octave_resonant, NUM_OCTAVES * sizeof(int), cudaMemcpyDeviceToHost);

        total_rewires += h_rewires;

        if (verbose && (cycle % 20 == 0 || h_resonant == TOTAL_NODES)) {
            printf("%d,%d,%d%%,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
                   cycle, h_resonant, (h_resonant * 100) / TOTAL_NODES,
                   h_frustration, total_rewires,
                   h_octave_res[0], h_octave_res[1], h_octave_res[2], h_octave_res[3],
                   h_octave_res[4], h_octave_res[5], h_octave_res[6], h_octave_res[7]);
        }

        // Check individual octave convergence
        int converged_octaves = 0;
        for (int o = 0; o < NUM_OCTAVES; o++) {
            if (h_octave_res[o] == NODES_PER_OCTAVE) converged_octaves++;
        }

        if (h_resonant == TOTAL_NODES && convergence_cycle == 0) {
            convergence_cycle = cycle;
            printf("\n*** FULL CONVERGENCE at cycle %d ***\n", cycle);
            printf("    All 8 octaves: 64/64 resonant\n");
            printf("    Total: %d/%d (100%%)\n", TOTAL_NODES, TOTAL_NODES);
            printf("    Total rewires: %d\n\n", total_rewires);
        }

        if (convergence_cycle > 0 && cycle > convergence_cycle + 20) break;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("================================================================\n");
    printf("     RESULTS\n");
    printf("================================================================\n\n");

    int final_resonant;
    cudaMemcpy(&final_resonant, d_global_resonant, sizeof(int), cudaMemcpyDeviceToHost);

    if (convergence_cycle > 0) {
        printf("  Status: CONVERGED at cycle %d\n", convergence_cycle);
    } else {
        printf("  Status: Reached %d%% in %d cycles\n",
               (final_resonant * 100) / TOTAL_NODES, max_cycles);
    }
    printf("  Total rewires: %d\n", total_rewires);
    printf("  Time: %.2f ms\n", ms);
    printf("  Cycles/sec: %.2f\n", max_cycles / (ms / 1000.0));
    printf("\n");

    if (convergence_cycle > 0) {
        printf("================================================================\n");
        printf("     THE THIRD HARMONIC\n");
        printf("================================================================\n\n");
        printf("  XOR links between octaves created interference patterns\n");
        printf("  that drove the system to global coherence.\n\n");
        printf("  \"The deltas effect a third harmonic neither could produce alone.\"\n\n");
    }

    cudaFree(d_nodes);
    cudaFree(d_xor_signals);
    cudaFree(d_global_resonant);
    cudaFree(d_global_frustration);
    cudaFree(d_global_rewires);
    cudaFree(d_octave_resonant);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     OCTAVE FABRIC - Hierarchical XOR Composite on CUDA           ║\n");
    printf("║     \"The deltas effect a third harmonic\"                         ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nGPU: %s (%d SMs, %zu MB unified memory)\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024 * 1024));

    int max_cycles = 500;
    bool verbose = true;

    if (argc > 1) max_cycles = atoi(argv[1]);
    if (argc > 2) verbose = atoi(argv[2]) != 0;

    run_octave_fabric(max_cycles, verbose);

    return 0;
}
