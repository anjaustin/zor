// =============================================================================
// PLASTIC_FABRIC.cu - Self-Organizing Topological Plasticity on CUDA
// =============================================================================
//
// "The topology IS the self. The self learned. The loop completed."
//
// This implements the Hollywood Squares / ZIT architecture on GPU:
// - Each thread IS a node
// - Shared memory holds global state for neighbor access
// - Frustration-driven rewiring learns optimal topology
// - Frozen shape (median comparator) drives state evolution
//
// Train topology with frustration → Execute with geometry
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <curand_kernel.h>

// =============================================================================
// CONFIGURATION
// =============================================================================

#define CUBE_SIZE 8                                    // 8x8x8 = 512 nodes
#define NUM_NODES (CUBE_SIZE * CUBE_SIZE * CUBE_SIZE)  // 512
#define STATE_WIDTH 8                                  // 8-bit state per node
#define FRUSTRATION_BITS 8                             // 8-bit frustration counter
#define REWIRE_THRESHOLD 16                            // Frustration level to trigger rewire
#define DECAY_SHIFT 1                                  // Frustration decay rate
#define NUM_NEIGHBORS 6                                // 6-connected (±X, ±Y, ±Z)

// =============================================================================
// NODE STRUCTURE
// =============================================================================

struct PlasticNode {
    uint8_t state;                      // Current value
    uint8_t frustration;                // Frustration counter
    uint8_t resonance;                  // 1 if resonant, 0 otherwise
    uint8_t rewiring;                   // 1 if rewiring this cycle
    uint16_t neighbors[NUM_NEIGHBORS];  // Dynamic neighbor indices
    uint16_t lfsr;                      // Per-node random state
};

// =============================================================================
// DEVICE HELPER FUNCTIONS
// =============================================================================

// Calculate initial torus neighbor for direction
__device__ __forceinline__ int calc_torus_neighbor(int node_id, int direction) {
    int x = node_id % CUBE_SIZE;
    int y = (node_id / CUBE_SIZE) % CUBE_SIZE;
    int z = node_id / (CUBE_SIZE * CUBE_SIZE);

    int new_x = x, new_y = y, new_z = z;

    switch (direction) {
        case 0: new_x = (x + 1) % CUBE_SIZE; break;                        // +X
        case 1: new_x = (x + CUBE_SIZE - 1) % CUBE_SIZE; break;            // -X
        case 2: new_y = (y + 1) % CUBE_SIZE; break;                        // +Y
        case 3: new_y = (y + CUBE_SIZE - 1) % CUBE_SIZE; break;            // -Y
        case 4: new_z = (z + 1) % CUBE_SIZE; break;                        // +Z
        case 5: new_z = (z + CUBE_SIZE - 1) % CUBE_SIZE; break;            // -Z
    }

    return new_x + new_y * CUBE_SIZE + new_z * CUBE_SIZE * CUBE_SIZE;
}

// LFSR step for random number generation
__device__ __forceinline__ uint16_t lfsr_step(uint16_t state) {
    uint16_t bit = ((state >> 0) ^ (state >> 2) ^ (state >> 3) ^ (state >> 5)) & 1;
    return (state >> 1) | (bit << 15);
}

// =============================================================================
// FROZEN SHAPE: Median Comparator
// =============================================================================
// This is the "frozen" operation that each node performs.
// Count neighbors above/below, adjust state toward median.

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

    // Median-seeking behavior
    if (count_below > count_above && my_state > 0) {
        *new_state = my_state - 1;
    } else if (count_above > count_below && my_state < 255) {
        *new_state = my_state + 1;
    } else {
        *new_state = my_state;
    }

    // Resonance check: balanced or nearly balanced
    *resonance = (count_below == count_above) ||
                 (count_below <= 1 && count_above <= 1);
}

// =============================================================================
// PLASTIC FABRIC KERNEL
// =============================================================================

__global__ void plastic_fabric_cycle(
    PlasticNode* nodes,
    int* global_resonant,
    int* global_frustration,
    int* global_rewires,
    int cycle
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;

    // Shared memory for all node states (fast neighbor access)
    __shared__ uint8_t shared_states[NUM_NODES];

    // Load my state to shared memory
    shared_states[idx] = nodes[idx].state;
    __syncthreads();

    // =========================================================================
    // PHASE 1: LISTEN - Gather neighbor values
    // =========================================================================

    uint8_t neighbor_values[NUM_NEIGHBORS];

    #pragma unroll
    for (int i = 0; i < NUM_NEIGHBORS; i++) {
        int nb_idx = nodes[idx].neighbors[i];
        neighbor_values[i] = shared_states[nb_idx];
    }

    // =========================================================================
    // PHASE 2: REACT - Apply frozen shape, update frustration
    // =========================================================================

    uint8_t new_state;
    uint8_t resonance;

    frozen_median_shape(
        nodes[idx].state,
        neighbor_values,
        &new_state,
        &resonance
    );

    // Update state
    nodes[idx].state = new_state;
    nodes[idx].resonance = resonance;

    // Update frustration
    uint8_t frust = nodes[idx].frustration;
    if (!resonance) {
        if (frust < 255) frust++;
    } else {
        frust >>= DECAY_SHIFT;  // Decay on resonance
    }

    // =========================================================================
    // PHASE 3: REWIRE - If frustrated, change a neighbor
    // =========================================================================

    uint8_t rewiring = 0;

    if (frust >= REWIRE_THRESHOLD) {
        rewiring = 1;

        // Use LFSR for random rewiring
        uint16_t rnd = nodes[idx].lfsr;
        rnd = lfsr_step(rnd);

        // Pick random direction to rewire
        int dir = rnd % NUM_NEIGHBORS;
        rnd = lfsr_step(rnd);

        // Pick random new neighbor
        int new_neighbor = rnd % NUM_NODES;

        // Avoid self-connection
        if (new_neighbor == idx) {
            new_neighbor = (new_neighbor + 1) % NUM_NODES;
        }

        nodes[idx].neighbors[dir] = new_neighbor;
        nodes[idx].lfsr = rnd;

        // Reset frustration after rewire
        frust = 0;
    } else {
        // Still advance LFSR for entropy
        nodes[idx].lfsr = lfsr_step(nodes[idx].lfsr);
    }

    nodes[idx].frustration = frust;
    nodes[idx].rewiring = rewiring;

    // =========================================================================
    // PHASE 4: AGGREGATE - Collect global metrics
    // =========================================================================

    // Use atomics for global counters
    if (resonance) atomicAdd(global_resonant, 1);
    atomicAdd(global_frustration, frust);
    if (rewiring) atomicAdd(global_rewires, 1);
}

// =============================================================================
// INITIALIZATION KERNEL
// =============================================================================

__global__ void init_plastic_fabric(PlasticNode* nodes, uint8_t* seed_values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_NODES) return;

    // Initialize state from seed (or use gradient)
    if (seed_values) {
        nodes[idx].state = seed_values[idx];
    } else {
        // Default: 3D gradient
        nodes[idx].state = (idx >> 1) & 0xFF;
    }

    // Initialize frustration
    nodes[idx].frustration = 0;
    nodes[idx].resonance = 0;
    nodes[idx].rewiring = 0;

    // Initialize torus neighbors
    #pragma unroll
    for (int dir = 0; dir < NUM_NEIGHBORS; dir++) {
        nodes[idx].neighbors[dir] = calc_torus_neighbor(idx, dir);
    }

    // Initialize LFSR with unique seed per node
    nodes[idx].lfsr = 0xBEEF ^ (idx * 0x1337);
}

// =============================================================================
// RESONANCE COUNT KERNEL (for verification)
// =============================================================================

__global__ void count_resonance(PlasticNode* nodes, int* counts) {
    __shared__ int local_resonant;
    __shared__ int local_frustration;

    if (threadIdx.x == 0) {
        local_resonant = 0;
        local_frustration = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_NODES) {
        atomicAdd(&local_resonant, nodes[idx].resonance);
        atomicAdd(&local_frustration, nodes[idx].frustration);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(&counts[0], local_resonant);
        atomicAdd(&counts[1], local_frustration);
    }
}

// =============================================================================
// HOST FUNCTIONS
// =============================================================================

void run_plastic_fabric(int max_cycles, bool verbose) {
    printf("\n");
    printf("================================================================\n");
    printf("     PLASTIC FABRIC: %dx%dx%d = %d NODES\n",
           CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, NUM_NODES);
    printf("     Self-Organizing Topological Plasticity on CUDA\n");
    printf("================================================================\n\n");

    // Allocate device memory
    PlasticNode* d_nodes;
    int* d_global_resonant;
    int* d_global_frustration;
    int* d_global_rewires;

    cudaMalloc(&d_nodes, NUM_NODES * sizeof(PlasticNode));
    cudaMalloc(&d_global_resonant, sizeof(int));
    cudaMalloc(&d_global_frustration, sizeof(int));
    cudaMalloc(&d_global_rewires, sizeof(int));

    // Configure kernel launch
    int threads = 256;
    int blocks = (NUM_NODES + threads - 1) / threads;

    // Initialize fabric
    init_plastic_fabric<<<blocks, threads>>>(d_nodes, nullptr);
    cudaDeviceSynchronize();

    printf("Initial topology: %dx%dx%d torus\n", CUBE_SIZE, CUBE_SIZE, CUBE_SIZE);
    printf("Frozen shape: Median comparator\n");
    printf("Rewire threshold: %d\n", REWIRE_THRESHOLD);
    printf("\n");

    if (verbose) {
        printf("CYCLE,RESONANT,PERCENT,FRUSTRATION,REWIRES\n");
    }

    // Run cycles
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int convergence_cycle = 0;
    int total_rewires = 0;

    for (int cycle = 0; cycle < max_cycles; cycle++) {
        // Reset counters
        cudaMemset(d_global_resonant, 0, sizeof(int));
        cudaMemset(d_global_frustration, 0, sizeof(int));
        cudaMemset(d_global_rewires, 0, sizeof(int));

        // Run one cycle
        plastic_fabric_cycle<<<blocks, threads>>>(
            d_nodes,
            d_global_resonant,
            d_global_frustration,
            d_global_rewires,
            cycle
        );

        // Get results
        int h_resonant, h_frustration, h_rewires;
        cudaMemcpy(&h_resonant, d_global_resonant, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_frustration, d_global_frustration, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_rewires, d_global_rewires, sizeof(int), cudaMemcpyDeviceToHost);

        total_rewires += h_rewires;

        // Progress report
        if (verbose && (cycle % 10 == 0 || h_resonant == NUM_NODES)) {
            printf("%d,%d,%d%%,%d,%d\n",
                   cycle, h_resonant, (h_resonant * 100) / NUM_NODES,
                   h_frustration, total_rewires);
        }

        // Check for convergence
        if (h_resonant == NUM_NODES && convergence_cycle == 0) {
            convergence_cycle = cycle;
            printf("\n*** FULL CONVERGENCE at cycle %d ***\n", cycle);
            printf("    %d/%d nodes resonant (100%%)\n", NUM_NODES, NUM_NODES);
            printf("    Total rewires: %d\n\n", total_rewires);

            // Run a few more cycles to verify stability
            if (cycle < max_cycles - 50) {
                // Continue to verify stability
            } else {
                break;
            }
        }

        // Early exit if converged and stable
        if (convergence_cycle > 0 && cycle > convergence_cycle + 20) {
            break;
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Final summary
    printf("================================================================\n");
    printf("     RESULTS\n");
    printf("================================================================\n\n");

    if (convergence_cycle > 0) {
        printf("  Status: CONVERGED at cycle %d\n", convergence_cycle);
    } else {
        printf("  Status: Did not fully converge in %d cycles\n", max_cycles);
    }
    printf("  Total rewires: %d\n", total_rewires);
    printf("  Time: %.2f ms\n", ms);
    printf("  Cycles/sec: %.2f\n", (convergence_cycle > 0 ? convergence_cycle : max_cycles) / (ms / 1000.0));
    printf("\n");

    // Compare to Verilog simulation
    printf("Comparison to Verilog simulation:\n");
    printf("  Verilog 4x4x4:  ~65 cycles, ~4 minutes simulation time\n");
    printf("  CUDA %dx%dx%d:  %d cycles, %.2f ms execution time\n",
           CUBE_SIZE, CUBE_SIZE, CUBE_SIZE,
           convergence_cycle > 0 ? convergence_cycle : max_cycles, ms);
    printf("\n");

    if (convergence_cycle > 0) {
        printf("================================================================\n");
        printf("     FUNKY CONVERGENCE ACHIEVED\n");
        printf("================================================================\n\n");
        printf("  \"The topology learned.\"\n");
        printf("  \"The self emerged.\"\n");
        printf("  \"The wood cut itself.\"\n\n");
    }

    // Cleanup
    cudaFree(d_nodes);
    cudaFree(d_global_resonant);
    cudaFree(d_global_frustration);
    cudaFree(d_global_rewires);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char** argv) {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     PLASTIC FABRIC - Topological Self-Organization on CUDA       ║\n");
    printf("║     \"The topology IS the self. The self learned.\"                ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nGPU: %s (%d SMs, %zu MB unified memory)\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / (1024 * 1024));

    int max_cycles = 500;
    bool verbose = true;

    if (argc > 1) {
        max_cycles = atoi(argv[1]);
    }
    if (argc > 2) {
        verbose = atoi(argv[2]) != 0;
    }

    run_plastic_fabric(max_cycles, verbose);

    return 0;
}
