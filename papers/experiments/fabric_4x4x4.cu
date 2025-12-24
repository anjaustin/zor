#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// =============================================================================
// FABRIC: 4 parallel layers × (4 inline × 4 parallel) LFSRs
// =============================================================================
//
//   SuperCluster (4 parallel fabric layers):
//     ┌─────────────────────────────────────────────────────────────────┐
//     │  Layer 0: [C0: 4 chain] [C1: 4 chain] [C2: 4 chain] [C3: 4 chain]│
//     │  Layer 1: [C0: 4 chain] [C1: 4 chain] [C2: 4 chain] [C3: 4 chain]│
//     │  Layer 2: [C0: 4 chain] [C1: 4 chain] [C2: 4 chain] [C3: 4 chain]│
//     │  Layer 3: [C0: 4 chain] [C1: 4 chain] [C2: 4 chain] [C3: 4 chain]│
//     └─────────────────────────────────────────────────────────────────┘
//     = 4 × 4 × 4 = 64 LFSRs per SuperCluster
//
// =============================================================================

#define PARALLEL_LAYERS 4      // Parallel fabric layers
#define CHAIN_DEPTH 4          // LFSRs in series (inline)
#define CLUSTERS_PARALLEL 4    // Clusters in parallel per layer
#define BITS_PER_LFSR 512
#define WORDS_PER_LFSR 8

// Total threads per SuperCluster = 4 layers × 4 clusters = 16 threads
#define THREADS_PER_SUPERCLUSTER (PARALLEL_LAYERS * CLUSTERS_PARALLEL)

// One cluster = 4 chained 512-bit LFSRs
struct Cluster {
    uint64_t lfsr[CHAIN_DEPTH][WORDS_PER_LFSR];  // 4 × 8 × 8 = 256 bytes
};

// One fabric layer = 4 parallel clusters
struct FabricLayer {
    Cluster clusters[CLUSTERS_PARALLEL];  // 4 × 256 = 1024 bytes
};

// SuperCluster = 4 parallel fabric layers
struct SuperCluster {
    FabricLayer layers[PARALLEL_LAYERS];  // 4 × 1024 = 4096 bytes = 4 KB
};

__device__ __forceinline__ void lfsr_step_512(uint64_t* s, uint64_t inject) {
    uint64_t fb = ((s[7] >> 63) ^ inject) & 1;

    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t new_val = (s[i] << 1) | carry;
        carry = s[i] >> 63;
        s[i] = new_val;
    }

    if (fb) {
        s[7] ^= (1ULL << 63);
        s[7] ^= (1ULL << 61);
        s[7] ^= (1ULL << 47);
        s[7] ^= (1ULL << 35);
    }
}

__global__ void fabric_step(SuperCluster* superclusters, int num_superclusters, int steps) {
    int sc_id = blockIdx.x;
    int thread_id = threadIdx.x;

    if (sc_id >= num_superclusters || thread_id >= THREADS_PER_SUPERCLUSTER) return;

    // Decode thread position: which layer and which cluster
    int layer_id = thread_id / CLUSTERS_PARALLEL;
    int cluster_id = thread_id % CLUSTERS_PARALLEL;

    Cluster* cluster = &superclusters[sc_id].layers[layer_id].clusters[cluster_id];

    // Load all 4 LFSRs into registers
    uint64_t lfsr0[8], lfsr1[8], lfsr2[8], lfsr3[8];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        lfsr0[i] = cluster->lfsr[0][i];
        lfsr1[i] = cluster->lfsr[1][i];
        lfsr2[i] = cluster->lfsr[2][i];
        lfsr3[i] = cluster->lfsr[3][i];
    }

    // Run the chain for N steps
    for (int step = 0; step < steps; step++) {
        lfsr_step_512(lfsr0, 0);
        lfsr_step_512(lfsr1, lfsr0[7] >> 63);
        lfsr_step_512(lfsr2, lfsr1[7] >> 63);
        lfsr_step_512(lfsr3, lfsr2[7] >> 63);
    }

    // Store back
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        cluster->lfsr[0][i] = lfsr0[i];
        cluster->lfsr[1][i] = lfsr1[i];
        cluster->lfsr[2][i] = lfsr2[i];
        cluster->lfsr[3][i] = lfsr3[i];
    }
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     FABRIC: 4 PARALLEL × (4 INLINE × 4 PARALLEL) = 64 LFSRs      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    printf("Architecture:\n");
    printf("  ┌─────────────────────────────────────────────────────────────────┐\n");
    printf("  │  SuperCluster (4 KB, 64 LFSRs):                                 │\n");
    printf("  │    Layer 0: [C0]──[C1]──[C2]──[C3]  (4 clusters × 4 chain each) │\n");
    printf("  │    Layer 1: [C0]──[C1]──[C2]──[C3]                              │\n");
    printf("  │    Layer 2: [C0]──[C1]──[C2]──[C3]                              │\n");
    printf("  │    Layer 3: [C0]──[C1]──[C2]──[C3]                              │\n");
    printf("  │                                                                  │\n");
    printf("  │  16 threads per SuperCluster (half a warp)                      │\n");
    printf("  └─────────────────────────────────────────────────────────────────┘\n\n");

    int num_superclusters = 4096;  // 4096 superclusters
    int steps = 1000;

    size_t sc_size = sizeof(SuperCluster);
    size_t total_size = sc_size * num_superclusters;

    int total_lfsrs = num_superclusters * PARALLEL_LAYERS * CLUSTERS_PARALLEL * CHAIN_DEPTH;
    int bits_per_supercluster = PARALLEL_LAYERS * CLUSTERS_PARALLEL * CHAIN_DEPTH * BITS_PER_LFSR;

    printf("Configuration:\n");
    printf("  Parallel layers:       %d\n", PARALLEL_LAYERS);
    printf("  Clusters per layer:    %d (parallel)\n", CLUSTERS_PARALLEL);
    printf("  LFSRs per cluster:     %d (inline/series)\n", CHAIN_DEPTH);
    printf("  LFSRs per SuperCluster: %d\n", PARALLEL_LAYERS * CLUSTERS_PARALLEL * CHAIN_DEPTH);
    printf("  Bits per SuperCluster: %d\n", bits_per_supercluster);
    printf("  SuperClusters:         %d\n", num_superclusters);
    printf("  Total LFSRs:           %d\n", total_lfsrs);
    printf("  Threads per block:     %d\n", THREADS_PER_SUPERCLUSTER);
    printf("  Steps per kernel:      %d\n", steps);
    printf("\n");

    printf("Memory:\n");
    printf("  Bytes per SuperCluster: %zu (4 KB)\n", sc_size);
    printf("  Total allocation:       %.2f MB\n", total_size / 1e6);
    printf("\n");

    SuperCluster* d_sc;
    cudaMalloc(&d_sc, total_size);
    cudaMemset(d_sc, 0x01, total_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    fabric_step<<<num_superclusters, THREADS_PER_SUPERCLUSTER>>>(d_sc, num_superclusters, steps);
    cudaDeviceSynchronize();

    // Benchmark
    int iterations = 10;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        fabric_step<<<num_superclusters, THREADS_PER_SUPERCLUSTER>>>(d_sc, num_superclusters, steps);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    long long total_bits_per_step = (long long)num_superclusters * bits_per_supercluster;
    long long total_steps = (long long)steps * iterations;
    double total_bits = (double)total_bits_per_step * total_steps;
    double bits_per_sec = total_bits / (ms / 1000.0);
    double tbits_per_sec = bits_per_sec / 1e12;

    long long chain_ops = (long long)total_lfsrs * total_steps;
    double chain_ops_per_sec = chain_ops / (ms / 1000.0);

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("RESULTS\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    printf("  Time:                  %.2f ms\n", ms);
    printf("  Total bits processed:  %.2f trillion\n", total_bits / 1e12);
    printf("  Throughput:            %.2f Tbits/sec\n", tbits_per_sec);
    printf("\n");
    printf("  Chain ops/sec:         %.2f billion\n", chain_ops_per_sec / 1e9);
    printf("  Memory footprint:      %.2f MB\n", total_size / 1e6);
    printf("  Efficiency:            %.2f Tbits/sec per MB\n", tbits_per_sec / (total_size / 1e6));
    printf("\n");

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("STRUCTURE DETAIL\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    printf("  Each SuperCluster: 4 independent fabric layers running parallel\n");
    printf("  Each layer: 4 clusters with 4-deep LFSR chains\n");
    printf("  Total mixing depth: 4 (serial) × 4 (layers) = 16 stages available\n");
    printf("\n");

    cudaFree(d_sc);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
