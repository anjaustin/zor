#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// =============================================================================
// FABRIC: 4 LFSRs in series × 4 clusters per layer
// =============================================================================

#define CHAIN_DEPTH 4
#define CLUSTERS_PER_LAYER 4
#define BITS_PER_LFSR 512
#define WORDS_PER_LFSR 8

struct Cluster {
    uint64_t lfsr[CHAIN_DEPTH][WORDS_PER_LFSR];
};

struct FabricLayer {
    Cluster clusters[CLUSTERS_PER_LAYER];
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

__global__ void fabric_step(FabricLayer* layers, int num_layers, int steps) {
    int layer_id = blockIdx.x;
    int cluster_id = threadIdx.x;

    if (layer_id >= num_layers || cluster_id >= CLUSTERS_PER_LAYER) return;

    Cluster* cluster = &layers[layer_id].clusters[cluster_id];

    uint64_t lfsr0[8], lfsr1[8], lfsr2[8], lfsr3[8];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        lfsr0[i] = cluster->lfsr[0][i];
        lfsr1[i] = cluster->lfsr[1][i];
        lfsr2[i] = cluster->lfsr[2][i];
        lfsr3[i] = cluster->lfsr[3][i];
    }

    for (int step = 0; step < steps; step++) {
        lfsr_step_512(lfsr0, 0);
        lfsr_step_512(lfsr1, lfsr0[7] >> 63);
        lfsr_step_512(lfsr2, lfsr1[7] >> 63);
        lfsr_step_512(lfsr3, lfsr2[7] >> 63);
    }

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
    printf("║           FABRIC: 4×4 CHAINED LFSR CLUSTERS                       ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    int num_layers = 4096;  // More layers to compensate for fewer clusters
    int steps = 1000;

    size_t layer_size = sizeof(FabricLayer);
    size_t total_size = layer_size * num_layers;

    printf("Configuration:\n");
    printf("  LFSRs per cluster:     %d (in series)\n", CHAIN_DEPTH);
    printf("  Clusters per layer:    %d (parallel)\n", CLUSTERS_PER_LAYER);
    printf("  Bits per cluster:      %d\n", CHAIN_DEPTH * BITS_PER_LFSR);
    printf("  Bits per layer:        %d\n", CLUSTERS_PER_LAYER * CHAIN_DEPTH * BITS_PER_LFSR);
    printf("  Fabric layers:         %d\n", num_layers);
    printf("  Total LFSRs:           %d\n", num_layers * CLUSTERS_PER_LAYER * CHAIN_DEPTH);
    printf("  Steps per kernel:      %d\n", steps);
    printf("\n");

    printf("Memory:\n");
    printf("  Bytes per cluster:     %zu\n", sizeof(Cluster));
    printf("  Bytes per layer:       %zu\n", layer_size);
    printf("  Total allocation:      %.2f MB\n", total_size / 1e6);
    printf("\n");

    FabricLayer* d_layers;
    cudaMalloc(&d_layers, total_size);
    cudaMemset(d_layers, 0x01, total_size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    fabric_step<<<num_layers, CLUSTERS_PER_LAYER>>>(d_layers, num_layers, steps);
    cudaDeviceSynchronize();

    // Benchmark
    int iterations = 10;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        fabric_step<<<num_layers, CLUSTERS_PER_LAYER>>>(d_layers, num_layers, steps);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    long long total_bits_per_step = (long long)num_layers * CLUSTERS_PER_LAYER * CHAIN_DEPTH * BITS_PER_LFSR;
    long long total_steps = (long long)steps * iterations;
    double total_bits = (double)total_bits_per_step * total_steps;
    double bits_per_sec = total_bits / (ms / 1000.0);
    double tbits_per_sec = bits_per_sec / 1e12;

    long long chain_ops_per_step = (long long)num_layers * CLUSTERS_PER_LAYER * CHAIN_DEPTH;
    double chain_ops_per_sec = (double)chain_ops_per_step * total_steps / (ms / 1000.0);

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

    cudaFree(d_layers);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
