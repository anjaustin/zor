#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// ============================================================================
// GEOMETRY ADD: Using exact fabric structure but doing parallel prefix addition
// ============================================================================

#define PARALLEL_LAYERS 4
#define CHAIN_DEPTH 4
#define CLUSTERS_PARALLEL 4
#define WORDS_PER_CLUSTER 8
#define THREADS_PER_SUPERCLUSTER (PARALLEL_LAYERS * CLUSTERS_PARALLEL)

struct Cluster { 
    uint64_t a[CHAIN_DEPTH][WORDS_PER_CLUSTER];
    uint64_t b[CHAIN_DEPTH][WORDS_PER_CLUSTER];
    uint64_t result[CHAIN_DEPTH][WORDS_PER_CLUSTER];
};
struct FabricLayer { Cluster clusters[CLUSTERS_PARALLEL]; };
struct SuperCluster { FabricLayer layers[PARALLEL_LAYERS]; };

// Geometry add for one 64-bit value
__device__ __forceinline__ uint64_t geometry_add_64(uint64_t av, uint64_t bv) {
    // Byte-wise addition
    uint16_t s0 = (uint16_t)(av & 0xFF) + (bv & 0xFF);
    uint16_t s1 = (uint16_t)((av >> 8) & 0xFF) + ((bv >> 8) & 0xFF);
    uint16_t s2 = (uint16_t)((av >> 16) & 0xFF) + ((bv >> 16) & 0xFF);
    uint16_t s3 = (uint16_t)((av >> 24) & 0xFF) + ((bv >> 24) & 0xFF);
    uint16_t s4 = (uint16_t)((av >> 32) & 0xFF) + ((bv >> 32) & 0xFF);
    uint16_t s5 = (uint16_t)((av >> 40) & 0xFF) + ((bv >> 40) & 0xFF);
    uint16_t s6 = (uint16_t)((av >> 48) & 0xFF) + ((bv >> 48) & 0xFF);
    uint16_t s7 = (uint16_t)((av >> 56) & 0xFF) + ((bv >> 56) & 0xFF);

    // Generate/Propagate
    int g0 = s0 >> 8, g1 = s1 >> 8, g2 = s2 >> 8, g3 = s3 >> 8;
    int g4 = s4 >> 8, g5 = s5 >> 8, g6 = s6 >> 8, g7 = s7 >> 8;
    int p0 = (s0 & 0xFF) == 0xFF, p1 = (s1 & 0xFF) == 0xFF;
    int p2 = (s2 & 0xFF) == 0xFF, p3 = (s3 & 0xFF) == 0xFF;
    int p4 = (s4 & 0xFF) == 0xFF, p5 = (s5 & 0xFF) == 0xFF;
    int p6 = (s6 & 0xFF) == 0xFF, p7 = (s7 & 0xFF) == 0xFF;

    // Kogge-Stone level 0
    int G0 = g0, G1 = g1 | (p1 & g0), G2 = g2 | (p2 & g1), G3 = g3 | (p3 & g2);
    int G4 = g4 | (p4 & g3), G5 = g5 | (p5 & g4), G6 = g6 | (p6 & g5), G7 = g7 | (p7 & g6);
    int P2 = p2 & p1, P3 = p3 & p2, P4 = p4 & p3, P5 = p5 & p4, P6 = p6 & p5, P7 = p7 & p6;

    // Level 1
    g2 = G2 | (P2 & G0); g3 = G3 | (P3 & G1);
    g4 = G4 | (P4 & G2); g5 = G5 | (P5 & G3); g6 = G6 | (P6 & G4); g7 = G7 | (P7 & G5);

    // Level 2
    G4 = g4 | (P4 & P2 & G0); G5 = g5 | (P5 & P3 & G1); G6 = g6 | (P6 & P4 & g2); G7 = g7 | (P7 & P5 & g3);

    return ((s0 & 0xFF) | 
            ((uint64_t)((s1 + G0) & 0xFF) << 8) |
            ((uint64_t)((s2 + G1) & 0xFF) << 16) | 
            ((uint64_t)((s3 + g2) & 0xFF) << 24) |
            ((uint64_t)((s4 + g3) & 0xFF) << 32) | 
            ((uint64_t)((s5 + G4) & 0xFF) << 40) |
            ((uint64_t)((s6 + G5) & 0xFF) << 48) | 
            ((uint64_t)((s7 + G6) & 0xFF) << 56));
}

__global__ void geometry_step(SuperCluster* superclusters, int num_superclusters, int steps) {
    int sc_id = blockIdx.x;
    int thread_id = threadIdx.x;
    if (sc_id >= num_superclusters || thread_id >= THREADS_PER_SUPERCLUSTER) return;
    
    int layer_id = thread_id / CLUSTERS_PARALLEL;
    int cluster_id = thread_id % CLUSTERS_PARALLEL;
    Cluster* cluster = &superclusters[sc_id].layers[layer_id].clusters[cluster_id];
    
    uint64_t a[4][8], b[4][8], r[4][8];
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            a[i][j] = cluster->a[i][j];
            b[i][j] = cluster->b[i][j];
        }
    }
    
    for (int step = 0; step < steps; step++) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                r[i][j] = geometry_add_64(a[i][j], b[i][j]);
            }
        }
    }
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            cluster->result[i][j] = r[i][j];
        }
    }
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     GEOMETRY: PARALLEL PREFIX ADD IN FABRIC STRUCTURE            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    
    int num_superclusters = 4096;
    int steps = 1000;
    size_t total_size = sizeof(SuperCluster) * num_superclusters;
    
    printf("Configuration:\n");
    printf("  SuperClusters:    %d\n", num_superclusters);
    printf("  Steps:            %d\n", steps);
    printf("  Threads/block:    %d\n", THREADS_PER_SUPERCLUSTER);
    printf("  Memory:           %.2f MB\n\n", total_size / 1e6);
    
    SuperCluster* d_sc;
    cudaMalloc(&d_sc, total_size);
    cudaMemset(d_sc, 0x01, total_size);
    
    // Warmup
    geometry_step<<<num_superclusters, THREADS_PER_SUPERCLUSTER>>>(d_sc, num_superclusters, steps);
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        geometry_step<<<num_superclusters, THREADS_PER_SUPERCLUSTER>>>(d_sc, num_superclusters, steps);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    long long total_adds = (long long)num_superclusters * THREADS_PER_SUPERCLUSTER * 4 * 8 * steps * 10;
    double adds_per_sec = total_adds / (ms / 1000.0);
    
    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("RESULTS\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    printf("  Time:              %.2f ms\n", ms);
    printf("  Total 64-bit adds: %.2f billion\n", total_adds / 1e9);
    printf("  Throughput:        %.2f billion adds/sec\n", adds_per_sec / 1e9);
    printf("  ns per add:        %.4f ns\n", 1e9 / adds_per_sec);
    printf("\n");
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Status: SUCCESS\n");
    }
    
    cudaFree(d_sc);
    return 0;
}
