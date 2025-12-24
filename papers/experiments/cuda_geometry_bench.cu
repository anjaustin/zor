#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// ============================================================================
// MDS GEOMETRY CUDA BENCHMARK
// Parallel prefix carry computation for 64-bit addition
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

// Kogge-Stone parallel prefix for byte-lane carry propagation
__device__ __forceinline__ uint64_t geometry_add_64(uint64_t av, uint64_t bv) {
    // Byte-wise sums (8 parallel lanes)
    uint16_t s0 = (uint16_t)(av & 0xFF) + (bv & 0xFF);
    uint16_t s1 = (uint16_t)((av >> 8) & 0xFF) + ((bv >> 8) & 0xFF);
    uint16_t s2 = (uint16_t)((av >> 16) & 0xFF) + ((bv >> 16) & 0xFF);
    uint16_t s3 = (uint16_t)((av >> 24) & 0xFF) + ((bv >> 24) & 0xFF);
    uint16_t s4 = (uint16_t)((av >> 32) & 0xFF) + ((bv >> 32) & 0xFF);
    uint16_t s5 = (uint16_t)((av >> 40) & 0xFF) + ((bv >> 40) & 0xFF);
    uint16_t s6 = (uint16_t)((av >> 48) & 0xFF) + ((bv >> 48) & 0xFF);
    uint16_t s7 = (uint16_t)((av >> 56) & 0xFF) + ((bv >> 56) & 0xFF);

    // Generate: did this lane produce a carry?
    int g0 = s0 >> 8, g1 = s1 >> 8, g2 = s2 >> 8, g3 = s3 >> 8;
    int g4 = s4 >> 8, g5 = s5 >> 8, g6 = s6 >> 8, g7 = s7 >> 8;
    
    // Propagate: would an incoming carry propagate through?
    int p0 = (s0 & 0xFF) == 0xFF, p1 = (s1 & 0xFF) == 0xFF;
    int p2 = (s2 & 0xFF) == 0xFF, p3 = (s3 & 0xFF) == 0xFF;
    int p4 = (s4 & 0xFF) == 0xFF, p5 = (s5 & 0xFF) == 0xFF;
    int p6 = (s6 & 0xFF) == 0xFF, p7 = (s7 & 0xFF) == 0xFF;

    // Kogge-Stone level 0: stride 1
    int G0 = g0, G1 = g1 | (p1 & g0), G2 = g2 | (p2 & g1), G3 = g3 | (p3 & g2);
    int G4 = g4 | (p4 & g3), G5 = g5 | (p5 & g4), G6 = g6 | (p6 & g5), G7 = g7 | (p7 & g6);
    int P1 = p1 & p0, P2 = p2 & p1, P3 = p3 & p2;
    int P4 = p4 & p3, P5 = p5 & p4, P6 = p6 & p5, P7 = p7 & p6;

    // Level 1: stride 2
    g2 = G2 | (P2 & G0); g3 = G3 | (P3 & G1);
    g4 = G4 | (P4 & G2); g5 = G5 | (P5 & G3); g6 = G6 | (P6 & G4); g7 = G7 | (P7 & G5);
    P4 = P4 & P2; P5 = P5 & P3; P6 = P6 & P4; P7 = P7 & P5;

    // Level 2: stride 4
    G4 = g4 | (P4 & G0); G5 = g5 | (P5 & G1); G6 = g6 | (P6 & g2); G7 = g7 | (P7 & g3);

    // Assemble result: sum[i] = partial_sum[i] + carry_in[i]
    return ((s0 & 0xFF) | 
            ((uint64_t)((s1 + G0) & 0xFF) << 8) |
            ((uint64_t)((s2 + G1) & 0xFF) << 16) | 
            ((uint64_t)((s3 + g2) & 0xFF) << 24) |
            ((uint64_t)((s4 + g3) & 0xFF) << 32) | 
            ((uint64_t)((s5 + G4) & 0xFF) << 40) |
            ((uint64_t)((s6 + G5) & 0xFF) << 48) | 
            ((uint64_t)((s7 + G6) & 0xFF) << 56));
}

// Native addition for comparison
__device__ __forceinline__ uint64_t native_add_64(uint64_t a, uint64_t b) {
    return a + b;
}

__global__ void geometry_kernel(SuperCluster* superclusters, int num_superclusters, int steps) {
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
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
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

__global__ void native_kernel(SuperCluster* superclusters, int num_superclusters, int steps) {
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
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                r[i][j] = native_add_64(a[i][j], b[i][j]);
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
    printf("║     MDS GEOMETRY: CUDA PARALLEL PREFIX CARRY BENCHMARK           ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    printf("Theory:\n");
    printf("  ┌───────────────────────────────────────────────────────────────┐\n");
    printf("  │  Carries exist at positions - we read them, not compute them │\n");
    printf("  │  Kogge-Stone parallel prefix: O(log n) parallel operations   │\n");
    printf("  │  8 byte lanes × 3 levels = 24 operations vs 64 sequential    │\n");
    printf("  └───────────────────────────────────────────────────────────────┘\n\n");

    int num_superclusters = 4096;
    int steps = 1000;
    size_t total_size = sizeof(SuperCluster) * num_superclusters;

    int total_adds_per_step = num_superclusters * THREADS_PER_SUPERCLUSTER * 4 * 8;

    printf("Configuration:\n");
    printf("  SuperClusters:     %d\n", num_superclusters);
    printf("  Threads/cluster:   %d\n", THREADS_PER_SUPERCLUSTER);
    printf("  64-bit adds/step:  %d\n", total_adds_per_step);
    printf("  Steps:             %d\n", steps);
    printf("  Memory:            %.2f MB\n\n", total_size / 1e6);

    SuperCluster* d_sc;
    cudaMalloc(&d_sc, total_size);
    cudaMemset(d_sc, 0x01, total_size);

    // Warmup
    geometry_kernel<<<num_superclusters, THREADS_PER_SUPERCLUSTER>>>(d_sc, num_superclusters, steps);
    native_kernel<<<num_superclusters, THREADS_PER_SUPERCLUSTER>>>(d_sc, num_superclusters, steps);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Benchmark geometry
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        geometry_kernel<<<num_superclusters, THREADS_PER_SUPERCLUSTER>>>(d_sc, num_superclusters, steps);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float geom_ms;
    cudaEventElapsedTime(&geom_ms, start, stop);

    // Benchmark native
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        native_kernel<<<num_superclusters, THREADS_PER_SUPERCLUSTER>>>(d_sc, num_superclusters, steps);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float native_ms;
    cudaEventElapsedTime(&native_ms, start, stop);

    long long total_adds = (long long)total_adds_per_step * steps * 10;
    double geom_adds_sec = total_adds / (geom_ms / 1000.0);
    double native_adds_sec = total_adds / (native_ms / 1000.0);
    double geom_ns_per = 1e9 / geom_adds_sec;
    double native_ns_per = 1e9 / native_adds_sec;

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("RESULTS\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");
    printf("%-12s %10s %15s %12s\n", "Method", "Time", "Throughput", "ns/add");
    printf("───────────────────────────────────────────────────────────────────\n");
    printf("%-12s %8.2f ms %12.2f B/s %10.4f ns\n", "Geometry", geom_ms, geom_adds_sec/1e9, geom_ns_per);
    printf("%-12s %8.2f ms %12.2f B/s %10.4f ns\n", "Native", native_ms, native_adds_sec/1e9, native_ns_per);
    printf("───────────────────────────────────────────────────────────────────\n\n");

    printf("Analysis:\n");
    printf("  Geometry/Native ratio:  %.2fx\n", geom_ms / native_ms);
    printf("  Total 64-bit adds:      %.2f billion\n", total_adds / 1e9);
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
