#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// MDS Geometry: Kogge-Stone parallel prefix for 64-bit addition
// Same structure as fabric_4x4x4

#define PARALLEL_LAYERS 4
#define CHAIN_DEPTH 4
#define CLUSTERS_PARALLEL 4
#define WORDS_PER_CLUSTER 8
#define THREADS_PER_SUPERCLUSTER (PARALLEL_LAYERS * CLUSTERS_PARALLEL)

struct Cluster { uint64_t data[CHAIN_DEPTH][WORDS_PER_CLUSTER]; };
struct FabricLayer { Cluster clusters[CLUSTERS_PARALLEL]; };
struct SuperCluster { FabricLayer layers[PARALLEL_LAYERS]; };

__device__ __forceinline__ uint64_t geometry_add(uint64_t a, uint64_t b) {
    uint16_t s0 = (a & 0xFF) + (b & 0xFF);
    uint16_t s1 = ((a >> 8) & 0xFF) + ((b >> 8) & 0xFF);
    uint16_t s2 = ((a >> 16) & 0xFF) + ((b >> 16) & 0xFF);
    uint16_t s3 = ((a >> 24) & 0xFF) + ((b >> 24) & 0xFF);
    uint16_t s4 = ((a >> 32) & 0xFF) + ((b >> 32) & 0xFF);
    uint16_t s5 = ((a >> 40) & 0xFF) + ((b >> 40) & 0xFF);
    uint16_t s6 = ((a >> 48) & 0xFF) + ((b >> 48) & 0xFF);
    uint16_t s7 = ((a >> 56) & 0xFF) + ((b >> 56) & 0xFF);

    int g0 = s0 >> 8, g1 = s1 >> 8, g2 = s2 >> 8, g3 = s3 >> 8;
    int g4 = s4 >> 8, g5 = s5 >> 8, g6 = s6 >> 8, g7 = s7 >> 8;
    int p1 = (s1 & 0xFF) == 0xFF, p2 = (s2 & 0xFF) == 0xFF, p3 = (s3 & 0xFF) == 0xFF;
    int p4 = (s4 & 0xFF) == 0xFF, p5 = (s5 & 0xFF) == 0xFF, p6 = (s6 & 0xFF) == 0xFF, p7 = (s7 & 0xFF) == 0xFF;

    int G1 = g1 | (p1 & g0), G2 = g2 | (p2 & g1), G3 = g3 | (p3 & g2);
    int G4 = g4 | (p4 & g3), G5 = g5 | (p5 & g4), G6 = g6 | (p6 & g5), G7 = g7 | (p7 & g6);
    g2 = G2 | ((p2 & p1) & g0); g3 = G3 | ((p3 & p2) & G1);
    g4 = G4 | ((p4 & p3) & G2); g5 = G5 | ((p5 & p4) & G3);
    g6 = G6 | ((p6 & p5) & g4); g7 = G7 | ((p7 & p6) & g5);
    G4 = g4 | ((p4 & p3 & p2 & p1) & g0); G5 = g5 | ((p5 & p4 & p3 & p2) & G1);
    G6 = g6 | ((p6 & p5 & p4 & p3) & g2); G7 = g7 | ((p7 & p6 & p5 & p4) & g3);

    return (s0 & 0xFF) | ((uint64_t)((s1 + g0) & 0xFF) << 8) |
           ((uint64_t)((s2 + G1) & 0xFF) << 16) | ((uint64_t)((s3 + g2) & 0xFF) << 24) |
           ((uint64_t)((s4 + g3) & 0xFF) << 32) | ((uint64_t)((s5 + G4) & 0xFF) << 40) |
           ((uint64_t)((s6 + G5) & 0xFF) << 48) | ((uint64_t)((s7 + G6) & 0xFF) << 56);
}

__global__ void geometry_step(SuperCluster* sc, int num_sc, int steps) {
    int sc_id = blockIdx.x;
    int tid = threadIdx.x;
    if (sc_id >= num_sc || tid >= THREADS_PER_SUPERCLUSTER) return;

    Cluster* c = &sc[sc_id].layers[tid / CLUSTERS_PARALLEL].clusters[tid % CLUSTERS_PARALLEL];
    uint64_t d[4][8];
    
    #pragma unroll
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++)
            d[i][j] = c->data[i][j];

    for (int s = 0; s < steps; s++) {
        #pragma unroll
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++)
                d[i][j] = geometry_add(d[i][j], d[(i+1)%4][j]);
    }

    #pragma unroll
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++)
            c->data[i][j] = d[i][j];
}

int main() {
    printf("MDS GEOMETRY CUDA BENCHMARK\n");
    printf("===========================\n\n");

    int num_sc = 4096, steps = 1000;
    size_t size = sizeof(SuperCluster) * num_sc;
    int adds_per_step = num_sc * THREADS_PER_SUPERCLUSTER * 4 * 8;

    printf("SuperClusters: %d\n", num_sc);
    printf("64-bit adds/step: %d\n", adds_per_step);
    printf("Steps: %d\n", steps);
    printf("Memory: %.2f MB\n\n", size / 1e6);

    SuperCluster* d_sc;
    cudaMalloc(&d_sc, size);
    cudaMemset(d_sc, 0x01, size);

    geometry_step<<<num_sc, THREADS_PER_SUPERCLUSTER>>>(d_sc, num_sc, steps);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        geometry_step<<<num_sc, THREADS_PER_SUPERCLUSTER>>>(d_sc, num_sc, steps);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    long long total = (long long)adds_per_step * steps * 10;
    printf("Time: %.2f ms\n", ms);
    printf("Total adds: %.2f billion\n", total / 1e9);
    printf("Throughput: %.2f billion/sec\n", total / (ms / 1000.0) / 1e9);
    printf("ns/add: %.4f\n", (ms * 1e6) / total);

    cudaFree(d_sc);
    return 0;
}
