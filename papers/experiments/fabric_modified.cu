#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define PARALLEL_LAYERS 4
#define CHAIN_DEPTH 4
#define CLUSTERS_PARALLEL 4
#define BITS_PER_LFSR 512
#define WORDS_PER_LFSR 8
#define THREADS_PER_SUPERCLUSTER (PARALLEL_LAYERS * CLUSTERS_PARALLEL)

struct Cluster { uint64_t lfsr[CHAIN_DEPTH][WORDS_PER_LFSR]; };
struct FabricLayer { Cluster clusters[CLUSTERS_PARALLEL]; };
struct SuperCluster { FabricLayer layers[PARALLEL_LAYERS]; };

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
    int layer_id = thread_id / CLUSTERS_PARALLEL;
    int cluster_id = thread_id % CLUSTERS_PARALLEL;
    Cluster* cluster = &superclusters[sc_id].layers[layer_id].clusters[cluster_id];
    
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
    printf("FABRIC MODIFIED TEST\n\n");  // Just added extra newline here
    
    int num_superclusters = 4096;
    int steps = 1000;
    size_t total_size = sizeof(SuperCluster) * num_superclusters;
    
    printf("Allocating %.2f MB\n", total_size / 1e6);
    
    SuperCluster* d_sc;
    cudaMalloc(&d_sc, total_size);
    cudaMemset(d_sc, 0x01, total_size);
    
    fabric_step<<<num_superclusters, THREADS_PER_SUPERCLUSTER>>>(d_sc, num_superclusters, steps);
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        fabric_step<<<num_superclusters, THREADS_PER_SUPERCLUSTER>>>(d_sc, num_superclusters, steps);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("Time: %.2f ms\n", ms);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Success!\n");
    }
    
    cudaFree(d_sc);
    return 0;
}
