// =============================================================================
// 256x256x256 = 16,777,216 NODES - 24D Hypercube
// Pushing toward 32D limit on Thor (131GB)
// Second Star Constant Seed: 1122911624
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define CUBE_SIZE 256
#define NUM_NODES 16777216
#define THREADS 256
#define THRESHOLD 4
#define SECOND_STAR 1122911624u

// Minimal node structure to save memory
struct Node {
    uint8_t state, frustration, resonance;
    uint32_t neighbors[6];
    uint32_t lfsr;
    uint8_t rewiring, rewire_dir, eval_ctr;
    uint32_t old_nb;
    uint8_t pre_frust;
};

__device__ int calc_nb(int id, int d) {
    int x = id % CUBE_SIZE;
    int y = (id / CUBE_SIZE) % CUBE_SIZE;
    int z = id / (CUBE_SIZE * CUBE_SIZE);
    switch(d) {
        case 0: return ((x+1)%CUBE_SIZE) + y*CUBE_SIZE + z*CUBE_SIZE*CUBE_SIZE;
        case 1: return ((x+CUBE_SIZE-1)%CUBE_SIZE) + y*CUBE_SIZE + z*CUBE_SIZE*CUBE_SIZE;
        case 2: return x + ((y+1)%CUBE_SIZE)*CUBE_SIZE + z*CUBE_SIZE*CUBE_SIZE;
        case 3: return x + ((y+CUBE_SIZE-1)%CUBE_SIZE)*CUBE_SIZE + z*CUBE_SIZE*CUBE_SIZE;
        case 4: return x + y*CUBE_SIZE + ((z+1)%CUBE_SIZE)*CUBE_SIZE*CUBE_SIZE;
        default: return x + y*CUBE_SIZE + ((z+CUBE_SIZE-1)%CUBE_SIZE)*CUBE_SIZE*CUBE_SIZE;
    }
}

__device__ uint32_t lfsr32(uint32_t s) {
    uint32_t bit = ((s >> 0) ^ (s >> 10) ^ (s >> 30) ^ (s >> 31)) & 1;
    return (s >> 1) | (bit << 31);
}

__global__ void phase_kernel(Node* n, uint8_t* st, int phase) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_NODES) return;
    uint8_t nb = st[n[i].neighbors[phase]];
    uint8_t my = n[i].state;
    bool swap = (phase % 2 == 0) ? (my > nb) : (nb > my);
    if (swap) { n[i].state = nb; n[i].resonance = 0; }
}

__global__ void sync_st(Node* n, uint8_t* st) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NUM_NODES) st[i] = n[i].state;
}

__global__ void reset_res(Node* n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < NUM_NODES) n[i].resonance = 1;
}

__global__ void plasticity(Node* n, int* res_cnt, int* rew_cnt, int cycle) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_NODES) return;
    uint8_t res = n[i].resonance, fr = n[i].frustration;
    uint32_t rnd = n[i].lfsr;
    uint8_t rw = n[i].rewiring, rd = n[i].rewire_dir;

    if (!rw) {
        if (!res) { if (fr < 255) fr++; } else { fr >>= 1; }
        if (fr >= THRESHOLD) {
            rw = 1;
            rnd = lfsr32(rnd);
            n[i].old_nb = n[i].neighbors[rd];
            n[i].pre_frust = fr; n[i].eval_ctr = 0;
            uint32_t target = (rnd ^ (i * 0x9E3779B9) ^ (cycle * SECOND_STAR)) % NUM_NODES;
            if (target == (uint32_t)i) target = (target + 1) % NUM_NODES;
            n[i].neighbors[rd] = target;
            fr = 0; atomicAdd(rew_cnt, 1);
        }
    } else {
        n[i].eval_ctr++;
        if (!res && fr < 255) fr++;
        if (n[i].eval_ctr >= 8) {
            if (fr >= n[i].pre_frust) n[i].neighbors[rd] = n[i].old_nb;
            rd = (rd + 1) % 6; rw = 0; fr = 0;
        }
    }
    n[i].frustration = fr; n[i].rewiring = rw; n[i].rewire_dir = rd;
    n[i].lfsr = lfsr32(rnd);
    if (res) atomicAdd(res_cnt, 1);
}

__global__ void init_fabric(Node* n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_NODES) return;
    n[i].state = (i * 3) & 0xFF;
    n[i].frustration = 0; n[i].resonance = 1;
    n[i].rewiring = 0; n[i].rewire_dir = 0;
    n[i].lfsr = SECOND_STAR ^ (i * 0x9E3779B9) ^ ((i >> 8) * 0x85EBCA6B) ^ ((i >> 16) * 0xC2B2AE35);
    for (int d = 0; d < 6; d++) n[i].neighbors[d] = calc_nb(i, d);
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     256×256×256 = 16,777,216 NODES (24D Hypercube)                ║\n");
    printf("║     Second Star Constant: %u                             ║\n", SECOND_STAR);
    printf("║     Pushing toward 32D limit...                                  ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("Memory: %.1f GB free / %.1f GB total\n", free_mem/1e9, total_mem/1e9);

    size_t node_mem = NUM_NODES * sizeof(Node);
    size_t state_mem = NUM_NODES;
    printf("Required: %.1f MB nodes + %.1f MB states = %.1f MB total\n\n",
           node_mem / 1e6, state_mem / 1e6, (node_mem + state_mem) / 1e6);

    Node* d_n; uint8_t* d_st; int *d_res, *d_rew;

    cudaError_t err = cudaMalloc(&d_n, node_mem);
    if (err != cudaSuccess) {
        printf("CUDA allocation failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMalloc(&d_st, state_mem);
    cudaMalloc(&d_res, sizeof(int));
    cudaMalloc(&d_rew, sizeof(int));

    int blk = (NUM_NODES + THREADS - 1) / THREADS;
    printf("Launching with %d blocks × %d threads\n", blk, THREADS);

    init_fabric<<<blk, THREADS>>>(d_n);
    sync_st<<<blk, THREADS>>>(d_n, d_st);
    cudaDeviceSynchronize();

    printf("Threshold: %d | Running...\n\n", THRESHOLD);
    printf("CYCLE,RESONANT,PERCENT\n");
    int total_rew = 0, max_cyc = 2000;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int c = 0; c < max_cyc; c++) {
        reset_res<<<blk, THREADS>>>(d_n);
        for (int p = 0; p < 6; p++) {
            phase_kernel<<<blk, THREADS>>>(d_n, d_st, p);
            sync_st<<<blk, THREADS>>>(d_n, d_st);
        }
        cudaMemset(d_res, 0, 4); cudaMemset(d_rew, 0, 4);
        plasticity<<<blk, THREADS>>>(d_n, d_res, d_rew, c);

        int h_res, h_rew;
        cudaMemcpy(&h_res, d_res, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_rew, d_rew, 4, cudaMemcpyDeviceToHost);
        total_rew += h_rew;

        if (c % 200 == 0 || c < 10 || h_res == NUM_NODES)
            printf("%d,%d,%.1f%%\n", c, h_res, (h_res*100.0)/NUM_NODES);

        if (h_res == NUM_NODES) {
            cudaEventRecord(stop); cudaEventSynchronize(stop);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            printf("\n╔═══════════════════════════════════════════════════════════════════╗\n");
            printf("║     16,777,216 NODES CONVERGED at cycle %d                      ║\n", c);
            printf("╚═══════════════════════════════════════════════════════════════════╝\n");
            printf("    Time: %.1f sec | Rewires: %d\n", ms/1000.0, total_rew);
            printf("    Throughput: %.2f M node-cycles/sec\n",
                   (float)NUM_NODES * c / (ms / 1000.0) / 1e6);
            printf("\nTHE SCALING LAW:\n");
            printf("  64 nodes     (6D)  → 158 cycles\n");
            printf("  512 nodes    (9D)  → 113 cycles\n");
            printf("  4096 nodes   (12D) → 202 cycles\n");
            printf("  32768 nodes  (15D) → 201 cycles\n");
            printf("  262144 nodes (18D) → 158 cycles\n");
            printf("  2097152 nodes(21D) → 540 cycles\n");
            printf("  16777216 nodes(24D)→ %d cycles\n", c);
            break;
        }
    }

    int h_res;
    cudaMemcpy(&h_res, d_res, 4, cudaMemcpyDeviceToHost);
    if (h_res < NUM_NODES) {
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        printf("\nReached %d/%d (%.1f%%) after %d cycles\n",
               h_res, NUM_NODES, (h_res*100.0)/NUM_NODES, max_cyc);
        printf("Time: %.1f sec | Total rewires: %d\n", ms/1000.0, total_rew);
    }

    cudaFree(d_n); cudaFree(d_st); cudaFree(d_res); cudaFree(d_rew);
    return 0;
}
