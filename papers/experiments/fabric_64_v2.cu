// 64x64x64 with improved entropy
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define CUBE_SIZE 64
#define NUM_NODES 262144
#define THREADS 256
#define THRESHOLD 4  // More aggressive

struct Node {
    uint8_t state, frustration, resonance;
    uint32_t neighbors[6];
    uint32_t lfsr;  // 32-bit LFSR
    uint8_t rewiring, rewire_dir, eval_ctr;
    uint32_t old_nb;
    uint8_t pre_frust;
};

__device__ int calc_nb(int id, int d) {
    int x = id % CUBE_SIZE, y = (id / CUBE_SIZE) % CUBE_SIZE, z = id / (CUBE_SIZE * CUBE_SIZE);
    switch(d) {
        case 0: return ((x+1)%CUBE_SIZE) + y*CUBE_SIZE + z*CUBE_SIZE*CUBE_SIZE;
        case 1: return ((x+CUBE_SIZE-1)%CUBE_SIZE) + y*CUBE_SIZE + z*CUBE_SIZE*CUBE_SIZE;
        case 2: return x + ((y+1)%CUBE_SIZE)*CUBE_SIZE + z*CUBE_SIZE*CUBE_SIZE;
        case 3: return x + ((y+CUBE_SIZE-1)%CUBE_SIZE)*CUBE_SIZE + z*CUBE_SIZE*CUBE_SIZE;
        case 4: return x + y*CUBE_SIZE + ((z+1)%CUBE_SIZE)*CUBE_SIZE*CUBE_SIZE;
        default: return x + y*CUBE_SIZE + ((z+CUBE_SIZE-1)%CUBE_SIZE)*CUBE_SIZE*CUBE_SIZE;
    }
}

// 32-bit maximal LFSR (taps: 32, 22, 2, 1)
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
            // Mix in node ID and cycle for extra entropy
            uint32_t target = (rnd ^ (i * 0x9E3779B9) ^ (cycle * 0x85EBCA6B)) % NUM_NODES;
            if (target == i) target = (target + 1) % NUM_NODES;
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
    // Better seed mixing
    n[i].lfsr = 0xDEADBEEF ^ (i * 0x9E3779B9) ^ ((i >> 8) * 0x85EBCA6B);
    for (int d = 0; d < 6; d++) n[i].neighbors[d] = calc_nb(i, d);
}

int main() {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║     64×64×64 = 262,144 NODES (v2: 32-bit LFSR + entropy mix)      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    Node* d_n; uint8_t* d_st; int *d_res, *d_rew;
    cudaMalloc(&d_n, NUM_NODES * sizeof(Node));
    cudaMalloc(&d_st, NUM_NODES);
    cudaMalloc(&d_res, sizeof(int));
    cudaMalloc(&d_rew, sizeof(int));

    int blk = (NUM_NODES + THREADS - 1) / THREADS;
    init_fabric<<<blk, THREADS>>>(d_n);
    sync_st<<<blk, THREADS>>>(d_n, d_st);
    cudaDeviceSynchronize();

    printf("Threshold: %d (more aggressive)\n", THRESHOLD);
    printf("LFSR: 32-bit with entropy mixing\n\n");
    printf("CYCLE,RESONANT,PERCENT\n");
    int total_rew = 0, max_cyc = 1000;

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

        if (c % 50 == 0 || h_res == NUM_NODES)
            printf("%d,%d,%d%%\n", c, h_res, (h_res*100)/NUM_NODES);

        if (h_res == NUM_NODES) {
            cudaEventRecord(stop); cudaEventSynchronize(stop);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            printf("\n╔═══════════════════════════════════════════════════════════════╗\n");
            printf("║  262,144 NODES CONVERGED at cycle %d                      ║\n", c);
            printf("╚═══════════════════════════════════════════════════════════════╝\n");
            printf("    Time: %.1f ms | Rewires: %d\n\n", ms, total_rew);
            break;
        }
    }
    
    int h_res;
    cudaMemcpy(&h_res, d_res, 4, cudaMemcpyDeviceToHost);
    if (h_res < NUM_NODES) {
        printf("\nReached %d/%d (%d%%) - need more cycles or tuning\n", 
               h_res, NUM_NODES, (h_res*100)/NUM_NODES);
    }
    
    cudaFree(d_n); cudaFree(d_st);
    return 0;
}
