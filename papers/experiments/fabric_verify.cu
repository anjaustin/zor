#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define THREADS_PER_BLOCK 16

struct Data {
    uint64_t lfsr[4][8];
};

__device__ __forceinline__ void lfsr_step(uint64_t* s) {
    uint64_t fb = (s[7] >> 63) & 1;
    uint64_t carry = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t new_val = (s[i] << 1) | carry;
        carry = s[i] >> 63;
        s[i] = new_val;
    }
    if (fb) s[7] ^= (1ULL << 63);
}

__global__ void step_kernel(Data* data, int num_blocks, int steps) {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    if (block_id >= num_blocks || thread_id >= THREADS_PER_BLOCK) return;
    
    Data* d = &data[block_id * THREADS_PER_BLOCK + thread_id];
    
    uint64_t local[4][8];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            local[i][j] = d->lfsr[i][j];
        }
    }
    
    for (int step = 0; step < steps; step++) {
        for (int i = 0; i < 4; i++) {
            lfsr_step(local[i]);
        }
    }
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            d->lfsr[i][j] = local[i][j];
        }
    }
}

int main() {
    printf("FABRIC VERIFY\n\n");
    
    int num_blocks = 4096;
    int steps = 1000;
    size_t total_size = sizeof(Data) * num_blocks * THREADS_PER_BLOCK;
    
    printf("Allocating %.2f MB\n", total_size / 1e6);
    
    Data* h_data = (Data*)malloc(total_size);
    Data* d_data;
    
    // Initialize with non-zero values
    for (size_t i = 0; i < num_blocks * THREADS_PER_BLOCK; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 8; k++) {
                h_data[i].lfsr[j][k] = (i + 1) * (j + 1) * (k + 1);
            }
        }
    }
    
    cudaMalloc(&d_data, total_size);
    cudaMemcpy(d_data, h_data, total_size, cudaMemcpyHostToDevice);
    
    step_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_data, num_blocks, steps);
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    printf("Kernel: %s\n", cudaGetErrorString(err));
    
    cudaMemcpy(h_data, d_data, total_size, cudaMemcpyDeviceToHost);
    
    // Check first element
    printf("\nFirst element after %d steps:\n", steps);
    for (int j = 0; j < 4; j++) {
        printf("  lfsr[%d][0] = 0x%016lx\n", j, h_data[0].lfsr[j][0]);
    }
    
    // Check if any values are non-zero
    int non_zero = 0;
    for (int j = 0; j < 4; j++) {
        for (int k = 0; k < 8; k++) {
            if (h_data[0].lfsr[j][k] != 0) non_zero++;
        }
    }
    printf("\nNon-zero values in first element: %d/32\n", non_zero);
    printf("Status: %s\n", non_zero > 0 ? "KERNEL EXECUTED" : "KERNEL FAILED");
    
    free(h_data);
    cudaFree(d_data);
    return 0;
}
