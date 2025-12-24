#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// Match fabric's thread count exactly: 16 threads per block

#define THREADS_PER_BLOCK 16

struct GeomData {
    uint64_t a[4];
    uint64_t b[4];
    uint64_t result[4];
};

__global__ void geometry_kernel(GeomData* data, int num_blocks, int steps) {
    int block_id = blockIdx.x;
    int thread_id = threadIdx.x;
    
    if (block_id >= num_blocks || thread_id >= THREADS_PER_BLOCK) return;
    
    GeomData* d = &data[block_id * THREADS_PER_BLOCK + thread_id];
    
    uint64_t local_a[4], local_b[4], local_r[4];
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        local_a[i] = d->a[i];
        local_b[i] = d->b[i];
    }
    
    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < 4; i++) {
            local_r[i] = local_a[i] + local_b[i];
        }
    }
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        d->result[i] = local_r[i];
    }
}

int main() {
    printf("GEOMETRY TEST (16 threads/block like fabric)\n");
    
    int num_blocks = 4096;
    int steps = 1000;
    
    size_t total_size = sizeof(GeomData) * num_blocks * THREADS_PER_BLOCK;
    printf("Allocating %.2f MB\n", total_size / 1e6);
    
    GeomData* d_data;
    cudaMalloc(&d_data, total_size);
    cudaMemset(d_data, 0x01, total_size);
    
    geometry_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(d_data, num_blocks, steps);
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Success!\n");
    }
    
    cudaFree(d_data);
    return 0;
}
