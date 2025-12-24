#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// Simple verification without timing

__global__ void add_kernel(uint64_t* a, uint64_t* b, uint64_t* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    printf("CUDA Verification (No Timing)\n");
    printf("========================================\n\n");
    
    int n = 1000;
    size_t size = n * sizeof(uint64_t);
    
    uint64_t* h_a = (uint64_t*)malloc(size);
    uint64_t* h_b = (uint64_t*)malloc(size);
    uint64_t* h_c = (uint64_t*)malloc(size);
    
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    uint64_t *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    add_kernel<<<(n+255)/256, 256>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    printf("Kernel launch: %s\n", cudaGetErrorString(err));
    
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            errors++;
            if (errors <= 5) {
                printf("Error at %d: got %lu, expected %lu\n", i, h_c[i], h_a[i] + h_b[i]);
            }
        }
    }
    
    printf("\nResults: %d errors out of %d\n", errors, n);
    printf("Status: %s\n", errors == 0 ? "SUCCESS" : "FAIL");
    
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
