#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// Use unified memory instead of separate device memory

__global__ void add_kernel(uint64_t* c, uint64_t* a, uint64_t* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    printf("Testing Unified Memory\n");
    printf("======================\n\n");
    
    int n = 1000;
    size_t size = n * sizeof(uint64_t);
    
    uint64_t *a, *b, *c;
    
    // Use cudaMallocManaged for unified memory
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);
    
    for (int i = 0; i < n; i++) {
        a[i] = i;
        b[i] = i * 2;
        c[i] = 0;
    }
    
    add_kernel<<<(n+255)/256, 256>>>(c, a, b, n);
    cudaDeviceSynchronize();
    
    cudaError_t err = cudaGetLastError();
    printf("Kernel: %s\n", cudaGetErrorString(err));
    
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (c[i] != a[i] + b[i]) {
            errors++;
            if (errors <= 3) {
                printf("Error at %d: got %lu, expected %lu\n", i, c[i], a[i] + b[i]);
            }
        }
    }
    
    printf("\nResults: %d errors out of %d\n", errors, n);
    printf("Status: %s\n", errors == 0 ? "SUCCESS" : "FAIL");
    
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}
