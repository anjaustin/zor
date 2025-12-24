#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>

#define THREADS 16
struct Data { uint64_t v[32]; };

__global__ void add_kernel(Data* d, int n, int steps) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    
    uint64_t local[32];
    for (int i = 0; i < 32; i++) local[i] = d[id].v[i];
    
    for (int s = 0; s < steps; s++)
        for (int i = 0; i < 32; i++)
            local[i] = local[i] + local[(i+1) % 32];
    
    for (int i = 0; i < 32; i++) d[id].v[i] = local[i];
}

int main() {
    printf("CUDA with cudaMallocManaged (Thor unified memory)\n\n");
    
    int n = 65536;
    int steps = 1000;
    size_t size = sizeof(Data) * n;
    
    Data* data = NULL;
    cudaError_t err = cudaMallocManaged(&data, size);
    printf("cudaMallocManaged: %s (ptr=%p)\n", cudaGetErrorString(err), data);
    
    if (data == NULL) {
        printf("Allocation failed!\n");
        return 1;
    }
    
    for (int i = 0; i < n; i++)
        for (int j = 0; j < 32; j++)
            data[i].v[j] = i + j;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    
    add_kernel<<<n/THREADS, THREADS>>>(data, n, steps);
    cudaDeviceSynchronize();
    
    clock_gettime(CLOCK_MONOTONIC, &t1);
    
    err = cudaGetLastError();
    printf("Kernel: %s\n", cudaGetErrorString(err));
    
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    long long total = (long long)n * 32 * steps;
    
    printf("\nTime: %.2f ms\n", ms);
    printf("Ops: %.2f billion\n", total / 1e9);
    printf("Throughput: %.2f billion/sec\n", total / (ms / 1000.0) / 1e9);
    printf("First result: %lu\n", data[0].v[0]);

    cudaFree(data);
    return 0;
}
