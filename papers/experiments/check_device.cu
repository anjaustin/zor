#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int count = 0;
    cudaGetDeviceCount(&count);
    printf("Device count: %d\n", count);
    
    if (count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Name: %s\n", prop.name);
        printf("Compute: %d.%d\n", prop.major, prop.minor);
        printf("Integrated: %d\n", prop.integrated);
        printf("Unified addressing: %d\n", prop.unifiedAddressing);
        printf("Managed memory: %d\n", prop.managedMemory);
        printf("Concurrent managed access: %d\n", prop.concurrentManagedAccess);
        printf("Total global mem: %zu MB\n", prop.totalGlobalMem / 1024 / 1024);
    }
    
    return 0;
}
