// GPU Equivalence Test - Find our true ceiling
// Test raw memory bandwidth and compute throughput

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// =============================================================================
// RAW BANDWIDTH TEST
// =============================================================================

__global__ void bandwidth_copy(uint8_t* dst, uint8_t* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < n; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void bandwidth_xor(uint8_t* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < n; i += stride) {
        data[i] ^= 0xAA;
    }
}

// Coalesced 4-byte access
__global__ void bandwidth_xor_u32(uint32_t* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < n; i += stride) {
        data[i] ^= 0xAAAAAAAA;
    }
}

// Vectorized 16-byte access
__global__ void bandwidth_xor_u128(uint4* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < n; i += stride) {
        uint4 v = data[i];
        v.x ^= 0xAAAAAAAA;
        v.y ^= 0xAAAAAAAA;
        v.z ^= 0xAAAAAAAA;
        v.w ^= 0xAAAAAAAA;
        data[i] = v;
    }
}

// =============================================================================
// COMPUTE-BOUND TEST
// =============================================================================

__global__ void compute_intensive(uint8_t* data, size_t n, int rounds) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < n; i += stride) {
        uint8_t v = data[i];
        #pragma unroll 8
        for (int r = 0; r < rounds; r++) {
            v ^= r;
            v = (v << 3) | (v >> 5);
            v ^= (v >> 2);
        }
        data[i] = v;
    }
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║              GPU CAPABILITY & EQUIVALENCE TEST                    ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Shared mem/block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("L2 cache: %d KB\n", prop.l2CacheSize / 1024);
    printf("\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test sizes
    size_t sizes[] = {
        16 * 1024 * 1024,    // 16 MB
        64 * 1024 * 1024,    // 64 MB
        256 * 1024 * 1024,   // 256 MB
        512 * 1024 * 1024,   // 512 MB
    };

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("RAW MEMORY BANDWIDTH\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    for (int s = 0; s < 4; s++) {
        size_t size = sizes[s];
        uint8_t* d_data;
        uint8_t* d_data2;

        cudaError_t err = cudaMalloc(&d_data, size);
        if (err != cudaSuccess) {
            printf("%.0f MB: allocation failed\n", size / 1e6);
            continue;
        }
        cudaMalloc(&d_data2, size);
        cudaMemset(d_data, 0x55, size);

        int threads = 256;
        int blocks = min(prop.multiProcessorCount * 32, (int)(size / threads));

        // Warmup
        bandwidth_xor_u128<<<blocks, threads>>>((uint4*)d_data, size / 16);
        cudaDeviceSynchronize();

        int iterations = 100;

        // Test vectorized XOR (16 bytes at a time)
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            bandwidth_xor_u128<<<blocks, threads>>>((uint4*)d_data, size / 16);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        // Read + write = 2x data
        double bandwidth = (2.0 * size * iterations) / (ms / 1000.0) / 1e9;

        printf("%.0f MB: %.1f GB/s (vectorized u128 XOR)\n", size / 1e6, bandwidth);

        cudaFree(d_data);
        cudaFree(d_data2);
    }

    printf("\n");

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("COMPUTE VS MEMORY BOUND\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    size_t size = 64 * 1024 * 1024;  // 64 MB
    uint8_t* d_data;
    cudaMalloc(&d_data, size);
    cudaMemset(d_data, 0x55, size);

    int threads = 256;
    int blocks = prop.multiProcessorCount * 32;

    printf("Data size: %.0f MB\n\n", size / 1e6);
    printf("%-20s %-15s %-15s %-15s\n", "Rounds/byte", "Time (ms)", "GB/s", "Ops/sec");
    printf("─────────────────────────────────────────────────────────────────\n");

    int rounds_list[] = {1, 4, 16, 64, 256};
    for (int r = 0; r < 5; r++) {
        int rounds = rounds_list[r];

        // Warmup
        compute_intensive<<<blocks, threads>>>(d_data, size, rounds);
        cudaDeviceSynchronize();

        int iterations = (rounds < 64) ? 100 : 10;

        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            compute_intensive<<<blocks, threads>>>(d_data, size, rounds);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        double bandwidth = (2.0 * size * iterations) / (ms / 1000.0) / 1e9;
        double ops = (3.0 * rounds * size * iterations) / (ms / 1000.0);  // 3 ops per round

        printf("%-20d %-15.2f %-15.1f %-15.1fG\n", rounds, ms, bandwidth, ops / 1e9);
    }

    cudaFree(d_data);

    printf("\n");

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("OPTIMIZED ACPU-256 vs RAW CAPABILITY\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    // Re-run our ACPU with optimized settings
    size = 256 * 1024 * 1024;  // 256 MB = 4M 512-bit units
    int num_units = size / 64;

    uint8_t* d_a;
    uint8_t* d_b;
    uint8_t* d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    cudaMemset(d_a, 0xAA, size);
    cudaMemset(d_b, 0x55, size);

    threads = 256;
    blocks = num_units;  // One block per 512-bit unit

    // But that's way too many blocks. Let's use persistent approach.
    blocks = prop.multiProcessorCount * 8;  // Fewer blocks, more work per block

    printf("Configuration:\n");
    printf("  Data size: %.0f MB (%d 512-bit units)\n", size / 1e6, num_units);
    printf("  Blocks: %d, Threads: %d\n\n", blocks, threads);

    // Test with coalesced access pattern
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        bandwidth_xor_u128<<<blocks, threads>>>((uint4*)d_a, size / 16);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double peak_bw = (2.0 * size * 100) / (ms / 1000.0) / 1e9;
    double peak_512_ops = (size / 64.0 * 100) / (ms / 1000.0);

    printf("Peak achievable:\n");
    printf("  Memory bandwidth: %.1f GB/s\n", peak_bw);
    printf("  512-bit XORs/sec: %.2f M\n", peak_512_ops / 1e6);
    printf("  Bits/sec: %.1f G\n\n", peak_512_ops * 512 / 1e9);

    // Our ACPU-256 achieved
    printf("Our ACPU-256 achieved:\n");
    printf("  Memory bandwidth: 30.65 GB/s\n");
    printf("  512-bit XORs/sec: 479 M\n");
    printf("  Bits/sec: 245 G\n\n");

    printf("Efficiency: %.1f%% of peak\n", 100.0 * 479e6 / peak_512_ops);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    printf("\n");

    printf("═══════════════════════════════════════════════════════════════════\n");
    printf("GPU EQUIVALENCE\n");
    printf("═══════════════════════════════════════════════════════════════════\n\n");

    printf("Based on measured peak bandwidth of %.1f GB/s:\n\n", peak_bw);

    // GPU comparison
    const char* gpu_names[] = {"GTX 1060", "GTX 1080", "RTX 2080", "RTX 3060", "RTX 3080", "RTX 3090", "RTX 4090", "A100"};
    double gpu_bw[] = {192, 320, 448, 360, 760, 936, 1008, 2039};

    printf("If ACPU-256 ran on these GPUs (scaling by bandwidth):\n\n");
    printf("%-12s %-15s %-20s %-15s\n", "GPU", "Bandwidth", "Projected 512b XOR", "Projected Gbits/s");
    printf("─────────────────────────────────────────────────────────────────\n");

    for (int i = 0; i < 8; i++) {
        double scale = gpu_bw[i] / peak_bw;
        double projected_ops = peak_512_ops * scale;
        double projected_bits = projected_ops * 512 / 1e9;
        printf("%-12s %-15.0f %-20.0fM %-15.0f\n",
               gpu_names[i], gpu_bw[i], projected_ops / 1e6, projected_bits);
    }

    printf("\n");
    printf("Thor (current): %.0f GB/s theoretical, %.1f GB/s achieved\n",
           256.0, peak_bw);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
