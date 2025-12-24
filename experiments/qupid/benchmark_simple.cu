/**
 * GCS Simple Resource Benchmark
 *
 * Uses CUDA profiling and /sys thermal readings on Jetson.
 *
 * Build: nvcc -std=c++17 -O3 -o benchmark_simple benchmark_simple.cu
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

// =============================================================================
// FABRIC CONFIG & SHAPES
// =============================================================================

struct FabricConfig {
    uint8_t layer1_shapes[8];
    uint8_t layer1_input_a[8];
    uint8_t layer1_input_b[8];
    uint8_t layer2_shapes[8];
    uint8_t layer2_input_a[8];
    uint8_t layer2_input_b[8];
};

__device__ __forceinline__ uint8_t execute_shape(uint8_t shape_id, uint8_t a, uint8_t b) {
    switch (shape_id & 0x0F) {
        case 0:  return a + b;
        case 1:  return a - b;
        case 2:  return a & b;
        case 3:  return a | b;
        case 4:  return a ^ b;
        case 5:  return a << (b & 7);
        case 6:  return a >> (b & 7);
        case 7:  return a * b;
        case 8:  return a < b ? a : b;
        case 9:  return a > b ? a : b;
        case 10: return (a + b) >> 1;
        case 11: return a > b ? a - b : b - a;
        case 12: return a;
        case 13: return b;
        case 14: return ~a;
        case 15: return 0;
        default: return a;
    }
}

__global__ void fabric_8x2_kernel(
    const uint8_t* __restrict__ inputs,
    uint8_t* __restrict__ outputs,
    const FabricConfig* __restrict__ config,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint8_t in[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        in[i] = inputs[idx * 8 + i];
    }

    uint8_t l1[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t a = in[config->layer1_input_a[i] & 7];
        uint8_t b = in[config->layer1_input_b[i] & 7];
        l1[i] = execute_shape(config->layer1_shapes[i], a, b);
    }

    uint8_t l2[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t a = l1[config->layer2_input_a[i] & 7];
        uint8_t b = l1[config->layer2_input_b[i] & 7];
        l2[i] = execute_shape(config->layer2_shapes[i], a, b);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        outputs[idx * 8 + i] = l2[i];
    }
}

// Read temperature from /sys
float read_gpu_temp() {
    std::ifstream f("/sys/class/thermal/thermal_zone1/temp");
    if (f) {
        int temp_mC;
        f >> temp_mC;
        return temp_mC / 1000.0f;
    }
    return -1;
}

float read_cpu_temp() {
    std::ifstream f("/sys/class/thermal/thermal_zone2/temp");
    if (f) {
        int temp_mC;
        f >> temp_mC;
        return temp_mC / 1000.0f;
    }
    return -1;
}

float read_tj_temp() {
    std::ifstream f("/sys/class/thermal/thermal_zone0/temp");
    if (f) {
        int temp_mC;
        f >> temp_mC;
        return temp_mC / 1000.0f;
    }
    return -1;
}

int main() {
    printf("=======================================================================\n");
    printf("GCS RESOURCE BENCHMARK (Jetson)\n");
    printf("=======================================================================\n\n");

    // Device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Device: %s\n", prop.name);
    printf("Compute: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max threads/block: %d\n", prop.maxThreadsPerBlock);
    printf("Shared memory/block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("L2 cache: %d KB\n", prop.l2CacheSize / 1024);
    printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
    // Thor uses unified memory, estimate bandwidth from bus width
    double peak_bw = 256.0;  // GB/s estimate for Thor unified memory
    printf("Estimated peak bandwidth: %.1f GB/s (unified memory)\n", peak_bw);
    printf("\n");

    // Initial temps
    printf("Initial Temperatures:\n");
    printf("  GPU: %.1f C\n", read_gpu_temp());
    printf("  CPU: %.1f C\n", read_cpu_temp());
    printf("  TJ:  %.1f C\n", read_tj_temp());
    printf("\n");

    // Memory info
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("Memory: %.1f GB free / %.1f GB total\n\n",
           free_mem / 1e9, total_mem / 1e9);

    // Setup
    FabricConfig config = {};
    config.layer1_shapes[0] = 8;  config.layer1_shapes[1] = 9;
    config.layer1_shapes[2] = 8;  config.layer1_shapes[3] = 9;
    for (int i = 4; i < 8; i++) config.layer1_shapes[i] = 15;
    config.layer1_input_a[0] = 0; config.layer1_input_b[0] = 1;
    config.layer1_input_a[1] = 0; config.layer1_input_b[1] = 1;
    config.layer1_input_a[2] = 2; config.layer1_input_b[2] = 3;
    config.layer1_input_a[3] = 2; config.layer1_input_b[3] = 3;
    config.layer2_shapes[0] = 8;  config.layer2_shapes[1] = 8;
    config.layer2_shapes[2] = 9;  config.layer2_shapes[3] = 9;
    for (int i = 4; i < 8; i++) config.layer2_shapes[i] = 15;
    config.layer2_input_a[0] = 0; config.layer2_input_b[0] = 2;
    config.layer2_input_a[1] = 1; config.layer2_input_b[1] = 3;
    config.layer2_input_a[2] = 0; config.layer2_input_b[2] = 2;
    config.layer2_input_a[3] = 1; config.layer2_input_b[3] = 3;

    const uint32_t N = 16 * 1024 * 1024;
    const int iterations = 500;

    FabricConfig* d_config;
    uint8_t* d_inputs;
    uint8_t* d_outputs;

    cudaMalloc(&d_config, sizeof(FabricConfig));
    cudaMalloc(&d_inputs, N * 8);
    cudaMalloc(&d_outputs, N * 8);
    cudaMemcpy(d_config, &config, sizeof(FabricConfig), cudaMemcpyHostToDevice);

    // Fill with data
    uint8_t* h_inputs = new uint8_t[N * 8];
    for (uint32_t i = 0; i < N * 8; i++) h_inputs[i] = rand() & 0xFF;
    cudaMemcpy(d_inputs, h_inputs, N * 8, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    // Get occupancy info
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, fabric_8x2_kernel, block_size, 0);

    int active_warps = max_blocks_per_sm * (block_size / 32);
    int max_warps = prop.maxThreadsPerMultiProcessor / 32;
    float occupancy = (float)active_warps / max_warps * 100;

    printf("Kernel Configuration:\n");
    printf("  Grid: %d blocks\n", grid_size);
    printf("  Block: %d threads\n", block_size);
    printf("  Blocks per SM: %d\n", max_blocks_per_sm);
    printf("  Active warps per SM: %d / %d\n", active_warps, max_warps);
    printf("  Theoretical occupancy: %.1f%%\n", occupancy);
    printf("\n");

    // Warm up
    printf("Warming up...\n");
    for (int i = 0; i < 50; i++) {
        fabric_8x2_kernel<<<grid_size, block_size>>>(d_inputs, d_outputs, d_config, N);
    }
    cudaDeviceSynchronize();

    printf("Post-warmup temps:\n");
    printf("  GPU: %.1f C\n", read_gpu_temp());
    printf("  CPU: %.1f C\n", read_cpu_temp());
    printf("  TJ:  %.1f C\n", read_tj_temp());
    printf("\n");

    // Benchmark
    printf("Running %d iterations of %dM samples...\n\n", iterations, N / (1024*1024));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float peak_gpu_temp = read_gpu_temp();

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        fabric_8x2_kernel<<<grid_size, block_size>>>(d_inputs, d_outputs, d_config, N);

        // Sample temp every 50 iterations
        if (i % 50 == 0) {
            cudaDeviceSynchronize();
            float t = read_gpu_temp();
            if (t > peak_gpu_temp) peak_gpu_temp = t;
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float final_gpu_temp = read_gpu_temp();
    float final_cpu_temp = read_cpu_temp();
    float final_tj_temp = read_tj_temp();

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double total_samples = (double)N * iterations;
    double samples_per_sec = total_samples / (ms / 1000.0);
    double ops_per_sec = samples_per_sec * 16;
    double bytes_per_sec = samples_per_sec * 16;  // 8 read + 8 write per sample

    printf("=======================================================================\n");
    printf("RESULTS\n");
    printf("=======================================================================\n\n");

    printf("Performance:\n");
    printf("  Total time: %.2f ms\n", ms);
    printf("  Samples processed: %.2f B\n", total_samples / 1e9);
    printf("  Throughput: %.2f B samples/sec\n", samples_per_sec / 1e9);
    printf("  ALU ops/sec: %.2f B\n", ops_per_sec / 1e9);
    printf("\n");

    printf("Memory Bandwidth (achieved):\n");
    printf("  Read:  %.1f GB/s\n", bytes_per_sec / 1e9 / 2);
    printf("  Write: %.1f GB/s\n", bytes_per_sec / 1e9 / 2);
    printf("  Total: %.1f GB/s\n", bytes_per_sec / 1e9);
    printf("\n");

    double peak_bw_final = 256.0;  // GB/s estimate for Thor
    printf("Memory Efficiency:\n");
    printf("  Achieved / Peak: %.1f / %.1f GB/s = %.1f%%\n",
           bytes_per_sec / 1e9, peak_bw_final, bytes_per_sec / 1e9 / peak_bw_final * 100);
    printf("\n");

    printf("Thermal:\n");
    printf("  GPU temp (peak):  %.1f C\n", peak_gpu_temp);
    printf("  GPU temp (final): %.1f C\n", final_gpu_temp);
    printf("  CPU temp (final): %.1f C\n", final_cpu_temp);
    printf("  TJ temp (final):  %.1f C\n", final_tj_temp);
    printf("\n");

    printf("=======================================================================\n");
    printf("SUMMARY\n");
    printf("=======================================================================\n\n");

    printf("The 8x2 GCS fabric achieves:\n");
    printf("  - %.1f billion ALU ops/sec\n", ops_per_sec / 1e9);
    printf("  - %.1f%% theoretical occupancy\n", occupancy);
    printf("  - %.1f%% memory bandwidth utilization\n", bytes_per_sec / 1e9 / peak_bw_final * 100);
    printf("  - Peak GPU temp: %.1f C\n", peak_gpu_temp);
    printf("\n");

    // Cleanup
    delete[] h_inputs;
    cudaFree(d_config);
    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
