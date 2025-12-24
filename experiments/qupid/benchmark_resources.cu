/**
 * GCS Resource Benchmark
 *
 * Measures GPU resource utilization during peak throughput.
 *
 * Build: nvcc -std=c++17 -O3 -o benchmark_resources benchmark_resources.cu
 * Run:   ./benchmark_resources
 */

#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <atomic>
#include <chrono>

// =============================================================================
// FABRIC CONFIG & SHAPES (from fabric_8x2.cu)
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

// =============================================================================
// NVML MONITORING
// =============================================================================

struct GPUMetrics {
    unsigned int gpu_util;        // GPU utilization %
    unsigned int mem_util;        // Memory utilization %
    unsigned int power_mw;        // Power in milliwatts
    unsigned int temp;            // Temperature in C
    unsigned long long mem_used;  // Memory used in bytes
    unsigned long long mem_total; // Total memory in bytes
    unsigned int sm_clock;        // SM clock in MHz
    unsigned int mem_clock;       // Memory clock in MHz
};

std::atomic<bool> monitoring(false);
std::atomic<bool> stop_monitoring(false);
GPUMetrics peak_metrics = {};
GPUMetrics avg_metrics = {};
int sample_count = 0;

void monitor_gpu(nvmlDevice_t device) {
    while (!stop_monitoring) {
        if (monitoring) {
            GPUMetrics m = {};

            nvmlUtilization_t util;
            if (nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS) {
                m.gpu_util = util.gpu;
                m.mem_util = util.memory;
            }

            nvmlDeviceGetPowerUsage(device, &m.power_mw);
            nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &m.temp);

            nvmlMemory_t mem;
            if (nvmlDeviceGetMemoryInfo(device, &mem) == NVML_SUCCESS) {
                m.mem_used = mem.used;
                m.mem_total = mem.total;
            }

            nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &m.sm_clock);
            nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &m.mem_clock);

            // Update peak
            if (m.gpu_util > peak_metrics.gpu_util) peak_metrics.gpu_util = m.gpu_util;
            if (m.mem_util > peak_metrics.mem_util) peak_metrics.mem_util = m.mem_util;
            if (m.power_mw > peak_metrics.power_mw) peak_metrics.power_mw = m.power_mw;
            if (m.temp > peak_metrics.temp) peak_metrics.temp = m.temp;
            if (m.mem_used > peak_metrics.mem_used) peak_metrics.mem_used = m.mem_used;
            peak_metrics.mem_total = m.mem_total;
            if (m.sm_clock > peak_metrics.sm_clock) peak_metrics.sm_clock = m.sm_clock;
            if (m.mem_clock > peak_metrics.mem_clock) peak_metrics.mem_clock = m.mem_clock;

            // Update average
            avg_metrics.gpu_util += m.gpu_util;
            avg_metrics.mem_util += m.mem_util;
            avg_metrics.power_mw += m.power_mw;
            avg_metrics.temp += m.temp;
            sample_count++;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// =============================================================================
// MAIN BENCHMARK
// =============================================================================

int main() {
    printf("=======================================================================\n");
    printf("GCS RESOURCE BENCHMARK\n");
    printf("=======================================================================\n\n");

    // Initialize NVML
    nvmlReturn_t nvml_result = nvmlInit();
    if (nvml_result != NVML_SUCCESS) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(nvml_result));
        printf("Running without GPU monitoring.\n\n");
    }

    nvmlDevice_t device;
    char name[256];
    if (nvml_result == NVML_SUCCESS) {
        nvmlDeviceGetHandleByIndex(0, &device);
        nvmlDeviceGetName(device, name, sizeof(name));
        printf("Device: %s\n", name);

        // Get initial metrics
        nvmlMemory_t mem;
        nvmlDeviceGetMemoryInfo(device, &mem);
        printf("Memory: %.1f GB / %.1f GB\n",
               mem.used / 1e9, mem.total / 1e9);

        unsigned int power;
        nvmlDeviceGetPowerUsage(device, &power);
        printf("Idle Power: %.1f W\n", power / 1000.0);

        unsigned int temp;
        nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
        printf("Idle Temp: %d C\n\n", temp);
    }

    // Setup fabric config (sorter)
    FabricConfig config = {};
    config.layer1_shapes[0] = 8;  // MIN
    config.layer1_shapes[1] = 9;  // MAX
    config.layer1_shapes[2] = 8;
    config.layer1_shapes[3] = 9;
    for (int i = 4; i < 8; i++) config.layer1_shapes[i] = 15;

    config.layer1_input_a[0] = 0; config.layer1_input_b[0] = 1;
    config.layer1_input_a[1] = 0; config.layer1_input_b[1] = 1;
    config.layer1_input_a[2] = 2; config.layer1_input_b[2] = 3;
    config.layer1_input_a[3] = 2; config.layer1_input_b[3] = 3;

    config.layer2_shapes[0] = 8;  // MIN
    config.layer2_shapes[1] = 8;
    config.layer2_shapes[2] = 9;  // MAX
    config.layer2_shapes[3] = 9;
    for (int i = 4; i < 8; i++) config.layer2_shapes[i] = 15;

    config.layer2_input_a[0] = 0; config.layer2_input_b[0] = 2;
    config.layer2_input_a[1] = 1; config.layer2_input_b[1] = 3;
    config.layer2_input_a[2] = 0; config.layer2_input_b[2] = 2;
    config.layer2_input_a[3] = 1; config.layer2_input_b[3] = 3;

    // Allocate
    const uint32_t N = 16 * 1024 * 1024;  // 16M samples
    const int iterations = 500;

    FabricConfig* d_config;
    uint8_t* d_inputs;
    uint8_t* d_outputs;

    cudaMalloc(&d_config, sizeof(FabricConfig));
    cudaMalloc(&d_inputs, N * 8);
    cudaMalloc(&d_outputs, N * 8);

    cudaMemcpy(d_config, &config, sizeof(FabricConfig), cudaMemcpyHostToDevice);

    // Fill with random data
    uint8_t* h_inputs = new uint8_t[N * 8];
    for (uint32_t i = 0; i < N * 8; i++) {
        h_inputs[i] = rand() & 0xFF;
    }
    cudaMemcpy(d_inputs, h_inputs, N * 8, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    printf("Benchmark Configuration:\n");
    printf("  Samples per iteration: %d (%.1f M)\n", N, N / 1e6);
    printf("  Iterations: %d\n", iterations);
    printf("  Total samples: %.1f B\n", (double)N * iterations / 1e9);
    printf("  ALU ops per sample: 16\n");
    printf("  Total ALU ops: %.1f T\n", (double)N * iterations * 16 / 1e12);
    printf("  Grid: %d blocks × %d threads\n\n", grid_size, block_size);

    // Warm up
    printf("Warming up...\n");
    for (int i = 0; i < 50; i++) {
        fabric_8x2_kernel<<<grid_size, block_size>>>(d_inputs, d_outputs, d_config, N);
    }
    cudaDeviceSynchronize();

    // Start monitoring thread
    std::thread monitor_thread;
    if (nvml_result == NVML_SUCCESS) {
        monitor_thread = std::thread(monitor_gpu, device);
    }

    printf("Running benchmark with monitoring...\n\n");

    // Start monitoring
    monitoring = true;

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        fabric_8x2_kernel<<<grid_size, block_size>>>(d_inputs, d_outputs, d_config, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Stop monitoring
    monitoring = false;
    stop_monitoring = true;
    if (monitor_thread.joinable()) {
        monitor_thread.join();
    }

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double samples_per_sec = (double)N * iterations / (ms / 1000.0);
    double ops_per_sec = samples_per_sec * 16;
    double bytes_read = (double)N * iterations * 8;  // 8 bytes per sample input
    double bytes_written = (double)N * iterations * 8;
    double bandwidth_read = bytes_read / (ms / 1000.0) / 1e9;
    double bandwidth_write = bytes_written / (ms / 1000.0) / 1e9;

    printf("=======================================================================\n");
    printf("PERFORMANCE RESULTS\n");
    printf("=======================================================================\n\n");

    printf("Throughput:\n");
    printf("  Time: %.2f ms\n", ms);
    printf("  Samples/sec: %.2f B\n", samples_per_sec / 1e9);
    printf("  ALU ops/sec: %.2f B\n", ops_per_sec / 1e9);
    printf("\n");

    printf("Memory Bandwidth:\n");
    printf("  Read:  %.1f GB/s\n", bandwidth_read);
    printf("  Write: %.1f GB/s\n", bandwidth_write);
    printf("  Total: %.1f GB/s\n", bandwidth_read + bandwidth_write);
    printf("\n");

    if (nvml_result == NVML_SUCCESS && sample_count > 0) {
        printf("=======================================================================\n");
        printf("GPU RESOURCE UTILIZATION\n");
        printf("=======================================================================\n\n");

        printf("Peak Metrics:\n");
        printf("  GPU Utilization: %u%%\n", peak_metrics.gpu_util);
        printf("  Memory Utilization: %u%%\n", peak_metrics.mem_util);
        printf("  Power: %.1f W\n", peak_metrics.power_mw / 1000.0);
        printf("  Temperature: %u C\n", peak_metrics.temp);
        printf("  Memory Used: %.2f GB / %.2f GB\n",
               peak_metrics.mem_used / 1e9, peak_metrics.mem_total / 1e9);
        printf("  SM Clock: %u MHz\n", peak_metrics.sm_clock);
        printf("  Memory Clock: %u MHz\n", peak_metrics.mem_clock);
        printf("\n");

        printf("Average Metrics (over %d samples):\n", sample_count);
        printf("  GPU Utilization: %.1f%%\n", (double)avg_metrics.gpu_util / sample_count);
        printf("  Memory Utilization: %.1f%%\n", (double)avg_metrics.mem_util / sample_count);
        printf("  Power: %.1f W\n", (double)avg_metrics.power_mw / sample_count / 1000.0);
        printf("  Temperature: %.1f C\n", (double)avg_metrics.temp / sample_count);
        printf("\n");

        // Efficiency calculations
        double ops_per_watt = ops_per_sec / (peak_metrics.power_mw / 1000.0);
        printf("Efficiency:\n");
        printf("  ALU ops per Watt: %.2f B ops/W\n", ops_per_watt / 1e9);
        printf("  ALU ops per Joule: %.2f B ops/J\n", ops_per_watt / 1e9);
        printf("\n");
    }

    // Cleanup
    delete[] h_inputs;
    cudaFree(d_config);
    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (nvml_result == NVML_SUCCESS) {
        nvmlShutdown();
    }

    printf("=======================================================================\n");
    printf("SUMMARY\n");
    printf("=======================================================================\n\n");
    printf("The 8x2 fabric running at %.1f B ops/sec uses:\n", ops_per_sec / 1e9);
    if (nvml_result == NVML_SUCCESS && sample_count > 0) {
        printf("  - %.1f%% GPU utilization\n", (double)avg_metrics.gpu_util / sample_count);
        printf("  - %.1f W power\n", (double)avg_metrics.power_mw / sample_count / 1000.0);
        printf("  - %.1f GB/s memory bandwidth\n", bandwidth_read + bandwidth_write);
    }
    printf("\n");

    return 0;
}
