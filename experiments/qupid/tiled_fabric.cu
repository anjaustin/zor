/**
 * Tiled Fabric - Hollywood Squares Virtual RAM PoC
 *
 * Demonstrates keeping data in shared memory across multiple fabric stages.
 * Tests the memory access reduction at various stage depths (RAID ratios).
 *
 * Build: nvcc -std=c++17 -O3 -o tiled_fabric tiled_fabric.cu
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

// =============================================================================
// FROZEN SHAPES
// =============================================================================

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

// =============================================================================
// FABRIC CONFIGURATIONS
// =============================================================================

struct FabricConfig {
    uint8_t shapes[8];
    uint8_t input_a[8];
    uint8_t input_b[8];
};

struct TiledPipelineConfig {
    int num_stages;
    FabricConfig stages[256];
};

// =============================================================================
// NAIVE KERNEL - Global memory between each stage
// =============================================================================

__global__ void naive_fabric_stage(
    const uint8_t* __restrict__ inputs,
    uint8_t* __restrict__ outputs,
    const FabricConfig* __restrict__ config,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint8_t in[8], out[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) in[i] = inputs[idx * 8 + i];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i] = execute_shape(config->shapes[i],
                               in[config->input_a[i] & 7],
                               in[config->input_b[i] & 7]);
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) outputs[idx * 8 + i] = out[i];
}

// =============================================================================
// TILED KERNEL - All stages in shared memory
// =============================================================================

#define TILES_PER_BLOCK 256
#define TILE_SIZE 8

__global__ void tiled_fabric_pipeline(
    const uint8_t* __restrict__ inputs,
    uint8_t* __restrict__ outputs,
    const TiledPipelineConfig* __restrict__ config,
    uint32_t n
) {
    __shared__ uint8_t tile_data[TILES_PER_BLOCK][TILE_SIZE];
    __shared__ uint8_t tile_temp[TILES_PER_BLOCK][TILE_SIZE];

    uint32_t block_start = blockIdx.x * TILES_PER_BLOCK;
    uint32_t local_idx = threadIdx.x;

    // Load from global memory (ONCE)
    if (block_start + local_idx < n) {
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            tile_data[local_idx][i] = inputs[(block_start + local_idx) * TILE_SIZE + i];
        }
    }
    __syncthreads();

    // Execute all stages IN SHARED MEMORY
    for (int stage = 0; stage < config->num_stages; stage++) {
        const FabricConfig* sc = &config->stages[stage];

        if (block_start + local_idx < n) {
            uint8_t* in = tile_data[local_idx];
            uint8_t* out = tile_temp[local_idx];

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                out[i] = execute_shape(sc->shapes[i],
                                       in[sc->input_a[i] & 7],
                                       in[sc->input_b[i] & 7]);
            }
        }
        __syncthreads();

        // Swap buffers
        if (block_start + local_idx < n) {
            #pragma unroll
            for (int i = 0; i < TILE_SIZE; i++) {
                tile_data[local_idx][i] = tile_temp[local_idx][i];
            }
        }
        __syncthreads();
    }

    // Store to global memory (ONCE)
    if (block_start + local_idx < n) {
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            outputs[(block_start + local_idx) * TILE_SIZE + i] = tile_data[local_idx][i];
        }
    }
}

// =============================================================================
// BENCHMARK FUNCTIONS
// =============================================================================

double benchmark_naive(FabricConfig* stages, int num_stages,
                       uint8_t* d_buf1, uint8_t* d_buf2,
                       FabricConfig* d_config,
                       uint32_t n, int iterations) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm up
    for (int stage = 0; stage < num_stages; stage++) {
        cudaMemcpy(d_config, &stages[stage], sizeof(FabricConfig), cudaMemcpyHostToDevice);
        naive_fabric_stage<<<grid_size, block_size>>>(
            (stage % 2 == 0) ? d_buf1 : d_buf2,
            (stage % 2 == 0) ? d_buf2 : d_buf1,
            d_config, n);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int iter = 0; iter < iterations; iter++) {
        for (int stage = 0; stage < num_stages; stage++) {
            cudaMemcpy(d_config, &stages[stage], sizeof(FabricConfig), cudaMemcpyHostToDevice);
            naive_fabric_stage<<<grid_size, block_size>>>(
                (stage % 2 == 0) ? d_buf1 : d_buf2,
                (stage % 2 == 0) ? d_buf2 : d_buf1,
                d_config, n);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (double)n * num_stages * 8 * iterations / (ms / 1000.0) / 1e9;  // B ops/sec
}

double benchmark_tiled(TiledPipelineConfig* config,
                       uint8_t* d_inputs, uint8_t* d_outputs,
                       TiledPipelineConfig* d_config,
                       uint32_t n, int iterations) {
    int grid_size = (n + TILES_PER_BLOCK - 1) / TILES_PER_BLOCK;

    cudaMemcpy(d_config, config, sizeof(TiledPipelineConfig), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm up
    for (int i = 0; i < 10; i++) {
        tiled_fabric_pipeline<<<grid_size, TILES_PER_BLOCK>>>(d_inputs, d_outputs, d_config, n);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int iter = 0; iter < iterations; iter++) {
        tiled_fabric_pipeline<<<grid_size, TILES_PER_BLOCK>>>(d_inputs, d_outputs, d_config, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (double)n * config->num_stages * 8 * iterations / (ms / 1000.0) / 1e9;
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    printf("=======================================================================\n");
    printf("HOLLYWOOD SQUARES VIRTUAL RAM - RAID FABRIC TEST\n");
    printf("=======================================================================\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (Shared: %zu KB/block)\n\n", prop.name, prop.sharedMemPerBlock / 1024);

    // Allocate once
    const uint32_t N = 4 * 1024 * 1024;
    const int iterations = 50;

    uint8_t* d_buf1;
    uint8_t* d_buf2;
    FabricConfig* d_config;
    TiledPipelineConfig* d_tiled_config;

    cudaMalloc(&d_buf1, N * 8);
    cudaMalloc(&d_buf2, N * 8);
    cudaMalloc(&d_config, sizeof(FabricConfig));
    cudaMalloc(&d_tiled_config, sizeof(TiledPipelineConfig));

    uint8_t* h_data = new uint8_t[N * 8];
    for (uint32_t i = 0; i < N * 8; i++) h_data[i] = rand() & 0xFF;

    // Test various RAID ratios (stages:1 memory access ratio)
    printf("| Stages | RAID Ratio | Naive (B ops/s) | Tiled (B ops/s) | Speedup |\n");
    printf("|--------|------------|-----------------|-----------------|--------|\n");

    int stage_counts[] = {4, 8, 16, 32, 64, 128, 256};
    int num_tests = sizeof(stage_counts) / sizeof(stage_counts[0]);

    for (int t = 0; t < num_tests; t++) {
        int num_stages = stage_counts[t];

        // Create stage configs
        FabricConfig stages[256];
        TiledPipelineConfig tiled_config;
        tiled_config.num_stages = num_stages;

        for (int s = 0; s < num_stages; s++) {
            // Alternating XOR/ADD transforms
            for (int i = 0; i < 8; i++) {
                stages[s].shapes[i] = (s % 2 == 0) ? 4 : 0;  // XOR or ADD
                stages[s].input_a[i] = i;
                stages[s].input_b[i] = (i + 1) % 8;
            }
            tiled_config.stages[s] = stages[s];
        }

        // Reset input data
        cudaMemcpy(d_buf1, h_data, N * 8, cudaMemcpyHostToDevice);

        // Benchmark naive
        double naive_ops = benchmark_naive(stages, num_stages, d_buf1, d_buf2, d_config, N, iterations);

        // Reset input data
        cudaMemcpy(d_buf1, h_data, N * 8, cudaMemcpyHostToDevice);

        // Benchmark tiled
        double tiled_ops = benchmark_tiled(&tiled_config, d_buf1, d_buf2, d_tiled_config, N, iterations);

        double speedup = tiled_ops / naive_ops;
        int raid_ratio = num_stages;  // N stages : 1 memory access

        printf("| %6d | %5d:1    | %15.1f | %15.1f | %6.2fx |\n",
               num_stages, raid_ratio, naive_ops, tiled_ops, speedup);
    }

    printf("\n");
    printf("=======================================================================\n");
    printf("ANALYSIS\n");
    printf("=======================================================================\n\n");

    printf("RAID FABRIC Principle:\n");
    printf("  - Naive: 2N global memory accesses for N stages\n");
    printf("  - Tiled: 2 global memory accesses regardless of N stages\n");
    printf("  - RAID N:1 means N compute stages per 1 memory access\n\n");

    printf("At high stage counts, tiled wins because:\n");
    printf("  - Compute happens in shared memory (~20 cycle latency)\n");
    printf("  - Global memory avoided (~400 cycle latency)\n");
    printf("  - The Hollywood Squares router only touches RAM at boundaries\n\n");

    // Cleanup
    delete[] h_data;
    cudaFree(d_buf1);
    cudaFree(d_buf2);
    cudaFree(d_config);
    cudaFree(d_tiled_config);

    return 0;
}
