/**
 * Precise CUDA timing test
 *
 * Measures exactly where time is spent.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <chrono>

__global__ void dummy_kernel(uint32_t* data, uint32_t n) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 1;
    }
}

int main() {
    const uint32_t N = 1024 * 1024;
    const int iterations = 1000;

    // Allocate
    uint32_t* d_data;
    cudaMallocManaged(&d_data, N * sizeof(uint32_t));

    // Create stream and graph
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    int grid = (N + 255) / 256;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    dummy_kernel<<<grid, 256, 0, stream>>>(d_data, N);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // Warm up
    for (int i = 0; i < 10; i++) {
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
    }

    printf("=======================================================================\n");
    printf("CUDA TIMING BREAKDOWN\n");
    printf("=======================================================================\n\n");

    // Measure graph launch only (no sync)
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        cudaGraphLaunch(graphExec, stream);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto launch_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    cudaStreamSynchronize(stream);  // Wait for all to complete

    double launch_us = (double)launch_time / iterations / 1000.0;
    printf("[Graph Launch Only]\n");
    printf("  Time per launch: %.2f μs\n\n", launch_us);

    // Measure sync only
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        cudaStreamSynchronize(stream);
    }
    end = std::chrono::high_resolution_clock::now();
    auto sync_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    double sync_us = (double)sync_time / iterations / 1000.0;
    printf("[Stream Sync Only (no work)]\n");
    printf("  Time per sync: %.2f μs\n\n", sync_us);

    // Measure launch + sync (the real path)
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);
    }
    end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    double total_us = (double)total_time / iterations / 1000.0;
    printf("[Graph Launch + Sync]\n");
    printf("  Time per iteration: %.2f μs\n", total_us);
    printf("  (Launch: %.2f μs + Kernel: %.2f μs)\n\n", launch_us, total_us - launch_us);

    // Measure raw kernel (no graph)
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        dummy_kernel<<<grid, 256>>>(d_data, N);
        cudaDeviceSynchronize();
    }
    end = std::chrono::high_resolution_clock::now();
    auto raw_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    double raw_us = (double)raw_time / iterations / 1000.0;
    printf("[Raw Kernel + DeviceSync]\n");
    printf("  Time per iteration: %.2f μs\n\n", raw_us);

    printf("=======================================================================\n");
    printf("SUMMARY\n");
    printf("=======================================================================\n");
    printf("\n");
    printf("  Graph launch overhead: %.2f μs\n", launch_us);
    printf("  Empty sync overhead:   %.2f μs\n", sync_us);
    printf("  Full graph path:       %.2f μs\n", total_us);
    printf("  Raw kernel path:       %.2f μs\n", raw_us);
    printf("\n");
    printf("  Graph vs Raw: %.2fx faster\n", raw_us / total_us);
    printf("\n");

    // Cleanup
    cudaFree(d_data);
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    return 0;
}
