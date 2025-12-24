/**
 * GCS Pipeline Test
 *
 * Tests the YAML -> C++ -> GPU pipeline.
 *
 * Build:
 *   nvcc -std=c++17 -O3 -o test_gcs test_gcs.cu
 *
 * Run:
 *   ./test_gcs
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>

// =============================================================================
// FABRIC CONFIG (from fabric_8x2.cu)
// =============================================================================

struct FabricConfig {
    uint8_t layer1_shapes[8];
    uint8_t layer1_input_a[8];
    uint8_t layer1_input_b[8];
    uint8_t layer2_shapes[8];
    uint8_t layer2_input_a[8];
    uint8_t layer2_input_b[8];
};

// =============================================================================
// FROZEN SHAPES
// =============================================================================

__device__ __forceinline__ uint8_t execute_shape(uint8_t shape_id, uint8_t a, uint8_t b) {
    switch (shape_id & 0x0F) {
        case 0:  return a + b;           // ADD
        case 1:  return a - b;           // SUB
        case 2:  return a & b;           // AND
        case 3:  return a | b;           // OR
        case 4:  return a ^ b;           // XOR
        case 5:  return a << (b & 7);    // SHL
        case 6:  return a >> (b & 7);    // SHR
        case 7:  return a * b;           // MUL
        case 8:  return a < b ? a : b;   // MIN
        case 9:  return a > b ? a : b;   // MAX
        case 10: return (a + b) >> 1;    // AVG
        case 11: return a > b ? a - b : b - a;  // DIFF
        case 12: return a;               // PASS_A
        case 13: return b;               // PASS_B
        case 14: return ~a;              // NOT
        case 15: return 0;               // ZERO/NOP
        default: return a;
    }
}

// =============================================================================
// FABRIC KERNEL
// =============================================================================

__global__ void fabric_8x2_kernel(
    const uint8_t* __restrict__ inputs,
    uint8_t* __restrict__ outputs,
    const FabricConfig* __restrict__ config,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Load 8 inputs
    uint8_t in[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        in[i] = inputs[idx * 8 + i];
    }

    // Layer 1
    uint8_t l1[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t a = in[config->layer1_input_a[i] & 7];
        uint8_t b = in[config->layer1_input_b[i] & 7];
        l1[i] = execute_shape(config->layer1_shapes[i], a, b);
    }

    // Layer 2
    uint8_t l2[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t a = l1[config->layer2_input_a[i] & 7];
        uint8_t b = l1[config->layer2_input_b[i] & 7];
        l2[i] = execute_shape(config->layer2_shapes[i], a, b);
    }

    // Store outputs
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        outputs[idx * 8 + i] = l2[i];
    }
}

// =============================================================================
// INCLUDE COMPILED FABRICS
// =============================================================================

#include "compiled/sorter_4.hpp"
#include "compiled/adder_tree.hpp"

// =============================================================================
// TEST HARNESS
// =============================================================================

void test_sorter_4() {
    printf("=== Testing sorter_4 (compiled from YAML) ===\n\n");

    // Get config from compiled header
    FabricConfig config = gcs::sorter_4::to_fabric_config();

    // Print configuration
    const char* shape_names[] = {
        "ADD", "SUB", "AND", "OR", "XOR", "SHL", "SHR", "MUL",
        "MIN", "MAX", "AVG", "DIFF", "PASS_A", "PASS_B", "NOT", "ZERO"
    };

    printf("Layer 1:\n");
    for (int i = 0; i < 4; i++) {
        printf("  ALU%d: %s(in[%d], in[%d])\n",
               i, shape_names[config.layer1_shapes[i]],
               config.layer1_input_a[i], config.layer1_input_b[i]);
    }

    printf("\nLayer 2:\n");
    for (int i = 0; i < 4; i++) {
        printf("  ALU%d: %s(L0[%d], L0[%d])\n",
               i, shape_names[config.layer2_shapes[i]],
               config.layer2_input_a[i], config.layer2_input_b[i]);
    }

    // Allocate
    FabricConfig* d_config;
    uint8_t* d_inputs;
    uint8_t* d_outputs;
    cudaMalloc(&d_config, sizeof(FabricConfig));
    cudaMalloc(&d_inputs, 8);
    cudaMalloc(&d_outputs, 8);

    cudaMemcpy(d_config, &config, sizeof(FabricConfig), cudaMemcpyHostToDevice);

    // Test cases
    struct TestCase {
        uint8_t inputs[4];
        const char* description;
    };

    TestCase tests[] = {
        {{7, 2, 9, 1}, "Unsorted: [7,2,9,1]"},
        {{1, 2, 3, 4}, "Already sorted: [1,2,3,4]"},
        {{4, 3, 2, 1}, "Reverse sorted: [4,3,2,1]"},
        {{5, 5, 5, 5}, "All same: [5,5,5,5]"},
        {{100, 200, 50, 150}, "Larger values: [100,200,50,150]"},
    };

    printf("\n--- Test Results ---\n\n");

    int passed = 0;
    for (const auto& test : tests) {
        // Prepare 8-byte input (only first 4 used)
        uint8_t inputs[8] = {test.inputs[0], test.inputs[1], test.inputs[2], test.inputs[3], 0, 0, 0, 0};
        uint8_t outputs[8];

        cudaMemcpy(d_inputs, inputs, 8, cudaMemcpyHostToDevice);
        fabric_8x2_kernel<<<1, 1>>>(d_inputs, d_outputs, d_config, 1);
        cudaMemcpy(outputs, d_outputs, 8, cudaMemcpyDeviceToHost);

        // Expected: outputs[0] = min, outputs[3] = max
        uint8_t expected_min = *std::min_element(test.inputs, test.inputs + 4);
        uint8_t expected_max = *std::max_element(test.inputs, test.inputs + 4);

        bool min_ok = (outputs[0] == expected_min);
        bool max_ok = (outputs[2] == expected_max || outputs[3] == expected_max);

        printf("%s\n", test.description);
        printf("  Input:  [%3d, %3d, %3d, %3d]\n",
               test.inputs[0], test.inputs[1], test.inputs[2], test.inputs[3]);
        printf("  Output: [%3d, %3d, %3d, %3d]\n",
               outputs[0], outputs[1], outputs[2], outputs[3]);
        printf("  Min: %d (expected %d) %s\n",
               outputs[0], expected_min, min_ok ? "OK" : "FAIL");
        printf("  Max: %d (expected %d) %s\n",
               outputs[3], expected_max, (outputs[3] == expected_max) ? "OK" : "FAIL");
        printf("\n");

        if (min_ok && outputs[3] == expected_max) passed++;
    }

    printf("=== %d/%lu tests passed ===\n\n", passed, sizeof(tests)/sizeof(tests[0]));

    cudaFree(d_config);
    cudaFree(d_inputs);
    cudaFree(d_outputs);
}

void benchmark_sorter_4() {
    printf("=== Benchmark sorter_4 ===\n\n");

    FabricConfig config = gcs::sorter_4::to_fabric_config();

    const uint32_t N = 1024 * 1024;
    const int iterations = 100;

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

    // Warm up
    for (int i = 0; i < 10; i++) {
        fabric_8x2_kernel<<<grid_size, block_size>>>(d_inputs, d_outputs, d_config, N);
    }
    cudaDeviceSynchronize();

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

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double samples_per_sec = (double)N * iterations / (ms / 1000.0);
    double ops_per_sec = samples_per_sec * 16;

    printf("Samples: %d × %d iterations\n", N, iterations);
    printf("Time: %.2f ms\n", ms);
    printf("Samples/sec: %.2f M\n", samples_per_sec / 1e6);
    printf("ALU ops/sec: %.2f B\n", ops_per_sec / 1e9);
    printf("\n");

    delete[] h_inputs;
    cudaFree(d_config);
    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void test_adder_tree() {
    printf("=== Testing adder_tree (compiled from YAML) ===\n\n");

    FabricConfig config = gcs::adder_tree::to_fabric_config();

    const char* shape_names[] = {
        "ADD", "SUB", "AND", "OR", "XOR", "SHL", "SHR", "MUL",
        "MIN", "MAX", "AVG", "DIFF", "PASS_A", "PASS_B", "NOT", "ZERO"
    };

    printf("Layer 1:\n");
    for (int i = 0; i < 4; i++) {
        printf("  ALU%d: %s(in[%d], in[%d])\n",
               i, shape_names[config.layer1_shapes[i]],
               config.layer1_input_a[i], config.layer1_input_b[i]);
    }

    printf("\nLayer 2:\n");
    for (int i = 0; i < 2; i++) {
        printf("  ALU%d: %s(L0[%d], L0[%d])\n",
               i, shape_names[config.layer2_shapes[i]],
               config.layer2_input_a[i], config.layer2_input_b[i]);
    }

    FabricConfig* d_config;
    uint8_t* d_inputs;
    uint8_t* d_outputs;
    cudaMalloc(&d_config, sizeof(FabricConfig));
    cudaMalloc(&d_inputs, 8);
    cudaMalloc(&d_outputs, 8);
    cudaMemcpy(d_config, &config, sizeof(FabricConfig), cudaMemcpyHostToDevice);

    struct TestCase {
        uint8_t inputs[8];
        uint8_t expected_sum_0_3;  // sum of inputs 0-3
        uint8_t expected_sum_4_7;  // sum of inputs 4-7
        const char* description;
    };

    TestCase tests[] = {
        {{1, 2, 3, 4, 5, 6, 7, 8}, 10, 26, "Sequential: [1,2,3,4,5,6,7,8]"},
        {{10, 10, 10, 10, 10, 10, 10, 10}, 40, 40, "All tens: [10,10,10,10,10,10,10,10]"},
        {{0, 0, 0, 0, 1, 1, 1, 1}, 0, 4, "Split: [0,0,0,0,1,1,1,1]"},
        {{50, 50, 50, 50, 0, 0, 0, 0}, 200, 0, "Left heavy: [50,50,50,50,0,0,0,0]"},
    };

    printf("\n--- Test Results ---\n\n");

    int passed = 0;
    for (const auto& test : tests) {
        uint8_t outputs[8];
        cudaMemcpy(d_inputs, test.inputs, 8, cudaMemcpyHostToDevice);
        fabric_8x2_kernel<<<1, 1>>>(d_inputs, d_outputs, d_config, 1);
        cudaMemcpy(outputs, d_outputs, 8, cudaMemcpyDeviceToHost);

        bool sum03_ok = (outputs[0] == test.expected_sum_0_3);
        bool sum47_ok = (outputs[1] == test.expected_sum_4_7);

        printf("%s\n", test.description);
        printf("  Input:  [%d, %d, %d, %d, %d, %d, %d, %d]\n",
               test.inputs[0], test.inputs[1], test.inputs[2], test.inputs[3],
               test.inputs[4], test.inputs[5], test.inputs[6], test.inputs[7]);
        printf("  Output[0] (sum 0-3): %d (expected %d) %s\n",
               outputs[0], test.expected_sum_0_3, sum03_ok ? "OK" : "FAIL");
        printf("  Output[1] (sum 4-7): %d (expected %d) %s\n",
               outputs[1], test.expected_sum_4_7, sum47_ok ? "OK" : "FAIL");
        printf("\n");

        if (sum03_ok && sum47_ok) passed++;
    }

    printf("=== %d/%lu tests passed ===\n\n", passed, sizeof(tests)/sizeof(tests[0]));

    cudaFree(d_config);
    cudaFree(d_inputs);
    cudaFree(d_outputs);
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    printf("=======================================================================\n");
    printf("GCS PIPELINE TEST\n");
    printf("=======================================================================\n");
    printf("\nYAML -> gcs_compile.py -> C++ header -> nvcc -> GPU execution\n\n");

    // Device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute: %d.%d\n\n", prop.major, prop.minor);

    test_sorter_4();
    test_adder_tree();
    benchmark_sorter_4();

    printf("=======================================================================\n");
    printf("PIPELINE COMPLETE\n");
    printf("=======================================================================\n");
    printf("\nThe fabric was defined in YAML, compiled to C++, and executed on GPU.\n");
    printf("This is the GCS pipeline in action.\n\n");

    return 0;
}
