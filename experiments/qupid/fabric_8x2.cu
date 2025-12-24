/**
 * 8x2 6502 Fabric
 *
 * Two layers of 8 parallel 6502 ALUs.
 * Each ALU picks its operation via learned routing.
 *
 * Layer 1: 8 inputs → 8 ALUs → 8 intermediate
 * Layer 2: 8 intermediate → 8 ALUs → 8 outputs
 *
 * This is a computational fabric where:
 * - Weights = routing tables (which shape to use)
 * - Activations = frozen shapes (ADD, XOR, AND, etc.)
 * - Connections = wiring between layers
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

// =============================================================================
// FROZEN SHAPES (same as before)
// =============================================================================

__device__ __forceinline__ uint8_t shape_add(uint8_t a, uint8_t b) {
    return a + b;
}

__device__ __forceinline__ uint8_t shape_sub(uint8_t a, uint8_t b) {
    return a - b;
}

__device__ __forceinline__ uint8_t shape_and(uint8_t a, uint8_t b) {
    return a & b;
}

__device__ __forceinline__ uint8_t shape_or(uint8_t a, uint8_t b) {
    return a | b;
}

__device__ __forceinline__ uint8_t shape_xor(uint8_t a, uint8_t b) {
    return a ^ b;
}

__device__ __forceinline__ uint8_t shape_shl(uint8_t a, uint8_t b) {
    return a << (b & 7);
}

__device__ __forceinline__ uint8_t shape_shr(uint8_t a, uint8_t b) {
    return a >> (b & 7);
}

__device__ __forceinline__ uint8_t shape_mul(uint8_t a, uint8_t b) {
    return a * b;  // Low 8 bits of product
}

__device__ __forceinline__ uint8_t shape_min(uint8_t a, uint8_t b) {
    return a < b ? a : b;
}

__device__ __forceinline__ uint8_t shape_max(uint8_t a, uint8_t b) {
    return a > b ? a : b;
}

__device__ __forceinline__ uint8_t shape_avg(uint8_t a, uint8_t b) {
    return (a + b) >> 1;
}

__device__ __forceinline__ uint8_t shape_diff(uint8_t a, uint8_t b) {
    return a > b ? a - b : b - a;  // Absolute difference
}

__device__ __forceinline__ uint8_t shape_pass_a(uint8_t a, uint8_t b) {
    return a;
}

__device__ __forceinline__ uint8_t shape_pass_b(uint8_t a, uint8_t b) {
    return b;
}

__device__ __forceinline__ uint8_t shape_not_a(uint8_t a, uint8_t b) {
    return ~a;
}

__device__ __forceinline__ uint8_t shape_const_zero(uint8_t a, uint8_t b) {
    return 0;
}

__device__ __forceinline__ uint8_t execute_shape(uint8_t shape_id, uint8_t a, uint8_t b) {
    switch (shape_id & 0x0F) {
        case 0:  return shape_add(a, b);
        case 1:  return shape_sub(a, b);
        case 2:  return shape_and(a, b);
        case 3:  return shape_or(a, b);
        case 4:  return shape_xor(a, b);
        case 5:  return shape_shl(a, b);
        case 6:  return shape_shr(a, b);
        case 7:  return shape_mul(a, b);
        case 8:  return shape_min(a, b);
        case 9:  return shape_max(a, b);
        case 10: return shape_avg(a, b);
        case 11: return shape_diff(a, b);
        case 12: return shape_pass_a(a, b);
        case 13: return shape_pass_b(a, b);
        case 14: return shape_not_a(a, b);
        case 15: return shape_const_zero(a, b);
        default: return a;
    }
}

// =============================================================================
// FABRIC CONFIGURATION
// =============================================================================

struct FabricConfig {
    // Layer 1: 8 ALUs
    uint8_t layer1_shapes[8];      // Which shape each ALU uses
    uint8_t layer1_input_a[8];     // Which input feeds port A (0-7 = inputs)
    uint8_t layer1_input_b[8];     // Which input feeds port B (0-7 = inputs)

    // Layer 2: 8 ALUs
    uint8_t layer2_shapes[8];      // Which shape each ALU uses
    uint8_t layer2_input_a[8];     // Which L1 output feeds port A (0-7 = L1 outputs)
    uint8_t layer2_input_b[8];     // Which L1 output feeds port B (0-7 = L1 outputs)
};

// =============================================================================
// FABRIC KERNEL
// =============================================================================

__global__ void fabric_8x2_kernel(
    const uint8_t* __restrict__ inputs,     // [N, 8] input values
    uint8_t* __restrict__ outputs,          // [N, 8] output values
    uint8_t* __restrict__ intermediates,    // [N, 8] layer 1 outputs (optional, for debugging)
    const FabricConfig* __restrict__ config,
    uint32_t n
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Load 8 inputs for this sample
    uint8_t in[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        in[i] = inputs[idx * 8 + i];
    }

    // Layer 1: 8 parallel ALUs
    uint8_t l1[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t a = in[config->layer1_input_a[i] & 7];
        uint8_t b = in[config->layer1_input_b[i] & 7];
        l1[i] = execute_shape(config->layer1_shapes[i], a, b);
    }

    // Store intermediates (optional)
    if (intermediates) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            intermediates[idx * 8 + i] = l1[i];
        }
    }

    // Layer 2: 8 parallel ALUs
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
// EXAMPLE CONFIGURATIONS
// =============================================================================

// Config 1: 8-bit parallel adder tree (sum all 8 inputs)
FabricConfig make_adder_tree() {
    FabricConfig c = {};

    // Layer 1: Pairwise additions
    // ALU0: in[0] + in[1], ALU1: in[2] + in[3], ALU2: in[4] + in[5], ALU3: in[6] + in[7]
    for (int i = 0; i < 4; i++) {
        c.layer1_shapes[i] = 0;  // ADD
        c.layer1_input_a[i] = i * 2;
        c.layer1_input_b[i] = i * 2 + 1;
    }
    // ALU4-7: Pass through (unused)
    for (int i = 4; i < 8; i++) {
        c.layer1_shapes[i] = 12;  // PASS_A
        c.layer1_input_a[i] = 0;
        c.layer1_input_b[i] = 0;
    }

    // Layer 2: Sum pairs from Layer 1
    // ALU0: l1[0] + l1[1], ALU1: l1[2] + l1[3]
    c.layer2_shapes[0] = 0;  // ADD
    c.layer2_input_a[0] = 0;
    c.layer2_input_b[0] = 1;

    c.layer2_shapes[1] = 0;  // ADD
    c.layer2_input_a[1] = 2;
    c.layer2_input_b[1] = 3;

    // ALU2: l2[0] + l2[1] (final sum in output[2])
    // Actually we can't do this in 2 layers - need 3 for full tree
    // So output[0] = sum of in[0-3], output[1] = sum of in[4-7]

    for (int i = 2; i < 8; i++) {
        c.layer2_shapes[i] = 12;  // PASS_A
        c.layer2_input_a[i] = i;
        c.layer2_input_b[i] = 0;
    }

    return c;
}

// Config 2: Bitwise logic fabric (AND/OR/XOR tree)
FabricConfig make_logic_tree() {
    FabricConfig c = {};

    // Layer 1: AND pairs
    for (int i = 0; i < 4; i++) {
        c.layer1_shapes[i] = 2;  // AND
        c.layer1_input_a[i] = i * 2;
        c.layer1_input_b[i] = i * 2 + 1;
    }
    // OR pairs
    for (int i = 4; i < 8; i++) {
        c.layer1_shapes[i] = 3;  // OR
        c.layer1_input_a[i] = (i - 4) * 2;
        c.layer1_input_b[i] = (i - 4) * 2 + 1;
    }

    // Layer 2: XOR the AND and OR results
    for (int i = 0; i < 4; i++) {
        c.layer2_shapes[i] = 4;  // XOR
        c.layer2_input_a[i] = i;      // AND result
        c.layer2_input_b[i] = i + 4;  // OR result
    }
    for (int i = 4; i < 8; i++) {
        c.layer2_shapes[i] = 12;  // PASS
        c.layer2_input_a[i] = i - 4;
        c.layer2_input_b[i] = 0;
    }

    return c;
}

// Config 3: Min/Max finder
FabricConfig make_minmax_tree() {
    FabricConfig c = {};

    // Layer 1: Pairwise min and max
    for (int i = 0; i < 4; i++) {
        c.layer1_shapes[i] = 8;  // MIN
        c.layer1_input_a[i] = i * 2;
        c.layer1_input_b[i] = i * 2 + 1;
    }
    for (int i = 4; i < 8; i++) {
        c.layer1_shapes[i] = 9;  // MAX
        c.layer1_input_a[i] = (i - 4) * 2;
        c.layer1_input_b[i] = (i - 4) * 2 + 1;
    }

    // Layer 2: Further reduce
    c.layer2_shapes[0] = 8;  // MIN of mins
    c.layer2_input_a[0] = 0;
    c.layer2_input_b[0] = 1;

    c.layer2_shapes[1] = 8;  // MIN of mins
    c.layer2_input_a[1] = 2;
    c.layer2_input_b[1] = 3;

    c.layer2_shapes[2] = 9;  // MAX of maxes
    c.layer2_input_a[2] = 4;
    c.layer2_input_b[2] = 5;

    c.layer2_shapes[3] = 9;  // MAX of maxes
    c.layer2_input_a[3] = 6;
    c.layer2_input_b[3] = 7;

    // output[0-1] = partial mins, output[2-3] = partial maxes
    for (int i = 4; i < 8; i++) {
        c.layer2_shapes[i] = 11;  // DIFF (range)
        c.layer2_input_a[i] = 2;  // partial max
        c.layer2_input_b[i] = 0;  // partial min
    }

    return c;
}

// Config 4: Difference detector (edge detection in 1D)
FabricConfig make_diff_detector() {
    FabricConfig c = {};

    // Layer 1: Adjacent differences
    for (int i = 0; i < 7; i++) {
        c.layer1_shapes[i] = 11;  // DIFF (absolute difference)
        c.layer1_input_a[i] = i;
        c.layer1_input_b[i] = i + 1;
    }
    c.layer1_shapes[7] = 12;  // PASS
    c.layer1_input_a[7] = 7;
    c.layer1_input_b[7] = 0;

    // Layer 2: Max of differences (find biggest edge)
    c.layer2_shapes[0] = 9;  // MAX
    c.layer2_input_a[0] = 0;
    c.layer2_input_b[0] = 1;

    c.layer2_shapes[1] = 9;  // MAX
    c.layer2_input_a[1] = 2;
    c.layer2_input_b[1] = 3;

    c.layer2_shapes[2] = 9;  // MAX
    c.layer2_input_a[2] = 4;
    c.layer2_input_b[2] = 5;

    c.layer2_shapes[3] = 9;  // MAX
    c.layer2_input_a[3] = 6;
    c.layer2_input_b[3] = 7;

    // Sum of differences (total variation)
    c.layer2_shapes[4] = 0;  // ADD
    c.layer2_input_a[4] = 0;
    c.layer2_input_b[4] = 1;

    c.layer2_shapes[5] = 0;  // ADD
    c.layer2_input_a[5] = 2;
    c.layer2_input_b[5] = 3;

    for (int i = 6; i < 8; i++) {
        c.layer2_shapes[i] = 12;
        c.layer2_input_a[i] = i - 2;
        c.layer2_input_b[i] = 0;
    }

    return c;
}

// =============================================================================
// TEST HARNESS
// =============================================================================

void print_config(const char* name, const FabricConfig& c) {
    const char* shape_names[] = {
        "ADD", "SUB", "AND", "OR", "XOR", "SHL", "SHR", "MUL",
        "MIN", "MAX", "AVG", "DIFF", "PASS_A", "PASS_B", "NOT", "ZERO"
    };

    printf("\n=== %s ===\n", name);
    printf("Layer 1:\n");
    for (int i = 0; i < 8; i++) {
        printf("  ALU%d: %s(in[%d], in[%d])\n",
               i, shape_names[c.layer1_shapes[i] & 0xF],
               c.layer1_input_a[i], c.layer1_input_b[i]);
    }
    printf("Layer 2:\n");
    for (int i = 0; i < 8; i++) {
        printf("  ALU%d: %s(l1[%d], l1[%d])\n",
               i, shape_names[c.layer2_shapes[i] & 0xF],
               c.layer2_input_a[i], c.layer2_input_b[i]);
    }
}

void test_fabric(const char* name, const FabricConfig& config,
                 const uint8_t inputs[8], uint8_t outputs[8]) {
    FabricConfig* d_config;
    uint8_t* d_inputs;
    uint8_t* d_outputs;

    cudaMalloc(&d_config, sizeof(FabricConfig));
    cudaMalloc(&d_inputs, 8);
    cudaMalloc(&d_outputs, 8);

    cudaMemcpy(d_config, &config, sizeof(FabricConfig), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs, inputs, 8, cudaMemcpyHostToDevice);

    fabric_8x2_kernel<<<1, 1>>>(d_inputs, d_outputs, nullptr, d_config, 1);

    cudaMemcpy(outputs, d_outputs, 8, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaMemcpy(outputs, d_outputs, 8, cudaMemcpyDeviceToHost);

    printf("\n%s:\n", name);
    printf("  Input:  [%3d %3d %3d %3d %3d %3d %3d %3d]\n",
           inputs[0], inputs[1], inputs[2], inputs[3],
           inputs[4], inputs[5], inputs[6], inputs[7]);
    printf("  Output: [%3d %3d %3d %3d %3d %3d %3d %3d]\n",
           outputs[0], outputs[1], outputs[2], outputs[3],
           outputs[4], outputs[5], outputs[6], outputs[7]);

    cudaFree(d_config);
    cudaFree(d_inputs);
    cudaFree(d_outputs);
}

void benchmark_fabric(const FabricConfig& config, uint32_t n, int iterations) {
    FabricConfig* d_config;
    uint8_t* d_inputs;
    uint8_t* d_outputs;

    cudaMalloc(&d_config, sizeof(FabricConfig));
    cudaMalloc(&d_inputs, n * 8);
    cudaMalloc(&d_outputs, n * 8);

    cudaMemcpy(d_config, &config, sizeof(FabricConfig), cudaMemcpyHostToDevice);

    // Fill with random data
    uint8_t* h_inputs = new uint8_t[n * 8];
    for (uint32_t i = 0; i < n * 8; i++) {
        h_inputs[i] = rand() & 0xFF;
    }
    cudaMemcpy(d_inputs, h_inputs, n * 8, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    // Warm up
    for (int i = 0; i < 10; i++) {
        fabric_8x2_kernel<<<grid_size, block_size>>>(d_inputs, d_outputs, nullptr, d_config, n);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        fabric_8x2_kernel<<<grid_size, block_size>>>(d_inputs, d_outputs, nullptr, d_config, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double samples_per_sec = (double)n * iterations / (ms / 1000.0);
    double ops_per_sec = samples_per_sec * 16;  // 16 ALU ops per sample (8 per layer × 2 layers)

    printf("\nBenchmark: %d samples × %d iterations\n", n, iterations);
    printf("  Time: %.2f ms\n", ms);
    printf("  Samples/sec: %.2f M\n", samples_per_sec / 1e6);
    printf("  ALU ops/sec: %.2f B\n", ops_per_sec / 1e9);

    delete[] h_inputs;
    cudaFree(d_config);
    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    printf("=======================================================================\n");
    printf("8x2 6502 FABRIC\n");
    printf("=======================================================================\n");
    printf("\n2 layers × 8 parallel ALUs = 16 frozen shape operations per sample\n");

    // Create configurations
    FabricConfig adder = make_adder_tree();
    FabricConfig logic = make_logic_tree();
    FabricConfig minmax = make_minmax_tree();
    FabricConfig diff = make_diff_detector();

    // Print configs
    print_config("Adder Tree", adder);
    print_config("Logic Tree", logic);
    print_config("MinMax Tree", minmax);
    print_config("Diff Detector", diff);

    // Test with sample data
    printf("\n=======================================================================\n");
    printf("TESTS\n");
    printf("=======================================================================\n");

    uint8_t inputs1[] = {10, 20, 30, 40, 50, 60, 70, 80};
    uint8_t outputs[8];

    test_fabric("Adder Tree", adder, inputs1, outputs);
    // Expected: partial sums (10+20=30, 30+40=70, 50+60=110, 70+80=150)

    test_fabric("MinMax Tree", minmax, inputs1, outputs);
    // Expected: mins and maxes propagating

    uint8_t inputs2[] = {100, 105, 103, 110, 108, 150, 145, 140};  // Signal with edges
    test_fabric("Diff Detector", diff, inputs2, outputs);
    // Expected: differences highlight the jump at position 5 (108→150 = 42)

    // Benchmark
    printf("\n=======================================================================\n");
    printf("BENCHMARK\n");
    printf("=======================================================================\n");

    benchmark_fabric(adder, 1024 * 1024, 100);

    printf("\n=======================================================================\n");
    printf("SUMMARY\n");
    printf("=======================================================================\n");
    printf("\nThe 8x2 fabric is a programmable ALU array.\n");
    printf("- 16 frozen shapes per sample\n");
    printf("- Routing tables define the computation\n");
    printf("- No weights to learn - just wire it up\n");
    printf("\nPossible applications:\n");
    printf("- Image processing (edge detection, thresholding)\n");
    printf("- Signal processing (filters, transforms)\n");
    printf("- Cryptographic primitives (hash functions)\n");
    printf("- Custom arithmetic (saturating, fixed-point)\n");
    printf("\n");

    return 0;
}
