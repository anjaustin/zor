/**
 * Frozen Shapes - C++ Main Entry Point
 *
 * Pure C++ execution of frozen 6502 ALU shapes.
 *
 * Build: nvcc -std=c++17 -O3 -o frozen main.cpp -x cu
 * Run:   ./frozen
 */

#include "frozen_shapes.hpp"
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <cstring>

using namespace frozen;
using namespace std::chrono;

// =============================================================================
// Verification
// =============================================================================

bool verify_shapes() {
    std::cout << "=======================================================================\n";
    std::cout << "VERIFYING FROZEN SHAPES\n";
    std::cout << "=======================================================================\n\n";

    ShapeExecutor exec(1024);

    // Test cases: shape_id, a, b, c, expected_result, expected_carry
    struct TestCase {
        uint8_t shape_id;
        uint8_t a, b, c;
        uint8_t expected_result;
        uint8_t expected_carry;
        const char* name;
    };

    TestCase tests[] = {
        {0, 0x10, 0x20, 0, 0x30, 0, "ADD"},
        {0, 0xFF, 0x01, 0, 0x00, 1, "ADD carry"},
        {1, 0x50, 0x20, 1, 0x30, 1, "SUB"},
        {2, 0xF0, 0x0F, 0, 0x00, 0, "AND"},
        {3, 0xF0, 0x0F, 0, 0xFF, 0, "OR"},
        {4, 0xFF, 0xAA, 0, 0x55, 0, "XOR"},
        {5, 0x40, 0x00, 0, 0x80, 0, "ASL"},
        {5, 0x80, 0x00, 0, 0x00, 1, "ASL carry"},
        {6, 0x02, 0x00, 0, 0x01, 0, "LSR"},
        {7, 0x40, 0x00, 1, 0x81, 0, "ROL"},
        {8, 0x02, 0x00, 1, 0x81, 0, "ROR"},
        {9, 0xFF, 0x00, 0, 0x00, 0, "INC wrap"},
        {10, 0x00, 0x00, 0, 0xFF, 0, "DEC wrap"},
    };

    int passed = 0, failed = 0;

    for (const auto& test : tests) {
        exec.opcodes[0] = test.shape_id;
        exec.operand_a[0] = test.a;
        exec.operand_b[0] = test.b;
        exec.carry_in[0] = test.c;

        exec.execute(1);

        bool ok = (exec.result[0] == test.expected_result) &&
                  (exec.carry_out[0] == test.expected_carry);

        if (ok) {
            passed++;
            std::cout << "  " << std::setw(12) << test.name << ": PASS\n";
        } else {
            failed++;
            std::cout << "  " << std::setw(12) << test.name << ": FAIL"
                      << " (got " << std::hex << (int)exec.result[0]
                      << " c=" << (int)exec.carry_out[0]
                      << ", expected " << (int)test.expected_result
                      << " c=" << (int)test.expected_carry << ")\n";
        }
    }

    std::cout << "\n  " << passed << " passed, " << failed << " failed\n\n";
    return failed == 0;
}

// =============================================================================
// Benchmark
// =============================================================================

void benchmark() {
    std::cout << "=======================================================================\n";
    std::cout << "FROZEN SHAPES BENCHMARK\n";
    std::cout << "=======================================================================\n\n";

    const uint32_t N = 1024 * 1024;  // 1M ops
    const int iterations = 1000;

    ShapeExecutor exec(N);

    // Set 6502 routing
    uint8_t routing[256];
    for (int i = 0; i < 11; i++) routing[i] = i;
    for (int i = 11; i < 256; i++) routing[i] = 15;
    exec.set_routing(routing);

    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> op_dist(0, 10);
    std::uniform_int_distribution<> byte_dist(0, 255);
    std::uniform_int_distribution<> bit_dist(0, 1);

    for (uint32_t i = 0; i < N; i++) {
        exec.opcodes[i] = op_dist(gen);
        exec.operand_a[i] = byte_dist(gen);
        exec.operand_b[i] = byte_dist(gen);
        exec.carry_in[i] = bit_dist(gen);
    }

    // Create graph
    exec.create_graph(N);

    // Warm up
    for (int i = 0; i < 10; i++) {
        exec.execute_graph();
    }

    // Benchmark raw kernel
    auto start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        exec.execute(N);
    }
    auto end = high_resolution_clock::now();

    auto raw_us = duration_cast<microseconds>(end - start).count() / (double)iterations;
    double raw_ops = (double)N * iterations / (duration_cast<microseconds>(end - start).count() / 1e6);

    // Benchmark graph
    start = high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        exec.execute_graph();
    }
    end = high_resolution_clock::now();

    auto graph_us = duration_cast<microseconds>(end - start).count() / (double)iterations;
    double graph_ops = (double)N * iterations / (duration_cast<microseconds>(end - start).count() / 1e6);

    // Print results
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "[Raw Kernel]\n";
    std::cout << "  Latency:    " << raw_us << " μs\n";
    std::cout << "  Throughput: " << (raw_ops / 1e9) << " B ops/sec\n\n";

    std::cout << "[CUDA Graph]\n";
    std::cout << "  Latency:    " << graph_us << " μs\n";
    std::cout << "  Throughput: " << (graph_ops / 1e9) << " B ops/sec\n\n";

    std::cout << "  Speedup: " << (raw_us / graph_us) << "x\n\n";

    // Comparison to original 6502
    double speedup = graph_ops / 1e6;  // Original 6502 was 1MHz
    std::cout << "  vs 1MHz 6502: " << std::setprecision(0) << speedup << "× faster\n\n";
}

// =============================================================================
// Interactive Mode
// =============================================================================

void interactive() {
    std::cout << "=======================================================================\n";
    std::cout << "FROZEN SHAPES - INTERACTIVE MODE\n";
    std::cout << "=======================================================================\n\n";

    ShapeExecutor exec(16);

    std::cout << "Enter: <shape_id> <A> <B> <C> (hex, e.g., '0 FF 01 0' for ADD)\n";
    std::cout << "Shape IDs: 0=ADD, 1=SUB, 2=AND, 3=OR, 4=XOR, 5=ASL, 6=LSR\n";
    std::cout << "           7=ROL, 8=ROR, 9=INC, 10=DEC\n";
    std::cout << "Type 'q' to quit\n\n";

    std::string line;
    while (true) {
        std::cout << "> ";
        std::getline(std::cin, line);

        if (line == "q" || line == "quit") break;

        int shape_id, a, b, c;
        if (sscanf(line.c_str(), "%x %x %x %x", &shape_id, &a, &b, &c) != 4) {
            std::cout << "Invalid input\n";
            continue;
        }

        exec.opcodes[0] = shape_id;
        exec.operand_a[0] = a;
        exec.operand_b[0] = b;
        exec.carry_in[0] = c;

        exec.execute(1);

        std::cout << "  Result: " << std::hex << std::uppercase
                  << std::setw(2) << std::setfill('0') << (int)exec.result[0]
                  << " Carry: " << (int)exec.carry_out[0]
                  << std::dec << "\n";
    }
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    // Parse args
    bool run_benchmark = true;
    bool run_interactive = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0) {
            run_interactive = true;
            run_benchmark = false;
        }
        if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--benchmark") == 0) {
            run_benchmark = true;
        }
    }

    // Print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute: " << prop.major << "." << prop.minor << "\n\n";

    // Run verification
    if (!verify_shapes()) {
        return 1;
    }

    // Run benchmark or interactive
    if (run_benchmark) {
        benchmark();
    }

    if (run_interactive) {
        interactive();
    }

    std::cout << "=======================================================================\n";
    std::cout << "FROZEN SHAPES - COMPLETE\n";
    std::cout << "=======================================================================\n\n";
    std::cout << "Stack: C++ → CUDA Runtime → GPU Hardware\n\n";
    std::cout << "The shapes are frozen. The routing is learned. The 6502 emerges.\n\n";

    return 0;
}
