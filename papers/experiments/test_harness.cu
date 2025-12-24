// =============================================================================
// ZIT-1 RIGOROUS TEST HARNESS
// =============================================================================
// Validates convergence behavior across multiple scales
// Run with: ./test_harness [--quick|--full|--stress]
// =============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define SECOND_STAR 1122911624u
#define THRESHOLD 4
#define THREADS 256

// Test configuration
struct TestConfig {
    int cube_size;
    int num_nodes;
    int max_cycles;
    int expected_min_cycles;
    int expected_max_cycles;
    const char* name;
};

// Node structure (same as production)
struct Node {
    uint8_t state, frustration, resonance;
    uint32_t neighbors[6];
    uint32_t lfsr;
    uint8_t rewiring, rewire_dir, eval_ctr;
    uint32_t old_nb;
    uint8_t pre_frust;
};

// Device functions
__device__ int calc_neighbor(int id, int cube_size, int d) {
    int x = id % cube_size;
    int y = (id / cube_size) % cube_size;
    int z = id / (cube_size * cube_size);
    switch(d) {
        case 0: return ((x+1)%cube_size) + y*cube_size + z*cube_size*cube_size;
        case 1: return ((x+cube_size-1)%cube_size) + y*cube_size + z*cube_size*cube_size;
        case 2: return x + ((y+1)%cube_size)*cube_size + z*cube_size*cube_size;
        case 3: return x + ((y+cube_size-1)%cube_size)*cube_size + z*cube_size*cube_size;
        case 4: return x + y*cube_size + ((z+1)%cube_size)*cube_size*cube_size;
        default: return x + y*cube_size + ((z+cube_size-1)%cube_size)*cube_size*cube_size;
    }
}

__device__ uint32_t lfsr32(uint32_t s) {
    uint32_t bit = ((s >> 0) ^ (s >> 10) ^ (s >> 30) ^ (s >> 31)) & 1;
    return (s >> 1) | (bit << 31);
}

// Kernels with num_nodes parameter
__global__ void init_fabric_k(Node* n, int num_nodes, int cube_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;
    n[i].state = (i * 3) & 0xFF;
    n[i].frustration = 0; n[i].resonance = 1;
    n[i].rewiring = 0; n[i].rewire_dir = 0;
    n[i].lfsr = SECOND_STAR ^ (i * 0x9E3779B9) ^ ((i >> 8) * 0x85EBCA6B) ^ ((i >> 16) * 0xC2B2AE35);
    for (int d = 0; d < 6; d++) n[i].neighbors[d] = calc_neighbor(i, cube_size, d);
}

__global__ void phase_kernel_k(Node* n, uint8_t* st, int phase, int num_nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;
    uint8_t nb = st[n[i].neighbors[phase]];
    uint8_t my = n[i].state;
    bool swap = (phase % 2 == 0) ? (my > nb) : (nb > my);
    if (swap) { n[i].state = nb; n[i].resonance = 0; }
}

__global__ void sync_st_k(Node* n, uint8_t* st, int num_nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) st[i] = n[i].state;
}

__global__ void reset_res_k(Node* n, int num_nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_nodes) n[i].resonance = 1;
}

__global__ void plasticity_k(Node* n, int* res_cnt, int* rew_cnt, int cycle, int num_nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;
    uint8_t res = n[i].resonance, fr = n[i].frustration;
    uint32_t rnd = n[i].lfsr;
    uint8_t rw = n[i].rewiring, rd = n[i].rewire_dir;

    if (!rw) {
        if (!res) { if (fr < 255) fr++; } else { fr >>= 1; }
        if (fr >= THRESHOLD) {
            rw = 1;
            rnd = lfsr32(rnd);
            n[i].old_nb = n[i].neighbors[rd];
            n[i].pre_frust = fr; n[i].eval_ctr = 0;
            uint32_t target = (rnd ^ (i * 0x9E3779B9) ^ (cycle * SECOND_STAR)) % num_nodes;
            if (target == (uint32_t)i) target = (target + 1) % num_nodes;
            n[i].neighbors[rd] = target;
            fr = 0; atomicAdd(rew_cnt, 1);
        }
    } else {
        n[i].eval_ctr++;
        if (!res && fr < 255) fr++;
        if (n[i].eval_ctr >= 8) {
            if (fr >= n[i].pre_frust) n[i].neighbors[rd] = n[i].old_nb;
            rd = (rd + 1) % 6; rw = 0; fr = 0;
        }
    }
    n[i].frustration = fr; n[i].rewiring = rw; n[i].rewire_dir = rd;
    n[i].lfsr = lfsr32(rnd);
    if (res) atomicAdd(res_cnt, 1);
}

// Run a single test configuration
int run_test(TestConfig cfg, bool verbose) {
    Node* d_n; uint8_t* d_st; int *d_res, *d_rew;

    size_t node_mem = cfg.num_nodes * sizeof(Node);
    size_t state_mem = cfg.num_nodes;

    cudaError_t err = cudaMalloc(&d_n, node_mem);
    if (err != cudaSuccess) {
        printf("  [SKIP] %s - insufficient memory\n", cfg.name);
        return -1; // Skip, not fail
    }
    cudaMalloc(&d_st, state_mem);
    cudaMalloc(&d_res, sizeof(int));
    cudaMalloc(&d_rew, sizeof(int));

    int blk = (cfg.num_nodes + THREADS - 1) / THREADS;

    init_fabric_k<<<blk, THREADS>>>(d_n, cfg.num_nodes, cfg.cube_size);
    sync_st_k<<<blk, THREADS>>>(d_n, d_st, cfg.num_nodes);
    cudaDeviceSynchronize();

    int convergence_cycle = -1;
    int total_rewires = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int c = 0; c < cfg.max_cycles; c++) {
        reset_res_k<<<blk, THREADS>>>(d_n, cfg.num_nodes);
        for (int p = 0; p < 6; p++) {
            phase_kernel_k<<<blk, THREADS>>>(d_n, d_st, p, cfg.num_nodes);
            sync_st_k<<<blk, THREADS>>>(d_n, d_st, cfg.num_nodes);
        }
        cudaMemset(d_res, 0, 4); cudaMemset(d_rew, 0, 4);
        plasticity_k<<<blk, THREADS>>>(d_n, d_res, d_rew, c, cfg.num_nodes);

        int h_res, h_rew;
        cudaMemcpy(&h_res, d_res, 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_rew, d_rew, 4, cudaMemcpyDeviceToHost);
        total_rewires += h_rew;

        if (h_res == cfg.num_nodes) {
            convergence_cycle = c;
            break;
        }
    }

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop);

    cudaFree(d_n); cudaFree(d_st); cudaFree(d_res); cudaFree(d_rew);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    // Verify results
    int result = 0; // 0 = pass

    if (convergence_cycle < 0) {
        printf("  [FAIL] %s - did not converge in %d cycles\n", cfg.name, cfg.max_cycles);
        result = 1;
    } else if (convergence_cycle < cfg.expected_min_cycles) {
        printf("  [WARN] %s - converged too fast: %d cycles (expected >= %d)\n",
               cfg.name, convergence_cycle, cfg.expected_min_cycles);
        // Warning, not failure - faster is okay
    } else if (convergence_cycle > cfg.expected_max_cycles) {
        printf("  [FAIL] %s - converged too slow: %d cycles (expected <= %d)\n",
               cfg.name, convergence_cycle, cfg.expected_max_cycles);
        result = 1;
    } else {
        if (verbose) {
            printf("  [PASS] %s - %d cycles, %.1fms, %d rewires\n",
                   cfg.name, convergence_cycle, ms, total_rewires);
        } else {
            printf("  [PASS] %s\n", cfg.name);
        }
    }

    return result;
}

// Verify topology properties
int test_topology_invariants() {
    printf("\n=== Topology Invariant Tests ===\n");

    int cube_size = 4;
    int num_nodes = 64;
    Node* h_nodes = (Node*)malloc(num_nodes * sizeof(Node));
    Node* d_nodes;

    cudaMalloc(&d_nodes, num_nodes * sizeof(Node));
    int blk = (num_nodes + THREADS - 1) / THREADS;
    init_fabric_k<<<blk, THREADS>>>(d_nodes, num_nodes, cube_size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_nodes, d_nodes, num_nodes * sizeof(Node), cudaMemcpyDeviceToHost);

    int failures = 0;

    // Test 1: All neighbors are valid node indices
    bool valid_neighbors = true;
    for (int i = 0; i < num_nodes; i++) {
        for (int d = 0; d < 6; d++) {
            if (h_nodes[i].neighbors[d] >= (uint32_t)num_nodes) {
                valid_neighbors = false;
                break;
            }
        }
    }
    if (valid_neighbors) {
        printf("  [PASS] All neighbor indices valid\n");
    } else {
        printf("  [FAIL] Invalid neighbor indices found\n");
        failures++;
    }

    // Test 2: No self-loops initially
    bool no_self_loops = true;
    for (int i = 0; i < num_nodes; i++) {
        for (int d = 0; d < 6; d++) {
            if (h_nodes[i].neighbors[d] == (uint32_t)i) {
                no_self_loops = false;
                break;
            }
        }
    }
    if (no_self_loops) {
        printf("  [PASS] No self-loops in initial topology\n");
    } else {
        printf("  [FAIL] Self-loops found in initial topology\n");
        failures++;
    }

    // Test 3: LFSR seeds are unique
    bool unique_seeds = true;
    for (int i = 0; i < num_nodes && unique_seeds; i++) {
        for (int j = i + 1; j < num_nodes; j++) {
            if (h_nodes[i].lfsr == h_nodes[j].lfsr) {
                unique_seeds = false;
                break;
            }
        }
    }
    if (unique_seeds) {
        printf("  [PASS] All LFSR seeds unique\n");
    } else {
        printf("  [FAIL] Duplicate LFSR seeds found\n");
        failures++;
    }

    // Test 4: Initial states follow pattern
    bool correct_states = true;
    for (int i = 0; i < num_nodes; i++) {
        if (h_nodes[i].state != ((i * 3) & 0xFF)) {
            correct_states = false;
            break;
        }
    }
    if (correct_states) {
        printf("  [PASS] Initial states follow (i*3)&0xFF pattern\n");
    } else {
        printf("  [FAIL] Initial states incorrect\n");
        failures++;
    }

    free(h_nodes);
    cudaFree(d_nodes);

    return failures;
}

// Test determinism with same seed
int test_determinism() {
    printf("\n=== Determinism Tests ===\n");

    TestConfig cfg = {8, 512, 200, 50, 200, "8x8x8 determinism"};

    // Run twice and compare
    int cycles[2];
    int rewires[2];

    for (int run = 0; run < 2; run++) {
        Node* d_n; uint8_t* d_st; int *d_res, *d_rew;
        cudaMalloc(&d_n, cfg.num_nodes * sizeof(Node));
        cudaMalloc(&d_st, cfg.num_nodes);
        cudaMalloc(&d_res, sizeof(int));
        cudaMalloc(&d_rew, sizeof(int));

        int blk = (cfg.num_nodes + THREADS - 1) / THREADS;
        init_fabric_k<<<blk, THREADS>>>(d_n, cfg.num_nodes, cfg.cube_size);
        sync_st_k<<<blk, THREADS>>>(d_n, d_st, cfg.num_nodes);
        cudaDeviceSynchronize();

        cycles[run] = -1;
        rewires[run] = 0;

        for (int c = 0; c < cfg.max_cycles; c++) {
            reset_res_k<<<blk, THREADS>>>(d_n, cfg.num_nodes);
            for (int p = 0; p < 6; p++) {
                phase_kernel_k<<<blk, THREADS>>>(d_n, d_st, p, cfg.num_nodes);
                sync_st_k<<<blk, THREADS>>>(d_n, d_st, cfg.num_nodes);
            }
            cudaMemset(d_res, 0, 4); cudaMemset(d_rew, 0, 4);
            plasticity_k<<<blk, THREADS>>>(d_n, d_res, d_rew, c, cfg.num_nodes);

            int h_res, h_rew;
            cudaMemcpy(&h_res, d_res, 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_rew, d_rew, 4, cudaMemcpyDeviceToHost);
            rewires[run] += h_rew;

            if (h_res == cfg.num_nodes) {
                cycles[run] = c;
                break;
            }
        }

        cudaFree(d_n); cudaFree(d_st); cudaFree(d_res); cudaFree(d_rew);
    }

    int failures = 0;

    if (cycles[0] == cycles[1]) {
        printf("  [PASS] Convergence cycle deterministic: %d\n", cycles[0]);
    } else {
        printf("  [FAIL] Convergence cycle non-deterministic: %d vs %d\n", cycles[0], cycles[1]);
        failures++;
    }

    if (rewires[0] == rewires[1]) {
        printf("  [PASS] Rewire count deterministic: %d\n", rewires[0]);
    } else {
        printf("  [FAIL] Rewire count non-deterministic: %d vs %d\n", rewires[0], rewires[1]);
        failures++;
    }

    return failures;
}

int main(int argc, char** argv) {
    bool quick_mode = false;
    bool full_mode = false;
    bool stress_mode = false;
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--quick") == 0) quick_mode = true;
        else if (strcmp(argv[i], "--full") == 0) full_mode = true;
        else if (strcmp(argv[i], "--stress") == 0) stress_mode = true;
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) verbose = true;
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("ZIT-1 Test Harness\n");
            printf("Usage: %s [OPTIONS]\n\n", argv[0]);
            printf("Options:\n");
            printf("  --quick    Run minimal tests (4x4x4, 8x8x8)\n");
            printf("  --full     Run all tests including large scales\n");
            printf("  --stress   Run stress tests with maximum scale\n");
            printf("  -v         Verbose output\n");
            printf("  --help     Show this help\n");
            printf("\nDefault: runs standard test suite\n");
            return 0;
        }
    }

    // Default to standard mode if nothing specified
    if (!quick_mode && !full_mode && !stress_mode) {
        full_mode = true; // Standard behavior
    }

    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║             ZIT-1 RIGOROUS TEST HARNESS                           ║\n");
    printf("║             Second Star Constant: %u                     ║\n", SECOND_STAR);
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("\nGPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("Memory: %.1f GB free / %.1f GB total\n", free_mem/1e9, total_mem/1e9);

    int total_failures = 0;
    int total_tests = 0;

    // Quick tests (always run)
    printf("\n=== Quick Convergence Tests ===\n");

    TestConfig quick_tests[] = {
        {4, 64, 300, 50, 250, "4x4x4 (64 nodes)"},
        {8, 512, 200, 50, 180, "8x8x8 (512 nodes)"},
    };

    for (int i = 0; i < 2; i++) {
        int result = run_test(quick_tests[i], verbose);
        if (result > 0) total_failures++;
        if (result >= 0) total_tests++;
    }

    // Topology and determinism tests
    total_failures += test_topology_invariants();
    total_tests += 4;

    total_failures += test_determinism();
    total_tests += 2;

    if (full_mode || stress_mode) {
        printf("\n=== Scale Tests ===\n");

        TestConfig scale_tests[] = {
            {16, 4096, 400, 100, 350, "16x16x16 (4K nodes)"},
            {32, 32768, 400, 100, 350, "32x32x32 (32K nodes)"},
            {64, 262144, 300, 80, 250, "64x64x64 (262K nodes)"},
        };

        for (int i = 0; i < 3; i++) {
            int result = run_test(scale_tests[i], verbose);
            if (result > 0) total_failures++;
            if (result >= 0) total_tests++;
        }
    }

    if (stress_mode) {
        printf("\n=== Stress Tests ===\n");

        TestConfig stress_tests[] = {
            {128, 2097152, 800, 300, 700, "128x128x128 (2M nodes)"},
            {256, 16777216, 1500, 600, 1400, "256x256x256 (16.7M nodes)"},
        };

        for (int i = 0; i < 2; i++) {
            int result = run_test(stress_tests[i], verbose);
            if (result > 0) total_failures++;
            if (result >= 0) total_tests++;
        }
    }

    // Summary
    printf("\n═══════════════════════════════════════════════════════════════════\n");
    if (total_failures == 0) {
        printf("  ALL %d TESTS PASSED\n", total_tests);
        printf("═══════════════════════════════════════════════════════════════════\n");
        return 0;
    } else {
        printf("  %d/%d TESTS FAILED\n", total_failures, total_tests);
        printf("═══════════════════════════════════════════════════════════════════\n");
        return 1;
    }
}
