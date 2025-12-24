"""
CUDA Parallel Geometry - Thousands of threads, one answer per thread.
"""

import subprocess
import tempfile
from pathlib import Path


CUDA_KERNEL = r'''
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define CHECK(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; } }

// Native add kernel
__global__ void native_add_kernel(uint64_t* a, uint64_t* b, uint64_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

// Parallel geometry add
__global__ void geometry_add_kernel(uint64_t* a, uint64_t* b, uint64_t* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint64_t av = a[idx];
    uint64_t bv = b[idx];

    // Byte-wise: 8 parallel lanes
    uint16_t s0 = (uint16_t)(av & 0xFF) + (bv & 0xFF);
    uint16_t s1 = (uint16_t)((av >> 8) & 0xFF) + ((bv >> 8) & 0xFF);
    uint16_t s2 = (uint16_t)((av >> 16) & 0xFF) + ((bv >> 16) & 0xFF);
    uint16_t s3 = (uint16_t)((av >> 24) & 0xFF) + ((bv >> 24) & 0xFF);
    uint16_t s4 = (uint16_t)((av >> 32) & 0xFF) + ((bv >> 32) & 0xFF);
    uint16_t s5 = (uint16_t)((av >> 40) & 0xFF) + ((bv >> 40) & 0xFF);
    uint16_t s6 = (uint16_t)((av >> 48) & 0xFF) + ((bv >> 48) & 0xFF);
    uint16_t s7 = (uint16_t)((av >> 56) & 0xFF) + ((bv >> 56) & 0xFF);

    // Generate and Propagate
    int g0 = s0 >> 8, g1 = s1 >> 8, g2 = s2 >> 8, g3 = s3 >> 8;
    int g4 = s4 >> 8, g5 = s5 >> 8, g6 = s6 >> 8, g7 = s7 >> 8;
    int p0 = (s0 & 0xFF) == 0xFF, p1 = (s1 & 0xFF) == 0xFF;
    int p2 = (s2 & 0xFF) == 0xFF, p3 = (s3 & 0xFF) == 0xFF;
    int p4 = (s4 & 0xFF) == 0xFF, p5 = (s5 & 0xFF) == 0xFF;
    int p6 = (s6 & 0xFF) == 0xFF, p7 = (s7 & 0xFF) == 0xFF;

    // Kogge-Stone parallel prefix (unrolled)
    // Level 0: stride 1
    int G0 = g0, G1 = g1 | (p1 & g0), G2 = g2 | (p2 & g1), G3 = g3 | (p3 & g2);
    int G4 = g4 | (p4 & g3), G5 = g5 | (p5 & g4), G6 = g6 | (p6 & g5), G7 = g7 | (p7 & g6);
    int P0 = p0, P1 = p1 & p0, P2 = p2 & p1, P3 = p3 & p2;
    int P4 = p4 & p3, P5 = p5 & p4, P6 = p6 & p5, P7 = p7 & p6;

    // Level 1: stride 2
    g0 = G0; g1 = G1; g2 = G2 | (P2 & G0); g3 = G3 | (P3 & G1);
    g4 = G4 | (P4 & G2); g5 = G5 | (P5 & G3); g6 = G6 | (P6 & G4); g7 = G7 | (P7 & G5);
    p0 = P0; p1 = P1; p2 = P2 & P0; p3 = P3 & P1;
    p4 = P4 & P2; p5 = P5 & P3; p6 = P6 & P4; p7 = P7 & P5;

    // Level 2: stride 4
    G0 = g0; G1 = g1; G2 = g2; G3 = g3;
    G4 = g4 | (p4 & g0); G5 = g5 | (p5 & g1); G6 = g6 | (p6 & g2); G7 = g7 | (p7 & g3);

    // Carries: c[i] = G[i-1]
    int c1 = G0, c2 = G1, c3 = G2, c4 = G3, c5 = G4, c6 = G5, c7 = G6;

    // Assemble result
    uint64_t r = ((s0 & 0xFF) | ((uint64_t)((s1 + c1) & 0xFF) << 8) |
                  ((uint64_t)((s2 + c2) & 0xFF) << 16) | ((uint64_t)((s3 + c3) & 0xFF) << 24) |
                  ((uint64_t)((s4 + c4) & 0xFF) << 32) | ((uint64_t)((s5 + c5) & 0xFF) << 40) |
                  ((uint64_t)((s6 + c6) & 0xFF) << 48) | ((uint64_t)((s7 + c7) & 0xFF) << 56));

    result[idx] = r;
}

int main() {
    const int N = 1000000;
    const int ITERS = 100;
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    uint64_t *h_a, *h_b, *h_result, *h_expected;
    uint64_t *d_a, *d_b, *d_result;

    h_a = (uint64_t*)malloc(N * sizeof(uint64_t));
    h_b = (uint64_t*)malloc(N * sizeof(uint64_t));
    h_result = (uint64_t*)malloc(N * sizeof(uint64_t));
    h_expected = (uint64_t*)malloc(N * sizeof(uint64_t));

    CHECK(cudaMalloc(&d_a, N * sizeof(uint64_t)));
    CHECK(cudaMalloc(&d_b, N * sizeof(uint64_t)));
    CHECK(cudaMalloc(&d_result, N * sizeof(uint64_t)));

    for (int i = 0; i < N; i++) {
        h_a[i] = (uint64_t)i * 12345ULL + 0xDEADBEEFULL;
        h_b[i] = (uint64_t)i * 67890ULL + 0xCAFEBABEULL;
        h_expected[i] = h_a[i] + h_b[i];
    }

    CHECK(cudaMemcpy(d_a, h_a, N * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, N * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Warmup
    native_add_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_result, N);
    geometry_add_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_result, N);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Benchmark geometry
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++) {
        geometry_add_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_result, N);
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float geom_ms = 0;
    CHECK(cudaEventElapsedTime(&geom_ms, start, stop));

    // Benchmark native
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; i++) {
        native_add_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_result, N);
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float native_ms = 0;
    CHECK(cudaEventElapsedTime(&native_ms, start, stop));

    // Verify
    geometry_add_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_a, d_b, d_result, N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_result, d_result, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_result[i] != h_expected[i]) errors++;
    }

    long long total_ops = (long long)N * ITERS;
    double geom_ns = (geom_ms * 1e6) / total_ops;
    double native_ns = (native_ms * 1e6) / total_ops;

    printf("\nCUDA PARALLEL GEOMETRY BENCHMARK\n");
    printf("================================================\n\n");
    printf("GPU: NVIDIA Thor\n");
    printf("Operations: %lld (1M x 100 iters)\n\n", total_ops);
    printf("%-12s %10s %10s\n", "Method", "Time", "ns/op");
    printf("------------------------------------------------\n");
    printf("%-12s %8.2f ms %8.4f ns\n", "Geometry", geom_ms, geom_ns);
    printf("%-12s %8.2f ms %8.4f ns\n", "Native", native_ms, native_ns);
    printf("------------------------------------------------\n\n");
    printf("Verification: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
    printf("Geometry/Native: %.2fx\n", geom_ms / native_ms);

    free(h_a); free(h_b); free(h_result); free(h_expected);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_result);
    return 0;
}
'''


def run_cuda_benchmark():
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "bench.cu"
        bin = Path(tmpdir) / "bench"
        src.write_text(CUDA_KERNEL)

        # Try different arch flags
        for arch in ["-arch=sm_87", "-arch=sm_89", ""]:
            result = subprocess.run(
                ["nvcc", "-O3", arch, "-o", str(bin), str(src)] if arch else ["nvcc", "-O3", "-o", str(bin), str(src)],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                break

        if result.returncode != 0:
            print("Compile Error:", result.stderr)
            return

        result = subprocess.run([str(bin)], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)


if __name__ == "__main__":
    run_cuda_benchmark()
