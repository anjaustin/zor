/**
 * TriX Native Ops - Benchmarks
 * 
 * Measure performance of self-hosted operations vs hypothetical BLAS.
 */

#include "trix_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define WARMUP_ITERS 10
#define BENCH_ITERS 100

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void fill_random(float* arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        arr[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

static void fill_ternary(float* arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        int r = rand() % 3;
        arr[i] = (r == 0) ? -1.0f : (r == 1) ? 0.0f : 1.0f;
    }
}

/* ============================================================================
 * BENCHMARKS
 * ============================================================================ */

void bench_ternary_matvec(size_t rows, size_t cols) {
    printf("Ternary MatVec [%zu x %zu]:\n", rows, cols);
    
    float* weights = malloc(rows * cols * sizeof(float));
    float* x = malloc(cols * sizeof(float));
    float* y = malloc(rows * sizeof(float));
    float* scale = malloc(rows * sizeof(float));
    
    fill_ternary(weights, rows * cols);
    fill_random(x, cols);
    for (size_t i = 0; i < rows; i++) scale[i] = 1.0f;
    
    TernaryMatrix mat = trix_ternary_from_float(weights, rows, cols);
    
    /* Warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        trix_ternary_matvec(&mat, x, y, scale);
    }
    
    /* Benchmark */
    double start = get_time_ms();
    for (int i = 0; i < BENCH_ITERS; i++) {
        trix_ternary_matvec(&mat, x, y, scale);
    }
    double end = get_time_ms();
    
    double ms_per_iter = (end - start) / BENCH_ITERS;
    double ops = 2.0 * rows * cols;  /* 2 ops per weight: compare + add/sub */
    double gops = (ops / ms_per_iter) / 1e6;
    
    printf("  Time: %.3f ms/iter\n", ms_per_iter);
    printf("  Throughput: %.2f GOP/s (ternary ops)\n", gops);
    
    /* Count non-zeros for comparison */
    size_t nnz = 0;
    for (size_t i = 0; i < rows * cols; i++) {
        if (weights[i] != 0.0f) nnz++;
    }
    printf("  Sparsity: %.1f%% zeros\n", 100.0 * (1.0 - (double)nnz / (rows * cols)));
    
    trix_ternary_free(&mat);
    free(weights);
    free(x);
    free(y);
    free(scale);
    printf("\n");
}

void bench_ternary_matmul(size_t batch, size_t rows, size_t cols) {
    printf("Ternary MatMul [%zu x %zu x %zu]:\n", batch, rows, cols);
    
    float* weights = malloc(rows * cols * sizeof(float));
    float* X = malloc(batch * cols * sizeof(float));
    float* Y = malloc(batch * rows * sizeof(float));
    float* scale = malloc(rows * sizeof(float));
    
    fill_ternary(weights, rows * cols);
    fill_random(X, batch * cols);
    for (size_t i = 0; i < rows; i++) scale[i] = 1.0f;
    
    TernaryMatrix mat = trix_ternary_from_float(weights, rows, cols);
    
    /* Warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        trix_ternary_matmul(&mat, X, Y, scale, batch);
    }
    
    /* Benchmark */
    double start = get_time_ms();
    for (int i = 0; i < BENCH_ITERS; i++) {
        trix_ternary_matmul(&mat, X, Y, scale, batch);
    }
    double end = get_time_ms();
    
    double ms_per_iter = (end - start) / BENCH_ITERS;
    double ops = 2.0 * batch * rows * cols;
    double gops = (ops / ms_per_iter) / 1e6;
    
    printf("  Time: %.3f ms/iter\n", ms_per_iter);
    printf("  Throughput: %.2f GOP/s\n", gops);
    
    trix_ternary_free(&mat);
    free(weights);
    free(X);
    free(Y);
    free(scale);
    printf("\n");
}

void bench_tile_forward(size_t batch, size_t d_model, size_t d_hidden) {
    printf("Tile Forward [batch=%zu, d_model=%zu, d_hidden=%zu]:\n", 
           batch, d_model, d_hidden);
    
    TrixTile* tile = trix_tile_create(d_model, d_hidden);
    
    float* up = malloc(d_hidden * d_model * sizeof(float));
    float* down = malloc(d_model * d_hidden * sizeof(float));
    float* bias = malloc(d_hidden * sizeof(float));
    float* scales_up = malloc(d_hidden * sizeof(float));
    float* scales_down = malloc(d_model * sizeof(float));
    
    fill_ternary(up, d_hidden * d_model);
    fill_ternary(down, d_model * d_hidden);
    for (size_t i = 0; i < d_hidden; i++) { bias[i] = 0.0f; scales_up[i] = 0.1f; }
    for (size_t i = 0; i < d_model; i++) scales_down[i] = 0.1f;
    
    trix_tile_init_weights(tile, up, down, bias, scales_up, scales_down);
    
    float* x = malloc(batch * d_model * sizeof(float));
    float* y = malloc(batch * d_model * sizeof(float));
    fill_random(x, batch * d_model);
    
    /* Warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        trix_tile_forward(tile, x, y, batch);
    }
    
    /* Benchmark */
    double start = get_time_ms();
    for (int i = 0; i < BENCH_ITERS; i++) {
        trix_tile_forward(tile, x, y, batch);
    }
    double end = get_time_ms();
    
    double ms_per_iter = (end - start) / BENCH_ITERS;
    /* Two matmuls + ReLU */
    double ops = 2.0 * batch * (d_model * d_hidden + d_hidden * d_model) + batch * d_hidden;
    double gops = (ops / ms_per_iter) / 1e6;
    
    printf("  Time: %.3f ms/iter\n", ms_per_iter);
    printf("  Throughput: %.2f GOP/s\n", gops);
    
    trix_tile_free(tile);
    free(up);
    free(down);
    free(bias);
    free(scales_up);
    free(scales_down);
    free(x);
    free(y);
    printf("\n");
}

void bench_routing(size_t dim, size_t num_sigs, size_t queries) {
    printf("Hamming Routing [dim=%zu, sigs=%zu, queries=%zu]:\n", 
           dim, num_sigs, queries);
    
    size_t packed_dim = (dim + 7) / 8;
    
    uint8_t* sigs = malloc(num_sigs * packed_dim);
    uint8_t* inputs = malloc(queries * packed_dim);
    
    for (size_t i = 0; i < num_sigs * packed_dim; i++) sigs[i] = rand() & 0xFF;
    for (size_t i = 0; i < queries * packed_dim; i++) inputs[i] = rand() & 0xFF;
    
    size_t* results = malloc(queries * sizeof(size_t));
    
    /* Warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        for (size_t q = 0; q < queries; q++) {
            results[q] = trix_route_hamming(inputs + q * packed_dim, sigs, num_sigs, packed_dim);
        }
    }
    
    /* Benchmark */
    double start = get_time_ms();
    for (int i = 0; i < BENCH_ITERS; i++) {
        for (size_t q = 0; q < queries; q++) {
            results[q] = trix_route_hamming(inputs + q * packed_dim, sigs, num_sigs, packed_dim);
        }
    }
    double end = get_time_ms();
    
    double ms_per_iter = (end - start) / BENCH_ITERS;
    double routes_per_sec = (queries / ms_per_iter) * 1000.0;
    
    printf("  Time: %.3f ms/iter\n", ms_per_iter);
    printf("  Throughput: %.2f M routes/sec\n", routes_per_sec / 1e6);
    
    free(sigs);
    free(inputs);
    free(results);
    printf("\n");
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(void) {
    printf("TriX Native Ops Benchmarks\n");
    printf("==========================\n\n");
    
    trix_ops_init();
    printf("Version: %s\n", trix_ops_version());
    printf("Iterations: %d (after %d warmup)\n\n", BENCH_ITERS, WARMUP_ITERS);
    
    srand(42);
    
    /* MatVec benchmarks */
    printf("=== Matrix-Vector (Ternary) ===\n\n");
    bench_ternary_matvec(256, 256);
    bench_ternary_matvec(512, 512);
    bench_ternary_matvec(1024, 1024);
    
    /* MatMul benchmarks */
    printf("=== Matrix-Matrix (Ternary) ===\n\n");
    bench_ternary_matmul(32, 256, 256);
    bench_ternary_matmul(32, 512, 512);
    bench_ternary_matmul(64, 512, 512);
    
    /* Tile benchmarks */
    printf("=== Tile Forward ===\n\n");
    bench_tile_forward(32, 256, 512);
    bench_tile_forward(32, 512, 1024);
    bench_tile_forward(64, 512, 1024);
    
    /* Routing benchmarks */
    printf("=== Hamming Routing ===\n\n");
    bench_routing(256, 64, 1024);
    bench_routing(512, 128, 1024);
    bench_routing(1024, 256, 1024);
    
    /* Binary vs Polynomial shape benchmarks */
    printf("=== Binary vs Polynomial Shapes ===\n\n");
    
    {
        size_t n = 1000000;  /* 1M elements = 125KB packed */
        size_t packed = n / 8;
        
        uint8_t* a = malloc(packed);
        uint8_t* b = malloc(packed);
        uint8_t* out = malloc(packed);
        float* af = malloc(n * sizeof(float));
        float* bf = malloc(n * sizeof(float));
        float* outf = malloc(n * sizeof(float));
        
        for (size_t i = 0; i < packed; i++) { a[i] = rand(); b[i] = rand(); }
        for (size_t i = 0; i < n; i++) { af[i] = (rand() % 2); bf[i] = (rand() % 2); }
        
        printf("XOR on %zu elements:\n", n);
        
        /* Binary XOR */
        for (int i = 0; i < WARMUP_ITERS; i++) trix_apply_binary_xor(a, b, out, packed);
        double start = get_time_ms();
        for (int i = 0; i < BENCH_ITERS; i++) trix_apply_binary_xor(a, b, out, packed);
        double binary_ms = (get_time_ms() - start) / BENCH_ITERS;
        
        /* Polynomial XOR */
        for (int i = 0; i < WARMUP_ITERS; i++) {
            for (size_t j = 0; j < n; j++) outf[j] = shape_xor(af[j], bf[j]);
        }
        start = get_time_ms();
        for (int i = 0; i < BENCH_ITERS; i++) {
            for (size_t j = 0; j < n; j++) outf[j] = shape_xor(af[j], bf[j]);
        }
        double poly_ms = (get_time_ms() - start) / BENCH_ITERS;
        
        printf("  Binary:     %.3f ms (%.2f GB/s)\n", binary_ms, (packed * 3.0 / binary_ms) / 1e6);
        printf("  Polynomial: %.3f ms\n", poly_ms);
        printf("  Speedup:    %.1fx\n", poly_ms / binary_ms);
        printf("  Memory:     %zu KB (binary) vs %zu KB (float)\n", packed/1024, n*4/1024);
        
        free(a); free(b); free(out);
        free(af); free(bf); free(outf);
    }
    printf("\n");
    
    printf("==========================\n");
    printf("Benchmarks complete.\n");
    
    return 0;
}
