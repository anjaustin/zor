"""
Hollywood Squares FFN - FAST Edition

Shared memory caching. Tiles loaded once, reused across threadblock.
Each threadblock IS a Hollywood Squares grid.

"512 squares. Tiles in shared memory. Broadcast routing."
"""

import numpy as np
import cupy as cp
import torch
import time


# =============================================================================
# OPTIMIZED KERNEL WITH SHARED MEMORY
# =============================================================================

FAST_KERNEL = r'''
extern "C" {

// Cubic B-spline
__device__ __forceinline__ float cubic_bspline(float t) {
    t = fabsf(t);
    if (t < 1.0f) return (2.0f/3.0f) - t*t + 0.5f * t*t*t;
    if (t < 2.0f) { float tmp = 2.0f - t; return (1.0f/6.0f) * tmp * tmp * tmp; }
    return 0.0f;
}

// =============================================================================
// FAST KERNEL: Shared memory tile caching
// Each threadblock = 256 threads, processes 256 tokens
// Tiles loaded to shared memory ONCE, reused across all 256 tokens
// =============================================================================

#define THREADS_PER_BLOCK 256
#define MAX_D_MODEL 128
#define MAX_TILES 32

__global__ void sparse_lookup_ffn_fast(
    const float* __restrict__ x,
    const float* __restrict__ signatures,
    const float* __restrict__ spline_coeffs,
    const float* __restrict__ directions,
    const float* __restrict__ positions,
    float* __restrict__ output,
    int batch_size,
    int d_model,
    int num_tiles,
    int grid_size,
    float position_spread,
    float output_scale
) {
    // Shared memory for tile signatures and directions
    __shared__ float s_signatures[MAX_TILES][MAX_D_MODEL];
    __shared__ float s_directions[MAX_TILES][MAX_D_MODEL];

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // === COOPERATIVE LOAD: All threads load signatures/directions ===
    // Each thread loads multiple elements
    int total_sig_elements = num_tiles * d_model;
    for (int i = tid; i < total_sig_elements; i += THREADS_PER_BLOCK) {
        int t = i / d_model;
        int d = i % d_model;
        s_signatures[t][d] = signatures[i];
        s_directions[t][d] = directions[i];
    }
    __syncthreads();

    if (global_tid >= batch_size) return;

    // Load input
    float x_local[MAX_D_MODEL];
    for (int d = 0; d < d_model; d++) {
        x_local[d] = x[global_tid * d_model + d];
    }
    float pos = positions[global_tid];

    // === PARALLEL TILE SEARCH ===
    // Each thread evaluates ALL tiles - but signatures are in fast shared memory!
    float best_score = -1e30f;
    int best_tile = 0;

    for (int t = 0; t < num_tiles; t++) {
        float content_score = 0.0f;

        // Dot product with cached signature
        #pragma unroll 8
        for (int d = 0; d < d_model; d++) {
            content_score += x_local[d] * s_signatures[t][d];
        }

        // Spatial score
        float tile_center = (float)t / (float)num_tiles * 64.0f;
        float spatial_score = cubic_bspline((pos - tile_center) / position_spread);

        float combined = content_score * spatial_score;
        if (combined > best_score) {
            best_score = combined;
            best_tile = t;
        }
    }

    // === SPLINE LOOKUP ===
    float a = tanhf(x_local[0] * 0.1f);
    float b = tanhf(x_local[1] * 0.1f);

    int idx_a = (int)(((a + 1.0f) / 2.0f) * (float)grid_size);
    int idx_b = (int)(((b + 1.0f) / 2.0f) * (float)grid_size);
    idx_a = min(max(idx_a, 0), grid_size - 1);
    idx_b = min(max(idx_b, 0), grid_size - 1);

    int coeff_idx = best_tile * grid_size * grid_size * 3 + idx_a * grid_size * 3 + idx_b * 3;
    float base = spline_coeffs[coeff_idx];
    float slope_a = spline_coeffs[coeff_idx + 1];
    float slope_b = spline_coeffs[coeff_idx + 2];

    float cell_size = 2.0f / (float)grid_size;
    float local_a = ((a + 1.0f) - idx_a * cell_size) / cell_size;
    float local_b = ((b + 1.0f) - idx_b * cell_size) / cell_size;
    float scale = (base + slope_a * local_a + slope_b * local_b) * output_scale;

    // === OUTPUT with cached direction ===
    for (int d = 0; d < d_model; d++) {
        output[global_tid * d_model + d] = x_local[d] + scale * s_directions[best_tile][d];
    }
}

// =============================================================================
// EVEN FASTER: Warp-level reduction for tile selection
// =============================================================================

__global__ void sparse_lookup_ffn_warp(
    const float* __restrict__ x,
    const float* __restrict__ signatures,
    const float* __restrict__ spline_coeffs,
    const float* __restrict__ directions,
    const float* __restrict__ positions,
    float* __restrict__ output,
    int batch_size,
    int d_model,
    int num_tiles,
    int grid_size,
    float position_spread,
    float output_scale
) {
    // Each warp (32 threads) processes one token
    // Threads within warp evaluate different tiles in parallel
    // Warp-level reduction finds best tile

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_id >= batch_size) return;

    // Shared memory for this block's tiles
    __shared__ float s_signatures[MAX_TILES][MAX_D_MODEL];
    __shared__ float s_directions[MAX_TILES][MAX_D_MODEL];

    // Cooperative load
    int tid = threadIdx.x;
    int total_elements = num_tiles * d_model;
    for (int i = tid; i < total_elements; i += blockDim.x) {
        int t = i / d_model;
        int d = i % d_model;
        s_signatures[t][d] = signatures[i];
        s_directions[t][d] = directions[i];
    }
    __syncthreads();

    // Load input for this warp
    float x_local[MAX_D_MODEL];
    float pos;
    if (lane_id == 0) {
        for (int d = 0; d < d_model; d++) {
            x_local[d] = x[warp_id * d_model + d];
        }
        pos = positions[warp_id];
    }

    // Broadcast x_local to all lanes
    for (int d = 0; d < d_model; d++) {
        x_local[d] = __shfl_sync(0xFFFFFFFF, x_local[d], 0);
    }
    pos = __shfl_sync(0xFFFFFFFF, pos, 0);

    // === PARALLEL TILE EVALUATION ===
    // Each lane evaluates (num_tiles / 32) tiles
    float my_best_score = -1e30f;
    int my_best_tile = 0;

    for (int t = lane_id; t < num_tiles; t += 32) {
        float content_score = 0.0f;

        for (int d = 0; d < d_model; d++) {
            content_score += x_local[d] * s_signatures[t][d];
        }

        float tile_center = (float)t / (float)num_tiles * 64.0f;
        float spatial_score = cubic_bspline((pos - tile_center) / position_spread);
        float combined = content_score * spatial_score;

        if (combined > my_best_score) {
            my_best_score = combined;
            my_best_tile = t;
        }
    }

    // === WARP REDUCTION: Find global best ===
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_score = __shfl_down_sync(0xFFFFFFFF, my_best_score, offset);
        int other_tile = __shfl_down_sync(0xFFFFFFFF, my_best_tile, offset);
        if (other_score > my_best_score) {
            my_best_score = other_score;
            my_best_tile = other_tile;
        }
    }

    // Lane 0 has the winner
    int best_tile = __shfl_sync(0xFFFFFFFF, my_best_tile, 0);

    // Only lane 0 does the spline lookup and output
    if (lane_id == 0) {
        float a = tanhf(x_local[0] * 0.1f);
        float b = tanhf(x_local[1] * 0.1f);

        int idx_a = (int)(((a + 1.0f) / 2.0f) * (float)grid_size);
        int idx_b = (int)(((b + 1.0f) / 2.0f) * (float)grid_size);
        idx_a = min(max(idx_a, 0), grid_size - 1);
        idx_b = min(max(idx_b, 0), grid_size - 1);

        int coeff_idx = best_tile * grid_size * grid_size * 3 + idx_a * grid_size * 3 + idx_b * 3;
        float scale = spline_coeffs[coeff_idx] * output_scale;

        for (int d = 0; d < d_model; d++) {
            output[warp_id * d_model + d] = x_local[d] + scale * s_directions[best_tile][d];
        }
    }
}

}  // extern "C"
'''


class HollywoodSquaresFFNFast:
    """Optimized Hollywood Squares FFN with shared memory."""

    def __init__(
        self,
        d_model: int = 128,
        num_tiles: int = 16,
        grid_size: int = 16,
        position_spread: float = 2.0,
    ):
        print("Initializing Hollywood Squares FFN FAST...")

        self.d_model = d_model
        self.num_tiles = num_tiles
        self.grid_size = grid_size
        self.position_spread = position_spread

        # Weights
        self.signatures = cp.random.randn(num_tiles, d_model).astype(cp.float32)
        self.signatures = cp.sign(self.signatures)
        self.spline_coeffs = cp.random.randn(num_tiles, grid_size, grid_size, 3).astype(cp.float32) * 0.5
        self.directions = cp.random.randn(num_tiles, d_model).astype(cp.float32) * 0.02

        # Compile
        print("  Compiling optimized kernels...", end=" ", flush=True)
        self.module = cp.RawModule(code=FAST_KERNEL)
        self.kernel_fast = self.module.get_function('sparse_lookup_ffn_fast')
        self.kernel_warp = self.module.get_function('sparse_lookup_ffn_warp')
        print("done.")

        self.threads_per_block = 256
        self.output_scale = 0.1

        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        print(f"  Device: {props['name'].decode()}")
        print("  Ready.")

    def forward_fast(self, x: cp.ndarray, positions: cp.ndarray = None) -> cp.ndarray:
        batch_size = x.shape[0]
        if positions is None:
            positions = cp.arange(batch_size, dtype=cp.float32) % 64

        output = cp.zeros_like(x)
        blocks = (batch_size + self.threads_per_block - 1) // self.threads_per_block

        self.kernel_fast(
            (blocks,), (self.threads_per_block,),
            (
                x, self.signatures.reshape(-1), self.spline_coeffs.reshape(-1),
                self.directions.reshape(-1), positions, output,
                batch_size, self.d_model, self.num_tiles, self.grid_size,
                self.position_spread, self.output_scale
            )
        )
        return output

    def forward_warp(self, x: cp.ndarray, positions: cp.ndarray = None) -> cp.ndarray:
        batch_size = x.shape[0]
        if positions is None:
            positions = cp.arange(batch_size, dtype=cp.float32) % 64

        output = cp.zeros_like(x)
        # Each warp processes one token, so we need batch_size warps
        warps_per_block = self.threads_per_block // 32
        blocks = (batch_size + warps_per_block - 1) // warps_per_block

        self.kernel_warp(
            (blocks,), (self.threads_per_block,),
            (
                x, self.signatures.reshape(-1), self.spline_coeffs.reshape(-1),
                self.directions.reshape(-1), positions, output,
                batch_size, self.d_model, self.num_tiles, self.grid_size,
                self.position_spread, self.output_scale
            )
        )
        return output

    def benchmark(self, batch_size: int = 100000) -> dict:
        cp.get_default_memory_pool().free_all_blocks()

        x = cp.random.randn(batch_size, self.d_model).astype(cp.float32)
        positions = cp.arange(batch_size, dtype=cp.float32) % 64

        # Warmup
        _ = self.forward_fast(x, positions)
        cp.cuda.Stream.null.synchronize()

        # Benchmark fast kernel
        times_fast = []
        for _ in range(5):
            start = time.perf_counter()
            _ = self.forward_fast(x, positions)
            cp.cuda.Stream.null.synchronize()
            times_fast.append(time.perf_counter() - start)

        # Benchmark warp kernel
        times_warp = []
        for _ in range(5):
            start = time.perf_counter()
            _ = self.forward_warp(x, positions)
            cp.cuda.Stream.null.synchronize()
            times_warp.append(time.perf_counter() - start)

        fast_best = min(times_fast)
        warp_best = min(times_warp)

        return {
            "batch_size": batch_size,
            "fast_throughput": batch_size / fast_best / 1e6,
            "warp_throughput": batch_size / warp_best / 1e6,
        }


def benchmark_all():
    """Full benchmark comparison."""
    import sys
    sys.path.insert(0, '/workspace/trix_latest/TriXO/src')

    print("=" * 70)
    print("HOLLYWOOD SQUARES FFN - ALL VARIANTS")
    print("=" * 70)
    print()

    d_model = 128
    num_tiles = 16

    # Load all versions
    hs_fast = HollywoodSquaresFFNFast(d_model=d_model, num_tiles=num_tiles)

    from trix.nn.sparse_lookup_v4 import SparseLookupFFNv4
    torch_ffn = SparseLookupFFNv4(d_model=d_model, num_tiles=num_tiles).cuda()

    print()
    print("Benchmark:")
    print("-" * 70)
    print(f"{'Batch':>12} {'PyTorch':>12} {'HS-Fast':>12} {'HS-Warp':>12} {'Best Speedup':>14}")
    print("-" * 70)

    for batch_size in [10000, 50000, 100000, 500000, 1000000]:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()

            # Hollywood Squares
            result = hs_fast.benchmark(batch_size)

            # PyTorch
            x_torch = torch.randn(1, batch_size, d_model).cuda()
            with torch.no_grad():
                _ = torch_ffn(x_torch)
            torch.cuda.synchronize()

            torch_times = []
            for _ in range(5):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = torch_ffn(x_torch)
                torch.cuda.synchronize()
                torch_times.append(time.perf_counter() - start)

            torch_tp = batch_size / min(torch_times) / 1e6

            best_hs = max(result['fast_throughput'], result['warp_throughput'])
            speedup = best_hs / torch_tp

            print(f"{batch_size:>12,} {torch_tp:>10.2f}M/s {result['fast_throughput']:>10.2f}M/s "
                  f"{result['warp_throughput']:>10.2f}M/s {speedup:>12.1f}×")

        except Exception as e:
            print(f"{batch_size:>12,}: Error - {str(e)[:40]}")
            break

    print("-" * 70)
    print()
    print("=" * 70)


if __name__ == "__main__":
    benchmark_all()
