"""
Hollywood Squares SparseLookupFFN - GPU Edition

Position is computation. All tiles process in parallel.
No Python loops in the hot path.

"512 squares. Each input finds its tile. All lookups simultaneous."
"""

import numpy as np
import cupy as cp
import torch
import time


# =============================================================================
# HOLLYWOOD SQUARES FFN KERNEL
# =============================================================================

SPARSE_LOOKUP_KERNEL = r'''
extern "C" {

// Cubic B-spline kernel - C² continuous
__device__ __forceinline__ float cubic_bspline(float t) {
    t = fabsf(t);
    if (t < 1.0f) {
        return (2.0f/3.0f) - t*t + 0.5f * t*t*t;
    } else if (t < 2.0f) {
        float tmp = 2.0f - t;
        return (1.0f/6.0f) * tmp * tmp * tmp;
    }
    return 0.0f;
}

// =============================================================================
// SINGLE-KERNEL SPARSE LOOKUP FFN
// Each thread processes one input token
// All tiles evaluated in parallel within the thread
// =============================================================================

__global__ void sparse_lookup_ffn_kernel(
    const float* x,              // [batch_size, d_model]
    const float* signatures,     // [num_tiles, d_model]
    const float* spline_coeffs,  // [num_tiles, grid_size, grid_size, 3]
    const float* directions,     // [num_tiles, d_model]
    const float* positions,      // [batch_size] - position of each token
    float* output,               // [batch_size, d_model]
    int batch_size,
    int d_model,
    int num_tiles,
    int grid_size,
    float position_spread,
    float output_scale
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    // Load input for this token
    const float* x_in = &x[tid * d_model];
    float pos = positions[tid];

    // === CONTENT ROUTING: dot product with all tiles ===
    float best_score = -1e30f;
    int best_tile = 0;

    for (int t = 0; t < num_tiles; t++) {
        // Content score: x · signature[t]
        float content_score = 0.0f;
        const float* sig = &signatures[t * d_model];

        for (int d = 0; d < d_model; d++) {
            content_score += x_in[d] * sig[d];
        }

        // Spatial score: B-spline based on position
        float tile_center = (float)t / (float)num_tiles * 64.0f;  // max_seq_len=64
        float pos_diff = (pos - tile_center) / position_spread;
        float spatial_score = cubic_bspline(pos_diff);

        // Combined score (content × spatial)
        float combined = content_score * spatial_score;

        if (combined > best_score) {
            best_score = combined;
            best_tile = t;
        }
    }

    // === SPLINE LOOKUP for winning tile ===
    // Compress input to 2D coordinates (simplified: use first two dims)
    // Real version would use learned compression
    float a = tanhf(x_in[0] * 0.1f);  // Map to [-1, 1]
    float b = tanhf(x_in[1] * 0.1f);

    // Grid indices
    int idx_a = (int)(((a + 1.0f) / 2.0f) * (float)grid_size);
    int idx_b = (int)(((b + 1.0f) / 2.0f) * (float)grid_size);
    idx_a = min(max(idx_a, 0), grid_size - 1);
    idx_b = min(max(idx_b, 0), grid_size - 1);

    // Get spline coefficients for this tile/cell
    int coeff_idx = best_tile * grid_size * grid_size * 3 + idx_a * grid_size * 3 + idx_b * 3;
    float base = spline_coeffs[coeff_idx];
    float slope_a = spline_coeffs[coeff_idx + 1];
    float slope_b = spline_coeffs[coeff_idx + 2];

    // Local coordinates within cell
    float cell_size = 2.0f / (float)grid_size;
    float local_a = ((a + 1.0f) - idx_a * cell_size) / cell_size;
    float local_b = ((b + 1.0f) - idx_b * cell_size) / cell_size;

    // Evaluate spline
    float scale = base + slope_a * local_a + slope_b * local_b;
    scale *= output_scale;

    // === OUTPUT: scale × direction[best_tile] + residual ===
    const float* dir = &directions[best_tile * d_model];
    float* out = &output[tid * d_model];

    for (int d = 0; d < d_model; d++) {
        out[d] = x_in[d] + scale * dir[d];  // Residual connection
    }
}

// =============================================================================
// BATCHED VERSION - multiple tokens per thread for better occupancy
// =============================================================================

__global__ void sparse_lookup_ffn_batched(
    const float* x,
    const float* signatures,
    const float* spline_coeffs,
    const float* directions,
    const float* positions,
    float* output,
    int batch_size,
    int d_model,
    int num_tiles,
    int grid_size,
    float position_spread,
    float output_scale,
    int tokens_per_thread
) {
    int base_tid = (blockIdx.x * blockDim.x + threadIdx.x) * tokens_per_thread;

    for (int i = 0; i < tokens_per_thread; i++) {
        int tid = base_tid + i;
        if (tid >= batch_size) return;

        const float* x_in = &x[tid * d_model];
        float pos = positions[tid];

        // Find best tile
        float best_score = -1e30f;
        int best_tile = 0;

        for (int t = 0; t < num_tiles; t++) {
            float content_score = 0.0f;
            const float* sig = &signatures[t * d_model];

            #pragma unroll 4
            for (int d = 0; d < d_model; d++) {
                content_score += x_in[d] * sig[d];
            }

            float tile_center = (float)t / (float)num_tiles * 64.0f;
            float pos_diff = (pos - tile_center) / position_spread;
            float spatial_score = cubic_bspline(pos_diff);
            float combined = content_score * spatial_score;

            if (combined > best_score) {
                best_score = combined;
                best_tile = t;
            }
        }

        // Spline lookup
        float a = tanhf(x_in[0] * 0.1f);
        float b = tanhf(x_in[1] * 0.1f);

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

        const float* dir = &directions[best_tile * d_model];
        float* out = &output[tid * d_model];

        #pragma unroll 4
        for (int d = 0; d < d_model; d++) {
            out[d] = x_in[d] + scale * dir[d];
        }
    }
}

}  // extern "C"
'''


class HollywoodSquaresFFN:
    """
    Hollywood Squares SparseLookupFFN on GPU.

    All tiles process in parallel. Position IS computation.
    No Python loops. Pure geometric lookup.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_tiles: int = 16,
        grid_size: int = 16,
        position_spread: float = 2.0,
    ):
        print("Initializing Hollywood Squares FFN...")

        self.d_model = d_model
        self.num_tiles = num_tiles
        self.grid_size = grid_size
        self.position_spread = position_spread

        # Initialize weights on GPU
        print("  Allocating tile parameters...", end=" ", flush=True)

        # Signatures: [num_tiles, d_model] - ternary values
        self.signatures = cp.random.randn(num_tiles, d_model).astype(cp.float32)
        self.signatures = cp.sign(self.signatures)  # Ternary

        # Spline coefficients: [num_tiles, grid_size, grid_size, 3]
        self.spline_coeffs = cp.random.randn(
            num_tiles, grid_size, grid_size, 3
        ).astype(cp.float32) * 0.5

        # Directions: [num_tiles, d_model]
        self.directions = cp.random.randn(num_tiles, d_model).astype(cp.float32) * 0.02

        param_count = (
            self.signatures.size +
            self.spline_coeffs.size +
            self.directions.size
        )
        print(f"done. ({param_count:,} params)")

        # Compile kernel
        print("  Compiling CUDA kernel...", end=" ", flush=True)
        self.module = cp.RawModule(code=SPARSE_LOOKUP_KERNEL)
        self.kernel = self.module.get_function('sparse_lookup_ffn_kernel')
        self.kernel_batched = self.module.get_function('sparse_lookup_ffn_batched')
        print("done.")

        self.threads_per_block = 256
        self.output_scale = 0.1

        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        print(f"  Device: {props['name'].decode()}")
        print(f"  SMs: {props['multiProcessorCount']}")
        print("  Ready.")

    def forward(self, x: cp.ndarray, positions: cp.ndarray = None) -> cp.ndarray:
        """
        Forward pass.

        Args:
            x: [batch_size, d_model] input
            positions: [batch_size] position indices (default: 0, 1, 2, ...)

        Returns:
            output: [batch_size, d_model]
        """
        batch_size = x.shape[0]

        if positions is None:
            positions = cp.arange(batch_size, dtype=cp.float32)

        output = cp.zeros_like(x)

        blocks_per_grid = (batch_size + self.threads_per_block - 1) // self.threads_per_block

        self.kernel(
            (blocks_per_grid,), (self.threads_per_block,),
            (
                x, self.signatures, self.spline_coeffs.reshape(-1),
                self.directions, positions, output,
                batch_size, self.d_model, self.num_tiles, self.grid_size,
                self.position_spread, self.output_scale
            )
        )

        return output

    def forward_batched(self, x: cp.ndarray, positions: cp.ndarray = None, tokens_per_thread: int = 4) -> cp.ndarray:
        """Forward with multiple tokens per thread."""
        batch_size = x.shape[0]

        if positions is None:
            positions = cp.arange(batch_size, dtype=cp.float32)

        output = cp.zeros_like(x)

        effective_batch = (batch_size + tokens_per_thread - 1) // tokens_per_thread
        blocks_per_grid = (effective_batch + self.threads_per_block - 1) // self.threads_per_block

        self.kernel_batched(
            (blocks_per_grid,), (self.threads_per_block,),
            (
                x, self.signatures, self.spline_coeffs.reshape(-1),
                self.directions, positions, output,
                batch_size, self.d_model, self.num_tiles, self.grid_size,
                self.position_spread, self.output_scale, tokens_per_thread
            )
        )

        return output

    def benchmark(self, batch_size: int = 100000) -> dict:
        """Benchmark the GPU FFN."""
        cp.get_default_memory_pool().free_all_blocks()

        x = cp.random.randn(batch_size, self.d_model).astype(cp.float32)
        positions = cp.arange(batch_size, dtype=cp.float32) % 64

        # Warmup
        _ = self.forward(x, positions)
        cp.cuda.Stream.null.synchronize()

        # Benchmark standard
        times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = self.forward(x, positions)
            cp.cuda.Stream.null.synchronize()
            times.append(time.perf_counter() - start)

        best_time = min(times)
        throughput = batch_size / best_time

        # Benchmark batched
        times_batched = []
        for _ in range(5):
            start = time.perf_counter()
            _ = self.forward_batched(x, positions, tokens_per_thread=4)
            cp.cuda.Stream.null.synchronize()
            times_batched.append(time.perf_counter() - start)

        best_batched = min(times_batched)
        throughput_batched = batch_size / best_batched

        return {
            "batch_size": batch_size,
            "elapsed": best_time,
            "throughput": throughput,
            "throughput_m": throughput / 1e6,
            "elapsed_batched": best_batched,
            "throughput_batched": throughput_batched,
            "throughput_batched_m": throughput_batched / 1e6,
        }


def benchmark_pytorch_comparison():
    """Compare Hollywood Squares FFN to PyTorch version."""
    import sys
    sys.path.insert(0, '/workspace/trix_latest/TriXO/src')

    print("=" * 70)
    print("HOLLYWOOD SQUARES FFN - GPU vs PyTorch")
    print("=" * 70)
    print()

    d_model = 128
    num_tiles = 16

    # Hollywood Squares GPU
    hs_ffn = HollywoodSquaresFFN(d_model=d_model, num_tiles=num_tiles)

    # PyTorch version
    print("\nLoading PyTorch SparseLookupFFNv4...", end=" ", flush=True)
    from trix.nn.sparse_lookup_v4 import SparseLookupFFNv4
    torch_ffn = SparseLookupFFNv4(d_model=d_model, num_tiles=num_tiles).cuda()
    print("done.")

    print()
    print("Comparison:")
    print("-" * 60)

    for batch_size in [10000, 100000, 500000, 1000000]:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            torch.cuda.empty_cache()

            # Hollywood Squares GPU
            result = hs_ffn.benchmark(batch_size)

            # PyTorch
            x_torch = torch.randn(1, batch_size, d_model).cuda()

            # Warmup
            with torch.no_grad():
                _ = torch_ffn(x_torch)
            torch.cuda.synchronize()

            # Benchmark
            torch_times = []
            for _ in range(5):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = torch_ffn(x_torch)
                torch.cuda.synchronize()
                torch_times.append(time.perf_counter() - start)

            torch_best = min(torch_times)
            torch_throughput = batch_size / torch_best

            speedup = result['throughput'] / torch_throughput

            print(f"  {batch_size:>10,}: PyTorch {torch_throughput/1e6:.2f}M/s, "
                  f"HS {result['throughput_m']:.2f}M/s, "
                  f"Speedup: {speedup:.1f}×")

        except Exception as e:
            print(f"  {batch_size:>10,}: Error - {str(e)[:40]}")
            break

    # Final comparison at optimal batch
    print()
    print("=" * 70)
    print("FINAL BENCHMARK")
    print("=" * 70)

    batch_size = 500000
    cp.get_default_memory_pool().free_all_blocks()
    torch.cuda.empty_cache()

    result = hs_ffn.benchmark(batch_size)

    print(f"\nHollywood Squares FFN @ {batch_size:,} tokens:")
    print(f"  Standard:  {result['throughput_m']:.2f} M tokens/sec")
    print(f"  Batched:   {result['throughput_batched_m']:.2f} M tokens/sec")

    print()
    print("=" * 70)

    return result['throughput']


if __name__ == "__main__":
    throughput = benchmark_pytorch_comparison()
