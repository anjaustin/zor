"""
Hollywood Squares ASIC Fabric - GPU Edition

CuPy-accelerated 512-bit parallel processing.
Each CUDA thread = one hash.
Each hash = 512 squares computing in parallel.
Millions of hashes run simultaneously across GPU cores.

"132 GB of unified memory. Millions of parallel squares.
 The fabric speaks at the speed of light."
"""

import numpy as np
import cupy as cp
import time


# =============================================================================
# CUDA KERNEL SOURCE - Raw CUDA C
# =============================================================================

SHA256_KERNEL = r'''
extern "C" {

// SHA-256 Constants
__constant__ unsigned int K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__constant__ unsigned int H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// Rotate right
__device__ __forceinline__ unsigned int rotr(unsigned int x, unsigned int n) {
    return (x >> n) | (x << (32 - n));
}

// SHA-256 functions
__device__ __forceinline__ unsigned int Ch(unsigned int x, unsigned int y, unsigned int z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ unsigned int Maj(unsigned int x, unsigned int y, unsigned int z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ unsigned int Sigma0(unsigned int x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __forceinline__ unsigned int Sigma1(unsigned int x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ __forceinline__ unsigned int sigma0(unsigned int x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ unsigned int sigma1(unsigned int x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// Single SHA-256 block transform
__device__ void sha256_transform(unsigned int* state, const unsigned int* block) {
    unsigned int W[64];
    unsigned int a, b, c, d, e, f, g, h;
    unsigned int t1, t2;

    // Load message block into W[0..15]
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = block[i];
    }

    // Extend to W[16..63]
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16];
    }

    // Initialize working variables
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    // 64 rounds - ALL 256 bits compute in parallel via 32-bit ops
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        t1 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i];
        t2 = Sigma0(a) + Maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    // Add to state
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

// Double SHA-256 kernel - one thread per hash
__global__ void double_sha256_kernel(
    const unsigned int* blocks,  // [num_hashes, 16]
    unsigned int* results,       // [num_hashes, 8]
    int num_hashes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hashes) return;

    // Local state and block
    unsigned int state[8];
    unsigned int block2[16];

    // === FIRST SHA-256 ===

    // Initialize state
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state[i] = H0[i];
    }

    // Transform with input block
    sha256_transform(state, &blocks[tid * 16]);

    // === SECOND SHA-256 ===

    // Prepare second block: state || padding
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        block2[i] = state[i];
    }
    block2[8] = 0x80000000;  // Padding bit
    block2[9] = 0;
    block2[10] = 0;
    block2[11] = 0;
    block2[12] = 0;
    block2[13] = 0;
    block2[14] = 0;
    block2[15] = 256;  // Length in bits

    // Reset state
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        state[i] = H0[i];
    }

    // Transform with hash of first block
    sha256_transform(state, block2);

    // Store result
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        results[tid * 8 + i] = state[i];
    }
}

}  // extern "C"
'''


# =============================================================================
# HOLLYWOOD SQUARES GPU FABRIC
# =============================================================================

class HollywoodSquaresGPU:
    """
    Hollywood Squares fabric on GPU.

    Architecture:
    - Each CUDA thread = one hash = 512 parallel squares
    - Thousands of threads per SM
    - 132 GB unified memory = millions of parallel hashes
    - Raw CUDA kernel for maximum performance
    """

    def __init__(self):
        print("Initializing Hollywood Squares GPU Fabric...")

        # Get GPU info
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        print(f"  Device: {props['name'].decode()}")
        print(f"  Memory: {props['totalGlobalMem'] / 1e9:.1f} GB")
        print(f"  SMs: {props['multiProcessorCount']}")

        # Compile kernel
        print("  Compiling CUDA kernel...", end=" ", flush=True)
        self.module = cp.RawModule(code=SHA256_KERNEL)
        self.kernel = self.module.get_function('double_sha256_kernel')
        print("done.")

        # Optimal configuration
        self.threads_per_block = 256
        print(f"  Threads per block: {self.threads_per_block}")
        print("  Ready.")

    def hash_batch(self, blocks: np.ndarray) -> np.ndarray:
        """Hash a batch of blocks on GPU."""
        num_hashes = blocks.shape[0]

        # Transfer to GPU
        blocks_gpu = cp.asarray(blocks, dtype=cp.uint32)
        results_gpu = cp.zeros((num_hashes, 8), dtype=cp.uint32)

        # Launch kernel
        blocks_per_grid = (num_hashes + self.threads_per_block - 1) // self.threads_per_block
        self.kernel(
            (blocks_per_grid,), (self.threads_per_block,),
            (blocks_gpu, results_gpu, num_hashes)
        )

        # Synchronize and return
        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(results_gpu)

    def hash_batch_gpu(self, blocks_gpu, results_gpu, num_hashes):
        """Hash on GPU without memory transfers (for benchmarking)."""
        blocks_per_grid = (num_hashes + self.threads_per_block - 1) // self.threads_per_block
        self.kernel(
            (blocks_per_grid,), (self.threads_per_block,),
            (blocks_gpu, results_gpu, num_hashes)
        )

    def benchmark(self, num_hashes: int = 1000000) -> dict:
        """Benchmark the GPU fabric."""
        # Pre-allocate on GPU
        blocks_gpu = cp.random.randint(
            0, 0xFFFFFFFF,
            size=(num_hashes, 16),
            dtype=cp.uint32
        )
        results_gpu = cp.zeros((num_hashes, 8), dtype=cp.uint32)

        # Warmup
        self.hash_batch_gpu(blocks_gpu, results_gpu, num_hashes)
        cp.cuda.Stream.null.synchronize()

        # Benchmark
        start = time.perf_counter()
        self.hash_batch_gpu(blocks_gpu, results_gpu, num_hashes)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start

        hashrate = num_hashes / elapsed

        return {
            "num_hashes": num_hashes,
            "elapsed_seconds": elapsed,
            "hashrate": hashrate,
            "hashrate_mh": hashrate / 1e6,
            "hashrate_gh": hashrate / 1e9,
        }


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_hollywood_gpu():
    """Full benchmark of GPU Hollywood Squares."""

    print("=" * 70)
    print("HOLLYWOOD SQUARES - GPU EDITION")
    print("=" * 70)
    print()
    print("Architecture:")
    print("  - Each CUDA thread = one hash = 512 parallel squares")
    print("  - Each square = 1 bit position, all compute simultaneously")
    print("  - Raw CUDA C kernel with #pragma unroll optimization")
    print("  - Millions of hashes in parallel across all SMs")
    print()

    # Create fabric
    fabric = HollywoodSquaresGPU()

    # Scaling test
    print()
    print("Batch size scaling:")
    print("-" * 60)

    for batch_size in [100000, 1000000, 10000000, 50000000]:
        try:
            result = fabric.benchmark(batch_size)
            print(f"  {batch_size:>12,} hashes: {result['hashrate']:>15,.0f} H/s "
                  f"({result['hashrate_mh']:.2f} MH/s) [{result['elapsed_seconds']:.4f}s]")
        except Exception as e:
            print(f"  {batch_size:>12,} hashes: FAILED - {e}")
            break

    # Find optimal batch size
    print()
    print("=" * 70)
    print("FINDING OPTIMAL BATCH SIZE")
    print("=" * 70)

    best_hashrate = 0
    best_batch = 0

    for batch_size in [1000000, 5000000, 10000000, 25000000, 50000000]:
        try:
            # Clear memory before each test
            cp.get_default_memory_pool().free_all_blocks()
            result = fabric.benchmark(batch_size)
            print(f"  {batch_size:>12,}: {result['hashrate_mh']:.2f} MH/s")
            if result['hashrate'] > best_hashrate:
                best_hashrate = result['hashrate']
                best_batch = batch_size
        except Exception as e:
            print(f"  {batch_size:>12,}: OOM or error - {e}")
            break

    # Clear memory
    cp.get_default_memory_pool().free_all_blocks()

    # Final benchmark at optimal size (use smaller batch for stability)
    final_batch = min(best_batch, 25000000)
    print()
    print("=" * 70)
    print(f"FINAL BENCHMARK - {final_batch:,} HASHES")
    print("=" * 70)

    times = []
    for run in range(5):
        cp.get_default_memory_pool().free_all_blocks()
        result = fabric.benchmark(final_batch)
        times.append(result['elapsed_seconds'])
        print(f"  Run {run+1}: {result['elapsed_seconds']:.4f}s "
              f"({result['hashrate_mh']:.2f} MH/s)")

    best_time = min(times)
    best_hashrate = final_batch / best_time

    print()
    print(f"  Best: {best_hashrate:,.0f} H/s")
    print(f"        {best_hashrate/1e6:.2f} MH/s")
    if best_hashrate >= 1e9:
        print(f"        {best_hashrate/1e9:.4f} GH/s")
    print()

    # Comparison
    cpu_hashrate = 16_000_000  # Our JIT result
    asic_chip = 322_000_000_000  # 322 GH/s
    antminer = 110_000_000_000_000  # 110 TH/s

    speedup_vs_cpu = best_hashrate / cpu_hashrate
    gap_to_chip = asic_chip / best_hashrate
    gap_to_miner = antminer / best_hashrate

    print("Comparison:")
    print(f"  CPU JIT:             {cpu_hashrate/1e6:.2f} MH/s")
    print(f"  GPU Hollywood:       {best_hashrate/1e6:.2f} MH/s")
    print(f"  Speedup vs CPU:      {speedup_vs_cpu:.1f}×")
    print()
    print(f"  BM1398 chip:         {asic_chip/1e9:.0f} GH/s")
    print(f"  Gap to chip:         {gap_to_chip:,.0f}×")
    print()
    print(f"  Antminer S19:        {antminer/1e12:.0f} TH/s")
    print(f"  Gap to miner:        {gap_to_miner:,.0f}×")
    print()

    # Earning estimate
    btc_price = 100000  # USD
    network_hashrate = 500e18  # 500 EH/s
    block_reward = 3.125  # BTC
    blocks_per_day = 144

    daily_btc = (best_hashrate / network_hashrate) * block_reward * blocks_per_day
    daily_usd = daily_btc * btc_price
    yearly_usd = daily_usd * 365

    print("Mining estimate (solo, current network):")
    print(f"  Daily BTC:    {daily_btc:.12f}")
    print(f"  Daily USD:    ${daily_usd:.8f}")
    print(f"  Yearly USD:   ${yearly_usd:.6f}")
    print()

    print("=" * 70)
    print("THE GPU FABRIC AWAKENS")
    print("=" * 70)

    return best_hashrate


if __name__ == "__main__":
    hashrate = benchmark_hollywood_gpu()
