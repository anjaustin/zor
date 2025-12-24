"""
Hollywood Squares ASIC Fabric - TUNED Edition

Systematic tuning of thread/block configuration.
Finding the optimal occupancy for Thor's SMs.

"20 SMs. 256 threads each. 5,120 simultaneous hashes.
 But what if we go wider?"
"""

import numpy as np
import cupy as cp
import time


# Same proven kernel - we'll tune the launch config
SHA256_KERNEL = r'''
extern "C" {

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

__device__ __forceinline__ unsigned int rotr(unsigned int x, unsigned int n) {
    return (x >> n) | (x << (32 - n));
}

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

__device__ void sha256_transform(unsigned int* state, const unsigned int* block) {
    unsigned int W[64];
    unsigned int a, b, c, d, e, f, g, h;
    unsigned int t1, t2;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = block[i];
    }

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16];
    }

    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        t1 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i];
        t2 = Sigma0(a) + Maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__global__ void double_sha256_kernel(
    const unsigned int* blocks,
    unsigned int* results,
    int num_hashes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hashes) return;

    unsigned int state[8];
    unsigned int block2[16];

    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = H0[i];

    sha256_transform(state, &blocks[tid * 16]);

    #pragma unroll
    for (int i = 0; i < 8; i++) block2[i] = state[i];
    block2[8] = 0x80000000;
    for (int i = 9; i < 15; i++) block2[i] = 0;
    block2[15] = 256;

    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = H0[i];

    sha256_transform(state, block2);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        results[tid * 8 + i] = state[i];
    }
}

}
'''


class HollywoodSquaresTuned:
    """Tuned Hollywood Squares - finding optimal thread configuration."""

    def __init__(self):
        print("Initializing Hollywood Squares TUNED...")

        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        self.sm_count = props['multiProcessorCount']
        self.max_threads = props['maxThreadsPerBlock']

        print(f"  Device: {props['name'].decode()}")
        print(f"  SMs: {self.sm_count}")
        print(f"  Max threads/block: {self.max_threads}")

        print("  Compiling kernel...", end=" ", flush=True)
        self.module = cp.RawModule(code=SHA256_KERNEL)
        self.kernel = self.module.get_function('double_sha256_kernel')
        print("done.")

    def benchmark_config(self, num_hashes: int, threads_per_block: int) -> dict:
        """Benchmark with specific thread configuration."""
        cp.get_default_memory_pool().free_all_blocks()

        blocks_gpu = cp.random.randint(
            0, 0xFFFFFFFF, size=(num_hashes, 16), dtype=cp.uint32
        )
        results_gpu = cp.zeros((num_hashes, 8), dtype=cp.uint32)

        blocks_per_grid = (num_hashes + threads_per_block - 1) // threads_per_block

        # Warmup
        self.kernel((blocks_per_grid,), (threads_per_block,),
                   (blocks_gpu, results_gpu, num_hashes))
        cp.cuda.Stream.null.synchronize()

        # Benchmark - multiple runs
        times = []
        for _ in range(3):
            start = time.perf_counter()
            self.kernel((blocks_per_grid,), (threads_per_block,),
                       (blocks_gpu, results_gpu, num_hashes))
            cp.cuda.Stream.null.synchronize()
            times.append(time.perf_counter() - start)

        best_time = min(times)
        hashrate = num_hashes / best_time

        return {
            "threads": threads_per_block,
            "blocks": blocks_per_grid,
            "elapsed": best_time,
            "hashrate": hashrate,
            "hashrate_mh": hashrate / 1e6,
        }


def benchmark_tuned():
    """Find optimal thread configuration."""

    print("=" * 70)
    print("HOLLYWOOD SQUARES - TUNING")
    print("=" * 70)
    print()

    fabric = HollywoodSquaresTuned()

    # Test different thread counts
    print()
    print("Thread configuration sweep (25M hashes):")
    print("-" * 60)
    print(f"{'Threads':>8} {'Blocks':>10} {'MH/s':>12} {'Δ from 256':>12}")
    print("-" * 60)

    results = []
    baseline = None

    for threads in [64, 128, 192, 256, 320, 384, 448, 512]:
        try:
            result = fabric.benchmark_config(25000000, threads)
            results.append(result)

            if threads == 256:
                baseline = result['hashrate']

            delta = ""
            if baseline:
                pct = (result['hashrate'] / baseline - 1) * 100
                delta = f"{pct:+.1f}%"

            print(f"{threads:>8} {result['blocks']:>10,} {result['hashrate_mh']:>12.2f} {delta:>12}")

        except Exception as e:
            print(f"{threads:>8} {'ERROR':>10} {str(e)[:30]}")

    # Find best
    if results:
        best = max(results, key=lambda x: x['hashrate'])
        print()
        print(f"OPTIMAL: {best['threads']} threads/block → {best['hashrate_mh']:.2f} MH/s")

    # Now test batch sizes with optimal config
    print()
    print("=" * 70)
    print("BATCH SIZE SWEEP (optimal threads)")
    print("=" * 70)

    optimal_threads = best['threads'] if results else 256

    for batch in [5000000, 10000000, 25000000, 50000000]:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            result = fabric.benchmark_config(batch, optimal_threads)
            print(f"  {batch:>12,}: {result['hashrate_mh']:.2f} MH/s")
        except:
            print(f"  {batch:>12,}: OOM")
            break

    # Final benchmark
    print()
    print("=" * 70)
    print("FINAL BENCHMARK")
    print("=" * 70)

    best_hashrate = 0
    for _ in range(5):
        cp.get_default_memory_pool().free_all_blocks()
        result = fabric.benchmark_config(25000000, optimal_threads)
        if result['hashrate'] > best_hashrate:
            best_hashrate = result['hashrate']
        print(f"  Run: {result['hashrate_mh']:.2f} MH/s")

    print()
    print(f"  BEST: {best_hashrate/1e6:.2f} MH/s")

    # Comparison
    prev = 778_000_000
    improvement = (best_hashrate / prev - 1) * 100

    print()
    print("Comparison:")
    print(f"  Previous (256 threads): 778.74 MH/s")
    print(f"  Tuned ({optimal_threads} threads):    {best_hashrate/1e6:.2f} MH/s")
    print(f"  Improvement:            {improvement:+.1f}%")

    print()
    print("=" * 70)

    return best_hashrate


if __name__ == "__main__":
    hashrate = benchmark_tuned()
