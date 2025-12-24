"""
Hollywood Squares - POSITION IS COMPUTATION Edition

Carry propagation as geometric navigation.
Addition becomes lookup - position IS the answer.

"32 sequential carries? No. 4 lookups. Position is computation."
"""

import numpy as np
import cupy as cp
import time


# =============================================================================
# POSITION-BASED SHA-256 KERNEL
# =============================================================================

POSITION_KERNEL = r'''
extern "C" {

// SHA-256 constants
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

// =============================================================================
// ADDITION LOOKUP TABLES IN SHARED MEMORY
// Position (a, b) → sum, carry
// 256 × 256 × 2 bytes = 128 KB (fits in shared memory on modern GPUs)
// =============================================================================

// We'll use texture memory for the lookup - faster random access
// Table: add_table[a][b] = (sum & 0xFF) | ((carry & 1) << 8)
// Packed into 16-bit values

// Global lookup table (will be loaded to texture/constant memory)
__device__ unsigned short add_lut[256 * 256];

// =============================================================================
// POSITION-BASED ADDITION
// 32-bit add via 4 × 8-bit lookups
// =============================================================================

__device__ __forceinline__ unsigned int position_add(
    unsigned int a, unsigned int b,
    const unsigned short* lut
) {
    // Split into 8-bit chunks
    unsigned int a0 = a & 0xFF;
    unsigned int a1 = (a >> 8) & 0xFF;
    unsigned int a2 = (a >> 16) & 0xFF;
    unsigned int a3 = (a >> 24) & 0xFF;

    unsigned int b0 = b & 0xFF;
    unsigned int b1 = (b >> 8) & 0xFF;
    unsigned int b2 = (b >> 16) & 0xFF;
    unsigned int b3 = (b >> 24) & 0xFF;

    // Lookup byte 0 - no carry in
    unsigned short r0 = lut[a0 * 256 + b0];
    unsigned int s0 = r0 & 0xFF;
    unsigned int c0 = (r0 >> 8) & 1;

    // Lookup byte 1 - must account for carry from byte 0
    // Carry-select: compute both, select based on c0
    unsigned short r1_nc = lut[a1 * 256 + b1];
    unsigned short r1_wc = lut[(a1 * 256 + b1 + 256) & 0xFFFF];  // a1 + carry

    // Actually simpler: just add c0 to the sum
    unsigned int sum1 = (r1_nc & 0xFF) + c0;
    unsigned int s1 = sum1 & 0xFF;
    unsigned int c1 = ((r1_nc >> 8) & 1) | (sum1 >> 8);

    // Byte 2
    unsigned short r2 = lut[a2 * 256 + b2];
    unsigned int sum2 = (r2 & 0xFF) + c1;
    unsigned int s2 = sum2 & 0xFF;
    unsigned int c2 = ((r2 >> 8) & 1) | (sum2 >> 8);

    // Byte 3 (top byte, carry discarded)
    unsigned short r3 = lut[a3 * 256 + b3];
    unsigned int sum3 = (r3 & 0xFF) + c2;
    unsigned int s3 = sum3 & 0xFF;

    return s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
}

// =============================================================================
// SHA-256 HELPER FUNCTIONS
// =============================================================================

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

// =============================================================================
// SHA-256 TRANSFORM USING POSITION-BASED ADDITION
// =============================================================================

__device__ void sha256_transform_position(
    unsigned int* state,
    const unsigned int* block,
    const unsigned short* lut
) {
    unsigned int W[64];
    unsigned int a, b, c, d, e, f, g, h;
    unsigned int t1, t2;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = block[i];
    }

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        // W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16]
        unsigned int s1 = sigma1(W[i-2]);
        unsigned int s0 = sigma0(W[i-15]);
        unsigned int tmp = position_add(s1, W[i-7], lut);
        tmp = position_add(tmp, s0, lut);
        W[i] = position_add(tmp, W[i-16], lut);
    }

    a = state[0]; b = state[1]; c = state[2]; d = state[3];
    e = state[4]; f = state[5]; g = state[6]; h = state[7];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        // t1 = h + Sigma1(e) + Ch(e,f,g) + K[i] + W[i]
        unsigned int ch = Ch(e, f, g);
        unsigned int sig1 = Sigma1(e);
        t1 = position_add(h, sig1, lut);
        t1 = position_add(t1, ch, lut);
        t1 = position_add(t1, K[i], lut);
        t1 = position_add(t1, W[i], lut);

        // t2 = Sigma0(a) + Maj(a,b,c)
        unsigned int maj = Maj(a, b, c);
        unsigned int sig0 = Sigma0(a);
        t2 = position_add(sig0, maj, lut);

        h = g; g = f; f = e;
        e = position_add(d, t1, lut);
        d = c; c = b; b = a;
        a = position_add(t1, t2, lut);
    }

    state[0] = position_add(state[0], a, lut);
    state[1] = position_add(state[1], b, lut);
    state[2] = position_add(state[2], c, lut);
    state[3] = position_add(state[3], d, lut);
    state[4] = position_add(state[4], e, lut);
    state[5] = position_add(state[5], f, lut);
    state[6] = position_add(state[6], g, lut);
    state[7] = position_add(state[7], h, lut);
}

// =============================================================================
// MAIN KERNEL
// =============================================================================

__global__ void double_sha256_position(
    const unsigned int* blocks,
    unsigned int* results,
    const unsigned short* lut,
    int num_hashes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_hashes) return;

    unsigned int state[8];
    unsigned int block2[16];

    // First SHA-256
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = H0[i];

    sha256_transform_position(state, &blocks[tid * 16], lut);

    // Prepare second block
    #pragma unroll
    for (int i = 0; i < 8; i++) block2[i] = state[i];
    block2[8] = 0x80000000;
    for (int i = 9; i < 15; i++) block2[i] = 0;
    block2[15] = 256;

    // Second SHA-256
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = H0[i];

    sha256_transform_position(state, block2, lut);

    // Store result
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        results[tid * 8 + i] = state[i];
    }
}

// Standard kernel for comparison
__device__ void sha256_transform_standard(unsigned int* state, const unsigned int* block) {
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

__global__ void double_sha256_standard(
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
    sha256_transform_standard(state, &blocks[tid * 16]);

    #pragma unroll
    for (int i = 0; i < 8; i++) block2[i] = state[i];
    block2[8] = 0x80000000;
    for (int i = 9; i < 15; i++) block2[i] = 0;
    block2[15] = 256;

    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = H0[i];
    sha256_transform_standard(state, block2);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        results[tid * 8 + i] = state[i];
    }
}

}  // extern "C"
'''


class HollywoodSquaresPosition:
    """Position-based addition - carry propagation via lookup."""

    def __init__(self):
        print("Initializing Hollywood Squares POSITION...")

        # Build the addition lookup table
        print("  Building addition LUT (256×256)...", end=" ", flush=True)
        lut = np.zeros((256, 256), dtype=np.uint16)
        for a in range(256):
            for b in range(256):
                result = a + b
                lut[a, b] = (result & 0xFF) | ((result >> 8) << 8)
        self.lut_gpu = cp.asarray(lut.flatten())
        print(f"done. ({self.lut_gpu.nbytes / 1024:.1f} KB)")

        # Get device info
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        self.sm_count = props['multiProcessorCount']
        print(f"  Device: {props['name'].decode()}")
        print(f"  SMs: {self.sm_count}")

        # Compile kernel
        print("  Compiling kernels...", end=" ", flush=True)
        self.module = cp.RawModule(code=POSITION_KERNEL)
        self.kernel_position = self.module.get_function('double_sha256_position')
        self.kernel_standard = self.module.get_function('double_sha256_standard')
        print("done.")

        self.threads_per_block = 256
        print("  Ready.")

    def benchmark_standard(self, num_hashes: int) -> dict:
        """Benchmark standard kernel."""
        cp.get_default_memory_pool().free_all_blocks()

        blocks_gpu = cp.random.randint(0, 0xFFFFFFFF, size=(num_hashes, 16), dtype=cp.uint32)
        results_gpu = cp.zeros((num_hashes, 8), dtype=cp.uint32)

        blocks_per_grid = (num_hashes + self.threads_per_block - 1) // self.threads_per_block

        # Warmup
        self.kernel_standard((blocks_per_grid,), (self.threads_per_block,),
                            (blocks_gpu, results_gpu, num_hashes))
        cp.cuda.Stream.null.synchronize()

        # Benchmark
        times = []
        for _ in range(3):
            start = time.perf_counter()
            self.kernel_standard((blocks_per_grid,), (self.threads_per_block,),
                                (blocks_gpu, results_gpu, num_hashes))
            cp.cuda.Stream.null.synchronize()
            times.append(time.perf_counter() - start)

        best_time = min(times)
        return {
            "elapsed": best_time,
            "hashrate": num_hashes / best_time,
            "hashrate_mh": num_hashes / best_time / 1e6,
        }

    def benchmark_position(self, num_hashes: int) -> dict:
        """Benchmark position-based kernel."""
        cp.get_default_memory_pool().free_all_blocks()

        blocks_gpu = cp.random.randint(0, 0xFFFFFFFF, size=(num_hashes, 16), dtype=cp.uint32)
        results_gpu = cp.zeros((num_hashes, 8), dtype=cp.uint32)

        blocks_per_grid = (num_hashes + self.threads_per_block - 1) // self.threads_per_block

        # Warmup
        self.kernel_position((blocks_per_grid,), (self.threads_per_block,),
                            (blocks_gpu, results_gpu, self.lut_gpu, num_hashes))
        cp.cuda.Stream.null.synchronize()

        # Benchmark
        times = []
        for _ in range(3):
            start = time.perf_counter()
            self.kernel_position((blocks_per_grid,), (self.threads_per_block,),
                                (blocks_gpu, results_gpu, self.lut_gpu, num_hashes))
            cp.cuda.Stream.null.synchronize()
            times.append(time.perf_counter() - start)

        best_time = min(times)
        return {
            "elapsed": best_time,
            "hashrate": num_hashes / best_time,
            "hashrate_mh": num_hashes / best_time / 1e6,
        }


def benchmark_position():
    """Benchmark position-based addition."""

    print("=" * 70)
    print("HOLLYWOOD SQUARES - POSITION IS COMPUTATION")
    print("=" * 70)
    print()
    print("Architecture:")
    print("  - 8-bit addition lookup table (128 KB)")
    print("  - 32-bit add = 4 lookups + 3 carry propagations")
    print("  - Position (a,b) IS the answer - frozen shape navigation")
    print()

    fabric = HollywoodSquaresPosition()

    # Compare
    print()
    print("Comparison (10M hashes):")
    print("-" * 50)

    std_result = fabric.benchmark_standard(10_000_000)
    print(f"  Standard addition:  {std_result['hashrate_mh']:.2f} MH/s")

    pos_result = fabric.benchmark_position(10_000_000)
    print(f"  Position lookup:    {pos_result['hashrate_mh']:.2f} MH/s")

    ratio = pos_result['hashrate'] / std_result['hashrate']
    print(f"  Ratio:              {ratio:.2f}×")

    # Larger batch
    print()
    print("Scaling test:")
    print("-" * 50)

    for batch in [1_000_000, 5_000_000, 10_000_000, 25_000_000]:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            std = fabric.benchmark_standard(batch)
            pos = fabric.benchmark_position(batch)
            print(f"  {batch:>12,}: Std {std['hashrate_mh']:.1f} vs Pos {pos['hashrate_mh']:.1f} MH/s "
                  f"({pos['hashrate']/std['hashrate']:.2f}×)")
        except Exception as e:
            print(f"  {batch:>12,}: Error - {e}")
            break

    print()
    print("=" * 70)

    return pos_result['hashrate']


if __name__ == "__main__":
    benchmark_position()
