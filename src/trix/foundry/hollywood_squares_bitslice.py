"""
Hollywood Squares - BIT-SLICED Edition

The Ultimate Parallelism: Process 32 hashes as ONE operation.

Traditional: 1 hash with 256-bit state (8 × uint32)
Bit-sliced:  32 hashes where each uint32 holds bit N of all 32 hashes

XOR across 32 hashes = 1 instruction
AND across 32 hashes = 1 instruction
ADD across 32 hashes = parallel prefix (carry-lookahead)

"32 hashes for the price of 1. The fabric multiplies."
"""

import numpy as np
import cupy as cp
import time


# =============================================================================
# BIT-SLICED SHA-256 KERNEL
# =============================================================================

BITSLICE_KERNEL = r'''
extern "C" {

// SHA-256 constants - one per round
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

// Initial hash - bit-sliced (all 32 lanes have same value)
__constant__ unsigned int H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// =============================================================================
// BIT-SLICED PRIMITIVES
// =============================================================================

// XOR across 32 lanes - trivial
__device__ __forceinline__ void bs_xor32(unsigned int* r,
    const unsigned int* a, const unsigned int* b) {
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        r[i] = a[i] ^ b[i];
    }
}

// AND across 32 lanes - trivial
__device__ __forceinline__ void bs_and32(unsigned int* r,
    const unsigned int* a, const unsigned int* b) {
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        r[i] = a[i] & b[i];
    }
}

// NOT across 32 lanes
__device__ __forceinline__ void bs_not32(unsigned int* r, const unsigned int* a) {
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        r[i] = ~a[i];
    }
}

// OR across 32 lanes
__device__ __forceinline__ void bs_or32(unsigned int* r,
    const unsigned int* a, const unsigned int* b) {
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        r[i] = a[i] | b[i];
    }
}

// XOR3 across 32 lanes
__device__ __forceinline__ void bs_xor3_32(unsigned int* r,
    const unsigned int* a, const unsigned int* b, const unsigned int* c) {
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        r[i] = a[i] ^ b[i] ^ c[i];
    }
}

// =============================================================================
// BIT-SLICED ADDITION (Parallel Prefix / Carry-Lookahead)
// =============================================================================

// Add two 32-bit values across 32 lanes using carry-save then final propagate
__device__ void bs_add32(unsigned int* r,
    const unsigned int* a, const unsigned int* b) {

    // Ripple carry across bit positions (unavoidable sequential dependency)
    // But each bit operation processes 32 hashes in parallel!

    unsigned int carry = 0;

    #pragma unroll
    for (int bit = 0; bit < 32; bit++) {
        unsigned int a_bit = a[bit];
        unsigned int b_bit = b[bit];

        // Full adder: sum = a ^ b ^ carry, carry_out = (a & b) | (carry & (a ^ b))
        unsigned int sum = a_bit ^ b_bit ^ carry;
        unsigned int carry_out = (a_bit & b_bit) | (carry & (a_bit ^ b_bit));

        r[bit] = sum;
        carry = carry_out;
    }
}

// Add constant to bit-sliced value
__device__ void bs_add_const32(unsigned int* r,
    const unsigned int* a, unsigned int constant) {

    unsigned int carry = 0;

    #pragma unroll
    for (int bit = 0; bit < 32; bit++) {
        unsigned int a_bit = a[bit];
        unsigned int c_bit = (constant >> bit) & 1 ? 0xFFFFFFFF : 0;

        unsigned int sum = a_bit ^ c_bit ^ carry;
        unsigned int carry_out = (a_bit & c_bit) | (carry & (a_bit ^ c_bit));

        r[bit] = sum;
        carry = carry_out;
    }
}

// =============================================================================
// BIT-SLICED ROTATE
// =============================================================================

// Rotate right - just reindex the bit positions
__device__ __forceinline__ void bs_rotr32(unsigned int* r,
    const unsigned int* a, int n) {
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        r[i] = a[(i + n) & 31];
    }
}

// Shift right - zero-fill high bits
__device__ __forceinline__ void bs_shr32(unsigned int* r,
    const unsigned int* a, int n) {
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int src = i + n;
        r[i] = (src < 32) ? a[src] : 0;
    }
}

// =============================================================================
// BIT-SLICED SHA-256 FUNCTIONS
// =============================================================================

// Ch(x, y, z) = (x & y) ^ (~x & z)
__device__ void bs_Ch(unsigned int* r,
    const unsigned int* x, const unsigned int* y, const unsigned int* z) {

    unsigned int t1[32], t2[32], nx[32];
    bs_and32(t1, x, y);      // x & y
    bs_not32(nx, x);         // ~x
    bs_and32(t2, nx, z);     // ~x & z
    bs_xor32(r, t1, t2);     // (x & y) ^ (~x & z)
}

// Maj(x, y, z) = (x & y) ^ (x & z) ^ (y & z)
__device__ void bs_Maj(unsigned int* r,
    const unsigned int* x, const unsigned int* y, const unsigned int* z) {

    unsigned int t1[32], t2[32], t3[32];
    bs_and32(t1, x, y);
    bs_and32(t2, x, z);
    bs_and32(t3, y, z);
    bs_xor3_32(r, t1, t2, t3);
}

// Sigma0(x) = rotr(x,2) ^ rotr(x,13) ^ rotr(x,22)
__device__ void bs_Sigma0(unsigned int* r, const unsigned int* x) {
    unsigned int t1[32], t2[32], t3[32];
    bs_rotr32(t1, x, 2);
    bs_rotr32(t2, x, 13);
    bs_rotr32(t3, x, 22);
    bs_xor3_32(r, t1, t2, t3);
}

// Sigma1(x) = rotr(x,6) ^ rotr(x,11) ^ rotr(x,25)
__device__ void bs_Sigma1(unsigned int* r, const unsigned int* x) {
    unsigned int t1[32], t2[32], t3[32];
    bs_rotr32(t1, x, 6);
    bs_rotr32(t2, x, 11);
    bs_rotr32(t3, x, 25);
    bs_xor3_32(r, t1, t2, t3);
}

// sigma0(x) = rotr(x,7) ^ rotr(x,18) ^ shr(x,3)
__device__ void bs_sigma0(unsigned int* r, const unsigned int* x) {
    unsigned int t1[32], t2[32], t3[32];
    bs_rotr32(t1, x, 7);
    bs_rotr32(t2, x, 18);
    bs_shr32(t3, x, 3);
    bs_xor3_32(r, t1, t2, t3);
}

// sigma1(x) = rotr(x,17) ^ rotr(x,19) ^ shr(x,10)
__device__ void bs_sigma1(unsigned int* r, const unsigned int* x) {
    unsigned int t1[32], t2[32], t3[32];
    bs_rotr32(t1, x, 17);
    bs_rotr32(t2, x, 19);
    bs_shr32(t3, x, 10);
    bs_xor3_32(r, t1, t2, t3);
}

// =============================================================================
// BIT-SLICED SHA-256 TRANSFORM
// =============================================================================

// Transform 32 message blocks simultaneously
__device__ void bs_sha256_transform(
    unsigned int state[8][32],      // 8 words × 32 bits, each bit holds 32 hashes
    const unsigned int block[16][32] // 16 words × 32 bits
) {
    // Message schedule
    unsigned int W[64][32];

    // Load first 16 words
    for (int i = 0; i < 16; i++) {
        for (int b = 0; b < 32; b++) {
            W[i][b] = block[i][b];
        }
    }

    // Expand to 64 words
    for (int i = 16; i < 64; i++) {
        unsigned int s0[32], s1[32], t1[32], t2[32];

        bs_sigma1(s1, W[i-2]);
        bs_sigma0(s0, W[i-15]);

        // W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16]
        bs_add32(t1, s1, W[i-7]);
        bs_add32(t2, s0, W[i-16]);
        bs_add32(W[i], t1, t2);
    }

    // Working variables
    unsigned int a[32], b[32], c[32], d[32];
    unsigned int e[32], f[32], g[32], h[32];

    for (int bit = 0; bit < 32; bit++) {
        a[bit] = state[0][bit];
        b[bit] = state[1][bit];
        c[bit] = state[2][bit];
        d[bit] = state[3][bit];
        e[bit] = state[4][bit];
        f[bit] = state[5][bit];
        g[bit] = state[6][bit];
        h[bit] = state[7][bit];
    }

    // 64 rounds
    for (int i = 0; i < 64; i++) {
        unsigned int S1[32], ch[32], t1[32], t2[32];
        unsigned int S0[32], maj[32];

        bs_Sigma1(S1, e);
        bs_Ch(ch, e, f, g);

        // t1 = h + S1 + ch + K[i] + W[i]
        bs_add32(t1, h, S1);
        bs_add32(t1, t1, ch);
        bs_add_const32(t1, t1, K[i]);
        bs_add32(t1, t1, W[i]);

        bs_Sigma0(S0, a);
        bs_Maj(maj, a, b, c);

        // t2 = S0 + maj
        bs_add32(t2, S0, maj);

        // Rotate working variables
        for (int bit = 0; bit < 32; bit++) {
            h[bit] = g[bit];
            g[bit] = f[bit];
            f[bit] = e[bit];
        }
        bs_add32(e, d, t1);  // e = d + t1

        for (int bit = 0; bit < 32; bit++) {
            d[bit] = c[bit];
            c[bit] = b[bit];
            b[bit] = a[bit];
        }
        bs_add32(a, t1, t2);  // a = t1 + t2
    }

    // Add to state
    unsigned int t[32];
    bs_add32(t, state[0], a); for (int b = 0; b < 32; b++) state[0][b] = t[b];
    bs_add32(t, state[1], b); for (int bb = 0; bb < 32; bb++) state[1][bb] = t[bb];
    bs_add32(t, state[2], c); for (int b = 0; b < 32; b++) state[2][b] = t[b];
    bs_add32(t, state[3], d); for (int b = 0; b < 32; b++) state[3][b] = t[b];
    bs_add32(t, state[4], e); for (int b = 0; b < 32; b++) state[4][b] = t[b];
    bs_add32(t, state[5], f); for (int b = 0; b < 32; b++) state[5][b] = t[b];
    bs_add32(t, state[6], g); for (int b = 0; b < 32; b++) state[6][b] = t[b];
    bs_add32(t, state[7], h); for (int b = 0; b < 32; b++) state[7][b] = t[b];
}

// =============================================================================
// MAIN KERNEL: Process 32 hashes per thread
// =============================================================================

__global__ void double_sha256_bitslice(
    const unsigned int* blocks,  // [num_groups, 32, 16] - 32 hashes × 16 words
    unsigned int* results,       // [num_groups, 32, 8] - 32 hashes × 8 words
    int num_groups               // Number of 32-hash groups
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_groups) return;

    // Load 32 message blocks into bit-sliced form
    unsigned int block[16][32];  // 16 words × 32 bits

    for (int word = 0; word < 16; word++) {
        for (int bit = 0; bit < 32; bit++) {
            unsigned int mask = 0;
            for (int hash = 0; hash < 32; hash++) {
                unsigned int val = blocks[(gid * 32 + hash) * 16 + word];
                if ((val >> bit) & 1) {
                    mask |= (1u << hash);
                }
            }
            block[word][bit] = mask;
        }
    }

    // Initialize state (bit-sliced, same for all 32 hashes)
    unsigned int state[8][32];
    for (int word = 0; word < 8; word++) {
        for (int bit = 0; bit < 32; bit++) {
            // All 32 hashes start with same H0 value
            state[word][bit] = ((H0[word] >> bit) & 1) ? 0xFFFFFFFF : 0;
        }
    }

    // First SHA-256
    bs_sha256_transform(state, block);

    // Prepare second block: state || padding
    for (int word = 0; word < 8; word++) {
        for (int bit = 0; bit < 32; bit++) {
            block[word][bit] = state[word][bit];
        }
    }
    // Padding
    for (int bit = 0; bit < 32; bit++) {
        block[8][bit] = (bit == 31) ? 0xFFFFFFFF : 0;  // 0x80000000
        block[9][bit] = 0;
        block[10][bit] = 0;
        block[11][bit] = 0;
        block[12][bit] = 0;
        block[13][bit] = 0;
        block[14][bit] = 0;
        block[15][bit] = (bit == 8) ? 0xFFFFFFFF : 0;  // 256
    }

    // Reset state
    for (int word = 0; word < 8; word++) {
        for (int bit = 0; bit < 32; bit++) {
            state[word][bit] = ((H0[word] >> bit) & 1) ? 0xFFFFFFFF : 0;
        }
    }

    // Second SHA-256
    bs_sha256_transform(state, block);

    // Extract results (un-bitslice)
    for (int hash = 0; hash < 32; hash++) {
        for (int word = 0; word < 8; word++) {
            unsigned int val = 0;
            for (int bit = 0; bit < 32; bit++) {
                if ((state[word][bit] >> hash) & 1) {
                    val |= (1u << bit);
                }
            }
            results[(gid * 32 + hash) * 8 + word] = val;
        }
    }
}

}  // extern "C"
'''


# =============================================================================
# HOLLYWOOD SQUARES BITSLICE
# =============================================================================

class HollywoodSquaresBitslice:
    """
    Bit-sliced Hollywood Squares.

    Each thread processes 32 hashes simultaneously.
    XOR/AND/OR across 32 hashes = single instruction.
    """

    def __init__(self):
        print("Initializing Hollywood Squares BITSLICE...")

        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        print(f"  Device: {props['name'].decode()}")
        print(f"  SMs: {props['multiProcessorCount']}")

        print("  Compiling bit-sliced kernel...", end=" ", flush=True)
        self.module = cp.RawModule(code=BITSLICE_KERNEL)
        self.kernel = self.module.get_function('double_sha256_bitslice')
        print("done.")

        self.threads_per_block = 64  # Lower due to high register usage
        print(f"  Hashes per thread: 32")
        print(f"  Threads per block: {self.threads_per_block}")
        print("  Ready.")

    def benchmark(self, num_hashes: int = 1000000) -> dict:
        """Benchmark bit-sliced hashing."""
        # Round up to multiple of 32
        num_groups = (num_hashes + 31) // 32
        actual_hashes = num_groups * 32

        cp.get_default_memory_pool().free_all_blocks()

        # Allocate
        blocks_gpu = cp.random.randint(
            0, 0xFFFFFFFF,
            size=(actual_hashes, 16),
            dtype=cp.uint32
        )
        results_gpu = cp.zeros((actual_hashes, 8), dtype=cp.uint32)

        blocks_per_grid = (num_groups + self.threads_per_block - 1) // self.threads_per_block

        # Warmup
        self.kernel(
            (blocks_per_grid,), (self.threads_per_block,),
            (blocks_gpu, results_gpu, num_groups)
        )
        cp.cuda.Stream.null.synchronize()

        # Benchmark
        start = time.perf_counter()
        self.kernel(
            (blocks_per_grid,), (self.threads_per_block,),
            (blocks_gpu, results_gpu, num_groups)
        )
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start

        hashrate = actual_hashes / elapsed

        return {
            "num_hashes": actual_hashes,
            "num_groups": num_groups,
            "elapsed": elapsed,
            "hashrate": hashrate,
            "hashrate_mh": hashrate / 1e6,
        }


def benchmark_bitslice():
    """Benchmark bit-sliced implementation."""

    print("=" * 70)
    print("HOLLYWOOD SQUARES - BITSLICE EDITION")
    print("=" * 70)
    print()
    print("Architecture:")
    print("  - Each thread processes 32 hashes simultaneously")
    print("  - XOR/AND/OR across 32 hashes = 1 instruction")
    print("  - Addition uses parallel ripple carry")
    print("  - 32× theoretical throughput improvement")
    print()

    fabric = HollywoodSquaresBitslice()

    print()
    print("Batch scaling:")
    print("-" * 50)

    for num_hashes in [32000, 320000, 1000000, 3200000]:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            result = fabric.benchmark(num_hashes)
            print(f"  {result['num_hashes']:>10,}: {result['hashrate_mh']:.2f} MH/s "
                  f"({result['num_groups']:,} groups)")
        except Exception as e:
            print(f"  {num_hashes:>10,}: Error - {e}")
            break

    # Comparison
    print()
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    try:
        cp.get_default_memory_pool().free_all_blocks()
        result = fabric.benchmark(3200000)

        standard = 781_000_000  # Previous best
        bitslice = result['hashrate']

        print(f"  Standard kernel: {standard/1e6:.2f} MH/s")
        print(f"  Bitslice kernel: {bitslice/1e6:.2f} MH/s")
        print(f"  Ratio: {bitslice/standard:.2f}×")

    except Exception as e:
        print(f"  Error: {e}")

    print()
    print("=" * 70)

    return result['hashrate'] if 'result' in dir() else 0


if __name__ == "__main__":
    hashrate = benchmark_bitslice()
