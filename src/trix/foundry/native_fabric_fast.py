"""
Native ASIC Fabric - FAST VERSION

Vectorized NumPy operations for true parallelism.
Processes thousands of hashes simultaneously via SIMD.
"""

import numpy as np
from typing import Dict, Tuple
import time


# =============================================================================
# SHA-256 CONSTANTS (as numpy arrays for broadcasting)
# =============================================================================

K = np.array([
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
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
], dtype=np.uint32)

H0 = np.array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
], dtype=np.uint32)


# =============================================================================
# VECTORIZED NATIVE SHAPES
# =============================================================================

def rotr32(x: np.ndarray, n: int) -> np.ndarray:
    """Vectorized rotate right - operates on [batch] of uint32."""
    return ((x >> n) | (x << (32 - n))).astype(np.uint32)


def ch(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Vectorized Ch function."""
    return ((x & y) ^ ((~x) & z)).astype(np.uint32)


def maj(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Vectorized Maj function."""
    return ((x & y) ^ (x & z) ^ (y & z)).astype(np.uint32)


def sigma0(x: np.ndarray) -> np.ndarray:
    """Vectorized Σ0."""
    return (rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22)).astype(np.uint32)


def sigma1(x: np.ndarray) -> np.ndarray:
    """Vectorized Σ1."""
    return (rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25)).astype(np.uint32)


def gamma0(x: np.ndarray) -> np.ndarray:
    """Vectorized σ0."""
    return (rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3)).astype(np.uint32)


def gamma1(x: np.ndarray) -> np.ndarray:
    """Vectorized σ1."""
    return (rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10)).astype(np.uint32)


# =============================================================================
# VECTORIZED SHA-256 - BATCH PROCESSING
# =============================================================================

def sha256_batch(blocks: np.ndarray) -> np.ndarray:
    """
    Compute SHA-256 for a batch of blocks simultaneously.

    Args:
        blocks: [batch, 16] array of uint32 (512-bit blocks)

    Returns:
        [batch, 8] array of uint32 (256-bit hashes)
    """
    batch_size = blocks.shape[0]

    # Initialize hash state for all blocks
    H = np.tile(H0, (batch_size, 1))  # [batch, 8]

    # Message schedule W[batch, 64]
    W = np.zeros((batch_size, 64), dtype=np.uint32)
    W[:, :16] = blocks

    # Expand message schedule
    for i in range(16, 64):
        W[:, i] = (
            gamma1(W[:, i-2]) +
            W[:, i-7] +
            gamma0(W[:, i-15]) +
            W[:, i-16]
        ).astype(np.uint32)

    # Working variables [batch]
    a = H[:, 0].copy()
    b = H[:, 1].copy()
    c = H[:, 2].copy()
    d = H[:, 3].copy()
    e = H[:, 4].copy()
    f = H[:, 5].copy()
    g = H[:, 6].copy()
    h = H[:, 7].copy()

    # 64 rounds
    for i in range(64):
        S1 = sigma1(e)
        ch_val = ch(e, f, g)
        t1 = (h.astype(np.uint64) + S1 + ch_val + K[i] + W[:, i]).astype(np.uint32)

        S0 = sigma0(a)
        maj_val = maj(a, b, c)
        t2 = (S0.astype(np.uint64) + maj_val).astype(np.uint32)

        h = g
        g = f
        f = e
        e = (d.astype(np.uint64) + t1).astype(np.uint32)
        d = c
        c = b
        b = a
        a = (t1.astype(np.uint64) + t2).astype(np.uint32)

    # Add to initial hash
    result = np.zeros((batch_size, 8), dtype=np.uint32)
    result[:, 0] = (H[:, 0].astype(np.uint64) + a).astype(np.uint32)
    result[:, 1] = (H[:, 1].astype(np.uint64) + b).astype(np.uint32)
    result[:, 2] = (H[:, 2].astype(np.uint64) + c).astype(np.uint32)
    result[:, 3] = (H[:, 3].astype(np.uint64) + d).astype(np.uint32)
    result[:, 4] = (H[:, 4].astype(np.uint64) + e).astype(np.uint32)
    result[:, 5] = (H[:, 5].astype(np.uint64) + f).astype(np.uint32)
    result[:, 6] = (H[:, 6].astype(np.uint64) + g).astype(np.uint32)
    result[:, 7] = (H[:, 7].astype(np.uint64) + h).astype(np.uint32)

    return result


def double_sha256_batch(blocks: np.ndarray) -> np.ndarray:
    """
    Double SHA-256 (Bitcoin style) for a batch.

    Args:
        blocks: [batch, 16] array of uint32

    Returns:
        [batch, 8] array of uint32 (256-bit hashes)
    """
    # First hash
    H1 = sha256_batch(blocks)

    # Prepare second block: H1 || padding
    batch_size = blocks.shape[0]
    blocks2 = np.zeros((batch_size, 16), dtype=np.uint32)
    blocks2[:, :8] = H1
    blocks2[:, 8] = 0x80000000  # Padding bit
    blocks2[:, 15] = 256  # Length

    # Second hash
    H2 = sha256_batch(blocks2)

    return H2


# =============================================================================
# ASIC FABRIC LAYER
# =============================================================================

class VectorizedFabric:
    """
    ASIC fabric using vectorized operations.
    Each "square" is a lane in the SIMD vector.
    """

    def __init__(self, num_squares: int = 10240):
        self.num_squares = num_squares
        print(f"Creating Vectorized ASIC Fabric...")
        print(f"  Virtual squares: {num_squares:,}")
        print(f"  (Each square is a SIMD lane)")

    def hash_batch(self, num_hashes: int) -> np.ndarray:
        """Generate random blocks and hash them."""
        # Random blocks
        blocks = np.random.randint(
            0, 0xFFFFFFFF,
            size=(num_hashes, 16),
            dtype=np.uint32
        )

        # Hash all at once
        return double_sha256_batch(blocks)

    def benchmark(self, num_hashes: int = 100000) -> Dict:
        """Benchmark the fabric."""
        print(f"\nBenchmarking {num_hashes:,} hashes...")

        # Warm up
        _ = self.hash_batch(1000)

        # Actual benchmark
        start = time.perf_counter()
        _ = self.hash_batch(num_hashes)
        elapsed = time.perf_counter() - start

        hashrate = num_hashes / elapsed

        return {
            "num_hashes": num_hashes,
            "elapsed_seconds": elapsed,
            "hashrate": hashrate,
            "hashrate_per_square": hashrate / self.num_squares,
        }


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def benchmark_native_fabric():
    """Full benchmark of native ASIC fabric."""

    print("=" * 70)
    print("NATIVE ASIC FABRIC - VECTORIZED (FAST)")
    print("=" * 70)
    print()

    # Create fabric
    fabric = VectorizedFabric(num_squares=10240)

    # Benchmark with increasing batch sizes
    print()
    print("Batch size scaling:")
    print("-" * 50)

    for batch_size in [1000, 10000, 100000, 500000]:
        result = fabric.benchmark(batch_size)
        print(f"  {batch_size:>7,} hashes: {result['hashrate']:>12,.0f} H/s "
              f"({result['elapsed_seconds']:.3f}s)")

    # Final benchmark
    print()
    print("=" * 70)
    print("FINAL BENCHMARK (1M hashes)")
    print("=" * 70)

    result = fabric.benchmark(1_000_000)

    print(f"\n  Hashes:     {result['num_hashes']:,}")
    print(f"  Time:       {result['elapsed_seconds']:.3f} seconds")
    print(f"  Hashrate:   {result['hashrate']:,.0f} H/s")
    print(f"              {result['hashrate']/1e6:.2f} MH/s")
    print()

    # Comparison
    asic_hashrate = 322_000_000_000  # 322 GH/s
    gap = asic_hashrate / result['hashrate']

    print("=" * 70)
    print("COMPARISON TO HARDWARE ASIC")
    print("=" * 70)
    print(f"  Our fabric:      {result['hashrate']/1e6:,.2f} MH/s")
    print(f"  BM1398 chip:     {asic_hashrate/1e9:,.0f} GH/s")
    print(f"  Gap:             {gap:,.0f}x")
    print()

    # What we can match
    antminer_hashrate = 110_000_000_000_000  # 110 TH/s
    squares_to_match = antminer_hashrate / result['hashrate']

    print(f"  To match 1 Antminer S19 Pro ({antminer_hashrate/1e12:.0f} TH/s):")
    print(f"    Need {squares_to_match:,.0f} parallel fabric instances")
    print()

    # Memory estimate
    bytes_per_hash = 16 * 4 + 64 * 4 + 8 * 4  # block + W + H
    memory_per_million = bytes_per_hash * 1_000_000 / 1e9

    print(f"  Memory per 1M parallel hashes: {memory_per_million:.2f} GB")
    print(f"  On 128GB Thor: ~{128/memory_per_million:.0f}M parallel hashes possible")
    print()

    print("=" * 70)
    print("THE FABRIC IS ALIVE")
    print("=" * 70)

    return fabric, result


if __name__ == "__main__":
    fabric, result = benchmark_native_fabric()
