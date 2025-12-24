"""
SIMD Parallel Geometry - 4 lanes of 64-bit in parallel.

256 bits in the time of 64. Using ARM NEON.
"""

import subprocess
import tempfile
from pathlib import Path


def emit_simd_256():
    """256-bit adder using 4x64-bit with NEON (2x128-bit registers)."""
    return '''
#include <arm_neon.h>

// 256-bit addition: 4 parallel 64-bit lanes + carry propagation
static inline void simd_add_256(const uint64_t *a, const uint64_t *b, uint64_t *sum) {
    // Load into two 128-bit NEON registers (2 x 64-bit lanes each)
    uint64x2_t va0 = vld1q_u64(a);
    uint64x2_t va1 = vld1q_u64(a + 2);
    uint64x2_t vb0 = vld1q_u64(b);
    uint64x2_t vb1 = vld1q_u64(b + 2);

    // Add lanes in parallel
    uint64x2_t vs0 = vaddq_u64(va0, vb0);
    uint64x2_t vs1 = vaddq_u64(va1, vb1);

    // Store results
    vst1q_u64(sum, vs0);
    vst1q_u64(sum + 2, vs1);

    // Handle inter-lane carries using geometry
    uint64_t s0 = sum[0], s1 = sum[1], s2 = sum[2], s3 = sum[3];
    uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];

    // Generate: overflow detection
    int g0 = (s0 < a0);
    int g1 = (s1 < a1);
    int g2 = (s2 < a2);
    int g3 = (s3 < a3);

    // Propagate: would carry propagate?
    int p0 = (s0 == 0xFFFFFFFFFFFFFFFFULL);
    int p1 = (s1 == 0xFFFFFFFFFFFFFFFFULL);
    int p2 = (s2 == 0xFFFFFFFFFFFFFFFFULL);
    int p3 = (s3 == 0xFFFFFFFFFFFFFFFFULL);

    // Parallel prefix carries - all positions read directly
    int c1 = g0;
    int c2 = g1 | (p1 & g0);
    int c3 = g2 | (p2 & g1) | (p2 & p1 & g0);
    int c4 = g3 | (p3 & g2) | (p3 & p2 & g1) | (p3 & p2 & p1 & g0);

    // Apply carries
    sum[1] += c1;
    sum[2] += c2;
    sum[3] += c3;
    sum[4] = c4;
}
'''


def emit_sequential_256():
    """Sequential 256-bit for comparison."""
    return '''
static inline void seq_add_256(const uint64_t *a, const uint64_t *b, uint64_t *sum) {
    __uint128_t acc = 0;
    for (int i = 0; i < 4; i++) {
        acc += (__uint128_t)a[i] + b[i];
        sum[i] = (uint64_t)acc;
        acc >>= 64;
    }
    sum[4] = (uint64_t)acc;
}
'''


def emit_parallel_64():
    """Our parallel 64-bit geometry for reference."""
    return '''
static inline uint64_t par_64(uint64_t a, uint64_t b) {
    // Use the 8-byte chunked approach
    uint8_t *ap = (uint8_t*)&a;
    uint8_t *bp = (uint8_t*)&b;

    uint16_t s[8];
    int G[8], P[8];

    // All chunks in parallel
    for (int i = 0; i < 8; i++) {
        s[i] = (uint16_t)ap[i] + bp[i];
        G[i] = s[i] >> 8;
        P[i] = (s[i] & 0xFF) == 0xFF;
    }

    // Parallel prefix (3 levels for 8 elements)
    int G2[8], P2[8];

    // Level 0
    for (int i = 0; i < 8; i++) {
        if (i >= 1) { G2[i] = G[i] | (P[i] & G[i-1]); P2[i] = P[i] & P[i-1]; }
        else { G2[i] = G[i]; P2[i] = P[i]; }
    }
    for (int i = 0; i < 8; i++) { G[i] = G2[i]; P[i] = P2[i]; }

    // Level 1
    for (int i = 0; i < 8; i++) {
        if (i >= 2) { G2[i] = G[i] | (P[i] & G[i-2]); P2[i] = P[i] & P[i-2]; }
        else { G2[i] = G[i]; P2[i] = P[i]; }
    }
    for (int i = 0; i < 8; i++) { G[i] = G2[i]; P[i] = P2[i]; }

    // Level 2
    for (int i = 0; i < 8; i++) {
        if (i >= 4) { G2[i] = G[i] | (P[i] & G[i-4]); P2[i] = P[i] & P[i-4]; }
        else { G2[i] = G[i]; P2[i] = P[i]; }
    }

    // Assemble result
    uint64_t result = 0;
    uint8_t *rp = (uint8_t*)&result;
    int carry = 0;
    for (int i = 0; i < 8; i++) {
        rp[i] = (s[i] + carry) & 0xFF;
        carry = (i > 0) ? G2[i-1] : 0;
    }

    return result;
}
'''


def benchmark(name, code, func_call, iterations):
    """Run benchmark."""
    harness = f'''
#include <stdio.h>
#include <stdint.h>
#include <time.h>

{code}

int main() {{
    uint64_t a[4] = {{0}};
    uint64_t b[4] = {{0}};
    uint64_t sum[5] = {{0}};

    struct timespec start, end;

    // Warmup
    for (int i = 0; i < 1000; i++) {{
        a[0] = i; b[0] = i;
        {func_call};
    }}

    clock_gettime(CLOCK_MONOTONIC, &start);

    volatile uint64_t acc = 0;
    for (uint64_t i = 0; i < {iterations}ULL; i++) {{
        a[0] = i;
        a[1] = i >> 16;
        b[0] = i >> 8;
        b[1] = i >> 24;
        {func_call};
        acc += sum[0];
    }}

    clock_gettime(CLOCK_MONOTONIC, &end);
    double ns = ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) / {iterations} * 1e9;
    printf("%.2f", ns);
    return (int)(acc & 0xFF);
}}
'''

    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "b.c"
        bin = Path(tmpdir) / "b"
        src.write_text(harness)

        r = subprocess.run(
            ["gcc", "-O3", "-march=native", "-o", str(bin), str(src)],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            print(f"Compile error ({name}):")
            print(r.stderr[:500])
            return None

        r = subprocess.run([str(bin)], capture_output=True, text=True)
        try:
            return float(r.stdout.strip())
        except:
            print(f"Runtime error: {r.stderr}")
            return None


def main():
    print("SIMD GEOMETRY: 256-bit in parallel lanes")
    print("=" * 60)
    print()

    iterations = 10000000

    # Sequential 256-bit
    seq_ns = benchmark("seq", emit_sequential_256(), "seq_add_256(a, b, sum)", iterations)
    print(f"Sequential 256-bit:    {seq_ns:.2f} ns" if seq_ns else "Sequential: error")

    # SIMD parallel geometry 256-bit
    simd_ns = benchmark("simd", emit_simd_256(), "simd_add_256(a, b, sum)", iterations)
    print(f"SIMD Geometry 256-bit: {simd_ns:.2f} ns" if simd_ns else "SIMD: error")

    if seq_ns and simd_ns:
        print(f"Speedup:               {seq_ns / simd_ns:.2f}x")

    print()
    print("─" * 60)
    print()
    print("4 x 64-bit lanes computed in parallel.")
    print("Inter-lane carries: parallel prefix geometry.")
    print("256 bits. One shot.")


if __name__ == "__main__":
    main()
