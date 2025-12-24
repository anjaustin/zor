"""
Push SIMD Geometry further - 512, 1024 bits
"""

import subprocess
import tempfile
from pathlib import Path


def emit_simd_512():
    """512-bit: 8 x 64-bit lanes."""
    return '''
#include <arm_neon.h>

static inline void simd_add_512(const uint64_t *a, const uint64_t *b, uint64_t *sum) {
    // Load and add 8 lanes in parallel (4 NEON registers)
    uint64x2_t vs0 = vaddq_u64(vld1q_u64(a), vld1q_u64(b));
    uint64x2_t vs1 = vaddq_u64(vld1q_u64(a+2), vld1q_u64(b+2));
    uint64x2_t vs2 = vaddq_u64(vld1q_u64(a+4), vld1q_u64(b+4));
    uint64x2_t vs3 = vaddq_u64(vld1q_u64(a+6), vld1q_u64(b+6));

    vst1q_u64(sum, vs0);
    vst1q_u64(sum+2, vs1);
    vst1q_u64(sum+4, vs2);
    vst1q_u64(sum+6, vs3);

    // Inter-lane carry geometry (8 lanes = 3 prefix levels)
    int G[8], P[8];
    for (int i = 0; i < 8; i++) {
        G[i] = (sum[i] < a[i]);
        P[i] = (sum[i] == 0xFFFFFFFFFFFFFFFFULL);
    }

    // Parallel prefix
    int G2[8], P2[8];
    for (int i = 0; i < 8; i++) {
        G2[i] = (i >= 1) ? G[i] | (P[i] & G[i-1]) : G[i];
        P2[i] = (i >= 1) ? P[i] & P[i-1] : P[i];
    }
    for (int i = 0; i < 8; i++) { G[i] = G2[i]; P[i] = P2[i]; }

    for (int i = 0; i < 8; i++) {
        G2[i] = (i >= 2) ? G[i] | (P[i] & G[i-2]) : G[i];
        P2[i] = (i >= 2) ? P[i] & P[i-2] : P[i];
    }
    for (int i = 0; i < 8; i++) { G[i] = G2[i]; P[i] = P2[i]; }

    for (int i = 0; i < 8; i++) {
        G2[i] = (i >= 4) ? G[i] | (P[i] & G[i-4]) : G[i];
    }

    // Apply carries
    for (int i = 1; i < 8; i++) sum[i] += G2[i-1];
    sum[8] = G2[7];
}
'''


def emit_simd_1024():
    """1024-bit: 16 x 64-bit lanes."""
    return '''
#include <arm_neon.h>

static inline void simd_add_1024(const uint64_t *a, const uint64_t *b, uint64_t *sum) {
    // Load and add 16 lanes in parallel
    for (int i = 0; i < 16; i += 2) {
        uint64x2_t vs = vaddq_u64(vld1q_u64(a+i), vld1q_u64(b+i));
        vst1q_u64(sum+i, vs);
    }

    // Inter-lane carry geometry (16 lanes = 4 prefix levels)
    int G[16], P[16];
    for (int i = 0; i < 16; i++) {
        G[i] = (sum[i] < a[i]);
        P[i] = (sum[i] == 0xFFFFFFFFFFFFFFFFULL);
    }

    int G2[16], P2[16];
    for (int stride = 1; stride <= 8; stride *= 2) {
        for (int i = 0; i < 16; i++) {
            G2[i] = (i >= stride) ? G[i] | (P[i] & G[i-stride]) : G[i];
            P2[i] = (i >= stride) ? P[i] & P[i-stride] : P[i];
        }
        for (int i = 0; i < 16; i++) { G[i] = G2[i]; P[i] = P2[i]; }
    }

    for (int i = 1; i < 16; i++) sum[i] += G[i-1];
    sum[16] = G[15];
}
'''


def benchmark(name, code, lanes, iterations):
    """Run benchmark."""
    harness = f'''
#include <stdio.h>
#include <stdint.h>
#include <time.h>

{code}

int main() {{
    uint64_t a[{lanes}] = {{0}};
    uint64_t b[{lanes}] = {{0}};
    uint64_t sum[{lanes + 1}] = {{0}};

    struct timespec start, end;

    for (int i = 0; i < 1000; i++) {{
        a[0] = i; b[0] = i;
        {name}(a, b, sum);
    }}

    clock_gettime(CLOCK_MONOTONIC, &start);

    volatile uint64_t acc = 0;
    for (uint64_t i = 0; i < {iterations}ULL; i++) {{
        a[0] = i; a[1] = i >> 16;
        b[0] = i >> 8; b[1] = i >> 24;
        {name}(a, b, sum);
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

        r = subprocess.run(["gcc", "-O3", "-march=native", "-o", str(bin), str(src)],
                          capture_output=True, text=True)
        if r.returncode != 0:
            print(f"Error: {r.stderr[:300]}")
            return None

        r = subprocess.run([str(bin)], capture_output=True, text=True)
        return float(r.stdout.strip()) if r.stdout.strip() else None


def main():
    print("SIMD GEOMETRY SCALING")
    print("=" * 50)
    print()
    print(f"{'Bits':>6} │ {'Lanes':>6} │ {'Time':>10} │ {'ns/lane':>10}")
    print("─" * 50)

    for bits, lanes, code, func in [
        (256, 4, '''
#include <arm_neon.h>
static inline void simd_add_256(const uint64_t *a, const uint64_t *b, uint64_t *sum) {
    uint64x2_t vs0 = vaddq_u64(vld1q_u64(a), vld1q_u64(b));
    uint64x2_t vs1 = vaddq_u64(vld1q_u64(a+2), vld1q_u64(b+2));
    vst1q_u64(sum, vs0); vst1q_u64(sum+2, vs1);
    int G[4], P[4];
    for (int i = 0; i < 4; i++) { G[i] = sum[i] < a[i]; P[i] = sum[i] == 0xFFFFFFFFFFFFFFFFULL; }
    int c1 = G[0], c2 = G[1]|(P[1]&G[0]), c3 = G[2]|(P[2]&G[1])|(P[2]&P[1]&G[0]);
    int c4 = G[3]|(P[3]&G[2])|(P[3]&P[2]&G[1])|(P[3]&P[2]&P[1]&G[0]);
    sum[1] += c1; sum[2] += c2; sum[3] += c3; sum[4] = c4;
}
''', "simd_add_256"),
        (512, 8, emit_simd_512(), "simd_add_512"),
        (1024, 16, emit_simd_1024(), "simd_add_1024"),
    ]:
        iters = 10000000 if bits <= 512 else 1000000
        ns = benchmark(func, code, lanes, iters)
        if ns:
            print(f"{bits:>6} │ {lanes:>6} │ {ns:>8.2f}ns │ {ns/lanes:>8.2f}ns")

    print("─" * 50)
    print()
    print("More bits, same ballpark time.")
    print("Parallel lanes + geometry = constant-ish latency.")


if __name__ == "__main__":
    main()
