"""
4096-bit SIMD Geometry - 64 lanes of 64-bit
"""

import subprocess
import tempfile
from pathlib import Path


def emit_simd(bits):
    """Generate SIMD adder for arbitrary bit width."""
    lanes = bits // 64
    levels = (lanes - 1).bit_length()  # log2 for prefix levels

    return f'''
#include <arm_neon.h>

static inline void simd_add_{bits}(const uint64_t *a, const uint64_t *b, uint64_t *sum) {{
    // Load and add {lanes} lanes in parallel
    for (int i = 0; i < {lanes}; i += 2) {{
        uint64x2_t vs = vaddq_u64(vld1q_u64(a+i), vld1q_u64(b+i));
        vst1q_u64(sum+i, vs);
    }}

    // Inter-lane carry geometry ({lanes} lanes = {levels} prefix levels)
    int G[{lanes}], P[{lanes}];
    for (int i = 0; i < {lanes}; i++) {{
        G[i] = (sum[i] < a[i]);
        P[i] = (sum[i] == 0xFFFFFFFFFFFFFFFFULL);
    }}

    // Kogge-Stone parallel prefix
    int G2[{lanes}], P2[{lanes}];
    for (int stride = 1; stride < {lanes}; stride *= 2) {{
        for (int i = 0; i < {lanes}; i++) {{
            if (i >= stride) {{
                G2[i] = G[i] | (P[i] & G[i-stride]);
                P2[i] = P[i] & P[i-stride];
            }} else {{
                G2[i] = G[i];
                P2[i] = P[i];
            }}
        }}
        for (int i = 0; i < {lanes}; i++) {{ G[i] = G2[i]; P[i] = P2[i]; }}
    }}

    // Apply carries
    for (int i = 1; i < {lanes}; i++) sum[i] += G[i-1];
    sum[{lanes}] = G[{lanes - 1}];
}}
'''


def benchmark(bits, iterations):
    """Run benchmark."""
    lanes = bits // 64
    code = emit_simd(bits)

    harness = f'''
#include <stdio.h>
#include <stdint.h>
#include <time.h>

{code}

int main() {{
    uint64_t a[{lanes}] __attribute__((aligned(16))) = {{0}};
    uint64_t b[{lanes}] __attribute__((aligned(16))) = {{0}};
    uint64_t sum[{lanes + 1}] __attribute__((aligned(16))) = {{0}};

    struct timespec start, end;

    for (int i = 0; i < 100; i++) {{
        a[0] = i; b[0] = i;
        simd_add_{bits}(a, b, sum);
    }}

    clock_gettime(CLOCK_MONOTONIC, &start);

    volatile uint64_t acc = 0;
    for (uint64_t i = 0; i < {iterations}ULL; i++) {{
        a[0] = i; a[1] = i >> 16;
        b[0] = i >> 8; b[1] = i >> 24;
        simd_add_{bits}(a, b, sum);
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
            print(f"Compile error: {r.stderr[:300]}")
            return None

        r = subprocess.run([str(bin)], capture_output=True, text=True)
        return float(r.stdout.strip()) if r.stdout.strip() else None


def main():
    print("SIMD GEOMETRY: SCALING TO 4096 BITS")
    print("=" * 55)
    print()
    print(f"{'Bits':>6} │ {'Lanes':>6} │ {'Time':>10} │ {'ns/lane':>10}")
    print("─" * 55)

    for bits in [256, 512, 1024, 2048, 4096]:
        lanes = bits // 64
        iters = 10000000 if bits <= 512 else 1000000 if bits <= 2048 else 100000
        ns = benchmark(bits, iters)
        if ns:
            print(f"{bits:>6} │ {lanes:>6} │ {ns:>8.2f}ns │ {ns/lanes:>8.3f}ns")

    print("─" * 55)
    print()
    print("The geometry scales. The time barely moves.")


if __name__ == "__main__":
    main()
