"""
Optimized SIMD Geometry - cache-aware chunking
"""

import subprocess
import tempfile
from pathlib import Path


def emit_chunked(bits):
    """Hierarchical: 8-lane chunks with inter-chunk geometry."""
    lanes = bits // 64
    chunks = lanes // 8

    return f'''
#include <arm_neon.h>

static inline void simd_add_{bits}(const uint64_t *a, const uint64_t *b, uint64_t *sum) {{
    // Process in 8-lane chunks (fits in cache/registers)
    int chunk_G[{chunks}], chunk_P[{chunks}];

    for (int c = 0; c < {chunks}; c++) {{
        int base = c * 8;

        // Add 8 lanes in parallel
        for (int i = 0; i < 8; i += 2) {{
            uint64x2_t vs = vaddq_u64(vld1q_u64(a + base + i), vld1q_u64(b + base + i));
            vst1q_u64(sum + base + i, vs);
        }}

        // Intra-chunk carry geometry
        int G[8], P[8];
        for (int i = 0; i < 8; i++) {{
            G[i] = (sum[base + i] < a[base + i]);
            P[i] = (sum[base + i] == 0xFFFFFFFFFFFFFFFFULL);
        }}

        // 3-level prefix within chunk
        int G2[8], P2[8];
        for (int stride = 1; stride <= 4; stride *= 2) {{
            for (int i = 0; i < 8; i++) {{
                G2[i] = (i >= stride) ? G[i] | (P[i] & G[i-stride]) : G[i];
                P2[i] = (i >= stride) ? P[i] & P[i-stride] : P[i];
            }}
            for (int i = 0; i < 8; i++) {{ G[i] = G2[i]; P[i] = P2[i]; }}
        }}

        // Apply intra-chunk carries
        for (int i = 1; i < 8; i++) sum[base + i] += G[i-1];

        // Chunk generate/propagate for inter-chunk geometry
        chunk_G[c] = G[7];
        chunk_P[c] = P[7];
        for (int i = 0; i < 7; i++) chunk_P[c] &= P[i];
    }}

    // Inter-chunk carry geometry
    int cG[{chunks}], cP[{chunks}];
    for (int i = 0; i < {chunks}; i++) {{ cG[i] = chunk_G[i]; cP[i] = chunk_P[i]; }}

    int cG2[{chunks}], cP2[{chunks}];
    for (int stride = 1; stride < {chunks}; stride *= 2) {{
        for (int i = 0; i < {chunks}; i++) {{
            cG2[i] = (i >= stride) ? cG[i] | (cP[i] & cG[i-stride]) : cG[i];
            cP2[i] = (i >= stride) ? cP[i] & cP[i-stride] : cP[i];
        }}
        for (int i = 0; i < {chunks}; i++) {{ cG[i] = cG2[i]; cP[i] = cP2[i]; }}
    }}

    // Apply inter-chunk carries
    for (int c = 1; c < {chunks}; c++) {{
        if (cG[c-1]) {{
            for (int i = 0; i < 8; i++) {{
                sum[c*8 + i]++;
                if (sum[c*8 + i] != 0) break;
            }}
        }}
    }}

    sum[{lanes}] = cG[{chunks - 1}];
}}
'''


def benchmark(bits, iterations, optimized=False):
    """Run benchmark."""
    lanes = bits // 64

    if optimized and lanes >= 16:
        code = emit_chunked(bits)
    else:
        # Simple version for small sizes
        levels = (lanes - 1).bit_length()
        code = f'''
#include <arm_neon.h>
static inline void simd_add_{bits}(const uint64_t *a, const uint64_t *b, uint64_t *sum) {{
    for (int i = 0; i < {lanes}; i += 2) {{
        uint64x2_t vs = vaddq_u64(vld1q_u64(a+i), vld1q_u64(b+i));
        vst1q_u64(sum+i, vs);
    }}
    int G[{lanes}], P[{lanes}];
    for (int i = 0; i < {lanes}; i++) {{
        G[i] = (sum[i] < a[i]);
        P[i] = (sum[i] == 0xFFFFFFFFFFFFFFFFULL);
    }}
    int G2[{lanes}], P2[{lanes}];
    for (int stride = 1; stride < {lanes}; stride *= 2) {{
        for (int i = 0; i < {lanes}; i++) {{
            G2[i] = (i >= stride) ? G[i] | (P[i] & G[i-stride]) : G[i];
            P2[i] = (i >= stride) ? P[i] & P[i-stride] : P[i];
        }}
        for (int i = 0; i < {lanes}; i++) {{ G[i] = G2[i]; P[i] = P2[i]; }}
    }}
    for (int i = 1; i < {lanes}; i++) sum[i] += G[i-1];
    sum[{lanes}] = G[{lanes - 1}];
}}
'''

    harness = f'''
#include <stdio.h>
#include <stdint.h>
#include <time.h>

{code}

int main() {{
    uint64_t a[{lanes}] __attribute__((aligned(64))) = {{0}};
    uint64_t b[{lanes}] __attribute__((aligned(64))) = {{0}};
    uint64_t sum[{lanes + 1}] __attribute__((aligned(64))) = {{0}};

    for (int i = 0; i < 100; i++) {{ a[0] = i; simd_add_{bits}(a, b, sum); }}

    struct timespec start, end;
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
            return None

        r = subprocess.run([str(bin)], capture_output=True, text=True)
        return float(r.stdout.strip()) if r.stdout.strip() else None


def main():
    print("SIMD GEOMETRY: SIMPLE vs CHUNKED")
    print("=" * 65)
    print()
    print(f"{'Bits':>6} │ {'Simple':>12} │ {'Chunked':>12} │ {'Speedup':>10}")
    print("─" * 65)

    for bits in [256, 512, 1024, 2048, 4096]:
        iters = 1000000 if bits <= 1024 else 100000
        simple = benchmark(bits, iters, optimized=False)
        chunked = benchmark(bits, iters, optimized=True) if bits >= 1024 else simple

        if simple and chunked:
            speedup = simple / chunked
            print(f"{bits:>6} │ {simple:>10.2f}ns │ {chunked:>10.2f}ns │ {speedup:>9.2f}x")

    print("─" * 65)


if __name__ == "__main__":
    main()
