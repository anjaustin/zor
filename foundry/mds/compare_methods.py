"""
Compare MDS Geometry against standard big integer libraries.

1. GMP (GNU Multiple Precision) - the gold standard
2. __int128 chained - compiler intrinsic
3. Sequential carry chain - naive approach
4. SIMD Geometry - our method
"""

import subprocess
import tempfile
from pathlib import Path


def check_gmp():
    """Check if GMP is available."""
    result = subprocess.run(
        ["gcc", "-xc", "-", "-lgmp", "-o", "/dev/null"],
        input=b"#include <gmp.h>\nint main(){return 0;}",
        capture_output=True
    )
    return result.returncode == 0


def emit_gmp(bits):
    """GMP addition."""
    return f'''
#include <gmp.h>

static mpz_t ga, gb, gs;
static int gmp_init_done = 0;

static inline void gmp_add_{bits}(const uint64_t *a, const uint64_t *b, uint64_t *sum) {{
    if (!gmp_init_done) {{
        mpz_init(ga); mpz_init(gb); mpz_init(gs);
        gmp_init_done = 1;
    }}
    mpz_import(ga, {bits // 64}, -1, 8, 0, 0, a);
    mpz_import(gb, {bits // 64}, -1, 8, 0, 0, b);
    mpz_add(gs, ga, gb);
    size_t count;
    mpz_export(sum, &count, -1, 8, 0, 0, gs);
    for (size_t i = count; i <= {bits // 64}; i++) sum[i] = 0;
}}
'''


def emit_int128(bits):
    """Chained __int128 addition."""
    lanes = bits // 64
    return f'''
static inline void int128_add_{bits}(const uint64_t *a, const uint64_t *b, uint64_t *sum) {{
    __uint128_t carry = 0;
    for (int i = 0; i < {lanes}; i++) {{
        carry += (__uint128_t)a[i] + b[i];
        sum[i] = (uint64_t)carry;
        carry >>= 64;
    }}
    sum[{lanes}] = (uint64_t)carry;
}}
'''


def emit_sequential(bits):
    """Bit-by-bit sequential."""
    return f'''
static inline void seq_add_{bits}(const uint64_t *a, const uint64_t *b, uint64_t *sum) {{
    int carry = 0;
    for (int i = 0; i < {bits}; i++) {{
        int ai = (a[i/64] >> (i%64)) & 1;
        int bi = (b[i/64] >> (i%64)) & 1;
        int s = ai ^ bi ^ carry;
        carry = (ai & bi) | (carry & (ai ^ bi));
        if (s) sum[i/64] |= (1ULL << (i%64));
        else sum[i/64] &= ~(1ULL << (i%64));
    }}
    if (carry) sum[{bits // 64}] = 1; else sum[{bits // 64}] = 0;
}}
'''


def emit_simd_geometry(bits):
    """Our SIMD geometry approach."""
    lanes = bits // 64

    if lanes <= 16:
        # Simple SIMD for small sizes
        return f'''
#include <arm_neon.h>

static inline void geom_add_{bits}(const uint64_t *a, const uint64_t *b, uint64_t *sum) {{
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
    else:
        # Chunked for larger sizes
        chunks = lanes // 8
        return f'''
#include <arm_neon.h>

static inline void geom_add_{bits}(const uint64_t *a, const uint64_t *b, uint64_t *sum) {{
    int chunk_G[{chunks}], chunk_P[{chunks}];
    for (int c = 0; c < {chunks}; c++) {{
        int base = c * 8;
        for (int i = 0; i < 8; i += 2) {{
            uint64x2_t vs = vaddq_u64(vld1q_u64(a + base + i), vld1q_u64(b + base + i));
            vst1q_u64(sum + base + i, vs);
        }}
        int G[8], P[8];
        for (int i = 0; i < 8; i++) {{
            G[i] = (sum[base + i] < a[base + i]);
            P[i] = (sum[base + i] == 0xFFFFFFFFFFFFFFFFULL);
        }}
        int G2[8], P2[8];
        for (int stride = 1; stride <= 4; stride *= 2) {{
            for (int i = 0; i < 8; i++) {{
                G2[i] = (i >= stride) ? G[i] | (P[i] & G[i-stride]) : G[i];
                P2[i] = (i >= stride) ? P[i] & P[i-stride] : P[i];
            }}
            for (int i = 0; i < 8; i++) {{ G[i] = G2[i]; P[i] = P2[i]; }}
        }}
        for (int i = 1; i < 8; i++) sum[base + i] += G[i-1];
        chunk_G[c] = G[7];
        chunk_P[c] = P[7];
        for (int i = 0; i < 7; i++) chunk_P[c] &= P[i];
    }}
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


def benchmark(name, code, func_name, bits, iterations, extra_flags=""):
    """Run a benchmark."""
    lanes = bits // 64

    harness = f'''
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <string.h>

{code}

int main() {{
    uint64_t a[{lanes + 1}] __attribute__((aligned(64))) = {{0}};
    uint64_t b[{lanes + 1}] __attribute__((aligned(64))) = {{0}};
    uint64_t sum[{lanes + 2}] __attribute__((aligned(64))) = {{0}};

    // Warmup
    for (int i = 0; i < 100; i++) {{
        a[0] = i; b[0] = i;
        {func_name}(a, b, sum);
    }}

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    volatile uint64_t acc = 0;
    for (uint64_t i = 0; i < {iterations}ULL; i++) {{
        a[0] = i; a[1] = i >> 16;
        b[0] = i >> 8; b[1] = i >> 24;
        {func_name}(a, b, sum);
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

        cmd = ["gcc", "-O3", "-march=native", "-o", str(bin), str(src)]
        if extra_flags:
            cmd.extend(extra_flags.split())

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return None

        r = subprocess.run([str(bin)], capture_output=True, text=True)
        try:
            return float(r.stdout.strip())
        except:
            return None


def main():
    print("MDS GEOMETRY vs STANDARD METHODS")
    print("=" * 75)
    print()

    has_gmp = check_gmp()
    if has_gmp:
        print("GMP: Available")
    else:
        print("GMP: Not installed (skipping)")
    print()

    print(f"{'Bits':>6} │ {'Sequential':>11} │ {'__int128':>11} │ {'GMP':>11} │ {'Geometry':>11} │ {'Winner':>10}")
    print("─" * 75)

    for bits in [256, 512, 1024, 2048, 4096]:
        iters = 1000000 if bits <= 1024 else 100000

        # Sequential
        seq_ns = benchmark("seq", emit_sequential(bits), f"seq_add_{bits}", bits, iters)

        # __int128
        i128_ns = benchmark("i128", emit_int128(bits), f"int128_add_{bits}", bits, iters)

        # GMP
        gmp_ns = None
        if has_gmp:
            gmp_ns = benchmark("gmp", emit_gmp(bits), f"gmp_add_{bits}", bits, iters, "-lgmp")

        # Geometry
        geom_ns = benchmark("geom", emit_simd_geometry(bits), f"geom_add_{bits}", bits, iters)

        # Find winner
        results = [("Seq", seq_ns), ("i128", i128_ns), ("GMP", gmp_ns), ("Geom", geom_ns)]
        valid = [(n, v) for n, v in results if v is not None]
        winner = min(valid, key=lambda x: x[1])[0] if valid else "?"

        seq_str = f"{seq_ns:.1f}ns" if seq_ns else "err"
        i128_str = f"{i128_ns:.1f}ns" if i128_ns else "err"
        gmp_str = f"{gmp_ns:.1f}ns" if gmp_ns else "n/a"
        geom_str = f"{geom_ns:.1f}ns" if geom_ns else "err"

        print(f"{bits:>6} │ {seq_str:>11} │ {i128_str:>11} │ {gmp_str:>11} │ {geom_str:>11} │ {winner:>10}")

    print("─" * 75)
    print()
    print("Sequential: Bit-by-bit carry chain (naive)")
    print("__int128:   Compiler intrinsic 128-bit arithmetic")
    print("GMP:        GNU Multiple Precision library")
    print("Geometry:   SIMD parallel prefix (our method)")


if __name__ == "__main__":
    main()
