"""
Large-scale MDS Benchmark: Sequential vs Parallel Prefix Geometry

Using Kogge-Stone parallel prefix for O(n log n) work instead of O(n²).
"""

import subprocess
import tempfile
from pathlib import Path
import math


def emit_sequential(bits):
    """Sequential carry chain - O(n) latency, O(n) work."""
    lines = []
    lines.append(f"static inline void seq_{bits}(const uint8_t *a, const uint8_t *b, uint8_t *sum) {{")
    lines.append(f"    int carry = 0;")
    lines.append(f"    for (int i = 0; i < {bits}; i++) {{")
    lines.append(f"        int ai = (a[i/8] >> (i%8)) & 1;")
    lines.append(f"        int bi = (b[i/8] >> (i%8)) & 1;")
    lines.append(f"        int s = ai ^ bi ^ carry;")
    lines.append(f"        carry = (ai & bi) | (carry & (ai ^ bi));")
    lines.append(f"        if (s) sum[i/8] |= (1 << (i%8)); else sum[i/8] &= ~(1 << (i%8));")
    lines.append(f"    }}")
    lines.append(f"    if (carry) sum[{bits}/8] |= (1 << ({bits}%8)); else sum[{bits}/8] &= ~(1 << ({bits}%8));")
    lines.append(f"}}")
    return "\n".join(lines)


def emit_parallel_prefix(bits):
    """Kogge-Stone parallel prefix - O(log n) latency, O(n log n) work."""
    num_bytes = (bits + 7) // 8

    lines = []
    lines.append(f"// Parallel Prefix {bits}-bit adder - Kogge-Stone")
    lines.append(f"static inline void par_{bits}(const uint8_t *a, const uint8_t *b, uint8_t *sum) {{")

    # Work with bytes for efficiency
    lines.append(f"    // Initial G,P per byte")
    lines.append(f"    uint16_t G[{num_bytes}], P[{num_bytes}];")
    lines.append(f"    for (int i = 0; i < {num_bytes}; i++) {{")
    lines.append(f"        uint16_t s = (uint16_t)a[i] + (uint16_t)b[i];")
    lines.append(f"        G[i] = s >> 8;")  # Carry out
    lines.append(f"        P[i] = (s & 0xFF) == 0xFF ? 1 : 0;")  # Would propagate
    lines.append(f"    }}")

    # Kogge-Stone parallel prefix
    levels = math.ceil(math.log2(num_bytes)) if num_bytes > 1 else 1
    lines.append(f"    // Parallel prefix - {levels} levels")
    lines.append(f"    uint16_t G2[{num_bytes}], P2[{num_bytes}];")

    for level in range(levels):
        stride = 1 << level
        lines.append(f"    // Level {level}: stride {stride}")
        lines.append(f"    for (int i = 0; i < {num_bytes}; i++) {{")
        lines.append(f"        if (i >= {stride}) {{")
        lines.append(f"            G2[i] = G[i] | (P[i] & G[i - {stride}]);")
        lines.append(f"            P2[i] = P[i] & P[i - {stride}];")
        lines.append(f"        }} else {{")
        lines.append(f"            G2[i] = G[i]; P2[i] = P[i];")
        lines.append(f"        }}")
        lines.append(f"    }}")
        lines.append(f"    for (int i = 0; i < {num_bytes}; i++) {{ G[i] = G2[i]; P[i] = P2[i]; }}")

    # Final sum using prefix carries
    lines.append(f"    // Final sum with carries")
    lines.append(f"    uint16_t carry = 0;")
    lines.append(f"    for (int i = 0; i < {num_bytes}; i++) {{")
    lines.append(f"        uint16_t s = (uint16_t)a[i] + (uint16_t)b[i] + carry;")
    lines.append(f"        sum[i] = s & 0xFF;")
    lines.append(f"        carry = (i > 0) ? G[i-1] : 0;")
    lines.append(f"    }}")
    lines.append(f"    sum[{num_bytes}] = G[{num_bytes - 1}];")
    lines.append(f"}}")
    return "\n".join(lines)


def benchmark(name, code, bits, iterations):
    """Run benchmark."""
    bytes_needed = (bits + 7) // 8 + 1

    harness = f'''
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

{code}

int main() {{
    uint8_t a[{bytes_needed}] = {{0}};
    uint8_t b[{bytes_needed}] = {{0}};
    uint8_t sum[{bytes_needed + 1}] = {{0}};

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    volatile uint64_t acc = 0;
    for (uint64_t i = 0; i < {iterations}ULL; i++) {{
        a[0] = i & 0xFF;
        a[1] = (i >> 8) & 0xFF;
        b[0] = (i >> 4) & 0xFF;
        b[1] = (i >> 12) & 0xFF;
        {name}_{bits}(a, b, sum);
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
            print(f"Error ({bits}-bit {name}):", r.stderr[:300])
            return None

        r = subprocess.run([str(bin)], capture_output=True, text=True)
        try:
            return float(r.stdout.strip())
        except:
            return None


def main():
    print("LARGE-SCALE MDS BENCHMARK (Kogge-Stone Parallel Prefix)")
    print("=" * 70)
    print()
    print(f"{'Bits':>6} │ {'Sequential':>14} │ {'Parallel':>14} │ {'Speedup':>10}")
    print("─" * 70)

    for bits in [64, 256, 512, 2048, 4096]:
        iters = 1000000 if bits <= 256 else 100000 if bits <= 1024 else 10000

        seq_code = emit_sequential(bits)
        par_code = emit_parallel_prefix(bits)

        seq_ns = benchmark("seq", seq_code, bits, iters)
        par_ns = benchmark("par", par_code, bits, iters)

        if seq_ns and par_ns:
            speedup = seq_ns / par_ns
            print(f"{bits:>6} │ {seq_ns:>11.2f} ns │ {par_ns:>11.2f} ns │ {speedup:>9.2f}x")
        else:
            print(f"{bits:>6} │ {'error':>14} │ {'error':>14} │ {'?':>10}")

    print("─" * 70)
    print()
    print("Sequential: O(n) latency - carry chain")
    print("Parallel:   O(log n) latency - prefix tree (Kogge-Stone)")
    print()
    print("In hardware: all prefix levels compute simultaneously.")
    print("4096-bit addition in log2(512) = 9 gate delays.")


if __name__ == "__main__":
    main()
