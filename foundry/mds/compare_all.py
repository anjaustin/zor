"""
Complete MDS Benchmark: All approaches compared.

1. Sequential Frozen Shapes (carry chain)
2. Parallel Geometry (direct carry read)
3. Native CPU Addition
"""

import subprocess
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from forge import freeze_factored


def get_type(bits, output=False):
    b = bits + 1 if output else bits
    if b <= 8: return "uint8_t"
    if b <= 16: return "uint16_t"
    if b <= 32: return "uint32_t"
    return "uint64_t"


def emit_parallel(bits):
    """Parallel carry geometry."""
    in_t, out_t = get_type(bits), get_type(bits, True)
    lines = [f"static inline {out_t} parallel_{bits}({in_t} a, {in_t} b) {{"]

    for i in range(bits):
        lines.append(f"    int g{i} = (a >> {i}) & 1 & ((b >> {i}) & 1);")
        lines.append(f"    int p{i} = ((a >> {i}) & 1) ^ ((b >> {i}) & 1);")

    lines.append("    int c0 = 0;")
    for i in range(1, bits + 1):
        terms = []
        for k in range(i):
            if k == i - 1:
                terms.append(f"g{k}")
            else:
                props = " & ".join(f"p{j}" for j in range(k + 1, i))
                terms.append(f"(g{k} & {props})")
        lines.append(f"    int c{i} = {' | '.join(terms)};")

    lines.append(f"    {out_t} sum = 0;")
    for i in range(bits):
        lines.append(f"    sum |= ({out_t})(((a >> {i}) & 1) ^ ((b >> {i}) & 1) ^ c{i}) << {i};")
    lines.append(f"    sum |= ({out_t})c{bits} << {bits};")
    lines.append("    return sum;")
    lines.append("}")
    return "\n".join(lines)


def benchmark(name, code, func_call, bits, iterations):
    """Run a benchmark."""
    mask = (1 << bits) - 1
    out_t = get_type(bits, True)

    harness = f'''
#include <stdio.h>
#include <stdint.h>
#include <time.h>

{code}

int main() {{
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    volatile uint64_t sum = 0;
    for (uint64_t i = 0; i < {iterations}ULL; i++) {{
        {out_t} a = i & {mask}ULL;
        {out_t} b = (i >> {bits//2}) & {mask}ULL;
        sum += {func_call};
    }}

    clock_gettime(CLOCK_MONOTONIC, &end);
    double ns = ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) / {iterations} * 1e9;
    printf("%.2f", ns);
    return (int)(sum & 0xFF);
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
        return float(r.stdout.strip())


def main():
    print("MDS COMPLETE BENCHMARK")
    print("=" * 70)
    print()
    print(f"{'Bits':>6} │ {'Sequential':>12} │ {'Parallel':>12} │ {'Native':>12} │ {'Winner':>10}")
    print("─" * 70)

    for bits in [4, 8, 16, 32]:
        iters = 10000000 if bits <= 8 else 1000000 if bits <= 16 else 100000

        # Sequential frozen
        verilog = f"""
module adder{bits}(input [{bits-1}:0] a, input [{bits-1}:0] b, output [{bits}:0] sum);
    assign sum = a + b;
endmodule
"""
        _, src = freeze_factored(verilog)
        src_clean = "\n".join(l for l in src.split("\n") if '#include "frozen_' not in l)
        out_t = get_type(bits, True)
        seq_ns = benchmark("seq", src_clean, f"({{ {out_t} r; frozen_adder{bits}(a, b, &r); r; }})", bits, iters)

        # Parallel geometry
        par_code = emit_parallel(bits)
        par_ns = benchmark("par", par_code, f"parallel_{bits}(a, b)", bits, iters)

        # Native
        nat_ns = benchmark("nat", "", "a + b", bits, iters)

        # Determine winner
        results = [("Sequential", seq_ns), ("Parallel", par_ns), ("Native", nat_ns)]
        results = [(n, v) for n, v in results if v is not None]
        winner = min(results, key=lambda x: x[1])[0] if results else "?"

        seq_str = f"{seq_ns:.2f} ns" if seq_ns else "error"
        par_str = f"{par_ns:.2f} ns" if par_ns else "error"
        nat_str = f"{nat_ns:.2f} ns" if nat_ns else "error"

        print(f"{bits:>6} │ {seq_str:>12} │ {par_str:>12} │ {nat_str:>12} │ {winner:>10}")

    print("─" * 70)
    print()
    print("INSIGHTS:")
    print("  • Sequential: Carry chain propagation (like original hardware)")
    print("  • Parallel:   Direct geometry read (no dependencies)")
    print("  • Native:     CPU's optimized ADD instruction")
    print()
    print("  At small bit widths, READING geometry beats COMPUTING.")
    print("  The shape contains the answer. We just look it up.")


if __name__ == "__main__":
    main()
