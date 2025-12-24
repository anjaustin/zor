"""
Freeze Parallel Carry Geometry to C.

All carries computed independently. No chain dependencies.
"""

import subprocess
import tempfile
from pathlib import Path


def emit_parallel_adder_c(bits: int) -> str:
    """Generate C code for parallel carry adder."""

    # Determine types
    if bits <= 8:
        in_type = "uint8_t"
    elif bits <= 16:
        in_type = "uint16_t"
    elif bits <= 32:
        in_type = "uint32_t"
    else:
        in_type = "uint64_t"

    if bits + 1 <= 8:
        out_type = "uint8_t"
    elif bits + 1 <= 16:
        out_type = "uint16_t"
    elif bits + 1 <= 32:
        out_type = "uint32_t"
    else:
        out_type = "uint64_t"

    lines = []
    lines.append(f"// Parallel Carry Geometry - {bits}-bit adder")
    lines.append(f"// All carries read directly. No sequential dependency.")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"static inline {out_type} parallel_add_{bits}({in_type} a, {in_type} b) {{")

    # Generate G and P for each bit
    lines.append(f"    // Generate (G) and Propagate (P) - all parallel")
    for i in range(bits):
        lines.append(f"    int g{i} = (a >> {i}) & 1 & ((b >> {i}) & 1);")
        lines.append(f"    int p{i} = ((a >> {i}) & 1) ^ ((b >> {i}) & 1);")

    lines.append("")
    lines.append(f"    // Carries - each computed directly from geometry")
    lines.append(f"    int c0 = 0;")

    # Generate carry equations
    for i in range(1, bits + 1):
        # Carry[i] = G[i-1] | (P[i-1] & G[i-2]) | (P[i-1] & P[i-2] & G[i-3]) | ...
        terms = []
        for k in range(i):
            # Term: G[k] & P[k+1] & P[k+2] & ... & P[i-1]
            if k == i - 1:
                terms.append(f"g{k}")
            else:
                props = " & ".join(f"p{j}" for j in range(k + 1, i))
                terms.append(f"(g{k} & {props})")

        carry_expr = " | ".join(terms)
        lines.append(f"    int c{i} = {carry_expr};")

    lines.append("")
    lines.append(f"    // Sum bits - all parallel")
    lines.append(f"    {out_type} sum = 0;")
    for i in range(bits):
        lines.append(f"    sum |= ({out_type})(((a >> {i}) & 1) ^ ((b >> {i}) & 1) ^ c{i}) << {i};")
    lines.append(f"    sum |= ({out_type})c{bits} << {bits};")

    lines.append("")
    lines.append("    return sum;")
    lines.append("}")

    return "\n".join(lines)


def benchmark_parallel(bits: int):
    """Benchmark parallel geometry C code."""

    code = emit_parallel_adder_c(bits)

    iterations = 10000000 if bits <= 8 else 1000000 if bits <= 16 else 100000
    mask = (1 << bits) - 1

    harness = f'''
#include <stdio.h>
#include <stdint.h>
#include <time.h>

{code}

int main() {{
    struct timespec start, end;
    uint64_t iterations = {iterations};

    // Warm up
    for (int i = 0; i < 1000; i++) {{
        parallel_add_{bits}(i, i);
    }}

    clock_gettime(CLOCK_MONOTONIC, &start);

    volatile uint64_t sum = 0;
    for (uint64_t i = 0; i < iterations; i++) {{
        sum += parallel_add_{bits}(i & {mask}ULL, (i >> {bits//2}) & {mask}ULL);
    }}

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double ns_per_op = (elapsed / iterations) * 1e9;

    printf("parallel,%d,%lu,%.2f\\n", {bits}, iterations, ns_per_op);
    return (int)(sum & 0xFF);
}}
'''

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "bench.c"
        bin_path = Path(tmpdir) / "bench"

        src_path.write_text(harness)

        result = subprocess.run(
            ["gcc", "-O3", "-march=native", "-o", str(bin_path), str(src_path)],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"Compile error for {bits}-bit:")
            print(result.stderr[:500])
            return None

        result = subprocess.run([str(bin_path)], capture_output=True, text=True)
        return result.stdout.strip()


def benchmark_native(bits: int):
    """Benchmark native addition."""

    iterations = 10000000 if bits <= 8 else 1000000 if bits <= 16 else 100000
    mask = (1 << bits) - 1

    harness = f'''
#include <stdio.h>
#include <stdint.h>
#include <time.h>

int main() {{
    struct timespec start, end;
    uint64_t iterations = {iterations};

    clock_gettime(CLOCK_MONOTONIC, &start);

    volatile uint64_t sum = 0;
    for (uint64_t i = 0; i < iterations; i++) {{
        uint64_t a = i & {mask}ULL;
        uint64_t b = (i >> {bits//2}) & {mask}ULL;
        sum += a + b;
    }}

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double ns_per_op = (elapsed / iterations) * 1e9;

    printf("native,%d,%lu,%.2f\\n", {bits}, iterations, ns_per_op);
    return (int)(sum & 0xFF);
}}
'''

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "bench.c"
        bin_path = Path(tmpdir) / "bench"

        src_path.write_text(harness)

        subprocess.run(
            ["gcc", "-O3", "-march=native", "-o", str(bin_path), str(src_path)],
            capture_output=True, text=True
        )

        result = subprocess.run([str(bin_path)], capture_output=True, text=True)
        return result.stdout.strip()


if __name__ == "__main__":
    print("PARALLEL GEOMETRY vs NATIVE")
    print("=" * 60)
    print()
    print(f"{'Type':<12} {'Bits':>6} {'Iterations':>12} {'ns/op':>10}")
    print("-" * 60)

    results = []

    for bits in [4, 8, 16, 32]:
        par_result = benchmark_parallel(bits)
        nat_result = benchmark_native(bits)

        if par_result:
            parts = par_result.split(',')
            print(f"{'Parallel':<12} {parts[1]:>6} {parts[2]:>12} {parts[3]:>10}")
            results.append(('parallel', int(parts[1]), float(parts[3])))

        if nat_result:
            parts = nat_result.split(',')
            print(f"{'Native':<12} {parts[1]:>6} {parts[2]:>12} {parts[3]:>10}")
            results.append(('native', int(parts[1]), float(parts[3])))

        print()

    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    for bits in [4, 8, 16, 32]:
        par_ns = next((r[2] for r in results if r[0] == 'parallel' and r[1] == bits), None)
        nat_ns = next((r[2] for r in results if r[0] == 'native' and r[1] == bits), None)

        if par_ns and nat_ns:
            ratio = par_ns / nat_ns
            print(f"{bits:>3}-bit: Parallel geometry is {ratio:.1f}x {'slower' if ratio > 1 else 'faster'}")

    print()
    print("No carry chain. Every position read directly.")
