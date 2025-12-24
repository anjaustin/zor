"""
MDS Benchmark: Frozen Shapes vs Native Arithmetic

Generate actual frozen C code, compile it, and benchmark.
"""

import subprocess
import tempfile
import time
import os
from pathlib import Path

# Add forge to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from forge import freeze_factored


def generate_adder_verilog(bits):
    """Generate Verilog for n-bit adder."""
    return f"""
module adder{bits}(
    input [{bits-1}:0] a,
    input [{bits-1}:0] b,
    output [{bits}:0] sum
);
    assign sum = a + b;
endmodule
"""


def get_output_type(bits):
    """Get the C type for output based on bit width."""
    if bits + 1 <= 8:
        return "uint8_t"
    elif bits + 1 <= 16:
        return "uint16_t"
    elif bits + 1 <= 32:
        return "uint32_t"
    elif bits + 1 <= 64:
        return "uint64_t"
    else:
        return "__uint128_t"


def benchmark_frozen(bits):
    """Benchmark frozen shape execution."""

    verilog = generate_adder_verilog(bits)
    header, source = freeze_factored(verilog)

    # Remove the include line from source - we'll inline everything
    source_lines = source.split('\n')
    source_clean = '\n'.join(line for line in source_lines if '#include "frozen_' not in line)

    iterations = 10000000 if bits <= 8 else 1000000 if bits <= 16 else 100000
    mask = (1 << bits) - 1
    out_type = get_output_type(bits)

    # Create benchmark harness with inlined frozen code
    harness = f'''
#include <stdio.h>
#include <stdint.h>
#include <time.h>

{source_clean}

int main() {{
    struct timespec start, end;
    uint64_t iterations = {iterations};
    {out_type} result;

    // Warm up
    for (int i = 0; i < 1000; i++) {{
        frozen_adder{bits}(i, i, &result);
    }}

    clock_gettime(CLOCK_MONOTONIC, &start);

    volatile uint64_t sum = 0;
    for (uint64_t i = 0; i < iterations; i++) {{
        frozen_adder{bits}(i & {mask}ULL, (i >> {bits//2}) & {mask}ULL, &result);
        sum += result;
    }}

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double ns_per_op = (elapsed / iterations) * 1e9;

    printf("frozen,%d,%lu,%.2f\\n", {bits}, iterations, ns_per_op);

    return (int)(sum & 0xFF);  // Prevent optimization
}}
'''

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "bench.c"
        bin_path = Path(tmpdir) / "bench"

        src_path.write_text(harness)

        # Compile with optimization
        result = subprocess.run(
            ["gcc", "-O3", "-march=native", "-o", str(bin_path), str(src_path)],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"Compile error for {bits}-bit:")
            print(result.stderr[:500])
            return None

        # Run benchmark
        result = subprocess.run([str(bin_path)], capture_output=True, text=True)
        return result.stdout.strip()


def benchmark_native(bits):
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

        result = subprocess.run(
            ["gcc", "-O3", "-march=native", "-o", str(bin_path), str(src_path)],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            return None

        result = subprocess.run([str(bin_path)], capture_output=True, text=True)
        return result.stdout.strip()


if __name__ == "__main__":
    print("FROZEN SHAPES vs NATIVE ARITHMETIC")
    print("=" * 60)
    print()
    print(f"{'Type':<10} {'Bits':>6} {'Iterations':>12} {'ns/op':>10}")
    print("-" * 60)

    results = []

    for bits in [4, 8, 16, 32]:
        frozen_result = benchmark_frozen(bits)
        native_result = benchmark_native(bits)

        if frozen_result:
            parts = frozen_result.split(',')
            print(f"{'Frozen':<10} {parts[1]:>6} {parts[2]:>12} {parts[3]:>10}")
            results.append(('frozen', int(parts[1]), float(parts[3])))

        if native_result:
            parts = native_result.split(',')
            print(f"{'Native':<10} {parts[1]:>6} {parts[2]:>12} {parts[3]:>10}")
            results.append(('native', int(parts[1]), float(parts[3])))

        print()

    print("=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    # Calculate ratios
    for bits in [4, 8, 16, 32]:
        frozen_ns = next((r[2] for r in results if r[0] == 'frozen' and r[1] == bits), None)
        native_ns = next((r[2] for r in results if r[0] == 'native' and r[1] == bits), None)

        if frozen_ns and native_ns:
            ratio = frozen_ns / native_ns
            print(f"{bits:>3}-bit: Frozen is {ratio:.1f}x {'slower' if ratio > 1 else 'faster'} than native")
