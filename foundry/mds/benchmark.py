"""
MDS Benchmark: Navigation vs Arithmetic
"""

import time
from lattice import build_adder_lattice

def benchmark_bits(bits):
    """Benchmark a specific bit width."""
    lattice = build_adder_lattice(bits)
    max_val = 2 ** bits
    total_cases = max_val * max_val

    # Navigation
    start = time.perf_counter()
    for a in range(max_val):
        for b in range(max_val):
            inputs = {f'a{i}': (a >> i) & 1 for i in range(bits)}
            inputs.update({f'b{i}': (b >> i) & 1 for i in range(bits)})
            lattice.navigate(inputs)
    nav_time = time.perf_counter() - start

    # Arithmetic
    start = time.perf_counter()
    for a in range(max_val):
        for b in range(max_val):
            _ = a + b
    arith_time = time.perf_counter() - start

    return {
        'bits': bits,
        'cases': total_cases,
        'nodes': lattice.node_count(),
        'nav_time': nav_time,
        'arith_time': arith_time,
        'nav_per_op': nav_time / total_cases * 1e6,  # microseconds
        'arith_per_op': arith_time / total_cases * 1e6,
    }

print("MDS BENCHMARK")
print("=" * 60)
print()
print("SOFTWARE SIMULATION (Python)")
print("-" * 60)
print(f"{'Bits':>6} {'Cases':>10} {'Nodes':>8} {'Nav/op':>12} {'Arith/op':>12}")
print("-" * 60)

for bits in [2, 4, 6, 8]:
    r = benchmark_bits(bits)
    print(f"{r['bits']:>6} {r['cases']:>10,} {r['nodes']:>8} {r['nav_per_op']:>10.2f}us {r['arith_per_op']:>10.2f}us")

print()
print("=" * 60)
print("ANALYSIS")
print("=" * 60)
print()
print("Navigation is slower in Python because:")
print("  - Dictionary lookups per node")
print("  - Object attribute access")
print("  - Method call overhead")
print("  - Input bit unpacking")
print()
print("In HARDWARE (silicon):")
print("  - Navigation = wire propagation (picoseconds)")
print("  - No computation, just signal routing")
print("  - Latency = gate depth, not operation count")
print()
print("The insight isn't speed. It's STRUCTURE.")
print("The lattice contains the answers. We route to them.")
