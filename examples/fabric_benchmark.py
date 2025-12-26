#!/usr/bin/env python3
"""
TriX Fabric Benchmarks

Measuring the performance of the unified database-computer.
"""

import sys
import time
from typing import List, Tuple

sys.path.insert(0, 'src')

from trix.fabric import Fabric, FabricProcessor, Message, parse_fdl
from trix.fabric.query import fabric_stats


def benchmark(name: str, func, iterations: int = 1) -> Tuple[float, any]:
    """Run a benchmark and return (time_seconds, result)."""
    start = time.perf_counter()
    for _ in range(iterations):
        result = func()
    elapsed = time.perf_counter() - start
    return elapsed, result


def format_rate(count: int, seconds: float) -> str:
    """Format operations per second."""
    if seconds == 0:
        return "inf"
    rate = count / seconds
    if rate >= 1e9:
        return f"{rate/1e9:.2f}B"
    elif rate >= 1e6:
        return f"{rate/1e6:.2f}M"
    elif rate >= 1e3:
        return f"{rate/1e3:.2f}K"
    else:
        return f"{rate:.2f}"


def separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def bench_fabric_creation(sizes: List[int]):
    """Benchmark fabric creation at various sizes."""
    separator("Fabric Creation")
    print(f"{'Processors':<12} {'Time (ms)':<12} {'Rate (proc/s)':<15}")
    print("-" * 40)

    for size in sizes:
        def create():
            fabric = Fabric(name=f"bench_{size}")
            for i in range(size):
                cluster = i // 256
                local = i % 256
                fabric.add_processor(FabricProcessor(
                    signature=f"/c{cluster}/0b{local:08b}",
                    shape="xor",
                    name=f"p{i}"
                ))
            return fabric

        elapsed, fabric = benchmark(f"create_{size}", create)
        rate = format_rate(size, elapsed)
        print(f"{size:<12} {elapsed*1000:<12.2f} {rate:<15}")


def bench_message_injection(sizes: List[int]):
    """Benchmark message injection and execution."""
    separator("Message Injection (single processor)")
    print(f"{'Messages':<12} {'Time (ms)':<12} {'Rate (msg/s)':<15}")
    print("-" * 40)

    # Create a simple fabric
    fabric = Fabric(name="bench_inject")
    fabric.add_processor(FabricProcessor(
        signature="/0/0b00",
        shape="xor",
        name="gate"
    ))
    fabric.set_export("/0/0b00")

    for count in sizes:
        def inject():
            for i in range(count):
                msg = Message(
                    target="/0/0b00",
                    payload={"a": 0.5, "b": 0.5}
                )
                fabric.inject(msg)

        elapsed, _ = benchmark(f"inject_{count}", inject)
        rate = format_rate(count, elapsed)
        print(f"{count:<12} {elapsed*1000:<12.2f} {rate:<15}")


def bench_chain_execution(depths: List[int]):
    """Benchmark execution through processor chains."""
    separator("Chain Execution (depth varies)")
    print(f"{'Depth':<12} {'Time (ms)':<12} {'Rate (eval/s)':<15}")
    print("-" * 40)

    for depth in depths:
        # Create a chain of processors
        fabric = Fabric(name=f"chain_{depth}")

        for i in range(depth):
            proc = FabricProcessor(
                signature=f"/0/0b{i:08b}",
                shape="xor",
                name=f"p{i}"
            )
            fabric.add_processor(proc)

            if i > 0:
                fabric.connect(f"/0/0b{i-1:08b}", f"/0/0b{i:08b}", "result", "a")

        fabric.set_export(f"/0/0b{depth-1:08b}")

        iterations = 1000
        def execute():
            for _ in range(iterations):
                msg = Message(target="/0/0b00", payload={"a": 1.0, "b": 0.0})
                fabric.inject(msg)

        elapsed, _ = benchmark(f"chain_{depth}", execute)
        total_evals = iterations * depth
        rate = format_rate(total_evals, elapsed)
        print(f"{depth:<12} {elapsed*1000:<12.2f} {rate:<15}")


def bench_parallel_execution(widths: List[int]):
    """Benchmark parallel execution (many independent processors)."""
    separator("Parallel Execution (width varies)")
    print(f"{'Width':<12} {'Time (ms)':<12} {'Rate (eval/s)':<15}")
    print("-" * 40)

    for width in widths:
        fabric = Fabric(name=f"parallel_{width}")

        for i in range(width):
            fabric.add_processor(FabricProcessor(
                signature=f"/0/0b{i:08b}",
                shape="xor",
                name=f"p{i}"
            ))
            fabric.set_export(f"/0/0b{i:08b}")

        iterations = 100
        def execute():
            for _ in range(iterations):
                inputs = {
                    f"/0/0b{i:08b}": {"a": 0.5, "b": 0.5}
                    for i in range(width)
                }
                fabric.execute(inputs)

        elapsed, _ = benchmark(f"parallel_{width}", execute)
        total_evals = iterations * width
        rate = format_rate(total_evals, elapsed)
        print(f"{width:<12} {elapsed*1000:<12.2f} {rate:<15}")


def bench_fdl_parsing(sizes: List[int]):
    """Benchmark FDL parsing."""
    separator("FDL Parsing")
    print(f"{'Processors':<12} {'Time (ms)':<12} {'Rate (proc/s)':<15}")
    print("-" * 40)

    for size in sizes:
        # Generate FDL source
        lines = [f'fabric "bench_{size}" {{']
        for i in range(size):
            lines.append(f'    processor p{i} {{')
            lines.append(f'        shape: xor')
            lines.append(f'        signature: /c{i//256}/0b{i%256:08b}')
            lines.append(f'    }}')
        lines.append('}')
        fdl_source = '\n'.join(lines)

        def parse():
            return parse_fdl(fdl_source)

        elapsed, fabric = benchmark(f"parse_{size}", parse)
        rate = format_rate(size, elapsed)
        print(f"{size:<12} {elapsed*1000:<12.2f} {rate:<15}")


def bench_query(sizes: List[int]):
    """Benchmark fabric queries."""
    separator("Fabric Query")
    print(f"{'Fabric Size':<12} {'Query Time (ms)':<15} {'Rate (query/s)':<15}")
    print("-" * 45)

    from trix.fabric import query_fabric

    for size in sizes:
        # Create fabric
        fabric = Fabric(name=f"query_{size}")
        for i in range(size):
            shape = ["xor", "and", "or", "not"][i % 4]
            fabric.add_processor(FabricProcessor(
                signature=f"/c{i//100}/0b{i%256:08b}",
                shape=shape,
                name=f"p{i}"
            ))

        iterations = 1000
        def query():
            for _ in range(iterations):
                query_fabric(fabric, shape="xor")

        elapsed, _ = benchmark(f"query_{size}", query)
        rate = format_rate(iterations, elapsed)
        print(f"{size:<12} {elapsed*1000:<15.2f} {rate:<15}")


def bench_routing(sizes: List[int]):
    """Benchmark Hamming routing."""
    separator("Hamming Routing")
    print(f"{'Fabric Size':<12} {'Routes':<10} {'Time (ms)':<12} {'Rate (route/s)':<15}")
    print("-" * 50)

    for size in sizes:
        fabric = Fabric(name=f"route_{size}")
        for i in range(size):
            fabric.add_processor(FabricProcessor(
                signature=f"/0/0b{i:08b}",
                shape="xor",
                name=f"p{i}"
            ))

        routes = 10000
        def route():
            import random
            for _ in range(routes):
                target = f"/0/0b{random.randint(0, 255):08b}"
                fabric.router.route(target)

        elapsed, _ = benchmark(f"route_{size}", route)
        rate = format_rate(routes, elapsed)
        print(f"{size:<12} {routes:<10} {elapsed*1000:<12.2f} {rate:<15}")


def main():
    print("=" * 60)
    print("  TRIX FABRIC BENCHMARKS")
    print("  Measuring the unified database-computer")
    print("=" * 60)

    # Fabric creation
    bench_fabric_creation([100, 500, 1000, 5000])

    # Message injection
    bench_message_injection([1000, 10000, 100000])

    # Chain execution
    bench_chain_execution([1, 5, 10, 20, 50])

    # Parallel execution
    bench_parallel_execution([10, 50, 100, 200])

    # FDL parsing
    bench_fdl_parsing([100, 500, 1000])

    # Query
    bench_query([100, 500, 1000, 5000])

    # Routing
    bench_routing([16, 64, 128, 256])

    separator("SUMMARY")
    print("""
TriX Fabric Performance (Python/HSOS layer):

This is the DEVELOPMENT tier (Layer 1: HSOS).
For production performance, compile to:
  - GILLIES (C):     ~280x faster
  - GILLIES Vulkan:  ~6000x faster

The shape polynomial a + b - 2ab computes identically
at all tiers. Only the substrate changes.
""")


if __name__ == "__main__":
    main()
