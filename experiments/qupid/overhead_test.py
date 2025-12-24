"""
Measure pure Python → C call overhead.
"""

import ctypes
import time

# Create a simple shared library
import subprocess
import tempfile
import os

src = """
#include <stdint.h>

extern "C" {

volatile uint32_t counter = 0;

void increment() {
    counter++;
}

void increment_n(uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        counter++;
    }
}

uint32_t get_counter() {
    return counter;
}

}
"""

# Write and compile
with tempfile.NamedTemporaryFile(suffix='.cu', delete=False) as f:
    f.write(src.encode())
    src_path = f.name

lib_path = src_path.replace('.cu', '.so')
subprocess.run(['nvcc', '-shared', '-O3', '-o', lib_path, src_path, '-Xcompiler', '-fPIC'],
               check=True, capture_output=True)

# Load library
lib = ctypes.CDLL(lib_path)
lib.increment.restype = None
lib.increment_n.argtypes = [ctypes.c_uint32]
lib.increment_n.restype = None
lib.get_counter.restype = ctypes.c_uint32

# Warm up
for _ in range(1000):
    lib.increment()

print("=" * 70)
print("PYTHON → C CALL OVERHEAD MEASUREMENT")
print("=" * 70)
print()

# Measure single call overhead
iterations = 100000
start = time.perf_counter()
for _ in range(iterations):
    lib.increment()
elapsed = time.perf_counter() - start

call_overhead_us = elapsed * 1000000 / iterations
calls_per_sec = iterations / elapsed

print(f"[Single Call]")
print(f"  {iterations:,} calls in {elapsed*1000:.1f} ms")
print(f"  Overhead: {call_overhead_us:.3f} μs/call")
print(f"  Rate: {calls_per_sec:,.0f} calls/sec")
print()

# Measure batched call (call once, do N ops in C)
N = 1024 * 1024
iterations = 1000

start = time.perf_counter()
for _ in range(iterations):
    lib.increment_n(N)
elapsed = time.perf_counter() - start

total_ops = N * iterations
ops_per_sec = total_ops / elapsed
overhead_us = elapsed * 1000000 / iterations

print(f"[Batched Call - {N:,} ops per call]")
print(f"  {iterations:,} calls, {total_ops:,} ops")
print(f"  Time: {elapsed*1000:.1f} ms")
print(f"  Overhead: {overhead_us:.3f} μs/call (includes work)")
print(f"  Rate: {ops_per_sec/1e9:.2f} B ops/sec")
print()

# What does this tell us?
print("=" * 70)
print("ANALYSIS")
print("=" * 70)
print()
print(f"Single ctypes call: ~{call_overhead_us:.1f} μs")
print()
print(f"For CUDA kernel execution:")
print(f"  Python ctypes overhead: ~{call_overhead_us:.1f} μs")
print(f"  CUDA graph launch: ~70 μs (bare metal)")
print(f"  Total Python path: ~{call_overhead_us + 70:.1f} μs")
print()
print(f"  Measured Python path: ~220 μs")
print(f"  Gap: ~{220 - 70 - call_overhead_us:.1f} μs")
print()

# Cleanup
os.unlink(src_path)
os.unlink(lib_path)
