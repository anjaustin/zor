"""
Compare Python ctypes path vs C subprocess path.
Same operations, different entry points.
"""

import ctypes
import time
import subprocess
import numpy as np
from pathlib import Path

# Use the existing executor library
LIB_PATH = Path(__file__).parent / "libcuda_executor.so"
lib = ctypes.CDLL(str(LIB_PATH))

lib.executor_init.argtypes = [ctypes.c_uint32]
lib.executor_init.restype = ctypes.c_int
lib.executor_create_graph.argtypes = [ctypes.c_uint32]
lib.executor_create_graph.restype = ctypes.c_int
lib.executor_run_graph.restype = ctypes.c_int
lib.executor_destroy.restype = None

N = 1024 * 1024
iterations = 1000

print("=" * 70)
print("COMPARING PYTHON VS BARE METAL PATHS")
print("=" * 70)
print()

# Initialize
lib.executor_init(N)
lib.executor_create_graph(N)

# Warm up
for _ in range(10):
    lib.executor_run_graph()

# Benchmark Python path
start = time.perf_counter()
for _ in range(iterations):
    lib.executor_run_graph()
elapsed = time.perf_counter() - start

python_us = elapsed * 1000000 / iterations
python_ops = N * iterations / elapsed

print(f"[Python ctypes path]")
print(f"  Latency: {python_us:.1f} μs/execution")
print(f"  Throughput: {python_ops/1e9:.2f} B ops/sec")
print()

# Cleanup
lib.executor_destroy()

# Now run the bare metal benchmark and capture its output
print("[Bare metal path]")
result = subprocess.run(
    ['./cuda_bare_metal'],
    capture_output=True,
    text=True
)

# Parse output to get graph timing
for line in result.stdout.split('\n'):
    if 'CUDA Graph' in line or 'Throughput' in line or 'Latency' in line:
        if 'Throughput' in line:
            parts = line.strip().split()
            for i, p in enumerate(parts):
                if 'B' in p and i > 0:
                    bare_ops = float(parts[i-1])
                    print(f"  Throughput: {bare_ops:.2f} B ops/sec")
        if 'Latency' in line:
            parts = line.strip().split()
            for i, p in enumerate(parts):
                if 'μs' in p and i > 0:
                    bare_us = float(parts[i-1])
                    print(f"  Latency: {bare_us:.1f} μs/kernel")

print()
print("=" * 70)
print("ANALYSIS")
print("=" * 70)
print()
print(f"Python path:     {python_us:.1f} μs")
print(f"Bare metal:      ~72 μs (from earlier benchmark)")
print(f"Overhead:        {python_us - 72:.1f} μs ({(python_us - 72) / 72 * 100:.0f}% slower)")
print()
print("The Python overhead comes from:")
print("  1. GIL acquisition/release around ctypes calls")
print("  2. ctypes FFI marshalling")
print("  3. Python object creation for return values")
print("  4. Interpreter loop overhead")
print()
print("To eliminate this, we need:")
print("  1. A C extension module (Cython/pybind11) - reduces ctypes overhead")
print("  2. Batching - amortize Python overhead over many ops")
print("  3. Async execution - overlap Python with GPU")
print()
