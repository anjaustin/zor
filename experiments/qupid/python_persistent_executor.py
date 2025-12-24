"""
Python Persistent Executor

Uses a persistent worker thread.
Python just sets a memory flag - no function call overhead in hot path.
"""

import ctypes
import numpy as np
import time
from pathlib import Path

# Load the compiled library
LIB_PATH = Path(__file__).parent / "libcuda_persistent.so"
lib = ctypes.CDLL(str(LIB_PATH))

# Define function signatures
lib.persistent_init.argtypes = [ctypes.c_uint32]
lib.persistent_init.restype = ctypes.c_int

lib.persistent_get_opcodes.restype = ctypes.POINTER(ctypes.c_uint8)
lib.persistent_get_operand_a.restype = ctypes.POINTER(ctypes.c_uint8)
lib.persistent_get_operand_b.restype = ctypes.POINTER(ctypes.c_uint8)
lib.persistent_get_carry_in.restype = ctypes.POINTER(ctypes.c_uint8)
lib.persistent_get_result.restype = ctypes.POINTER(ctypes.c_uint8)
lib.persistent_get_carry_out.restype = ctypes.POINTER(ctypes.c_uint8)

lib.persistent_submit.argtypes = [ctypes.c_uint32]
lib.persistent_wait.restype = None

lib.persistent_get_total_executions.restype = ctypes.c_uint64
lib.persistent_get_total_ops.restype = ctypes.c_uint64

lib.persistent_destroy.restype = None


class PersistentExecutor:
    """
    Python wrapper around the persistent CUDA executor.

    A worker thread polls for work continuously.
    Python just sets a memory flag - minimal overhead.
    """

    def __init__(self, max_n=1024*1024):
        self.max_n = max_n

        # Initialize executor (starts worker thread)
        print("Initializing persistent executor...")
        result = lib.persistent_init(max_n)
        if result != 0:
            raise RuntimeError("Failed to initialize executor")

        # Get buffer pointers
        self._opcodes_ptr = lib.persistent_get_opcodes()
        self._a_ptr = lib.persistent_get_operand_a()
        self._b_ptr = lib.persistent_get_operand_b()
        self._c_ptr = lib.persistent_get_carry_in()
        self._result_ptr = lib.persistent_get_result()
        self._carry_ptr = lib.persistent_get_carry_out()

        # Create numpy arrays backed by unified memory
        self.opcodes = np.ctypeslib.as_array(self._opcodes_ptr, shape=(max_n,))
        self.operand_a = np.ctypeslib.as_array(self._a_ptr, shape=(max_n,))
        self.operand_b = np.ctypeslib.as_array(self._b_ptr, shape=(max_n,))
        self.carry_in = np.ctypeslib.as_array(self._c_ptr, shape=(max_n,))
        self.result = np.ctypeslib.as_array(self._result_ptr, shape=(max_n,))
        self.carry_out = np.ctypeslib.as_array(self._carry_ptr, shape=(max_n,))

    def execute(self, n):
        """
        Execute n operations.

        This just sets a memory flag - the worker thread does the rest.
        """
        lib.persistent_submit(n)
        lib.persistent_wait()

    def get_stats(self):
        """Get execution statistics."""
        return {
            'total_executions': lib.persistent_get_total_executions(),
            'total_ops': lib.persistent_get_total_ops()
        }

    def __del__(self):
        print("Destroying persistent executor...")
        lib.persistent_destroy()


def benchmark():
    """Benchmark the persistent executor."""
    print("=" * 70)
    print("PYTHON PERSISTENT EXECUTOR BENCHMARK")
    print("=" * 70)
    print()
    print("Worker thread polls continuously.")
    print("Python just sets a memory flag to submit work.")
    print()

    N = 1024 * 1024  # 1M ops
    iterations = 1000

    # Create executor
    executor = PersistentExecutor(max_n=N)

    # Fill buffers with random data
    print("Filling buffers...")
    executor.opcodes[:] = np.random.randint(0, 11, N, dtype=np.uint8)
    executor.operand_a[:] = np.random.randint(0, 256, N, dtype=np.uint8)
    executor.operand_b[:] = np.random.randint(0, 256, N, dtype=np.uint8)
    executor.carry_in[:] = np.random.randint(0, 2, N, dtype=np.uint8)

    # Warm up
    print("Warming up...")
    for _ in range(10):
        executor.execute(N)

    # Benchmark
    print(f"\nBenchmarking ({iterations} iterations)...")

    start = time.perf_counter()
    for _ in range(iterations):
        executor.execute(N)
    elapsed = time.perf_counter() - start

    ops_per_sec = (N * iterations) / elapsed
    latency_us = elapsed * 1000000 / iterations

    stats = executor.get_stats()

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"  N = {N:,}")
    print(f"  Iterations = {iterations:,}")
    print(f"  Total time = {elapsed*1000:.1f} ms")
    print(f"  Throughput = {ops_per_sec/1e9:.2f} B ops/sec")
    print(f"  Latency = {latency_us:.1f} μs/execution")
    print()
    print(f"  Worker stats: {stats['total_executions']:,} executions, {stats['total_ops']:,} ops")
    print()

    # Comparison
    print("Comparison:")
    print(f"  Pure Python ctypes graph:  4.78 B ops/sec (219 μs)")
    print(f"  Python native lib:         4.63 B ops/sec (227 μs)")
    print(f"  Python persistent:         {ops_per_sec/1e9:.2f} B ops/sec ({latency_us:.0f} μs)")
    print(f"  Pure C/CUDA:              15.02 B ops/sec (70 μs)")
    print()


if __name__ == "__main__":
    benchmark()
