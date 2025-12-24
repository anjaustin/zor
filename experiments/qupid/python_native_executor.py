"""
Python Native Executor

Uses the compiled CUDA executor library.
Python only does setup - execution is pure C/CUDA.
"""

import ctypes
import numpy as np
import time
from pathlib import Path

# Load the compiled library
LIB_PATH = Path(__file__).parent / "libcuda_executor.so"
lib = ctypes.CDLL(str(LIB_PATH))

# Define function signatures
lib.executor_init.argtypes = [ctypes.c_uint32]
lib.executor_init.restype = ctypes.c_int

lib.executor_get_opcodes.restype = ctypes.POINTER(ctypes.c_uint8)
lib.executor_get_operand_a.restype = ctypes.POINTER(ctypes.c_uint8)
lib.executor_get_operand_b.restype = ctypes.POINTER(ctypes.c_uint8)
lib.executor_get_carry_in.restype = ctypes.POINTER(ctypes.c_uint8)
lib.executor_get_result.restype = ctypes.POINTER(ctypes.c_uint8)
lib.executor_get_carry_out.restype = ctypes.POINTER(ctypes.c_uint8)

lib.executor_set_routing.argtypes = [ctypes.POINTER(ctypes.c_uint8)]

lib.executor_create_graph.argtypes = [ctypes.c_uint32]
lib.executor_create_graph.restype = ctypes.c_int

lib.executor_run_graph.restype = ctypes.c_int
lib.executor_run.argtypes = [ctypes.c_uint32]
lib.executor_run.restype = ctypes.c_int

lib.executor_destroy.restype = None


class NativeExecutor:
    """
    Python wrapper around the native CUDA executor.

    Data buffers are in unified memory.
    Python writes directly to GPU-accessible memory.
    Execution is pure CUDA graph replay.
    """

    def __init__(self, max_n=1024*1024):
        self.max_n = max_n

        # Initialize C executor
        result = lib.executor_init(max_n)
        if result != 0:
            raise RuntimeError("Failed to initialize executor")

        # Get buffer pointers (unified memory)
        self._opcodes_ptr = lib.executor_get_opcodes()
        self._a_ptr = lib.executor_get_operand_a()
        self._b_ptr = lib.executor_get_operand_b()
        self._c_ptr = lib.executor_get_carry_in()
        self._result_ptr = lib.executor_get_result()
        self._carry_ptr = lib.executor_get_carry_out()

        # Create numpy arrays backed by unified memory
        # This avoids copies - writes go directly to GPU memory
        self.opcodes = np.ctypeslib.as_array(self._opcodes_ptr, shape=(max_n,))
        self.operand_a = np.ctypeslib.as_array(self._a_ptr, shape=(max_n,))
        self.operand_b = np.ctypeslib.as_array(self._b_ptr, shape=(max_n,))
        self.carry_in = np.ctypeslib.as_array(self._c_ptr, shape=(max_n,))
        self.result = np.ctypeslib.as_array(self._result_ptr, shape=(max_n,))
        self.carry_out = np.ctypeslib.as_array(self._carry_ptr, shape=(max_n,))

        self.graph_n = 0

    def set_routing(self, routing):
        """Set opcode -> shape routing table."""
        table = np.zeros(256, dtype=np.uint8)
        for i, shape_id in enumerate(routing[:256]):
            table[i] = shape_id
        lib.executor_set_routing(table.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))

    def create_graph(self, n):
        """Create CUDA graph for fixed batch size."""
        result = lib.executor_create_graph(n)
        if result != 0:
            raise RuntimeError(f"Failed to create graph for n={n}")
        self.graph_n = n

    def execute_graph(self):
        """Execute pre-created graph (minimal overhead)."""
        result = lib.executor_run_graph()
        if result != 0:
            raise RuntimeError("Graph execution failed")

    def execute(self, n):
        """Execute n operations (raw kernel)."""
        result = lib.executor_run(n)
        if result != 0:
            raise RuntimeError(f"Execution failed for n={n}")

    def __del__(self):
        lib.executor_destroy()


def benchmark():
    """Benchmark the native executor."""
    print("=" * 70)
    print("PYTHON NATIVE EXECUTOR BENCHMARK")
    print("=" * 70)
    print()
    print("Python writes directly to unified memory.")
    print("Execution is pure C/CUDA graph replay.")
    print()

    N = 1024 * 1024  # 1M ops
    iterations = 1000

    # Create executor
    print(f"Initializing executor (max_n={N:,})...")
    executor = NativeExecutor(max_n=N)

    # Set routing
    routing = list(range(11)) + [15] * 245
    executor.set_routing(routing)

    # Fill buffers with random data
    print("Filling buffers...")
    executor.opcodes[:] = np.random.randint(0, 11, N, dtype=np.uint8)
    executor.operand_a[:] = np.random.randint(0, 256, N, dtype=np.uint8)
    executor.operand_b[:] = np.random.randint(0, 256, N, dtype=np.uint8)
    executor.carry_in[:] = np.random.randint(0, 2, N, dtype=np.uint8)

    # Create graph
    print("Creating CUDA graph...")
    executor.create_graph(N)

    # Warm up
    print("Warming up...")
    for _ in range(10):
        executor.execute_graph()

    # Benchmark graph execution
    print(f"\nBenchmarking graph execution ({iterations} iterations)...")

    start = time.perf_counter()
    for _ in range(iterations):
        executor.execute_graph()
    elapsed = time.perf_counter() - start

    ops_per_sec = (N * iterations) / elapsed
    latency_us = elapsed * 1000000 / iterations

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

    # Compare to old approach
    print("Comparison:")
    print(f"  Pure Python ctypes: ~4.5 B ops/sec")
    print(f"  Python + native lib: {ops_per_sec/1e9:.2f} B ops/sec")
    print(f"  Pure C/CUDA: ~15 B ops/sec")
    print()
    print(f"  Speedup vs ctypes: {ops_per_sec/4.5e9:.2f}x")
    print(f"  Gap to bare metal: {15e9/ops_per_sec:.2f}x")
    print()


if __name__ == "__main__":
    benchmark()
