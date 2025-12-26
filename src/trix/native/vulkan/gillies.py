"""
GILLIES Vulkan Python Bindings

17 billion XOR ops/sec. From Python.

Usage:
    from trix.native.vulkan.gillies import GilliesVulkan

    gpu = GilliesVulkan(max_elements=1_000_000)
    result = gpu.xor(a, b)  # NumPy arrays
    gpu.close()
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Optional

# Find the shared library
_LIB_PATH = Path(__file__).parent / "libgillies.so"

class GilliesVulkan:
    """
    GILLIES Vulkan compute interface.

    Executes frozen shapes on GPU at 17+ billion ops/sec.
    """

    def __init__(self, max_elements: int = 16_777_216):
        """
        Initialize GILLIES Vulkan context.

        Args:
            max_elements: Maximum number of elements per operation.
                         Default 16M = 64MB per buffer.
        """
        if not _LIB_PATH.exists():
            raise RuntimeError(
                f"GILLIES library not found at {_LIB_PATH}. "
                "Run 'make libgillies.so' in src/trix/native/vulkan/"
            )

        self._lib = ctypes.CDLL(str(_LIB_PATH))

        # Define function signatures
        self._lib.gillies_create.argtypes = [ctypes.c_size_t]
        self._lib.gillies_create.restype = ctypes.c_void_p

        self._lib.gillies_destroy.argtypes = [ctypes.c_void_p]
        self._lib.gillies_destroy.restype = None

        self._lib.gillies_xor.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t
        ]
        self._lib.gillies_xor.restype = ctypes.c_int

        self._lib.gillies_max_elements.argtypes = [ctypes.c_void_p]
        self._lib.gillies_max_elements.restype = ctypes.c_size_t

        # Create context
        self._ctx = self._lib.gillies_create(max_elements)
        if not self._ctx:
            raise RuntimeError("Failed to create GILLIES Vulkan context")

        self._max_elements = max_elements

    def xor(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute XOR shape: out = a + b - 2*a*b

        Args:
            a: Input array (float32)
            b: Input array (float32)

        Returns:
            Result array (float32)
        """
        # Ensure contiguous float32
        a = np.ascontiguousarray(a, dtype=np.float32)
        b = np.ascontiguousarray(b, dtype=np.float32)

        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

        n = a.size
        if n > self._max_elements:
            raise ValueError(f"Array too large: {n} > {self._max_elements}")

        # Allocate output
        out = np.empty(a.shape, dtype=np.float32)

        # Get pointers
        a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Execute
        result = self._lib.gillies_xor(self._ctx, a_ptr, b_ptr, out_ptr, n)
        if result != 0:
            raise RuntimeError("GILLIES XOR execution failed")

        return out

    def close(self):
        """Release GPU resources."""
        if self._ctx:
            self._lib.gillies_destroy(self._ctx)
            self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


def benchmark():
    """Run GILLIES Vulkan benchmark."""
    import time

    print("GILLIES Vulkan Python Benchmark")
    print("=" * 50)

    with GilliesVulkan(max_elements=16_000_000) as gpu:
        sizes = [10_000, 100_000, 1_000_000, 10_000_000]

        for size in sizes:
            a = np.random.rand(size).astype(np.float32)
            b = np.random.rand(size).astype(np.float32)

            # Warmup
            _ = gpu.xor(a, b)

            # Benchmark
            start = time.perf_counter()
            for _ in range(100):
                out = gpu.xor(a, b)
            elapsed = time.perf_counter() - start

            ops_per_sec = size * 100 / elapsed
            print(f"  Size {size:>10,}: {ops_per_sec / 1e9:.2f} B ops/sec")

        # Verify correctness
        a = np.array([0, 1, 0, 1], dtype=np.float32)
        b = np.array([0, 0, 1, 1], dtype=np.float32)
        out = gpu.xor(a, b)
        expected = np.array([0, 1, 1, 0], dtype=np.float32)

        if np.allclose(out, expected):
            print("\nCorrectness: PASSED")
        else:
            print(f"\nCorrectness: FAILED - got {out}, expected {expected}")


if __name__ == "__main__":
    benchmark()
