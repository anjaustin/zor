"""
GILLIES Vulkan Frozen Shapes

Drop-in replacement for trix.native.FrozenShapes using GPU acceleration.
17 billion ops/sec on Thor GPU.

Usage:
    from trix.native.vulkan.shapes import GilliesFrozenShapes

    shapes = GilliesFrozenShapes()
    result = shapes.xor(a, b)  # GPU-accelerated
"""

import numpy as np
from typing import Optional
from .gillies import GilliesVulkan


class GilliesFrozenShapes:
    """
    GPU-accelerated frozen shapes via GILLIES Vulkan.

    Same interface as trix.native.FrozenShapes, but runs on GPU.
    Falls back to CPU for shapes not yet implemented in GILLIES.
    """

    def __init__(self, max_elements: int = 16_777_216):
        """
        Initialize GPU-accelerated shapes.

        Args:
            max_elements: Maximum elements per operation (default 16M).
        """
        self._gpu = GilliesVulkan(max_elements=max_elements)
        self._max_elements = max_elements

    def xor(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """XOR shape: a + b - 2*a*b (GPU)"""
        return self._gpu.xor(a, b)

    def and_op(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """AND shape: a * b (CPU fallback)"""
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        return a * b

    def or_op(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """OR shape: a + b - a*b (CPU fallback)"""
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        return a + b - a * b

    def not_op(self, a: np.ndarray) -> np.ndarray:
        """NOT shape: 1 - a (CPU fallback)"""
        a = np.asarray(a, dtype=np.float32)
        return 1.0 - a

    def nand(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """NAND shape: 1 - a*b (CPU fallback)"""
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        return 1.0 - a * b

    def nor(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """NOR shape: 1 - (a + b - a*b) (CPU fallback)"""
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        return 1.0 - (a + b - a * b)

    def xnor(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """XNOR shape: 1 - (a + b - 2*a*b) (CPU fallback)"""
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        return 1.0 - (a + b - 2 * a * b)

    def half_adder(self, a: np.ndarray, b: np.ndarray):
        """Half adder: sum=XOR(a,b), carry=AND(a,b)"""
        s = self.xor(a, b)
        c = self.and_op(a, b)
        return s, c

    def full_adder(self, a: np.ndarray, b: np.ndarray, c_in: np.ndarray):
        """Full adder: sum, carry_out"""
        # sum = a XOR b XOR c_in
        # carry = (a AND b) OR (c_in AND (a XOR b))
        ab_xor = self.xor(a, b)
        s = self.xor(ab_xor, c_in)
        ab_and = self.and_op(a, b)
        c_ab = self.and_op(c_in, ab_xor)
        c_out = self.or_op(ab_and, c_ab)
        return s, c_out

    def close(self):
        """Release GPU resources."""
        if self._gpu:
            self._gpu.close()
            self._gpu = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()


def benchmark_comparison():
    """Compare GILLIES vs NumPy performance."""
    import time
    from trix.native import FrozenShapes

    print("GILLIES vs NumPy Comparison")
    print("=" * 60)

    gpu_shapes = GilliesFrozenShapes()
    cpu_shapes = FrozenShapes()

    size = 1_000_000
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)

    # Warmup
    _ = gpu_shapes.xor(a, b)
    _ = cpu_shapes.xor(a, b)

    # Benchmark XOR
    start = time.perf_counter()
    for _ in range(100):
        _ = gpu_shapes.xor(a, b)
    gpu_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(100):
        _ = cpu_shapes.xor(a, b)
    cpu_time = time.perf_counter() - start

    gpu_rate = size * 100 / gpu_time / 1e9
    cpu_rate = size * 100 / cpu_time / 1e9

    print(f"XOR ({size:,} elements, 100 iterations):")
    print(f"  GILLIES Vulkan: {gpu_rate:.2f} B ops/sec")
    print(f"  NumPy CPU:      {cpu_rate:.2f} B ops/sec")
    print(f"  Speedup:        {gpu_rate/cpu_rate:.1f}x")

    gpu_shapes.close()


if __name__ == "__main__":
    benchmark_comparison()
