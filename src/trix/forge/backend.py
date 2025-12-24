"""
Hardware backend interface for XORPU execution.

The backend layer abstracts hardware execution. Application code uses
the same interface regardless of whether shapes execute on CUDA, FPGA,
or custom silicon.

Usage:
    from trix.forge.backend import get_backend, CUDABackend

    backend = get_backend()  # Auto-detect best available
    result = backend.execute("xor", a, b)

    # Or explicitly:
    backend = CUDABackend()
    results = backend.execute_batch("and", a_batch, b_batch)

The Three Atoms execute identically across all backends:
    - XOR(a, b) = a + b - 2ab  →  a ^ b
    - AND(a, b) = ab           →  a & b
    - NOT(a)    = 1 - a        →  ~a
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import subprocess
import tempfile
import struct
from pathlib import Path

from .term import (
    ShapeTerms, generate_all_shapes, generate_shape,
    SHAPE_GENERATORS, FAST_PATH_SHAPES, compute_fast,
    evaluate_batch, evaluate_batch_fast,
)
from .cuda import shape_to_cuda, export_cuda, generate_cuda_header


# =============================================================================
# BACKEND INTERFACE
# =============================================================================

@dataclass
class BackendInfo:
    """Information about a hardware backend."""
    name: str
    device: str
    available: bool
    compute_units: int = 0
    memory_bytes: int = 0
    driver_version: str = ""


class Backend(ABC):
    """Abstract base class for hardware backends."""

    @abstractmethod
    def info(self) -> BackendInfo:
        """Get backend information."""
        pass

    @abstractmethod
    def available_shapes(self) -> List[str]:
        """List shapes this backend can execute."""
        pass

    @abstractmethod
    def execute(self, shape: str, a: int, b: int = 0, bits: int = 32) -> int:
        """Execute a shape on single inputs."""
        pass

    @abstractmethod
    def execute_batch(
        self,
        shape: str,
        a_batch: List[int],
        b_batch: Optional[List[int]] = None,
        bits: int = 32
    ) -> List[int]:
        """Execute a shape on batched inputs."""
        pass

    def compile_shapes(self, shapes: List[str], bits: int = 32) -> bool:
        """Pre-compile shapes for faster execution. Returns success."""
        return True  # Default: no-op

    def supports_shape(self, shape: str) -> bool:
        """Check if backend supports a shape."""
        return shape.lower() in self.available_shapes()


# =============================================================================
# CPU BACKEND (Reference Implementation)
# =============================================================================

class CPUBackend(Backend):
    """
    CPU backend using Python evaluation.

    This is the reference implementation. Always available, always correct,
    but slower than hardware backends.
    """

    def __init__(self, bits: int = 32):
        self.bits = bits
        self._shapes: Dict[str, ShapeTerms] = {}

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="CPU",
            device="Python interpreter",
            available=True,
            compute_units=1,
        )

    def available_shapes(self) -> List[str]:
        return list(SHAPE_GENERATORS.keys())

    def _get_shape(self, name: str, bits: int) -> ShapeTerms:
        """Get or generate a shape."""
        key = f"{name}_{bits}"
        if key not in self._shapes:
            self._shapes[key] = generate_shape(name, bits)
        return self._shapes[key]

    def execute(self, shape: str, a: int, b: int = 0, bits: int = 32) -> int:
        """Execute using fast path or polynomial evaluation."""
        shape_lower = shape.lower()

        # Fast path for simple shapes
        if shape_lower in FAST_PATH_SHAPES:
            return compute_fast(shape_lower, a, b, bits)

        # Polynomial evaluation via batch (single element)
        shape_terms = self._get_shape(shape_lower, bits)
        results = evaluate_batch(shape_terms, [a], [b])
        return results[0]

    def execute_batch(
        self,
        shape: str,
        a_batch: List[int],
        b_batch: Optional[List[int]] = None,
        bits: int = 32
    ) -> List[int]:
        """Execute on batch using optimized evaluation."""
        shape_lower = shape.lower()

        # Fast path for simple shapes
        if shape_lower in FAST_PATH_SHAPES:
            return evaluate_batch_fast(shape_lower, a_batch, b_batch, bits)

        # Polynomial batch evaluation
        if b_batch is None:
            b_batch = [0] * len(a_batch)
        shape_terms = self._get_shape(shape_lower, bits)
        return evaluate_batch(shape_terms, a_batch, b_batch)


# =============================================================================
# CUDA BACKEND
# =============================================================================

class CUDABackend(Backend):
    """
    CUDA backend for GPU execution.

    Compiles shapes to CUDA kernels and executes on NVIDIA GPU.
    """

    def __init__(self, bits: int = 32, cache_dir: Optional[str] = None):
        self.bits = bits
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="xorpu_cuda_"))
        self._compiled = False
        self._binary_path: Optional[Path] = None
        self._gpu_info: Optional[BackendInfo] = None
        self._shapes: Dict[str, ShapeTerms] = {}

        # Check CUDA availability
        self._cuda_available = self._check_cuda()

    def _check_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(", ")
                self._gpu_info = BackendInfo(
                    name="CUDA",
                    device=parts[0] if parts else "Unknown GPU",
                    available=True,
                    memory_bytes=self._parse_memory(parts[1]) if len(parts) > 1 else 0,
                )
                # Get SM count
                sm_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                return True
            return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _parse_memory(self, mem_str: str) -> int:
        """Parse memory string like '8192 MiB' to bytes."""
        try:
            parts = mem_str.strip().split()
            value = float(parts[0])
            if len(parts) > 1 and 'MiB' in parts[1]:
                return int(value * 1024 * 1024)
            elif len(parts) > 1 and 'GiB' in parts[1]:
                return int(value * 1024 * 1024 * 1024)
            return int(value)
        except:
            return 0

    def info(self) -> BackendInfo:
        if self._gpu_info:
            return self._gpu_info
        return BackendInfo(
            name="CUDA",
            device="Not available",
            available=False,
        )

    def available_shapes(self) -> List[str]:
        # All shapes can be compiled to CUDA
        return list(SHAPE_GENERATORS.keys())

    def compile_shapes(self, shapes: List[str], bits: int = 32) -> bool:
        """Compile shapes to CUDA binary."""
        if not self._cuda_available:
            return False

        self.bits = bits

        # Generate shape terms
        shape_dict = {}
        for name in shapes:
            shape_dict[name] = generate_shape(name, bits)
            self._shapes[name] = shape_dict[name]

        # Export to CUDA
        export_cuda(shape_dict, str(self.cache_dir))

        # Compile
        try:
            result = subprocess.run(
                ["make", "-C", str(self.cache_dir)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                self._binary_path = self.cache_dir / "xorpu_test"
                self._compiled = True
                return True
            else:
                print(f"Compilation failed: {result.stderr}")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Compilation error: {e}")
            return False

    def execute(self, shape: str, a: int, b: int = 0, bits: int = 32) -> int:
        """Execute single operation via CUDA."""
        # For single operations, CPU is actually faster due to kernel launch overhead
        # Use CPU backend for single ops, CUDA for batches
        return CPUBackend(bits).execute(shape, a, b, bits)

    def execute_batch(
        self,
        shape: str,
        a_batch: List[int],
        b_batch: Optional[List[int]] = None,
        bits: int = 32
    ) -> List[int]:
        """Execute batch via CUDA kernel."""
        if not self._cuda_available:
            # Fallback to CPU
            return CPUBackend(bits).execute_batch(shape, a_batch, b_batch, bits)

        # For small batches, CPU is faster
        if len(a_batch) < 1000:
            return CPUBackend(bits).execute_batch(shape, a_batch, b_batch, bits)

        # TODO: Implement proper CUDA batch execution with pycuda or ctypes
        # For now, fall back to CPU for correctness
        return CPUBackend(bits).execute_batch(shape, a_batch, b_batch, bits)


# =============================================================================
# BACKEND REGISTRY
# =============================================================================

_backends: Dict[str, type] = {
    "cpu": CPUBackend,
    "cuda": CUDABackend,
}


def register_backend(name: str, backend_class: type):
    """Register a new backend."""
    _backends[name.lower()] = backend_class


def get_backend(name: Optional[str] = None, **kwargs) -> Backend:
    """
    Get a backend by name, or auto-detect best available.

    Args:
        name: Backend name ("cpu", "cuda") or None for auto-detect
        **kwargs: Passed to backend constructor

    Returns:
        Backend instance
    """
    if name:
        name = name.lower()
        if name not in _backends:
            raise ValueError(f"Unknown backend: {name}. Available: {list(_backends.keys())}")
        return _backends[name](**kwargs)

    # Auto-detect: try CUDA first, fall back to CPU
    cuda = CUDABackend(**kwargs)
    if cuda.info().available:
        return cuda
    return CPUBackend(**kwargs)


def list_backends() -> Dict[str, BackendInfo]:
    """List all registered backends and their availability."""
    result = {}
    for name, cls in _backends.items():
        try:
            backend = cls()
            result[name] = backend.info()
        except Exception as e:
            result[name] = BackendInfo(name=name, device=str(e), available=False)
    return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Interface
    "Backend",
    "BackendInfo",

    # Implementations
    "CPUBackend",
    "CUDABackend",

    # Registry
    "register_backend",
    "get_backend",
    "list_backends",
]
