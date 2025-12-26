"""
TriX Native Ops - Python Bindings

Self-hosted computation primitives via ctypes.
No external math libraries. Ternary weights = routing, not multiplication.

Usage:
    from trix.native.ops import TrixOps
    
    ops = TrixOps()
    
    # Ternary matrix-vector (no actual multiplication)
    y = ops.ternary_matvec(weights, x, scale)
    
    # Tile forward/backward
    tile = ops.create_tile(d_model=64, d_hidden=128)
    y = ops.tile_forward(tile, x)
    d_x = ops.tile_backward(tile, d_y)
    
    # Hamming routing
    best_idx = ops.route_hamming(input_binary, signatures)
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# Find the shared library
_LIB_PATH = Path(__file__).parent / "libtrix_ops.so"


class TrixOps:
    """
    Python interface to TriX native operations.
    
    All computation uses ternary weights {-1, 0, +1} encoded as bitmasks.
    Matrix multiply becomes routing: no actual multiplication needed.
    """
    
    def __init__(self):
        if not _LIB_PATH.exists():
            raise RuntimeError(
                f"libtrix_ops.so not found at {_LIB_PATH}. "
                "Run 'make' in src/trix/native/ops/ to build."
            )
        
        self._lib = ctypes.CDLL(str(_LIB_PATH))
        self._setup_functions()
        self._lib.trix_ops_init()
    
    def _setup_functions(self):
        """Configure ctypes function signatures."""
        
        # Version and SIMD checks
        self._lib.trix_ops_version.restype = ctypes.c_char_p
        self._lib.trix_has_neon.restype = ctypes.c_int
        self._lib.trix_has_avx2.restype = ctypes.c_int
        
        # Ternary matvec
        self._lib.trix_ternary_matvec.argtypes = [
            ctypes.c_void_p,  # TernaryMatrix*
            ctypes.POINTER(ctypes.c_float),  # x
            ctypes.POINTER(ctypes.c_float),  # y
            ctypes.POINTER(ctypes.c_float),  # scale
        ]
        
        # Ternary matmul
        self._lib.trix_ternary_matmul.argtypes = [
            ctypes.c_void_p,  # TernaryMatrix*
            ctypes.POINTER(ctypes.c_float),  # X
            ctypes.POINTER(ctypes.c_float),  # Y
            ctypes.POINTER(ctypes.c_float),  # scale
            ctypes.c_size_t,  # batch
        ]
        
        # Tile operations
        self._lib.trix_tile_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        self._lib.trix_tile_create.restype = ctypes.c_void_p
        
        self._lib.trix_tile_free.argtypes = [ctypes.c_void_p]
        
        self._lib.trix_tile_forward.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        
        self._lib.trix_tile_backward.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        
        # Reductions
        self._lib.trix_sum.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        self._lib.trix_sum.restype = ctypes.c_float
        
        self._lib.trix_norm.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        self._lib.trix_norm.restype = ctypes.c_float
        
        # Routing
        self._lib.trix_hamming_distance.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        self._lib.trix_hamming_distance.restype = ctypes.c_size_t
        
        self._lib.trix_route_hamming.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        self._lib.trix_route_hamming.restype = ctypes.c_size_t
        
        self._lib.trix_binarize.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
    
    @property
    def version(self) -> str:
        return self._lib.trix_ops_version().decode('utf-8')
    
    @property
    def has_neon(self) -> bool:
        return bool(self._lib.trix_has_neon())
    
    @property
    def has_avx2(self) -> bool:
        return bool(self._lib.trix_has_avx2())
    
    @property
    def simd(self) -> str:
        """Return active SIMD instruction set."""
        if self.has_neon:
            return "NEON"
        elif self.has_avx2:
            return "AVX2"
        else:
            return "scalar"
    
    # =========================================================================
    # Frozen Shapes (pure Python, matching C implementation)
    # =========================================================================
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, x)"""
        return np.maximum(x, 0)
    
    @staticmethod
    def xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """XOR polynomial: a + b - 2ab"""
        return a + b - 2 * a * b
    
    @staticmethod
    def and_(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """AND polynomial: ab"""
        return a * b
    
    @staticmethod
    def or_(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """OR polynomial: a + b - ab"""
        return a + b - a * b
    
    @staticmethod
    def not_(a: np.ndarray) -> np.ndarray:
        """NOT polynomial: 1 - a"""
        return 1 - a
    
    # =========================================================================
    # Reductions
    # =========================================================================
    
    def sum(self, x: np.ndarray) -> float:
        """Sum via adder tree (pairwise summation)."""
        x = np.ascontiguousarray(x, dtype=np.float32).flatten()
        return self._lib.trix_sum(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(x)
        )
    
    def norm(self, x: np.ndarray) -> float:
        """L2 norm via adder tree."""
        x = np.ascontiguousarray(x, dtype=np.float32).flatten()
        return self._lib.trix_norm(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            len(x)
        )
    
    # =========================================================================
    # Routing
    # =========================================================================
    
    def binarize(self, x: np.ndarray) -> np.ndarray:
        """Convert float array to packed binary (x >= 0 -> 1)."""
        x = np.ascontiguousarray(x, dtype=np.float32).flatten()
        packed_size = (len(x) + 7) // 8
        out = np.zeros(packed_size, dtype=np.uint8)
        
        self._lib.trix_binarize(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(x)
        )
        return out
    
    def hamming_distance(self, a: np.ndarray, b: np.ndarray) -> int:
        """Hamming distance between packed binary arrays."""
        a = np.ascontiguousarray(a, dtype=np.uint8)
        b = np.ascontiguousarray(b, dtype=np.uint8)
        assert len(a) == len(b)
        
        return self._lib.trix_hamming_distance(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(a)
        )
    
    def route_hamming(
        self,
        input_binary: np.ndarray,
        signatures: np.ndarray
    ) -> int:
        """
        Route input to best-matching signature via Hamming distance.
        
        Args:
            input_binary: Packed binary input [packed_dim]
            signatures: Packed binary signatures [num_sigs, packed_dim]
            
        Returns:
            Index of closest signature
        """
        input_binary = np.ascontiguousarray(input_binary, dtype=np.uint8)
        signatures = np.ascontiguousarray(signatures, dtype=np.uint8)
        
        num_sigs, packed_dim = signatures.shape
        
        return self._lib.trix_route_hamming(
            input_binary.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            signatures.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            num_sigs,
            packed_dim
        )


# Convenience: module-level instance
_ops: Optional[TrixOps] = None

def get_ops() -> TrixOps:
    """Get or create singleton TrixOps instance."""
    global _ops
    if _ops is None:
        _ops = TrixOps()
    return _ops


# Export frozen shapes at module level
def relu(x): return TrixOps.relu(x)
def xor(a, b): return TrixOps.xor(a, b)
def and_(a, b): return TrixOps.and_(a, b)
def or_(a, b): return TrixOps.or_(a, b)
def not_(a): return TrixOps.not_(a)
