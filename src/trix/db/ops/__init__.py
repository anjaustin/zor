"""
DB Cooper Native Operations - Python Bindings

NEON-optimized ternary similarity for Octave DB.
Pure bit operations. No floats at query time.

Usage:
    from trix.db.ops import CooperOps
    
    ops = CooperOps()
    print(f"SIMD: {ops.simd}")
    
    score = ops.ternary_similarity(q_pos, q_neg, d_pos, d_neg)
    scores = ops.batch_similarity(q_pos, q_neg, d_pos_batch, d_neg_batch)
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class CooperOps:
    """
    Python interface to DB Cooper native operations.
    
    Auto-detects SIMD (NEON on ARM, AVX2 on x86) and falls back
    to optimized scalar code if neither is available.
    """
    
    _lib: Optional[ctypes.CDLL] = None
    _lib_path: Optional[Path] = None
    
    def __init__(self):
        """Initialize and load native library."""
        self._load_library()
    
    def _load_library(self) -> None:
        """Load the native library."""
        if CooperOps._lib is not None:
            return
        
        # Find library
        ops_dir = Path(__file__).parent
        
        # Try different extensions
        for ext in ['so', 'dylib']:
            lib_path = ops_dir / f'libcooper.{ext}'
            if lib_path.exists():
                CooperOps._lib_path = lib_path
                break
        
        if CooperOps._lib_path is None:
            raise RuntimeError(
                f"Native library not found. Build it with:\n"
                f"  cd {ops_dir} && make"
            )
        
        # Load library
        CooperOps._lib = ctypes.CDLL(str(CooperOps._lib_path))
        
        # Set up function signatures
        self._setup_signatures()
        
        # Initialize
        CooperOps._lib.cooper_ops_init()
    
    def _setup_signatures(self) -> None:
        """Set up ctypes function signatures."""
        lib = CooperOps._lib
        
        # Version
        lib.cooper_ops_version.restype = ctypes.c_char_p
        lib.cooper_ops_version.argtypes = []
        
        # SIMD detection
        lib.cooper_has_neon.restype = ctypes.c_int
        lib.cooper_has_neon.argtypes = []
        
        lib.cooper_has_avx2.restype = ctypes.c_int
        lib.cooper_has_avx2.argtypes = []
        
        # Ternary similarity
        lib.cooper_ternary_similarity.restype = ctypes.c_int32
        lib.cooper_ternary_similarity.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),  # q_pos
            ctypes.POINTER(ctypes.c_uint8),  # q_neg
            ctypes.POINTER(ctypes.c_uint8),  # d_pos
            ctypes.POINTER(ctypes.c_uint8),  # d_neg
            ctypes.c_size_t,                  # packed_bytes
        ]
        
        # Batch similarity
        lib.cooper_batch_similarity.restype = None
        lib.cooper_batch_similarity.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),  # q_pos
            ctypes.POINTER(ctypes.c_uint8),  # q_neg
            ctypes.POINTER(ctypes.c_uint8),  # d_pos
            ctypes.POINTER(ctypes.c_uint8),  # d_neg
            ctypes.POINTER(ctypes.c_int32),  # scores
            ctypes.c_size_t,                  # n_docs
            ctypes.c_size_t,                  # packed_bytes
        ]
        
        # Derive coarse
        lib.cooper_derive_coarse.restype = None
        lib.cooper_derive_coarse.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),  # fine_pos
            ctypes.POINTER(ctypes.c_uint8),  # fine_neg
            ctypes.POINTER(ctypes.c_uint8),  # coarse_pos
            ctypes.POINTER(ctypes.c_uint8),  # coarse_neg
            ctypes.c_size_t,                  # fine_dims
            ctypes.c_size_t,                  # pool_factor
        ]
        
        # Pack ternary
        lib.cooper_pack_ternary.restype = None
        lib.cooper_pack_ternary.argtypes = [
            ctypes.POINTER(ctypes.c_int8),   # ternary
            ctypes.POINTER(ctypes.c_uint8),  # pos_out
            ctypes.POINTER(ctypes.c_uint8),  # neg_out
            ctypes.c_size_t,                  # n_dims
        ]
        
        # Unpack ternary
        lib.cooper_unpack_ternary.restype = None
        lib.cooper_unpack_ternary.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),  # pos
            ctypes.POINTER(ctypes.c_uint8),  # neg
            ctypes.POINTER(ctypes.c_int8),   # ternary_out
            ctypes.c_size_t,                  # n_dims
        ]
        
        # Hamming distance
        lib.cooper_hamming_distance.restype = ctypes.c_int32
        lib.cooper_hamming_distance.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        
        # Popcount
        lib.cooper_popcount.restype = ctypes.c_int32
        lib.cooper_popcount.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
    
    @property
    def version(self) -> str:
        """Library version."""
        return CooperOps._lib.cooper_ops_version().decode('utf-8')
    
    @property
    def has_neon(self) -> bool:
        """Check if NEON SIMD is available."""
        return bool(CooperOps._lib.cooper_has_neon())
    
    @property
    def has_avx2(self) -> bool:
        """Check if AVX2 SIMD is available."""
        return bool(CooperOps._lib.cooper_has_avx2())
    
    @property
    def simd(self) -> str:
        """Active SIMD instruction set."""
        if self.has_neon:
            return "NEON"
        elif self.has_avx2:
            return "AVX2"
        else:
            return "scalar"
    
    def _as_ptr(self, arr: np.ndarray, dtype) -> ctypes.POINTER:
        """Convert numpy array to ctypes pointer."""
        return arr.ctypes.data_as(ctypes.POINTER(dtype))
    
    def ternary_similarity(
        self,
        q_pos: np.ndarray,
        q_neg: np.ndarray,
        d_pos: np.ndarray,
        d_neg: np.ndarray,
    ) -> int:
        """
        Compute ternary similarity between query and document.
        
        Args:
            q_pos: Query positive bitmask (uint8)
            q_neg: Query negative bitmask (uint8)
            d_pos: Document positive bitmask (uint8)
            d_neg: Document negative bitmask (uint8)
        
        Returns:
            Integer similarity score (agreement - conflict)
        """
        q_pos = np.ascontiguousarray(q_pos, dtype=np.uint8)
        q_neg = np.ascontiguousarray(q_neg, dtype=np.uint8)
        d_pos = np.ascontiguousarray(d_pos, dtype=np.uint8)
        d_neg = np.ascontiguousarray(d_neg, dtype=np.uint8)
        
        packed_bytes = len(q_pos)
        
        return CooperOps._lib.cooper_ternary_similarity(
            self._as_ptr(q_pos, ctypes.c_uint8),
            self._as_ptr(q_neg, ctypes.c_uint8),
            self._as_ptr(d_pos, ctypes.c_uint8),
            self._as_ptr(d_neg, ctypes.c_uint8),
            packed_bytes,
        )
    
    def batch_similarity(
        self,
        q_pos: np.ndarray,
        q_neg: np.ndarray,
        d_pos: np.ndarray,
        d_neg: np.ndarray,
    ) -> np.ndarray:
        """
        Compute similarity between one query and multiple documents.
        
        Args:
            q_pos: Query positive bitmask [packed_bytes]
            q_neg: Query negative bitmask [packed_bytes]
            d_pos: Document positive bitmasks [n_docs, packed_bytes]
            d_neg: Document negative bitmasks [n_docs, packed_bytes]
        
        Returns:
            Array of similarity scores [n_docs]
        """
        q_pos = np.ascontiguousarray(q_pos, dtype=np.uint8)
        q_neg = np.ascontiguousarray(q_neg, dtype=np.uint8)
        d_pos = np.ascontiguousarray(d_pos, dtype=np.uint8)
        d_neg = np.ascontiguousarray(d_neg, dtype=np.uint8)
        
        n_docs = d_pos.shape[0]
        packed_bytes = q_pos.shape[0]
        
        scores = np.zeros(n_docs, dtype=np.int32)
        
        CooperOps._lib.cooper_batch_similarity(
            self._as_ptr(q_pos, ctypes.c_uint8),
            self._as_ptr(q_neg, ctypes.c_uint8),
            self._as_ptr(d_pos.ravel(), ctypes.c_uint8),
            self._as_ptr(d_neg.ravel(), ctypes.c_uint8),
            self._as_ptr(scores, ctypes.c_int32),
            n_docs,
            packed_bytes,
        )
        
        return scores
    
    def derive_coarse(
        self,
        fine_pos: np.ndarray,
        fine_neg: np.ndarray,
        fine_dims: int,
        pool_factor: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Derive coarse signature from fine via pooling.
        
        Args:
            fine_pos: Fine positive bitmask
            fine_neg: Fine negative bitmask
            fine_dims: Number of fine dimensions
            pool_factor: How many fine dims per coarse dim
        
        Returns:
            (coarse_pos, coarse_neg) bitmasks
        """
        fine_pos = np.ascontiguousarray(fine_pos, dtype=np.uint8)
        fine_neg = np.ascontiguousarray(fine_neg, dtype=np.uint8)
        
        coarse_dims = fine_dims // pool_factor
        coarse_bytes = (coarse_dims + 7) // 8
        
        coarse_pos = np.zeros(coarse_bytes, dtype=np.uint8)
        coarse_neg = np.zeros(coarse_bytes, dtype=np.uint8)
        
        CooperOps._lib.cooper_derive_coarse(
            self._as_ptr(fine_pos, ctypes.c_uint8),
            self._as_ptr(fine_neg, ctypes.c_uint8),
            self._as_ptr(coarse_pos, ctypes.c_uint8),
            self._as_ptr(coarse_neg, ctypes.c_uint8),
            fine_dims,
            pool_factor,
        )
        
        return coarse_pos, coarse_neg
    
    def pack_ternary(self, ternary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pack ternary values to bitmasks.
        
        Args:
            ternary: Array of {-1, 0, +1} as int8
        
        Returns:
            (pos_mask, neg_mask) as uint8 arrays
        """
        ternary = np.ascontiguousarray(ternary, dtype=np.int8)
        n_dims = len(ternary)
        n_bytes = (n_dims + 7) // 8
        
        pos = np.zeros(n_bytes, dtype=np.uint8)
        neg = np.zeros(n_bytes, dtype=np.uint8)
        
        CooperOps._lib.cooper_pack_ternary(
            self._as_ptr(ternary, ctypes.c_int8),
            self._as_ptr(pos, ctypes.c_uint8),
            self._as_ptr(neg, ctypes.c_uint8),
            n_dims,
        )
        
        return pos, neg
    
    def unpack_ternary(
        self,
        pos: np.ndarray,
        neg: np.ndarray,
        n_dims: int,
    ) -> np.ndarray:
        """
        Unpack bitmasks to ternary values.
        
        Args:
            pos: Positive bitmask
            neg: Negative bitmask
            n_dims: Number of dimensions
        
        Returns:
            Array of {-1, 0, +1} as int8
        """
        pos = np.ascontiguousarray(pos, dtype=np.uint8)
        neg = np.ascontiguousarray(neg, dtype=np.uint8)
        
        ternary = np.zeros(n_dims, dtype=np.int8)
        
        CooperOps._lib.cooper_unpack_ternary(
            self._as_ptr(pos, ctypes.c_uint8),
            self._as_ptr(neg, ctypes.c_uint8),
            self._as_ptr(ternary, ctypes.c_int8),
            n_dims,
        )
        
        return ternary
    
    def hamming_distance(self, a: np.ndarray, b: np.ndarray) -> int:
        """Compute Hamming distance between two bitmasks."""
        a = np.ascontiguousarray(a, dtype=np.uint8)
        b = np.ascontiguousarray(b, dtype=np.uint8)
        
        return CooperOps._lib.cooper_hamming_distance(
            self._as_ptr(a, ctypes.c_uint8),
            self._as_ptr(b, ctypes.c_uint8),
            len(a),
        )
    
    def popcount(self, bits: np.ndarray) -> int:
        """Count number of set bits in bitmask."""
        bits = np.ascontiguousarray(bits, dtype=np.uint8)
        
        return CooperOps._lib.cooper_popcount(
            self._as_ptr(bits, ctypes.c_uint8),
            len(bits),
        )


# Convenience function to check if native ops are available
def native_ops_available() -> bool:
    """Check if DB Cooper native operations are available."""
    try:
        ops = CooperOps()
        return True
    except RuntimeError:
        return False
