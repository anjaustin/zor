"""
TriX Kernel (DEPRECATED)

This module is deprecated. Use trix.native.ops instead.

The kernel uses PyTorch bindings and STE (Straight-Through Estimator).
The native.ops module is self-hosted (pure C) with real gradients via
the Gradient Truth paradigm.

Migration:
    # OLD (deprecated)
    from trix.kernel import TriXLinear, STESign
    
    # NEW (recommended)
    from trix.native.ops import TrixOps
    from trix.nn import HierarchicalTriXFFN  # Uses Gradient Truth by default

The native.ops module provides:
- NEON (ARM) and AVX2 (x86) SIMD acceleration
- No PyTorch dependency for inference
- Binary frozen shapes for maximum speed
"""

import warnings

warnings.warn(
    "trix.kernel is deprecated. Use trix.native.ops for inference "
    "and trix.nn with use_gradient_truth=True for training. "
    "STESign is no longer recommended.",
    DeprecationWarning,
    stacklevel=2
)

from .bindings import (
    TriXLinear,
    STESign,
    pack_weights,
    unpack_weights,
    trix_forward,
    is_neon_available,
)

__all__ = [
    "TriXLinear",
    "STESign",
    "pack_weights",
    "unpack_weights",
    "trix_forward",
    "is_neon_available",
]
