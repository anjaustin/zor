"""
TriX PyTorch - Legacy PyTorch Modules

NOTICE: The primary API is now trix.native.
        PyTorch modules are maintained for backward compatibility.

For new projects, use:
    from trix.native import NativeHollywoodSquares, Trainer

This module re-exports from trix.nn for backward compatibility.
"""

import warnings

# Re-export everything from nn
from ..nn import *
from ..nn import __all__ as nn_all

__all__ = nn_all


def __getattr__(name):
    """Emit deprecation warning on first access."""
    if name in nn_all:
        warnings.warn(
            f"trix.pytorch.{name} is deprecated. "
            "The primary API is now trix.native. "
            "PyTorch modules are maintained for backward compatibility.",
            DeprecationWarning,
            stacklevel=2
        )
        from .. import nn
        return getattr(nn, name)
    raise AttributeError(f"module 'trix.pytorch' has no attribute '{name}'")
