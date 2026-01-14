"""
Shape Fabric Backends - Different execution targets for shapes.

Available backends:
- python: Pure Python reference implementation
- numpy: NumPy-accelerated for batched operations
- native: C extension for maximum performance (future)
- cuda: GPU execution via CUDA (future)
"""

from typing import Dict, Any

# Backend registry
BACKENDS: Dict[str, Any] = {}


def register_backend(name: str, backend: Any) -> None:
    """Register a new backend."""
    BACKENDS[name] = backend


def get_backend(name: str) -> Any:
    """Get a backend by name."""
    return BACKENDS.get(name)


def list_backends() -> list:
    """List available backends."""
    return list(BACKENDS.keys())
