"""
TriX → XOR Fabric Pipeline

A mature, production-quality pipeline for:
- Training neural computation with TriX (smooth gradients)
- Freezing learned shapes (no learnable parameters)
- Compiling to CUDA XOR fabric (hardware execution)
- Verifying correctness (TriX output = CUDA output)

Usage:
    from fabric_pipeline import Pipeline

    pipeline = Pipeline(config)
    pipeline.train(task)
    pipeline.freeze()
    pipeline.compile()
    pipeline.verify()
    pipeline.benchmark()
"""

from .config import FabricConfig, TaskConfig
from .shapes import ShapeRegistry, Shape
from .compiler import FabricCompiler
from .verifier import FabricVerifier
from .pipeline import Pipeline

__version__ = "1.0.0"
__all__ = [
    "Pipeline",
    "FabricConfig",
    "TaskConfig",
    "ShapeRegistry",
    "Shape",
    "FabricCompiler",
    "FabricVerifier",
]
