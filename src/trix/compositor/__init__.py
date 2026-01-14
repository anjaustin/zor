"""
TriX Compositor - Automatic structural discovery from flat gate graphs.

The compositor takes a flat graph of gates (from ingest) and discovers
hierarchical structure through pattern recognition and composition.

Usage:
    from trix.compositor import compose, visualize

    # Load a flat graph from ingest
    from trix.forge.ingest import ingest_yosys_json
    system = ingest_yosys_json("design.json")

    # Discover structure
    tree = compose(system)

    # View the hierarchy
    print(visualize(tree))
"""

__version__ = "0.1.0"

from .composer import compose, ComposedShape, ShapeTree
from .hasher import hash_neighborhood, NeighborhoodHash
from .matcher import find_patterns, Pattern
from .visualizer import visualize, visualize_stats

__all__ = [
    "compose",
    "ComposedShape",
    "ShapeTree",
    "hash_neighborhood",
    "NeighborhoodHash",
    "find_patterns",
    "Pattern",
    "visualize",
    "visualize_stats",
]
