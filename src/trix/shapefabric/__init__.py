"""
TriX Shape Fabric - Shape-Routed Compute Architecture.

The Shape Fabric is a dataflow compute architecture where:
- Programs are graphs of shape compositions
- Execution is determined by data flow, not instruction sequence
- Parallelism is automatic and implicit
- 32 atomic shapes form the basis for all computation

Usage:
    from trix.shapefabric import ShapeGraph, execute

    # Build a graph
    graph = ShapeGraph()
    a = graph.input("a")
    b = graph.input("b")
    sum_node = graph.add("ADD", [a, b])
    graph.output("result", sum_node)

    # Execute
    result = execute(graph, {"a": 5, "b": 3})
    assert result["result"] == 8
"""

__version__ = "0.1.0"

from .shapes import SHAPES, Shape, get_shape, list_shapes
from .graph import ShapeGraph, Node, Edge
from .executor import execute, DataflowExecutor
from .fabric import ShapeFabric, PatternRouter, ShapeTile, create_fabric

__all__ = [
    "SHAPES",
    "Shape",
    "get_shape",
    "list_shapes",
    "ShapeGraph",
    "Node",
    "Edge",
    "execute",
    "DataflowExecutor",
    "ShapeFabric",
    "PatternRouter",
    "ShapeTile",
    "create_fabric",
]
