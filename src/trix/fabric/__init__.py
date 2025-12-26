"""
TriX Fabric: Unified Database-Computer Architecture

The fabric IS the program.
The fabric IS the database.
The fabric IS the knowledge.

Components:
- Fabric: Recursive topology of processors
- Cluster: Tree node containing hypercube of processors
- Processor: Shape + Signature (stateless)
- Router: Hierarchical routing (prefix → Hamming)
- FDL: Fabric Definition Language (define/persist/query)
"""

from .fabric import Fabric
from .cluster import Cluster
from .processor import FabricProcessor
from .router import Router
from .fdl import parse_fdl, emit_fdl
from .query import query_fabric
from .message import Message

__all__ = [
    'Fabric',
    'Cluster',
    'FabricProcessor',
    'Router',
    'Message',
    'parse_fdl',
    'emit_fdl',
    'query_fabric',
]
