"""
DB Cooper: Octave DB
Multi-resolution ternary retrieval with glassbox explainability.

    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ COARSE  │ ←─ │ MEDIUM  │ ←─ │  FINE   │   derived hierarchy
    │ D/16    │    │  D/4    │    │    D    │
    └────┬────┘    └────┬────┘    └────┬────┘
         │              │              │
         ▼              ▼              ▼
    CONTEXT       SIMILARITY      EXACTNESS

Usage:
    from trix.db import OctaveDB
    
    db = OctaveDB(dimensions=384)
    db.add("doc1", embedding)
    results = db.search(query_embedding, mode="similar", top_k=10)
"""

from .core import (
    ternary_quantize,
    derive_coarse,
    ternary_similarity,
    pack_ternary,
    unpack_ternary,
)

from .index import OctaveIndex

from .octave_db import OctaveDB

from .explorer import (
    PossibilityExplorer,
    ExplorationResult,
    compare_approaches,
)

__all__ = [
    "OctaveDB",
    "OctaveIndex",
    "PossibilityExplorer",
    "ExplorationResult",
    "compare_approaches",
    "ternary_quantize",
    "derive_coarse",
    "ternary_similarity",
    "pack_ternary",
    "unpack_ternary",
]
