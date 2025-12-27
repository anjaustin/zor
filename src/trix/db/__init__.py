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

from .codex import (
    CodexField,
    CodexLayer,
    GeneKeyTile,
    CodexEdge,
    load_codex_from_json,
)

from .codex_keywords import (
    KeywordCodex,
    KeywordCodexLayer,
    GeneKeyNode,
    create_codex,
    GENE_KEYS,
    PARTNERS,
    CODON_RINGS,
)

from .hsquares_index import (
    HollywoodSquaresIndex,
    create_hsquares_index,
)

from .codex_squares import (
    CodexSquaresIndex,
    CodexResult,
    CodexTile,
    create_codex_squares,
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
    # Codex Layer (embedding-based)
    "CodexField",
    "CodexLayer", 
    "GeneKeyTile",
    "CodexEdge",
    "load_codex_from_json",
    # Codex Layer (keyword-based)
    "KeywordCodex",
    "KeywordCodexLayer",
    "GeneKeyNode",
    "create_codex",
    "GENE_KEYS",
    "PARTNERS",
    "CODON_RINGS",
    # Hollywood Squares Index
    "HollywoodSquaresIndex",
    "create_hsquares_index",
    # Codex Squares Index
    "CodexSquaresIndex",
    "CodexResult",
    "CodexTile",
    "create_codex_squares",
]
