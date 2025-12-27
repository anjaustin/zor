"""
DB Cooper: OctaveIndex - Hierarchical ternary index.

Coarse buckets for fast filtering, fine signatures for precise ranking.
NEON-accelerated when native ops are available.

The Secret Sauce: Magnitude-weighted similarity.
Sign tells direction. Magnitude tells importance.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass

from .core import (
    ternary_similarity,
    derive_hierarchy,
    pack_ternary,
    explain_match,
    octave_quantize,
    octave_similarity,
    octave_similarity_batch,
)

# Try to import native ops
try:
    from .ops import CooperOps, native_ops_available
    _NATIVE_OPS = CooperOps() if native_ops_available() else None
except Exception:
    _NATIVE_OPS = None


@dataclass
class Document:
    """Stored document with octave signatures."""
    id: str
    fine: np.ndarray       # Signs {-1, 0, +1}
    medium: np.ndarray
    coarse: np.ndarray
    magnitudes: np.ndarray  # The Secret Sauce: magnitude weights
    metadata: Optional[dict] = None
    
    # Packed representations for fast similarity
    fine_pos: Optional[np.ndarray] = None
    fine_neg: Optional[np.ndarray] = None


@dataclass 
class SearchResult:
    """Single search result with explanation."""
    id: str
    score: float
    level_scores: Dict[str, float]
    metadata: Optional[dict] = None
    explanation: Optional[dict] = None


class OctaveIndex:
    """
    Hierarchical ternary index for multi-resolution search.
    
    Structure:
        COARSE BUCKETS (hash by coarse signature)
        └── Documents with that coarse signature
            └── Each has medium and fine signatures
    
    Search:
        1. Hash query coarse → candidate buckets
        2. Filter by coarse similarity
        3. Rerank by medium
        4. Rerank by fine
    """
    
    def __init__(
        self,
        dimensions: int,
        pool_factor: int = 4,
        coarse_threshold: float = 0.5,
    ):
        """
        Initialize Octave Index.
        
        Args:
            dimensions: Fine embedding dimensionality
            pool_factor: Pooling factor between levels
            coarse_threshold: Minimum coarse similarity to consider
        """
        self.dimensions = dimensions
        self.pool_factor = pool_factor
        self.coarse_threshold = coarse_threshold
        
        # Derived dimensions
        self.medium_dims = dimensions // pool_factor
        self.coarse_dims = self.medium_dims // pool_factor
        
        # Storage
        self.documents: Dict[str, Document] = {}
        self.coarse_buckets: Dict[tuple, List[str]] = defaultdict(list)
        
        # Stats
        self.num_documents = 0
    
    def _coarse_key(self, coarse: np.ndarray) -> tuple:
        """Convert coarse signature to hashable key."""
        return tuple(coarse.flatten().tolist())
    
    def add(
        self,
        doc_id: str,
        fine: np.ndarray,
        medium: Optional[np.ndarray] = None,
        coarse: Optional[np.ndarray] = None,
        magnitudes: Optional[np.ndarray] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add document to index.
        
        Args:
            doc_id: Unique document identifier
            fine: Fine ternary signature [D] (signs)
            medium: Medium signature (derived if not provided)
            coarse: Coarse signature (derived if not provided)
            magnitudes: Magnitude weights (for Secret Sauce similarity)
            metadata: Optional document metadata
        """
        fine = np.asarray(fine, dtype=np.int8)
        
        # Derive if not provided
        if medium is None or coarse is None:
            levels = derive_hierarchy(fine, self.pool_factor, num_levels=3)
            fine, medium, coarse = levels[0], levels[1], levels[2]
        else:
            medium = np.asarray(medium, dtype=np.int8)
            coarse = np.asarray(coarse, dtype=np.int8)
        
        # Default magnitudes to uniform if not provided (backward compat)
        if magnitudes is None:
            magnitudes = np.ones(len(fine), dtype=np.float32)
        else:
            magnitudes = np.asarray(magnitudes, dtype=np.float32)
        
        # Pack for fast similarity
        fine_pos, fine_neg = pack_ternary(fine)
        
        # Create document
        doc = Document(
            id=doc_id,
            fine=fine,
            medium=medium,
            coarse=coarse,
            magnitudes=magnitudes,
            metadata=metadata,
            fine_pos=fine_pos,
            fine_neg=fine_neg,
        )
        
        # Store
        self.documents[doc_id] = doc
        
        # Index by coarse bucket
        key = self._coarse_key(coarse)
        self.coarse_buckets[key].append(doc_id)
        
        self.num_documents += 1
    
    def add_batch(
        self,
        doc_ids: List[str],
        fine_embeddings: np.ndarray,
        metadata_list: Optional[List[dict]] = None,
    ) -> None:
        """Add multiple documents efficiently."""
        if metadata_list is None:
            metadata_list = [None] * len(doc_ids)
        
        for doc_id, fine, meta in zip(doc_ids, fine_embeddings, metadata_list):
            self.add(doc_id, fine, metadata=meta)
    
    def remove(self, doc_id: str) -> bool:
        """Remove document from index."""
        if doc_id not in self.documents:
            return False
        
        doc = self.documents[doc_id]
        key = self._coarse_key(doc.coarse)
        
        # Remove from bucket
        if doc_id in self.coarse_buckets[key]:
            self.coarse_buckets[key].remove(doc_id)
            if not self.coarse_buckets[key]:
                del self.coarse_buckets[key]
        
        # Remove document
        del self.documents[doc_id]
        self.num_documents -= 1
        
        return True
    
    def search(
        self,
        query_fine: np.ndarray,
        query_medium: Optional[np.ndarray] = None,
        query_coarse: Optional[np.ndarray] = None,
        query_magnitudes: Optional[np.ndarray] = None,
        mode: str = "similar",
        top_k: int = 10,
        explain: bool = False,
    ) -> List[SearchResult]:
        """
        Cascading octave search: Oc2 → Oc1 → Oc0.
        
        The funnel:
            Oc2 (coarse): Fast fuzzy filter, broad candidates
            Oc1 (medium): Converge, narrow down
            Oc0 (fine):   Near-exact, final ranking (with Secret Sauce)
        
        Args:
            query_fine: Fine query signature (signs)
            query_medium: Medium signature (derived if not provided)
            query_coarse: Coarse signature (derived if not provided)
            query_magnitudes: Query magnitude weights (for Secret Sauce)
            mode: "exact" | "similar" | "context"
                - exact: tight funnel, strict convergence
                - similar: balanced funnel
                - context: wide funnel, broad discovery
            top_k: Number of results to return
            explain: Whether to include per-dimension explanations
        
        Returns:
            List of SearchResult sorted by relevance
        """
        query_fine = np.asarray(query_fine, dtype=np.int8)
        
        # Derive query hierarchy
        if query_medium is None or query_coarse is None:
            levels = derive_hierarchy(query_fine, self.pool_factor, num_levels=3)
            query_fine, query_medium, query_coarse = levels
        else:
            query_medium = np.asarray(query_medium, dtype=np.int8)
            query_coarse = np.asarray(query_coarse, dtype=np.int8)
        
        # Default query magnitudes to uniform if not provided
        if query_magnitudes is None:
            query_magnitudes = np.ones(len(query_fine), dtype=np.float32)
        else:
            query_magnitudes = np.asarray(query_magnitudes, dtype=np.float32)
        
        # Funnel widths based on mode
        if mode == "context":
            oc2_keep = self.num_documents  # Keep all at coarse
            oc1_keep = self.num_documents  # Keep all at medium
        elif mode == "exact":
            oc2_keep = max(top_k * 5, 20)   # Tight funnel
            oc1_keep = max(top_k * 2, 10)
        else:  # similar
            oc2_keep = max(top_k * 10, 50)  # Balanced funnel
            oc1_keep = max(top_k * 3, 15)
        
        # Normalization factors
        coarse_norm = max(np.sum(np.abs(query_coarse)), 1)
        medium_norm = max(np.sum(np.abs(query_medium)), 1)
        
        # ═══════════════════════════════════════════════════════════════
        # Oc2 (COARSE): Fuzzy, fastest - broad candidates
        # ═══════════════════════════════════════════════════════════════
        oc2_candidates = []
        for doc_id, doc in self.documents.items():
            coarse_score = ternary_similarity(query_coarse, doc.coarse) / coarse_norm
            oc2_candidates.append((doc_id, coarse_score))
        
        # Sort by coarse score, keep top candidates
        oc2_candidates.sort(key=lambda x: -x[1])
        oc2_candidates = oc2_candidates[:oc2_keep]
        
        # If context mode, return coarse results directly
        if mode == "context":
            results = []
            for doc_id, coarse_score in oc2_candidates[:top_k]:
                doc = self.documents[doc_id]
                results.append(SearchResult(
                    id=doc_id,
                    score=coarse_score,
                    level_scores={'coarse': coarse_score, 'medium': 0.0, 'fine': 0.0},
                    metadata=doc.metadata,
                    explanation=explain_match(query_coarse, doc.coarse) if explain else None,
                ))
            return results
        
        # ═══════════════════════════════════════════════════════════════
        # Oc1 (MEDIUM): More convergence, faster - narrow down
        # ═══════════════════════════════════════════════════════════════
        oc1_candidates = []
        for doc_id, coarse_score in oc2_candidates:
            doc = self.documents[doc_id]
            medium_score = ternary_similarity(query_medium, doc.medium) / medium_norm
            oc1_candidates.append((doc_id, coarse_score, medium_score))
        
        # Sort by medium score, keep top candidates
        oc1_candidates.sort(key=lambda x: -x[2])
        oc1_candidates = oc1_candidates[:oc1_keep]
        
        # ═══════════════════════════════════════════════════════════════
        # Oc0 (FINE): Near-exact convergence - final ranking
        # Uses SECRET SAUCE (magnitude-weighted similarity)
        # ═══════════════════════════════════════════════════════════════
        oc0_results = []
        for doc_id, coarse_score, medium_score in oc1_candidates:
            doc = self.documents[doc_id]
            
            # THE SECRET SAUCE: magnitude-weighted similarity
            octave_score = octave_similarity(
                query_fine, query_magnitudes,
                doc.fine, doc.magnitudes
            )
            
            # Normalize by max possible score
            max_score = np.sum(np.sqrt(query_magnitudes * doc.magnitudes))
            fine_score = octave_score / max_score if max_score > 0 else 0.0
            
            # Final score is fine score (the others were just for filtering)
            oc0_results.append((doc_id, fine_score, {
                'fine': fine_score,
                'medium': medium_score,
                'coarse': coarse_score,
            }))
        
        # Sort by fine score (the final word)
        oc0_results.sort(key=lambda x: -x[1])
        
        # Build results
        results = []
        for doc_id, score, level_scores in oc0_results[:top_k]:
            doc = self.documents[doc_id]
            results.append(SearchResult(
                id=doc_id,
                score=score,
                level_scores=level_scores,
                metadata=doc.metadata,
                explanation=explain_match(query_fine, doc.fine) if explain else None,
            ))
        
        return results
    
    def get(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self.documents.get(doc_id)
    
    def stats(self) -> dict:
        """Get index statistics."""
        bucket_sizes = [len(docs) for docs in self.coarse_buckets.values()]
        return {
            'num_documents': self.num_documents,
            'num_buckets': len(self.coarse_buckets),
            'avg_bucket_size': np.mean(bucket_sizes) if bucket_sizes else 0,
            'max_bucket_size': max(bucket_sizes) if bucket_sizes else 0,
            'dimensions': {
                'fine': self.dimensions,
                'medium': self.medium_dims,
                'coarse': self.coarse_dims,
            },
        }
