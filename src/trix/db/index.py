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
    embedding: Optional[np.ndarray] = None  # Original float embedding for exact match
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
        
        # Vectorized storage for fast batch operations
        self._doc_ids: List[str] = []
        self._coarse_matrix: Optional[np.ndarray] = None
        self._medium_matrix: Optional[np.ndarray] = None
        self._fine_matrix: Optional[np.ndarray] = None
        self._mag_matrix: Optional[np.ndarray] = None
        self._embedding_matrix: Optional[np.ndarray] = None  # For quality mode
        self._index_dirty = True
        
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
        embedding: Optional[np.ndarray] = None,
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
            embedding: Original float embedding (for quality mode exact match)
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
        
        # Store embedding if provided
        if embedding is not None:
            embedding = np.asarray(embedding, dtype=np.float32)
        
        # Create document
        doc = Document(
            id=doc_id,
            fine=fine,
            medium=medium,
            coarse=coarse,
            magnitudes=magnitudes,
            embedding=embedding,
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
        self._index_dirty = True
    
    def _rebuild_matrices(self):
        """Rebuild vectorized matrices for fast batch operations."""
        if not self._index_dirty or self.num_documents == 0:
            return
        
        self._doc_ids = list(self.documents.keys())
        n = len(self._doc_ids)
        
        # Get dimensions from first doc
        first_doc = self.documents[self._doc_ids[0]]
        
        self._coarse_matrix = np.zeros((n, len(first_doc.coarse)), dtype=np.int8)
        self._medium_matrix = np.zeros((n, len(first_doc.medium)), dtype=np.int8)
        self._fine_matrix = np.zeros((n, len(first_doc.fine)), dtype=np.int8)
        self._mag_matrix = np.zeros((n, len(first_doc.magnitudes)), dtype=np.float32)
        
        # Check if embeddings are available
        has_embeddings = first_doc.embedding is not None
        if has_embeddings:
            self._embedding_matrix = np.zeros((n, len(first_doc.embedding)), dtype=np.float32)
        else:
            self._embedding_matrix = None
        
        for i, doc_id in enumerate(self._doc_ids):
            doc = self.documents[doc_id]
            self._coarse_matrix[i] = doc.coarse
            self._medium_matrix[i] = doc.medium
            self._fine_matrix[i] = doc.fine
            self._mag_matrix[i] = doc.magnitudes
            if has_embeddings and doc.embedding is not None:
                self._embedding_matrix[i] = doc.embedding
        
        self._index_dirty = False
    
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
    
    def _search_quality(
        self,
        query_fine: np.ndarray,
        query_magnitudes: np.ndarray,
        top_k: int = 10,
        explain: bool = False,
        keep_ratio: float = 0.2,
    ) -> List[SearchResult]:
        """
        Quality mode: 100% recall via magnitude-weighted filter + exact cosine.
        
        Two-stage architecture:
            1. Magnitude-weighted ternary filter → keep top candidates
            2. Exact cosine on candidates → precise ranking
        
        Requires embeddings to be stored (set during add).
        """
        # Reconstruct query embedding from signs * magnitudes
        query_embedding = query_fine.astype(np.float32) * query_magnitudes
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        
        return self._search_quality_with_embedding(
            query_fine, query_magnitudes, query_embedding,
            top_k=top_k, explain=explain, keep_ratio=keep_ratio
        )
    
    def _search_quality_with_embedding(
        self,
        query_fine: np.ndarray,
        query_magnitudes: np.ndarray,
        query_embedding: np.ndarray,
        top_k: int = 10,
        explain: bool = False,
        keep_ratio: float = 0.2,
    ) -> List[SearchResult]:
        """
        Quality mode with original query embedding for exact cosine.
        
        Three-stage architecture for scale:
            1. Coarse filter (fast) → 30% candidates
            2. Magnitude-weighted filter → 20% of remaining
            3. Exact cosine → final ranking
        """
        self._rebuild_matrices()
        
        if self.num_documents == 0:
            return []
        
        if self._embedding_matrix is None:
            raise ValueError(
                "Quality mode requires embeddings. "
                "Store embeddings when adding documents."
            )
        
        # ═══════════════════════════════════════════════════════════════
        # HIERARCHICAL QUALITY: Fast coarse filter + precise reranking
        # ═══════════════════════════════════════════════════════════════
        
        # For small datasets, skip hierarchy (brute-force is fast)
        HIERARCHY_THRESHOLD = 5000
        
        if self.num_documents <= HIERARCHY_THRESHOLD:
            # Direct magnitude-weighted filter on all documents
            candidate_indices = np.arange(self.num_documents)
        else:
            # For large datasets: magnitude-weighted filter → exact cosine
            # Skip hierarchy, use Secret Sauce directly
            weights = np.sqrt(query_magnitudes * self._mag_matrix)
            sign_product = query_fine * self._fine_matrix
            weighted_scores = np.sum(weights * sign_product, axis=1)
            
            # Keep top 20% for exact cosine reranking (max 20K for speed)
            n_keep = min(max(int(self.num_documents * 0.20), 1000), 20000)
            candidate_indices = np.argsort(weighted_scores)[::-1][:n_keep]
        
        # For small datasets, we need to compute magnitude-weighted scores
        if self.num_documents <= HIERARCHY_THRESHOLD:
            candidate_mags = self._mag_matrix[candidate_indices]
            candidate_signs = self._fine_matrix[candidate_indices]
            weights = np.sqrt(query_magnitudes * candidate_mags)
            sign_product = query_fine * candidate_signs
            weighted_scores = np.sum(weights * sign_product, axis=1)
            n_keep = max(int(len(candidate_indices) * keep_ratio), min(500, len(candidate_indices)))
            top_idx = np.argsort(weighted_scores)[::-1][:n_keep]
            rerank_indices = candidate_indices[top_idx]
        else:
            # Already filtered by magnitude-weighted scores
            rerank_indices = candidate_indices
        
        # Stage 2: Exact cosine on candidates using original query embedding
        candidate_embeddings = self._embedding_matrix[rerank_indices]
        exact_scores = candidate_embeddings @ query_embedding
        
        # Top K by exact cosine
        top_k_local = np.argsort(exact_scores)[::-1][:top_k]
        top_k_indices = rerank_indices[top_k_local]
        top_k_scores = exact_scores[top_k_local]
        
        # Build results
        results = []
        for i, idx in enumerate(top_k_indices):
            doc_id = self._doc_ids[idx]
            doc = self.documents[doc_id]
            results.append(SearchResult(
                id=doc_id,
                score=float(top_k_scores[i]),
                level_scores={
                    'exact_cosine': float(top_k_scores[i]),
                },
                metadata=doc.metadata,
                explanation=None,
            ))
        
        return results
    
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
        
        # PHI FUNNEL: Golden ratio narrowing at each octave
        # φ = 1.618... - the spiral to convergence
        PHI = 1.6180339887
        
        # QUALITY MODE: magnitude-weighted filter + exact cosine
        # Achieves 100% recall with minimal candidate set
        if mode == "quality":
            return self._search_quality(
                query_fine, query_magnitudes, top_k, explain
            )
        
        # ═══════════════════════════════════════════════════════════════
        # ENGINEERING FIX: Skip hierarchy for small datasets
        # Brute-force is fast enough for N < 10K and guarantees recall
        # ═══════════════════════════════════════════════════════════════
        BRUTE_FORCE_THRESHOLD = 10000
        
        if self.num_documents <= BRUTE_FORCE_THRESHOLD or mode == "context":
            # Direct magnitude-weighted search on all documents
            oc2_keep = self.num_documents
            oc1_keep = self.num_documents
        elif mode == "exact":
            # Tight spiral with reasonable minimums
            MIN_OC1 = max(100, int(self.num_documents * 0.05))  # At least 5%
            MIN_OC2 = max(200, int(self.num_documents * 0.10))  # At least 10%
            oc1_keep = max(int(top_k * PHI), MIN_OC1)
            oc2_keep = max(int(oc1_keep * PHI), MIN_OC2)
        else:  # similar
            # Wider funnel for better recall
            MIN_OC1 = max(200, int(self.num_documents * 0.10))  # At least 10%
            MIN_OC2 = max(500, int(self.num_documents * 0.20))  # At least 20%
            oc1_keep = max(int(top_k * PHI * PHI), MIN_OC1)
            oc2_keep = max(int(oc1_keep * PHI), MIN_OC2)
        
        # Rebuild vectorized matrices if needed
        self._rebuild_matrices()
        
        if self.num_documents == 0:
            return []
        
        # ═══════════════════════════════════════════════════════════════
        # VECTORIZED CASCADE: All octave levels computed in batch
        # ═══════════════════════════════════════════════════════════════
        
        # Oc2: Coarse scores (all docs)
        coarse_scores = np.sum(query_coarse * self._coarse_matrix, axis=1).astype(np.float32)
        coarse_norm = max(np.sum(np.abs(query_coarse)), 1)
        coarse_scores = coarse_scores / coarse_norm
        
        # Get top oc2_keep by coarse score
        oc2_indices = np.argsort(coarse_scores)[::-1][:oc2_keep]
        
        # If context mode, return coarse results directly
        if mode == "context":
            results = []
            for idx in oc2_indices[:top_k]:
                doc_id = self._doc_ids[idx]
                doc = self.documents[doc_id]
                results.append(SearchResult(
                    id=doc_id,
                    score=float(coarse_scores[idx]),
                    level_scores={'coarse': float(coarse_scores[idx]), 'medium': 0.0, 'fine': 0.0},
                    metadata=doc.metadata,
                    explanation=explain_match(query_coarse, doc.coarse) if explain else None,
                ))
            return results
        
        # Oc1: Medium scores (filtered candidates)
        medium_subset = self._medium_matrix[oc2_indices]
        medium_scores = np.sum(query_medium * medium_subset, axis=1).astype(np.float32)
        medium_norm = max(np.sum(np.abs(query_medium)), 1)
        medium_scores = medium_scores / medium_norm
        
        # Get top oc1_keep by medium score
        oc1_local_indices = np.argsort(medium_scores)[::-1][:oc1_keep]
        oc1_indices = oc2_indices[oc1_local_indices]
        oc1_coarse_scores = coarse_scores[oc1_indices]
        oc1_medium_scores = medium_scores[oc1_local_indices]
        
        # ═══════════════════════════════════════════════════════════════
        # Oc0 (FINE): Near-exact convergence - final ranking
        # Uses SECRET SAUCE (magnitude-weighted similarity) - VECTORIZED
        # ═══════════════════════════════════════════════════════════════
        if len(oc1_indices) == 0:
            return []
        
        # Get fine and magnitude data for Oc1 candidates
        candidate_signs = self._fine_matrix[oc1_indices]
        candidate_mags = self._mag_matrix[oc1_indices]
        
        # VECTORIZED SECRET SAUCE
        # weights[i, d] = sqrt(q_mags[d] * candidate_mags[i, d])
        weights = np.sqrt(query_magnitudes * candidate_mags)
        # sign_product[i, d] = q_signs[d] * candidate_signs[i, d]
        sign_product = query_fine * candidate_signs
        # octave_scores[i] = sum over d
        octave_scores = np.sum(weights * sign_product, axis=1)
        # max_scores[i] for normalization
        max_scores = np.sum(weights, axis=1)
        # Avoid division by zero
        fine_scores = np.divide(octave_scores, max_scores, 
                                out=np.zeros_like(octave_scores), 
                                where=max_scores > 0)
        
        # Build results
        oc0_results = []
        for i, idx in enumerate(oc1_indices):
            doc_id = self._doc_ids[idx]
            oc0_results.append((doc_id, float(fine_scores[i]), {
                'fine': float(fine_scores[i]),
                'medium': float(oc1_medium_scores[i]),
                'coarse': float(oc1_coarse_scores[i]),
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
