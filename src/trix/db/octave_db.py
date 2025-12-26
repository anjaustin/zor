"""
DB Cooper: OctaveDB - The main interface.

Multi-resolution ternary retrieval with glassbox explainability.
Find the exact, the similar, and the related.
"""

import numpy as np
from typing import List, Optional, Union, Callable
from dataclasses import dataclass

from .core import ternary_quantize, derive_hierarchy, explain_match
from .index import OctaveIndex, SearchResult


@dataclass
class OctaveDBConfig:
    """Configuration for OctaveDB."""
    dimensions: int = 384
    pool_factor: int = 4
    quantize_threshold: float = 0.0
    quantize_sparsity: Optional[float] = 0.3  # 30% zeros
    coarse_threshold: float = 0.3


class OctaveDB:
    """
    Octave DB: Multi-resolution ternary retrieval.
    
    CODENAME: DB Cooper
    
    Features:
        - Three search modes: exact, similar, context
        - Glassbox explainability
        - Ternary storage (compact)
        - Hierarchical search (fast)
    
    Usage:
        db = OctaveDB(dimensions=384)
        
        # Add documents (float embeddings, auto-quantized)
        db.add("doc1", embedding1)
        db.add("doc2", embedding2)
        
        # Search with different modes
        results = db.search(query, mode="similar", top_k=10)
        results = db.search(query, mode="exact", top_k=1)
        results = db.search(query, mode="context", top_k=100)
        
        # Explain matches
        explanation = db.explain(query, "doc1")
    """
    
    def __init__(
        self,
        dimensions: int = 384,
        pool_factor: int = 4,
        quantize_threshold: float = 0.0,
        quantize_sparsity: Optional[float] = 0.3,
        coarse_threshold: float = 0.3,
        embedder: Optional[Callable] = None,
    ):
        """
        Initialize OctaveDB.
        
        Args:
            dimensions: Embedding dimensionality
            pool_factor: Pooling factor between octave levels
            quantize_threshold: Threshold for ternary quantization
            quantize_sparsity: Target sparsity (fraction of zeros)
            coarse_threshold: Minimum coarse similarity for candidates
            embedder: Optional function to embed text → float array
        """
        self.config = OctaveDBConfig(
            dimensions=dimensions,
            pool_factor=pool_factor,
            quantize_threshold=quantize_threshold,
            quantize_sparsity=quantize_sparsity,
            coarse_threshold=coarse_threshold,
        )
        
        self.index = OctaveIndex(
            dimensions=dimensions,
            pool_factor=pool_factor,
            coarse_threshold=coarse_threshold,
        )
        
        self.embedder = embedder
        
        # Track original float embeddings if needed for re-quantization
        self._float_cache: dict = {}
    
    def _to_ternary(self, embedding: np.ndarray) -> np.ndarray:
        """Convert float embedding to ternary."""
        return ternary_quantize(
            embedding,
            threshold=self.config.quantize_threshold,
            sparsity_target=self.config.quantize_sparsity,
        )
    
    def _embed(self, text: str) -> np.ndarray:
        """Embed text to float array."""
        if self.embedder is None:
            raise ValueError("No embedder configured. Provide embeddings directly.")
        return self.embedder(text)
    
    def add(
        self,
        doc_id: str,
        embedding: Union[np.ndarray, str],
        metadata: Optional[dict] = None,
        is_ternary: bool = False,
    ) -> None:
        """
        Add document to database.
        
        Args:
            doc_id: Unique document identifier
            embedding: Float embedding array OR text (if embedder configured)
            metadata: Optional metadata dict
            is_ternary: If True, embedding is already ternary
        """
        # Handle text input
        if isinstance(embedding, str):
            embedding = self._embed(embedding)
            is_ternary = False
        
        embedding = np.asarray(embedding)
        
        # Quantize if needed
        if is_ternary:
            ternary = embedding.astype(np.int8)
        else:
            ternary = self._to_ternary(embedding)
        
        # Add to index
        self.index.add(doc_id, ternary, metadata=metadata)
    
    def add_batch(
        self,
        doc_ids: List[str],
        embeddings: Union[np.ndarray, List[str]],
        metadata_list: Optional[List[dict]] = None,
        is_ternary: bool = False,
    ) -> None:
        """Add multiple documents efficiently."""
        # Handle text input
        if isinstance(embeddings[0], str):
            embeddings = np.array([self._embed(t) for t in embeddings])
            is_ternary = False
        
        embeddings = np.asarray(embeddings)
        
        # Quantize if needed
        if is_ternary:
            ternary = embeddings.astype(np.int8)
        else:
            ternary = self._to_ternary(embeddings)
        
        # Add to index
        self.index.add_batch(doc_ids, ternary, metadata_list)
    
    def remove(self, doc_id: str) -> bool:
        """Remove document from database."""
        return self.index.remove(doc_id)
    
    def search(
        self,
        query: Union[np.ndarray, str],
        mode: str = "similar",
        top_k: int = 10,
        explain: bool = False,
        is_ternary: bool = False,
    ) -> List[SearchResult]:
        """
        Search database.
        
        Args:
            query: Query embedding OR text (if embedder configured)
            mode: Search mode
                - "exact": Fine-level matching, strict identity
                - "similar": Multi-level matching, balanced
                - "context": Coarse-level matching, broad discovery
            top_k: Number of results
            explain: Include per-dimension explanations
            is_ternary: If True, query is already ternary
        
        Returns:
            List of SearchResult with scores and optional explanations
        """
        # Handle text input
        if isinstance(query, str):
            query = self._embed(query)
            is_ternary = False
        
        query = np.asarray(query)
        
        # Quantize if needed
        if is_ternary:
            query_ternary = query.astype(np.int8)
        else:
            query_ternary = self._to_ternary(query)
        
        # Search
        return self.index.search(
            query_ternary,
            mode=mode,
            top_k=top_k,
            explain=explain,
        )
    
    def explain(
        self,
        query: Union[np.ndarray, str],
        doc_id: str,
        is_ternary: bool = False,
    ) -> Optional[dict]:
        """
        Explain why query matches document. Glassbox.
        
        Args:
            query: Query embedding or text
            doc_id: Document ID to explain
            is_ternary: If True, query is already ternary
        
        Returns:
            Explanation dict with agreement/conflict dimensions
        """
        doc = self.index.get(doc_id)
        if doc is None:
            return None
        
        # Handle text input
        if isinstance(query, str):
            query = self._embed(query)
            is_ternary = False
        
        query = np.asarray(query)
        
        # Quantize if needed
        if is_ternary:
            query_ternary = query.astype(np.int8)
        else:
            query_ternary = self._to_ternary(query)
        
        return explain_match(query_ternary, doc.fine)
    
    def get(self, doc_id: str) -> Optional[dict]:
        """Get document by ID."""
        doc = self.index.get(doc_id)
        if doc is None:
            return None
        return {
            'id': doc.id,
            'fine': doc.fine,
            'medium': doc.medium,
            'coarse': doc.coarse,
            'metadata': doc.metadata,
        }
    
    def stats(self) -> dict:
        """Get database statistics."""
        return self.index.stats()
    
    def __len__(self) -> int:
        return self.index.num_documents
    
    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self.index.documents
