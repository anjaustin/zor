"""
Codex Squares Index: Hollywood Squares + Gene Keys

64 tiles = 64 Gene Keys (archetypes of human experience)
Centroids = Gene Key semantic embeddings  
Edges = Programming partners + Codon ring membership

Every search result comes with archetypal context.
The topology IS meaning. Structure IS the algorithm.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field


@dataclass
class CodexTile:
    """A Gene Key tile in the Codex field."""
    gk_number: int
    shadow: str
    gift: str
    siddhi: str
    centroid: np.ndarray
    vectors: List[Tuple[str, np.ndarray]] = field(default_factory=list)
    neighbors: Set[int] = field(default_factory=set)
    
    @property
    def name(self) -> str:
        return f"{self.shadow}→{self.gift}→{self.siddhi}"


@dataclass 
class CodexResult:
    """Search result with archetypal context."""
    doc_id: str
    score: float
    gk_number: int
    shadow: str
    gift: str
    siddhi: str
    
    def __repr__(self):
        return f"CodexResult({self.doc_id}, score={self.score:.3f}, GK{self.gk_number}: {self.gift})"


class CodexSquaresIndex:
    """
    Hollywood Squares Index with Gene Key topology.
    
    Standard IVF:  Vectors → K-Means Centroids → Nearest Cluster → Search
    CodexSquares:  Vectors → Gene Key Archetypes → Meaning Cluster → Search
    
    Same algorithm, but centroids ARE the 64 archetypes.
    """
    
    def __init__(self, gk_data: dict, embedder=None):
        """
        Initialize with Gene Keys data and optional embedder.
        
        Args:
            gk_data: Gene Keys dictionary with 'gene_keys', 'partners', 'codon_rings'
            embedder: Function that takes list of strings, returns (N, D) embeddings
        """
        self.dimensions = 384
        self.n_tiles = 64
        self.tiles: Dict[int, CodexTile] = {}
        self.centroids: Optional[np.ndarray] = None
        
        # Contiguous storage
        self._vectors: Optional[np.ndarray] = None
        self._ids: List[str] = []
        self._tile_ranges: Dict[int, Tuple[int, int]] = {}
        self._idx_to_gk: Dict[int, int] = {}
        self._built = False
        self.num_documents = 0
        
        # Build tiles
        self._build_tiles(gk_data, embedder)
        self._build_topology(gk_data)
    
    def _build_tiles(self, gk_data: dict, embedder):
        """Create tiles from Gene Keys."""
        gene_keys = gk_data.get('gene_keys', {})
        
        descriptions = []
        for i in range(1, 65):
            gk = gene_keys.get(str(i), {})
            shadow = gk.get('shadow', f'Shadow{i}')
            gift = gk.get('gift', f'Gift{i}')
            siddhi = gk.get('siddhi', f'Siddhi{i}')
            
            self.tiles[i] = CodexTile(
                gk_number=i,
                shadow=shadow,
                gift=gift,
                siddhi=siddhi,
                centroid=np.zeros(self.dimensions, dtype=np.float32),
                neighbors=set()
            )
            
            descriptions.append(
                f"Gene Key {i}: {shadow} transforms through {gift} to {siddhi}"
            )
        
        # Embed descriptions
        if embedder is not None:
            embeddings = embedder(descriptions)
            if hasattr(embeddings, 'numpy'):
                embeddings = embeddings.numpy()
            embeddings = np.asarray(embeddings, dtype=np.float32)
            
            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-10)
            
            for i, emb in enumerate(embeddings, 1):
                self.tiles[i].centroid = emb
                
            self.dimensions = embeddings.shape[1]
        
        # Stack centroids
        self.centroids = np.array([self.tiles[i].centroid for i in range(1, 65)])
    
    def _build_topology(self, gk_data: dict):
        """Build edges from partners and codon rings."""
        partners = gk_data.get('partners', {})
        rings = gk_data.get('codon_rings', {})
        
        # Partner edges (bidirectional)
        for gk_str, partner in partners.items():
            try:
                gk = int(gk_str)
                if 1 <= gk <= 64 and 1 <= partner <= 64:
                    self.tiles[gk].neighbors.add(partner)
                    self.tiles[partner].neighbors.add(gk)
            except (ValueError, TypeError):
                continue
        
        # Ring edges (all members connected)
        for ring_name, members in rings.items():
            valid_members = [m for m in members if 1 <= m <= 64]
            for i, m1 in enumerate(valid_members):
                for m2 in valid_members[i+1:]:
                    self.tiles[m1].neighbors.add(m2)
                    self.tiles[m2].neighbors.add(m1)
    
    def set_centroids(self, centroids: np.ndarray):
        """Set centroids directly (64 x D array)."""
        centroids = np.asarray(centroids, dtype=np.float32)
        if centroids.shape[0] != 64:
            raise ValueError(f"Expected 64 centroids, got {centroids.shape[0]}")
        
        self.dimensions = centroids.shape[1]
        self.centroids = centroids
        for i in range(1, 65):
            self.tiles[i].centroid = centroids[i-1]
    
    def add(self, doc_id: str, vector: np.ndarray):
        """Add vector to nearest Gene Key tile."""
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        # Find nearest Gene Key centroid
        scores = np.dot(self.centroids, vector)
        tile_id = int(np.argmax(scores)) + 1  # Gene Keys are 1-indexed
        
        self.tiles[tile_id].vectors.append((doc_id, vector))
        self.num_documents += 1
        self._built = False
    
    def build(self):
        """Build contiguous storage for fast search."""
        all_vecs = []
        all_ids = []
        
        for gk in range(1, 65):
            tile = self.tiles[gk]
            start = len(all_vecs)
            for doc_id, vec in tile.vectors:
                idx = len(all_vecs)
                all_vecs.append(vec)
                all_ids.append(doc_id)
                self._idx_to_gk[idx] = gk
            self._tile_ranges[gk] = (start, len(all_vecs))
        
        if all_vecs:
            self._vectors = np.array(all_vecs, dtype=np.float32)
            self._ids = all_ids
        else:
            self._vectors = np.zeros((0, self.dimensions), dtype=np.float32)
            self._ids = []
        
        self._built = True
    
    def search(self, query: np.ndarray, top_k: int = 10, 
               n_probes: int = 8) -> List[CodexResult]:
        """
        Search with archetypal context.
        
        Returns list of CodexResult with doc_id, score, and Gene Key info.
        """
        query = np.asarray(query, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        
        if not self._built:
            self.build()
        
        if self._vectors is None or len(self._vectors) == 0:
            return []
        
        # Find top Gene Key tiles by centroid similarity
        centroid_scores = np.dot(self.centroids, query)
        top_gks = np.argsort(centroid_scores)[::-1][:n_probes] + 1
        
        # Collect candidate indices
        indices = []
        for gk in top_gks:
            start, end = self._tile_ranges.get(gk, (0, 0))
            indices.extend(range(start, end))
        
        if not indices:
            return []
        
        # Score all candidates in single matrix multiply
        indices = np.array(indices)
        scores = np.dot(self._vectors[indices], query)
        
        # Get top-k
        k = min(top_k, len(scores))
        if k == len(scores):
            top_local = np.argsort(scores)[::-1]
        else:
            top_local = np.argpartition(scores, -k)[-k:]
            top_local = top_local[np.argsort(scores[top_local])[::-1]]
        
        # Build results with archetypal context
        results = []
        for i in top_local:
            idx = indices[i]
            gk = self._idx_to_gk[idx]
            tile = self.tiles[gk]
            results.append(CodexResult(
                doc_id=self._ids[idx],
                score=float(scores[i]),
                gk_number=gk,
                shadow=tile.shadow,
                gift=tile.gift,
                siddhi=tile.siddhi
            ))
        
        return results
    
    def search_brute(self, query: np.ndarray, top_k: int = 10) -> List[CodexResult]:
        """Brute force search over all vectors."""
        query = np.asarray(query, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        
        if not self._built:
            self.build()
        
        if self._vectors is None or len(self._vectors) == 0:
            return []
        
        scores = np.dot(self._vectors, query)
        
        k = min(top_k, len(scores))
        if k == len(scores):
            top_idx = np.argsort(scores)[::-1]
        else:
            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        
        results = []
        for idx in top_idx:
            gk = self._idx_to_gk[idx]
            tile = self.tiles[gk]
            results.append(CodexResult(
                doc_id=self._ids[idx],
                score=float(scores[idx]),
                gk_number=gk,
                shadow=tile.shadow,
                gift=tile.gift,
                siddhi=tile.siddhi
            ))
        
        return results
    
    def get_tile_distribution(self) -> Dict[int, int]:
        """Get number of documents per Gene Key tile."""
        return {gk: len(tile.vectors) for gk, tile in self.tiles.items()}
    
    def get_dominant_archetype(self, doc_id: str) -> Optional[CodexTile]:
        """Get the Gene Key tile for a document."""
        if not self._built:
            self.build()
        
        if doc_id in self._ids:
            idx = self._ids.index(doc_id)
            gk = self._idx_to_gk[idx]
            return self.tiles[gk]
        return None
    
    def explain_result(self, result: CodexResult) -> str:
        """Generate archetypal explanation for a result."""
        return (
            f"This content resonates with Gene Key {result.gk_number}: "
            f"the journey from {result.shadow} (shadow) through {result.gift} (gift) "
            f"to {result.siddhi} (highest expression). "
            f"Relevance score: {result.score:.3f}"
        )


def create_codex_squares(gk_data: dict, embedder=None) -> CodexSquaresIndex:
    """Factory function to create CodexSquaresIndex."""
    return CodexSquaresIndex(gk_data, embedder)
