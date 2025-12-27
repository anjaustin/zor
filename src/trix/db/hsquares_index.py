"""
Hollywood Squares Index: Vector Search via Message Passing

The topology IS the algorithm.
64 tiles. Partner edges. Ring clusters.
The field relaxes toward the answer.

Structure is meaning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import heapq


@dataclass
class Tile:
    """A single tile in the Hollywood Squares field."""
    id: int
    centroid: np.ndarray
    vectors: List[Tuple[str, np.ndarray]]  # (doc_id, vector)
    neighbors: Set[int]  # Connected tiles
    
    # Message passing state
    inbox: List[Tuple[float, str, np.ndarray]] = field(default_factory=list)  # (score, doc_id, vector)
    best_local: List[Tuple[float, str]] = field(default_factory=list)  # Top-k from this tile
    
    def search_local(self, query: np.ndarray, k: int) -> List[Tuple[float, str]]:
        """Search vectors in this tile."""
        if not self.vectors:
            return []
        
        scores = []
        for doc_id, vec in self.vectors:
            score = float(np.dot(query, vec))
            scores.append((score, doc_id))
        
        scores.sort(reverse=True)
        return scores[:k]


class HollywoodSquaresIndex:
    """
    Vector search index using Hollywood Squares topology.
    
    64 tiles arranged in 8x8 grid with:
    - Partner edges (horizontal neighbors)
    - Ring edges (vertical/diagonal clusters)
    
    Search = message passing until field relaxes.
    """
    
    def __init__(self, dimensions: int, n_tiles: int = 64):
        self.dimensions = dimensions
        self.n_tiles = n_tiles
        self.grid_size = int(np.sqrt(n_tiles))
        
        # Tiles
        self.tiles: Dict[int, Tile] = {}
        self.centroids: Optional[np.ndarray] = None
        
        # Document tracking
        self.doc_to_tile: Dict[str, int] = {}
        self.num_documents = 0
        
        # Contiguous storage (for fast vectorized search)
        self._vectors: Optional[np.ndarray] = None  # (N, D) matrix
        self._ids: List[str] = []  # Doc IDs in order
        self._tile_ranges: Dict[int, Tuple[int, int]] = {}  # tile_id -> (start, end)
        self._built = False
        
        # Build topology
        self._build_topology()
    
    def _build_topology(self):
        """Build the 8x8 grid with neighbor connections."""
        for i in range(self.n_tiles):
            row, col = i // self.grid_size, i % self.grid_size
            neighbors = set()
            
            # 8-connected neighbors (horizontal, vertical, diagonal)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                        neighbors.add(nr * self.grid_size + nc)
            
            self.tiles[i] = Tile(
                id=i,
                centroid=np.zeros(self.dimensions, dtype=np.float32),
                vectors=[],
                neighbors=neighbors,
            )
    
    def _assign_to_tile(self, vector: np.ndarray) -> int:
        """Assign vector to nearest centroid tile."""
        if self.centroids is None:
            return 0  # Default to first tile before centroids are computed
        
        similarities = np.dot(self.centroids, vector)
        return int(np.argmax(similarities))
    
    def add(self, doc_id: str, vector: np.ndarray):
        """Add vector to index."""
        vector = np.asarray(vector, dtype=np.float32)
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        # Assign to tile
        tile_id = self._assign_to_tile(vector)
        self.tiles[tile_id].vectors.append((doc_id, vector))
        self.doc_to_tile[doc_id] = tile_id
        self.num_documents += 1
        self._built = False  # Invalidate contiguous storage
    
    def build_centroids(self, n_iterations: int = 10):
        """Build tile centroids using k-means."""
        if self.num_documents == 0:
            return
        
        # Collect all vectors
        all_vectors = []
        all_ids = []
        for tile in self.tiles.values():
            for doc_id, vec in tile.vectors:
                all_vectors.append(vec)
                all_ids.append(doc_id)
        
        if not all_vectors:
            return
        
        vectors = np.array(all_vectors)
        n = len(vectors)
        
        # Initialize centroids (k-means++)
        centroids = np.zeros((self.n_tiles, self.dimensions), dtype=np.float32)
        
        # First centroid: random
        idx = np.random.randint(n)
        centroids[0] = vectors[idx]
        
        # Remaining centroids: weighted by distance
        for k in range(1, self.n_tiles):
            # Distance to nearest centroid
            dists = np.zeros(n)
            for i in range(n):
                min_dist = float('inf')
                for j in range(k):
                    d = 1 - np.dot(vectors[i], centroids[j])
                    min_dist = min(min_dist, d)
                dists[i] = min_dist
            
            # Sample proportional to distance squared
            probs = dists ** 2
            probs = probs / probs.sum()
            idx = np.random.choice(n, p=probs)
            centroids[k] = vectors[idx]
        
        # K-means iterations
        for _ in range(n_iterations):
            # Assign vectors to nearest centroid
            assignments = np.argmax(np.dot(vectors, centroids.T), axis=1)
            
            # Update centroids
            for k in range(self.n_tiles):
                mask = assignments == k
                if mask.sum() > 0:
                    centroids[k] = vectors[mask].mean(axis=0)
                    centroids[k] = centroids[k] / (np.linalg.norm(centroids[k]) + 1e-10)
        
        self.centroids = centroids
        
        # Update tile centroids
        for k in range(self.n_tiles):
            self.tiles[k].centroid = centroids[k]
        
        # Reassign vectors to tiles
        self._reassign_vectors()
    
    def _reassign_vectors(self):
        """Reassign all vectors to their nearest centroid tile."""
        # Collect all vectors
        all_docs = []
        for tile in self.tiles.values():
            all_docs.extend(tile.vectors)
            tile.vectors = []
        
        # Clear mappings
        self.doc_to_tile.clear()
        
        # Reassign
        for doc_id, vec in all_docs:
            tile_id = self._assign_to_tile(vec)
            self.tiles[tile_id].vectors.append((doc_id, vec))
            self.doc_to_tile[doc_id] = tile_id
        
        # Build contiguous storage
        self._build_contiguous()
    
    def _build_contiguous(self):
        """Build contiguous storage for fast vectorized search."""
        all_vecs = []
        all_ids = []
        
        # Store vectors grouped by tile
        for tile_id in range(self.n_tiles):
            tile = self.tiles[tile_id]
            start = len(all_vecs)
            for doc_id, vec in tile.vectors:
                all_vecs.append(vec)
                all_ids.append(doc_id)
            end = len(all_vecs)
            self._tile_ranges[tile_id] = (start, end)
        
        if all_vecs:
            self._vectors = np.array(all_vecs, dtype=np.float32)
            self._ids = all_ids
        else:
            self._vectors = np.zeros((0, self.dimensions), dtype=np.float32)
            self._ids = []
        
        self._built = True
    
    def search(self, query: np.ndarray, top_k: int = 10, n_probes: int = 8) -> List[Tuple[str, float]]:
        """
        Search by probing top-n tiles by centroid similarity.
        
        Uses contiguous storage for FAST vectorized search.
        """
        query = np.asarray(query, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        
        if self.centroids is None or self.num_documents == 0 or not self._built:
            return []
        
        # Rank tiles by centroid similarity
        centroid_scores = np.dot(self.centroids, query)
        top_tiles = np.argsort(centroid_scores)[::-1][:n_probes]
        
        # Collect indices from top tiles
        indices = []
        for tile_id in top_tiles:
            start, end = self._tile_ranges[int(tile_id)]
            if start < end:
                indices.extend(range(start, end))
        
        if not indices:
            return []
        
        # Single matrix multiply using contiguous storage
        indices = np.array(indices)
        candidate_vecs = self._vectors[indices]
        scores = np.dot(candidate_vecs, query)
        
        # Get top-k
        if len(scores) <= top_k:
            top_local = np.argsort(scores)[::-1]
        else:
            top_local = np.argpartition(scores, -top_k)[-top_k:]
            top_local = top_local[np.argsort(scores[top_local])[::-1]]
        
        results = [(self._ids[indices[i]], float(scores[i])) for i in top_local]
        return results
    
    def search_quality(self, query: np.ndarray, top_k: int = 10, 
                       n_probes: int = 32) -> List[Tuple[str, float]]:
        """
        Quality mode: more probes for higher recall.
        """
        return self.search(query, top_k=top_k, n_probes=n_probes)
    
    def search_brute(self, query: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """Brute force search over all vectors."""
        query = np.asarray(query, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        
        if not self._built or self._vectors is None:
            return []
        
        scores = np.dot(self._vectors, query)
        
        if len(scores) <= top_k:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        
        return [(self._ids[i], float(scores[i])) for i in top_indices]
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        tile_sizes = [len(t.vectors) for t in self.tiles.values()]
        return {
            "num_documents": self.num_documents,
            "num_tiles": self.n_tiles,
            "tile_size_mean": np.mean(tile_sizes) if tile_sizes else 0,
            "tile_size_std": np.std(tile_sizes) if tile_sizes else 0,
            "tile_size_min": min(tile_sizes) if tile_sizes else 0,
            "tile_size_max": max(tile_sizes) if tile_sizes else 0,
            "empty_tiles": sum(1 for s in tile_sizes if s == 0),
        }


# Factory function
def create_hsquares_index(dimensions: int, n_tiles: int = 64) -> HollywoodSquaresIndex:
    """Create a Hollywood Squares index."""
    return HollywoodSquaresIndex(dimensions=dimensions, n_tiles=n_tiles)
