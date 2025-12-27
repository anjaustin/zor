"""
Codex Layer - A Hollywood Squares Field for Meaning

The Codex is a coordination OS for archetypal meaning.
64 Gene Keys as 64 tiles in a field, connected by edges (partners, rings, transforms).
Activation propagates via message passing until the field converges.

"Structure is meaning. The wiring determines the behavior."
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import json


@dataclass
class GeneKeyTile:
    """
    A single Gene Key as a tile in the Codex field.
    
    Like NativeProgrammableTile but for meaning, not computation.
    """
    id: int  # 1-64
    shadow: str
    gift: str
    siddhi: str
    
    # Embedding = the tile's signature for routing
    embedding: Optional[np.ndarray] = None
    
    # Edges to other tiles (message channels)
    partner: Optional[int] = None  # Programming partner (bidirectional)
    ring_members: List[int] = field(default_factory=list)  # Codon ring
    
    # Activation state
    activation: float = 0.0
    _prev_activation: float = 0.0
    
    # Message inbox (activation received from neighbors)
    _inbox: float = 0.0
    
    def receive(self, amount: float):
        """Receive activation from a neighbor."""
        self._inbox += amount
    
    def process(self) -> float:
        """
        Process inbox and update activation.
        Returns the change in activation (for convergence detection).
        """
        self._prev_activation = self.activation
        # Combine direct activation with received messages
        self.activation = max(self.activation, self._inbox)
        self._inbox = 0.0
        return abs(self.activation - self._prev_activation)
    
    def reset(self):
        """Reset activation state."""
        self.activation = 0.0
        self._prev_activation = 0.0
        self._inbox = 0.0


@dataclass
class CodexEdge:
    """An edge in the Codex graph."""
    source: int
    target: int
    edge_type: str  # 'partner', 'ring', 'transform'
    weight: float


class CodexField:
    """
    The Codex as a Hollywood Squares field.
    
    64 tiles (Gene Keys) with topology defined by:
    - Partner edges (32 pairs, bidirectional, weight=0.8)
    - Ring edges (codon rings, weight=0.5)
    - Transform edges (shadow→gift→siddhi, weight=0.6)
    
    Query injection activates tiles.
    Message passing propagates activation.
    Field relaxes to stable pattern.
    """
    
    # Edge weights
    PARTNER_WEIGHT = 0.8
    RING_WEIGHT = 0.5
    TRANSFORM_WEIGHT = 0.6
    
    # Convergence threshold
    CONVERGENCE_THRESHOLD = 1e-4
    MAX_ITERATIONS = 10
    
    def __init__(self):
        self.tiles: Dict[int, GeneKeyTile] = {}
        self.edges: List[CodexEdge] = []
        self._adjacency: Dict[int, List[Tuple[int, float]]] = {}  # id -> [(neighbor_id, weight), ...]
        self._embeddings: Optional[np.ndarray] = None  # [64, d_model]
        self._id_to_idx: Dict[int, int] = {}  # Gene Key ID -> matrix index
        self._idx_to_id: Dict[int, int] = {}  # matrix index -> Gene Key ID
    
    def load_gene_keys(self, graph_data: Dict):
        """
        Load Gene Keys from extracted graph data.
        
        Args:
            graph_data: Dict mapping Gene Key ID to {genekey, shadow, gift, siddhi, partner, ...}
        """
        # Create tiles
        for key, data in graph_data.items():
            gk_id = int(key)
            tile = GeneKeyTile(
                id=gk_id,
                shadow=data.get('shadow') or '',
                gift=data.get('gift') or '',
                siddhi=data.get('siddhi') or '',
                partner=data.get('partner'),
            )
            self.tiles[gk_id] = tile
            idx = len(self._id_to_idx)
            self._id_to_idx[gk_id] = idx
            self._idx_to_id[idx] = gk_id
        
        # Create partner edges (bidirectional)
        seen_partners = set()
        for gk_id, tile in self.tiles.items():
            if tile.partner and tile.partner in self.tiles:
                pair = tuple(sorted([gk_id, tile.partner]))
                if pair not in seen_partners:
                    seen_partners.add(pair)
                    # Bidirectional edges
                    self.edges.append(CodexEdge(gk_id, tile.partner, 'partner', self.PARTNER_WEIGHT))
                    self.edges.append(CodexEdge(tile.partner, gk_id, 'partner', self.PARTNER_WEIGHT))
        
        # Build adjacency list
        self._build_adjacency()
        
        print(f"Loaded {len(self.tiles)} Gene Keys, {len(self.edges)} edges")
    
    def _build_adjacency(self):
        """Build adjacency list from edges."""
        self._adjacency = {gk_id: [] for gk_id in self.tiles}
        for edge in self.edges:
            self._adjacency[edge.source].append((edge.target, edge.weight))
    
    def set_embeddings(self, embeddings: np.ndarray, ids: Optional[List[int]] = None):
        """
        Set embeddings for Gene Keys.
        
        Args:
            embeddings: [n, d_model] embedding matrix
            ids: Optional list of Gene Key IDs corresponding to each row.
                 If None, assumes rows 0-63 map to Gene Keys 1-64.
        """
        if ids is None:
            # Assume sequential mapping
            for i in range(min(embeddings.shape[0], 64)):
                gk_id = i + 1
                if gk_id in self.tiles:
                    self.tiles[gk_id].embedding = embeddings[i]
        else:
            for i, gk_id in enumerate(ids):
                if gk_id in self.tiles:
                    self.tiles[gk_id].embedding = embeddings[i]
        
        # Build embedding matrix for fast similarity computation
        self._build_embedding_matrix()
    
    def _build_embedding_matrix(self):
        """Build matrix of all embeddings for vectorized operations."""
        if not self.tiles:
            return
        
        d_model = None
        for tile in self.tiles.values():
            if tile.embedding is not None:
                d_model = tile.embedding.shape[0]
                break
        
        if d_model is None:
            return
        
        self._embeddings = np.zeros((len(self.tiles), d_model), dtype=np.float32)
        for gk_id, tile in self.tiles.items():
            idx = self._id_to_idx[gk_id]
            if tile.embedding is not None:
                self._embeddings[idx] = tile.embedding
    
    # === Core Operations ===
    
    def activate(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Compute initial activation pattern from query.
        
        Args:
            query_embedding: [d_model] query vector
            
        Returns:
            [64] activation pattern (cosine similarity to each Gene Key)
        """
        if self._embeddings is None:
            raise ValueError("Embeddings not set. Call set_embeddings() first.")
        
        # Normalize for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        emb_norms = self._embeddings / (np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity
        similarities = emb_norms @ query_norm
        
        # ReLU (only positive activations)
        activations = np.maximum(similarities, 0)
        
        # Set tile activations
        for gk_id, tile in self.tiles.items():
            idx = self._id_to_idx[gk_id]
            tile.activation = float(activations[idx])
            tile._prev_activation = 0.0
            tile._inbox = 0.0
        
        return activations
    
    def propagate_step(self) -> float:
        """
        One step of message passing.
        
        Each tile sends its activation * edge_weight to neighbors.
        Returns max change in activation (for convergence detection).
        """
        # Phase 1: Send messages
        for gk_id, tile in self.tiles.items():
            if tile.activation > 0.01:  # Only send if activated
                for neighbor_id, weight in self._adjacency.get(gk_id, []):
                    message = tile.activation * weight
                    self.tiles[neighbor_id].receive(message)
        
        # Phase 2: Process messages
        max_change = 0.0
        for tile in self.tiles.values():
            change = tile.process()
            max_change = max(max_change, change)
        
        return max_change
    
    def propagate(self, max_iters: Optional[int] = None) -> Tuple[int, np.ndarray]:
        """
        Propagate activation until convergence.
        
        Returns:
            (num_iterations, final_activation_pattern)
        """
        max_iters = max_iters or self.MAX_ITERATIONS
        
        for i in range(max_iters):
            change = self.propagate_step()
            if change < self.CONVERGENCE_THRESHOLD:
                break
        
        # Extract final pattern
        pattern = self.get_activation_pattern()
        return i + 1, pattern
    
    def get_activation_pattern(self) -> np.ndarray:
        """Get current activation as [64] vector."""
        pattern = np.zeros(len(self.tiles), dtype=np.float32)
        for gk_id, tile in self.tiles.items():
            idx = self._id_to_idx[gk_id]
            pattern[idx] = tile.activation
        return pattern
    
    def reset(self):
        """Reset all tile activations."""
        for tile in self.tiles.values():
            tile.reset()
    
    # === Query Interface ===
    
    def illuminate(self, query_embedding: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, str, float]]]:
        """
        Full illumination pipeline: activate → propagate → explain.
        
        Args:
            query_embedding: [d_model] query vector
            
        Returns:
            (activation_pattern, explanations)
            explanations: [(gk_id, gift_name, activation), ...] top activated keys
        """
        self.reset()
        
        # Activate
        initial = self.activate(query_embedding)
        
        # Propagate
        iters, final = self.propagate()
        
        # Explain top activations
        explanations = []
        for gk_id, tile in self.tiles.items():
            if tile.activation > 0.1:
                explanations.append((gk_id, tile.gift, tile.activation))
        
        explanations.sort(key=lambda x: -x[2])
        
        return final, explanations[:5]
    
    def get_illuminated_keys(self, threshold: float = 0.1) -> List[int]:
        """Get IDs of Gene Keys above activation threshold."""
        return [gk_id for gk_id, tile in self.tiles.items() if tile.activation > threshold]
    
    # === Tracing (Hollywood Squares style) ===
    
    def trace(self, query_embedding: np.ndarray) -> List[Dict]:
        """
        Trace the full activation propagation.
        
        Returns list of events: [{iter, gk_id, activation, source}, ...]
        """
        self.reset()
        events = []
        
        # Initial activation
        self.activate(query_embedding)
        for gk_id, tile in self.tiles.items():
            if tile.activation > 0.01:
                events.append({
                    'iter': 0,
                    'gk_id': gk_id,
                    'gift': tile.gift,
                    'activation': tile.activation,
                    'source': 'query'
                })
        
        # Propagation
        for i in range(self.MAX_ITERATIONS):
            # Record who will send
            senders = [(gk_id, tile.activation) for gk_id, tile in self.tiles.items() if tile.activation > 0.01]
            
            change = self.propagate_step()
            
            # Record changes
            for gk_id, tile in self.tiles.items():
                if tile.activation > tile._prev_activation + 0.01:
                    # Find who sent to us
                    sources = []
                    for sender_id, sender_act in senders:
                        for neighbor_id, weight in self._adjacency.get(sender_id, []):
                            if neighbor_id == gk_id:
                                sources.append(sender_id)
                    
                    events.append({
                        'iter': i + 1,
                        'gk_id': gk_id,
                        'gift': tile.gift,
                        'activation': tile.activation,
                        'source': f'propagation from {sources}'
                    })
            
            if change < self.CONVERGENCE_THRESHOLD:
                break
        
        return events


class CodexLayer:
    """
    Integration layer between Cooper and the Codex field.
    
    Computes document archetypal fingerprints and illuminates queries.
    """
    
    def __init__(self, codex_field: CodexField, alpha: float = 0.7):
        """
        Args:
            codex_field: Initialized CodexField with embeddings
            alpha: Weight for vector similarity vs archetypal alignment (0-1)
        """
        self.field = codex_field
        self.alpha = alpha
    
    def compute_document_fingerprint(self, doc_embedding: np.ndarray) -> np.ndarray:
        """
        Compute archetypal fingerprint for a document.
        
        Args:
            doc_embedding: [d_model] document embedding
            
        Returns:
            [64] archetypal activation pattern
        """
        pattern, _ = self.field.illuminate(doc_embedding)
        return pattern
    
    def compute_batch_fingerprints(self, doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute fingerprints for batch of documents.
        
        Args:
            doc_embeddings: [n_docs, d_model]
            
        Returns:
            [n_docs, 64] archetypal patterns
        """
        n_docs = doc_embeddings.shape[0]
        n_keys = len(self.field.tiles)
        fingerprints = np.zeros((n_docs, n_keys), dtype=np.float32)
        
        for i in range(n_docs):
            fingerprints[i] = self.compute_document_fingerprint(doc_embeddings[i])
        
        return fingerprints
    
    def illuminate_query(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidate_fingerprints: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, str, float]]]:
        """
        Score candidates using both vector similarity and archetypal alignment.
        
        Args:
            query_embedding: [d_model]
            candidate_embeddings: [n_candidates, d_model]
            candidate_fingerprints: [n_candidates, 64]
            
        Returns:
            (combined_scores, archetypal_scores, query_explanation)
        """
        # Query archetypal pattern
        query_pattern, explanation = self.field.illuminate(query_embedding)
        
        # Vector similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        cand_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
        vector_scores = cand_norms @ query_norm
        
        # Archetypal alignment (cosine similarity of fingerprints)
        query_pattern_norm = query_pattern / (np.linalg.norm(query_pattern) + 1e-8)
        cand_pattern_norms = candidate_fingerprints / (np.linalg.norm(candidate_fingerprints, axis=1, keepdims=True) + 1e-8)
        archetypal_scores = cand_pattern_norms @ query_pattern_norm
        
        # Combined score
        combined_scores = self.alpha * vector_scores + (1 - self.alpha) * archetypal_scores
        
        return combined_scores, archetypal_scores, explanation


def load_codex_from_json(path: str) -> CodexField:
    """Load a CodexField from Gene Keys JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    field = CodexField()
    field.load_gene_keys(data)
    return field
