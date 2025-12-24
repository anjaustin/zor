"""
Doula VDB - Vector Database for Learning Dynamics.

Mesa 16.2: The Doula

The VDB stores embeddings of learning dynamics, enabling:
- Similarity search to find past moments like the current one
- Full transparency (all data accessible, all queries logged)
- Persistence to disk for cross-training wisdom

Glass box: everything is recorded, nothing hidden.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

from .signature import DoulaEntry, DynamicsSignature


class DoulaVDB:
    """
    Vector database for Doula observations.

    Simple implementation using cosine similarity.
    Can be swapped for Qdrant/Pinecone/etc for scale.

    Glass box principles:
    - All data accessible via get_all_entries()
    - All queries logged via get_query_log()
    - Full inspection methods for analysis
    """

    def __init__(
        self,
        path: Optional[str] = None,
        embedding_dim: int = 64,
        auto_save_interval: int = 100,
    ):
        """
        Initialize the VDB.

        Args:
            path: Path for persistence (None = in-memory only)
            embedding_dim: Dimension of dynamics embeddings
            auto_save_interval: Save every N entries (0 = no auto-save)
        """
        self.path = Path(path) if path else None
        self.embedding_dim = embedding_dim
        self.auto_save_interval = auto_save_interval

        # Storage
        self.entries: List[DoulaEntry] = []
        self.embeddings: Tensor = torch.empty(0, embedding_dim)

        # Glass box: log all queries
        self.query_log: List[Dict] = []

        # Metadata
        self.created_at = datetime.now().isoformat()
        self.domain: Optional[str] = None

        # Load if exists
        if self.path and self.path.exists():
            self.load()

    def add(self, entry: DoulaEntry):
        """
        Add entry to VDB.

        Args:
            entry: DoulaEntry to store
        """
        self.entries.append(entry)

        # Add embedding
        emb = entry.dynamics_embedding.unsqueeze(0)
        if self.embeddings.numel() == 0:
            self.embeddings = emb
        else:
            self.embeddings = torch.cat([self.embeddings, emb])

        # Auto-save
        if self.auto_save_interval > 0 and len(self.entries) % self.auto_save_interval == 0:
            if self.path:
                self.save()

    def search(
        self,
        query_embedding: Tensor,
        k: int = 5,
        phase_filter: Optional[str] = None,
        layer_filter: Optional[int] = None,
    ) -> List[DoulaEntry]:
        """
        Search for similar entries.

        Args:
            query_embedding: Query vector
            k: Number of results
            phase_filter: Only return entries from this phase
            layer_filter: Only return entries from this layer

        Returns:
            List of similar DoulaEntry objects
        """
        if len(self.entries) == 0:
            return []

        # Apply filters first
        mask = torch.ones(len(self.entries), dtype=torch.bool)

        if phase_filter:
            mask &= torch.tensor([e.phase == phase_filter for e in self.entries])
        if layer_filter is not None:
            mask &= torch.tensor([e.layer == layer_filter for e in self.entries])

        if not mask.any():
            return []

        # Get filtered embeddings
        filtered_indices = mask.nonzero(as_tuple=True)[0]
        filtered_embeddings = self.embeddings[filtered_indices]

        # Cosine similarity
        query_norm = F.normalize(query_embedding.unsqueeze(0), dim=1)
        embeddings_norm = F.normalize(filtered_embeddings, dim=1)
        similarities = (query_norm @ embeddings_norm.T).squeeze(0)

        # Top-k
        k = min(k, len(filtered_indices))
        top_k = similarities.topk(k)

        results = [self.entries[filtered_indices[i]] for i in top_k.indices]

        # Log query (glass box)
        self.query_log.append({
            'timestamp': datetime.now().isoformat(),
            'query_embedding': query_embedding.tolist(),
            'k': k,
            'phase_filter': phase_filter,
            'layer_filter': layer_filter,
            'result_ids': [e.id for e in results],
            'similarities': top_k.values.tolist(),
        })

        return results

    def search_by_signature(
        self,
        signature: DynamicsSignature,
        k: int = 5,
        **filters,
    ) -> List[DoulaEntry]:
        """
        Search using a DynamicsSignature directly.

        Args:
            signature: Query signature
            k: Number of results
            **filters: Passed to search()

        Returns:
            List of similar entries
        """
        embedding = signature.to_embedding(self.embedding_dim)
        return self.search(embedding, k=k, **filters)

    # =========================================================================
    # GLASS BOX: Full Transparency Methods
    # =========================================================================

    def get_all_entries(self) -> List[DoulaEntry]:
        """Get all stored observations."""
        return self.entries

    def get_query_log(self) -> List[Dict]:
        """Get log of all queries made."""
        return self.query_log

    def get_entry_by_id(self, entry_id: str) -> Optional[DoulaEntry]:
        """Get specific entry by ID."""
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    def get_entries_by_phase(self, phase: str) -> List[DoulaEntry]:
        """Get all entries from a specific phase."""
        return [e for e in self.entries if e.phase == phase]

    def get_entries_by_layer(self, layer: int) -> List[DoulaEntry]:
        """Get all entries from a specific layer."""
        return [e for e in self.entries if e.layer == layer]

    def get_entries_by_step_range(
        self,
        start_step: int,
        end_step: int,
    ) -> List[DoulaEntry]:
        """Get entries within a step range."""
        return [e for e in self.entries if start_step <= e.step <= end_step]

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def analyze_phases(self) -> Dict[str, int]:
        """Count entries per phase."""
        from collections import Counter
        return dict(Counter(e.phase for e in self.entries))

    def get_emergence_moments(self) -> List[DoulaEntry]:
        """Get entries where new shapes emerged."""
        return [e for e in self.entries if e.signature.new_shape_activated]

    def get_cluster_formations(self) -> List[DoulaEntry]:
        """Get entries where clusters formed."""
        return [e for e in self.entries if e.signature.cluster_formed]

    def get_crystallization_trajectory(self, layer: int) -> List[Tuple[int, float]]:
        """
        Get stability over time for a layer.

        Returns:
            List of (step, stability) tuples
        """
        layer_entries = [e for e in self.entries if e.layer == layer]
        layer_entries.sort(key=lambda e: e.step)
        return [(e.step, e.signature.routing_stability) for e in layer_entries]

    def get_entropy_trajectory(self, layer: int) -> List[Tuple[int, float]]:
        """
        Get entropy over time for a layer.

        Returns:
            List of (step, entropy) tuples
        """
        layer_entries = [e for e in self.entries if e.layer == layer]
        layer_entries.sort(key=lambda e: e.step)
        return [(e.step, e.signature.shape_entropy) for e in layer_entries]

    def get_loss_trajectory(self) -> List[Tuple[int, float]]:
        """
        Get loss over time.

        Returns:
            List of (step, loss) tuples
        """
        # Use entries from layer 0 to avoid duplicates
        layer0_entries = [e for e in self.entries if e.layer == 0]
        layer0_entries.sort(key=lambda e: e.step)
        return [(e.step, e.loss_at_step) for e in layer0_entries]

    def summarize(self) -> Dict:
        """
        Get summary statistics of the VDB.
        """
        if not self.entries:
            return {'empty': True}

        phases = self.analyze_phases()
        layers = set(e.layer for e in self.entries)

        return {
            'total_entries': len(self.entries),
            'phases': phases,
            'num_layers': len(layers),
            'step_range': (self.entries[0].step, self.entries[-1].step),
            'emergence_count': len(self.get_emergence_moments()),
            'cluster_count': len(self.get_cluster_formations()),
            'query_count': len(self.query_log),
            'domain': self.domain,
            'created_at': self.created_at,
        }

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self):
        """Save VDB to disk."""
        if not self.path:
            raise ValueError("No path set for VDB persistence")

        self.path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'created_at': self.created_at,
            'domain': self.domain,
            'embedding_dim': self.embedding_dim,
            'entries': [e.to_dict() for e in self.entries],
            'query_log': self.query_log,
        }

        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load VDB from disk."""
        if not self.path or not self.path.exists():
            return

        with open(self.path, 'r') as f:
            data = json.load(f)

        self.created_at = data.get('created_at', self.created_at)
        self.domain = data.get('domain')
        self.embedding_dim = data.get('embedding_dim', self.embedding_dim)
        self.query_log = data.get('query_log', [])

        self.entries = [DoulaEntry.from_dict(d) for d in data.get('entries', [])]

        # Rebuild embeddings tensor
        if self.entries:
            self.embeddings = torch.stack([e.dynamics_embedding for e in self.entries])
        else:
            self.embeddings = torch.empty(0, self.embedding_dim)

    def set_domain(self, domain: str):
        """Set the domain/task name for this VDB."""
        self.domain = domain

    def clear(self):
        """Clear all entries (but keep metadata)."""
        self.entries = []
        self.embeddings = torch.empty(0, self.embedding_dim)
        self.query_log = []
