"""
Cluster: A tree node in the TriX Fabric hierarchy.

Each cluster contains:
- Child clusters (for tree navigation)
- Processors (hypercube within this cluster)

Address format: /cluster/subcluster/0b1010
- Path navigates tree
- Final component is local signature (Hamming-routable)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from .processor import FabricProcessor


def hamming_distance(sig1: str, sig2: str) -> int:
    """
    Compute Hamming distance between two binary signatures.

    Signatures can be in format "0b1010" or just "1010".
    """
    # Strip 0b prefix if present
    s1 = sig1[2:] if sig1.startswith('0b') else sig1
    s2 = sig2[2:] if sig2.startswith('0b') else sig2

    # Pad to same length
    max_len = max(len(s1), len(s2))
    s1 = s1.zfill(max_len)
    s2 = s2.zfill(max_len)

    # Count differing bits
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


@dataclass
class Cluster:
    """
    A cluster in the fabric hierarchy.

    Attributes:
        path: The cluster's path in the tree (e.g., "/0/1")
        name: Human-readable name (optional)
        children: Child clusters
        processors: Processors in this cluster (local hypercube)
    """
    path: str
    name: Optional[str] = None
    children: Dict[str, 'Cluster'] = field(default_factory=dict)
    processors: Dict[str, FabricProcessor] = field(default_factory=dict)

    def __post_init__(self):
        # Normalize path
        if not self.path.startswith('/'):
            self.path = '/' + self.path
        if self.path != '/' and self.path.endswith('/'):
            self.path = self.path[:-1]

    def add_child(self, key: str, cluster: Optional['Cluster'] = None) -> 'Cluster':
        """Add a child cluster. Creates new cluster if not provided."""
        if cluster is None:
            child_path = f"{self.path}/{key}" if self.path != '/' else f"/{key}"
            cluster = Cluster(path=child_path)
        self.children[key] = cluster
        return cluster

    def get_child(self, key: str) -> Optional['Cluster']:
        """Get a child cluster by key."""
        return self.children.get(key)

    def add_processor(self, processor: FabricProcessor) -> None:
        """Add a processor to this cluster's hypercube."""
        local_sig = processor.local_signature
        self.processors[local_sig] = processor

    def get_processor(self, local_signature: str) -> Optional[FabricProcessor]:
        """Get a processor by its local signature."""
        return self.processors.get(local_signature)

    def find_nearest_processor(self, target_local_sig: str) -> Optional[FabricProcessor]:
        """
        Find the processor nearest to target signature by Hamming distance.

        This is the core of hypercube routing within a cluster.
        """
        if not self.processors:
            return None

        # Exact match first
        if target_local_sig in self.processors:
            return self.processors[target_local_sig]

        # Find nearest by Hamming distance
        best_proc = None
        best_dist = float('inf')

        for sig, proc in self.processors.items():
            dist = hamming_distance(sig, target_local_sig)
            if dist < best_dist:
                best_dist = dist
                best_proc = proc

        return best_proc

    def iter_processors(self) -> Iterator[FabricProcessor]:
        """Iterate over all processors in this cluster (not children)."""
        yield from self.processors.values()

    def iter_all_processors(self) -> Iterator[FabricProcessor]:
        """Iterate over all processors in this cluster and descendants."""
        yield from self.processors.values()
        for child in self.children.values():
            yield from child.iter_all_processors()

    @property
    def processor_count(self) -> int:
        """Count processors in this cluster only."""
        return len(self.processors)

    @property
    def total_processor_count(self) -> int:
        """Count all processors in this cluster and descendants."""
        count = len(self.processors)
        for child in self.children.values():
            count += child.total_processor_count
        return count

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'path': self.path,
            'name': self.name,
            'children': {k: v.to_dict() for k, v in self.children.items()},
            'processors': {k: v.to_dict() for k, v in self.processors.items()}
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Cluster':
        """Deserialize from dictionary."""
        cluster = cls(
            path=d['path'],
            name=d.get('name')
        )
        for key, child_dict in d.get('children', {}).items():
            cluster.children[key] = cls.from_dict(child_dict)
        for key, proc_dict in d.get('processors', {}).items():
            cluster.processors[key] = FabricProcessor.from_dict(proc_dict)
        return cluster

    def __repr__(self):
        return f"Cluster({self.path!r}, processors={self.processor_count}, children={len(self.children)})"
