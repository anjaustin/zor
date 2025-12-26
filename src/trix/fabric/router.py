"""
Router: Hierarchical routing for TriX Fabric.

Routing algorithm:
1. Parse target signature into path components
2. Navigate tree by prefix matching
3. Within cluster, route by Hamming distance

Address format: /cluster/subcluster/0b1010
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .cluster import Cluster
from .processor import FabricProcessor


@dataclass
class RouteResult:
    """Result of routing a message."""
    processor: Optional[FabricProcessor]
    cluster: Cluster
    path_taken: List[str]
    exact_match: bool


class Router:
    """
    Hierarchical router for TriX Fabric.

    Routes messages through tree of clusters,
    then uses Hamming distance within clusters.
    """

    def __init__(self, root: Cluster):
        """
        Initialize router with root cluster.

        Args:
            root: The root cluster of the fabric
        """
        self.root = root

    @staticmethod
    def parse_signature(signature: str) -> Tuple[List[str], str]:
        """
        Parse a hierarchical signature into path components and local signature.

        Args:
            signature: Full signature like "/0/1/0b1010"

        Returns:
            Tuple of (path_components, local_signature)
            e.g., (["/", "0", "1"], "0b1010")
        """
        # Handle root
        if signature == '/':
            return ['/'], ''

        # Split by /
        parts = signature.split('/')

        # First element is empty string before leading /
        if parts[0] == '':
            parts = parts[1:]

        if not parts:
            return ['/'], ''

        # Last component is local signature if it looks like binary
        last = parts[-1]
        if last.startswith('0b') or (last and all(c in '01' for c in last)):
            path_parts = ['/'] + parts[:-1]
            return path_parts, last
        else:
            # All parts are path
            return ['/'] + parts, ''

    def route(self, target_signature: str) -> RouteResult:
        """
        Route to a target signature.

        Args:
            target_signature: Full hierarchical signature

        Returns:
            RouteResult with found processor (or None) and routing info
        """
        path_parts, local_sig = self.parse_signature(target_signature)
        path_taken = []

        # Start at root
        current = self.root
        path_taken.append(current.path)

        # Navigate tree by path
        for i, part in enumerate(path_parts):
            if part == '/':
                continue  # Skip root marker

            child = current.get_child(part)
            if child is None:
                # Path not found, stay at current cluster
                break
            current = child
            path_taken.append(current.path)

        # Now we're at the target cluster, find processor
        if local_sig:
            processor = current.get_processor(local_sig)
            if processor:
                return RouteResult(
                    processor=processor,
                    cluster=current,
                    path_taken=path_taken,
                    exact_match=True
                )
            # Try Hamming routing
            nearest = current.find_nearest_processor(local_sig)
            return RouteResult(
                processor=nearest,
                cluster=current,
                path_taken=path_taken,
                exact_match=False
            )
        else:
            # No local signature, return first processor in cluster
            procs = list(current.iter_processors())
            return RouteResult(
                processor=procs[0] if procs else None,
                cluster=current,
                path_taken=path_taken,
                exact_match=False
            )

    def resolve_path(self, path: str) -> Optional[Cluster]:
        """
        Resolve a cluster path.

        Args:
            path: Cluster path like "/0/1"

        Returns:
            The cluster at that path, or None
        """
        if path == '/':
            return self.root

        parts = path.split('/')
        if parts[0] == '':
            parts = parts[1:]

        current = self.root
        for part in parts:
            if not part:
                continue
            child = current.get_child(part)
            if child is None:
                return None
            current = child

        return current

    def get_processor(self, signature: str) -> Optional[FabricProcessor]:
        """
        Get a processor by exact signature.

        Args:
            signature: Full hierarchical signature

        Returns:
            The processor, or None if not found
        """
        result = self.route(signature)
        if result.exact_match:
            return result.processor
        return None

    def find_all_processors(self, pattern: str = '*') -> List[FabricProcessor]:
        """
        Find all processors matching a pattern.

        Args:
            pattern: Glob-like pattern (currently just '*' for all)

        Returns:
            List of matching processors
        """
        if pattern == '*':
            return list(self.root.iter_all_processors())

        # TODO: Implement more complex pattern matching
        return list(self.root.iter_all_processors())
