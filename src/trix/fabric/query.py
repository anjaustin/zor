"""
Query: Pattern matching on TriX Fabrics.

Queries allow finding processors by:
- Signature pattern (glob-like)
- Shape type
- Cluster path
- Connection patterns
"""

import fnmatch
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

from .fabric import Fabric
from .processor import FabricProcessor
from .cluster import Cluster


@dataclass
class QueryResult:
    """Result of a fabric query."""
    processors: List[FabricProcessor]
    count: int
    query: str


def query_fabric(
    fabric: Fabric,
    signature: Optional[str] = None,
    shape: Optional[str] = None,
    cluster: Optional[str] = None,
    name: Optional[str] = None,
    has_connections: Optional[bool] = None,
    predicate: Optional[Callable[[FabricProcessor], bool]] = None
) -> QueryResult:
    """
    Query a fabric for processors matching criteria.

    Args:
        fabric: The fabric to query
        signature: Glob pattern for signature (e.g., "/0/*", "*/0b1*")
        shape: Shape name to match
        cluster: Cluster path prefix
        name: Processor name pattern (glob)
        has_connections: If True, only processors with outputs
        predicate: Custom filter function

    Returns:
        QueryResult with matching processors
    """
    results = []
    query_parts = []

    for proc in fabric.iter_processors():
        match = True

        # Check signature pattern
        if signature is not None:
            if not _match_signature(proc.signature, signature):
                match = False

        # Check shape
        if shape is not None and match:
            if proc.shape != shape:
                match = False

        # Check cluster path
        if cluster is not None and match:
            if not proc.signature.startswith(cluster):
                match = False

        # Check name pattern
        if name is not None and match:
            if proc.name is None or not fnmatch.fnmatch(proc.name, name):
                match = False

        # Check connections
        if has_connections is not None and match:
            has_conn = len(proc.connections) > 0
            if has_connections != has_conn:
                match = False

        # Check custom predicate
        if predicate is not None and match:
            if not predicate(proc):
                match = False

        if match:
            results.append(proc)

    # Build query description
    if signature:
        query_parts.append(f"signature={signature}")
    if shape:
        query_parts.append(f"shape={shape}")
    if cluster:
        query_parts.append(f"cluster={cluster}")
    if name:
        query_parts.append(f"name={name}")

    query_str = " AND ".join(query_parts) if query_parts else "*"

    return QueryResult(
        processors=results,
        count=len(results),
        query=query_str
    )


def _match_signature(signature: str, pattern: str) -> bool:
    """
    Match a signature against a glob-like pattern.

    Supports:
    - * matches any single path component
    - ** matches any number of path components
    - ? matches any single character
    """
    # Convert pattern to regex
    regex_parts = []
    parts = pattern.split('/')

    for part in parts:
        if part == '**':
            regex_parts.append('.*')
        elif part == '*':
            regex_parts.append('[^/]+')
        else:
            # Escape special chars and convert ? and *
            escaped = re.escape(part)
            escaped = escaped.replace(r'\?', '.')
            escaped = escaped.replace(r'\*', '[^/]*')
            regex_parts.append(escaped)

    regex = '/'.join(regex_parts)
    regex = f'^{regex}$'

    return bool(re.match(regex, signature))


def find_by_shape(fabric: Fabric, shape: str) -> List[FabricProcessor]:
    """Find all processors with a given shape."""
    return query_fabric(fabric, shape=shape).processors


def find_by_cluster(fabric: Fabric, cluster_path: str) -> List[FabricProcessor]:
    """Find all processors in a cluster (and subclusters)."""
    return query_fabric(fabric, cluster=cluster_path).processors


def find_exports(fabric: Fabric) -> List[FabricProcessor]:
    """Find all export processors."""
    return [
        fabric.get_processor(sig)
        for sig in fabric.exports
        if fabric.get_processor(sig) is not None
    ]


def find_entry_points(fabric: Fabric) -> List[FabricProcessor]:
    """Find all entry point processors."""
    return [
        fabric.get_processor(sig)
        for sig in fabric.entry_points
        if fabric.get_processor(sig) is not None
    ]


def find_connected_to(fabric: Fabric, signature: str) -> List[FabricProcessor]:
    """Find all processors that receive input from the given processor."""
    source = fabric.get_processor(signature)
    if source is None:
        return []

    targets = []
    for conn in source.connections:
        target = fabric.get_processor(conn.target_signature)
        if target:
            targets.append(target)

    return targets


def find_inputs_of(fabric: Fabric, signature: str) -> List[FabricProcessor]:
    """Find all processors that provide input to the given processor."""
    target = fabric.get_processor(signature)
    if target is None:
        return []

    sources = []
    for input_name, source_ref in target.input_map.items():
        # Parse source_ref to get signature
        if '.' in source_ref:
            source_sig = source_ref.rsplit('.', 1)[0]
        else:
            source_sig = source_ref

        source = fabric.get_processor(source_sig)
        if source:
            sources.append(source)

    return sources


def trace_path(fabric: Fabric, start_sig: str, end_sig: str) -> Optional[List[str]]:
    """
    Find a path from start to end processor via connections.

    Returns list of signatures in the path, or None if no path exists.
    """
    visited: Set[str] = set()
    queue: List[List[str]] = [[start_sig]]

    while queue:
        path = queue.pop(0)
        current = path[-1]

        if current == end_sig:
            return path

        if current in visited:
            continue
        visited.add(current)

        proc = fabric.get_processor(current)
        if proc is None:
            continue

        for conn in proc.connections:
            if conn.target_signature not in visited:
                new_path = path + [conn.target_signature]
                queue.append(new_path)

    return None


def fabric_stats(fabric: Fabric) -> Dict[str, Any]:
    """Get statistics about a fabric."""
    processors = list(fabric.iter_processors())

    # Count shapes
    shape_counts: Dict[str, int] = {}
    for proc in processors:
        shape_counts[proc.shape] = shape_counts.get(proc.shape, 0) + 1

    # Count clusters
    clusters: Set[str] = set()
    for proc in processors:
        clusters.add(proc.cluster_path)

    # Count connections
    total_connections = sum(len(proc.connections) for proc in processors)

    return {
        'name': fabric.name,
        'total_processors': len(processors),
        'total_clusters': len(clusters),
        'total_connections': total_connections,
        'shapes': shape_counts,
        'exports': len(fabric.exports),
        'entry_points': len(fabric.entry_points)
    }
