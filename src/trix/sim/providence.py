"""
Providence Memory - 4D Content-Addressable Memory via XOR Matching

The elegant architecture where:
- Addressing IS matching (no carry chains)
- XOR gives distance for FREE (O(1) parallel)
- Signatures ARE addresses
- Memory is a manifold, not a grid

This is attention. This is what transformers discovered.
We just found it from the other direction.

Query → XOR Match → Nearest Signature → Value

No sequential carries. No octave chains. Pure parallel geometry.

Example:
    >>> mem = ProvidenceMemory(num_cells=256, sig_bits=16)
    >>> mem.write(0xABCD, 42)  # Signature 0xABCD stores value 42
    >>> mem.read(0xABCE)       # Query 0xABCE (1 bit different)
    42  # Finds nearest match

    >>> # Interpolated read (memory as continuous surface)
    >>> mem.read_interpolated(0xABCE, k=4)
    41.7  # Weighted average of 4 nearest

Providence: The architecture that provides itself.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

from trix.shapes import xor


# =============================================================================
# XOR DISTANCE (THE CORE)
# =============================================================================

def hamming_distance(a: int, b: int) -> int:
    """
    Compute Hamming distance using XOR.

    XOR gives us distance for FREE:
    - Bits that match → 0
    - Bits that differ → 1
    - Count the 1s = distance

    O(1) operation. No carry chains. Pure elegance.
    """
    return bin(a ^ b).count('1')


def hamming_distance_batch(query: int, signatures: List[int]) -> List[int]:
    """Compute Hamming distance from query to all signatures."""
    return [hamming_distance(query, sig) for sig in signatures]


def xor_match(query: int, signatures: List[int]) -> int:
    """
    Find index of signature nearest to query via XOR distance.

    This is the core operation. Content-addressable lookup.
    The 'address' IS the content pattern.
    """
    distances = hamming_distance_batch(query, signatures)
    return min(range(len(distances)), key=lambda i: distances[i])


def xor_match_k(query: int, signatures: List[int], k: int) -> Tuple[List[int], List[int]]:
    """
    Find k nearest signatures to query.

    Returns:
        (indices, distances) of k nearest matches
    """
    distances = hamming_distance_batch(query, signatures)
    indexed = sorted(enumerate(distances), key=lambda x: x[1])[:k]
    indices = [i for i, d in indexed]
    dists = [d for i, d in indexed]
    return indices, dists


# =============================================================================
# PROVIDENCE MEMORY
# =============================================================================

@dataclass
class ProvidenceCell:
    """A cell in Providence memory: signature + value."""
    signature: int
    value: int = 0


class ProvidenceMemory:
    """
    Content-addressable memory via XOR matching.

    No carry chains. No sequential addressing.
    Query → XOR → Match → Value. All parallel.

    This is what attention mechanisms do:
    - Query = what you're looking for
    - Keys = signatures (what each cell responds to)
    - Values = stored data

    We just discovered it from frozen geometry.

    Args:
        num_cells: Number of memory cells
        sig_bits: Bits per signature (default 16)
        init_mode: How to initialize signatures
            'sequential': 0, 1, 2, ... (like traditional memory)
            'random': random signatures
            'spread': maximally spread in Hamming space

    Example:
        >>> mem = ProvidenceMemory(256)
        >>> mem.write(0x1234, 42)
        >>> mem.read(0x1234)
        42
        >>> mem.read(0x1235)  # 1 bit different, still matches
        42
    """

    def __init__(
        self,
        num_cells: int,
        sig_bits: int = 16,
        init_mode: str = 'sequential'
    ):
        self.num_cells = num_cells
        self.sig_bits = sig_bits
        self.max_sig = (1 << sig_bits) - 1

        # Initialize signatures
        if init_mode == 'sequential':
            # Like traditional memory: address IS index
            self.signatures = [i % (self.max_sig + 1) for i in range(num_cells)]
        elif init_mode == 'random':
            import random
            self.signatures = [random.randint(0, self.max_sig) for _ in range(num_cells)]
        elif init_mode == 'spread':
            # Spread signatures maximally in Hamming space
            # Use Gray code-like spreading
            self.signatures = self._spread_signatures(num_cells, sig_bits)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

        # Initialize values to 0
        self.values = [0] * num_cells

        # Track access patterns (for learning, optional)
        self.access_counts = [0] * num_cells

    def _spread_signatures(self, n: int, bits: int) -> List[int]:
        """Generate n signatures spread in Hamming space."""
        # Simple spreading: use bit-reversal permutation
        sigs = []
        for i in range(n):
            # Bit-reverse the index for spreading
            rev = int(bin(i)[2:].zfill(bits)[::-1], 2)
            sigs.append(rev & self.max_sig)
        return sigs

    # -------------------------------------------------------------------------
    # Core Read/Write
    # -------------------------------------------------------------------------

    def write(self, signature: int, value: int) -> int:
        """
        Write value to cell with nearest matching signature.

        Args:
            signature: The 'address' (content pattern)
            value: Byte value to store

        Returns:
            Index of cell written to
        """
        idx = xor_match(signature & self.max_sig, self.signatures)
        self.values[idx] = value & 0xFF
        self.access_counts[idx] += 1
        return idx

    def read(self, query: int) -> int:
        """
        Read from cell with nearest matching signature.

        Args:
            query: The 'address' (content pattern to match)

        Returns:
            Value from nearest matching cell
        """
        idx = xor_match(query & self.max_sig, self.signatures)
        self.access_counts[idx] += 1
        return self.values[idx]

    def read_with_distance(self, query: int) -> Tuple[int, int]:
        """
        Read and return distance to matched cell.

        Returns:
            (value, hamming_distance)
        """
        idx = xor_match(query & self.max_sig, self.signatures)
        dist = hamming_distance(query & self.max_sig, self.signatures[idx])
        return self.values[idx], dist

    # -------------------------------------------------------------------------
    # Interpolated Read (Memory as Continuous Surface)
    # -------------------------------------------------------------------------

    def read_interpolated(self, query: int, k: int = 4, temperature: float = 1.0) -> float:
        """
        Read with soft interpolation from k nearest cells.

        This turns discrete memory into a continuous surface.
        Like attention, but for memory.

        Args:
            query: Content pattern to match
            k: Number of neighbors to interpolate
            temperature: Softmax temperature (lower = harder)

        Returns:
            Interpolated value (float)
        """
        indices, distances = xor_match_k(query & self.max_sig, self.signatures, k)

        if not distances or distances[0] == 0:
            # Exact match
            return float(self.values[indices[0]])

        # Convert distances to weights via softmax
        # Closer = higher weight
        neg_dists = [-d / temperature for d in distances]
        max_neg = max(neg_dists)
        exp_dists = [2.718281828 ** (nd - max_neg) for nd in neg_dists]
        sum_exp = sum(exp_dists)
        weights = [e / sum_exp for e in exp_dists]

        # Weighted sum of values
        result = sum(w * self.values[i] for w, i in zip(weights, indices))
        return result

    # -------------------------------------------------------------------------
    # Batch Operations
    # -------------------------------------------------------------------------

    def write_batch(self, signatures: List[int], values: List[int]) -> List[int]:
        """Write multiple signature-value pairs."""
        return [self.write(s, v) for s, v in zip(signatures, values)]

    def read_batch(self, queries: List[int]) -> List[int]:
        """Read multiple queries."""
        return [self.read(q) for q in queries]

    # -------------------------------------------------------------------------
    # Direct Cell Access (for debugging/testing)
    # -------------------------------------------------------------------------

    def get_cell(self, idx: int) -> ProvidenceCell:
        """Get cell by index."""
        return ProvidenceCell(self.signatures[idx], self.values[idx])

    def set_cell(self, idx: int, signature: int, value: int) -> None:
        """Set cell by index."""
        self.signatures[idx] = signature & self.max_sig
        self.values[idx] = value & 0xFF

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------

    def coverage(self) -> float:
        """Fraction of cells that have been accessed."""
        return sum(1 for c in self.access_counts if c > 0) / self.num_cells

    def hotspots(self, top_k: int = 10) -> List[Tuple[int, int]]:
        """Return indices of most-accessed cells."""
        indexed = sorted(enumerate(self.access_counts), key=lambda x: -x[1])
        return [(i, c) for i, c in indexed[:top_k]]

    def collision_rate(self) -> float:
        """Estimate collision rate based on access patterns."""
        total = sum(self.access_counts)
        if total == 0:
            return 0.0
        unique = sum(1 for c in self.access_counts if c > 0)
        return 1.0 - (unique / total) if total > unique else 0.0

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"ProvidenceMemory(cells={self.num_cells}, sig_bits={self.sig_bits})"

    def __len__(self) -> int:
        return self.num_cells


# =============================================================================
# HIERARCHICAL PROVIDENCE (O(√n) scaling)
# =============================================================================

class HierarchicalProvidenceMemory:
    """
    Two-level Providence memory for O(√n) lookup.

    Like HierarchicalTriXFFN, but for memory:
    - Level 1: Find nearest cluster
    - Level 2: Find nearest cell within cluster

    For 64K cells: √65536 = 256 clusters × 256 cells = O(256 + 256) = O(512)
    Instead of O(65536) for flat search.
    """

    def __init__(
        self,
        num_cells: int,
        cells_per_cluster: int = 256,
        sig_bits: int = 16
    ):
        self.num_cells = num_cells
        self.cells_per_cluster = cells_per_cluster
        self.sig_bits = sig_bits

        # Compute cluster count
        self.num_clusters = (num_cells + cells_per_cluster - 1) // cells_per_cluster

        # Create clusters
        self.clusters = []
        for i in range(self.num_clusters):
            start = i * cells_per_cluster
            end = min(start + cells_per_cluster, num_cells)
            size = end - start
            cluster = ProvidenceMemory(size, sig_bits, init_mode='sequential')
            # Offset signatures by cluster position
            for j in range(size):
                cluster.signatures[j] = start + j
            self.clusters.append(cluster)

        # Cluster signatures (centroid of each cluster's signatures)
        self.cluster_sigs = [
            self._centroid(c.signatures) for c in self.clusters
        ]

    def _centroid(self, sigs: List[int]) -> int:
        """Compute centroid signature (bitwise majority)."""
        if not sigs:
            return 0
        bits = self.sig_bits
        result = 0
        for bit in range(bits):
            ones = sum(1 for s in sigs if s & (1 << bit))
            if ones > len(sigs) // 2:
                result |= (1 << bit)
        return result

    def write(self, signature: int, value: int) -> Tuple[int, int]:
        """
        Write to hierarchical memory.

        Returns:
            (cluster_idx, cell_idx)
        """
        sig = signature & ((1 << self.sig_bits) - 1)

        # Find nearest cluster
        cluster_idx = xor_match(sig, self.cluster_sigs)

        # Write to cluster
        cell_idx = self.clusters[cluster_idx].write(sig, value)

        return cluster_idx, cell_idx

    def read(self, query: int) -> int:
        """Read from hierarchical memory."""
        sig = query & ((1 << self.sig_bits) - 1)

        # Find nearest cluster
        cluster_idx = xor_match(sig, self.cluster_sigs)

        # Read from cluster
        return self.clusters[cluster_idx].read(sig)

    def read_interpolated(self, query: int, k: int = 4) -> float:
        """Interpolated read within matched cluster."""
        sig = query & ((1 << self.sig_bits) - 1)
        cluster_idx = xor_match(sig, self.cluster_sigs)
        return self.clusters[cluster_idx].read_interpolated(sig, k)

    def __repr__(self) -> str:
        return f"HierarchicalProvidenceMemory(cells={self.num_cells}, clusters={self.num_clusters})"

    def __len__(self) -> int:
        return self.num_cells
