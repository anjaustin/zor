"""
trix.sim - Machine simulation with frozen geometry.

Providence: The architecture that provides itself.

Two memory models:
    OctaveMemory     - Linear addressing, chained frozen adds (traditional)
    ProvidenceMemory - Content-addressable, XOR matching (elegant)

OctaveMemory:
    >>> from trix.sim import OctaveMemory
    >>> mem = OctaveMemory(octave=1)  # 16-bit, 64KB
    >>> mem.write(0x42, 0x0200)
    >>> mem.read(0x0200)
    66

ProvidenceMemory:
    >>> from trix.sim import ProvidenceMemory
    >>> mem = ProvidenceMemory(256, sig_bits=8)
    >>> mem.write(0xAB, 42)  # Signature 0xAB stores 42
    >>> mem.read(0xAB)       # Query 0xAB → finds match
    42
    >>> mem.read(0xAC)       # Query 0xAC → finds nearest (1 bit off)
    42

The elegance:
    Traditional ADD: O(n) carry chain
    Providence XOR:  O(1) parallel matching

This is attention. Query → Key match → Value retrieval.
We found it from frozen geometry.
"""

from .memory import OctaveMemory, add_octave, int_to_bytes, bytes_to_int
from .providence import (
    ProvidenceMemory,
    HierarchicalProvidenceMemory,
    hamming_distance,
    xor_match,
    xor_match_k,
)

__all__ = [
    # Octave (linear, carry chains)
    'OctaveMemory',
    'add_octave',
    'int_to_bytes',
    'bytes_to_int',

    # Providence (content-addressable, XOR matching)
    'ProvidenceMemory',
    'HierarchicalProvidenceMemory',
    'hamming_distance',
    'xor_match',
    'xor_match_k',
]
