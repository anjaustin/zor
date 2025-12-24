"""
Term-Based Signature Derivation.

Signatures are ternary vectors {-1, 0, +1} that identify shapes.
They are computed deterministically from term structure, not sampled from behavior.

Key insight: The signature is a HASH of the shape's structure.
    - +1 = wants this input pattern
    - -1 = wants opposite pattern
    -  0 = doesn't care

Usage:
    from trix.forge.signature import signature_from_terms

    shape = generate_xor_shape(bits=8)
    sig = signature_from_terms(shape)  # Tensor of {-1, 0, +1}

    # Signatures enable content-addressable dispatch:
    # scores = input @ signatures.T
    # winner = argmax(scores)
"""

from typing import List, Dict, Optional, Tuple
import hashlib

import torch
from torch import Tensor

from .term import Term, BitTerms, ShapeTerms


# =============================================================================
# SIGNATURE PARAMETERS
# =============================================================================

DEFAULT_SIG_DIM = 64  # Default signature dimension


# =============================================================================
# TERM-BASED SIGNATURE DERIVATION
# =============================================================================

def signature_from_terms(shape: ShapeTerms, sig_dim: int = DEFAULT_SIG_DIM) -> Tensor:
    """
    Derive ternary signature from term structure.

    The signature encodes "what this shape cares about":
        - Which input bits it uses
        - How terms combine (coefficient patterns)
        - Structural properties of the polynomial

    Args:
        shape: ShapeTerms to derive signature from
        sig_dim: Dimension of signature vector

    Returns:
        Tensor of shape [sig_dim] with values in {-1, 0, +1}
    """
    # Build feature vector from term properties
    features = _extract_term_features(shape)

    # Project to signature dimension using deterministic random projection
    signature = _project_and_ternarize(features, shape.name, sig_dim)

    return signature


def _extract_term_features(shape: ShapeTerms) -> Tensor:
    """
    Extract feature vector from shape terms.

    Features include:
        - Term count per output bit
        - Coefficient distribution
        - Variable usage patterns
        - Polynomial degree statistics
    """
    features = []

    # Global statistics
    total_terms = shape.total_terms()
    features.append(total_terms / 100.0)  # Normalized term count

    # Per-bit statistics
    for bt in shape.bit_terms:
        # Term count for this bit
        features.append(len(bt.terms) / 10.0)

        # Coefficient statistics
        coeffs = [t.coefficient for t in bt.terms]
        if coeffs:
            features.append(sum(coeffs) / len(coeffs) / 2.0)  # Mean coefficient
            features.append(max(coeffs) / 2.0)  # Max coefficient
            features.append(min(coeffs) / 2.0)  # Min coefficient
        else:
            features.extend([0.0, 0.0, 0.0])

        # Variable usage
        all_vars = set()
        for term in bt.terms:
            all_vars.update(term.variables)

        # Encode which input bits are used
        for i in range(min(16, shape.input_bits)):
            features.append(1.0 if i in all_vars else 0.0)

        # Degree statistics
        degrees = [t.degree() for t in bt.terms]
        if degrees:
            features.append(sum(degrees) / len(degrees) / 3.0)  # Mean degree
            features.append(max(degrees) / 3.0)  # Max degree
        else:
            features.extend([0.0, 0.0])

    # Pad or truncate to fixed size
    target_size = 256
    if len(features) < target_size:
        features.extend([0.0] * (target_size - len(features)))
    else:
        features = features[:target_size]

    return torch.tensor(features, dtype=torch.float32)


def _project_and_ternarize(features: Tensor, name: str, sig_dim: int) -> Tensor:
    """
    Project feature vector to signature dimension and ternarize.

    Uses deterministic random projection based on shape name hash.
    """
    # Deterministic seed from shape name
    name_hash = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    generator = torch.Generator()
    generator.manual_seed(name_hash)

    # Random projection matrix
    projection = torch.randn(len(features), sig_dim, generator=generator)
    projection = projection / projection.norm(dim=0, keepdim=True)

    # Project
    projected = features @ projection

    # Ternarize with threshold
    threshold = projected.std() * 0.5
    signature = torch.zeros(sig_dim)
    signature[projected > threshold] = 1
    signature[projected < -threshold] = -1

    return signature


# =============================================================================
# SIGNATURE OPERATIONS
# =============================================================================

def signature_distance(a: Tensor, b: Tensor) -> float:
    """
    Compute Hamming-like distance between two signatures.

    Distance is number of positions where signs differ.
    """
    # Treat 0 as "don't care" - only count mismatches where both are non-zero
    mask = (a != 0) & (b != 0)
    mismatches = ((a != b) & mask).sum().item()
    return mismatches


def signature_similarity(a: Tensor, b: Tensor) -> float:
    """
    Compute similarity between two signatures.

    Returns value in [0, 1] where 1 is identical.
    """
    # Dot product normalized by number of non-zero positions
    n_active = ((a != 0) | (b != 0)).sum().item()
    if n_active == 0:
        return 1.0

    dot = (a * b).sum().item()
    return (dot / n_active + 1) / 2  # Map from [-1, 1] to [0, 1]


def match_signature(query: Tensor, signatures: Dict[str, Tensor]) -> Tuple[str, float]:
    """
    Find best matching signature from a dictionary.

    Args:
        query: Query signature
        signatures: Dict mapping names to signatures

    Returns:
        (best_name, similarity_score)
    """
    best_name = None
    best_similarity = -float('inf')

    for name, sig in signatures.items():
        sim = signature_similarity(query, sig)
        if sim > best_similarity:
            best_similarity = sim
            best_name = name

    return best_name, best_similarity


# =============================================================================
# SIGNATURE TABLE
# =============================================================================

class SignatureTable:
    """
    Table of signatures for content-addressable dispatch.

    Enables O(1) lookup of shapes by input pattern matching.
    """

    def __init__(self, sig_dim: int = DEFAULT_SIG_DIM):
        self.sig_dim = sig_dim
        self.signatures: Dict[str, Tensor] = {}
        self._signature_matrix: Optional[Tensor] = None
        self._names: List[str] = []

    def add(self, name: str, shape: ShapeTerms):
        """Add shape to table."""
        sig = signature_from_terms(shape, self.sig_dim)
        self.signatures[name] = sig
        self._signature_matrix = None  # Invalidate cache

    def add_signature(self, name: str, signature: Tensor):
        """Add pre-computed signature to table."""
        self.signatures[name] = signature
        self._signature_matrix = None  # Invalidate cache

    def _build_matrix(self):
        """Build signature matrix for vectorized lookup."""
        if self._signature_matrix is None:
            self._names = list(self.signatures.keys())
            sigs = [self.signatures[n] for n in self._names]
            self._signature_matrix = torch.stack(sigs)  # [n_shapes, sig_dim]

    def lookup(self, query: Tensor) -> Tuple[str, float]:
        """
        Find best matching shape for query signature.

        Args:
            query: Query signature tensor

        Returns:
            (shape_name, score)
        """
        self._build_matrix()

        # Compute scores: query @ signatures.T
        scores = query @ self._signature_matrix.T  # [n_shapes]

        # Find winner
        winner_idx = scores.argmax().item()
        winner_name = self._names[winner_idx]
        winner_score = scores[winner_idx].item()

        return winner_name, winner_score

    def lookup_batch(self, queries: Tensor) -> Tuple[List[str], Tensor]:
        """
        Batch lookup for multiple queries.

        Args:
            queries: [batch, sig_dim] tensor of query signatures

        Returns:
            (winner_names, scores)
        """
        self._build_matrix()

        # Compute all scores: queries @ signatures.T
        scores = queries @ self._signature_matrix.T  # [batch, n_shapes]

        # Find winners
        winner_indices = scores.argmax(dim=1)  # [batch]
        winner_names = [self._names[i.item()] for i in winner_indices]
        winner_scores = scores.gather(1, winner_indices.unsqueeze(1)).squeeze(1)

        return winner_names, winner_scores

    def __len__(self) -> int:
        return len(self.signatures)

    def __contains__(self, name: str) -> bool:
        return name in self.signatures


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def derive_all_signatures(
    shapes: Dict[str, ShapeTerms],
    sig_dim: int = DEFAULT_SIG_DIM
) -> Dict[str, Tensor]:
    """
    Derive signatures for all shapes in a dictionary.

    Args:
        shapes: Dict mapping names to ShapeTerms
        sig_dim: Signature dimension

    Returns:
        Dict mapping names to signature tensors
    """
    return {name: signature_from_terms(shape, sig_dim)
            for name, shape in shapes.items()}


def build_signature_table(
    shapes: Dict[str, ShapeTerms],
    sig_dim: int = DEFAULT_SIG_DIM
) -> SignatureTable:
    """
    Build signature table from shapes.

    Args:
        shapes: Dict mapping names to ShapeTerms
        sig_dim: Signature dimension

    Returns:
        SignatureTable ready for lookup
    """
    table = SignatureTable(sig_dim)
    for name, shape in shapes.items():
        table.add(name, shape)
    return table


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main function
    "signature_from_terms",

    # Signature operations
    "signature_distance",
    "signature_similarity",
    "match_signature",

    # Signature table
    "SignatureTable",
    "derive_all_signatures",
    "build_signature_table",

    # Parameters
    "DEFAULT_SIG_DIM",
]
