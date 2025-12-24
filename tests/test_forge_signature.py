"""
Tests for Signature Derivation.

Tests cover:
    - signature_from_terms: Deriving signatures from ShapeTerms
    - SignatureTable: Content-addressable signature lookup
    - Signature operations: distance, similarity, matching
"""

import pytest
import torch

from trix.forge import (
    signature_from_terms,
    signature_distance,
    signature_similarity,
    match_signature,
    SignatureTable,
    derive_all_signatures,
    build_signature_table,
    DEFAULT_SIG_DIM,
    generate_xor_shape,
    generate_and_shape,
    generate_or_shape,
    generate_all_shapes,
)


# =============================================================================
# SIGNATURE FROM TERMS TESTS
# =============================================================================

class TestSignatureFromTerms:
    """Tests for signature_from_terms function."""

    def test_signature_returns_tensor(self):
        """Test that signature_from_terms returns a tensor."""
        shape = generate_xor_shape(bits=8)
        sig = signature_from_terms(shape)

        assert isinstance(sig, torch.Tensor)

    def test_signature_default_dimension(self):
        """Test that signature has default dimension."""
        shape = generate_xor_shape(bits=8)
        sig = signature_from_terms(shape)

        assert sig.shape == (DEFAULT_SIG_DIM,)

    def test_signature_custom_dimension(self):
        """Test signature with custom dimension."""
        shape = generate_xor_shape(bits=8)
        sig = signature_from_terms(shape, sig_dim=128)

        assert sig.shape == (128,)

    def test_signature_ternary_values(self):
        """Test that signature contains only {-1, 0, +1}."""
        shape = generate_xor_shape(bits=8)
        sig = signature_from_terms(shape)

        unique_values = torch.unique(sig)
        for v in unique_values:
            assert v.item() in {-1, 0, 1}

    def test_signature_deterministic(self):
        """Test that signature is deterministic."""
        shape = generate_xor_shape(bits=8)

        sig1 = signature_from_terms(shape)
        sig2 = signature_from_terms(shape)

        assert torch.equal(sig1, sig2)

    def test_different_shapes_different_signatures(self):
        """Test that different shapes have different signatures."""
        xor_shape = generate_xor_shape(bits=8)
        and_shape = generate_and_shape(bits=8)

        xor_sig = signature_from_terms(xor_shape)
        and_sig = signature_from_terms(and_shape)

        # Signatures should be different
        assert not torch.equal(xor_sig, and_sig)

    def test_signature_varies_with_bits(self):
        """Test that signature varies with bit width."""
        shape_8 = generate_xor_shape(bits=8)
        shape_16 = generate_xor_shape(bits=16)

        sig_8 = signature_from_terms(shape_8)
        sig_16 = signature_from_terms(shape_16)

        # Different bit widths should give different signatures
        assert not torch.equal(sig_8, sig_16)


# =============================================================================
# SIGNATURE OPERATIONS TESTS
# =============================================================================

class TestSignatureOperations:
    """Tests for signature operations."""

    def test_signature_distance_identical(self):
        """Test distance between identical signatures."""
        shape = generate_xor_shape(bits=8)
        sig = signature_from_terms(shape)

        distance = signature_distance(sig, sig)
        assert distance == 0

    def test_signature_distance_different(self):
        """Test distance between different signatures."""
        xor_sig = signature_from_terms(generate_xor_shape(bits=8))
        and_sig = signature_from_terms(generate_and_shape(bits=8))

        distance = signature_distance(xor_sig, and_sig)
        assert distance > 0

    def test_signature_similarity_identical(self):
        """Test similarity between identical signatures."""
        shape = generate_xor_shape(bits=8)
        sig = signature_from_terms(shape)

        similarity = signature_similarity(sig, sig)
        assert similarity == 1.0

    def test_signature_similarity_range(self):
        """Test that similarity is in [0, 1]."""
        xor_sig = signature_from_terms(generate_xor_shape(bits=8))
        and_sig = signature_from_terms(generate_and_shape(bits=8))

        similarity = signature_similarity(xor_sig, and_sig)
        assert 0 <= similarity <= 1

    def test_match_signature(self):
        """Test matching a signature against a dictionary."""
        xor_shape = generate_xor_shape(bits=8)
        and_shape = generate_and_shape(bits=8)
        or_shape = generate_or_shape(bits=8)

        signatures = {
            "xor": signature_from_terms(xor_shape),
            "and": signature_from_terms(and_shape),
            "or": signature_from_terms(or_shape),
        }

        # Query with XOR signature should match XOR
        xor_query = signature_from_terms(xor_shape)
        best_name, score = match_signature(xor_query, signatures)

        assert best_name == "xor"
        assert score == 1.0  # Exact match


# =============================================================================
# SIGNATURE TABLE TESTS
# =============================================================================

class TestSignatureTable:
    """Tests for SignatureTable class."""

    def test_create_empty_table(self):
        """Test creating empty signature table."""
        table = SignatureTable()
        assert len(table) == 0

    def test_add_shape(self):
        """Test adding shape to table."""
        table = SignatureTable()
        shape = generate_xor_shape(bits=8)
        table.add("xor", shape)

        assert len(table) == 1
        assert "xor" in table

    def test_add_multiple_shapes(self):
        """Test adding multiple shapes."""
        table = SignatureTable()
        table.add("xor", generate_xor_shape(bits=8))
        table.add("and", generate_and_shape(bits=8))
        table.add("or", generate_or_shape(bits=8))

        assert len(table) == 3

    def test_lookup(self):
        """Test signature lookup."""
        table = SignatureTable()
        table.add("xor", generate_xor_shape(bits=8))
        table.add("and", generate_and_shape(bits=8))
        table.add("or", generate_or_shape(bits=8))

        # Query with XOR signature
        xor_sig = signature_from_terms(generate_xor_shape(bits=8))
        name, score = table.lookup(xor_sig)

        assert name == "xor"

    def test_lookup_batch(self):
        """Test batch signature lookup."""
        table = SignatureTable()
        table.add("xor", generate_xor_shape(bits=8))
        table.add("and", generate_and_shape(bits=8))

        # Query with both signatures
        queries = torch.stack([
            signature_from_terms(generate_xor_shape(bits=8)),
            signature_from_terms(generate_and_shape(bits=8)),
        ])

        names, scores = table.lookup_batch(queries)

        assert len(names) == 2
        assert "xor" in names
        assert "and" in names

    def test_custom_sig_dim(self):
        """Test table with custom signature dimension."""
        table = SignatureTable(sig_dim=128)
        table.add("xor", generate_xor_shape(bits=8))

        assert table.sig_dim == 128


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_derive_all_signatures(self):
        """Test deriving signatures for all shapes."""
        shapes = generate_all_shapes(bits=8)
        signatures = derive_all_signatures(shapes)

        assert len(signatures) == len(shapes)
        for name in shapes:
            assert name in signatures
            assert isinstance(signatures[name], torch.Tensor)

    def test_build_signature_table(self):
        """Test building signature table from shapes."""
        shapes = generate_all_shapes(bits=8)
        table = build_signature_table(shapes)

        assert len(table) == len(shapes)

    def test_build_signature_table_custom_dim(self):
        """Test building table with custom dimension."""
        shapes = {"xor": generate_xor_shape(bits=8)}
        table = build_signature_table(shapes, sig_dim=32)

        assert table.sig_dim == 32


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSignatureIntegration:
    """Integration tests for signature system."""

    def test_full_pipeline(self):
        """Test full signature pipeline: derive -> table -> lookup."""
        # Create shapes
        shapes = {
            "xor": generate_xor_shape(bits=8),
            "and": generate_and_shape(bits=8),
            "or": generate_or_shape(bits=8),
        }

        # Build table
        table = build_signature_table(shapes)

        # Lookup each shape - should find itself
        for name, shape in shapes.items():
            query = signature_from_terms(shape)
            found_name, score = table.lookup(query)
            assert found_name == name, f"Expected {name}, got {found_name}"

    def test_signatures_distinguish_all_shapes(self):
        """Test that signatures distinguish all standard shapes."""
        shapes = generate_all_shapes(bits=8)
        signatures = derive_all_signatures(shapes)

        # Each shape should have a unique signature
        seen = set()
        for name, sig in signatures.items():
            sig_tuple = tuple(sig.tolist())
            assert sig_tuple not in seen, f"Duplicate signature for {name}"
            seen.add(sig_tuple)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
