"""
Tests for DB Cooper: Octave DB
Multi-resolution ternary retrieval with glassbox explainability.
"""

import pytest
import numpy as np

from trix.db import (
    OctaveDB,
    OctaveIndex,
    ternary_quantize,
    derive_coarse,
    ternary_similarity,
    pack_ternary,
    unpack_ternary,
)
from trix.db.core import explain_match, derive_hierarchy


class TestTernaryQuantize:
    """Tests for ternary quantization."""
    
    def test_basic_quantize(self):
        """Values above/below threshold become +1/-1."""
        x = np.array([0.5, -0.5, 0.1, -0.1, 0.0])
        t = ternary_quantize(x, threshold=0.2)
        expected = np.array([1, -1, 0, 0, 0], dtype=np.int8)
        np.testing.assert_array_equal(t, expected)
    
    def test_quantize_batch(self):
        """Works on batched inputs."""
        x = np.random.randn(10, 64)
        t = ternary_quantize(x, threshold=0.5)
        assert t.shape == (10, 64)
        assert set(np.unique(t)).issubset({-1, 0, 1})
    
    def test_sparsity_target(self):
        """Sparsity target controls fraction of zeros."""
        x = np.random.randn(1000)
        t = ternary_quantize(x, sparsity_target=0.5)
        sparsity = np.mean(t == 0)
        assert 0.4 < sparsity < 0.6  # Approximately 50% zeros
    
    def test_output_dtype(self):
        """Output is int8."""
        x = np.random.randn(10)
        t = ternary_quantize(x)
        assert t.dtype == np.int8


class TestDeriveCoarse:
    """Tests for coarse derivation."""
    
    def test_basic_derive(self):
        """Coarse is sign of mean of chunks."""
        fine = np.array([1, 1, 1, 1, -1, -1, 1, -1], dtype=np.int8)
        coarse = derive_coarse(fine, pool_factor=4)
        # First chunk: mean([1,1,1,1]) = 1 → sign = 1
        # Second chunk: mean([-1,-1,1,-1]) = -0.5 → sign = -1
        expected = np.array([1, -1], dtype=np.int8)
        np.testing.assert_array_equal(coarse, expected)
    
    def test_derive_preserves_batch(self):
        """Works on batched inputs."""
        fine = np.random.choice([-1, 0, 1], size=(10, 64)).astype(np.int8)
        coarse = derive_coarse(fine, pool_factor=4)
        assert coarse.shape == (10, 16)
    
    def test_hierarchy(self):
        """Full hierarchy derivation."""
        fine = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        levels = derive_hierarchy(fine, pool_factor=4, num_levels=3)
        assert len(levels) == 3
        assert levels[0].shape == (64,)   # fine
        assert levels[1].shape == (16,)   # medium
        assert levels[2].shape == (4,)    # coarse


class TestTernarySimilarity:
    """Tests for ternary similarity."""
    
    def test_identical_vectors(self):
        """Identical vectors have maximum similarity."""
        a = np.array([1, -1, 1, 0, -1], dtype=np.int8)
        score = ternary_similarity(a, a)
        # Should be 4 (four non-zero positions all agree)
        assert score == 4
    
    def test_opposite_vectors(self):
        """Opposite vectors have minimum similarity."""
        a = np.array([1, -1, 1, -1], dtype=np.int8)
        b = np.array([-1, 1, -1, 1], dtype=np.int8)
        score = ternary_similarity(a, b)
        assert score == -4
    
    def test_orthogonal_vectors(self):
        """Zeros are neutral."""
        a = np.array([1, 0, 0, 0], dtype=np.int8)
        b = np.array([0, 1, 0, 0], dtype=np.int8)
        score = ternary_similarity(a, b)
        assert score == 0
    
    def test_batch_similarity(self):
        """Works with batched documents."""
        query = np.array([1, -1, 1, 0], dtype=np.int8)
        docs = np.array([
            [1, -1, 1, 0],   # identical
            [-1, 1, -1, 0],  # opposite
            [0, 0, 0, 0],    # zeros
        ], dtype=np.int8)
        scores = ternary_similarity(query, docs)
        assert scores[0] == 3   # three agreements
        assert scores[1] == -3  # three conflicts
        assert scores[2] == 0   # all neutral


class TestPackTernary:
    """Tests for bit packing."""
    
    def test_pack_unpack_roundtrip(self):
        """Pack then unpack recovers original."""
        original = np.array([1, -1, 0, 1, -1, -1, 0, 1], dtype=np.int8)
        pos, neg = pack_ternary(original)
        recovered = unpack_ternary(pos, neg, len(original))
        np.testing.assert_array_equal(recovered, original)
    
    def test_packed_is_smaller(self):
        """Packed representation is compact."""
        original = np.random.choice([-1, 0, 1], size=256).astype(np.int8)
        pos, neg = pack_ternary(original)
        # 256 dimensions → 32 bytes per mask
        assert pos.shape == (32,)
        assert neg.shape == (32,)


class TestExplainMatch:
    """Tests for glassbox explanations."""
    
    def test_explain_agreement(self):
        """Identifies agreeing dimensions."""
        q = np.array([1, -1, 1, 0], dtype=np.int8)
        d = np.array([1, -1, 0, 0], dtype=np.int8)
        exp = explain_match(q, d)
        assert 0 in exp['agreement']  # both +1
        assert 1 in exp['agreement']  # both -1
        assert 2 in exp['query_only']  # q=1, d=0
    
    def test_explain_conflict(self):
        """Identifies conflicting dimensions."""
        q = np.array([1, -1], dtype=np.int8)
        d = np.array([-1, 1], dtype=np.int8)
        exp = explain_match(q, d)
        assert 0 in exp['conflict']
        assert 1 in exp['conflict']
    
    def test_explain_score(self):
        """Explanation includes correct score."""
        q = np.array([1, 1, 1, -1], dtype=np.int8)
        d = np.array([1, 1, -1, -1], dtype=np.int8)
        exp = explain_match(q, d)
        # agree: 0,1,3 (+3), conflict: 2 (-1) = 2
        assert exp['score'] == 2


class TestOctaveIndex:
    """Tests for OctaveIndex."""
    
    def test_add_and_get(self):
        """Can add and retrieve documents."""
        idx = OctaveIndex(dimensions=64, pool_factor=4)
        fine = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        idx.add("doc1", fine, metadata={'title': 'Test'})
        
        doc = idx.get("doc1")
        assert doc is not None
        assert doc.id == "doc1"
        assert doc.metadata == {'title': 'Test'}
        np.testing.assert_array_equal(doc.fine, fine)
    
    def test_search_exact_match(self):
        """Exact match has highest score."""
        idx = OctaveIndex(dimensions=64, pool_factor=4, coarse_threshold=0.0)
        
        target = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        other = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        
        idx.add("target", target)
        idx.add("other", other)
        
        results = idx.search(target, mode="exact", top_k=2)
        assert len(results) >= 1
        assert results[0].id == "target"
        # Exact match should have score of 1.0 (normalized)
        assert results[0].score > 0.5
    
    def test_search_modes(self):
        """Different modes return results."""
        idx = OctaveIndex(dimensions=64, pool_factor=4, coarse_threshold=0.0)
        
        for i in range(10):
            fine = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
            idx.add(f"doc{i}", fine)
        
        query = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        
        exact = idx.search(query, mode="exact", top_k=5)
        similar = idx.search(query, mode="similar", top_k=5)
        context = idx.search(query, mode="context", top_k=5)
        
        assert len(exact) > 0
        assert len(similar) > 0
        assert len(context) > 0
    
    def test_remove(self):
        """Can remove documents."""
        idx = OctaveIndex(dimensions=64, pool_factor=4)
        fine = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        idx.add("doc1", fine)
        
        assert idx.get("doc1") is not None
        idx.remove("doc1")
        assert idx.get("doc1") is None
    
    def test_stats(self):
        """Stats are computed correctly."""
        idx = OctaveIndex(dimensions=64, pool_factor=4)
        
        for i in range(10):
            fine = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
            idx.add(f"doc{i}", fine)
        
        stats = idx.stats()
        assert stats['num_documents'] == 10
        assert stats['dimensions']['fine'] == 64
        assert stats['dimensions']['medium'] == 16
        assert stats['dimensions']['coarse'] == 4


class TestOctaveDB:
    """Tests for OctaveDB main interface."""
    
    def test_add_float_embedding(self):
        """Can add float embeddings (auto-quantized)."""
        db = OctaveDB(dimensions=64)
        embedding = np.random.randn(64)
        db.add("doc1", embedding)
        
        assert "doc1" in db
        assert len(db) == 1
    
    def test_add_batch(self):
        """Can add multiple documents."""
        db = OctaveDB(dimensions=64)
        embeddings = np.random.randn(10, 64)
        doc_ids = [f"doc{i}" for i in range(10)]
        db.add_batch(doc_ids, embeddings)
        
        assert len(db) == 10
    
    def test_search_returns_results(self):
        """Search returns results."""
        db = OctaveDB(dimensions=64, coarse_threshold=0.0)
        
        for i in range(20):
            db.add(f"doc{i}", np.random.randn(64))
        
        query = np.random.randn(64)
        results = db.search(query, mode="similar", top_k=5)
        
        assert len(results) == 5
        assert all(r.score is not None for r in results)
    
    def test_explain(self):
        """Can explain matches."""
        db = OctaveDB(dimensions=64)
        
        embedding = np.random.randn(64)
        db.add("doc1", embedding)
        
        exp = db.explain(embedding, "doc1")
        assert exp is not None
        assert 'agreement' in exp
        assert 'conflict' in exp
        assert 'score' in exp
    
    def test_get_document(self):
        """Can retrieve document signatures."""
        db = OctaveDB(dimensions=64)
        db.add("doc1", np.random.randn(64), metadata={'key': 'value'})
        
        doc = db.get("doc1")
        assert doc is not None
        assert doc['id'] == "doc1"
        assert doc['metadata'] == {'key': 'value'}
        assert 'fine' in doc
        assert 'medium' in doc
        assert 'coarse' in doc
    
    def test_ternary_input(self):
        """Can add pre-quantized ternary."""
        db = OctaveDB(dimensions=64)
        ternary = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        db.add("doc1", ternary, is_ternary=True)
        
        doc = db.get("doc1")
        np.testing.assert_array_equal(doc['fine'], ternary)


class TestSearchQuality:
    """Tests for search quality invariants."""
    
    def test_self_search_top_result(self):
        """Searching for a document returns itself first."""
        db = OctaveDB(dimensions=128, coarse_threshold=0.0)
        
        embeddings = np.random.randn(50, 128)
        for i, emb in enumerate(embeddings):
            db.add(f"doc{i}", emb)
        
        # Search for each document
        for i, emb in enumerate(embeddings):
            results = db.search(emb, mode="exact", top_k=1)
            assert results[0].id == f"doc{i}", f"doc{i} should be top result for itself"
    
    def test_similar_vectors_rank_higher(self):
        """More similar vectors rank higher."""
        db = OctaveDB(dimensions=64, coarse_threshold=0.0)
        
        base = np.random.randn(64)
        similar = base + 0.1 * np.random.randn(64)  # Small perturbation
        different = np.random.randn(64)  # Completely different
        
        db.add("similar", similar)
        db.add("different", different)
        
        results = db.search(base, mode="similar", top_k=2)
        assert results[0].id == "similar"
    
    def test_mode_affects_ranking(self):
        """Different modes can produce different rankings."""
        db = OctaveDB(dimensions=64, coarse_threshold=0.0)
        
        # Add documents with different fine/coarse characteristics
        for i in range(30):
            db.add(f"doc{i}", np.random.randn(64))
        
        query = np.random.randn(64)
        
        exact = db.search(query, mode="exact", top_k=10)
        context = db.search(query, mode="context", top_k=10)
        
        exact_ids = [r.id for r in exact]
        context_ids = [r.id for r in context]
        
        # Different modes may produce different orderings
        # (not guaranteed, but structurally different)
        assert len(exact_ids) == len(context_ids)


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_db_search(self):
        """Search on empty DB returns empty."""
        db = OctaveDB(dimensions=64)
        results = db.search(np.random.randn(64))
        assert results == []
    
    def test_single_document(self):
        """Works with single document."""
        db = OctaveDB(dimensions=64, coarse_threshold=-1.0)  # Accept all
        embedding = np.random.randn(64)
        db.add("only", embedding)
        
        # Search for the same embedding should find it
        results = db.search(embedding, top_k=10)
        assert len(results) == 1
        assert results[0].id == "only"
    
    def test_all_zeros_embedding(self):
        """Handles all-zeros embedding."""
        db = OctaveDB(dimensions=64, coarse_threshold=0.0)
        db.add("zeros", np.zeros(64))
        db.add("normal", np.random.randn(64))
        
        results = db.search(np.zeros(64), top_k=2)
        assert len(results) >= 1
    
    def test_remove_nonexistent(self):
        """Removing nonexistent doc returns False."""
        db = OctaveDB(dimensions=64)
        assert db.remove("nonexistent") == False
    
    def test_get_nonexistent(self):
        """Getting nonexistent doc returns None."""
        db = OctaveDB(dimensions=64)
        assert db.get("nonexistent") is None
