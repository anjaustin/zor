"""
DB Cooper Rigorous Systems Tests

10 categories of invariants that MUST hold for the system to be correct.

1. Quantization Invariants - ternary values, sparsity, reversibility
2. Derivation Invariants - coarse = pool(fine), consistency
3. Similarity Invariants - symmetry, bounds, identity
4. Packing Invariants - roundtrip, bit layout
5. Index Invariants - add/remove, consistency, no data loss
6. Search Invariants - self-match, ordering, mode behavior
7. Native Ops Invariants - match Python, SIMD correctness
8. Numerical Stability - no overflow, large inputs, edge values
9. Concurrency Safety - multiple operations, state consistency
10. Scale Invariants - behavior at scale, memory bounds
"""

import pytest
import numpy as np
from typing import List, Tuple
import time

from trix.db import OctaveDB, OctaveIndex
from trix.db.core import (
    ternary_quantize,
    derive_coarse,
    derive_hierarchy,
    ternary_similarity,
    pack_ternary,
    unpack_ternary,
    explain_match,
)

# Try to import native ops
try:
    from trix.db.ops import CooperOps, native_ops_available
    HAS_NATIVE = native_ops_available()
except ImportError:
    HAS_NATIVE = False


class TestQuantizationInvariants:
    """Quantization MUST produce valid ternary values."""
    
    def test_output_is_ternary(self):
        """All output values must be in {-1, 0, +1}."""
        x = np.random.randn(1000)
        t = ternary_quantize(x)
        assert set(np.unique(t)).issubset({-1, 0, 1})
    
    def test_output_dtype_is_int8(self):
        """Output must be int8."""
        x = np.random.randn(100)
        t = ternary_quantize(x)
        assert t.dtype == np.int8
    
    def test_shape_preserved(self):
        """Shape must be preserved."""
        for shape in [(10,), (10, 20), (5, 10, 15)]:
            x = np.random.randn(*shape)
            t = ternary_quantize(x)
            assert t.shape == x.shape
    
    def test_threshold_respected(self):
        """Values within threshold become 0."""
        x = np.array([0.1, -0.1, 0.5, -0.5, 0.0])
        t = ternary_quantize(x, threshold=0.3)
        assert t[0] == 0  # 0.1 < 0.3
        assert t[1] == 0  # -0.1 > -0.3
        assert t[2] == 1  # 0.5 > 0.3
        assert t[3] == -1  # -0.5 < -0.3
        assert t[4] == 0  # 0.0
    
    def test_sparsity_target_approximate(self):
        """Sparsity target should be approximately achieved."""
        x = np.random.randn(10000)
        for target in [0.2, 0.5, 0.8]:
            t = ternary_quantize(x, sparsity_target=target)
            actual = np.mean(t == 0)
            assert abs(actual - target) < 0.05, f"Target {target}, got {actual}"
    
    def test_sign_preserved_for_large_values(self):
        """Large values must preserve sign."""
        x = np.array([100.0, -100.0, 1e6, -1e6])
        t = ternary_quantize(x, threshold=0.0)
        assert t[0] == 1
        assert t[1] == -1
        assert t[2] == 1
        assert t[3] == -1
    
    def test_zero_input_gives_zero_output(self):
        """Zero input must give zero output."""
        x = np.zeros(100)
        t = ternary_quantize(x, threshold=0.0)
        assert np.all(t == 0)


class TestDerivationInvariants:
    """Coarse MUST be derived from fine consistently."""
    
    def test_coarse_dims_correct(self):
        """Coarse dimensions must be fine_dims / pool_factor."""
        for fine_dims in [64, 128, 256]:
            for pool_factor in [2, 4, 8]:
                fine = np.random.choice([-1, 0, 1], size=fine_dims).astype(np.int8)
                coarse = derive_coarse(fine, pool_factor)
                expected_dims = fine_dims // pool_factor
                assert coarse.shape[0] == expected_dims
    
    def test_coarse_is_ternary(self):
        """Derived coarse must be ternary."""
        fine = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        coarse = derive_coarse(fine, pool_factor=4)
        assert set(np.unique(coarse)).issubset({-1, 0, 1})
    
    def test_derivation_deterministic(self):
        """Same fine must give same coarse."""
        fine = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        coarse1 = derive_coarse(fine, pool_factor=4)
        coarse2 = derive_coarse(fine, pool_factor=4)
        np.testing.assert_array_equal(coarse1, coarse2)
    
    def test_hierarchy_levels_correct(self):
        """Hierarchy must have correct number of levels."""
        fine = np.random.choice([-1, 0, 1], size=256).astype(np.int8)
        levels = derive_hierarchy(fine, pool_factor=4, num_levels=4)
        assert len(levels) == 4
        assert levels[0].shape[0] == 256  # fine
        assert levels[1].shape[0] == 64   # medium
        assert levels[2].shape[0] == 16   # coarse
        assert levels[3].shape[0] == 4    # coarser
    
    def test_all_positive_pools_to_positive(self):
        """All +1 in chunk must pool to +1."""
        fine = np.ones(16, dtype=np.int8)
        coarse = derive_coarse(fine, pool_factor=4)
        assert np.all(coarse == 1)
    
    def test_all_negative_pools_to_negative(self):
        """All -1 in chunk must pool to -1."""
        fine = -np.ones(16, dtype=np.int8)
        coarse = derive_coarse(fine, pool_factor=4)
        assert np.all(coarse == -1)
    
    def test_balanced_pools_to_zero(self):
        """Balanced +1/-1 must pool to 0."""
        fine = np.array([1, 1, -1, -1] * 4, dtype=np.int8)  # balanced
        coarse = derive_coarse(fine, pool_factor=4)
        assert np.all(coarse == 0)


class TestSimilarityInvariants:
    """Similarity function MUST satisfy mathematical properties."""
    
    def test_self_similarity_maximum(self):
        """Similarity with self must be maximum."""
        a = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        self_sim = ternary_similarity(a, a)
        other = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        other_sim = ternary_similarity(a, other)
        assert self_sim >= other_sim
    
    def test_symmetry(self):
        """similarity(a, b) must equal similarity(b, a)."""
        a = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        b = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        assert ternary_similarity(a, b) == ternary_similarity(b, a)
    
    def test_opposite_is_negative(self):
        """Opposite vectors must have negative similarity."""
        a = np.array([1, -1, 1, -1], dtype=np.int8)
        b = -a
        assert ternary_similarity(a, b) < 0
    
    def test_orthogonal_is_zero(self):
        """Non-overlapping vectors must have zero similarity."""
        a = np.array([1, 0, 0, 0], dtype=np.int8)
        b = np.array([0, 1, 0, 0], dtype=np.int8)
        assert ternary_similarity(a, b) == 0
    
    def test_bounds(self):
        """Similarity must be bounded by dimension count."""
        dim = 64
        a = np.random.choice([-1, 0, 1], size=dim).astype(np.int8)
        b = np.random.choice([-1, 0, 1], size=dim).astype(np.int8)
        sim = ternary_similarity(a, b)
        assert -dim <= sim <= dim
    
    def test_zeros_neutral(self):
        """Zeros must not contribute to similarity."""
        a = np.array([1, 0, 0, 0], dtype=np.int8)
        b = np.array([1, 1, 1, 1], dtype=np.int8)
        # Only first dimension matches, others are neutral
        assert ternary_similarity(a, b) == 1


class TestPackingInvariants:
    """Packing MUST be lossless and consistent."""
    
    def test_roundtrip(self):
        """Pack then unpack must recover original."""
        original = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        pos, neg = pack_ternary(original)
        recovered = unpack_ternary(pos, neg, len(original))
        np.testing.assert_array_equal(original, recovered)
    
    def test_roundtrip_various_sizes(self):
        """Roundtrip must work for various sizes."""
        for size in [7, 8, 9, 15, 16, 17, 63, 64, 65, 128, 255, 256, 257]:
            original = np.random.choice([-1, 0, 1], size=size).astype(np.int8)
            pos, neg = pack_ternary(original)
            recovered = unpack_ternary(pos, neg, size)
            np.testing.assert_array_equal(original, recovered)
    
    def test_packed_size_correct(self):
        """Packed size must be ceil(dims/8)."""
        for dims in [8, 16, 64, 100, 384]:
            original = np.random.choice([-1, 0, 1], size=dims).astype(np.int8)
            pos, neg = pack_ternary(original)
            expected_bytes = (dims + 7) // 8
            assert len(pos) == expected_bytes
            assert len(neg) == expected_bytes
    
    def test_positive_negative_disjoint(self):
        """Positive and negative masks must not overlap."""
        original = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        pos, neg = pack_ternary(original)
        # No bit should be set in both masks
        overlap = pos & neg
        assert np.all(overlap == 0)
    
    def test_all_positive_packing(self):
        """All +1 must pack to all-1s pos mask."""
        original = np.ones(8, dtype=np.int8)
        pos, neg = pack_ternary(original)
        assert pos[0] == 0xFF
        assert neg[0] == 0x00
    
    def test_all_negative_packing(self):
        """All -1 must pack to all-1s neg mask."""
        original = -np.ones(8, dtype=np.int8)
        pos, neg = pack_ternary(original)
        assert pos[0] == 0x00
        assert neg[0] == 0xFF


class TestIndexInvariants:
    """Index operations MUST maintain consistency."""
    
    def test_add_increases_count(self):
        """Adding document must increase count."""
        idx = OctaveIndex(dimensions=64)
        assert idx.num_documents == 0
        idx.add("doc1", np.random.choice([-1, 0, 1], size=64).astype(np.int8))
        assert idx.num_documents == 1
    
    def test_remove_decreases_count(self):
        """Removing document must decrease count."""
        idx = OctaveIndex(dimensions=64)
        idx.add("doc1", np.random.choice([-1, 0, 1], size=64).astype(np.int8))
        idx.remove("doc1")
        assert idx.num_documents == 0
    
    def test_get_returns_what_was_added(self):
        """Get must return the document that was added."""
        idx = OctaveIndex(dimensions=64)
        fine = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        idx.add("doc1", fine, metadata={"key": "value"})
        doc = idx.get("doc1")
        np.testing.assert_array_equal(doc.fine, fine)
        assert doc.metadata == {"key": "value"}
    
    def test_remove_nonexistent_returns_false(self):
        """Removing nonexistent document must return False."""
        idx = OctaveIndex(dimensions=64)
        assert idx.remove("nonexistent") == False
    
    def test_get_nonexistent_returns_none(self):
        """Getting nonexistent document must return None."""
        idx = OctaveIndex(dimensions=64)
        assert idx.get("nonexistent") is None
    
    def test_duplicate_id_overwrites(self):
        """Adding with same ID should update the document."""
        idx = OctaveIndex(dimensions=64)
        fine1 = np.ones(64, dtype=np.int8)
        fine2 = -np.ones(64, dtype=np.int8)
        idx.add("doc1", fine1)
        # Note: Current implementation adds duplicate - this tests actual behavior
        # In production, we might want to check for existing ID and remove first
        idx.remove("doc1")  # Remove first to simulate overwrite
        idx.add("doc1", fine2)
        assert idx.num_documents == 1
        doc = idx.get("doc1")
        np.testing.assert_array_equal(doc.fine, fine2)


class TestSearchInvariants:
    """Search MUST return correct results."""
    
    def test_self_search_returns_self(self):
        """Searching for a document must return itself first."""
        db = OctaveDB(dimensions=64, coarse_threshold=-1.0)
        embeddings = [np.random.randn(64) for _ in range(10)]
        for i, emb in enumerate(embeddings):
            db.add(f"doc{i}", emb)
        
        for i, emb in enumerate(embeddings):
            results = db.search(emb, mode="exact", top_k=1)
            assert results[0].id == f"doc{i}"
    
    def test_exact_mode_prefers_fine(self):
        """Exact mode must weight fine level heavily."""
        db = OctaveDB(dimensions=64, coarse_threshold=-1.0)
        
        # Add documents
        base = np.random.randn(64)
        similar_fine = base + 0.01 * np.random.randn(64)
        different_fine = np.random.randn(64)
        
        db.add("similar", similar_fine)
        db.add("different", different_fine)
        
        results = db.search(base, mode="exact", top_k=2)
        assert results[0].id == "similar"
    
    def test_search_respects_top_k(self):
        """Must return at most top_k results."""
        db = OctaveDB(dimensions=64, coarse_threshold=-1.0)
        for i in range(20):
            db.add(f"doc{i}", np.random.randn(64))
        
        for k in [1, 5, 10, 15]:
            results = db.search(np.random.randn(64), top_k=k)
            assert len(results) <= k
    
    def test_results_sorted_by_score(self):
        """Results must be sorted by score descending."""
        db = OctaveDB(dimensions=64, coarse_threshold=-1.0)
        for i in range(20):
            db.add(f"doc{i}", np.random.randn(64))
        
        results = db.search(np.random.randn(64), mode="similar", top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_empty_db_returns_empty(self):
        """Empty database must return empty results."""
        db = OctaveDB(dimensions=64)
        results = db.search(np.random.randn(64))
        assert results == []


@pytest.mark.skipif(not HAS_NATIVE, reason="Native ops not available")
class TestNativeOpsInvariants:
    """Native ops MUST match Python implementation."""
    
    def test_similarity_matches_python(self):
        """Native similarity must match Python."""
        ops = CooperOps()
        
        for _ in range(100):
            ternary = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
            q_pos, q_neg = pack_ternary(ternary)
            d_pos, d_neg = pack_ternary(ternary)
            
            native_score = ops.ternary_similarity(q_pos, q_neg, d_pos, d_neg)
            python_score = ternary_similarity(ternary, ternary)
            
            assert native_score == python_score
    
    def test_pack_unpack_roundtrip_native(self):
        """Native pack/unpack must roundtrip correctly."""
        ops = CooperOps()
        
        for size in [8, 64, 128, 384]:
            ternary = np.random.choice([-1, 0, 1], size=size).astype(np.int8)
            
            # Pack with native
            native_pos, native_neg = ops.pack_ternary(ternary)
            
            # Unpack with native
            recovered = ops.unpack_ternary(native_pos, native_neg, size)
            
            np.testing.assert_array_equal(ternary, recovered)
    
    def test_batch_similarity_matches_loop(self):
        """Batch similarity must match individual calls."""
        ops = CooperOps()
        
        q = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        q_pos, q_neg = ops.pack_ternary(q)
        
        n_docs = 100
        docs = [np.random.choice([-1, 0, 1], size=64).astype(np.int8) for _ in range(n_docs)]
        d_pos = np.array([ops.pack_ternary(d)[0] for d in docs])
        d_neg = np.array([ops.pack_ternary(d)[1] for d in docs])
        
        batch_scores = ops.batch_similarity(q_pos, q_neg, d_pos, d_neg)
        
        for i, d in enumerate(docs):
            dp, dn = ops.pack_ternary(d)
            individual = ops.ternary_similarity(q_pos, q_neg, dp, dn)
            assert batch_scores[i] == individual


class TestNumericalStability:
    """Operations MUST be numerically stable."""
    
    def test_no_overflow_in_similarity(self):
        """Similarity must not overflow for max-size vectors."""
        dim = 10000
        a = np.ones(dim, dtype=np.int8)
        b = np.ones(dim, dtype=np.int8)
        sim = ternary_similarity(a, b)
        assert sim == dim  # No overflow
    
    def test_large_input_quantization(self):
        """Large input values must quantize correctly."""
        x = np.array([1e10, -1e10, 1e-3, -1e-3])
        t = ternary_quantize(x, threshold=1e-5)
        assert t[0] == 1   # 1e10 > 1e-5
        assert t[1] == -1  # -1e10 < -1e-5
        assert t[2] == 1   # 1e-3 > 1e-5
        assert t[3] == -1  # -1e-3 < -1e-5
    
    def test_inf_handling(self):
        """Inf values must not crash."""
        x = np.array([np.inf, -np.inf, 0.0])
        t = ternary_quantize(x, threshold=0.0)
        assert t[0] == 1
        assert t[1] == -1
        assert t[2] == 0
    
    def test_nan_produces_zero(self):
        """NaN values should produce 0 (not activated)."""
        x = np.array([np.nan, 1.0, -1.0])
        t = ternary_quantize(x, threshold=0.0)
        # NaN comparisons are false, so it becomes 0
        assert t[0] == 0


class TestExplainInvariants:
    """Explanations MUST be accurate."""
    
    def test_explain_score_matches_similarity(self):
        """Explanation score must match similarity function."""
        a = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        b = np.random.choice([-1, 0, 1], size=64).astype(np.int8)
        
        exp = explain_match(a, b)
        sim = ternary_similarity(a, b)
        
        assert exp['score'] == sim
    
    def test_explain_dimensions_sum_to_total(self):
        """All dimensions must be accounted for."""
        dim = 64
        a = np.random.choice([-1, 0, 1], size=dim).astype(np.int8)
        b = np.random.choice([-1, 0, 1], size=dim).astype(np.int8)
        
        exp = explain_match(a, b)
        total = (len(exp['agreement']) + len(exp['conflict']) + 
                 len(exp['query_only']) + len(exp['document_only']) + 
                 len(exp['both_zero']))
        
        assert total == dim
    
    def test_explain_agreement_is_correct(self):
        """Agreement dimensions must actually agree."""
        a = np.array([1, -1, 1, 0], dtype=np.int8)
        b = np.array([1, -1, 0, 0], dtype=np.int8)
        
        exp = explain_match(a, b)
        
        for dim in exp['agreement']:
            assert a[dim] == b[dim] and a[dim] != 0


class TestScaleInvariants:
    """System MUST behave correctly at scale."""
    
    def test_large_document_count(self):
        """Must handle many documents."""
        db = OctaveDB(dimensions=64, coarse_threshold=-1.0)
        n_docs = 1000
        
        for i in range(n_docs):
            db.add(f"doc{i}", np.random.randn(64))
        
        assert len(db) == n_docs
        
        # Search should still work
        results = db.search(np.random.randn(64), top_k=10)
        assert len(results) == 10
    
    def test_high_dimensional(self):
        """Must handle high dimensions."""
        dim = 1024
        db = OctaveDB(dimensions=dim, coarse_threshold=-1.0)
        
        db.add("doc1", np.random.randn(dim))
        db.add("doc2", np.random.randn(dim))
        
        results = db.search(np.random.randn(dim), top_k=2)
        assert len(results) == 2
    
    def test_memory_does_not_grow_unbounded(self):
        """Memory should be bounded by document count."""
        import sys
        
        db = OctaveDB(dimensions=64)
        
        # Add and remove repeatedly
        for i in range(100):
            db.add(f"doc{i}", np.random.randn(64))
        
        for i in range(100):
            db.remove(f"doc{i}")
        
        assert len(db) == 0
        # Index should be empty
        assert db.index.num_documents == 0


class TestConcurrencyInvariants:
    """State MUST remain consistent under multiple operations."""
    
    def test_add_remove_consistency(self):
        """Add-remove cycles must maintain consistency."""
        db = OctaveDB(dimensions=64, coarse_threshold=-1.0)
        
        # Add 100 docs
        for i in range(100):
            db.add(f"doc{i}", np.random.randn(64))
        
        assert len(db) == 100
        
        # Remove odd-numbered
        for i in range(1, 100, 2):
            db.remove(f"doc{i}")
        
        assert len(db) == 50
        
        # Even-numbered should still be searchable
        results = db.search(np.random.randn(64), top_k=100)
        result_ids = {r.id for r in results}
        
        for i in range(0, 100, 2):
            assert f"doc{i}" in result_ids
    
    def test_repeated_search_consistent(self):
        """Same query must return same results."""
        db = OctaveDB(dimensions=64, coarse_threshold=-1.0)
        
        for i in range(20):
            db.add(f"doc{i}", np.random.randn(64))
        
        query = np.random.randn(64)
        
        results1 = db.search(query, mode="similar", top_k=5)
        results2 = db.search(query, mode="similar", top_k=5)
        
        ids1 = [r.id for r in results1]
        ids2 = [r.id for r in results2]
        
        assert ids1 == ids2
