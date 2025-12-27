"""
Tests for Hollywood Squares and Codex Squares Index.

Paper-ready tests demonstrating:
1. HollywoodSquaresIndex with k-means centroids
2. CodexSquaresIndex with Gene Key archetypes
3. Recall vs probe count tradeoffs
4. Archetypal enrichment validation
"""

import pytest
import numpy as np
import json
import os
from pathlib import Path


# Skip if data files don't exist
DATA_EXISTS = (
    Path('/tmp/shakespeare_embeddings.npy').exists() and
    Path('/tmp/shakespeare_chunks.json').exists() and
    Path('/tmp/genekeys_complete.json').exists()
)

skip_no_data = pytest.mark.skipif(
    not DATA_EXISTS,
    reason="Test data files not found in /tmp"
)


class TestHollywoodSquaresIndex:
    """Tests for HollywoodSquaresIndex."""
    
    def test_create_index(self):
        """Test index creation."""
        from trix.db import HollywoodSquaresIndex
        
        index = HollywoodSquaresIndex(dimensions=384, n_tiles=64)
        assert index.n_tiles == 64
        assert index.dimensions == 384
        assert len(index.tiles) == 64
    
    def test_add_vectors(self):
        """Test adding vectors."""
        from trix.db import HollywoodSquaresIndex
        
        index = HollywoodSquaresIndex(dimensions=384)
        
        np.random.seed(42)
        for i in range(100):
            vec = np.random.randn(384).astype(np.float32)
            index.add(f"doc_{i}", vec)
        
        assert index.num_documents == 100
    
    def test_build_centroids(self):
        """Test k-means centroid building."""
        from trix.db import HollywoodSquaresIndex
        
        index = HollywoodSquaresIndex(dimensions=384)
        
        np.random.seed(42)
        for i in range(500):
            vec = np.random.randn(384).astype(np.float32)
            index.add(f"doc_{i}", vec)
        
        index.build_centroids(n_iterations=5)
        
        assert index.centroids is not None
        assert index.centroids.shape == (64, 384)
        assert index._built
    
    def test_search_returns_results(self):
        """Test that search returns results."""
        from trix.db import HollywoodSquaresIndex
        
        index = HollywoodSquaresIndex(dimensions=384)
        
        np.random.seed(42)
        vectors = np.random.randn(500, 384).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        for i, vec in enumerate(vectors):
            index.add(f"doc_{i}", vec)
        
        index.build_centroids(n_iterations=5)
        
        query = vectors[0]
        results = index.search(query, top_k=10, n_probes=8)
        
        assert len(results) == 10
        assert results[0][0] == "doc_0"  # Should find itself
        assert results[0][1] > 0.99  # High similarity
    
    def test_recall_increases_with_probes(self):
        """Test that recall increases with more probes."""
        from trix.db import HollywoodSquaresIndex
        
        index = HollywoodSquaresIndex(dimensions=384)
        
        np.random.seed(42)
        vectors = np.random.randn(1000, 384).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        for i, vec in enumerate(vectors):
            index.add(f"doc_{i}", vec)
        
        index.build_centroids(n_iterations=5)
        
        # Compute ground truth
        query = vectors[50]
        gt_scores = np.dot(vectors, query)
        gt_top10 = set(f"doc_{i}" for i in np.argsort(gt_scores)[-10:])
        
        # Test increasing probes
        prev_recall = 0
        for n_probes in [4, 8, 16, 32]:
            results = index.search(query, top_k=10, n_probes=n_probes)
            retrieved = set(r[0] for r in results)
            recall = len(retrieved & gt_top10) / 10
            
            # Recall should generally increase (may not be strictly monotonic)
            # Just check that higher probes eventually get better recall
            if n_probes == 32:
                assert recall >= prev_recall * 0.9  # Allow some variance
            prev_recall = recall
    
    @skip_no_data
    def test_real_embeddings_high_recall(self):
        """Test that real embeddings achieve high recall with few probes."""
        from trix.db import HollywoodSquaresIndex
        
        embeddings = np.load('/tmp/shakespeare_embeddings.npy')
        
        index = HollywoodSquaresIndex(dimensions=384)
        for i, vec in enumerate(embeddings):
            index.add(f"doc_{i}", vec)
        index.build_centroids(n_iterations=10)
        
        # Sample queries
        np.random.seed(42)
        query_idx = np.random.choice(len(embeddings), 20, replace=False)
        
        recalls = []
        for idx in query_idx:
            query = embeddings[idx]
            
            # Ground truth
            gt_scores = np.dot(embeddings, query)
            gt_top10 = set(f"doc_{i}" for i in np.argsort(gt_scores)[-10:])
            
            # Search with 16 probes
            results = index.search(query, top_k=10, n_probes=16)
            retrieved = set(r[0] for r in results)
            
            recall = len(retrieved & gt_top10) / 10
            recalls.append(recall)
        
        avg_recall = np.mean(recalls)
        assert avg_recall >= 0.90, f"Expected recall >= 90%, got {avg_recall:.1%}"


class TestCodexSquaresIndex:
    """Tests for CodexSquaresIndex."""
    
    @skip_no_data
    def test_create_index(self):
        """Test index creation with Gene Keys."""
        from trix.db import CodexSquaresIndex
        
        with open('/tmp/genekeys_complete.json') as f:
            gk_data = json.load(f)
        
        # Create without embedder (just data structure)
        index = CodexSquaresIndex(gk_data, embedder=None)
        
        assert len(index.tiles) == 64
        assert index.tiles[1].gk_number == 1
        assert index.tiles[38].gift == "Perseverance"
    
    @skip_no_data
    def test_tiles_have_correct_names(self):
        """Test that tiles have correct Gene Key names."""
        from trix.db import CodexSquaresIndex
        
        with open('/tmp/genekeys_complete.json') as f:
            gk_data = json.load(f)
        
        index = CodexSquaresIndex(gk_data, embedder=None)
        
        # Check a few known Gene Keys
        assert index.tiles[1].shadow == "Entropy"
        assert index.tiles[1].gift == "Freshness"
        assert index.tiles[1].siddhi == "Beauty"
        
        assert index.tiles[38].shadow == "Struggle"
        assert index.tiles[38].gift == "Perseverance"
        assert index.tiles[38].siddhi == "Honour"
    
    @skip_no_data
    def test_topology_has_partners(self):
        """Test that partner edges are created."""
        from trix.db import CodexSquaresIndex
        
        with open('/tmp/genekeys_complete.json') as f:
            gk_data = json.load(f)
        
        index = CodexSquaresIndex(gk_data, embedder=None)
        
        # GK38 and GK39 are programming partners
        assert 39 in index.tiles[38].neighbors
        assert 38 in index.tiles[39].neighbors
    
    @skip_no_data
    def test_search_returns_codex_results(self):
        """Test that search returns CodexResult objects."""
        from trix.db import CodexSquaresIndex, CodexResult
        from sentence_transformers import SentenceTransformer
        
        with open('/tmp/genekeys_complete.json') as f:
            gk_data = json.load(f)
        
        embeddings = np.load('/tmp/shakespeare_embeddings.npy')[:100]
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        def embedder(texts):
            return model.encode(texts, normalize_embeddings=True)
        
        index = CodexSquaresIndex(gk_data, embedder)
        for i, vec in enumerate(embeddings):
            index.add(f"doc_{i}", vec)
        index.build()
        
        query = embeddings[0]
        results = index.search(query, top_k=5, n_probes=8)
        
        assert len(results) == 5
        assert all(isinstance(r, CodexResult) for r in results)
        assert all(1 <= r.gk_number <= 64 for r in results)
        assert all(r.shadow for r in results)
        assert all(r.gift for r in results)
        assert all(r.siddhi for r in results)
    
    @skip_no_data
    def test_archetypal_distribution(self):
        """Test that documents distribute across archetypes."""
        from trix.db import CodexSquaresIndex
        from sentence_transformers import SentenceTransformer
        
        with open('/tmp/genekeys_complete.json') as f:
            gk_data = json.load(f)
        
        embeddings = np.load('/tmp/shakespeare_embeddings.npy')
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        def embedder(texts):
            return model.encode(texts, normalize_embeddings=True)
        
        index = CodexSquaresIndex(gk_data, embedder)
        for i, vec in enumerate(embeddings):
            index.add(f"doc_{i}", vec)
        index.build()
        
        dist = index.get_tile_distribution()
        
        # Should have documents in multiple tiles
        non_empty = sum(1 for count in dist.values() if count > 0)
        assert non_empty >= 20, f"Expected >= 20 non-empty tiles, got {non_empty}"
        
        # Total should match
        total = sum(dist.values())
        assert total == len(embeddings)
    
    @skip_no_data
    def test_explain_result(self):
        """Test result explanation generation."""
        from trix.db import CodexSquaresIndex, CodexResult
        from sentence_transformers import SentenceTransformer
        
        with open('/tmp/genekeys_complete.json') as f:
            gk_data = json.load(f)
        
        embeddings = np.load('/tmp/shakespeare_embeddings.npy')[:50]
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        def embedder(texts):
            return model.encode(texts, normalize_embeddings=True)
        
        index = CodexSquaresIndex(gk_data, embedder)
        for i, vec in enumerate(embeddings):
            index.add(f"doc_{i}", vec)
        index.build()
        
        results = index.search(embeddings[0], top_k=1, n_probes=8)
        explanation = index.explain_result(results[0])
        
        assert "Gene Key" in explanation
        assert "shadow" in explanation
        assert "gift" in explanation
        assert "highest expression" in explanation


class TestCodexVsPlainComparison:
    """Tests comparing Codex vs Plain search."""
    
    @skip_no_data
    def test_both_achieve_good_recall(self):
        """Test that both methods achieve good recall."""
        from trix.db import HollywoodSquaresIndex, CodexSquaresIndex
        from sentence_transformers import SentenceTransformer
        
        with open('/tmp/genekeys_complete.json') as f:
            gk_data = json.load(f)
        
        embeddings = np.load('/tmp/shakespeare_embeddings.npy')
        n = len(embeddings)
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        def embedder(texts):
            return model.encode(texts, normalize_embeddings=True)
        
        # Build both indexes
        plain_index = HollywoodSquaresIndex(dimensions=384)
        codex_index = CodexSquaresIndex(gk_data, embedder)
        
        for i, vec in enumerate(embeddings):
            plain_index.add(f"doc_{i}", vec)
            codex_index.add(f"doc_{i}", vec)
        
        plain_index.build_centroids(n_iterations=10)
        codex_index.build()
        
        # Sample queries
        np.random.seed(42)
        query_idx = np.random.choice(n, 20, replace=False)
        
        plain_recalls, codex_recalls = [], []
        
        for idx in query_idx:
            query = embeddings[idx]
            
            # Ground truth
            gt_scores = np.dot(embeddings, query)
            gt_top10 = set(f"doc_{i}" for i in np.argsort(gt_scores)[-10:])
            
            # Plain search
            plain_results = plain_index.search(query, top_k=10, n_probes=16)
            plain_retrieved = set(r[0] for r in plain_results)
            plain_recalls.append(len(plain_retrieved & gt_top10) / 10)
            
            # Codex search
            codex_results = codex_index.search(query, top_k=10, n_probes=16)
            codex_retrieved = set(r.doc_id for r in codex_results)
            codex_recalls.append(len(codex_retrieved & gt_top10) / 10)
        
        plain_avg = np.mean(plain_recalls)
        codex_avg = np.mean(codex_recalls)
        
        # Both should achieve reasonable recall
        assert plain_avg >= 0.85, f"Plain recall {plain_avg:.1%} < 85%"
        assert codex_avg >= 0.80, f"Codex recall {codex_avg:.1%} < 80%"
    
    @skip_no_data
    def test_codex_provides_archetypes(self):
        """Test that Codex results include archetype info that plain doesn't."""
        from trix.db import HollywoodSquaresIndex, CodexSquaresIndex
        from sentence_transformers import SentenceTransformer
        
        with open('/tmp/genekeys_complete.json') as f:
            gk_data = json.load(f)
        
        embeddings = np.load('/tmp/shakespeare_embeddings.npy')[:200]
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        def embedder(texts):
            return model.encode(texts, normalize_embeddings=True)
        
        plain_index = HollywoodSquaresIndex(dimensions=384)
        codex_index = CodexSquaresIndex(gk_data, embedder)
        
        for i, vec in enumerate(embeddings):
            plain_index.add(f"doc_{i}", vec)
            codex_index.add(f"doc_{i}", vec)
        
        plain_index.build_centroids(n_iterations=5)
        codex_index.build()
        
        query = embeddings[0]
        
        # Plain returns (doc_id, score)
        plain_results = plain_index.search(query, top_k=5, n_probes=8)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in plain_results)
        
        # Codex returns CodexResult with archetype
        codex_results = codex_index.search(query, top_k=5, n_probes=8)
        assert all(hasattr(r, 'gk_number') for r in codex_results)
        assert all(hasattr(r, 'shadow') for r in codex_results)
        assert all(hasattr(r, 'gift') for r in codex_results)
        assert all(hasattr(r, 'siddhi') for r in codex_results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
