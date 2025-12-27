"""
Rigorous tests for aboutness understanding.

These tests prove that DB Cooper can understand what a query is ABOUT,
not just find similar vectors.

Every test is:
- Transparent: you can see exactly what happened
- Reproducible: same seed = same results
- Meaningful: tests real semantic understanding
"""

import numpy as np
import pytest
from collections import Counter

from trix.db.core import octave_quantize
from trix.db.explorer import PossibilityExplorer, compare_approaches


class TestAboutnessUnderstanding:
    """Tests for semantic aboutness understanding."""
    
    @pytest.fixture
    def confuser_dataset(self):
        """
        Create a dataset with known confuser pairs.
        
        Confusers are concepts that are SIMILAR but DIFFERENT.
        Good retrieval should distinguish them.
        """
        np.random.seed(42)
        dims = 256
        
        # Define confuser pairs with target similarity
        pairs = [
            ('dog', 'cat', 0.7),        # Both pets
            ('car', 'truck', 0.8),       # Both vehicles
            ('happy', 'excited', 0.75),  # Both positive emotions
            ('rain', 'snow', 0.65),      # Both precipitation
            ('python', 'javascript', 0.6), # Both programming
        ]
        
        concept_bases = {}
        
        for c1, c2, target_sim in pairs:
            # Create c1
            base1 = np.random.randn(dims).astype(np.float32)
            base1 = base1 / np.linalg.norm(base1)
            
            # Create c2 with target similarity
            orthogonal = np.random.randn(dims).astype(np.float32)
            orthogonal = orthogonal - np.dot(orthogonal, base1) * base1
            orthogonal = orthogonal / np.linalg.norm(orthogonal)
            
            base2 = target_sim * base1 + np.sqrt(1 - target_sim**2) * orthogonal
            base2 = base2 / np.linalg.norm(base2)
            
            concept_bases[c1] = base1
            concept_bases[c2] = base2
        
        # Create documents
        docs_per_concept = 100
        doc_embeddings = []
        doc_concepts = []
        doc_ids = []
        
        for concept, base in concept_bases.items():
            for i in range(docs_per_concept):
                noise = 0.2 * np.random.randn(dims).astype(np.float32)
                emb = base + noise
                emb = emb / np.linalg.norm(emb)
                
                doc_embeddings.append(emb)
                doc_concepts.append(concept)
                doc_ids.append(f"{concept}_{i}")
        
        doc_embeddings = np.array(doc_embeddings)
        
        # Quantize
        doc_signs = np.array([octave_quantize(e)[0] for e in doc_embeddings])
        doc_mags = np.array([octave_quantize(e)[1] for e in doc_embeddings])
        
        return {
            'embeddings': doc_embeddings,
            'signs': doc_signs,
            'mags': doc_mags,
            'concepts': doc_concepts,
            'ids': doc_ids,
            'concept_bases': concept_bases,
            'pairs': pairs,
        }
    
    def test_confuser_rejection(self, confuser_dataset):
        """
        Test that Cooper rejects confusers better than naive top-k.
        
        This is the core aboutness test.
        """
        data = confuser_dataset
        
        explorer = PossibilityExplorer(
            doc_embeddings=data['embeddings'],
            doc_signs=data['signs'],
            doc_mags=data['mags'],
            doc_concepts=data['concepts'],
            doc_ids=data['ids'],
        )
        
        total_cooper_correct = 0
        total_cooper_confuser = 0
        total_naive_correct = 0
        total_naive_confuser = 0
        
        np.random.seed(123)  # Different seed for queries
        
        for c1, c2, _ in data['pairs']:
            # Test queries for c1 (c2 is confuser)
            for i in range(20):
                base = data['concept_bases'][c1]
                query = base + 0.15 * np.random.randn(256).astype(np.float32)
                query = query / np.linalg.norm(query)
                
                result = compare_approaches(
                    explorer, query, 
                    true_concept=c1, 
                    confuser_concept=c2,
                    query_id=f"{c1}_query_{i}"
                )
                
                total_cooper_correct += result['cooper']['correct']
                total_cooper_confuser += result['cooper']['confuser']
                total_naive_correct += result['naive']['correct']
                total_naive_confuser += result['naive']['confuser']
        
        cooper_purity = total_cooper_correct / (total_cooper_correct + total_cooper_confuser)
        naive_purity = total_naive_correct / (total_naive_correct + total_naive_confuser)
        
        # Cooper should have higher purity
        assert cooper_purity > naive_purity, (
            f"Cooper purity ({cooper_purity:.2%}) should exceed "
            f"naive purity ({naive_purity:.2%})"
        )
        
        # Cooper should have at least 10% better purity
        purity_advantage = cooper_purity - naive_purity
        assert purity_advantage >= 0.10, (
            f"Cooper should have at least 10% purity advantage, "
            f"got {purity_advantage:.2%}"
        )
    
    def test_explanation_transparency(self, confuser_dataset):
        """
        Test that every decision is explained.
        
        No black boxes.
        """
        data = confuser_dataset
        
        explorer = PossibilityExplorer(
            doc_embeddings=data['embeddings'],
            doc_signs=data['signs'],
            doc_mags=data['mags'],
            doc_concepts=data['concepts'],
            doc_ids=data['ids'],
        )
        
        np.random.seed(456)
        query = data['concept_bases']['dog'] + 0.1 * np.random.randn(256).astype(np.float32)
        query = query / np.linalg.norm(query)
        
        result = explorer.explore(query, query_id="test_transparency")
        
        # Check that explanation exists and is detailed
        explanation = result.explain()
        
        assert "EXPLORATION REPORT" in explanation
        assert "STAGE 1: EXPLORATION" in explanation
        assert "STAGE 2: STRUCTURE ANALYSIS" in explanation
        assert "STAGE 3: UNDERSTANDING" in explanation
        assert "DECISION TRACE" in explanation
        assert "FINAL RESULTS" in explanation
        
        # Check that decisions are recorded
        assert len(result.decisions) >= 3, "Should have at least 3 decision points"
        
        # Check that each result has a reason
        for r in result.results:
            assert r.reason != "", f"Result {r.id} should have a reason"
    
    def test_concept_understanding(self, confuser_dataset):
        """
        Test that Cooper correctly identifies the query's concept.
        """
        data = confuser_dataset
        
        explorer = PossibilityExplorer(
            doc_embeddings=data['embeddings'],
            doc_signs=data['signs'],
            doc_mags=data['mags'],
            doc_concepts=data['concepts'],
            doc_ids=data['ids'],
        )
        
        np.random.seed(789)
        
        correct_understanding = 0
        total_queries = 0
        
        for concept, base in data['concept_bases'].items():
            for i in range(10):
                query = base + 0.1 * np.random.randn(256).astype(np.float32)
                query = query / np.linalg.norm(query)
                
                result = explorer.explore(query)
                
                if result.dominant_concept == concept:
                    correct_understanding += 1
                total_queries += 1
        
        understanding_rate = correct_understanding / total_queries
        
        # Should understand at least 80% of queries correctly
        assert understanding_rate >= 0.80, (
            f"Should understand at least 80% of queries, "
            f"got {understanding_rate:.2%}"
        )
    
    def test_confidence_correlates_with_correctness(self, confuser_dataset):
        """
        Test that higher confidence means more likely to be correct.
        """
        data = confuser_dataset
        
        explorer = PossibilityExplorer(
            doc_embeddings=data['embeddings'],
            doc_signs=data['signs'],
            doc_mags=data['mags'],
            doc_concepts=data['concepts'],
            doc_ids=data['ids'],
        )
        
        np.random.seed(101112)
        
        high_conf_correct = 0
        high_conf_total = 0
        low_conf_correct = 0
        low_conf_total = 0
        
        for concept, base in data['concept_bases'].items():
            for i in range(20):
                # Vary query noise to get different confidences
                noise_level = 0.1 + 0.2 * np.random.rand()
                query = base + noise_level * np.random.randn(256).astype(np.float32)
                query = query / np.linalg.norm(query)
                
                result = explorer.explore(query)
                
                is_correct = result.dominant_concept == concept
                
                if result.dominant_confidence >= 0.7:
                    high_conf_total += 1
                    if is_correct:
                        high_conf_correct += 1
                else:
                    low_conf_total += 1
                    if is_correct:
                        low_conf_correct += 1
        
        high_conf_accuracy = high_conf_correct / (high_conf_total + 0.001)
        low_conf_accuracy = low_conf_correct / (low_conf_total + 0.001)
        
        # High confidence should have higher accuracy
        assert high_conf_accuracy >= low_conf_accuracy, (
            f"High confidence accuracy ({high_conf_accuracy:.2%}) should exceed "
            f"low confidence accuracy ({low_conf_accuracy:.2%})"
        )


class TestHardCases:
    """Test the hardest cases - very similar confusers."""
    
    @pytest.fixture
    def hard_confusers(self):
        """Create dataset with 90%+ similarity confusers."""
        np.random.seed(42)
        dims = 256
        
        # VERY similar pairs
        pairs = [
            ('golden_retriever', 'labrador', 0.92),
            ('sedan', 'coupe', 0.90),
            ('jazz', 'blues', 0.91),
        ]
        
        concept_bases = {}
        
        for c1, c2, target_sim in pairs:
            base1 = np.random.randn(dims).astype(np.float32)
            base1 = base1 / np.linalg.norm(base1)
            
            orthogonal = np.random.randn(dims).astype(np.float32)
            orthogonal = orthogonal - np.dot(orthogonal, base1) * base1
            orthogonal = orthogonal / np.linalg.norm(orthogonal)
            
            base2 = target_sim * base1 + np.sqrt(1 - target_sim**2) * orthogonal
            base2 = base2 / np.linalg.norm(base2)
            
            concept_bases[c1] = base1
            concept_bases[c2] = base2
        
        docs_per_concept = 150
        doc_embeddings = []
        doc_concepts = []
        doc_ids = []
        
        for concept, base in concept_bases.items():
            for i in range(docs_per_concept):
                noise = 0.15 * np.random.randn(dims).astype(np.float32)
                emb = base + noise
                emb = emb / np.linalg.norm(emb)
                
                doc_embeddings.append(emb)
                doc_concepts.append(concept)
                doc_ids.append(f"{concept}_{i}")
        
        doc_embeddings = np.array(doc_embeddings)
        doc_signs = np.array([octave_quantize(e)[0] for e in doc_embeddings])
        doc_mags = np.array([octave_quantize(e)[1] for e in doc_embeddings])
        
        return {
            'embeddings': doc_embeddings,
            'signs': doc_signs,
            'mags': doc_mags,
            'concepts': doc_concepts,
            'ids': doc_ids,
            'concept_bases': concept_bases,
            'pairs': pairs,
        }
    
    def test_hard_confusers(self, hard_confusers):
        """
        Even with 90%+ similar confusers, Cooper should outperform naive.
        """
        data = hard_confusers
        
        explorer = PossibilityExplorer(
            doc_embeddings=data['embeddings'],
            doc_signs=data['signs'],
            doc_mags=data['mags'],
            doc_concepts=data['concepts'],
            doc_ids=data['ids'],
        )
        
        np.random.seed(999)
        
        cooper_purity_sum = 0
        naive_purity_sum = 0
        n_tests = 0
        
        for c1, c2, sim in data['pairs']:
            for i in range(30):
                base = data['concept_bases'][c1]
                query = base + 0.1 * np.random.randn(256).astype(np.float32)
                query = query / np.linalg.norm(query)
                
                result = compare_approaches(
                    explorer, query,
                    true_concept=c1,
                    confuser_concept=c2,
                )
                
                cooper_purity_sum += result['cooper']['purity']
                naive_purity_sum += result['naive']['purity']
                n_tests += 1
        
        cooper_avg = cooper_purity_sum / n_tests
        naive_avg = naive_purity_sum / n_tests
        
        # Cooper should still be better even on hard cases
        assert cooper_avg > naive_avg, (
            f"Cooper ({cooper_avg:.2%}) should beat naive ({naive_avg:.2%}) "
            f"even on 90%+ similarity confusers"
        )


class TestReproducibility:
    """Test that results are reproducible."""
    
    def test_same_seed_same_results(self):
        """Same random seed should give identical results."""
        dims = 256
        
        results = []
        for run in range(3):
            np.random.seed(42)
            
            # Create simple dataset
            base1 = np.random.randn(dims).astype(np.float32)
            base1 = base1 / np.linalg.norm(base1)
            base2 = np.random.randn(dims).astype(np.float32)
            base2 = base2 / np.linalg.norm(base2)
            
            docs = []
            concepts = []
            ids = []
            
            for i in range(50):
                emb = base1 + 0.2 * np.random.randn(dims).astype(np.float32)
                emb = emb / np.linalg.norm(emb)
                docs.append(emb)
                concepts.append('A')
                ids.append(f'A_{i}')
            
            for i in range(50):
                emb = base2 + 0.2 * np.random.randn(dims).astype(np.float32)
                emb = emb / np.linalg.norm(emb)
                docs.append(emb)
                concepts.append('B')
                ids.append(f'B_{i}')
            
            docs = np.array(docs)
            signs = np.array([octave_quantize(e)[0] for e in docs])
            mags = np.array([octave_quantize(e)[1] for e in docs])
            
            explorer = PossibilityExplorer(docs, signs, mags, concepts, ids)
            
            query = base1 + 0.1 * np.random.randn(dims).astype(np.float32)
            query = query / np.linalg.norm(query)
            
            result = explorer.explore(query)
            results.append([r.id for r in result.results])
        
        # All runs should give same results
        assert results[0] == results[1] == results[2], "Results should be reproducible"
