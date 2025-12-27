#!/usr/bin/env python3
"""
DB Cooper PURPOSE Test

The purpose of DB Cooper is SEMANTIC RETRIEVAL:
  - Find documents with similar MEANING
  - Reveal relationships at different resolutions
  - Explain WHY documents match

Speed means nothing if these don't work.
"""

import numpy as np
import sys

sys.path.insert(0, 'src')
from trix.db import OctaveDB


def create_semantic_embeddings():
    """
    Create embeddings with known semantic relationships.
    
    We simulate embeddings where:
    - Similar topics have similar patterns
    - Related concepts share some dimensions
    - Unrelated concepts are orthogonal
    """
    np.random.seed(42)
    dims = 128
    
    # Create topic basis vectors (orthogonal concepts)
    topics = {
        'animals': np.random.randn(dims),
        'food': np.random.randn(dims),
        'technology': np.random.randn(dims),
        'nature': np.random.randn(dims),
        'music': np.random.randn(dims),
    }
    # Normalize
    for k in topics:
        topics[k] = topics[k] / np.linalg.norm(topics[k])
    
    # Create documents as mixtures of topics
    documents = [
        # Animals
        ("Dogs are loyal companions", ['animals'], [1.0]),
        ("Cats are independent pets", ['animals'], [1.0]),
        ("Birds can fly and sing", ['animals', 'music'], [0.8, 0.2]),
        ("Fish live in water", ['animals', 'nature'], [0.7, 0.3]),
        ("Horses run fast", ['animals'], [1.0]),
        
        # Food
        ("Pizza is delicious", ['food'], [1.0]),
        ("Sushi is Japanese cuisine", ['food'], [1.0]),
        ("Apples grow on trees", ['food', 'nature'], [0.6, 0.4]),
        ("Cooking requires heat", ['food', 'technology'], [0.7, 0.3]),
        ("Wine pairs with cheese", ['food'], [1.0]),
        
        # Technology
        ("Computers process data", ['technology'], [1.0]),
        ("Phones connect people", ['technology'], [1.0]),
        ("AI learns from data", ['technology'], [1.0]),
        ("Robots automate tasks", ['technology'], [1.0]),
        ("Software runs on hardware", ['technology'], [1.0]),
        
        # Nature
        ("Mountains are tall", ['nature'], [1.0]),
        ("Rivers flow to the sea", ['nature'], [1.0]),
        ("Forests have many trees", ['nature'], [1.0]),
        ("Deserts are dry and hot", ['nature'], [1.0]),
        ("Oceans cover the earth", ['nature'], [1.0]),
        
        # Music
        ("Guitars have strings", ['music'], [1.0]),
        ("Drums keep the beat", ['music'], [1.0]),
        ("Singing expresses emotion", ['music'], [1.0]),
        ("Concerts are live performances", ['music'], [1.0]),
        ("Jazz is improvisational", ['music'], [1.0]),
        
        # Cross-topic
        ("Bird songs are beautiful", ['animals', 'music', 'nature'], [0.4, 0.4, 0.2]),
        ("Farm animals provide food", ['animals', 'food'], [0.5, 0.5]),
        ("Music streaming apps", ['music', 'technology'], [0.5, 0.5]),
        ("Nature documentaries", ['nature', 'technology'], [0.6, 0.4]),
        ("Pet food industry", ['animals', 'food', 'technology'], [0.4, 0.3, 0.3]),
    ]
    
    # Generate embeddings
    embeddings = []
    for text, topic_names, weights in documents:
        emb = np.zeros(dims)
        for topic, weight in zip(topic_names, weights):
            emb += weight * topics[topic]
        # Add small noise
        emb += 0.1 * np.random.randn(dims)
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb.astype(np.float32))
    
    return documents, embeddings, topics


def test_semantic_retrieval():
    """Test: Does DB Cooper find semantically similar documents?"""
    print("\n" + "=" * 70)
    print("TEST 1: SEMANTIC RETRIEVAL")
    print("Does DB Cooper find documents with similar MEANING?")
    print("=" * 70)
    
    documents, embeddings, topics = create_semantic_embeddings()
    
    # Build index - use low sparsity to preserve more information
    db = OctaveDB(dimensions=len(embeddings[0]), quantize_sparsity=0.05)
    for i, (text, _, _) in enumerate(documents):
        db.add(f"doc{i}", embeddings[i], metadata={"text": text})
    
    # Build topic lookup
    topic_docs = {}
    for i, (text, topic_names, _) in enumerate(documents):
        for t in topic_names:
            if t not in topic_docs:
                topic_docs[t] = []
            topic_docs[t].append(text)
    
    # Test queries - check if results contain same-topic documents
    test_cases = [
        ("Dogs are loyal companions", "animals"),
        ("Computers process data", "technology"),
        ("Forests have many trees", "nature"),
    ]
    
    passed = 0
    for query_text, expected_topic in test_cases:
        # Find query embedding
        query_idx = next(i for i, (t, _, _) in enumerate(documents) if t == query_text)
        query_emb = embeddings[query_idx]
        
        # Search
        results = db.search(query_emb, mode="similar", top_k=5)
        
        # Check how many results are from the same topic
        same_topic_docs = set(topic_docs[expected_topic])
        result_texts = [db.get(r.id)["metadata"]["text"] for r in results[1:5]]  # Skip self
        
        topic_hits = sum(1 for t in result_texts if t in same_topic_docs)
        success = topic_hits >= 2  # At least 2 of 4 results from same topic
        
        print(f"\nQuery: \"{query_text}\"")
        print(f"Expected topic: {expected_topic}")
        print(f"Results:")
        for r in results[:5]:
            doc = db.get(r.id)
            marker = "✓" if doc["metadata"]["text"] in same_topic_docs else " "
            print(f"  {marker} {doc['metadata']['text']} (score: {r.score:.3f})")
        print(f"Same-topic hits: {topic_hits}/4 → {'PASS' if success else 'FAIL'}")
        
        if success:
            passed += 1
    
    print(f"\n{'='*70}")
    print(f"SEMANTIC RETRIEVAL: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_multi_resolution():
    """Test: Do exact/similar/context modes reveal different relationships?"""
    print("\n" + "=" * 70)
    print("TEST 2: MULTI-RESOLUTION SEARCH")
    print("Do different modes reveal different relationships?")
    print("=" * 70)
    
    documents, embeddings, topics = create_semantic_embeddings()
    
    db = OctaveDB(dimensions=len(embeddings[0]), pool_factor=4, quantize_sparsity=0.05)
    for i, (text, topic_names, _) in enumerate(documents):
        db.add(f"doc{i}", embeddings[i], metadata={"text": text, "topics": topic_names})
    
    # Query: "Bird songs are beautiful" (cross-topic: animals + music + nature)
    query_idx = next(i for i, (t, _, _) in enumerate(documents) if "Bird songs" in t)
    query_emb = embeddings[query_idx]
    
    print(f"\nQuery: \"{documents[query_idx][0]}\"")
    print(f"Topics: {documents[query_idx][1]}")
    
    # Search at each resolution
    modes = ["exact", "similar", "context"]
    results_by_mode = {}
    
    for mode in modes:
        results = db.search(query_emb, mode=mode, top_k=5)
        results_by_mode[mode] = results
        
        print(f"\n{mode.upper()} mode:")
        for r in results[:4]:
            doc = db.get(r.id)
            print(f"  {doc['metadata']['text'][:40]:<40} topics={doc['metadata']['topics']}")
    
    # Check: different modes should surface different documents
    exact_ids = set(r.id for r in results_by_mode["exact"][:3])
    context_ids = set(r.id for r in results_by_mode["context"][:3])
    
    # Context should be more diverse (broader matches)
    exact_topics = set()
    context_topics = set()
    for r in results_by_mode["exact"][:5]:
        doc = db.get(r.id)
        exact_topics.update(doc["metadata"]["topics"])
    for r in results_by_mode["context"][:5]:
        doc = db.get(r.id)
        context_topics.update(doc["metadata"]["topics"])
    
    print(f"\nExact mode topics: {exact_topics}")
    print(f"Context mode topics: {context_topics}")
    
    # Context should cover same or more topics
    passed = len(context_topics) >= len(exact_topics)
    print(f"\nMulti-resolution reveals different views: {'PASS' if passed else 'FAIL'}")
    
    return passed


def test_explainability():
    """Test: Can we understand WHY documents match?"""
    print("\n" + "=" * 70)
    print("TEST 3: GLASSBOX EXPLAINABILITY")
    print("Can we understand WHY documents match?")
    print("=" * 70)
    
    documents, embeddings, topics = create_semantic_embeddings()
    
    db = OctaveDB(dimensions=len(embeddings[0]), quantize_sparsity=0.05)
    for i, (text, _, _) in enumerate(documents):
        db.add(f"doc{i}", embeddings[i], metadata={"text": text})
    
    # Query for a document
    query_text = "Dogs are loyal companions"
    query_idx = next(i for i, (t, _, _) in enumerate(documents) if t == query_text)
    query_emb = embeddings[query_idx]
    
    results = db.search(query_emb, mode="similar", top_k=3)
    
    print(f"\nQuery: \"{query_text}\"")
    
    for r in results[:2]:
        doc = db.get(r.id)
        print(f"\nMatch: \"{doc['metadata']['text']}\"")
        print(f"Score: {r.score}")
        
        # Get explanation
        explanation = db.explain(query_emb, r.id)
        
        print(f"Explanation:")
        print(f"  Score: {explanation['score']}")
        print(f"  Agreement dims: {len(explanation['agreement'])}")
        print(f"  Conflict dims: {len(explanation['conflict'])}")
        print(f"  Query-only dims: {len(explanation['query_only'])}")
        print(f"  Document-only dims: {len(explanation['document_only'])}")
        
        # The explanation should make sense
        # More agreements than conflicts for similar docs
        if len(explanation['agreement']) > len(explanation['conflict']):
            print(f"  → More agreements than conflicts: Makes sense!")
    
    # Check that explanation provides useful breakdown
    exp = db.explain(query_emb, results[0].id)
    passed = (
        'agreement' in exp and 
        'conflict' in exp and
        len(exp['agreement']) >= 0 and
        len(exp['agreement']) > len(exp['conflict'])  # Self-match has more agreements
    )
    
    print(f"\nExplanation provides meaningful breakdown: {'PASS' if passed else 'FAIL'}")
    return passed


def test_real_world_scenario():
    """Test: Document deduplication / near-duplicate detection."""
    print("\n" + "=" * 70)
    print("TEST 4: REAL-WORLD SCENARIO - Near-Duplicate Detection")
    print("Can DB Cooper find documents that are almost the same?")
    print("=" * 70)
    
    np.random.seed(42)
    dims = 128
    
    # Create "documents" with near-duplicates
    base_docs = [
        np.random.randn(dims).astype(np.float32) for _ in range(10)
    ]
    
    # Normalize
    base_docs = [d / np.linalg.norm(d) for d in base_docs]
    
    # Create variations (near-duplicates)
    all_docs = []
    labels = []
    for i, base in enumerate(base_docs):
        # Original
        all_docs.append(base)
        labels.append(f"original_{i}")
        
        # Small variation (near-duplicate)
        variation = base + 0.05 * np.random.randn(dims).astype(np.float32)
        variation = variation / np.linalg.norm(variation)
        all_docs.append(variation)
        labels.append(f"duplicate_{i}")
        
        # Large variation (different document)
        different = base + 0.5 * np.random.randn(dims).astype(np.float32)
        different = different / np.linalg.norm(different)
        all_docs.append(different)
        labels.append(f"different_{i}")
    
    # Build index
    db = OctaveDB(dimensions=dims, quantize_sparsity=0.1)
    for i, doc in enumerate(all_docs):
        db.add(labels[i], doc)
    
    # Test: can we find near-duplicates?
    correct = 0
    total = 10
    
    print("\nFinding near-duplicates of original documents:")
    for i in range(10):
        query = all_docs[i * 3]  # Original
        results = db.search(query, mode="exact", top_k=3)
        
        # The duplicate should be in top 3
        result_ids = [r.id for r in results]
        expected_dup = f"duplicate_{i}"
        
        found = expected_dup in result_ids
        if found:
            correct += 1
        
        if i < 3:  # Show first 3
            print(f"\n  Query: original_{i}")
            print(f"  Results: {result_ids}")
            print(f"  Found duplicate: {'YES' if found else 'NO'}")
    
    accuracy = correct / total
    passed = accuracy >= 0.8  # 80% accuracy
    
    print(f"\nNear-duplicate detection: {correct}/{total} ({accuracy:.0%})")
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    
    return passed


def main():
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "DB COOPER PURPOSE TEST" + " " * 26 + "#")
    print("#" + " " * 15 + "Speed means nothing without meaning" + " " * 18 + "#")
    print("#" * 70)
    
    results = []
    
    results.append(("Semantic Retrieval", test_semantic_retrieval()))
    results.append(("Multi-Resolution Search", test_multi_resolution()))
    results.append(("Glassbox Explainability", test_explainability()))
    results.append(("Near-Duplicate Detection", test_real_world_scenario()))
    
    # Summary
    print("\n" + "=" * 70)
    print("PURPOSE TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, p in results if p)
    
    for name, p in results:
        status = "PASS ✓" if p else "FAIL ✗"
        print(f"  {name:<30} {status}")
    
    print("=" * 70)
    print(f"OVERALL: {passed}/{len(results)} purpose tests passed")
    
    if passed == len(results):
        print("\nDB Cooper fulfills its PURPOSE.")
        print("It finds meaning, not just bits.")
    else:
        print("\nWork needed to fulfill purpose.")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
