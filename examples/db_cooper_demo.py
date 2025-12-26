#!/usr/bin/env python3
"""
DB Cooper Demo: Octave DB in action.

Demonstrates multi-resolution ternary retrieval with glassbox explainability.
"""

import numpy as np
import time
from trix.db import OctaveDB


def demo_basic():
    """Basic usage demo."""
    print("=" * 60)
    print("DB COOPER: Basic Demo")
    print("=" * 60)
    
    # Create database
    db = OctaveDB(dimensions=128, pool_factor=4)
    
    # Add some documents with float embeddings
    np.random.seed(42)
    documents = {
        "doc_cat": np.random.randn(128) + np.array([1] * 64 + [0] * 64),
        "doc_dog": np.random.randn(128) + np.array([0.8] * 64 + [0] * 64),
        "doc_car": np.random.randn(128) + np.array([0] * 64 + [1] * 64),
        "doc_bike": np.random.randn(128) + np.array([0] * 64 + [0.8] * 64),
    }
    
    for doc_id, embedding in documents.items():
        db.add(doc_id, embedding, metadata={"name": doc_id})
    
    print(f"\nAdded {len(db)} documents")
    print(f"Stats: {db.stats()}")
    
    # Search for something cat-like
    query = np.random.randn(128) + np.array([0.9] * 64 + [0] * 64)
    
    print("\n--- Search: 'cat-like' query ---")
    results = db.search(query, mode="similar", top_k=4, explain=False)
    for r in results:
        print(f"  {r.id}: score={r.score:.3f}, levels={r.level_scores}")
    
    # Explain the top match
    print("\n--- Explain top match ---")
    exp = db.explain(query, results[0].id)
    print(f"  Agreement dims: {len(exp['agreement'])} dimensions agree")
    print(f"  Conflict dims:  {len(exp['conflict'])} dimensions conflict")
    print(f"  Score: {exp['score']}")


def demo_three_modes():
    """Demonstrate three search modes."""
    print("\n" + "=" * 60)
    print("DB COOPER: Three Search Modes")
    print("=" * 60)
    
    db = OctaveDB(dimensions=256, pool_factor=4, coarse_threshold=-1.0)
    
    # Create clustered documents
    np.random.seed(123)
    
    # Cluster A: "science" documents
    for i in range(10):
        emb = np.random.randn(256) * 0.3
        emb[:64] += 1.0  # Science signature
        db.add(f"science_{i}", emb, metadata={"cluster": "science"})
    
    # Cluster B: "art" documents  
    for i in range(10):
        emb = np.random.randn(256) * 0.3
        emb[64:128] += 1.0  # Art signature
        db.add(f"art_{i}", emb, metadata={"cluster": "art"})
    
    # Cluster C: "music" documents
    for i in range(10):
        emb = np.random.randn(256) * 0.3
        emb[128:192] += 1.0  # Music signature
        db.add(f"music_{i}", emb, metadata={"cluster": "music"})
    
    print(f"\nAdded {len(db)} documents in 3 clusters")
    
    # Query: something science-ish
    query = np.random.randn(256) * 0.3
    query[:64] += 0.8
    
    print("\n--- EXACT mode (fine-level precision) ---")
    results = db.search(query, mode="exact", top_k=5)
    for r in results:
        cluster = db.get(r.id)['metadata']['cluster']
        print(f"  {r.id} ({cluster}): score={r.score:.3f}")
    
    print("\n--- SIMILAR mode (balanced) ---")
    results = db.search(query, mode="similar", top_k=5)
    for r in results:
        cluster = db.get(r.id)['metadata']['cluster']
        print(f"  {r.id} ({cluster}): score={r.score:.3f}")
    
    print("\n--- CONTEXT mode (coarse discovery) ---")
    results = db.search(query, mode="context", top_k=5)
    for r in results:
        cluster = db.get(r.id)['metadata']['cluster']
        print(f"  {r.id} ({cluster}): score={r.score:.3f}")


def demo_benchmark():
    """Benchmark performance."""
    print("\n" + "=" * 60)
    print("DB COOPER: Performance Benchmark")
    print("=" * 60)
    
    dimensions = 384
    num_docs = 10000
    num_queries = 100
    
    print(f"\nConfig: {num_docs} documents, {dimensions} dimensions")
    
    # Create database and add documents
    db = OctaveDB(dimensions=dimensions, pool_factor=4, coarse_threshold=0.3)
    
    np.random.seed(42)
    embeddings = np.random.randn(num_docs, dimensions).astype(np.float32)
    doc_ids = [f"doc_{i}" for i in range(num_docs)]
    
    start = time.time()
    db.add_batch(doc_ids, embeddings)
    index_time = time.time() - start
    
    print(f"\nIndexing: {index_time:.2f}s ({num_docs/index_time:.0f} docs/sec)")
    print(f"Stats: {db.stats()}")
    
    # Generate queries
    queries = np.random.randn(num_queries, dimensions).astype(np.float32)
    
    # Benchmark search
    print("\n--- Search Benchmark ---")
    for mode in ["exact", "similar", "context"]:
        start = time.time()
        for q in queries:
            results = db.search(q, mode=mode, top_k=10)
        elapsed = time.time() - start
        qps = num_queries / elapsed
        avg_ms = (elapsed / num_queries) * 1000
        print(f"  {mode:8s}: {qps:.0f} queries/sec, {avg_ms:.2f} ms/query")


def demo_glassbox():
    """Demonstrate glassbox explainability."""
    print("\n" + "=" * 60)
    print("DB COOPER: Glassbox Explainability")
    print("=" * 60)
    
    # Disable sparsity targeting for this demo - we want exact ternary
    db = OctaveDB(dimensions=32, pool_factor=4, coarse_threshold=-1.0,
                  quantize_sparsity=None, quantize_threshold=0.5)
    
    # Create simple documents with clear patterns (use values > threshold)
    doc_a = np.array([1.0]*8 + [-1.0]*8 + [0.0]*8 + [1.0]*8, dtype=np.float32)
    doc_b = np.array([1.0]*8 + [1.0]*8 + [0.0]*8 + [-1.0]*8, dtype=np.float32)
    
    db.add("doc_a", doc_a)
    db.add("doc_b", doc_b)
    
    # Query similar to doc_a
    query = np.array([1.0]*8 + [-1.0]*8 + [0.0]*8 + [1.0]*8, dtype=np.float32)
    
    print("\nQuery pattern:  [+++++++|--------|00000000|++++++++]")
    print("Doc A pattern:  [+++++++|--------|00000000|++++++++]")
    print("Doc B pattern:  [+++++++|++++++++|00000000|--------]")
    
    print("\n--- Explain: Query vs Doc A ---")
    exp = db.explain(query, "doc_a")
    print(f"  Agreements: {len(exp['agreement'])} dims (should be ~24)")
    print(f"  Conflicts:  {len(exp['conflict'])} dims (should be 0)")
    print(f"  Score:      {exp['score']} (maximum possible)")
    
    print("\n--- Explain: Query vs Doc B ---")
    exp = db.explain(query, "doc_b")
    print(f"  Agreements: {len(exp['agreement'])} dims")
    print(f"  Conflicts:  {len(exp['conflict'])} dims")
    print(f"  Score:      {exp['score']}")
    
    # Show actual dimensions
    print(f"\n  Conflict dimensions: {exp['conflict'][:10]}...")
    print("  (These are where query and doc disagree)")


def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("#" + " " * 20 + "DB COOPER" + " " * 29 + "#")
    print("#" + " " * 10 + "Multi-Resolution Ternary Retrieval" + " " * 13 + "#")
    print("#" * 60)
    
    demo_basic()
    demo_three_modes()
    demo_glassbox()
    demo_benchmark()
    
    print("\n" + "=" * 60)
    print("DB Cooper: Find the exact, the similar, and the related.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
