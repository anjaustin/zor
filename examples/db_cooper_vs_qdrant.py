#!/usr/bin/env python3
"""
DB Cooper vs Qdrant

Direct comparison between DB Cooper and Qdrant vector database.
Both running in-memory for fair comparison.
"""

import numpy as np
import time
import sys
from dataclasses import dataclass
from typing import List

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# DB Cooper
sys.path.insert(0, 'src')
from trix.db.core import ternary_quantize, pack_ternary
from trix.db.ops import CooperOps, native_ops_available


@dataclass
class BenchmarkResult:
    name: str
    index_time_ms: float
    query_time_ms: float
    queries_per_sec: float
    recall_at_1: float
    recall_at_10: float
    memory_estimate_mb: float


def benchmark_qdrant(data: np.ndarray, queries: np.ndarray, k: int = 10) -> BenchmarkResult:
    """Benchmark Qdrant in-memory."""
    n_docs, dims = data.shape
    
    # Create in-memory client
    client = QdrantClient(":memory:")
    
    # Index
    start = time.perf_counter()
    
    # Create collection
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
    )
    
    # Add vectors in batches
    batch_size = 1000
    for i in range(0, n_docs, batch_size):
        batch_end = min(i + batch_size, n_docs)
        points = [
            PointStruct(id=j, vector=data[j].tolist())
            for j in range(i, batch_end)
        ]
        client.upsert(collection_name="test", points=points)
    
    index_time = (time.perf_counter() - start) * 1000
    
    # Query
    start = time.perf_counter()
    all_results = []
    for q in queries:
        results = client.query_points(
            collection_name="test",
            query=q.tolist(),
            limit=k,
        ).points
        all_results.append([r.id for r in results])
    query_time = (time.perf_counter() - start) * 1000
    
    # Self-retrieval test
    n_test = min(100, n_docs)
    self_hits = 0
    for i in range(n_test):
        results = client.query_points(
            collection_name="test",
            query=data[i].tolist(),
            limit=1,
        ).points
        if results and results[0].id == i:
            self_hits += 1
    recall_1 = self_hits / n_test
    
    # R@10 via self-retrieval (check if self is in top 10)
    hits_10 = 0
    for i in range(n_test):
        results = client.query_points(
            collection_name="test",
            query=data[i].tolist(),
            limit=k,
        ).points
        if any(r.id == i for r in results):
            hits_10 += 1
    recall_10 = hits_10 / n_test
    
    # Memory estimate (float32 vectors)
    memory_mb = data.nbytes / 1024 / 1024
    
    return BenchmarkResult(
        name="Qdrant (in-memory)",
        index_time_ms=index_time,
        query_time_ms=query_time,
        queries_per_sec=len(queries) / (query_time / 1000),
        recall_at_1=recall_1,
        recall_at_10=recall_10,
        memory_estimate_mb=memory_mb,
    )


def benchmark_qdrant_quantized(data: np.ndarray, queries: np.ndarray, k: int = 10) -> BenchmarkResult:
    """Benchmark Qdrant with scalar quantization."""
    n_docs, dims = data.shape
    
    from qdrant_client.models import ScalarQuantizationConfig, ScalarType
    
    client = QdrantClient(":memory:")
    
    # Index with quantization
    start = time.perf_counter()
    
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
        quantization_config=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            always_ram=True,
        ),
    )
    
    batch_size = 1000
    for i in range(0, n_docs, batch_size):
        batch_end = min(i + batch_size, n_docs)
        points = [
            PointStruct(id=j, vector=data[j].tolist())
            for j in range(i, batch_end)
        ]
        client.upsert(collection_name="test", points=points)
    
    index_time = (time.perf_counter() - start) * 1000
    
    # Query with quantization
    start = time.perf_counter()
    all_results = []
    for q in queries:
        results = client.query_points(
            collection_name="test",
            query=q.tolist(),
            limit=k,
        ).points
        all_results.append([r.id for r in results])
    query_time = (time.perf_counter() - start) * 1000
    
    # Self-retrieval
    n_test = min(100, n_docs)
    self_hits = 0
    for i in range(n_test):
        results = client.query_points(
            collection_name="test",
            query=data[i].tolist(),
            limit=1,
        ).points
        if results and results[0].id == i:
            self_hits += 1
    recall_1 = self_hits / n_test
    
    hits_10 = 0
    for i in range(n_test):
        results = client.query_points(
            collection_name="test",
            query=data[i].tolist(),
            limit=k,
        ).points
        if any(r.id == i for r in results):
            hits_10 += 1
    recall_10 = hits_10 / n_test
    
    # Memory (int8 = 1/4 of float32)
    memory_mb = data.nbytes / 4 / 1024 / 1024
    
    return BenchmarkResult(
        name="Qdrant (INT8 quant)",
        index_time_ms=index_time,
        query_time_ms=query_time,
        queries_per_sec=len(queries) / (query_time / 1000),
        recall_at_1=recall_1,
        recall_at_10=recall_10,
        memory_estimate_mb=memory_mb,
    )


def benchmark_db_cooper(data: np.ndarray, queries: np.ndarray, k: int = 10) -> BenchmarkResult:
    """Benchmark DB Cooper with NEON."""
    if not native_ops_available():
        return None
    
    ops = CooperOps()
    n_docs, dims = data.shape
    packed_bytes = (dims + 7) // 8
    
    # Index
    start = time.perf_counter()
    ternary_data = ternary_quantize(data, sparsity_target=0.1)
    d_pos = np.zeros((n_docs, packed_bytes), dtype=np.uint8)
    d_neg = np.zeros((n_docs, packed_bytes), dtype=np.uint8)
    for i, t in enumerate(ternary_data):
        p, n = pack_ternary(t)
        d_pos[i] = p
        d_neg[i] = n
    index_time = (time.perf_counter() - start) * 1000
    
    # Query
    start = time.perf_counter()
    ternary_queries = ternary_quantize(queries, sparsity_target=0.1)
    all_results = []
    for tq in ternary_queries:
        q_pos, q_neg = pack_ternary(tq)
        scores = ops.batch_similarity(q_pos, q_neg, d_pos, d_neg)
        all_results.append(np.argsort(scores)[::-1][:k])
    query_time = (time.perf_counter() - start) * 1000
    
    # Self-retrieval
    n_test = min(100, n_docs)
    self_hits = 0
    for i in range(n_test):
        q_pos, q_neg = pack_ternary(ternary_data[i])
        scores = ops.batch_similarity(q_pos, q_neg, d_pos, d_neg)
        if np.argmax(scores) == i:
            self_hits += 1
    recall_1 = self_hits / n_test
    
    hits_10 = 0
    for i in range(n_test):
        q_pos, q_neg = pack_ternary(ternary_data[i])
        scores = ops.batch_similarity(q_pos, q_neg, d_pos, d_neg)
        top_10 = np.argsort(scores)[::-1][:10]
        if i in top_10:
            hits_10 += 1
    recall_10 = hits_10 / n_test
    
    memory_mb = (d_pos.nbytes + d_neg.nbytes) / 1024 / 1024
    
    return BenchmarkResult(
        name=f"DB Cooper ({ops.simd})",
        index_time_ms=index_time,
        query_time_ms=query_time,
        queries_per_sec=len(queries) / (query_time / 1000),
        recall_at_1=recall_1,
        recall_at_10=recall_10,
        memory_estimate_mb=memory_mb,
    )


def print_comparison(results: List[BenchmarkResult], n_docs: int, dims: int, n_queries: int):
    """Print comparison table."""
    print("\n" + "=" * 100)
    print(f"DB COOPER vs QDRANT: {n_docs:,} docs, {dims} dims, {n_queries} queries")
    print("=" * 100)
    
    print(f"\n{'Method':<25} {'Index(ms)':<12} {'Query(ms)':<12} {'QPS':<12} "
          f"{'Self-R@1':<10} {'Self-R@10':<10} {'Memory':<10}")
    print("-" * 100)
    
    for r in results:
        if r is None:
            continue
        print(f"{r.name:<25} {r.index_time_ms:<12.1f} {r.query_time_ms:<12.1f} "
              f"{r.queries_per_sec:<12.0f} {r.recall_at_1:<10.1%} {r.recall_at_10:<10.1%} "
              f"{r.memory_estimate_mb:<10.1f} MB")
    
    print("=" * 100)
    
    # Analysis
    cooper = next((r for r in results if r and "Cooper" in r.name), None)
    qdrant = next((r for r in results if r and "Qdrant (in-memory)" in r.name), None)
    qdrant_q = next((r for r in results if r and "INT8" in r.name), None)
    
    if cooper and qdrant:
        print(f"\n📊 DB COOPER vs QDRANT (in-memory):")
        speedup = cooper.queries_per_sec / qdrant.queries_per_sec
        mem_ratio = qdrant.memory_estimate_mb / cooper.memory_estimate_mb
        print(f"   Speed:  {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")
        print(f"   Memory: {mem_ratio:.0f}x smaller")
        print(f"   Index:  {qdrant.index_time_ms/cooper.index_time_ms:.1f}x {'faster' if cooper.index_time_ms < qdrant.index_time_ms else 'slower'}")
    
    if cooper and qdrant_q:
        print(f"\n📊 DB COOPER vs QDRANT (INT8 quantized):")
        speedup = cooper.queries_per_sec / qdrant_q.queries_per_sec
        mem_ratio = qdrant_q.memory_estimate_mb / cooper.memory_estimate_mb
        print(f"   Speed:  {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")
        print(f"   Memory: {mem_ratio:.1f}x {'smaller' if mem_ratio > 1 else 'larger'}")


def run_comparison(n_docs: int = 10000, dims: int = 384, n_queries: int = 100):
    """Run head-to-head comparison."""
    print("\n" + "#" * 100)
    print("#" + " " * 35 + "DB COOPER vs QDRANT" + " " * 44 + "#")
    print("#" * 100)
    
    print(f"\nGenerating test data: {n_docs:,} documents, {dims} dimensions...")
    np.random.seed(42)
    
    # Clustered data for realistic scenario
    n_clusters = 50
    centers = np.random.randn(n_clusters, dims).astype(np.float32)
    data = []
    for i in range(n_docs):
        cluster = i % n_clusters
        point = centers[cluster] + 0.5 * np.random.randn(dims)
        data.append(point)
    data = np.array(data, dtype=np.float32)
    
    # Normalize for cosine similarity
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    
    queries = data[np.random.choice(n_docs, n_queries, replace=False)]
    
    print("Running benchmarks...")
    results = []
    
    print("  [1/3] Qdrant (in-memory, float32)...")
    results.append(benchmark_qdrant(data, queries))
    
    print("  [2/3] Qdrant (in-memory, INT8 quantized)...")
    try:
        results.append(benchmark_qdrant_quantized(data, queries))
    except Exception as e:
        print(f"        Skipped: {e}")
    
    print("  [3/3] DB Cooper (NEON)...")
    results.append(benchmark_db_cooper(data, queries))
    
    print_comparison(results, n_docs, dims, n_queries)
    
    # Verdict
    cooper = next((r for r in results if r and "Cooper" in r.name), None)
    qdrant = next((r for r in results if r and "in-memory" in r.name and "INT8" not in r.name), None)
    
    if cooper and qdrant:
        print("\n" + "=" * 100)
        print("THE VERDICT")
        print("=" * 100)
        
        speedup = cooper.queries_per_sec / qdrant.queries_per_sec
        mem_ratio = qdrant.memory_estimate_mb / cooper.memory_estimate_mb
        
        print(f"""
DB Cooper vs Qdrant:

  SPEED:   DB Cooper is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}
           ({cooper.queries_per_sec:,.0f} vs {qdrant.queries_per_sec:,.0f} QPS)

  MEMORY:  DB Cooper is {mem_ratio:.0f}x smaller
           ({cooper.memory_estimate_mb:.1f} MB vs {qdrant.memory_estimate_mb:.1f} MB)

  RECALL:  Both achieve {min(cooper.recall_at_1, qdrant.recall_at_1):.0%}+ self-retrieval

  UNIQUE TO DB COOPER:
    ✓ Glassbox explainability (see why documents match)
    ✓ Multi-resolution search (exact/similar/context)
    ✓ Ternary structure (2-bit, interpretable)
    ✓ No server required (pure library)
""")


def run_scale_comparison():
    """Compare at multiple scales."""
    print("\n" + "#" * 100)
    print("#" + " " * 35 + "SCALE COMPARISON" + " " * 47 + "#")
    print("#" * 100)
    
    scales = [
        (1000, 128),
        (5000, 256),
        (10000, 384),
        (25000, 384),
    ]
    
    results_table = []
    
    for n_docs, dims in scales:
        print(f"\n{'='*60}")
        print(f"Scale: {n_docs:,} docs, {dims} dims")
        print('='*60)
        
        np.random.seed(42)
        data = np.random.randn(n_docs, dims).astype(np.float32)
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        queries = data[:100]
        
        # Qdrant
        try:
            qdrant = benchmark_qdrant(data, queries)
            print(f"  Qdrant:     {qdrant.queries_per_sec:>6,.0f} QPS, {qdrant.memory_estimate_mb:>5.1f} MB")
        except Exception as e:
            print(f"  Qdrant:     Error - {e}")
            qdrant = None
        
        # DB Cooper
        cooper = benchmark_db_cooper(data, queries)
        if cooper:
            print(f"  DB Cooper:  {cooper.queries_per_sec:>6,.0f} QPS, {cooper.memory_estimate_mb:>5.1f} MB")
        
        if cooper and qdrant:
            speedup = cooper.queries_per_sec / qdrant.queries_per_sec
            print(f"  → DB Cooper is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")
            results_table.append((n_docs, dims, speedup))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: DB Cooper speedup vs Qdrant")
    print("=" * 60)
    for n_docs, dims, speedup in results_table:
        bar = "█" * int(speedup * 2)
        print(f"  {n_docs:>6,} docs, {dims}d: {speedup:>5.1f}x {bar}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=10000)
    parser.add_argument("--dims", type=int, default=384)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--scale", action="store_true")
    
    args = parser.parse_args()
    
    if args.scale:
        run_scale_comparison()
    else:
        run_comparison(args.docs, args.dims, args.queries)
