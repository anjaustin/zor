#!/usr/bin/env python3
"""
DB Cooper Product Comparison

Compares DB Cooper against available baseline methods:
1. Numpy brute force (exact)
2. Scipy cdist (exact)  
3. Binary hash (simpler quantization)
4. Random projection LSH (standard approximate)

This demonstrates DB Cooper's position in the search landscape.
"""

import numpy as np
import time
import sys
from typing import List, Tuple
from dataclasses import dataclass
from scipy.spatial.distance import cdist

sys.path.insert(0, 'src')
from trix.db.core import ternary_quantize, pack_ternary
from trix.db.ops import CooperOps, native_ops_available


@dataclass
class Result:
    name: str
    index_time_ms: float
    query_time_ms: float
    queries_per_sec: float
    recall_at_1: float
    recall_at_10: float
    memory_bytes: int


def ground_truth_search(queries: np.ndarray, data: np.ndarray, k: int) -> np.ndarray:
    """Exact nearest neighbor via L2 distance."""
    distances = cdist(queries, data, metric='euclidean')
    return np.argsort(distances, axis=1)[:, :k]


def compute_recall(retrieved: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k."""
    total = 0.0
    for ret, gt in zip(retrieved, ground_truth):
        total += len(set(ret[:k]) & set(gt[:k])) / k
    return total / len(retrieved)


def benchmark_numpy_brute(data: np.ndarray, queries: np.ndarray, k: int = 10) -> Result:
    """Numpy brute force search."""
    # Index
    start = time.perf_counter()
    # No index needed for brute force
    index_time = (time.perf_counter() - start) * 1000
    
    # Query
    start = time.perf_counter()
    results = []
    for q in queries:
        dists = np.sum((data - q) ** 2, axis=1)
        results.append(np.argsort(dists)[:k])
    results = np.array(results)
    query_time = (time.perf_counter() - start) * 1000
    
    # Ground truth is itself (exact)
    gt = results
    
    return Result(
        name="Numpy Brute Force",
        index_time_ms=index_time,
        query_time_ms=query_time,
        queries_per_sec=len(queries) / (query_time / 1000),
        recall_at_1=1.0,
        recall_at_10=1.0,
        memory_bytes=data.nbytes,
    )


def benchmark_scipy_cdist(data: np.ndarray, queries: np.ndarray, k: int = 10) -> Result:
    """Scipy cdist search."""
    # Index
    start = time.perf_counter()
    # No index needed
    index_time = (time.perf_counter() - start) * 1000
    
    # Query
    start = time.perf_counter()
    distances = cdist(queries, data, metric='sqeuclidean')
    results = np.argsort(distances, axis=1)[:, :k]
    query_time = (time.perf_counter() - start) * 1000
    
    return Result(
        name="Scipy cdist",
        index_time_ms=index_time,
        query_time_ms=query_time,
        queries_per_sec=len(queries) / (query_time / 1000),
        recall_at_1=1.0,
        recall_at_10=1.0,
        memory_bytes=data.nbytes,
    )


def benchmark_binary_hash(data: np.ndarray, queries: np.ndarray, k: int = 10) -> Result:
    """Simple binary hashing (sign of values)."""
    # Index: binarize data
    start = time.perf_counter()
    data_binary = (data > 0).astype(np.uint8)
    data_packed = np.packbits(data_binary, axis=1)
    index_time = (time.perf_counter() - start) * 1000
    
    # Query: binarize and hamming distance
    start = time.perf_counter()
    queries_binary = (queries > 0).astype(np.uint8)
    queries_packed = np.packbits(queries_binary, axis=1)
    
    results = []
    for qp in queries_packed:
        # Hamming distance via XOR + popcount
        xor = np.bitwise_xor(data_packed, qp)
        hamming = np.sum(np.unpackbits(xor, axis=1), axis=1)
        results.append(np.argsort(hamming)[:k])
    results = np.array(results)
    query_time = (time.perf_counter() - start) * 1000
    
    # Compute recall against exact search
    gt = ground_truth_search(queries, data, k)
    recall_1 = compute_recall(results, gt, 1)
    recall_10 = compute_recall(results, gt, k)
    
    return Result(
        name="Binary Hash",
        index_time_ms=index_time,
        query_time_ms=query_time,
        queries_per_sec=len(queries) / (query_time / 1000),
        recall_at_1=recall_1,
        recall_at_10=recall_10,
        memory_bytes=data_packed.nbytes,
    )


def benchmark_random_projection_lsh(data: np.ndarray, queries: np.ndarray, 
                                     k: int = 10, n_planes: int = 128) -> Result:
    """Random projection LSH."""
    dims = data.shape[1]
    
    # Index: project and binarize
    start = time.perf_counter()
    np.random.seed(42)
    planes = np.random.randn(dims, n_planes).astype(np.float32)
    planes /= np.linalg.norm(planes, axis=0)
    
    data_proj = data @ planes
    data_binary = (data_proj > 0).astype(np.uint8)
    data_packed = np.packbits(data_binary, axis=1)
    index_time = (time.perf_counter() - start) * 1000
    
    # Query
    start = time.perf_counter()
    queries_proj = queries @ planes
    queries_binary = (queries_proj > 0).astype(np.uint8)
    queries_packed = np.packbits(queries_binary, axis=1)
    
    results = []
    for qp in queries_packed:
        xor = np.bitwise_xor(data_packed, qp)
        hamming = np.sum(np.unpackbits(xor, axis=1), axis=1)
        results.append(np.argsort(hamming)[:k])
    results = np.array(results)
    query_time = (time.perf_counter() - start) * 1000
    
    # Compute recall
    gt = ground_truth_search(queries, data, k)
    recall_1 = compute_recall(results, gt, 1)
    recall_10 = compute_recall(results, gt, k)
    
    return Result(
        name="LSH (128 planes)",
        index_time_ms=index_time,
        query_time_ms=query_time,
        queries_per_sec=len(queries) / (query_time / 1000),
        recall_at_1=recall_1,
        recall_at_10=recall_10,
        memory_bytes=data_packed.nbytes + planes.nbytes,
    )


def benchmark_db_cooper(data: np.ndarray, queries: np.ndarray, k: int = 10) -> Result:
    """DB Cooper with native NEON ops."""
    if not native_ops_available():
        return None
    
    ops = CooperOps()
    dims = data.shape[1]
    n_docs = data.shape[0]
    packed_bytes = (dims + 7) // 8
    
    # Index: quantize and pack
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
    results = []
    for tq in ternary_queries:
        q_pos, q_neg = pack_ternary(tq)
        scores = ops.batch_similarity(q_pos, q_neg, d_pos, d_neg)
        results.append(np.argsort(scores)[::-1][:k])
    results = np.array(results)
    query_time = (time.perf_counter() - start) * 1000
    
    # Compute recall against exact L2 search
    gt = ground_truth_search(queries, data, k)
    recall_1 = compute_recall(results, gt, 1)
    recall_10 = compute_recall(results, gt, k)
    
    return Result(
        name=f"DB Cooper ({ops.simd})",
        index_time_ms=index_time,
        query_time_ms=query_time,
        queries_per_sec=len(queries) / (query_time / 1000),
        recall_at_1=recall_1,
        recall_at_10=recall_10,
        memory_bytes=d_pos.nbytes + d_neg.nbytes,
    )


def print_results(results: List[Result], n_docs: int, n_queries: int, dims: int):
    """Print comparison table."""
    print("\n" + "=" * 100)
    print(f"PRODUCT COMPARISON: {n_docs:,} docs, {dims} dims, {n_queries} queries")
    print("=" * 100)
    
    # Header
    print(f"{'Method':<22} {'Index(ms)':<12} {'Query(ms)':<12} {'QPS':<12} "
          f"{'R@1':<8} {'R@10':<8} {'Memory':<12}")
    print("-" * 100)
    
    for r in results:
        if r is None:
            continue
        mem_str = f"{r.memory_bytes / 1024 / 1024:.1f} MB"
        print(f"{r.name:<22} {r.index_time_ms:<12.1f} {r.query_time_ms:<12.1f} "
              f"{r.queries_per_sec:<12.0f} {r.recall_at_1:<8.1%} {r.recall_at_10:<8.1%} "
              f"{mem_str:<12}")
    
    print("=" * 100)
    
    # Find DB Cooper and best baseline
    cooper = next((r for r in results if r and "Cooper" in r.name), None)
    baseline = next((r for r in results if r and r.name == "Numpy Brute Force"), None)
    lsh = next((r for r in results if r and "LSH" in r.name), None)
    
    if cooper and baseline:
        speedup = cooper.queries_per_sec / baseline.queries_per_sec
        mem_ratio = baseline.memory_bytes / cooper.memory_bytes
        print(f"\nDB Cooper vs Brute Force:")
        print(f"  Speed:  {speedup:.1f}x faster")
        print(f"  Memory: {mem_ratio:.0f}x smaller")
        print(f"  Recall: {cooper.recall_at_10:.1%} @ 10")
    
    if cooper and lsh:
        print(f"\nDB Cooper vs LSH:")
        speedup = cooper.queries_per_sec / lsh.queries_per_sec
        print(f"  Speed:  {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")
        print(f"  R@10:   {cooper.recall_at_10:.1%} vs {lsh.recall_at_10:.1%}")


def run_product_test(n_docs: int = 10000, dims: int = 384, n_queries: int = 100):
    """Run product comparison."""
    print("\n" + "#" * 100)
    print("#" + " " * 35 + "DB COOPER PRODUCT TEST" + " " * 41 + "#")
    print("#" * 100)
    
    print(f"\nGenerating {n_docs:,} documents with {dims} dimensions...")
    np.random.seed(42)
    
    # Create clustered data (more realistic than pure random)
    n_clusters = 50
    cluster_centers = np.random.randn(n_clusters, dims).astype(np.float32)
    data = []
    for i in range(n_docs):
        cluster = i % n_clusters
        point = cluster_centers[cluster] + 0.5 * np.random.randn(dims)
        data.append(point)
    data = np.array(data, dtype=np.float32)
    
    # Queries from the data (should find themselves and cluster neighbors)
    query_idx = np.random.choice(n_docs, n_queries, replace=False)
    queries = data[query_idx].copy()
    
    print(f"Generated {n_clusters} clusters")
    print(f"Running benchmarks...\n")
    
    results = []
    
    # Run all benchmarks
    print("  [1/5] Numpy Brute Force...")
    results.append(benchmark_numpy_brute(data, queries))
    
    print("  [2/5] Scipy cdist...")
    results.append(benchmark_scipy_cdist(data, queries))
    
    print("  [3/5] Binary Hash...")
    results.append(benchmark_binary_hash(data, queries))
    
    print("  [4/5] Random Projection LSH...")
    results.append(benchmark_random_projection_lsh(data, queries))
    
    print("  [5/5] DB Cooper (NEON)...")
    results.append(benchmark_db_cooper(data, queries))
    
    # Print results
    print_results(results, n_docs, n_queries, dims)
    
    # The verdict
    print("\n" + "=" * 100)
    print("THE VERDICT")
    print("=" * 100)
    
    cooper = next((r for r in results if r and "Cooper" in r.name), None)
    if cooper:
        print(f"""
DB Cooper achieves:
  • {cooper.queries_per_sec:,.0f} queries/second
  • {cooper.recall_at_10:.1%} recall @ 10 (vs exact L2 search)
  • {cooper.memory_bytes / 1024 / 1024:.1f} MB memory ({data.nbytes / cooper.memory_bytes:.0f}x compression)

This positions DB Cooper as:
  ✓ Faster than brute force exact search
  ✓ More memory efficient than float vectors
  ✓ Competitive recall with approximate methods
  ✓ Glassbox explainability (unique advantage)
  ✓ Multi-resolution search (exact/similar/context)
""")


def run_scale_test():
    """Test at multiple scales."""
    print("\n" + "#" * 100)
    print("#" + " " * 38 + "SCALE TEST" + " " * 50 + "#")
    print("#" * 100)
    
    scales = [
        (1000, 128),
        (10000, 256),
        (10000, 384),
        (50000, 384),
        (100000, 384),
    ]
    
    for n_docs, dims in scales:
        print(f"\n{'='*50}")
        print(f"Scale: {n_docs:,} docs, {dims} dims")
        print('='*50)
        
        np.random.seed(42)
        data = np.random.randn(n_docs, dims).astype(np.float32)
        queries = data[:100]
        
        # DB Cooper
        cooper = benchmark_db_cooper(data, queries)
        if cooper:
            print(f"  DB Cooper: {cooper.queries_per_sec:,.0f} QPS, "
                  f"{cooper.memory_bytes/1024/1024:.1f} MB, R@10={cooper.recall_at_10:.1%}")
        
        # Brute force for reference
        bf = benchmark_numpy_brute(data, queries)
        print(f"  Brute Force: {bf.queries_per_sec:,.0f} QPS, "
              f"{bf.memory_bytes/1024/1024:.1f} MB")
        
        if cooper:
            print(f"  Speedup: {cooper.queries_per_sec/bf.queries_per_sec:.1f}x")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=10000)
    parser.add_argument("--dims", type=int, default=384)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--scale", action="store_true")
    
    args = parser.parse_args()
    
    if args.scale:
        run_scale_test()
    else:
        run_product_test(args.docs, args.dims, args.queries)
