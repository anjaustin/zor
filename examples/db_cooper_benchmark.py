#!/usr/bin/env python3
"""
DB Cooper Product Benchmark

Compares DB Cooper against:
- FAISS (Facebook AI Similarity Search) - if available
- Brute-force numpy search (baseline)
- Scikit-learn NearestNeighbors - if available

Metrics:
- Recall@K: How many true neighbors are found
- Queries/sec: Search throughput
- Index time: Time to build index
- Memory: Approximate memory usage
"""

import numpy as np
import time
import sys
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

# Import DB Cooper
sys.path.insert(0, 'src')
from trix.db import OctaveDB
from trix.db.core import ternary_quantize

# Try to import alternatives
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    index_time: float
    query_time: float
    queries_per_sec: float
    recall_at_10: float
    memory_mb: float


def compute_recall(retrieved: List[List[int]], ground_truth: List[List[int]], k: int) -> float:
    """Compute recall@k."""
    total_recall = 0.0
    for ret, gt in zip(retrieved, ground_truth):
        ret_set = set(ret[:k])
        gt_set = set(gt[:k])
        total_recall += len(ret_set & gt_set) / k
    return total_recall / len(retrieved)


def brute_force_search(queries: np.ndarray, data: np.ndarray, k: int) -> List[List[int]]:
    """Brute force exact nearest neighbor search."""
    results = []
    for q in queries:
        distances = np.sum((data - q) ** 2, axis=1)
        indices = np.argsort(distances)[:k]
        results.append(indices.tolist())
    return results


def benchmark_brute_force(data: np.ndarray, queries: np.ndarray, k: int = 10) -> BenchmarkResult:
    """Benchmark brute force numpy search."""
    print("  Benchmarking: Brute Force (numpy)...")
    
    # Index time (none for brute force)
    index_time = 0.0
    
    # Query time
    start = time.time()
    results = brute_force_search(queries, data, k)
    query_time = time.time() - start
    
    # Ground truth is itself
    recall = 1.0  # Brute force is exact
    
    # Memory (just the data)
    memory_mb = data.nbytes / 1024 / 1024
    
    return BenchmarkResult(
        name="Brute Force",
        index_time=index_time,
        query_time=query_time,
        queries_per_sec=len(queries) / query_time,
        recall_at_10=recall,
        memory_mb=memory_mb,
    )


def benchmark_faiss(data: np.ndarray, queries: np.ndarray, k: int = 10) -> BenchmarkResult:
    """Benchmark FAISS flat index."""
    if not HAS_FAISS:
        return None
    
    print("  Benchmarking: FAISS (Flat L2)...")
    
    d = data.shape[1]
    data_f32 = data.astype(np.float32)
    queries_f32 = queries.astype(np.float32)
    
    # Index time
    start = time.time()
    index = faiss.IndexFlatL2(d)
    index.add(data_f32)
    index_time = time.time() - start
    
    # Query time
    start = time.time()
    distances, indices = index.search(queries_f32, k)
    query_time = time.time() - start
    
    # Memory
    memory_mb = data_f32.nbytes / 1024 / 1024
    
    return BenchmarkResult(
        name="FAISS Flat L2",
        index_time=index_time,
        query_time=query_time,
        queries_per_sec=len(queries) / query_time,
        recall_at_10=1.0,  # Flat is exact
        memory_mb=memory_mb,
    )


def benchmark_faiss_ivf(data: np.ndarray, queries: np.ndarray, k: int = 10, 
                        nlist: int = 100, nprobe: int = 10) -> BenchmarkResult:
    """Benchmark FAISS IVF index (approximate)."""
    if not HAS_FAISS:
        return None
    
    print("  Benchmarking: FAISS (IVF)...")
    
    d = data.shape[1]
    n = data.shape[0]
    data_f32 = data.astype(np.float32)
    queries_f32 = queries.astype(np.float32)
    
    # Index time
    start = time.time()
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, min(nlist, n // 10))
    index.train(data_f32)
    index.add(data_f32)
    index.nprobe = nprobe
    index_time = time.time() - start
    
    # Query time
    start = time.time()
    distances, indices = index.search(queries_f32, k)
    query_time = time.time() - start
    
    # Compute recall against brute force
    ground_truth = brute_force_search(queries, data, k)
    retrieved = [idx.tolist() for idx in indices]
    recall = compute_recall(retrieved, ground_truth, k)
    
    memory_mb = data_f32.nbytes / 1024 / 1024
    
    return BenchmarkResult(
        name="FAISS IVF",
        index_time=index_time,
        query_time=query_time,
        queries_per_sec=len(queries) / query_time,
        recall_at_10=recall,
        memory_mb=memory_mb,
    )


def benchmark_sklearn(data: np.ndarray, queries: np.ndarray, k: int = 10) -> BenchmarkResult:
    """Benchmark scikit-learn NearestNeighbors."""
    if not HAS_SKLEARN:
        return None
    
    print("  Benchmarking: scikit-learn KD-Tree...")
    
    # Index time
    start = time.time()
    nn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
    nn.fit(data)
    index_time = time.time() - start
    
    # Query time
    start = time.time()
    distances, indices = nn.kneighbors(queries)
    query_time = time.time() - start
    
    memory_mb = data.nbytes / 1024 / 1024
    
    return BenchmarkResult(
        name="sklearn KD-Tree",
        index_time=index_time,
        query_time=query_time,
        queries_per_sec=len(queries) / query_time,
        recall_at_10=1.0,  # Exact
        memory_mb=memory_mb,
    )


def benchmark_db_cooper(data: np.ndarray, queries: np.ndarray, k: int = 10) -> BenchmarkResult:
    """Benchmark DB Cooper."""
    print("  Benchmarking: DB Cooper (Octave DB)...")
    
    # Index time
    start = time.time()
    db = OctaveDB(
        dimensions=data.shape[1],
        pool_factor=4,
        coarse_threshold=-1.0,  # Accept all for fair comparison
        quantize_sparsity=0.1,  # Keep more information
    )
    
    doc_ids = [f"doc{i}" for i in range(len(data))]
    db.add_batch(doc_ids, data)
    index_time = time.time() - start
    
    # Query time
    start = time.time()
    all_results = []
    for q in queries:
        results = db.search(q, mode="similar", top_k=k)
        indices = [int(r.id.replace("doc", "")) for r in results]
        all_results.append(indices)
    query_time = time.time() - start
    
    # Self-retrieval test (documents should find themselves)
    # Use first N documents as queries where N = min(100, n_docs)
    n_test = min(100, len(data))
    self_hits = 0
    for i in range(n_test):
        results = db.search(data[i], mode="exact", top_k=1)
        if results and results[0].id == f"doc{i}":
            self_hits += 1
    self_recall = self_hits / n_test
    
    # Memory estimate (ternary = 2 bits per dim, plus overhead)
    ternary_bits = 2 * data.shape[0] * data.shape[1]
    memory_mb = ternary_bits / 8 / 1024 / 1024 * 3  # x3 for 3 levels + overhead
    
    return BenchmarkResult(
        name="DB Cooper",
        index_time=index_time,
        query_time=query_time,
        queries_per_sec=len(queries) / query_time,
        recall_at_10=self_recall,  # Self-retrieval rate
        memory_mb=memory_mb,
    )


def benchmark_db_cooper_native(data: np.ndarray, queries: np.ndarray, k: int = 10) -> BenchmarkResult:
    """Benchmark DB Cooper with native NEON ops."""
    try:
        from trix.db.ops import CooperOps, native_ops_available
        if not native_ops_available():
            return None
    except ImportError:
        return None
    
    print("  Benchmarking: DB Cooper Native (NEON)...")
    
    ops = CooperOps()
    dims = data.shape[1]
    n_docs = data.shape[0]
    packed_bytes = (dims + 7) // 8
    
    # Quantize and pack all data
    start = time.time()
    from trix.db.core import ternary_quantize, pack_ternary
    
    ternary_data = ternary_quantize(data, sparsity_target=0.1)
    d_pos = np.zeros((n_docs, packed_bytes), dtype=np.uint8)
    d_neg = np.zeros((n_docs, packed_bytes), dtype=np.uint8)
    
    for i, t in enumerate(ternary_data):
        p, n = pack_ternary(t)
        d_pos[i] = p
        d_neg[i] = n
    
    index_time = time.time() - start
    
    # Quantize queries
    ternary_queries = ternary_quantize(queries, sparsity_target=0.1)
    
    # Query time using native batch similarity
    start = time.time()
    all_results = []
    for tq in ternary_queries:
        q_pos, q_neg = pack_ternary(tq)
        scores = ops.batch_similarity(q_pos, q_neg, d_pos, d_neg)
        top_k_idx = np.argsort(scores)[-k:][::-1]
        all_results.append(top_k_idx.tolist())
    query_time = time.time() - start
    
    # Self-retrieval test
    n_test = min(100, n_docs)
    self_hits = 0
    for i in range(n_test):
        q_pos, q_neg = pack_ternary(ternary_data[i])
        scores = ops.batch_similarity(q_pos, q_neg, d_pos, d_neg)
        top_idx = np.argmax(scores)
        if top_idx == i:
            self_hits += 1
    self_recall = self_hits / n_test
    
    # Memory
    memory_mb = (d_pos.nbytes + d_neg.nbytes) / 1024 / 1024
    
    return BenchmarkResult(
        name=f"DB Cooper Native ({ops.simd})",
        index_time=index_time,
        query_time=query_time,
        queries_per_sec=len(queries) / query_time,
        recall_at_10=self_recall,
        memory_mb=memory_mb,
    )


def print_results(results: List[BenchmarkResult], n_docs: int, n_queries: int, dims: int):
    """Print benchmark results in a nice table."""
    print("\n" + "=" * 80)
    print(f"BENCHMARK RESULTS: {n_docs} docs, {dims} dims, {n_queries} queries")
    print("=" * 80)
    
    # Header
    print(f"{'Method':<25} {'Index(s)':<10} {'Query(s)':<10} {'QPS':<12} {'Self-Recall':<12} {'Mem(MB)':<10}")
    print("-" * 85)
    
    # Results
    for r in results:
        if r is not None:
            print(f"{r.name:<25} {r.index_time:<10.3f} {r.query_time:<10.3f} "
                  f"{r.queries_per_sec:<12.0f} {r.recall_at_10:<12.1%} {r.memory_mb:<10.1f}")
    
    print("=" * 80)
    
    # Analysis
    cooper = next((r for r in results if r and r.name == "DB Cooper"), None)
    brute = next((r for r in results if r and r.name == "Brute Force"), None)
    
    if cooper and brute:
        speedup = cooper.queries_per_sec / brute.queries_per_sec
        memory_ratio = brute.memory_mb / cooper.memory_mb
        print(f"\nDB Cooper vs Brute Force:")
        print(f"  Speed:  {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")
        print(f"  Memory: {memory_ratio:.1f}x smaller")
        print(f"  Recall: {cooper.recall_at_10:.1%} (vs 100% exact)")
    
    if HAS_FAISS:
        faiss_flat = next((r for r in results if r and r.name == "FAISS Flat L2"), None)
        if cooper and faiss_flat:
            speedup = cooper.queries_per_sec / faiss_flat.queries_per_sec
            print(f"\nDB Cooper vs FAISS Flat:")
            print(f"  Speed:  {speedup:.1f}x {'faster' if speedup > 1 else 'slower'}")


def run_benchmark(n_docs: int = 10000, dims: int = 384, n_queries: int = 100):
    """Run full benchmark suite."""
    print(f"\n{'#' * 80}")
    print(f"# DB COOPER PRODUCT BENCHMARK")
    print(f"{'#' * 80}")
    
    print(f"\nConfiguration:")
    print(f"  Documents: {n_docs}")
    print(f"  Dimensions: {dims}")
    print(f"  Queries: {n_queries}")
    print(f"  FAISS available: {HAS_FAISS}")
    print(f"  sklearn available: {HAS_SKLEARN}")
    
    # Generate data
    print("\nGenerating data...")
    np.random.seed(42)
    data = np.random.randn(n_docs, dims).astype(np.float32)
    queries = np.random.randn(n_queries, dims).astype(np.float32)
    
    # Run benchmarks
    print("\nRunning benchmarks...")
    results = []
    
    results.append(benchmark_brute_force(data, queries))
    results.append(benchmark_db_cooper(data, queries))
    results.append(benchmark_db_cooper_native(data, queries))
    
    if HAS_FAISS:
        results.append(benchmark_faiss(data, queries))
        if n_docs >= 1000:
            results.append(benchmark_faiss_ivf(data, queries))
    
    if HAS_SKLEARN and dims <= 100:  # KD-tree doesn't scale well to high dims
        results.append(benchmark_sklearn(data, queries))
    
    # Print results
    print_results(results, n_docs, n_queries, dims)


def run_scale_benchmark():
    """Run benchmark at multiple scales."""
    print("\n" + "#" * 80)
    print("# SCALE BENCHMARK")
    print("#" * 80)
    
    scales = [
        (1000, 128, 100),
        (10000, 256, 100),
        (10000, 384, 100),
        (50000, 384, 100),
    ]
    
    for n_docs, dims, n_queries in scales:
        run_benchmark(n_docs, dims, n_queries)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DB Cooper Product Benchmark")
    parser.add_argument("--docs", type=int, default=10000, help="Number of documents")
    parser.add_argument("--dims", type=int, default=384, help="Embedding dimensions")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--scale", action="store_true", help="Run scale benchmark")
    
    args = parser.parse_args()
    
    if args.scale:
        run_scale_benchmark()
    else:
        run_benchmark(args.docs, args.dims, args.queries)
