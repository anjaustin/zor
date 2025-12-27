#!/usr/bin/env python3
"""
RIGOROUS BENCHMARK: DB Cooper vs Qdrant

This is essential research. Log everything, document everything, track everything.
We NEED our blindspots revealed.

Methodology:
1. Ground truth: Exact brute-force cosine similarity
2. Metrics: Recall@K, Precision@K, nDCG@K, MRR
3. Scales: 1K, 10K, 50K, 100K vectors
4. Dimensions: 128, 384, 768
5. Query sets: 100 queries per scale
6. Statistical rigor: Mean, std, min, max, percentiles

Output: JSON logs + human-readable report
"""

import sys
sys.path.insert(0, '/workspace/ZOR/src')

import numpy as np
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Set
import os

# Create output directory
OUTPUT_DIR = "/workspace/ZOR/benchmarks/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Timestamp for this run
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# METRICS
# ============================================================================

def compute_recall(retrieved: Set[int], relevant: Set[int]) -> float:
    """Recall: fraction of relevant items retrieved."""
    if not relevant:
        return 1.0
    return len(retrieved & relevant) / len(relevant)

def compute_precision(retrieved: Set[int], relevant: Set[int]) -> float:
    """Precision: fraction of retrieved items that are relevant."""
    if not retrieved:
        return 0.0
    return len(retrieved & relevant) / len(retrieved)

def compute_ndcg(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            dcg += 1.0 / np.log2(i + 2)
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def compute_mrr(retrieved: List[int], relevant: Set[int]) -> float:
    """Mean Reciprocal Rank."""
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0

def compute_jaccard(set1: Set[int], set2: Set[int]) -> float:
    """Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_random_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate normalized random vectors."""
    np.random.seed(seed)
    vectors = np.random.randn(n, dim).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors

def get_ground_truth(vectors: np.ndarray, query: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
    """Compute exact ground truth via brute-force cosine similarity."""
    similarities = np.dot(vectors, query)
    top_k_indices = np.argsort(similarities)[::-1][:k]
    top_k_scores = similarities[top_k_indices]
    return top_k_indices.tolist(), top_k_scores.tolist()

# ============================================================================
# BENCHMARK RUNNERS
# ============================================================================

@dataclass
class QueryResult:
    """Result for a single query."""
    query_id: int
    latency_ms: float
    retrieved_ids: List[int]
    retrieved_scores: List[float]
    ground_truth_ids: List[int]
    ground_truth_scores: List[float]
    recall_at_k: float
    precision_at_k: float
    ndcg_at_k: float
    mrr: float
    jaccard: float

@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    system: str
    mode: str
    n_vectors: int
    dimensions: int
    n_queries: int
    top_k: int
    
    # Latency stats (ms)
    latency_mean: float
    latency_std: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_min: float
    latency_max: float
    
    # Quality stats
    recall_mean: float
    recall_std: float
    precision_mean: float
    ndcg_mean: float
    mrr_mean: float
    jaccard_mean: float
    
    # Index stats
    index_time_s: float
    
    # Raw data for analysis
    query_results: List[QueryResult]

def benchmark_cooper(vectors: np.ndarray, queries: np.ndarray, k: int, 
                     mode: str = "quality") -> BenchmarkResult:
    """Benchmark DB Cooper."""
    from trix.db import OctaveDB
    
    n, dim = vectors.shape
    n_queries = len(queries)
    
    # Index
    print(f"    Indexing {n} vectors in Cooper ({mode} mode)...")
    t0 = time.time()
    db = OctaveDB(dimensions=dim)
    for i, vec in enumerate(vectors):
        db.add(f"doc_{i}", vec, store_embedding=(mode == "quality"))
    index_time = time.time() - t0
    print(f"    Indexed in {index_time:.2f}s")
    
    # Query
    print(f"    Running {n_queries} queries...")
    query_results = []
    
    for q_id, query in enumerate(queries):
        # Ground truth
        gt_ids, gt_scores = get_ground_truth(vectors, query, k)
        gt_set = set(gt_ids)
        
        # Cooper search
        t0 = time.time()
        results = db.search(query, top_k=k, mode=mode)
        latency_ms = (time.time() - t0) * 1000
        
        retrieved_ids = [int(r.id.split("_")[1]) for r in results]
        retrieved_scores = [r.score for r in results]
        retrieved_set = set(retrieved_ids)
        
        # Metrics
        qr = QueryResult(
            query_id=q_id,
            latency_ms=latency_ms,
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            ground_truth_ids=gt_ids,
            ground_truth_scores=gt_scores,
            recall_at_k=compute_recall(retrieved_set, gt_set),
            precision_at_k=compute_precision(retrieved_set, gt_set),
            ndcg_at_k=compute_ndcg(retrieved_ids, gt_set, k),
            mrr=compute_mrr(retrieved_ids, gt_set),
            jaccard=compute_jaccard(retrieved_set, gt_set),
        )
        query_results.append(qr)
    
    # Aggregate
    latencies = [qr.latency_ms for qr in query_results]
    recalls = [qr.recall_at_k for qr in query_results]
    precisions = [qr.precision_at_k for qr in query_results]
    ndcgs = [qr.ndcg_at_k for qr in query_results]
    mrrs = [qr.mrr for qr in query_results]
    jaccards = [qr.jaccard for qr in query_results]
    
    return BenchmarkResult(
        system="cooper",
        mode=mode,
        n_vectors=n,
        dimensions=dim,
        n_queries=n_queries,
        top_k=k,
        latency_mean=np.mean(latencies),
        latency_std=np.std(latencies),
        latency_p50=np.percentile(latencies, 50),
        latency_p95=np.percentile(latencies, 95),
        latency_p99=np.percentile(latencies, 99),
        latency_min=np.min(latencies),
        latency_max=np.max(latencies),
        recall_mean=np.mean(recalls),
        recall_std=np.std(recalls),
        precision_mean=np.mean(precisions),
        ndcg_mean=np.mean(ndcgs),
        mrr_mean=np.mean(mrrs),
        jaccard_mean=np.mean(jaccards),
        index_time_s=index_time,
        query_results=query_results,
    )

def benchmark_qdrant(vectors: np.ndarray, queries: np.ndarray, k: int) -> BenchmarkResult:
    """Benchmark Qdrant."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    
    n, dim = vectors.shape
    n_queries = len(queries)
    
    # Index
    print(f"    Indexing {n} vectors in Qdrant...")
    t0 = time.time()
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="benchmark",
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    points = [PointStruct(id=i, vector=vec.tolist()) for i, vec in enumerate(vectors)]
    client.upsert(collection_name="benchmark", points=points)
    index_time = time.time() - t0
    print(f"    Indexed in {index_time:.2f}s")
    
    # Query
    print(f"    Running {n_queries} queries...")
    query_results = []
    
    for q_id, query in enumerate(queries):
        # Ground truth
        gt_ids, gt_scores = get_ground_truth(vectors, query, k)
        gt_set = set(gt_ids)
        
        # Qdrant search
        t0 = time.time()
        results = client.query_points(
            collection_name="benchmark",
            query=query.tolist(),
            limit=k
        ).points
        latency_ms = (time.time() - t0) * 1000
        
        retrieved_ids = [r.id for r in results]
        retrieved_scores = [r.score for r in results]
        retrieved_set = set(retrieved_ids)
        
        # Metrics
        qr = QueryResult(
            query_id=q_id,
            latency_ms=latency_ms,
            retrieved_ids=retrieved_ids,
            retrieved_scores=retrieved_scores,
            ground_truth_ids=gt_ids,
            ground_truth_scores=gt_scores,
            recall_at_k=compute_recall(retrieved_set, gt_set),
            precision_at_k=compute_precision(retrieved_set, gt_set),
            ndcg_at_k=compute_ndcg(retrieved_ids, gt_set, k),
            mrr=compute_mrr(retrieved_ids, gt_set),
            jaccard=compute_jaccard(retrieved_set, gt_set),
        )
        query_results.append(qr)
    
    # Aggregate
    latencies = [qr.latency_ms for qr in query_results]
    recalls = [qr.recall_at_k for qr in query_results]
    precisions = [qr.precision_at_k for qr in query_results]
    ndcgs = [qr.ndcg_at_k for qr in query_results]
    mrrs = [qr.mrr for qr in query_results]
    jaccards = [qr.jaccard for qr in query_results]
    
    return BenchmarkResult(
        system="qdrant",
        mode="default",
        n_vectors=n,
        dimensions=dim,
        n_queries=n_queries,
        top_k=k,
        latency_mean=np.mean(latencies),
        latency_std=np.std(latencies),
        latency_p50=np.percentile(latencies, 50),
        latency_p95=np.percentile(latencies, 95),
        latency_p99=np.percentile(latencies, 99),
        latency_min=np.min(latencies),
        latency_max=np.max(latencies),
        recall_mean=np.mean(recalls),
        recall_std=np.std(recalls),
        precision_mean=np.mean(precisions),
        ndcg_mean=np.mean(ndcgs),
        mrr_mean=np.mean(mrrs),
        jaccard_mean=np.mean(jaccards),
        index_time_s=index_time,
        query_results=query_results,
    )

# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(results: List[BenchmarkResult], output_path: str):
    """Generate human-readable report."""
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RIGOROUS BENCHMARK: DB Cooper vs Qdrant\n")
        f.write(f"Run ID: {RUN_ID}\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        
        # Group by scale
        scales = sorted(set(r.n_vectors for r in results))
        
        for n in scales:
            f.write(f"\n{'='*80}\n")
            f.write(f"SCALE: {n:,} vectors\n")
            f.write(f"{'='*80}\n\n")
            
            scale_results = [r for r in results if r.n_vectors == n]
            
            # Table header
            f.write(f"{'System':<20} | {'Recall@10':>10} | {'nDCG@10':>10} | {'P50 (ms)':>10} | {'P95 (ms)':>10} | {'Index (s)':>10}\n")
            f.write("-" * 80 + "\n")
            
            for r in scale_results:
                label = f"{r.system} ({r.mode})"
                f.write(f"{label:<20} | {r.recall_mean:>10.2%} | {r.ndcg_mean:>10.4f} | {r.latency_p50:>10.2f} | {r.latency_p95:>10.2f} | {r.index_time_s:>10.2f}\n")
            
            f.write("\n")
            
            # Detailed stats
            f.write("DETAILED STATISTICS:\n")
            f.write("-" * 40 + "\n")
            
            for r in scale_results:
                f.write(f"\n{r.system} ({r.mode}):\n")
                f.write(f"  Latency: mean={r.latency_mean:.2f}ms, std={r.latency_std:.2f}ms, min={r.latency_min:.2f}ms, max={r.latency_max:.2f}ms\n")
                f.write(f"  Recall@{r.top_k}: mean={r.recall_mean:.4f}, std={r.recall_std:.4f}\n")
                f.write(f"  Precision@{r.top_k}: {r.precision_mean:.4f}\n")
                f.write(f"  nDCG@{r.top_k}: {r.ndcg_mean:.4f}\n")
                f.write(f"  MRR: {r.mrr_mean:.4f}\n")
                f.write(f"  Jaccard: {r.jaccard_mean:.4f}\n")
            
            # Head-to-head comparison
            cooper_results = [r for r in scale_results if r.system == "cooper"]
            qdrant_results = [r for r in scale_results if r.system == "qdrant"]
            
            if cooper_results and qdrant_results:
                f.write("\nHEAD-TO-HEAD:\n")
                f.write("-" * 40 + "\n")
                
                for cr in cooper_results:
                    qr = qdrant_results[0]
                    
                    speedup = qr.latency_p50 / cr.latency_p50 if cr.latency_p50 > 0 else 0
                    recall_diff = cr.recall_mean - qr.recall_mean
                    
                    f.write(f"\nCooper ({cr.mode}) vs Qdrant:\n")
                    f.write(f"  Speedup: {speedup:.2f}x {'(Cooper faster)' if speedup > 1 else '(Qdrant faster)'}\n")
                    f.write(f"  Recall difference: {recall_diff:+.4f} {'(Cooper better)' if recall_diff > 0 else '(Qdrant better)' if recall_diff < 0 else '(tied)'}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

def save_results_json(results: List[BenchmarkResult], output_path: str):
    """Save results as JSON for further analysis."""
    # Convert to serializable format
    data = {
        "run_id": RUN_ID,
        "timestamp": datetime.now().isoformat(),
        "results": []
    }
    
    for r in results:
        result_dict = asdict(r)
        # Convert query_results to simpler format (exclude raw data for size)
        result_dict["query_results"] = [
            {
                "query_id": qr.query_id,
                "latency_ms": qr.latency_ms,
                "recall_at_k": qr.recall_at_k,
                "precision_at_k": qr.precision_at_k,
                "ndcg_at_k": qr.ndcg_at_k,
                "mrr": qr.mrr,
                "jaccard": qr.jaccard,
            }
            for qr in r.query_results
        ]
        data["results"].append(result_dict)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("RIGOROUS BENCHMARK: DB Cooper vs Qdrant")
    print(f"Run ID: {RUN_ID}")
    print("=" * 80)
    
    # Configuration
    SCALES = [1000, 10000, 50000]  # Start conservative, can add 100K
    DIMENSIONS = 384  # Common embedding dimension
    N_QUERIES = 100
    TOP_K = 10
    SEED = 42
    
    all_results = []
    
    for n in SCALES:
        print(f"\n{'='*80}")
        print(f"BENCHMARK: {n:,} vectors, {DIMENSIONS} dimensions")
        print(f"{'='*80}")
        
        # Generate data
        print(f"\n  Generating {n:,} vectors...")
        vectors = generate_random_vectors(n, DIMENSIONS, SEED)
        queries = generate_random_vectors(N_QUERIES, DIMENSIONS, SEED + 1)
        
        # Benchmark Cooper (quality mode)
        print(f"\n  [Cooper - quality mode]")
        try:
            result = benchmark_cooper(vectors, queries, TOP_K, mode="quality")
            all_results.append(result)
            print(f"    Recall@{TOP_K}: {result.recall_mean:.2%}")
            print(f"    Latency P50: {result.latency_p50:.2f}ms")
        except Exception as e:
            print(f"    ERROR: {e}")
        
        # Benchmark Cooper (similar mode)
        print(f"\n  [Cooper - similar mode]")
        try:
            result = benchmark_cooper(vectors, queries, TOP_K, mode="similar")
            all_results.append(result)
            print(f"    Recall@{TOP_K}: {result.recall_mean:.2%}")
            print(f"    Latency P50: {result.latency_p50:.2f}ms")
        except Exception as e:
            print(f"    ERROR: {e}")
        
        # Benchmark Qdrant
        print(f"\n  [Qdrant]")
        try:
            result = benchmark_qdrant(vectors, queries, TOP_K)
            all_results.append(result)
            print(f"    Recall@{TOP_K}: {result.recall_mean:.2%}")
            print(f"    Latency P50: {result.latency_p50:.2f}ms")
        except Exception as e:
            print(f"    ERROR: {e}")
    
    # Generate outputs
    print(f"\n{'='*80}")
    print("GENERATING REPORTS")
    print(f"{'='*80}")
    
    report_path = f"{OUTPUT_DIR}/benchmark_{RUN_ID}.txt"
    json_path = f"{OUTPUT_DIR}/benchmark_{RUN_ID}.json"
    
    generate_report(all_results, report_path)
    print(f"  Report: {report_path}")
    
    save_results_json(all_results, json_path)
    print(f"  JSON: {json_path}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    for n in SCALES:
        scale_results = [r for r in all_results if r.n_vectors == n]
        if not scale_results:
            continue
            
        print(f"\n{n:,} vectors:")
        for r in scale_results:
            print(f"  {r.system:8} ({r.mode:8}): Recall={r.recall_mean:.2%}, P50={r.latency_p50:.1f}ms")
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
