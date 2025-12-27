"""
Paper Benchmarks: Cooper vs Qdrant, With/Without Codex

Rigorous evaluation for academic publication.
Metrics: Recall@K, Precision@K, nDCG@K, MRR, Latency, Throughput

Datasets:
- TinyShakespeare: 3,230 chunks (real-world text)
- Synthetic: 10K-100K random vectors (stress test)
"""

import numpy as np
import json
import time
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Add src to path
sys.path.insert(0, '/workspace/ZOR/src')


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    method: str
    dataset: str
    n_vectors: int
    n_queries: int
    
    # Quality metrics
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    precision_at_10: float
    ndcg_at_10: float
    mrr: float
    
    # Performance metrics
    index_time_ms: float
    query_time_ms: float
    queries_per_sec: float
    
    # Codex-specific
    archetypal_coverage: Optional[float] = None  # % of results with archetype
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def compute_dcg(relevances: List[float], k: int) -> float:
    """Compute Discounted Cumulative Gain."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def compute_ndcg(retrieved: List[str], relevant: set, k: int) -> float:
    """Compute Normalized DCG."""
    relevances = [1.0 if doc in relevant else 0.0 for doc in retrieved[:k]]
    dcg = compute_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = compute_dcg(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(retrieved: List[str], relevant: set) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


class PaperBenchmark:
    """Comprehensive benchmark suite for paper."""
    
    def __init__(self, output_dir: str = '/workspace/ZOR/benchmarks/results'):
        self.output_dir = output_dir
        self.results: List[BenchmarkResult] = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_shakespeare(self) -> Tuple[np.ndarray, List[dict]]:
        """Load TinyShakespeare dataset."""
        embeddings = np.load('/tmp/shakespeare_embeddings.npy')
        with open('/tmp/shakespeare_chunks.json') as f:
            chunks = json.load(f)
        return embeddings, chunks
    
    def load_genekeys(self) -> dict:
        """Load Gene Keys data."""
        with open('/tmp/genekeys_complete.json') as f:
            return json.load(f)
    
    def generate_synthetic(self, n: int, dim: int = 384) -> np.ndarray:
        """Generate synthetic embeddings."""
        vectors = np.random.randn(n, dim).astype(np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    def compute_ground_truth(self, vectors: np.ndarray, queries: np.ndarray, 
                            k: int = 10) -> List[set]:
        """Compute brute-force ground truth (returns doc_ids like 'doc_0')."""
        ground_truth = []
        for q in queries:
            scores = np.dot(vectors, q)
            top_k_indices = np.argsort(scores)[-k:]
            top_k = set(f"doc_{i}" for i in top_k_indices)
            ground_truth.append(top_k)
        return ground_truth
    
    def evaluate(self, results: List[List[str]], ground_truth: List[set]) -> dict:
        """Compute all quality metrics."""
        recalls_1, recalls_5, recalls_10 = [], [], []
        precisions_10, ndcgs_10, mrrs = [], [], []
        
        for retrieved, relevant in zip(results, ground_truth):
            retrieved_set = set(retrieved[:10])
            
            # Recall@K
            recalls_1.append(len(set(retrieved[:1]) & relevant) / min(1, len(relevant)))
            recalls_5.append(len(set(retrieved[:5]) & relevant) / min(5, len(relevant)))
            recalls_10.append(len(retrieved_set & relevant) / min(10, len(relevant)))
            
            # Precision@10
            precisions_10.append(len(retrieved_set & relevant) / 10)
            
            # nDCG@10
            ndcgs_10.append(compute_ndcg(retrieved, relevant, 10))
            
            # MRR
            mrrs.append(compute_mrr(retrieved, relevant))
        
        return {
            'recall_at_1': np.mean(recalls_1),
            'recall_at_5': np.mean(recalls_5),
            'recall_at_10': np.mean(recalls_10),
            'precision_at_10': np.mean(precisions_10),
            'ndcg_at_10': np.mean(ndcgs_10),
            'mrr': np.mean(mrrs),
        }
    
    def benchmark_qdrant(self, vectors: np.ndarray, queries: np.ndarray,
                        ground_truth: List[set], dataset_name: str) -> BenchmarkResult:
        """Benchmark Qdrant."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance, PointStruct
        
        n, dim = vectors.shape
        n_queries = len(queries)
        
        # Index
        client = QdrantClient(":memory:")
        t0 = time.time()
        client.create_collection("bench", vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
        points = [PointStruct(id=i, vector=vec.tolist()) for i, vec in enumerate(vectors)]
        client.upsert("bench", points, wait=True)
        index_time = (time.time() - t0) * 1000
        
        # Query
        results = []
        t0 = time.time()
        for q in queries:
            hits = client.query_points("bench", query=q.tolist(), limit=10)
            results.append([f"doc_{h.id}" for h in hits.points])
        query_time = (time.time() - t0) / n_queries * 1000
        
        # Evaluate
        metrics = self.evaluate(results, ground_truth)
        
        return BenchmarkResult(
            method="Qdrant",
            dataset=dataset_name,
            n_vectors=n,
            n_queries=n_queries,
            index_time_ms=index_time,
            query_time_ms=query_time,
            queries_per_sec=1000 / query_time,
            **metrics
        )
    
    def benchmark_cooper_brute(self, vectors: np.ndarray, queries: np.ndarray,
                               ground_truth: List[set], dataset_name: str) -> BenchmarkResult:
        """Benchmark Cooper brute force."""
        from trix.db import OctaveDB
        
        n, dim = vectors.shape
        n_queries = len(queries)
        
        # Index
        t0 = time.time()
        db = OctaveDB(dimensions=dim)
        for i, vec in enumerate(vectors):
            db.add(f"doc_{i}", vec)
        index_time = (time.time() - t0) * 1000
        
        # Query (quality mode = brute force)
        results = []
        t0 = time.time()
        for q in queries:
            hits = db.search(q, top_k=10, mode="quality")
            results.append([h.id for h in hits])
        query_time = (time.time() - t0) / n_queries * 1000
        
        # Evaluate
        metrics = self.evaluate(results, ground_truth)
        
        return BenchmarkResult(
            method="Cooper (brute)",
            dataset=dataset_name,
            n_vectors=n,
            n_queries=n_queries,
            index_time_ms=index_time,
            query_time_ms=query_time,
            queries_per_sec=1000 / query_time,
            **metrics
        )
    
    def benchmark_hsquares(self, vectors: np.ndarray, queries: np.ndarray,
                          ground_truth: List[set], dataset_name: str,
                          n_probes: int = 16) -> BenchmarkResult:
        """Benchmark Hollywood Squares Index."""
        from trix.db import HollywoodSquaresIndex
        
        n, dim = vectors.shape
        n_queries = len(queries)
        
        # Index
        t0 = time.time()
        index = HollywoodSquaresIndex(dimensions=dim, n_tiles=64)
        for i, vec in enumerate(vectors):
            index.add(f"doc_{i}", vec)
        index.build_centroids(n_iterations=10)
        index_time = (time.time() - t0) * 1000
        
        # Query
        results = []
        t0 = time.time()
        for q in queries:
            hits = index.search(q, top_k=10, n_probes=n_probes)
            results.append([h[0] for h in hits])
        query_time = (time.time() - t0) / n_queries * 1000
        
        # Evaluate
        metrics = self.evaluate(results, ground_truth)
        
        return BenchmarkResult(
            method=f"HSquares (n={n_probes})",
            dataset=dataset_name,
            n_vectors=n,
            n_queries=n_queries,
            index_time_ms=index_time,
            query_time_ms=query_time,
            queries_per_sec=1000 / query_time,
            **metrics
        )
    
    def benchmark_codex_squares(self, vectors: np.ndarray, queries: np.ndarray,
                                ground_truth: List[set], dataset_name: str,
                                gk_data: dict, n_probes: int = 16) -> BenchmarkResult:
        """Benchmark Codex Squares Index."""
        from trix.db import CodexSquaresIndex
        from sentence_transformers import SentenceTransformer
        
        n, dim = vectors.shape
        n_queries = len(queries)
        
        # Load model for centroids
        model = SentenceTransformer('all-MiniLM-L6-v2')
        def embedder(texts):
            return model.encode(texts, normalize_embeddings=True)
        
        # Index
        t0 = time.time()
        index = CodexSquaresIndex(gk_data, embedder)
        for i, vec in enumerate(vectors):
            index.add(f"doc_{i}", vec)
        index.build()
        index_time = (time.time() - t0) * 1000
        
        # Query
        results = []
        archetypes_found = 0
        t0 = time.time()
        for q in queries:
            hits = index.search(q, top_k=10, n_probes=n_probes)
            results.append([h.doc_id for h in hits])
            archetypes_found += sum(1 for h in hits if h.gk_number > 0)
        query_time = (time.time() - t0) / n_queries * 1000
        
        # Evaluate
        metrics = self.evaluate(results, ground_truth)
        
        return BenchmarkResult(
            method=f"CodexSquares (n={n_probes})",
            dataset=dataset_name,
            n_vectors=n,
            n_queries=n_queries,
            index_time_ms=index_time,
            query_time_ms=query_time,
            queries_per_sec=1000 / query_time,
            archetypal_coverage=archetypes_found / (n_queries * 10),
            **metrics
        )
    
    def run_shakespeare_benchmark(self):
        """Run full benchmark on Shakespeare dataset."""
        print("=" * 70)
        print("SHAKESPEARE BENCHMARK")
        print("=" * 70)
        
        # Load data
        embeddings, chunks = self.load_shakespeare()
        gk_data = self.load_genekeys()
        
        n, dim = embeddings.shape
        print(f"\nDataset: TinyShakespeare")
        print(f"  Vectors: {n:,}")
        print(f"  Dimensions: {dim}")
        
        # Sample queries
        np.random.seed(42)
        query_idx = np.random.choice(n, min(100, n), replace=False)
        queries = embeddings[query_idx]
        
        # Ground truth
        print("\nComputing ground truth...")
        gt = self.compute_ground_truth(embeddings, queries, k=10)
        
        # Run benchmarks
        print("\nRunning benchmarks...")
        
        # Qdrant
        print("  Qdrant...")
        self.results.append(self.benchmark_qdrant(embeddings, queries, gt, "Shakespeare"))
        
        # Cooper brute force
        print("  Cooper (brute)...")
        self.results.append(self.benchmark_cooper_brute(embeddings, queries, gt, "Shakespeare"))
        
        # Hollywood Squares
        for n_probes in [8, 16]:
            print(f"  HSquares (n={n_probes})...")
            self.results.append(self.benchmark_hsquares(embeddings, queries, gt, "Shakespeare", n_probes))
        
        # Codex Squares
        for n_probes in [8, 16]:
            print(f"  CodexSquares (n={n_probes})...")
            self.results.append(self.benchmark_codex_squares(embeddings, queries, gt, "Shakespeare", gk_data, n_probes))
    
    def run_scale_benchmark(self, sizes: List[int] = [10000, 50000]):
        """Run scale benchmark on synthetic data."""
        print("\n" + "=" * 70)
        print("SCALE BENCHMARK (Synthetic)")
        print("=" * 70)
        
        for n in sizes:
            print(f"\n--- {n:,} vectors ---")
            
            # Generate data
            vectors = self.generate_synthetic(n, 384)
            
            # Sample queries
            np.random.seed(42)
            query_idx = np.random.choice(n, 50, replace=False)
            queries = vectors[query_idx]
            
            # Ground truth
            print("  Computing ground truth...")
            gt = self.compute_ground_truth(vectors, queries, k=10)
            
            # Qdrant
            print("  Qdrant...")
            self.results.append(self.benchmark_qdrant(vectors, queries, gt, f"Synthetic-{n//1000}K"))
            
            # Cooper brute
            print("  Cooper (brute)...")
            self.results.append(self.benchmark_cooper_brute(vectors, queries, gt, f"Synthetic-{n//1000}K"))
            
            # Hollywood Squares
            print("  HSquares (n=16)...")
            self.results.append(self.benchmark_hsquares(vectors, queries, gt, f"Synthetic-{n//1000}K", n_probes=16))
    
    def print_results(self):
        """Print results as formatted table."""
        print("\n" + "=" * 100)
        print("BENCHMARK RESULTS")
        print("=" * 100)
        
        # Group by dataset
        by_dataset = defaultdict(list)
        for r in self.results:
            by_dataset[r.dataset].append(r)
        
        for dataset, results in by_dataset.items():
            print(f"\n{dataset}")
            print("-" * 100)
            
            header = f"{'Method':<25} | {'Recall@10':>10} | {'nDCG@10':>10} | {'MRR':>8} | {'Latency':>10} | {'QPS':>8}"
            print(header)
            print("-" * 100)
            
            for r in results:
                row = f"{r.method:<25} | {r.recall_at_10:>10.2%} | {r.ndcg_at_10:>10.4f} | {r.mrr:>8.4f} | {r.query_time_ms:>8.2f}ms | {r.queries_per_sec:>8.0f}"
                print(row)
    
    def save_results(self):
        """Save results to JSON and text files."""
        # JSON
        json_path = f"{self.output_dir}/paper_benchmark_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        print(f"\nResults saved to: {json_path}")
        
        # Text summary
        txt_path = f"{self.output_dir}/paper_benchmark_{self.timestamp}.txt"
        with open(txt_path, 'w') as f:
            f.write("PAPER BENCHMARK RESULTS\n")
            f.write(f"Generated: {self.timestamp}\n")
            f.write("=" * 100 + "\n\n")
            
            by_dataset = defaultdict(list)
            for r in self.results:
                by_dataset[r.dataset].append(r)
            
            for dataset, results in by_dataset.items():
                f.write(f"\n{dataset}\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Method':<25} | {'Recall@10':>10} | {'nDCG@10':>10} | {'MRR':>8} | {'Latency':>10} | {'QPS':>8}\n")
                f.write("-" * 100 + "\n")
                
                for r in results:
                    f.write(f"{r.method:<25} | {r.recall_at_10:>10.2%} | {r.ndcg_at_10:>10.4f} | {r.mrr:>8.4f} | {r.query_time_ms:>8.2f}ms | {r.queries_per_sec:>8.0f}\n")
        
        print(f"Summary saved to: {txt_path}")


def main():
    """Run all benchmarks."""
    benchmark = PaperBenchmark()
    
    # Shakespeare (real embeddings)
    benchmark.run_shakespeare_benchmark()
    
    # Scale test (synthetic)
    benchmark.run_scale_benchmark([10000])
    
    # Print and save
    benchmark.print_results()
    benchmark.save_results()


if __name__ == "__main__":
    main()
