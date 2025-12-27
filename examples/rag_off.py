#!/usr/bin/env python3
"""
RAG-OFF: DB Cooper vs Qdrant

A retrieval benchmark with semantic structure.
Not random vectors - documents with meaning.
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple
import sys

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# DB Cooper
sys.path.insert(0, 'src')
from trix.db import OctaveDB


@dataclass
class Document:
    id: str
    text: str
    topic: str
    embedding: np.ndarray


@dataclass
class Query:
    text: str
    relevant_topic: str
    embedding: np.ndarray


def create_rag_corpus(n_docs_per_topic: int = 50, dims: int = 256) -> Tuple[List[Document], Dict]:
    """
    Create a corpus with semantic structure.
    
    Topics have distinct embedding patterns.
    Documents within a topic are similar.
    Documents across topics are different.
    """
    np.random.seed(42)
    
    # Topic definitions with base embeddings
    topics = {
        'python': {
            'docs': [
                "Python is a high-level programming language",
                "Django is a Python web framework",
                "NumPy provides numerical computing in Python",
                "Flask is a lightweight Python web framework",
                "Pandas is used for data analysis in Python",
                "Python supports object-oriented programming",
                "List comprehensions are a Python feature",
                "Python has dynamic typing",
                "Pip is the Python package manager",
                "Virtual environments isolate Python projects",
            ],
            'base': None,  # Will be generated
        },
        'javascript': {
            'docs': [
                "JavaScript runs in web browsers",
                "React is a JavaScript UI library",
                "Node.js runs JavaScript on servers",
                "TypeScript adds types to JavaScript",
                "NPM manages JavaScript packages",
                "JavaScript uses prototype-based inheritance",
                "Async/await handles asynchronous JavaScript",
                "Vue.js is a JavaScript framework",
                "Webpack bundles JavaScript modules",
                "JavaScript has first-class functions",
            ],
            'base': None,
        },
        'databases': {
            'docs': [
                "PostgreSQL is a relational database",
                "MongoDB stores documents as JSON",
                "Redis is an in-memory key-value store",
                "SQL queries relational databases",
                "Indexes speed up database queries",
                "ACID ensures database transactions",
                "Sharding distributes database load",
                "Database normalization reduces redundancy",
                "Foreign keys enforce referential integrity",
                "Database migrations update schemas",
            ],
            'base': None,
        },
        'machine_learning': {
            'docs': [
                "Neural networks learn from data",
                "Gradient descent optimizes model weights",
                "Transformers use attention mechanisms",
                "CNNs process image data",
                "RNNs handle sequential data",
                "Backpropagation computes gradients",
                "Regularization prevents overfitting",
                "Embeddings represent words as vectors",
                "Loss functions measure prediction error",
                "Batch normalization stabilizes training",
            ],
            'base': None,
        },
        'devops': {
            'docs': [
                "Docker containers package applications",
                "Kubernetes orchestrates containers",
                "CI/CD automates deployment pipelines",
                "Infrastructure as code manages resources",
                "Monitoring tracks system health",
                "Load balancers distribute traffic",
                "Microservices decompose applications",
                "API gateways route requests",
                "Service mesh manages communication",
                "GitOps uses Git for operations",
            ],
            'base': None,
        },
    }
    
    # Generate orthogonal base vectors for each topic
    n_topics = len(topics)
    topic_dims = dims // n_topics
    
    for i, topic in enumerate(topics.keys()):
        base = np.zeros(dims, dtype=np.float32)
        # Each topic gets a distinct region of the embedding space
        start = i * topic_dims
        end = start + topic_dims
        base[start:end] = np.random.randn(topic_dims)
        base = base / np.linalg.norm(base)
        topics[topic]['base'] = base
    
    # Generate documents
    documents = []
    doc_id = 0
    
    for topic_name, topic_data in topics.items():
        base = topic_data['base']
        
        for i in range(n_docs_per_topic):
            # Text (cycle through examples, add variation)
            text_idx = i % len(topic_data['docs'])
            text = topic_data['docs'][text_idx]
            if i >= len(topic_data['docs']):
                text = f"{text} (variant {i // len(topic_data['docs'])})"
            
            # Embedding: base + small noise (preserves topic similarity)
            noise = 0.3 * np.random.randn(dims).astype(np.float32)
            embedding = base + noise
            embedding = embedding / np.linalg.norm(embedding)
            
            documents.append(Document(
                id=f"doc_{doc_id}",
                text=text,
                topic=topic_name,
                embedding=embedding,
            ))
            doc_id += 1
    
    return documents, topics


def create_queries(topics: Dict, n_queries_per_topic: int = 10, dims: int = 256) -> List[Query]:
    """Create queries that should retrieve specific topics."""
    np.random.seed(123)
    
    query_templates = {
        'python': [
            "How do I use Python for web development?",
            "What is the best Python data science library?",
            "Python programming tutorial",
            "Python package management",
            "Object oriented Python",
        ],
        'javascript': [
            "Frontend JavaScript frameworks",
            "How does Node.js work?",
            "JavaScript async programming",
            "TypeScript vs JavaScript",
            "JavaScript build tools",
        ],
        'databases': [
            "SQL database design",
            "NoSQL vs relational databases",
            "Database performance optimization",
            "How do database indexes work?",
            "Database transaction management",
        ],
        'machine_learning': [
            "Deep learning neural networks",
            "How does backpropagation work?",
            "Natural language processing models",
            "Machine learning optimization",
            "Preventing overfitting in ML",
        ],
        'devops': [
            "Container orchestration platforms",
            "CI/CD pipeline setup",
            "Cloud infrastructure management",
            "Microservices architecture",
            "Application monitoring tools",
        ],
    }
    
    queries = []
    
    for topic_name, base in [(t, topics[t]['base']) for t in topics]:
        for i in range(n_queries_per_topic):
            # Query text
            text_idx = i % len(query_templates[topic_name])
            text = query_templates[topic_name][text_idx]
            
            # Query embedding: close to topic base
            noise = 0.25 * np.random.randn(dims).astype(np.float32)
            embedding = base + noise
            embedding = embedding / np.linalg.norm(embedding)
            
            queries.append(Query(
                text=text,
                relevant_topic=topic_name,
                embedding=embedding,
            ))
    
    return queries


def benchmark_db_cooper(documents: List[Document], queries: List[Query], top_k: int = 10):
    """Benchmark DB Cooper retrieval."""
    dims = len(documents[0].embedding)
    
    # Index
    start = time.perf_counter()
    db = OctaveDB(dimensions=dims)
    for doc in documents:
        db.add(doc.id, doc.embedding, metadata={'topic': doc.topic, 'text': doc.text})
    index_time = time.perf_counter() - start
    
    # Query
    start = time.perf_counter()
    results = []
    for query in queries:
        hits = db.search(query.embedding, mode='similar', top_k=top_k)
        retrieved_topics = [db.get(h.id)['metadata']['topic'] for h in hits]
        results.append((query.relevant_topic, retrieved_topics))
    query_time = time.perf_counter() - start
    
    return results, index_time, query_time


def benchmark_qdrant(documents: List[Document], queries: List[Query], top_k: int = 10):
    """Benchmark Qdrant retrieval."""
    dims = len(documents[0].embedding)
    
    client = QdrantClient(":memory:")
    
    # Index
    start = time.perf_counter()
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=dims, distance=Distance.COSINE),
    )
    
    points = [
        PointStruct(
            id=i,
            vector=doc.embedding.tolist(),
            payload={'topic': doc.topic, 'text': doc.text, 'doc_id': doc.id}
        )
        for i, doc in enumerate(documents)
    ]
    client.upsert(collection_name="test", points=points)
    index_time = time.perf_counter() - start
    
    # Query
    start = time.perf_counter()
    results = []
    for query in queries:
        hits = client.query_points(
            collection_name="test",
            query=query.embedding.tolist(),
            limit=top_k,
        ).points
        retrieved_topics = [h.payload['topic'] for h in hits]
        results.append((query.relevant_topic, retrieved_topics))
    query_time = time.perf_counter() - start
    
    return results, index_time, query_time


def compute_metrics(results: List[Tuple[str, List[str]]], k_values: List[int] = [1, 5, 10]):
    """Compute precision@k and recall@k."""
    metrics = {}
    
    for k in k_values:
        precision_sum = 0
        recall_sum = 0
        
        for relevant_topic, retrieved_topics in results:
            retrieved_k = retrieved_topics[:k]
            hits = sum(1 for t in retrieved_k if t == relevant_topic)
            
            precision_sum += hits / k
            recall_sum += hits / k  # Assuming k relevant docs exist
        
        metrics[f'P@{k}'] = precision_sum / len(results)
        metrics[f'R@{k}'] = recall_sum / len(results)
    
    return metrics


def run_rag_off():
    """Run the RAG-off benchmark."""
    print()
    print("=" * 70)
    print("                         RAG-OFF")
    print("                  DB Cooper vs Qdrant")
    print("=" * 70)
    print()
    
    # Create corpus
    n_docs = 50  # per topic
    dims = 256
    
    print(f"Creating corpus: {n_docs} docs/topic, 5 topics, {dims} dims...")
    documents, topics = create_rag_corpus(n_docs_per_topic=n_docs, dims=dims)
    queries = create_queries(topics, n_queries_per_topic=20, dims=dims)
    
    print(f"Total documents: {len(documents)}")
    print(f"Total queries: {len(queries)}")
    print()
    
    # Benchmark
    print("Running benchmarks...")
    print()
    
    top_k = 10
    
    cooper_results, cooper_index, cooper_query = benchmark_db_cooper(documents, queries, top_k)
    qdrant_results, qdrant_index, qdrant_query = benchmark_qdrant(documents, queries, top_k)
    
    # Compute metrics
    cooper_metrics = compute_metrics(cooper_results)
    qdrant_metrics = compute_metrics(qdrant_results)
    
    # Print results
    print("=" * 70)
    print("                         RESULTS")
    print("=" * 70)
    print()
    
    print(f"{'Metric':<20} {'DB Cooper':<15} {'Qdrant':<15} {'Winner':<15}")
    print("-" * 70)
    
    # Retrieval quality
    for metric in ['P@1', 'P@5', 'P@10', 'R@10']:
        cooper_val = cooper_metrics[metric]
        qdrant_val = qdrant_metrics[metric]
        
        if cooper_val > qdrant_val + 0.01:
            winner = "DB Cooper"
        elif qdrant_val > cooper_val + 0.01:
            winner = "Qdrant"
        else:
            winner = "Tie"
        
        print(f"{metric:<20} {cooper_val:<15.1%} {qdrant_val:<15.1%} {winner:<15}")
    
    print("-" * 70)
    
    # Speed
    cooper_qps = len(queries) / cooper_query
    qdrant_qps = len(queries) / qdrant_query
    
    print(f"{'Index time':<20} {cooper_index*1000:<15.1f}ms {qdrant_index*1000:<15.1f}ms "
          f"{'DB Cooper' if cooper_index < qdrant_index else 'Qdrant':<15}")
    print(f"{'Query time':<20} {cooper_query*1000:<15.1f}ms {qdrant_query*1000:<15.1f}ms "
          f"{'DB Cooper' if cooper_query < qdrant_query else 'Qdrant':<15}")
    print(f"{'Queries/sec':<20} {cooper_qps:<15.0f} {qdrant_qps:<15.0f} "
          f"{'DB Cooper' if cooper_qps > qdrant_qps else 'Qdrant':<15}")
    
    print("=" * 70)
    print()
    
    # Summary
    print("SUMMARY:")
    print()
    
    # Count wins
    cooper_wins = 0
    qdrant_wins = 0
    
    for metric in ['P@1', 'P@5', 'P@10']:
        if cooper_metrics[metric] > qdrant_metrics[metric] + 0.01:
            cooper_wins += 1
        elif qdrant_metrics[metric] > cooper_metrics[metric] + 0.01:
            qdrant_wins += 1
    
    if cooper_qps > qdrant_qps:
        cooper_wins += 1
    else:
        qdrant_wins += 1
    
    print(f"  DB Cooper wins: {cooper_wins}")
    print(f"  Qdrant wins:    {qdrant_wins}")
    print()
    
    if cooper_wins > qdrant_wins:
        print("  🏆 DB COOPER WINS THE RAG-OFF")
    elif qdrant_wins > cooper_wins:
        print("  🏆 QDRANT WINS THE RAG-OFF")
    else:
        print("  🤝 IT'S A TIE")
    
    print()
    print("=" * 70)
    
    # Show some example retrievals
    print()
    print("EXAMPLE RETRIEVALS:")
    print("-" * 70)
    
    # Pick a few queries
    db = OctaveDB(dimensions=dims)
    for doc in documents:
        db.add(doc.id, doc.embedding, metadata={'topic': doc.topic, 'text': doc.text})
    
    for i in [0, 25, 50, 75]:
        query = queries[i]
        hits = db.search(query.embedding, mode='similar', top_k=3)
        
        print(f"\nQuery: \"{query.text}\"")
        print(f"Expected topic: {query.relevant_topic}")
        print("Top 3 results:")
        for h in hits:
            doc = db.get(h.id)
            match = "✓" if doc['metadata']['topic'] == query.relevant_topic else "✗"
            print(f"  {match} [{doc['metadata']['topic']}] {doc['metadata']['text'][:50]}...")


if __name__ == "__main__":
    run_rag_off()
