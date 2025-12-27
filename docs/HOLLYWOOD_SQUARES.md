# Hollywood Squares Index

**IVF-style vector search with semantic topology**

```
    ┌───┬───┬───┬───┬───┬───┬───┬───┐
    │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │
    ├───┼───┼───┼───┼───┼───┼───┼───┤
    │ 9 │10 │11 │12 │13 │14 │15 │16 │   64 tiles
    ├───┼───┼───┼───┼───┼───┼───┼───┤   K-means centroids
    │17 │18 │19 │20 │21 │22 │23 │24 │   Probe top-n by similarity
    ├───┼───┼───┼───┼───┼───┼───┼───┤
    │25 │26 │27 │28 │29 │30 │31 │32 │   The topology IS the algorithm
    ├───┼───┼───┼───┼───┼───┼───┼───┤
    │33 │34 │35 │36 │37 │38 │39 │40 │
    ├───┼───┼───┼───┼───┼───┼───┼───┤
    │41 │42 │43 │44 │45 │46 │47 │48 │
    ├───┼───┼───┼───┼───┼───┼───┼───┤
    │49 │50 │51 │52 │53 │54 │55 │56 │
    ├───┼───┼───┼───┼───┼───┼───┼───┤
    │57 │58 │59 │60 │61 │62 │63 │64 │
    └───┴───┴───┴───┴───┴───┴───┴───┘
```

## Overview

Hollywood Squares Index is an IVF (Inverted File Index) implementation that partitions vectors into 64 tiles using k-means clustering. Search probes the top-n tiles by centroid similarity, achieving sublinear search time while maintaining high recall.

**Key Insight**: Real embeddings have semantic structure that clustering exploits. Random vectors don't cluster well, but text/image embeddings naturally form neighborhoods.

## Benchmark Results

### Shakespeare (3,230 real embeddings)

| Method | Recall@10 | Latency | QPS | vs Qdrant |
|--------|-----------|---------|-----|-----------|
| Qdrant | 100.0% | 2.74ms | 365 | baseline |
| HSquares (n=8) | 92.8% | 0.24ms | 4,159 | **11.4x faster** |
| HSquares (n=16) | 96.9% | 0.52ms | 1,941 | **5.3x faster** |

### Synthetic (10,000 random vectors)

| Method | Recall@10 | Latency | Notes |
|--------|-----------|---------|-------|
| Qdrant | 100.0% | 7.41ms | |
| HSquares (n=16) | 48.6% | 1.36ms | Random vectors don't cluster |

## Usage

```python
from trix.db import HollywoodSquaresIndex

# Create index
index = HollywoodSquaresIndex(dimensions=384, n_tiles=64)

# Add vectors
for i, embedding in enumerate(embeddings):
    index.add(f"doc_{i}", embedding)

# Build k-means centroids
index.build_centroids(n_iterations=10)

# Search (probe top 16 tiles)
results = index.search(query, top_k=10, n_probes=16)
# Returns: [(doc_id, score), ...]
```

## API Reference

### HollywoodSquaresIndex

```python
class HollywoodSquaresIndex:
    def __init__(self, dimensions: int, n_tiles: int = 64):
        """
        Create Hollywood Squares index.
        
        Args:
            dimensions: Vector dimensionality
            n_tiles: Number of tiles (default 64 = 8x8 grid)
        """
    
    def add(self, doc_id: str, vector: np.ndarray):
        """Add vector to index."""
    
    def build_centroids(self, n_iterations: int = 10):
        """Build k-means centroids and reassign vectors to tiles."""
    
    def search(self, query: np.ndarray, top_k: int = 10, 
               n_probes: int = 8) -> List[Tuple[str, float]]:
        """
        Search by probing top-n tiles.
        
        Args:
            query: Query vector
            top_k: Number of results
            n_probes: Number of tiles to probe (more = higher recall, slower)
        
        Returns:
            List of (doc_id, score) tuples
        """
    
    def search_brute(self, query: np.ndarray, top_k: int = 10):
        """Brute force search (100% recall, slower)."""
    
    def get_stats(self) -> Dict:
        """Get index statistics (tile sizes, etc.)."""
```

## How It Works

1. **Indexing**: Vectors are initially assigned to tile 0
2. **Centroid Building**: K-means clusters vectors into 64 groups
3. **Reassignment**: Vectors move to their nearest centroid tile
4. **Search**: 
   - Compute query similarity to all 64 centroids
   - Probe top-n tiles by centroid similarity
   - Return top-k from probed vectors

## Tuning n_probes

| n_probes | % Data Searched | Expected Recall | Use Case |
|----------|-----------------|-----------------|----------|
| 4 | 6.25% | ~85% | Speed-critical |
| 8 | 12.5% | ~93% | Balanced |
| 16 | 25% | ~97% | Quality-focused |
| 32 | 50% | ~99% | Near-exact |
| 64 | 100% | 100% | Exact (brute force) |

## When to Use

**Good for:**
- Real embeddings (text, images, audio)
- Datasets with semantic structure
- Speed/recall tradeoff acceptable

**Not ideal for:**
- Random vectors (no structure to exploit)
- Exact search required (use brute force)
- Very small datasets (< 1000 vectors, brute force is fast enough)

## Relationship to Hollywood Squares OS

This index is inspired by [Hollywood Squares OS](https://github.com/anjaustin/hollywood-squares-os), a coordination operating system where:

- **64 tiles** = addressable processor field
- **Message passing** = query propagation
- **Topology** = algorithm structure

The key insight: *Structure is meaning. The wiring determines the behavior.*

---

See also: [CODEX_SQUARES.md](./CODEX_SQUARES.md) for Gene Key-infused search.
