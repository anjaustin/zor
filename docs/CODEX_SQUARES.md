# Codex Squares Index

**Hollywood Squares + Gene Keys: Search with Archetypal Meaning**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Standard IVF:   Vectors → K-Means Centroids → Nearest Cluster → Search     │
│                                                                             │
│  CodexSquares:   Vectors → Gene Key Archetypes → Meaning Cluster → Search   │
│                                                                             │
│  Same algorithm. Different centroids. Meaning built into the structure.     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Overview

Codex Squares replaces k-means centroids with the 64 Gene Keys—archetypal patterns of human experience. Every search result now carries its archetypal context: the Shadow (challenge), Gift (potential), and Siddhi (highest expression).

**The insight**: Instead of arbitrary cluster centers, use meaningful semantic anchors. The same IVF algorithm, but results carry *meaning*.

## Benchmark Results

### Shakespeare (3,230 embeddings)

| Method | Recall@10 | Latency | QPS | Archetypes |
|--------|-----------|---------|-----|------------|
| Qdrant | 100.0% | 2.74ms | 365 | ❌ |
| HSquares (n=16) | 96.9% | 0.52ms | 1,941 | ❌ |
| CodexSquares (n=16) | 94.5% | 0.75ms | 1,342 | ✅ |

### Shakespeare's Archetypal Distribution

| Gene Key | Gift | Shadow | % of Corpus |
|----------|------|--------|-------------|
| GK29 | Commitment | Half-Heartedness | 25% |
| GK31 | Leadership | Arrogance | 16% |
| GK22 | Graciousness | Dishonour | 15% |
| GK36 | Humanity | Turbulence | 8% |
| GK38 | Perseverance | Struggle | 4% |

## Usage

```python
from trix.db import CodexSquaresIndex, create_codex_squares
from sentence_transformers import SentenceTransformer
import json

# Load Gene Keys data
with open('genekeys_complete.json') as f:
    gk_data = json.load(f)

# Create embedder
model = SentenceTransformer('all-MiniLM-L6-v2')
def embedder(texts):
    return model.encode(texts, normalize_embeddings=True)

# Create index
index = create_codex_squares(gk_data, embedder)

# Add vectors
for i, embedding in enumerate(embeddings):
    index.add(f"doc_{i}", embedding)
index.build()

# Search with archetypal context
results = index.search(query, top_k=10, n_probes=16)

for result in results:
    print(f"{result.doc_id}: {result.score:.3f}")
    print(f"  GK{result.gk_number}: {result.shadow} → {result.gift} → {result.siddhi}")
    print(f"  {index.explain_result(result)}")
```

## API Reference

### CodexSquaresIndex

```python
class CodexSquaresIndex:
    def __init__(self, gk_data: dict, embedder=None):
        """
        Create Codex Squares index.
        
        Args:
            gk_data: Gene Keys dictionary with 'gene_keys', 'partners', 'codon_rings'
            embedder: Function that takes list of strings, returns embeddings
        """
    
    def add(self, doc_id: str, vector: np.ndarray):
        """Add vector to nearest Gene Key tile."""
    
    def build(self):
        """Build contiguous storage for fast search."""
    
    def search(self, query: np.ndarray, top_k: int = 10,
               n_probes: int = 8) -> List[CodexResult]:
        """
        Search with archetypal context.
        
        Returns:
            List of CodexResult objects with doc_id, score, and Gene Key info
        """
    
    def get_tile_distribution(self) -> Dict[int, int]:
        """Get document count per Gene Key tile."""
    
    def explain_result(self, result: CodexResult) -> str:
        """Generate archetypal explanation for a result."""
```

### CodexResult

```python
@dataclass
class CodexResult:
    doc_id: str      # Document identifier
    score: float     # Similarity score
    gk_number: int   # Gene Key number (1-64)
    shadow: str      # Shadow aspect (challenge)
    gift: str        # Gift aspect (potential)
    siddhi: str      # Siddhi aspect (highest expression)
```

### CodexTile

```python
@dataclass
class CodexTile:
    gk_number: int   # Gene Key number
    shadow: str      # Shadow name
    gift: str        # Gift name
    siddhi: str      # Siddhi name
    centroid: np.ndarray  # Semantic embedding
    neighbors: Set[int]   # Connected tiles (partners + rings)
```

## Semantic Query Examples

| Query | Archetype | Transformation Path |
|-------|-----------|---------------------|
| "love and devotion" | GK29 | Half-Heartedness → Commitment → Devotion |
| "death and dying" | GK40 | Exhaustion → Resolve → Divine Will |
| "power and throne" | GK34 | Force → Strength → Majesty |
| "war and battle" | GK6 | Conflict → Diplomacy → Peace |

## The 64 Gene Keys as Tiles

Each Gene Key represents a specific archetypal transformation:

| GK | Shadow | Gift | Siddhi |
|----|--------|------|--------|
| 1 | Entropy | Freshness | Beauty |
| 2 | Dislocation | Orientation | Unity |
| 3 | Chaos | Innovation | Innocence |
| ... | ... | ... | ... |
| 38 | Struggle | Perseverance | Honour |
| 39 | Provocation | Dynamism | Liberation |
| ... | ... | ... | ... |
| 64 | Confusion | Imagination | Illumination |

## Topology

CodexSquares preserves the Gene Keys topology:

- **32 Programming Partner pairs**: GK38 ↔ GK39 (bidirectional edges)
- **22 Codon Rings**: Groups of Gene Keys sharing DNA codon patterns
- **102 total edges** connecting the 64 tiles

This topology can be used for:
- Graph-based search expansion
- Archetypal relationship discovery
- Meaning propagation

## When to Use

**Use CodexSquares when:**
- Human experience is the domain (psychology, literature, personal development)
- Archetypal meaning adds value to search results
- Users benefit from transformation context (Shadow → Gift → Siddhi)

**Use plain HollywoodSquares when:**
- Technical domains (code, documentation, APIs)
- Archetypal framing is irrelevant
- Maximum speed is required

## Data Requirements

CodexSquares requires Gene Keys data in this format:

```json
{
  "gene_keys": {
    "1": {"shadow": "Entropy", "gift": "Freshness", "siddhi": "Beauty"},
    "2": {"shadow": "Dislocation", "gift": "Orientation", "siddhi": "Unity"},
    ...
  },
  "partners": {
    "1": 2,
    "38": 39,
    ...
  },
  "codon_rings": {
    "Ring of Fire": [1, 14, 34, 43],
    "Ring of Humanity": [10, 17, 21, 25, 38, 51],
    ...
  }
}
```

---

See also:
- [HOLLYWOOD_SQUARES.md](./HOLLYWOOD_SQUARES.md) for plain IVF search
- [CODEX.md](./CODEX.md) for keyword-based Codex layer
