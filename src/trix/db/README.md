# DB Cooper + Codex Layer

**Octave-quantized retrieval with optional archetypal illumination.**

## Components

### DB Cooper (Core)
Multi-resolution ternary retrieval with glassbox explainability.

```python
from trix.db import OctaveDB, PossibilityExplorer

db = OctaveDB(dimensions=384)
db.add("doc1", embedding)
results = db.search(query_embedding, top_k=10)
```

### Hollywood Squares Index
IVF-style search with 64 tiles and k-means centroids. **11x faster than Qdrant at 93% recall.**

```python
from trix.db import HollywoodSquaresIndex

index = HollywoodSquaresIndex(dimensions=384, n_tiles=64)
for i, vec in enumerate(embeddings):
    index.add(f"doc_{i}", vec)
index.build_centroids(n_iterations=10)

results = index.search(query, top_k=10, n_probes=16)
# Returns: [(doc_id, score), ...]
```

### Codex Squares Index
Hollywood Squares + Gene Keys: every result carries archetypal context.

```python
from trix.db import CodexSquaresIndex, create_codex_squares

index = create_codex_squares(gk_data, embedder)
for i, vec in enumerate(embeddings):
    index.add(f"doc_{i}", vec)
index.build()

results = index.search(query, top_k=10, n_probes=16)
for r in results:
    print(f"{r.doc_id}: GK{r.gk_number} ({r.shadow}→{r.gift}→{r.siddhi})")
```

### Codex Layer (Optional)
Archetypal meaning illumination for human-centered domains.

```python
from trix.db import KeywordCodex, create_codex

codex = create_codex()
activations, explanations = codex.illuminate("I feel stuck in a struggle")

# GK38 (Perseverance): 1.00 via shadow:struggle
# GK39 (Dynamism): 0.80 via partner:38 (propagation)
```

## When to Use Codex

| Domain | Use Codex? | Example |
|--------|------------|---------|
| Human experience | ✓ Yes | "dealing with addiction" |
| Psychology | ✓ Yes | "patterns of control" |
| Relationships | ✓ Yes | "conflict resolution" |
| Technical docs | ✗ No | "nginx configuration" |
| Code | ✗ No | "implement sorting" |
| APIs | ✗ No | "REST endpoints" |

The Codex applies where the **human equation** is in effect.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    COOPER CORE                              │
│              (All domains, always on)                       │
│                                                             │
│   Embed → Quantize → Explore → Aboutness → Results          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼ (optional, human domains)
┌─────────────────────────────────────────────────────────────┐
│                    CODEX LAYER                              │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              KEYWORD CODEX                          │   │
│   │   196 keywords → 64 Gene Keys → Propagation         │   │
│   │   "struggle" → GK38 → Partner GK39 → Ring members   │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   Shadow → Gift → Siddhi transformation paths               │
│   32 Programming Partner pairs                              │
│   21 Codon Rings (amino acid families)                      │
└─────────────────────────────────────────────────────────────┘
```

## Gene Keys Structure

Each of the 64 Gene Keys has:

- **Shadow**: Unconscious pattern (e.g., Struggle)
- **Gift**: Conscious expression (e.g., Perseverance)  
- **Siddhi**: Transcendent state (e.g., Honour)
- **Programming Partner**: Paired archetype (GK38 ↔ GK39)
- **Codon Ring**: Amino acid family (Ring of Humanity)

## API Reference

### KeywordCodex

```python
from trix.db import KeywordCodex

codex = KeywordCodex()

# Illuminate text
activations, explanations = codex.illuminate("your text here")

# Get activation pattern (64-dim vector)
pattern = codex.get_activation_pattern()

# Explain a specific Gene Key
print(codex.explain(38))
```

### KeywordCodexLayer

```python
from trix.db import KeywordCodexLayer, create_codex

codex = create_codex()
layer = KeywordCodexLayer(codex, alpha=0.5)

# Score document against query
score, activations, explanations = layer.score_document(
    query_text="struggle with purpose",
    doc_text="perseverance leads to honour"
)

# Find shared archetypes
shared = layer.find_shared_archetypes(query_text, doc_text)
```

### Gene Keys Data

```python
from trix.db import GENE_KEYS, PARTNERS, CODON_RINGS

# Access Gene Key data
gk38 = GENE_KEYS[38]
# {'shadow': 'Struggle', 'gift': 'Perseverance', 'siddhi': 'Honour'}

# Find programming partner
partner = PARTNERS[38]  # 39

# Get ring members
ring = CODON_RINGS["Ring of Humanity"]  # [10, 17, 21, 25, 38, 51]
```

## Propagation Weights

| Relationship | Weight | Description |
|--------------|--------|-------------|
| Direct keyword | 1.0 | Text contains Shadow/Gift/Siddhi |
| Partner | 0.8 | Programming partner activation |
| Ring | 0.5 | Codon ring member activation |

## Benchmark Results

### Shakespeare (3,230 real embeddings)

| Method | Recall@10 | Latency | vs Qdrant |
|--------|-----------|---------|-----------|
| Qdrant | 100.0% | 2.74ms | baseline |
| HSquares (n=8) | 92.8% | 0.24ms | **11.4x faster** |
| HSquares (n=16) | 96.9% | 0.52ms | **5.3x faster** |
| CodexSquares (n=16) | 94.5% | 0.75ms | **3.7x faster** + archetypes |

Key finding: Real embeddings have structure that clustering exploits.

## Files

- `core.py` - Ternary quantization primitives
- `index.py` - OctaveIndex implementation
- `octave_db.py` - OctaveDB high-level API
- `explorer.py` - PossibilityExplorer for aboutness
- `codex.py` - Embedding-based Codex (CodexField)
- `codex_keywords.py` - Keyword-based Codex (KeywordCodex)
- `hsquares_index.py` - Hollywood Squares IVF index
- `codex_squares.py` - Gene Key-infused Hollywood Squares

## Documentation

- [HOLLYWOOD_SQUARES.md](../../docs/HOLLYWOOD_SQUARES.md) - IVF search with topology
- [CODEX_SQUARES.md](../../docs/CODEX_SQUARES.md) - Search with archetypal meaning
- [CODEX.md](../../docs/CODEX.md) - Keyword-based Codex layer
