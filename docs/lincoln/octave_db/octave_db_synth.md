# SYNTHESIS: Octave DB

## The Vision

**Octave DB: Multi-resolution ternary retrieval with glassbox explainability.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OCTAVE DB                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                                │
│   │ COARSE  │ ←─ │ MEDIUM  │ ←─ │  FINE   │   ← derived hierarchy          │
│   │ D/16    │    │  D/4    │    │    D    │                                │
│   └────┬────┘    └────┬────┘    └────┬────┘                                │
│        │              │              │                                      │
│        ▼              ▼              ▼                                      │
│                                                                             │
│   CONTEXT       SIMILARITY      EXACTNESS                                   │
│   (related?)    (like this?)    (same thing?)                              │
│                                                                             │
│   Ternary {-1, 0, +1}  ·  Bit operations  ·  Glassbox                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Principles

### 1. Derived Hierarchy
```
coarse = sign(pool(fine))
```
Not independent levels. Views of the same data. Consistency guaranteed.

### 2. Ternary Signatures
```
{-1, 0, +1}

+1 = positive activation
-1 = negative activation  
 0 = not activated (don't care)
```
2 bits per dimension. Bit-parallel operations.

### 3. Three Search Modes
```
EXACT:   Fine-level match    → identity verification
SIMILAR: Medium-level match  → nearest neighbors
CONTEXT: Coarse-level match  → intertextual discovery
```

### 4. Glassbox
```
"Why did these match?"

Query:    [+1, -1, +1, 0, +1, -1, 0, ...]
Document: [+1, -1, +1, 0, -1, -1, +1, ...]
           ✓   ✓   ✓  ·   ✗   ✓   ·

Agreement: dims 0,1,2,5
Conflict:  dim 4
Neutral:   dims 3,6
```

---

## Architecture

### Embedding Pipeline

```
┌──────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  TEXT    │ ──→ │ TrueOctave Embedder │ ──→ │ (fine, med, crs) │
└──────────┘     │                     │     │  D, D/4, D/16    │
                 │  ternary output     │     │  ternary vectors │
                 └─────────────────────┘     └──────────────────┘
```

### Index Structure

```
COARSE INDEX (hash table)
├── signature_A → [doc_1, doc_7, doc_23, ...]
├── signature_B → [doc_2, doc_5, ...]
├── signature_C → [doc_3, doc_8, doc_14, doc_19, ...]
└── ...

DOCUMENT STORE
├── doc_1 → {fine: [...], medium: [...], coarse: [...]}
├── doc_2 → {fine: [...], medium: [...], coarse: [...]}
└── ...
```

### Search Algorithm

```
search(query, mode="similar", top_k=10):
    
    # 1. Derive query signatures
    q_fine, q_med, q_coarse = embed(query)
    
    # 2. Coarse filter (fast)
    buckets = find_matching_buckets(q_coarse, threshold=0.8)
    candidates = documents_in(buckets)  # maybe 1000s → 100s
    
    # 3. Medium rerank (if mode != "context")
    if mode in ["similar", "exact"]:
        candidates = rerank(candidates, q_med, top=100)
    
    # 4. Fine rerank (if mode == "exact" or "similar")
    if mode in ["similar", "exact"]:
        candidates = rerank(candidates, q_fine, top=top_k)
    
    # 5. Return with explanations
    return [(doc, score, explain(q_fine, doc.fine)) for doc in candidates]
```

---

## Similarity Function

For ternary vectors q and d, packed as bitmasks:

```
q_pos, q_neg = pack_ternary(q)  # two bitmasks
d_pos, d_neg = pack_ternary(d)

agreement = popcount(q_pos & d_pos) + popcount(q_neg & d_neg)
conflict  = popcount(q_pos & d_neg) + popcount(q_neg & d_pos)
score = agreement - conflict
```

All bit operations. No multiply. Fast.

---

## Implementation Plan

### Phase 1: Prototype (Validate Concept)

**Week 1-2: Quantized Embeddings**
```python
# Take any embedding model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed
float_embed = model.encode(texts)  # [N, 384]

# Quantize to ternary
threshold = 0.3
ternary = np.sign(float_embed) * (np.abs(float_embed) > threshold)

# Derive coarse levels
fine = ternary                          # [N, 384]
medium = sign(pool(fine, factor=4))     # [N, 96]
coarse = sign(pool(medium, factor=4))   # [N, 24]
```

**Week 2-3: Simple Index**
```python
class OctaveIndex:
    def __init__(self):
        self.coarse_buckets = defaultdict(list)
        self.documents = {}
    
    def add(self, doc_id, fine, medium, coarse):
        self.coarse_buckets[tuple(coarse)].append(doc_id)
        self.documents[doc_id] = (fine, medium, coarse)
    
    def search(self, query_fine, query_medium, query_coarse, top_k):
        # Coarse filter
        candidates = []
        for sig, docs in self.coarse_buckets.items():
            if ternary_similarity(query_coarse, sig) > threshold:
                candidates.extend(docs)
        
        # Fine rerank
        scored = [(doc, ternary_similarity(query_fine, self.documents[doc][0])) 
                  for doc in candidates]
        return sorted(scored, key=lambda x: -x[1])[:top_k]
```

**Week 3-4: Benchmark**
- Test on BEIR, MS MARCO subsets
- Compare: Octave vs flat float
- Measure: recall@10, recall@100, latency

### Phase 2: Native Embedder (If Phase 1 Validates)

**Design TrueOctave Embedder**
```python
class OctaveEmbedder(nn.Module):
    def __init__(self, d_model=384, num_fine_tiles=64):
        self.encoder = TransformerEncoder(...)  # text to hidden
        self.octave_ffn = TrueOctaveFFN(d_model, num_fine_tiles)
    
    def forward(self, text):
        hidden = self.encoder(text)       # [B, S, D]
        pooled = hidden.mean(dim=1)       # [B, D]
        
        # Octave FFN outputs all resolutions
        fine, medium, coarse = self.octave_ffn(pooled)
        
        return fine, medium, coarse  # ternary at each level
```

**Training**
- Contrastive loss on fine level
- Consistency loss: coarse ≈ pool(fine)
- Gradient Truth: frozen ternary, learned routing

### Phase 3: Production

- Optimized index (HNSW at coarse level?)
- Bit-packed storage
- SIMD similarity
- REST/gRPC API
- Persistence

---

## API Design

```python
# Initialize
db = OctaveDB(dimensions=384, pool_factor=4)

# Index
db.add(id="doc1", text="The cat sat on the mat.")
db.add(id="doc2", text="A dog ran through the park.")
db.add_batch(ids=[...], texts=[...])

# Search with mode
results = db.search("feline resting", mode="similar", top_k=10)
results = db.search("exact quote here", mode="exact", top_k=1)
results = db.search("animals outdoors", mode="context", top_k=100)

# Explain
explanation = db.explain(query="cat", doc_id="doc1")
# → {agreement: [3,7,12], conflict: [5], neutral: [0,1,2,...]}

# Introspect
signature = db.get_signature(doc_id="doc1")
# → {fine: [...], medium: [...], coarse: [...]}
```

---

## Success Metrics

| Metric | Target | Why |
|--------|--------|-----|
| Recall@10 | >90% of float baseline | Quality must not crater |
| Latency | <10ms for 1M docs | Speed is the point |
| Memory | <1GB for 1M docs | Ternary = compact |
| Explainability | Per-dimension breakdown | Glassbox |

---

## The One-Liner

**Octave DB: Find the exact, the similar, and the related — in one index, with full explainability.**

---

## Next Action

**Build the prototype.** Quantize existing embeddings. Build simple index. Benchmark.

If quality holds: proceed to native embedder.
If quality craters: understand why, iterate on quantization strategy.

The prototype is the test. Run it.
