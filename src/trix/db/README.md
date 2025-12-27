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

## Files

- `core.py` - Ternary quantization primitives
- `index.py` - OctaveIndex implementation
- `octave_db.py` - OctaveDB high-level API
- `explorer.py` - PossibilityExplorer for aboutness
- `codex.py` - Embedding-based Codex (CodexField)
- `codex_keywords.py` - Keyword-based Codex (KeywordCodex)
