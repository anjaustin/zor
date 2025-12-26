# NODES: Octave DB

## Identified Nodes

---

### Node 1: Derivation is the Differentiator

**Observation:** The key insight is coarse = pool(fine). Derived, not independent.

**Why it matters:** 
- Consistency across resolutions
- You're searching ONE space at different zoom levels
- Coarse candidates WILL refine correctly at fine level
- Unlike random projections (LSH), this is deterministic and meaningful

**Tension:** How do you derive? Mean pooling + sign? Max pooling? Learned pooling?

---

### Node 2: Ternary Similarity

**Observation:** Similarity between ternary vectors is just dot product, which is bit operations.

```
score = (q_pos AND d_pos).popcount() 
      + (q_neg AND d_neg).popcount()
      - (q_pos AND d_neg).popcount()
      - (q_neg AND d_pos).popcount()
```

**Why it matters:** No floats. No multiply. Just AND + popcount. CPU-friendly. GPU-trivial.

**Tension:** Does ternary preserve enough semantic information? What's the quality/speed tradeoff?

---

### Node 3: The Meaning of Zero

**Observation:** In ternary embeddings, 0 means "this dimension doesn't matter."

- +1: positively correlated with this feature
- -1: negatively correlated
- 0: neutral, irrelevant, not activated

**Why it matters:** Sparsity is information. The pattern of zeros tells you what's salient.

**Tension:** How many zeros is optimal? Too few = dense, slow. Too many = information loss.

---

### Node 4: Three Kinds of Search

**Observation:** Different resolutions answer different questions.

| Level | Question | Use Case |
|-------|----------|----------|
| Fine | Is this THE thing? | Deduplication, exact lookup |
| Medium | Is this LIKE the thing? | Similarity search, RAG |
| Coarse | Is this RELATED to the thing? | Discovery, intertextuality |

**Why it matters:** One index, three capabilities. Choose resolution for task.

**Tension:** How to expose this in API? Separate methods? Resolution parameter?

---

### Node 5: Glassbox Retrieval

**Observation:** You can show WHY matches match. Dimension by dimension, level by level.

```
Query:    [+1, -1, +1, 0, ...]
Document: [+1, -1, +1, 0, ...]
           ✓   ✓   ✓  ·
```

**Why it matters:** Explainability. Trust. Debugging. Legal/medical justification.

**Tension:** Is per-dimension explanation meaningful to users? Need to map back to features.

---

### Node 6: Native vs Quantized Embeddings

**Observation:** Two paths to ternary embeddings:

1. **Quantized:** Take float embeddings, apply sign threshold
2. **Native:** Train embedder that outputs ternary directly (Gradient Truth)

**Why it matters:** Quantized is faster to prototype. Native is philosophically aligned.

**Tension:** Does native ternary training actually work for embeddings? Open question.

---

### Node 7: TrueOctaveFFN as Embedder

**Observation:** If the embedder IS a TrueOctaveFFN, it outputs all resolutions natively.

```
text → TrueOctaveFFN → (fine, medium, coarse)
```

No post-hoc derivation. The model has octave structure. The output has octave structure.

**Why it matters:** End-to-end alignment. The embedder embodies the philosophy.

**Tension:** Need to design the architecture. What goes in, what comes out?

---

### Node 8: Index Structure

**Observation:** Natural hierarchical index:

```
COARSE BUCKETS
├── bucket_A (coarse signature A)
│   ├── medium index
│   │   ├── doc_1 (fine)
│   │   └── doc_2 (fine)
│   └── medium index
│       └── doc_3 (fine)
└── bucket_B (coarse signature B)
    └── ...
```

**Why it matters:** Coarse-first search is O(buckets) not O(documents). Fast.

**Tension:** What if coarse signatures aren't well distributed? Unbalanced buckets.

---

### Node 9: The Pooling Question

**Observation:** How exactly do you derive coarse from fine?

Option A: sign(mean(fine_vectors))
Option B: majority vote per dimension
Option C: learned pooling (but then not purely derived)

**Why it matters:** The derivation function determines the relationship between levels.

**Tension:** Trade-off between simplicity and expressiveness.

---

### Node 10: Benchmarking Strategy

**Observation:** Need to validate that ternary + hierarchy doesn't kill quality.

Benchmarks:
- Standard retrieval datasets (MS MARCO, BEIR, etc.)
- Compare: float vs ternary, flat vs hierarchical
- Measure: recall@k, latency, memory

**Why it matters:** If quality craters, the speed doesn't matter.

**Tension:** Academic benchmarks may not capture intertextuality benefits.

---

## Tensions Summary

| Tension | Between |
|---------|---------|
| Quality vs Speed | Ternary may lose information |
| Simplicity vs Power | Derived pooling vs learned pooling |
| Native vs Quantized | Principled but hard vs easy but hacky |
| Flat API vs Rich API | One search method vs resolution selection |
| Dense vs Sparse | How many zeros? |

---

## Core Dependencies

```
Node 7 (TrueOctaveFFN embedder)
    │
    ├── requires → Node 9 (pooling function)
    │
    └── enables → Node 4 (three kinds of search)
                       │
                       └── enables → Node 5 (glassbox)

Node 2 (ternary similarity)
    │
    └── enables → Node 8 (index structure)

Node 10 (benchmarking)
    │
    └── validates → everything
```
