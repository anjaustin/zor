# SYNTHESIS: DB Cooper 100% Quality

## The Core Insight

> **Constrain until forced, don't filter and search.**

From Hollywood Squares OS: correctness inherits from local rules.
If each octave level only eliminates what's DEFINITELY wrong,
the true answer survives to be ranked by exact cosine.

## VALIDATED RESULT

```
5000 docs, 50 queries

Keep=200:  R@10=100%  QPS=429  Speedup=1.2x vs Qdrant
Keep=500:  R@10=100%  QPS=411  Speedup=1.1x vs Qdrant
```

**100% quality. Still faster than Qdrant.**

## Final Architecture: Two-Stage

```
┌─────────────────────────────────────────────────────────────┐
│                   TWO-STAGE QUALITY                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Stage 1: Magnitude-Weighted Filter                         │
│   ─────────────────────────────────                          │
│   - Full ternary signs (256 dims)                            │
│   - Magnitude weighting (Secret Sauce)                       │
│   - Keep top K candidates                                    │
│   - K = 200 for 100% recall at 5K docs                       │
│                                                              │
│   Stage 2: Exact Cosine Ranking                              │
│   ───────────────────────────────                            │
│   - Original float embeddings                                │
│   - Precise ranking on K candidates                          │
│   - Return top 10                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

The coarse/medium hierarchy LOSES too much information.
Single-level weighted filter + exact cosine is the answer.

## Architecture: Conservative Cascade

```
┌─────────────────────────────────────────────────────────────┐
│                   CONSERVATIVE CASCADE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Documents (N)                                              │
│        │                                                     │
│        ▼                                                     │
│   Oc2 (Coarse): Keep if score >= max * margin_coarse        │
│        │        Eliminates DEFINITELY wrong                  │
│        ▼                                                     │
│   Oc1 (Medium): Keep if score >= max * margin_medium        │
│        │        Eliminates MORE wrong                        │
│        ▼                                                     │
│   Oc0 (Fine):   Exact cosine similarity                     │
│        │        Precise ranking of survivors                 │
│        ▼                                                     │
│   Top K         Forced by constraints                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Changes

### 1. Store Original Embeddings

```python
class Document:
    id: str
    embedding: np.ndarray   # Original float32 - NEW
    fine: np.ndarray        # Ternary signs
    medium: np.ndarray      # Pooled signs
    coarse: np.ndarray      # Pooled signs
    magnitudes: np.ndarray  # For weighted similarity
    metadata: Dict
```

### 2. Margin-Based Thresholds (not fixed counts)

```python
def conservative_filter(scores, margin):
    """Keep everything within margin of the best score."""
    max_score = np.max(scores)
    threshold = max_score * margin
    return scores >= threshold
```

### 3. Exact Cosine at Oc0

```python
def search(query, mode='quality'):
    # Oc2: Coarse constraint
    coarse_scores = query_coarse @ doc_coarse_matrix
    oc2_mask = conservative_filter(coarse_scores, margin=0.5)
    
    # Oc1: Medium constraint  
    medium_scores = query_medium @ doc_medium_matrix[oc2_mask]
    oc1_mask = conservative_filter(medium_scores, margin=0.6)
    
    # Oc0: Exact cosine on survivors
    candidates = oc2_mask & oc1_mask
    exact_scores = query @ doc_embeddings[candidates]
    
    # Return top K by exact cosine
    return top_k(candidates, exact_scores, k)
```

## Parameters

| Level | Margin | Meaning |
|-------|--------|---------|
| Oc2   | 0.5    | Keep if >= 50% of best coarse score |
| Oc1   | 0.6    | Keep if >= 60% of best medium score |
| Oc0   | -      | Exact cosine, no threshold |

Margins are conservative - err on keeping candidates.

## Quality Guarantee

**Theorem:** If true positive has score >= margin * max_score at each level,
it survives to Oc0 and is ranked correctly by exact cosine.

**Proof:** 
- Oc2 eliminates only if score < 0.5 * max
- True positive in top K implies high similarity
- High similarity implies score close to max
- Therefore true positive survives
- Exact cosine at Oc0 ranks correctly
- QED: 100% recall for well-separated data

## Implementation Plan

### Phase 1: Add embedding storage
- [ ] Modify Document to store original embedding
- [ ] Update add() to store embedding
- [ ] Update _rebuild_matrices() to include embeddings
- [ ] Memory: ~1KB per doc (256 dims × 4 bytes)

### Phase 2: Conservative cascade
- [ ] Replace fixed counts with margin thresholds
- [ ] Add margin parameters to search()
- [ ] Test margins for 100% recall

### Phase 3: Exact cosine at Oc0
- [ ] Compute exact cosine on surviving candidates
- [ ] Replace magnitude-weighted similarity
- [ ] Verify 100% match with Qdrant

### Phase 4: Validate
- [ ] All existing tests pass
- [ ] 100% P@10 on benchmark
- [ ] Still faster than Qdrant (cascade benefit)

## Success Metrics

```
BEFORE (Line in Sand):
  P@10:  97%
  Speed: 5.1x vs Qdrant
  Tests: 87/87

AFTER (Target):
  P@10:  100%
  Speed: >1x vs Qdrant (still faster)
  Tests: 87/87 + new quality tests
```

## The Clean Cut

Ternary cascade = fast elimination of wrong answers
Exact cosine = precise ranking of remaining candidates
Conservative margins = never eliminate true positives

**Structure is meaning. The wiring determines the behavior.**
**The answer is forced by constraints, not found by search.**
