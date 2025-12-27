# REFLECT: DB Cooper 100% Quality

## The Hollywood Squares Insight

From Hollywood Squares OS:

> **"Deterministic message passing + bounded local semantics + enforced observability ⇒ global convergence with inherited correctness."**

And from the Constraint Field:

> **"The answer was FORCED by the constraints. No search. No guessing. Just propagation."**

This reframes everything.

## The Reframe

We've been thinking: **filter then rank**
We should think: **constrain until forced**

Each octave level doesn't FILTER (keep some, drop others).
Each octave level CONSTRAINS (eliminate what's definitely wrong).

The difference:
- Filter: "Keep the top 50 by coarse score"
- Constrain: "Eliminate everything that CAN'T be the answer"

**Correctness inherits from local rules when the topology is right.**

## The 3% Error Source

Our cascade CAN eliminate true positives because we use a THRESHOLD.
"Keep top N" is a filter, not a constraint.

A true constraint would be:
"Keep everything that COULD be correct"
"Only eliminate what's DEFINITELY wrong"

## The Conservative Cascade

Instead of: "Top 50 by coarse score"
Use: "Everything with coarse score above MARGIN of top score"

```
top_score = max(coarse_scores)
threshold = top_score * margin  # e.g., 0.7
keep = coarse_scores >= threshold
```

This keeps all PLAUSIBLE candidates, not a fixed count.
The delta = uncertain items = KEEP THEM.

When in doubt, keep it. Let the next level decide.

## The Core Tension (Revised)

Not: "filter vs rank"
But: "when to eliminate"

Only eliminate when CERTAIN it's wrong.
Propagate constraints, don't impose arbitrary thresholds.

## The Hollywood Squares Architecture

```
Oc2 (Coarse):   Eliminate DEFINITELY wrong (high confidence)
                 │
                 ▼
Oc1 (Medium):   Eliminate MORE wrong (medium confidence)
                 │
                 ▼
Oc0 (Fine):     Rank what remains (exact cosine)
                 │
                 ▼
Result:         Forced by constraints, not searched
```

Each level only eliminates what it's CERTAIN about.
Uncertainty propagates forward for finer resolution.

## Why This Achieves 100%

If no level eliminates a true positive:
- True positive survives to Oc0
- Exact cosine ranks it correctly
- 100% recall guaranteed

The key: conservative elimination at each level.

## Implementation

1. Score-relative thresholds, not fixed counts
2. Margin parameter controls conservatism
3. Higher margin = more candidates = higher recall = slower
4. Lower margin = fewer candidates = faster = risk of drops

Find the margin where true positives are never dropped.

## The Laundry Resolution

Node 6 says: Partition first, check boundaries.

The cascade IS the partitioning.
The exact cosine IS the boundary check.
The hybrid approach IS the Laundry Method.

```
Cascade (ternary)  = Sort into piles
Exact (cosine)     = Pick the right sock from the pile
```

We were looking for elegance. The elegance is in the division:
- Ternary for fast exclusion (what's definitely NOT the answer)
- Float for precise inclusion (what IS the answer)

## Why Storing Floats is Okay

Node 7 worried about memory. But:
- 1KB per doc for 256-dim floats
- 1M docs = 1GB
- This is fine for quality-first applications
- Can always add compression later (float16, etc.)

The ternary representation is still valuable:
- Fast coarse filtering
- Bucket assignment
- Approximate nearest neighbor
- Memory-constrained deployments (optional mode)

## The Minimum Architecture for 100%

From our testing:
- Oc2=100, Oc1=50 candidates → 100% P@10
- That's cascade filtering to 10% of docs, then exact cosine

This IS the answer:
1. Store both ternary (coarse/medium/fine) AND float embeddings
2. Cascade filter with ternary (fast)
3. Final rank with exact cosine (accurate)
4. The delta between them is monitored, not used for decisions

## Node 8 Reconsideration

"Magnitude at every level" is interesting but unnecessary.
If we're doing exact cosine at the end anyway, we don't need
magnitude-weighted ternary at coarse/medium.
Pure ternary filtering + exact cosine ranking = simpler.

## The Insight

**Ternary is the filter. Float is the truth.**

Don't try to make ternary as good as float.
Use ternary for what it's good at: fast exclusion.
Use float for what it's good at: precise ranking.

The 100% architecture:
1. Ternary cascade rapidly excludes non-candidates
2. Float cosine precisely ranks the candidates
3. Boundary checking is implicit - exact cosine fixes any cascade errors

## Remaining Questions

1. What's the minimum funnel size for 100%?
   - Test: Oc2=50 Oc1=25? Oc2=30 Oc1=15?
   - Find the threshold where quality degrades

2. Can we detect when to expand the funnel?
   - Score variance? Margin between candidates?
   - Adaptive expansion when uncertain

3. Memory mode vs Quality mode?
   - Quality mode: store floats, exact cosine
   - Memory mode: ternary only, accept quality loss

## Success Criteria

- [ ] 100% P@10 matching Qdrant
- [ ] Still faster than Qdrant (cascade benefit)
- [ ] Clear architecture: filter (ternary) + rank (float)
- [ ] Configurable funnel sizes
- [ ] Optional memory-only mode
