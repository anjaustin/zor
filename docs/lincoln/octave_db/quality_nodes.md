# NODES: DB Cooper 100% Quality

## Node 1: The 3% Gap is Cascade Filtering
The problem isn't ranking - delta-weighted gets 99.8% on same candidates.
The problem is candidates get DROPPED before reaching fine stage.
Tension: Filtering is the speed. Filtering is also the error.

## Node 2: Magnitude Only Votes at Oc0
Coarse and Medium stages use pure ternary similarity.
Magnitude (the Secret Sauce) only applies at fine level.
By then, good candidates may already be gone.
Why it matters: The filter doesn't know about magnitude.

## Node 3: The Hybrid Works
Cascade filter (ternary) + Exact cosine (float) = 100% P@10
Oc2=100, Oc1=50 → only 10% of docs examined
This is proven. It works.
Tension: Two systems (ternary + float). Is this elegant or hacky?

## Node 4: Boundary Items Are The Error
Items near bucket boundaries get mis-assigned.
The sock that looks like a rag.
These are the 3%.
Solution shape: Identify boundaries, treat specially.

## Node 5: Delta = Confidence Signal
Low delta between cascade and exact → cascade is reliable
High delta → cascade is wrong
We can measure this, but only AFTER exact computation.
Tension: Need exact to know if cascade is wrong, but exact is what we're avoiding.

## Node 6: The Laundry Method Says Check Boundaries
Partition first, search within.
But CHECK the weird items that might be in wrong pile.
This is the delta - boundary items need verification.

## Node 7: Store Original Embeddings?
If we store float32 embeddings, we can always do exact cosine.
Memory cost: 256 dims × 4 bytes = 1KB per doc
But then what's the point of ternary?
Ternary is for: (a) memory compression, (b) fast similarity
If we store floats anyway, we lose (a).

## Node 8: Magnitude at Every Level
What if coarse/medium stages ALSO used magnitude-weighted similarity?
Pool magnitudes alongside signs.
Coarse magnitude = aggregated fine magnitudes.
This might prevent good candidates from being dropped.

## Node 9: Adaptive Funnel
Tight funnel when confident (low delta expected)
Wide funnel when uncertain (high delta expected)
But how to predict delta before computing it?
Query characteristics? Score distribution? Variance?

## Node 10: The Goal is 100%, Not Speed
User said: "quality of the product is primary"
Speed advantage is nice but optional.
If hybrid (cascade + exact) gives 100%, that's the answer.
Elegance is secondary to correctness.
