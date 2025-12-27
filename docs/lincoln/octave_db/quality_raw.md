# RAW: DB Cooper 100% Quality

## Stream of Consciousness

We're at 97% P@10 vs Qdrant's 100%. The 3% gap is the delta - where cascade and exact disagree.

The gap comes from quantization. We turn float32 into ternary {-1, 0, +1}. Information loss. The magnitude tells us HOW MUCH in that direction, and we're throwing it away in the cascade.

Wait - we have the Secret Sauce. Magnitude-weighted similarity. But that's only at Oc0 (fine level). The coarse and medium levels are still pure ternary. So the filtering stages can drop good candidates before magnitude ever gets a vote.

The laundry insight: partition first, search within. But what if the sock ends up in the wrong pile? It never gets found. That's our 3%.

Delta-weighted search got 99.8% on same candidates. So if we have the right candidates, we can rank them correctly. The problem is the CASCADE FILTERING - it's dropping good candidates.

Options:
1. Bigger funnel (keep more candidates) - but this is just brute force
2. Store original embeddings, use exact cosine at final stage - we tested this, got 100%
3. Multiple bucket membership - item can be in more than one pile
4. Boundary detection - identify items near bucket edges, pull them into adjacent buckets too

The hybrid approach (cascade filter + exact cosine) achieved 100% P@10 with Oc2=100, Oc1=50. That's only examining 100 candidates out of 1000. 10% of the work for 100% quality.

But is that the right architecture? We're using ternary for filtering and float for ranking. Two systems. Could we do better?

What if the cascade stages ALSO used magnitude? Not just Oc0. Magnitude at every level.

Actually... the coarse level pools the fine level. When we pool, we're averaging signs. But we could also pool magnitudes. Coarse magnitude = sum of fine magnitudes in that region.

The insight from the laundry method: the delta is where mistakes hide. We need to CHECK THE BOUNDARIES. Items with high uncertainty about their bucket assignment.

How do we know uncertainty? The score margin. If a document scores 0.9 in bucket A and 0.1 in bucket B, it's clearly A. If it scores 0.5 and 0.48, it's uncertain - it's on the boundary.

## Questions

- Is 100% quality achievable without storing original embeddings?
- What's the minimum candidate set for 100%?
- Can we predict which queries will have cascade errors?
- Is the 3% gap consistent or query-dependent?

## First Instincts

The hybrid approach works. Cascade for speed, exact for quality. But it feels like cheating - we're not really solving the quantization problem, just working around it.

The pure solution would be: make the cascade never drop a good candidate. But that might require keeping too many candidates, defeating the purpose.

Maybe the answer is adaptive: tight cascade when confident, loose cascade when uncertain. The delta tells us confidence.
