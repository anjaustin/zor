# RAW: Octave DB

## Stream of Consciousness

What is an Octave DB? Start from what we know.

Vector DBs store embeddings. Dense float vectors. You query with a vector, get back nearest neighbors by cosine or dot product. This works but it's:
- Float-heavy (memory, compute)
- Flat (one resolution)
- Blackbox (why did these match?)

Octave DB flips this. Multi-resolution. Ternary. Derived hierarchy. Glassbox.

---

The core insight: coarse = pool(fine).

If you have fine-grained embeddings, you can DERIVE coarser versions by pooling. Sign of the mean. The coarse signature isn't independent - it's a VIEW of the fine data.

This means:
- Coarse search is CONSISTENT with fine search
- You're not searching different spaces, you're searching different resolutions of the SAME space
- Candidates found at coarse level will refine correctly at fine level

---

What does "ternary embedding" even mean?

Traditional: embed("cat") → [0.23, -0.47, 0.81, ...]  (floats)
Ternary:    embed("cat") → [+1, -1, +1, 0, -1, ...]   ({-1, 0, +1})

How do you get ternary embeddings?
1. Train a float embedder
2. Quantize: sign(x) where |x| > threshold, else 0
3. Or: train directly with Gradient Truth?

The second is more interesting. What if the embedder is a TrueOctaveFFN? Then the embedding IS multi-resolution natively.

---

Search mechanics.

Float vector search: cosine(q, d) = dot(q, d) / (|q| * |d|)
Ternary search: Hamming distance? Or something else?

For ternary {-1, 0, +1}:
- Match: q[i] == d[i]
- Mismatch: q[i] != d[i] and both non-zero
- Neutral: either is 0

Score = matches - mismatches? Ignoring zeros?

Or treat as two bitmasks:
- positive_mask: where value is +1
- negative_mask: where value is -1

Then similarity is... overlap of masks?

Need to think about this more carefully.

---

The 0s again. In embeddings, 0 means "this dimension doesn't matter for this item."

"Cat" might have 0 in the "vehicle" dimension. Not positive, not negative. Irrelevant.

This is INFORMATION. The sparsity pattern tells you what's salient.

If query has +1 in dimension 5 and document has 0, that's not a mismatch. It's "document doesn't care about this."

Asymmetric? If document has +1 and query has 0... query doesn't care.

Maybe: score = sum(q[i] * d[i])? That's just dot product on ternary.
- (+1)(+1) = +1 (agreement)
- (-1)(-1) = +1 (agreement)
- (+1)(-1) = -1 (disagreement)
- (0)(anything) = 0 (neutral)

That works. And for ternary, dot product is just:
score = popcount(q_pos AND d_pos) + popcount(q_neg AND d_neg) - popcount(q_pos AND d_neg) - popcount(q_neg AND d_pos)

All bit operations. Fast.

---

Multi-resolution search.

```
QUERY: [fine embedding, D dimensions]
       ↓ derive
       [medium embedding, D/4 dimensions]
       ↓ derive
       [coarse embedding, D/16 dimensions]

SEARCH:
1. Compare query_coarse to all doc_coarse → top N candidates
2. Compare query_medium to candidate doc_medium → top M
3. Compare query_fine to candidate doc_fine → final results
```

The coarse search is 16x cheaper (1/16 dimensions). And you only do fine search on survivors.

This is hierarchical. Like a spatial index. But derived from the data itself.

---

Intertextuality. This is the interesting part.

Two documents might have:
- Different fine signatures (different specific content)
- Similar medium signatures (related topics)
- Same coarse signature (same broad context)

This captures RELATIONSHIP without exact match.

"Moby Dick" and a paper on whaling:
- Fine: different (one is novel, one is academic)
- Medium: similar (both about whales)
- Coarse: same (both in "maritime/nature" context)

The coarse signature is the intertextual link.

---

What about updates? Vector DBs struggle with updates.

For Octave DB:
- Insert: compute all three levels, add to index
- Delete: remove from all three levels
- Update: delete + insert

The derivation is one-way (fine → coarse), so you always have consistency.

What if you want to search by coarse and get fine? Easy - the coarse index points to documents, documents have fine embeddings.

---

Clustering emerges.

If you index by coarse signature, documents with same coarse signature cluster together. Natural organization.

The coarse space is small (D/16 dimensions, ternary). There are only so many possible coarse signatures. Each is a "bucket" of related documents.

This is like locality-sensitive hashing, but deterministic and derived.

---

Building the index.

Level 1 (coarse): hash table by coarse signature → list of doc IDs
Level 2 (medium): for each coarse bucket, index medium signatures
Level 3 (fine): for each document, store fine signature

Search:
1. Hash query_coarse → bucket
2. Scan bucket with query_medium → candidates
3. Score candidates with query_fine → results

Simple. Hierarchical. Fast.

---

What about the embedder?

Options:
1. Use existing embedder (BERT, etc.), quantize output
2. Train custom ternary embedder with Gradient Truth
3. Use TrueOctaveFFN as the embedder (native multi-resolution)

Option 3 is most aligned. The embedder itself has octave structure. Output each level directly.

```
text → TrueOctaveFFN → (fine_embed, medium_embed, coarse_embed)
```

No post-hoc derivation needed. The model outputs all resolutions.

---

Glassbox retrieval.

"Why did document D match query Q?"

Show the signatures at each level:
```
Query:    coarse [+1,-1,+1,0,...]  →  medium [...]  →  fine [...]
Document: coarse [+1,-1,+1,0,...]  →  medium [...]  →  fine [...]
                  ^^^^^^^^^^^^
                  "Matched here"
```

You can SEE the agreement. Dimension by dimension.

---

What I don't know:
- Optimal dimensionality at each level
- How to train ternary embeddings that preserve semantic similarity
- Whether Hamming-style search actually works for retrieval quality
- How this compares to existing ternary/binary embedding work

What I suspect:
- The derivation (coarse = pool(fine)) is the key differentiator
- Ternary is probably enough for most semantic tasks
- The glassbox property will be valuable for trust/debugging
- This could be very fast on CPU (bit operations)

---

Random thought: the coarse signature is like a "genre" or "topic" label, but learned not assigned. Emergent categorization.

Random thought: you could visualize the octave space. Coarse signatures form clusters. Medium refines. Fine individuates.

Random thought: this could work for images too. Octave features at multiple resolutions. Same principle.

---

First implementation?

1. Take existing embeddings (from any model)
2. Quantize to ternary
3. Derive coarse levels by pooling
4. Build simple index
5. Benchmark retrieval quality vs float vectors

Start simple. Validate the concept. Then build native octave embedder.

---

The three kinds of search:
1. Exactness: fine-level match, identity
2. Similarness: medium-level match, neighborhood
3. Intertextuality: coarse-level match, shared context

One index, three query modes. Choose your resolution.

Or: cascade through all three. Coarse for recall, fine for precision.

---

I keep coming back to the glassbox property. Traditional retrieval is "trust me, these are similar." Octave retrieval is "here's WHY they're similar, at each resolution."

This matters for:
- RAG: why did you retrieve this context?
- Legal: justify the precedent match
- Medical: explain the similar case
- Debugging: why wrong results?

Glassbox retrieval could be a killer feature.
