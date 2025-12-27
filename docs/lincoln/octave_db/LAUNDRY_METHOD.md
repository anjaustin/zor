# The Laundry Method

## Line in the Sand

```
87/87 tests passing
P@10:  97% (vs Qdrant 100%)  
Speed: 5.1x faster than Qdrant
```

## Current Approach

Rank everything → narrow via cascade → re-rank with exact

## The Laundry Insight

> "Instead of diving straight into the pile, I divide it up into smaller 
> piles by type of clothing; socks, undies, shirts, pants, etc.; then, 
> I take one major segment at a time and break it down by type of sock, 
> type of shirt, or type of pant/short/trouser."

**Partition first. Search within.**

## Translation

```
LAUNDRY                          DB COOPER
────────────────────────────────────────────────────────
Big pile                    →    All documents

Divide by type:             →    Oc2 (COARSE)
  socks, shirts, pants           Bucket by coarse signature

Within socks pile:          →    Oc1 (MEDIUM)  
  ankle, crew, dress             Narrow within bucket

Pick the exact sock:        →    Oc0 (FINE)
  the black ankle one            Exact match
```

## The Delta

The sock that looks like a rag. The shirt that could be a towel.

Items at bucket boundaries that might be mis-classified.

**These need verification with exact cosine.**

## Implementation Plan

1. Hash coarse signatures → bucket assignment
2. Query hits a bucket, search only within
3. Identify boundary items (high delta from bucket center)
4. Verify boundaries with exact cosine
5. Merge results

## Expected Benefits

- Don't search the whole pile
- Hard partitions faster than soft ranking  
- Only expensive verification on delta items
- Memory locality (bucket = contiguous)

## Delta-Weighted Result (from testing)

Same 50 candidates:
- Cascade only:   94.5%
- Exact only:     99.7%
- Delta-weighted: 99.8%

The delta knows where cascade lies.
