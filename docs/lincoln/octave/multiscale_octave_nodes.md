# Nodes of Interest: MultiScale Octave Architecture

## Node 1: Two Meanings of "Octave"

**Sparse Octave (Providence):** Bit-shift on lookup keys. Fine = 16 bits, Medium = 12 bits, Coarse = 8 bits. The SAME memory, different address precision.

**Our MultiScale:** Different tile counts per level. Fine = 64 tiles, Medium = 16 tiles, Coarse = 4 tiles. DIFFERENT memories, same address precision.

Why it matters: These are fundamentally different mechanisms. One coarsens the QUERY. One coarsens the BANK.

---

## Node 2: Derived vs Independent Structure

Current implementation: Each octave initialized with random ternary weights. No relationship between levels.

Alternative: Coarse patterns derived FROM fine patterns via pooling/averaging then ternarizing.

Why it matters: True octave structure implies hierarchy. Random init implies independent specialization. Which do we want?

---

## Node 3: Hamming ≈ Inverted Dot Product

For ternary vectors:
- Hamming distance = popcount(XOR(a, b)) = count of disagreements
- Dot product = sum(a * b) = count of agreements - count of disagreements

They're linearly related: dot(a,b) = len(a) - 2*hamming(a,b)

Why it matters: Providence (Hamming) and signature routing (dot product) are THE SAME THING with inverted similarity. We can unify them.

---

## Node 4: The Blend Network is the True Insight

Both Sparse Octave and our MultiScale have a learned blend network that combines outputs from different resolutions.

This is where "fuzzy" lives. The frozen structures provide options. The blend chooses.

Why it matters: Maybe the octave CONSTRUCTION is less important than the octave COMBINATION. The blend network might make any reasonable octave structure work.

Tension with Node 2: But if blend learns everything, why have structure at all? There must be inductive bias from the octave design.

---

## Node 5: Bit-Shift = Pooling on Ternary

Sparse Octave: key >> 4 means "ignore the low 4 bits"

For ternary weights, the equivalent might be: sign(mean(weight_chunk))

Both operations: reduce resolution by aggregating fine structure into coarse summary.

Why it matters: This gives us a principled way to DERIVE coarse octaves from fine octaves.

---

## Node 6: Content-Addressed vs Signature-Routed

Providence: Store (key, value) pairs. Query with key. Get values with similar keys.

Signature routing: Tiles have signatures. Input matches to signatures. Get tile with best match.

These are... the same? 
- Key = Signature
- Value = Tile computation
- Similarity = Dot product (or inverted Hamming)

Why it matters: We might not need two separate architectures. They're isomorphic.

---

## Node 7: Deterministic Mode = Fine Scale Only

In deterministic mode, we force scale_weights = [1, 0, 0]. Only fine octave matters.

But what if deterministic structures EXIST at multiple scales? The 6502 has both fine structure (individual opcodes) and coarse structure (instruction categories).

Tension: Maybe deterministic mode should still use all octaves, just with HARD routing instead of soft routing?

---

## Node 8: The Hierarchy Should Be Frozen Too

If coarse patterns derive from fine patterns, then the DERIVATION is frozen (it's just pooling + sign).

The only learned things should be:
1. Which fine patterns exist (currently: random init, could be: discovered)
2. How to blend octave outputs (the blend network)

Why it matters: This is more pure Gradient Truth. Less learned, more discovered.

---

## Node 9: Missing - Temporal/Sequential Octaves

Current octaves are spatial (different resolutions). 

What about temporal octaves? 
- Fine: this token
- Medium: this sentence  
- Coarse: this document

Why it matters: Multi-scale might not just be about resolution. It might be about SCOPE.

---

## Node 10: The "World Model" Claim

We said this could "model the world accurately while retaining generative capabilities."

For this to be true:
- Fine scale must capture EXACT dynamics (physics, logic, math)
- Coarse scale must capture FUZZY semantics (meaning, intent, context)
- Blend must learn when to be exact vs when to be fuzzy

Is our current design sufficient for this? Or do we need true octave derivation?

---

## Tensions Summary

| Tension | Nodes |
|---------|-------|
| Independent vs Derived octaves | 1, 2, 5, 8 |
| Two architectures vs One unified | 3, 6 |
| Construction vs Combination matters | 2, 4 |
| Spatial vs Temporal multi-scale | 9, 10 |
| Hard routing vs Soft routing in deterministic mode | 7 |
