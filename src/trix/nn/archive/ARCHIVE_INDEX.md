# Archive Index

**Archived:** December 27, 2025
**Reason:** Architecture consolidation - these modules are superseded or deprecated

This is essential research. Nothing is deleted. Everything is preserved for reference.

---

## Archived Modules

### 1. emergent.py (220 lines)
**Original purpose:** Emergent Routing for TriX
**Why archived:** Self-documented as DEPRECATED. Uses legacy STE-based training.
**Superseded by:** HierarchicalTriXFFN with `use_gradient_truth=True`
**Gold extracted:** The emergent routing concept (signatures emerge from weights) lives on in trix.py

```python
# To access archived module:
from trix.nn.archive.emergent import EmergentGatedFFN, EmergentTransformerBlock
```

### 2. sparse.py (429 lines)
**Original purpose:** SparseTriXFFN - Sparse Training Components (Option B)
**Why archived:** Code says "use HierarchicalTriXFFN instead"
**Superseded by:** HierarchicalTriXFFN
**Gold extracted:** One-hot routing pattern influenced hierarchical design

```python
# To access archived module:
from trix.nn.archive.sparse import SparseTriXFFN, SparseTriXBlock
```

### 3. sparse_lookup_v2.py (746 lines)
**Original purpose:** SparseLookupFFN v2 with Signature Surgery and Island Regularization
**Why archived:** Self-documented as DEPRECATED
**Superseded by:** SparseLookupFFN with `use_gradient_truth=True`
**Gold extracted:** Signature surgery concept may be worth revisiting

```python
# To access archived module:
from trix.nn.archive.sparse_lookup_v2 import SparseLookupFFNv2, SparseLookupBlockV2
```

### 4. sparse_lookup_v4.py (448 lines)
**Original purpose:** SparseLookupFFNv4 - SpatioTemporal TriX
**Why archived:** Self-documented as DEPRECATED, no test file exists
**Superseded by:** SparseLookupFFN with `use_gradient_truth=True`
**Gold extracted:** SpatioTemporal concept may inform future temporal work

```python
# To access archived module:
from trix.nn.archive.sparse_lookup_v4 import SparseLookupFFNv4
```

### 5. multiscale.py (378 lines)
**Original purpose:** MultiScaleTriXFFN - "Exact where exact, fuzzy where fuzzy"
**Why archived:** Superseded by TrueOctaveFFN (same concept, better implementation)
**Superseded by:** TrueOctaveFFN in octave.py
**Gold extracted:**
- 3-octave structure (fine/medium/coarse) → now in TrueOctaveFFN
- "Fuzziness in routing, not patterns" → core insight preserved
- Scale blending → revised to selection (commitment principle)

```python
# To access archived module:
from trix.nn.archive.multiscale import MultiScaleTriXFFN, MultiScaleTriXBlock, OctaveTile
```

---

## Why These Were Archived (Not Deleted)

1. **Historical value:** These modules represent the evolution of ideas
2. **Gold extraction:** Key patterns were identified and preserved in production code
3. **Reference:** Future work may need to revisit these approaches
4. **Testing:** Some test files still reference these for regression coverage

---

## Accessing Archived Code

All archived modules remain importable:

```python
# Direct import from archive
from trix.nn.archive import emergent, sparse, multiscale

# Or via main module (with deprecation warning)
from trix.nn import SparseTriXFFN  # Will warn, then import from archive
```

---

## Related Test Files

These test files still exist and may reference archived modules:
- tests/test_sparse.py
- tests/test_sparse_lookup_v2.py
- tests/test_multiscale.py

Tests are preserved for regression coverage and historical reference.

---

## Gold Nuggets Extracted to Production

| From | Pattern | Now In |
|------|---------|--------|
| emergent.py | Signature = sum(weights) | trix.py, hierarchical.py |
| sparse.py | One-hot routing | hierarchical.py (Gradient Truth mode) |
| sparse_lookup_v2.py | Signature surgery | (Consider for future) |
| sparse_lookup_v4.py | SpatioTemporal | hierarchical_temporal.py |
| multiscale.py | 3-octave structure | octave.py (TrueOctaveFFN) |
| multiscale.py | Derived octaves | octave.py (derive_octave) |
| multiscale.py | Scale blending | octave.py (revised to selection) |

---

## The Consolidation

**Before:** 20 FFN modules, 17,191 lines
**After:**
- 10 core modules (6,931 lines)
- 2 utility modules (1,236 lines)
- 5 archived modules (2,321 lines) ← you are here
- 4 research modules (1,396 lines)

The gold is extracted. The history is preserved. The path forward is clear.
