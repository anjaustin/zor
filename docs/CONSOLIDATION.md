# Architecture Consolidation Map

## The Sprawl: 20 FFN Modules → 17,191 Lines

```
src/trix/nn/
├── hierarchical.py      992 lines  ← CORE: O(√n) routing
├── frozen_6502.py       916 lines  ← CORE: CPU emulation
├── gradient_truth.py    843 lines  ← CORE: Training paradigm
├── providence.py        829 lines  ← CORE: Unified architecture
├── xor_superposition.py 791 lines  ← UTILITY: Compression
├── sparse_lookup_v2.py  746 lines  ← DEPRECATED
├── frozen.py            724 lines  ← CORE: Shape library
├── frozen_shapes.py     707 lines  ← CORE: Activation shapes
├── xor_ffn.py           623 lines  ← CORE: Hamming routing
├── hierarchical_temporal 558 lines ← EXPERIMENTAL: State
├── sparse_lookup.py     540 lines  ← EXPERIMENTAL: Spline routing
├── routed_memory.py     503 lines  ← EXPERIMENTAL: Memory
├── octave.py            494 lines  ← CORE: Multi-resolution
├── kan_hierarchical.py  487 lines  ← TRACK B: KAN
├── sdpmx_pipeline.py    479 lines  ← RESEARCH: Vi's synthesis
├── sparse_lookup_v4.py  448 lines  ← DEPRECATED (no tests)
├── compiled_dispatch.py 445 lines  ← UTILITY: Path compilation
├── additive_kan.py      430 lines  ← TRACK B: KAN
├── anchored.py          430 lines  ← CORE: Partition-first
├── sparse.py            429 lines  ← DEPRECATED
├── multiscale.py        378 lines  ← SUPERSEDED by octave
├── trix.py              373 lines  ← CORE: Base FFN
└── emergent.py          220 lines  ← DEPRECATED
```

---

## Gold Nuggets to Extract

### 1. GRADIENT TRUTH (The Training Paradigm)
**Source:** gradient_truth.py, hierarchical.py, sparse_lookup.py
**Pattern:** Frozen ternary weights + learned scales = real gradients
**Why it's gold:** No STE needed. Mathematically sound. Appears in 8+ modules.
**Keep in:** All production architectures

### 2. SIGNATURE ROUTING (Content-Addressable)
**Source:** trix.py, hierarchical.py, xor_ffn.py
**Pattern:** dot(input, signature) → winner-takes-all
**Variant:** hamming(binarize(input), signature) → XOR routing
**Why it's gold:** O(1) XOR vs O(d) multiply. Content-addressable semantics.
**Keep in:** All routing layers

### 3. HIERARCHICAL 2-LEVEL (O(√n) Scaling)
**Source:** hierarchical.py
**Pattern:** Cluster → Fine tile (64-1000+ tiles feasible)
**Why it's gold:** Production-proven at scale. Best-tested module.
**Keep in:** HierarchicalTriXFFN, AnchoredDualMode uses similar partitioning

### 4. FROZEN SHAPES (Computation as Geometry)
**Source:** frozen.py, frozen_shapes.py, frozen_6502.py
**Pattern:** FP4 compiler atoms → 100% accurate by construction
**Why it's gold:** Provably correct. 40x compression for 6502.
**Keep in:** FrozenTriXFFN, Providence, GradientTruth shape banks

### 5. TEMPERATURE ANNEALING (Soft → Hard)
**Source:** anchored.py, octave.py
**Pattern:** Start soft (exploration), end hard (commitment)
**Why it's gold:** Bridges generative/deterministic modes
**Keep in:** All dual-mode architectures

### 6. DERIVED OCTAVES (Multi-Resolution Views)
**Source:** octave.py (supersedes multiscale.py)
**Pattern:** Coarse = sign(pool(Fine)). Views, not banks.
**Why it's gold:** Bit-shift principle. Hierarchical compression.
**Keep in:** TrueOctaveFFN only (multiscale.py archived)

### 7. SUPERPOSITION COMPRESSION (11.6x)
**Source:** xor_superposition.py
**Pattern:** Base + sparse XOR deltas instead of full signatures
**Why it's gold:** Memory efficiency for large tile counts
**Keep in:** As utility for hierarchical.py (already used)

### 8. SDPMX OPERATORS (Hilbert Space)
**Source:** sdpmx_pipeline.py
**Pattern:** S→D→P→M→X pipeline (Smooth, Differentiate, Project, Mask, XOR)
**Why it's gold:** Unique mathematical foundation. "Geometry dictates order."
**Decision needed:** Integrate into Providence or keep as research branch?

### 9. KAN 1D SPLINES (Track B)
**Source:** additive_kan.py, kan_hierarchical.py
**Pattern:** f(x) = Σ φᵢ(xᵢ) where φᵢ is 1D spline
**Why it's gold:** Kolmogorov-Arnold alternative to MLP
**Decision needed:** Keep as separate track or archive?

---

## Consolidation Plan

### TIER 1: KEEP (Production Core)
| Module | Lines | Role | Status |
|--------|-------|------|--------|
| trix.py | 373 | Base FFN, emergent routing | KEEP |
| hierarchical.py | 992 | O(√n) scaling, content-addressable | KEEP |
| anchored.py | 430 | Partition-first, dual-mode | KEEP |
| octave.py | 494 | Multi-resolution, derived views | KEEP |
| providence.py | 829 | Unified: XOR + shapes + state | KEEP |
| gradient_truth.py | 843 | Explicit 3-layer decomposition | KEEP |
| xor_ffn.py | 623 | Hamming routing | KEEP |
| frozen.py | 724 | Shape library | KEEP |
| frozen_shapes.py | 707 | Activation shapes | KEEP |
| frozen_6502.py | 916 | CPU emulation | KEEP |

**Total: 6,931 lines (40% of sprawl)**

### TIER 2: UTILITY (Support Core)
| Module | Lines | Role | Status |
|--------|-------|------|--------|
| xor_superposition.py | 791 | Compression | KEEP as utility |
| compiled_dispatch.py | 445 | Path compilation | KEEP as utility |

**Total: 1,236 lines (7%)**

### TIER 3: ARCHIVE (Deprecated/Superseded)
| Module | Lines | Reason | Archive |
|--------|-------|--------|---------|
| emergent.py | 220 | DEPRECATED in code | YES |
| sparse.py | 429 | Points to hierarchical | YES |
| sparse_lookup_v2.py | 746 | DEPRECATED in code | YES |
| sparse_lookup_v4.py | 448 | DEPRECATED, no tests | YES |
| multiscale.py | 378 | Superseded by octave | YES |
| layers.py (GatedFFN) | ~100 | Points to hierarchical | YES |

**Total: 2,321 lines (14%) → ARCHIVE**

### TIER 4: RESEARCH BRANCHES (Keep Separate)
| Module | Lines | Value | Decision |
|--------|-------|-------|----------|
| additive_kan.py | 430 | Kolmogorov-Arnold | Keep as Track B |
| kan_hierarchical.py | 487 | KAN + hierarchy | Keep as Track B |
| sdpmx_pipeline.py | 479 | Vi's Synthesis | Keep for now |
| hierarchical_temporal.py | 558 | State persistence | Merge into Providence? |
| sparse_lookup.py | 540 | Spline routing | Keep, has Gradient Truth |
| routed_memory.py | 503 | Memory variant | Review vs Providence |

**Total: 2,997 lines (17%) → DECIDE**

---

## Integration Opportunities

### 1. Providence as the Unified Architecture
Providence already combines:
- XOR routing (from xor_ffn.py)
- Frozen shapes (from frozen.py)
- State persistence (from temporal tiles)

**Could absorb:**
- hierarchical_temporal.py (same state pattern)
- routed_memory.py (memory is Providence's specialty)

### 2. Anchored as the Deployment Architecture
Anchored is designed for:
- Chip deployment (deterministic, synthesizable)
- Modal models (shapes + probabilistic search)

**Best integration with:**
- Frozen shapes (already uses)
- Temperature annealing (already has)
- Could add octave-style multi-resolution?

### 3. SparseLookup as Spline Track
SparseLookup has unique value:
- "Routing IS computation"
- Spline modulation (TernarySpline2D)
- Gradient Truth support

**Keep as alternative routing approach**

---

## Proposed Final Architecture

```
trix.nn/
├── CORE (Production)
│   ├── trix.py              # Base FFN
│   ├── hierarchical.py      # O(√n) scaling
│   ├── anchored.py          # Partition-first
│   ├── octave.py            # Multi-resolution
│   ├── providence.py        # Unified (absorbs temporal)
│   └── gradient_truth.py    # Training paradigm
│
├── ROUTING
│   ├── xor_ffn.py           # Hamming routing
│   └── sparse_lookup.py     # Spline routing (Gradient Truth)
│
├── SHAPES (Frozen Computation)
│   ├── frozen.py            # Shape library
│   ├── frozen_shapes.py     # Activation shapes
│   └── frozen_6502.py       # CPU emulation
│
├── UTILITY
│   ├── xor_superposition.py # Compression
│   └── compiled_dispatch.py # Path compilation
│
├── RESEARCH (Separate tracks)
│   ├── kan/                 # Track B: Kolmogorov-Arnold
│   └── sdpmx/               # Vi's Synthesis
│
└── ARCHIVE (Move to archive/)
    ├── emergent.py
    ├── sparse.py
    ├── sparse_lookup_v2.py
    ├── sparse_lookup_v4.py
    ├── multiscale.py
    └── layers.py (GatedFFN only)
```

---

## Summary

| Category | Modules | Lines | % |
|----------|---------|-------|---|
| CORE (Keep) | 10 | 6,931 | 40% |
| UTILITY (Keep) | 2 | 1,236 | 7% |
| ARCHIVE | 6 | 2,321 | 14% |
| RESEARCH | 6 | 2,997 | 17% |
| Other (__init__, etc) | - | 3,706 | 22% |

**Action Items:**
1. Archive 6 deprecated modules (2,321 lines)
2. Merge hierarchical_temporal into Providence
3. Review routed_memory vs Providence (possible merge)
4. Move KAN to research/kan/ subdirectory
5. Keep SDPMX for now, revisit after Providence matures
6. Update __init__.py exports to reflect consolidation
