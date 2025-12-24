# TriX Master Index

> **Complete catalog of all modules, experiments, and documentation in this repository.**
>
> *"Know thy codebase."*

Last updated: 2025-12-20

---

## Table of Contents

1. [Core Modules (Production)](#core-modules-production)
2. [Experimental Modules (Active Research)](#experimental-modules-active-research)
3. [Orphaned Modules (Preserved Ideas)](#orphaned-modules-preserved-ideas)
4. [Subsystems](#subsystems)
5. [Mesa Experiments](#mesa-experiments)
6. [Documentation](#documentation)
7. [Tests](#tests)
8. [Version Timeline](#version-timeline)

---

## Status Legend

| Status | Meaning |
|--------|---------|
| `ACTIVE` | Production-ready, exported, tested |
| `EXPERIMENTAL` | Working but not finalized |
| `RESEARCH` | Exploratory, may not work |
| `ORPHANED` | Not exported, no tests, preserved for ideas |
| `DEPRECATED` | Superseded, kept for reference |

---

## Core Modules (Production)

### `src/trix/nn/` - Neural Network Layers

| Module | Status | Lines | Exported | Tests | Description |
|--------|--------|-------|----------|-------|-------------|
| `hierarchical.py` | `ACTIVE` | 858 | Yes | Yes | **HierarchicalTriXFFN** - 2-level routing for 64+ tiles, O(√n) complexity. The main architecture. |
| `sparse_lookup.py` | `ACTIVE` | 486 | Yes | Yes | **SparseLookupFFN** - "Routing IS the computation". Spline-based magnitude modulation. |
| `sparse_lookup_v2.py` | `ACTIVE` | 737 | Yes | Yes | **SparseLookupFFNv2** - Surgery API, claim tracking, island regularization. |
| `sparse.py` | `ACTIVE` | 232 | Yes | Yes | **SparseTriXFFN** - Simple 4-tile sparse FFN. Proven baseline. |
| `trix.py` | `ACTIVE` | 212 | Yes | Yes | **TriXFFN** - Original emergent routing. Reference implementation. |
| `temporal_tiles.py` | `ACTIVE` | 463 | Yes | Yes | **TemporalTileLayer** - State routing for temporal binding (Mesa 4). |
| `compiled_dispatch.py` | `ACTIVE` | 321 | Yes | Yes | **CompiledDispatch** - Path compilation for O(1) inference. |
| `xor_superposition.py` | `ACTIVE` | 450 | Yes | Yes | **XORSuperpositionFFN** - 129× signature compression via XOR deltas (Mesa 13). |
| `layers.py` | `ACTIVE` | 176 | Yes | - | **Top1Gate, GatedFFN** - Learned routing (alternative approach). |
| `emergent.py` | `ACTIVE` | 155 | Yes | - | **EmergentGatedFFN** - Emergent routing with gating. |
| `additive_kan.py` | `ACTIVE` | 430 | Yes | Yes | **AdditiveKAN** - Kolmogorov-Arnold via 1D splines. "768×16 = Doable." |
| `kan_hierarchical.py` | `ACTIVE` | 487 | Yes | Yes | **HierarchicalKANFFN** - KAN tiles + hierarchical routing. 98% param savings. |
| `sdpmx_pipeline.py` | `ACTIVE` | 350 | Yes | Yes | **SDPMXPipeline** - Vi's Synthesis: S→D→P→M→X Hilbert operators. |
| `frozen.py` | `ACTIVE` | 725 | Yes | Yes | **FrozenTile, FrozenTriXFFN** - Frozen shapes infrastructure (Mesa 14). |
| `frozen_6502.py` | `ACTIVE` | 894 | Yes | Yes | **Frozen6502** - 16 frozen shapes for 6502 emulation. 40× compression. |

### `src/trix/kernel/` - Low-Level Operations

| Module | Status | Lines | Description |
|--------|--------|-------|-------------|
| `bindings.py` | `ACTIVE` | 374 | **TriXLinear**, pack/unpack weights, NEON kernels |

### `src/trix/qat/` - Quantization-Aware Training

| Module | Status | Lines | Description |
|--------|--------|-------|-------------|
| `quantizers.py` | `ACTIVE` | 258 | **TernaryQuantizer**, **SoftTernaryQuantizer**, **QATTrainer** |

### `src/trix/viz/` - SDPMX Visualizer (Ghost in the Machine)

| Module | Status | Lines | Exported | Tests | Description |
|--------|--------|-------|----------|-------|-------------|
| `metrics.py` | `ACTIVE` | 210 | Yes | Yes | **HealthScore**, **MaskStats**, **OperatorDelta** - Metric computation |
| `buffer.py` | `ACTIVE` | 180 | Yes | Yes | **RingBuffer**, **Aggregates**, **ObservationStore** - Storage |
| `observer.py` | `ACTIVE` | 200 | Yes | Yes | **SDPMXObserver** - Hooks, sampling, async queue |
| `tui.py` | `ACTIVE` | 220 | Yes | Yes | **RichTUI**, **SimpleTUI** - Terminal dashboards |

**Total**: 4 files, ~810 lines, 18 exports, 18 tests

**Documentation**: `docs/SDPMX_VISUALIZER.md` (Complete)

**Examples**: `examples/viz/demo_observer.py`, `examples/viz/demo_live_dashboard.py`

**Key Features**:
- Zero-impact observation (async, sampled)
- Hierarchical logging levels (0-3)
- Health score with trend tracking
- Ring buffer + running aggregates
- Rich terminal dashboard or simple fallback

---

## Experimental Modules (Active Research)

| Module | Status | Lines | Exported | Description |
|--------|--------|-------|----------|-------------|
| `xor_routing.py` | `RESEARCH` | 388 | No | XOR-based superposition routing. Foundation for Mesa 13. |

---

## Orphaned Modules (Preserved Ideas)

> These modules contain valuable ideas but are not exported, tested, or integrated.
> **Do not delete.** They represent research directions worth revisiting.

| Module | Status | Lines | Born | Description | Key Insight |
|--------|--------|-------|------|-------------|-------------|
| `routed_memory.py` | `ORPHANED` | 503 | v0.5? | **Attention replacement**: Tile-routed memory slots | "Qdrant with a brain at every address" |
| `gated_spline.py` | `ORPHANED` | ~300 | v0.4? | **GSU**: Gated spline units | "Learn big (Hybrid), deploy small (Spline-6502)" |
| `sparse_lookup_v3.py` | `ORPHANED` | 608 | v0.5? | **Geometric TriX**: Yang-Mills gauge transforms | Navier-Stokes + Yang-Mills + Hodge cohomology |
| `sparse_lookup_v4.py` | `ORPHANED` | 439 | v0.6.0 | **SpatioTemporal routing**: 3D routing dimensions | Spatial + Temporal + Content routing combined |
| `spline.py` | `ORPHANED` | 371 | v0.3? | **SplineLayer**: Piecewise linear for 6502 hardware | "~12 cycles vs ~10,000 for MLP" |
| `spline2d.py` | `ORPHANED` | 374 | v0.4? | **Spline2D**: 2D KAN for binary operations | "65,536 inputs → 256 cells → 768 params" |
| `spline_adc.py` | `ORPHANED` | ~200 | v0.4? | **SplineADC**: Wrap-aware addition spline | Two splines: wrap vs no-wrap |
| `hybrid_kan.py` | `ORPHANED` | ~350 | v0.4? | **Hybrid KAN**: Bottleneck + spline tiles | PQH architecture that learns |

### Orphan Analysis

**Track B** (`additive_kan.py` → `kan_hierarchical.py`):
- **RESURRECTED 2024-12-20**: Now ACTIVE with tests
- Validated at 99.74% accuracy on 6502 with 30× fewer parameters
- Key insight proven: Per-dimension splines avoid curse of dimensionality

**SDPMX Pipeline** (`sdpmx_pipeline.py`):
- **CREATED 2024-12-20**: Vi's Synthesis of Hilbert space operators
- S→D→P→M→X: Smooth, Differentiate, Project, Mask, XOR
- Based on Robert's Hilbert geometry + V'Gem's architecture
- Validated at 90.32% on 6502 (needs P refinement)

**Geometric Series** (`sparse_lookup_v3.py`):
- Most ambitious module - references Millennium Problems
- Yang-Mills parallel transport between tiles
- Navier-Stokes for smooth/turbulent decomposition
- Hodge cohomology for topological compression
- **Why orphaned?** Too speculative, no validation

**Hardware Path** (`spline.py` → `spline2d.py` → `spline_adc.py`):
- Direct line to 6502 implementation
- Proven math (splines work)
- **Why orphaned?** Superseded by SparseLookupFFN

---

## Subsystems

### `src/trix/compiler/` - Circuit Compiler

| Module | Status | Lines | Description |
|--------|--------|-------|-------------|
| `atoms.py` | `ACTIVE` | 407 | Atom library base |
| `atoms_fp4.py` | `ACTIVE` | 597 | FP4 atom library with threshold circuits |
| `compiler.py` | `ACTIVE` | 349 | Main compiler orchestration |
| `compose.py` | `ACTIVE` | 413 | Circuit composition |
| `decompose.py` | `ACTIVE` | 240 | Spec decomposition |
| `emit.py` | `ACTIVE` | 535 | Code emission (TriX, FP4) |
| `fp4_pack.py` | `ACTIVE` | 351 | FP4 weight packing |
| `spec.py` | `ACTIVE` | 308 | Circuit specifications |
| `verify.py` | `ACTIVE` | 348 | Formal verification |

**Total**: 10 files, 3,636 lines, 43 exports

**Status**: Complete subsystem, under-documented. Not integrated with main tutorials.

### `src/trix/guardian/` - Adaptive Training Observer (Mesa 12)

| Module | Status | Lines | Description |
|--------|--------|-------|-------------|
| `guardian.py` | `EXPERIMENTAL` | 256 | **TrainingObserver** - Monitors and intervenes |
| `observer.py` | `EXPERIMENTAL` | 408 | **ObserverModel** - State encoding |
| `pipeline.py` | `EXPERIMENTAL` | 600 | **AdaptiveTrainingPipeline** - Multi-phase training |
| `programmable_tile.py` | `EXPERIMENTAL` | 335 | **ProgrammableTile** - Read/write interface |
| `reflector.py` | `EXPERIMENTAL` | 327 | **XORReflector** - Superposition-based reflection |
| `training.py` | `EXPERIMENTAL` | 446 | **GuardedTrainer** - Training with observation |

**Total**: 6 files, 2,372 lines, 15 exports

**Status**: Research code. Marked experimental. Has tests but no examples.

---

## Mesa Experiments

### Mesa 11: Unified Addressing Theory (UAT)

**Location**: `experiments/mesa11/`

| Experiment | File | Status | Result |
|------------|------|--------|--------|
| 01 | `01_pipeline_emulation.py` | Complete | 0.00 error, 100% routing |
| 02 | `02_mixed_signatures.py` | Complete | Temporal ⊂ Content proven |
| 02b | `02b_mixed_signatures_strict.py` | Complete | 95.6% vs 25%/50% baselines |
| 03 | `03_spatial_addressing.py` | Complete | 100% topology preservation |
| 04 | `04_manifold_visualization.py` | Complete | Training warps manifold (0.077 movement) |
| 05 | `05_geodesic_tracing.py` | Complete | Routing = shortest paths (100% match) |
| 06 | `06_metric_construction.py` | Complete | Metric determines routing |
| 06b | `06b_weighted_metric_control.py` | Complete | λ-slider control works |
| 07 | `07_curvature_generalization.py` | Complete | r=+0.712 correlation |

**Rigorous Validation**: `experiments/mesa11/rigorous/`
- `trixgr_6502_monolithic.py` - 6502 training with geometric validation (99.76% accuracy)

**Documentation**: `docs/MESA11_UAT.md` (Complete)

### Mesa 12: Guardian / Training Observer

**Location**: `experiments/mesa12/`

| Experiment | File | Status | Result |
|------------|------|--------|--------|
| HALO Pipeline | `halo_pipeline_test.py` | Complete | Pipeline works |
| Field Test | `trixgr_guardian_fieldtest.py` | Complete | Observer functional |

**Documentation**: `docs/MESA12.md` + `docs/legacy/MESA12_*.md`

**Status**: Experimental. Needs examples and clearer use cases.

### Mesa 13: XOR Superposition Compression

**Location**: `experiments/mesa13/`

| Experiment | File | Status | Result |
|------------|------|--------|--------|
| TriX129X 6502 | `train_6502_trix129x.py` | Active | 99.79% accuracy, 2.7× compression |

**Documentation**: `docs/MESA13_XOR_SUPERPOSITION.md` (Complete)

**Key Findings**:
- 129× compression on synthetic signatures
- 2.7× compression on trained signatures
- 0.000% accuracy loss after compression
- XOR mixer init sweet spot: 0.0173

### Mesa 14: Frozen Shapes Integration

**Location**: `src/trix/nn/frozen.py`, `src/trix/nn/frozen_6502.py`

**Documentation**:
- `docs/FROZEN_SHAPES.md` - Theory + API + Tutorial
- `docs/FROZEN_6502.md` - 6502-specific architecture
- `tmp/frozen_integration/` - Lincoln Manifold exploration (4 phases)

**Key Insight**: FP4 atoms from the compiler ARE frozen shapes. We wrap them with signatures for content-addressable routing.

**Core Thesis**: *"Computation is topology. Learning is routing."*

**What was built**:
- `FrozenShape` - Verified computation primitive with truth-table-derived signature
- `FrozenTile` - nn.Module wrapper compatible with TriXTile interface
- `FrozenShapeRegistry` - Catalog of available shapes (includes FP4 atoms)
- `FrozenTriXFFN` - FFN with content/opcode routing to frozen tiles
- `Frozen6502` - Complete 6502 ALU using 16 frozen shapes

**Result**:
- 100% accuracy by construction
- ~2,500 parameters (vs ~100,000 monolithic) = **40× compression**
- All 16 6502 shapes validated exhaustively

### Atomic Functions: Frozen 6502

**Location**: `experiments/atomic/`

| Experiment | File | Status | Result |
|------------|------|--------|--------|
| ADC Micro-NN | `train_adc_micronn.py` | Complete | 611 params → 98.8%, 16 params → 100% |
| Frozen v1 | `frozen_shapes_meaning.py` | Complete | 280 params, 357× compression, 100% |
| Frozen v2 | `frozen_6502_v2.py` | Complete | 720 params, 139× compression, 100% |

**Documentation**:
- `docs/ATOMIC_FUNCTIONS.md` - Pure math formulas
- `docs/FROZEN_6502.md` - Complete architecture
- `docs/OPCODE_MAP.md` - Opcode reference

**Key Discovery**: "Geometry is Computation"

The insight progression:
```
Flat MLP:        6,092 params → 3.4%   (wrong topology)
Ripple-carry:      611 params → 98.8%  (right topology)
Explicit XOR:       16 params → 100%   (right math)
Pure formula:        0 params → 100%   (math IS function)
```

**Architecture**:
- Level 0: Pure math primitives (XOR, AND, OR, NOT) - 0 params
- Level 1: Frozen shapes (16 topologies) - 0 params
- Level 2: Meaning layer (opcode routing) - ~2,500 params

**Compression**: ~100,000 (monolithic) → ~2,500 (frozen) = **40× reduction**

---

## Documentation

### Main Documentation (`docs/`)

| File | Status | Description |
|------|--------|-------------|
| `ARCHITECTURE.md` | Current | System architecture overview |
| `API.md` | Current | API reference for exported modules |
| `QUICKSTART.md` | Current | Getting started guide |
| `THEORY.md` | Current | Theoretical foundations |
| `BENCHMARKS.md` | Current | Performance benchmarks |
| `MESA11_UAT.md` | Complete | Unified Addressing Theory |
| `MESA12.md` | Partial | Guardian module overview |
| `MESA13_XOR_SUPERPOSITION.md` | Complete | XOR compression specification |
| `SDPMX_PIPELINE.md` | Complete | SDPMX operator sequence (Vi's Synthesis) |
| `SDPMX_VISUALIZER.md` | Complete | Ghost in the Machine visualizer |
| `ATOMIC_FUNCTIONS.md` | Complete | Pure math formulas for 6502 ops |
| `FROZEN_6502.md` | Complete | Frozen shapes architecture |
| `FROZEN_SHAPES.md` | Complete | Frozen shapes theory + API (Mesa 14) |
| `OPCODE_MAP.md` | Complete | 6502 opcode → frozen shape mapping |
| `LINCOLN_MANIFOLD_METHOD.md` | Complete | 4-phase exploration methodology (SOP) |
| `CONVERGENT_IDEAS.md` | Notes | Research convergence notes |
| `ACCESSIBILITY_PLAN.md` | Notes | Accessibility considerations |

### Legacy Documentation (`docs/legacy/`)

| File | Status | Description |
|------|--------|-------------|
| `MESA12_OBSERVER_ONTOLOGY.md` | Archive | Observer theory |
| `MESA12_HALO.md` | Archive | HALO pipeline design |
| `MESA12_REFLECTION.md` | Archive | Reflection mechanisms |
| `MESA12_ENGINEERING.md` | Archive | Engineering notes |

### Notes (`docs/notes/`)

| File | Status | Description |
|------|--------|-------------|
| `GRMAX.md` | Notes | GR-inspired routing ideas |
| `stocastic-deterministicFTW.md` | Notes | Stochastic vs deterministic routing |

### Root Documentation

| File | Status | Description |
|------|--------|-------------|
| `README.md` | Current | Project overview and quick start |
| `CHANGELOG.md` | Current | Version history (through v0.7.3) |
| `MASTER_INDEX.md` | **This file** | Complete repository catalog |

---

## Tests

**Location**: `tests/`

| Test File | Covers | Tests |
|-----------|--------|-------|
| `test_hierarchical.py` | HierarchicalTriXFFN | Comprehensive |
| `test_sparse_lookup.py` | SparseLookupFFN | Comprehensive |
| `test_sparse_lookup_v2.py` | SparseLookupFFNv2 | Comprehensive |
| `test_sparse.py` | SparseTriXFFN | Basic |
| `test_trix_ffn.py` | TriXFFN | Basic |
| `test_temporal_tiles.py` | TemporalTileLayer | Comprehensive |
| `test_compiled_dispatch.py` | CompiledDispatch | Basic |
| `test_xor_superposition.py` | XOR compression | Comprehensive (33 tests) |
| `test_guardian.py` | Guardian module | Basic |
| `test_qat.py` | QAT utilities | Basic |
| `test_kernel.py` | TriXLinear, packing | Basic |
| `test_spline.py` | Spline layers | Basic |
| `test_nn.py` | General NN tests | Mixed |
| `test_integration.py` | Integration tests | Mixed |
| `test_ab_harness.py` | A/B testing | Basic |
| `test_kan.py` | AdditiveKAN, HierarchicalKAN | 22 tests |
| `test_sdpmx.py` | SDPMX Pipeline | 21 tests |
| `test_sdpmx_rigorous.py` | SDPMX Pipeline (rigorous) | 34 tests |
| `test_viz/test_integration.py` | SDPMX Visualizer | 18 tests |
| `test_frozen.py` | Frozen Shapes infrastructure | 30+ tests |
| `test_frozen_6502.py` | Frozen 6502 (100% accuracy) | 40+ tests |
| `conftest.py` | Test fixtures | - |

**Not Tested**:
- All orphaned modules
- Compiler module (needs tests)
- spline2d.py, spline_adc.py
- hybrid_kan.py

---

## Version Timeline

| Version | Date | Milestone |
|---------|------|-----------|
| 0.1.0 | 2024-12-10 | Initial: TriXLinear, emergent routing |
| 0.2.0 | 2024-12-12 | Sparse training: SparseTriXFFN |
| 0.3.0 | 2024-12-13 | Hierarchical: HierarchicalTriXFFN |
| 0.4.0 | 2024-12-14 | SparseLookup: Routing IS computation |
| 0.5.0 | 2024-12-15 | CompiledDispatch, SparseLookupFFNv2 |
| 0.5.4 | 2024-12-16 | Temporal tiles (Mesa 4) |
| 0.6.0 | 2024-12-17 | SparseLookupFFNv4, XORRouter |
| 0.6.1 | 2024-12-18 | TriXO isolation |
| 0.7.0 | 2024-12-18 | Mesa 11: UAT |
| 0.7.1 | 2024-12-18 | Geometric framework |
| 0.7.2 | 2024-12-18 | All 8 UAT experiments confirmed |
| 0.7.3 | 2024-12-18 | λ-slider control |
| 0.12.0 | 2024-12-19 | Mesa 13: XOR Superposition (129× compression) |
| 0.13.0 | 2024-12-20 | Frozen 6502: Geometry as Computation (139× compression) |
| 0.14.0 | 2024-12-20 | Mesa 14: Frozen Shapes integrated into TriX (40× compression, 100% accuracy) |

---

## Unanswered Questions

1. **Why do v1, v2, v3, v4 of sparse_lookup coexist?**
   - v1: Original spline-based
   - v2: + Surgery API, regularization
   - v3: + Yang-Mills (orphaned)
   - v4: + SpatioTemporal (orphaned)
   - **Which should users prefer?** → v2 for most cases

2. **What is "Track B"?**
   - Alternative architecture path using Kolmogorov-Arnold representation
   - additive_kan → kan_hierarchical → hybrid_kan
   - Never completed or integrated

3. **Why was v4 SpatioTemporal added then orphaned?**
   - Added in v0.6.0
   - Never exported or tested
   - Possibly superseded by temporal_tiles.py

4. **Where's the Yang-Mills theory doc?**
   - Only in `sparse_lookup_v3.py` docstrings
   - Ambitious but unvalidated

5. **What's the relationship between Guardian and main training?**
   - Guardian is observer/advisor, not required
   - Experimental approach to adaptive training

---

## File Statistics

| Directory | Files | Lines | Purpose |
|-----------|-------|-------|---------|
| `src/trix/nn/` | 23 | ~11,100 | Neural network layers |
| `src/trix/compiler/` | 10 | ~3,600 | Circuit compiler |
| `src/trix/guardian/` | 6 | ~2,400 | Training observer |
| `src/trix/viz/` | 4 | ~810 | SDPMX Visualizer |
| `src/trix/qat/` | 2 | ~280 | Quantization |
| `src/trix/kernel/` | 2 | ~400 | Low-level ops |
| `tests/` | 23 | ~8,500 | Test suite |
| `experiments/` | 18 | ~5,500+ | Research experiments |
| `docs/` | 22 | ~4,000+ | Documentation |

**Total**: ~35,000+ lines of code

---

*"The map is not the territory, but it helps to have one."*
