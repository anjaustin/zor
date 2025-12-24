# XORPU Production Roadmap

**Neural-Geometric Distributed Processing Fabric**

> From proof-of-concept to silicon-ready specification

**Version:** 2.0 (Reordered by Fundamental Importance)
**Updated:** 2025-12-22

---

## Current State (A)

| Component | Status |
|-----------|--------|
| Frozen Geometry | Proven (Mesa 14) |
| Learning IS Routing | Proven (Mesa 15) |
| XORPU Spec | Complete (Mesa 16) |
| Shapes | 15 @ 100% accuracy |
| Terms | 2,432 total |
| Tests | 1027 passing (none for forge) |

**What works:**
- `trix.forge.Chip` - Declarative chip specification
- `trix.forge.XORPU` - 15 shapes, all validated
- Compute interface - Exact results

**What's missing:**
- Explicit term representation (implicit in code)
- Test coverage for trix.forge
- Export pipeline to hardware
- Production validation
- Multi-width support

---

## Target State (Z)

A complete, validated specification that can be:
1. Introspected (explicit terms, not hidden in functions)
2. Tested (comprehensive, tiered validation)
3. Exported (JSON, Verilog, C)
4. Verified (formal proofs)
5. Synthesized (FPGA, ASIC)

---

## Phase Execution Order

**Ordered by fundamental importance (keystone first):**

| Phase | Name | Why This Order |
|-------|------|----------------|
| 1 | Explicit Terms | Unlocks everything - can't export what you can't see |
| 2 | Test Suite | Validates terms are correct before building on them |
| 3 | JSON Export | Machine-readable spec enables tooling |
| 4 | Tiered Validation | Confidence before hardware commitment |
| 5 | Verilog Export | The silicon target |
| 6 | Hardware Model | Estimates for planning |
| 7 | FPGA Validation | Proof on real hardware |
| 8 | Multi-Width | Generalization (8/16/32/64-bit) |
| 9 | Optimization | Polish (fast path, parallel, compression) |

Each phase receives full Lincoln Manifold treatment:
**RAW → NODES → REFLECT → SYNTHESIZE → IMPLEMENT**

---

## Phase 1: Explicit Terms

**The Keystone**

> You cannot export what you cannot see.

### Goal
Transform implicit polynomial computation into explicit, introspectable data structures.

### Current State
```python
# Terms are hidden inside functions
def add_bit(a_bits, b_bits, bit_idx):
    # Polynomial logic embedded in code
    # Not exportable, not introspectable
```

### Target State
```python
@dataclass(frozen=True)
class Term:
    coefficient: int       # -2, -1, 1, 2
    variables: Tuple[int, ...]  # Input bit indices (sorted)

@dataclass
class ShapeSpec:
    name: str
    input_bits: int
    output_bits: int
    terms: List[List[Term]]  # terms[output_bit] = list of terms

    def term_count(self) -> int: ...
    def to_polynomial_str(self) -> str: ...
    def evaluate(self, a: int, b: int) -> int: ...
```

### Tasks
- [ ] Create `trix.forge.term` module
- [ ] Define Term dataclass (frozen, hashable)
- [ ] Define ShapeSpec dataclass
- [ ] Implement polynomial string representation
- [ ] Implement evaluation from explicit terms
- [ ] Refactor XORPU to generate ShapeSpecs
- [ ] Verify term counts match current spec (2,432 total)
- [ ] Add term introspection API

### Success Criteria
- All 15 shapes represented as explicit ShapeSpecs
- Term count verified: 2,432 total
- `shape.evaluate(a, b)` matches `xorpu.compute(a, b, shape_name)`
- Terms are introspectable: `shape.terms[0]` returns actual Term objects

### Working Directory
`tmp/xorpu_production/phase_01_terms/`

---

## Phase 2: Test Suite

**Validate the Foundation**

> Tests are the contract. Without them, nothing is guaranteed.

### Goal
Comprehensive pytest coverage for trix.forge module.

### Test Files
```
tests/test_forge_term.py     - Term dataclass, operations
tests/test_forge_shape.py    - ShapeSpec, evaluation
tests/test_forge_xorpu.py    - All 15 shapes, validation
tests/test_forge_chip.py     - Chip DSL, compilation, routing
```

### Test Categories
1. **Unit tests** - Term creation, ShapeSpec methods
2. **Integration tests** - XORPU shape generation
3. **Property tests** - Hypothesis-based fuzzing
4. **Edge case tests** - Zeros, ones, overflow, alternating

### Tasks
- [ ] Create test_forge_term.py
- [ ] Create test_forge_shape.py
- [ ] Create test_forge_xorpu.py
- [ ] Create test_forge_chip.py
- [ ] Add hypothesis property tests
- [ ] Add edge case test suite
- [ ] Achieve >90% coverage

### Success Criteria
- All tests passing
- Coverage >90% for trix.forge
- Edge cases explicitly tested
- Property tests find no failures

### Working Directory
`tmp/xorpu_production/phase_02_tests/`

---

## Phase 3: JSON Export

**Machine-Readable Specification**

> If a machine can't read it, a machine can't build it.

### Goal
Export ShapeSpecs to JSON for tooling, storage, and interchange.

### Target Format
```json
{
  "version": "1.0.0",
  "name": "xorpu_32",
  "bits": 32,
  "shapes": [
    {
      "id": 0,
      "name": "add",
      "input_bits": 64,
      "output_bits": 32,
      "term_count": 256,
      "terms": [
        [
          {"coeff": 1, "vars": [0]},
          {"coeff": 1, "vars": [32]},
          {"coeff": -2, "vars": [0, 32]}
        ]
      ]
    }
  ],
  "total_terms": 2432,
  "generated": "2025-12-22T00:00:00Z"
}
```

### Tasks
- [ ] Define JSON schema (jsonschema)
- [ ] Implement ShapeSpec.to_dict()
- [ ] Implement XORPU.export_json(path)
- [ ] Implement XORPU.from_json(path) - roundtrip
- [ ] Add schema validation
- [ ] Version the format

### Success Criteria
- Roundtrip: export → import → export produces identical output
- Schema validates all exports
- Format is human-readable and machine-parseable

### Working Directory
`tmp/xorpu_production/phase_03_json/`

---

## Phase 4: Tiered Validation

**Confidence Ladder**

> Random sampling catches gross errors. Edge cases catch subtle ones. Formal proofs catch all.

### Goal
Multiple validation tiers from fast to exhaustive.

### Validation Tiers

| Tier | Method | Coverage | Speed |
|------|--------|----------|-------|
| 0 | Quick | 1,000 random | <1s |
| 1 | Edge | 100 specific patterns | <1s |
| 2 | Exhaustive-8 | 2^16 (8-bit proxy) | ~10s |
| 3 | Exhaustive-32 | 2^64 (impossible) | N/A |
| 4 | Formal | SMT proof | varies |

### Edge Cases
```python
EDGE_CASES = [
    # Zeros and ones
    (0x00000000, 0x00000000),
    (0xFFFFFFFF, 0xFFFFFFFF),
    (0x00000000, 0xFFFFFFFF),

    # Alternating patterns
    (0xAAAAAAAA, 0x55555555),
    (0x55555555, 0xAAAAAAAA),

    # Powers of two
    (0x00000001, 0x00000001),
    (0x80000000, 0x80000000),

    # Overflow boundaries
    (0x7FFFFFFF, 0x00000001),  # Max positive + 1
    (0xFFFFFFFF, 0x00000001),  # -1 + 1

    # Single bits
    *[(1 << i, 1 << j) for i in range(32) for j in range(32) if i <= j],
]
```

### Tasks
- [ ] Implement validate(mode="quick")
- [ ] Implement validate(mode="edge")
- [ ] Implement validate(mode="exhaustive") for 8-bit
- [ ] Research SMT-LIB format
- [ ] Implement export_smt() for atoms
- [ ] Document validation coverage

### Success Criteria
- All tiers passing for all 15 shapes
- 8-bit exhaustive proves polynomial correctness
- At least XOR shape has SMT proof

### Working Directory
`tmp/xorpu_production/phase_04_validation/`

---

## Phase 5: Verilog Export

**The Silicon Target**

> Verilog is the bridge to physical reality.

### Goal
Generate synthesizable Verilog RTL from ShapeSpecs.

### Target Output
```verilog
// Auto-generated by trix.forge
// Shape: add (32-bit)
// Terms: 256

module xorpu_add (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] result
);
    // Bit 0: a[0] + b[0] - 2*a[0]*b[0]
    wire t0_0 = a[0];
    wire t0_1 = b[0];
    wire t0_2 = a[0] & b[0];
    assign result[0] = t0_0 ^ t0_1 ^ t0_2 ^ t0_2;  // XOR reduction

    // ... remaining bits ...
endmodule
```

### Tasks
- [ ] Implement ShapeSpec.to_verilog()
- [ ] Handle coefficient mapping (-2 → double XOR)
- [ ] Generate per-shape modules
- [ ] Generate top-level XORPU wrapper
- [ ] Generate testbench
- [ ] Validate with Icarus Verilog (if available)

### Success Criteria
- Verilog compiles without errors
- Testbench passes for all shapes
- Output matches Python reference

### Working Directory
`tmp/xorpu_production/phase_05_verilog/`

---

## Phase 6: Hardware Model

**Estimation Before Fabrication**

> Measure twice, synthesize once.

### Goal
Accurate resource and timing estimates.

### Models

```python
@dataclass
class HardwareEstimate:
    luts: int           # Lookup tables
    ffs: int            # Flip-flops
    bram_kb: float      # Block RAM
    cycles: int         # Execution cycles
    power_mw: float     # Estimated power

def estimate_luts(shape: ShapeSpec) -> int:
    """
    LUT estimation model:
    - Degree 0 (constant): 0 LUTs
    - Degree 1 (single var): 1 LUT
    - Degree 2 (AND): 1 LUT
    - Degree 3+: ceil(degree/4) LUTs (6-LUT architecture)
    - XOR reduction: log2(terms) LUTs per bit
    """
```

### Tasks
- [ ] Implement LUT estimation
- [ ] Implement FF estimation
- [ ] Implement cycle estimation
- [ ] Implement power estimation (rough)
- [ ] Add estimate() method to ShapeSpec
- [ ] Add estimate_total() to XORPU
- [ ] Document model assumptions

### Success Criteria
- Estimates produced for all shapes
- Model documented with assumptions
- (Future: validate against actual synthesis)

### Working Directory
`tmp/xorpu_production/phase_06_hardware_model/`

---

## Phase 7: FPGA Validation

**Proof on Silicon**

> The map is not the territory. Synthesize to know.

### Goal
Validate at least one shape on real FPGA.

### Target
- FPGA: Lattice iCE40 (open toolchain) or Xilinx Artix-7
- Shape: ADD (most complex commonly used)
- Validation: Compare against Python reference

### Tasks
- [ ] Select target FPGA and toolchain
- [ ] Synthesize ADD shape
- [ ] Compare actual LUTs to estimate
- [ ] Create test harness
- [ ] Validate functional correctness
- [ ] Document synthesis flow

### Success Criteria
- ADD shape synthesizes
- LUT count within ±20% of estimate
- All test vectors pass
- Reproducible build flow

### Working Directory
`tmp/xorpu_production/phase_07_fpga/`

---

## Phase 8: Multi-Width

**Generalization**

> If it works for 32 bits, make it work for all widths.

### Goal
Parameterized shapes for 8/16/32/64-bit.

### Scaling Model
| Width | Expected Terms | Cycles (ADD) |
|-------|---------------|--------------|
| 8     | ~150          | ~13          |
| 16    | ~600          | ~21          |
| 32    | ~2,400        | ~37          |
| 64    | ~9,700        | ~69          |

### Tasks
- [ ] Parameterize shape generation by bit width
- [ ] Implement XORPU(bits=8)
- [ ] Implement XORPU(bits=16)
- [ ] Implement XORPU(bits=64)
- [ ] Validate all widths
- [ ] Document scaling characteristics
- [ ] Add width-specific optimizations

### Success Criteria
- All widths generate correct shapes
- Term counts match scaling model (±10%)
- All widths pass tiered validation

### Working Directory
`tmp/xorpu_production/phase_08_multiwidth/`

---

## Phase 9: Optimization

**Production Polish**

> Make it work, make it right, make it fast.

### Goal
Performance optimizations for production use.

### Optimizations

#### 9.1 Hardwired Fast Path
```
Atoms (XOR, AND, NOT): 1 cycle (bypass polynomial evaluation)
Complex: N cycles (full evaluation)
```

#### 9.2 Parallel Term Evaluation
```
Serial:   37 cycles for ADD
8-wide:   ~10 cycles
16-wide:  ~7 cycles
```

#### 9.3 Shape Compression
```
Full storage: 2,432 terms × 8 bytes = 19KB
XOR superposition: 129× compression = ~150 bytes
```

### Tasks
- [ ] Implement atom bypass
- [ ] Design parallel evaluation unit
- [ ] Implement configurable parallelism
- [ ] Implement shape compression
- [ ] Benchmark all optimizations
- [ ] Trade-off analysis

### Success Criteria
- Atoms execute in 1 cycle
- Parallel evaluation achieves target speedup
- Compression ratio documented

### Working Directory
`tmp/xorpu_production/phase_09_optimization/`

---

## Milestone Summary

| Milestone | Phases | Key Deliverable |
|-----------|--------|-----------------|
| M1 | 1-2 | Explicit terms + tests |
| M2 | 3 | JSON export |
| M3 | 4 | Tiered validation |
| M4 | 5-6 | Verilog + estimates |
| M5 | 7 | FPGA proof |
| M6 | 8 | Multi-width |
| M7 | 9 | Optimized production |

---

## Success Criteria (Overall)

| Metric | Target |
|--------|--------|
| Shape accuracy | 100% (all shapes, all widths) |
| Test coverage | >90% for trix.forge |
| Verilog synthesis | Clean (no warnings) |
| LUT estimate accuracy | ±20% vs actual |
| Formal verification | At least atoms proven |
| FPGA validation | At least ADD working |

---

## Session Tracking

Progress is tracked in: `tmp/xorpu_production/SESSION_LOG.md`

Each phase has a working directory: `tmp/xorpu_production/phase_NN_name/`

Lincoln Manifold documents are archived to: `experiments/xorpu_production/`

---

## Related Work

- [Fungible Computation](https://github.com/anjaustin/fungible-computation) - Research paper
- Mesa 14: Frozen Shapes - Computation IS geometry
- Mesa 15: Learning IS Routing - 78× fewer parameters
- Mesa 16: XORPU - Geometry in silicon

---

*The blade is sharpened. Ready for silicon.*

*Geometry in Motion.*
