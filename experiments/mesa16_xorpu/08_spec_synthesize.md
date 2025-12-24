# SYNTHESIZE: Production Forge Specification

> Lincoln Manifold Method - Final Optimization Plan
> Date: 2025-12-22

---

## Executive Summary

The Forge produces correct XORPU specs. To be production-ready, it needs:
1. Hardware-oriented term export
2. Cycle/gate estimation
3. Multi-width support
4. Enhanced validation

**Timeline:** 1 sprint (2 weeks)

---

## 1. Term Export Format

### Current State
Terms are implicit in Python functions.

### Target State
```python
@dataclass
class TermSpec:
    coeff: int           # -2 to +2
    vars: Tuple[int]     # Variable indices

@dataclass
class ShapeSpec:
    id: int
    name: str
    input_bits: int
    output_bits: int
    terms: List[TermSpec]
    cycles: int

    def to_verilog(self) -> str:
        """Generate Verilog implementation."""

    def to_c(self) -> str:
        """Generate C implementation."""
```

### Export Formats
```
xorpu_spec.json     # Full specification
xorpu_shapes.v      # Verilog RTL
xorpu_shapes.h      # C header
xorpu_shapes.py     # Python reference
```

---

## 2. Hardware Estimation Model

### LUT Estimation
```python
def estimate_luts(shape: ShapeSpec) -> int:
    luts = 0
    for term in shape.terms:
        degree = len(term.vars)
        if degree == 0:
            luts += 1  # Constant
        elif degree == 1:
            luts += 2  # Input + coefficient
        elif degree == 2:
            luts += 3  # AND + coefficient
        else:
            luts += degree + 1  # Multi-input AND + coefficient
    return luts
```

### Cycle Estimation
```python
def estimate_cycles(shape: ShapeSpec, parallelism: int = 1) -> int:
    pipeline_overhead = 5  # Fetch, decode, setup, reduce, writeback
    term_cycles = math.ceil(len(shape.terms) / parallelism)
    return pipeline_overhead + term_cycles
```

### Power Estimation
```python
def estimate_power_mw(shape: ShapeSpec, freq_mhz: int = 200) -> float:
    # Rough model: 0.1 mW per LUT at 200 MHz
    luts = estimate_luts(shape)
    return luts * 0.1 * (freq_mhz / 200)
```

---

## 3. Multi-Width Support

### Parameterized Shapes
```python
class XORPU:
    def __init__(self, bits: int = 32):
        self.bits = bits
        self.shapes = self._generate_shapes()

    def _generate_shapes(self) -> List[ShapeSpec]:
        return [
            self._make_xor(),
            self._make_and(),
            self._make_add(),
            # ... parameterized by self.bits
        ]
```

### Width Presets
```python
XORPU_8 = XORPU(8)    # Embedded, byte ops
XORPU_16 = XORPU(16)  # Legacy, DSP
XORPU_32 = XORPU(32)  # Standard (RV32)
XORPU_64 = XORPU(64)  # Modern (RV64)
```

### Scaling Estimates
| Width | Terms | LUTs | Cycles (ADD) |
|-------|-------|------|--------------|
| 8     | 152   | 380  | 13           |
| 16    | 608   | 1520 | 21           |
| 32    | 2432  | 6080 | 37           |
| 64    | 9728  | 24K  | 69           |

---

## 4. Enhanced Validation

### Tier 1: Quick (default)
```python
xorpu.validate(mode="quick")  # 1,000 random samples
```

### Tier 2: Edge Cases
```python
EDGE_CASES = [
    (0x00000000, 0x00000000),  # All zeros
    (0xFFFFFFFF, 0xFFFFFFFF),  # All ones
    (0xAAAAAAAA, 0x55555555),  # Alternating
    (0x80000000, 0x00000001),  # Sign bit + LSB
    (0x7FFFFFFF, 0x00000001),  # Max positive + 1
    # ... 50 more cases
]

xorpu.validate(mode="edge")  # Targeted cases
```

### Tier 3: Exhaustive (8-bit proxy)
```python
xorpu.validate(mode="exhaustive")
# For 8-bit: 2^16 = 65,536 cases per op
# For 32-bit: Uses 8-bit exhaustive + 32-bit sampling
```

### Tier 4: Formal (external tool)
```python
xorpu.export_smt("shapes.smt2")  # For Z3/CVC5 verification
```

---

## 5. Hardware Interface Spec

### Memory Map
```python
XORPU_MEMORY_MAP = {
    "CTRL":      {"offset": 0x00, "width": 32, "access": "RW"},
    "STATUS":    {"offset": 0x04, "width": 32, "access": "RO"},
    "SRC1":      {"offset": 0x08, "width": 32, "access": "RW"},
    "SRC2":      {"offset": 0x0C, "width": 32, "access": "RW"},
    "SHAPE_ID":  {"offset": 0x10, "width": 8,  "access": "RW"},
    "RESULT":    {"offset": 0x14, "width": 32, "access": "RO"},
    "RESULT_HI": {"offset": 0x18, "width": 32, "access": "RO"},
    "CYCLES":    {"offset": 0x1C, "width": 16, "access": "RO"},
}
```

### Register Definitions
```python
CTRL_BITS = {
    "START": 0,
    "RESET": 1,
    "MODE":  2,  # 0=32-bit, 1=64-bit
}

STATUS_BITS = {
    "READY": 0,
    "DONE":  1,
    "ERROR": 2,
}
```

### Export
```python
xorpu.export_hwspec("xorpu_hwspec.json")
xorpu.export_svd("xorpu.svd")  # CMSIS-SVD for tooling
```

---

## 6. Verilog Generation

### Template
```verilog
module xorpu_shape_add (
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] result
);
    // Generated polynomial evaluation
    // 256 terms for 32-bit add
    wire [31:0] term_0 = a[0];
    wire [31:0] term_1 = b[0];
    wire [31:0] term_2 = a[0] & b[0];
    // ... 253 more terms ...

    assign result = term_0 ^ term_1 ^ ... ; // XOR reduction
endmodule
```

### Generation API
```python
xorpu.export_verilog("rtl/xorpu_shapes.v")
```

---

## 7. Implementation Plan

### Week 1: Core
- [ ] TermSpec dataclass with export
- [ ] Hardware estimation functions
- [ ] Multi-width XORPU class
- [ ] JSON/Verilog/C export

### Week 2: Polish
- [ ] Enhanced validation tiers
- [ ] Hardware interface spec
- [ ] SVD generation
- [ ] Documentation

---

## 8. API Changes

### Before (Current)
```python
from trix.forge.xorpu_spec import XORPU

xorpu = XORPU()
xorpu.build()
xorpu.validate()
result = xorpu.compute(a, b, "add")
```

### After (Production)
```python
from trix.forge import XORPU

# Multi-width support
xorpu = XORPU(bits=32)

# Build with options
xorpu.build(
    include_fused=True,     # INC, DEC, NEG
    optimize_atoms=True,    # Hardwire XOR/AND/NOT
)

# Tiered validation
xorpu.validate(mode="edge")

# Hardware estimation
print(xorpu.estimate())
# LUTs: 6080, FFs: 2432, BRAM: 24KB, Power: 608mW

# Rich export
xorpu.export(
    json="xorpu_spec.json",
    verilog="rtl/",
    c_header="include/",
    svd="xorpu.svd",
)

# Compute with timing
result, cycles = xorpu.compute_timed(a, b, "add")
```

---

## 9. Success Criteria

| Metric | Target |
|--------|--------|
| All shapes 100% | Yes |
| Verilog synthesis | Clean |
| LUT estimate accuracy | ±20% |
| Cycle estimate accuracy | ±10% |
| Multi-width support | 8/16/32/64 |
| Export formats | JSON, Verilog, C, SVD |

---

## 10. The Deliverable

**trix.forge.XORPU: Production-Ready Chip Specification Tool**

Input: Bit width, options
Output: Complete hardware specification including:
- Polynomial terms (explicit)
- Verilog RTL
- C reference
- Hardware interface (memory map, registers)
- Timing/area/power estimates
- Validation report

The Forge becomes the single source of truth for XORPU silicon.

---

*End of SYNTHESIZE - Production Specification*

*The blade is sharpened. Ready for silicon.*
