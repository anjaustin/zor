# REFLECT: Forge-to-Production Tensions

> Lincoln Manifold Method - Resolving Implementation Gaps
> Date: 2025-12-22

---

## Tension 1: Software Accuracy vs Hardware Timing

**Problem:** Forge validates correctness but not timing.

**Resolution:** Add cycle-accurate modeling to Forge.

```python
# Current
result = xorpu.compute(a, b, "add")  # Correct, but timing unknown

# Proposed
result, cycles = xorpu.compute_timed(a, b, "add")
# Returns: (result, 37)  # 37 cycles for ADD
```

The spec already has cycle estimates. Surface them in API.

---

## Tension 2: Polynomial Terms vs Hardware Gates

**Problem:** Terms ≠ gates. Mapping is non-trivial.

**Resolution:** Add gate estimation model.

```
Term with degree 1: ~2 LUTs (coefficient + input)
Term with degree 2: ~3 LUTs (AND + coefficient)
Term with degree 3+: ~4 LUTs

Total estimate: terms × 2.5 LUTs
2,432 terms × 2.5 = ~6,000 LUTs
```

Add to spec export:
```json
{
  "hardware_estimate": {
    "luts": 6080,
    "ffs": 2432,
    "bram_kb": 24
  }
}
```

---

## Tension 3: Individual Shapes vs Fused Operations

**Problem:** Some operations naturally fuse (INC = ADD 1) but we treat them separately.

**Resolution:** Add fusion hints to spec.

```python
FUSED_OPS = {
    "inc": ("add", {"b": 1}),      # INC = ADD with b=1
    "dec": ("sub", {"b": 1}),      # DEC = SUB with b=1
    "neg": ("sub", {"a": 0}),      # NEG = SUB with a=0
    "mov": ("or", {"b": 0}),       # MOV = OR with b=0
}
```

Hardware can implement fused ops with reduced terms.

---

## Tension 4: 32-bit Fixed vs Variable Width

**Problem:** Spec is 32-bit only. Real use cases need 8/16/64/128.

**Resolution:** Parameterized shape generation.

```python
class XORPU:
    def __init__(self, bits: int = 32):
        self.bits = bits
        self._build_shapes()  # Generates shapes for specified width

# Usage
xorpu_8 = XORPU(bits=8)    # 8-bit, ~150 terms
xorpu_32 = XORPU(bits=32)  # 32-bit, ~2,400 terms
xorpu_64 = XORPU(bits=64)  # 64-bit, ~4,800 terms
```

Terms scale linearly with bit width (for ripple operations).

---

## Tension 5: Forge API vs Hardware Interface

**Problem:** Forge API is Pythonic. Hardware needs registers/memory-mapped.

**Resolution:** Dual interface in spec.

```python
# Software API (Forge)
result = xorpu.compute(a, b, "add")

# Hardware API (spec export)
spec.hardware_interface = {
    "registers": {
        "SRC1": {"offset": 0x08, "width": 32},
        "SRC2": {"offset": 0x0C, "width": 32},
        "SHAPE": {"offset": 0x10, "width": 8},
        "RESULT": {"offset": 0x14, "width": 32},
    },
    "protocol": "AXI4-Lite",
}
```

Forge generates both software reference AND hardware interface spec.

---

## Tension 6: Validation Depth

**Problem:** 2,000 random samples is fast but not exhaustive. 2^64 inputs is impossible.

**Resolution:** Tiered validation strategy.

```
Tier 1: Random sampling (2,000 samples) - FAST
  └── Catches gross errors
  └── 100% required to proceed

Tier 2: Edge cases (specific patterns) - MEDIUM
  └── All zeros, all ones, alternating
  └── Max positive, max negative
  └── Powers of 2

Tier 3: Exhaustive (8-bit equivalent) - SLOW
  └── 2^16 cases per operation
  └── Proves polynomial correctness
  └── Run once per shape definition

Tier 4: Formal verification - DEFINITIVE
  └── Prove polynomial = truth function
  └── Mathematical, not empirical
```

Current Forge does Tier 1. Production needs Tier 2-3.

---

## Tension 7: Term Representation

**Problem:** Current term format is implicit in code, not exportable.

**Resolution:** Explicit term format in spec.

```python
@dataclass
class Term:
    coefficient: int      # -2, -1, 0, 1, 2
    variables: List[int]  # Indices of input bits

@dataclass
class Shape:
    name: str
    input_bits: int
    output_bits: int
    terms: List[Term]     # Explicit polynomial

# Example: XOR bit 0
# a0 + b0 - 2*a0*b0
xor_bit0 = [
    Term(1, [0]),       # +a0
    Term(1, [32]),      # +b0 (b is offset by 32)
    Term(-2, [0, 32]),  # -2*a0*b0
]
```

This format is hardware-synthesizable.

---

## Tension 8: Test Coverage vs Test Speed

**Problem:** More tests = more confidence, but slower iteration.

**Resolution:** Test pyramid.

```
              /\
             /  \  Exhaustive (CI nightly)
            /____\
           /      \  Edge cases (CI per-commit)
          /________\
         /          \  Random sample (local dev)
        /______________\
```

Forge supports all three, developer chooses.

---

## Production-Ready Forge Checklist

After resolving tensions:

- [x] 100% accuracy on all shapes
- [ ] Cycle-accurate timing model
- [ ] Hardware gate estimation
- [ ] Fused operation hints
- [ ] Parameterized bit widths
- [ ] Hardware interface spec export
- [ ] Tiered validation (edge cases)
- [ ] Explicit term representation
- [ ] Formal verification hooks

**Status: 1/9 complete**

The Forge is functionally correct. It needs production polish.

---

*End of REFLECT - Spec Analysis*
