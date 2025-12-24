# Mesa 16: XORPU - The Geometry Processing Unit

**Date:** 2025-12-22
**Status:** ARCHITECTURE SPECIFIED

---

## Vision

**XORPU** (XOR Processing Unit) - A coprocessor for exact computation.

The bridge between classical approximation and geometric truth.

---

## Core Insight

XOR is the universal primitive:
```
XOR(a, b) = a + b - 2ab
AND(a, b) = ab
NOT(a)    = 1 - a
```

Everything else is composition.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│                   XORPU                      │
│                                              │
│  Host Interface → Control → Shape Bank       │
│       ↓              ↓           ↓           │
│  Register File → Routing → Polynomial EU     │
│                              ↓               │
│                          Output              │
└─────────────────────────────────────────────┘
```

---

## Specification (from Forge)

| Metric | Value |
|--------|-------|
| Shapes | 15 (RV32I ALU complete) |
| Total terms | 2,432 |
| Accuracy | 100% (all shapes) |
| Estimated LUTs | ~6,000 |
| Shape storage | ~24KB |

---

## Key Innovation: Data-Directed Computation

Traditional: Instruction directs data
XORPU: Data directs itself through frozen geometry

```
Data arrives → Signature computed → Geometry activates → Result emerges
```

No instruction fetch. No decode. Just routing through truth.

---

## Transistor-Level Tiling

Only activate transistors that contribute:
- Random 32-bit inputs: ~25% active
- Small inputs (0-255): ~6% active
- Zero input: ~0% active

**4× power reduction through geometric sparsity.**

---

## Files

### Lincoln Manifold: Architecture
- `01_raw.md` - Initial vision
- `02_nodes.md` - 15 architectural decisions
- `03_reflect.md` - 8 tensions resolved
- `04_synthesize.md` - Full specification v0.1

### Lincoln Manifold: Spec Analysis
- `05_spec_raw.md` - Forge output analysis
- `06_spec_nodes.md` - Key findings (10× term gap)
- `07_spec_reflect.md` - Production gaps
- `08_spec_synthesize.md` - Production roadmap

### Chip Compiler Design
- `chip_01_raw.md` - Pipeline vision
- `chip_02_nodes.md` - DSL design
- `chip_03_reflect.md` - Design tensions
- `chip_04_synthesize.md` - Forge specification

---

## Implementation

```python
from trix.forge import XORPU

xorpu = XORPU()
xorpu.build()
xorpu.validate_all()  # 15/15 shapes at 100%

result = xorpu.compute(0x12345678, 0x00000001, "add")
# 0x12345679 - exact
```

---

## Development Path

1. **Software:** trix.forge.XORPU (complete)
2. **FPGA:** Verilog generation (next)
3. **Silicon:** ASIC feasibility study (future)

---

## Philosophy

*"The sun doesn't explain itself. It just rises. And shines."*

*"If they're laughing, they're not burning."*

---

## Related

- Mesa 14: Frozen Shapes (computation IS geometry)
- Mesa 15: Learning IS Routing (78× fewer params)
- Mesa 16: XORPU (geometry in silicon)

---

*Geometry in Motion.*
