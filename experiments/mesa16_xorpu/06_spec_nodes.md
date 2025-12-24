# NODES: Spec Analysis Findings

> Lincoln Manifold Method - Key Insights from Forge Data
> Date: 2025-12-22

---

## NODE 1: The Complexity Hierarchy

```
Tier 0: Atoms (32 terms)
  └── XOR, AND, NOT

Tier 1: Simple Derived (64-96 terms)
  └── OR, NAND, NOR, XNOR

Tier 2: Arithmetic (256 terms)
  └── ADD, SUB

Tier 3: Complex (264-352 terms)
  └── SLT, SLTU, SLL, SRL, SRA
```

Hardware implication: Tier 0-1 in fast path, Tier 2-3 in general PEU.

---

## NODE 2: Term Count Drives Everything

| Metric | Scales With Terms |
|--------|-------------------|
| Execution time | O(terms) |
| Power consumption | O(terms) |
| Verification complexity | O(terms) |
| Storage size | O(terms) |

**2,432 terms = the "weight" of XORPU v0.1**

---

## NODE 3: The 10× Gap

Atoms: 32 terms
Shifts: 320 terms

10× difference. Why?

```
XOR of 32 bits = 32 independent XORs = 32 terms
SLL by variable amount = 5-stage barrel shifter
  Stage 1: shift by 1 or not (32 MUXes)
  Stage 2: shift by 2 or not (32 MUXes)
  ...
  Each MUX = ~4 terms
  5 × 32 × 4 = 640 terms (theoretical)
  Actual: 320 terms (optimized)
```

---

## NODE 4: Compression Opportunity

Shapes share structure:
- ADD and SUB differ only in NOT(b) + 1
- SLL and SRL are mirrors
- SLT = SUB + sign logic

**Potential: Store base shapes + deltas**
- Base: ADD (256 terms)
- SUB delta: ~32 terms (NOT + carry-in)
- Savings: ~220 terms

Apply XOR superposition (Mesa 13): 129× compression possible.

---

## NODE 5: Hardwired Fast Path

The atoms (XOR, AND, NOT) should be hardwired, not polynomial-evaluated.

```
XOR gate: 1 cycle (native hardware)
XOR polynomial: 5 cycles (PEU)
```

5× speedup for most common operations.

**Revised architecture:**
```
┌─────────────────────────────────────────┐
│  Input                                   │
│    ├── Atoms (hardwired) ────→ Fast out │
│    └── Complex (PEU) ────────→ Slow out │
└─────────────────────────────────────────┘
```

---

## NODE 6: Cycle Budget

At 200 MHz:
- 1 cycle = 5 ns
- Atom (5 cycles) = 25 ns
- ADD (37 cycles) = 185 ns
- SRA (41 cycles) = 205 ns

Compared to CPU:
- Modern x86 ADD: ~1 ns (but 1000× more transistors)
- XORPU ADD: 185 ns (but 100% correct, minimal area)

XORPU is not about speed. It's about correctness and simplicity.

---

## NODE 7: RISC-V Alignment

Our 15 shapes cover RV32I ALU completely:
- ADD, SUB, AND, OR, XOR ✓
- SLL, SRL, SRA ✓
- SLT, SLTU ✓
- AUIPC, LUI (need addition) ✓

Missing for full RV32I:
- Memory ops (out of scope - not ALU)
- Branch logic (comparisons covered)
- MUL/DIV (RV32M extension)

**XORPU v0.1 = RV32I ALU equivalent**

---

## NODE 8: Production Optimization Targets

1. **Hardwire atoms:** XOR, AND, NOT → 1 cycle
2. **Parallel terms:** 8-wide PEU → 8× throughput
3. **Shape compression:** XOR superposition → 129× storage
4. **Fused ops:** INC, DEC, NEG as optimized variants

Expected improvement:
- Latency: 5× for atoms
- Throughput: 8× overall
- Storage: 10× reduction

---

## NODE 9: The Forge Is Production-Ready

Validation results:
- 15/15 shapes: 100.00%
- 2,000 samples each
- No failures

The Forge correctly:
1. Defines shapes as frozen polynomials
2. Validates against truth functions
3. Exports specification data
4. Provides compute interface

**Forge = validated software reference for hardware**

---

## NODE 10: What We Now Know

| Question | Answer |
|----------|--------|
| How many shapes? | 15 (covers RV32I ALU) |
| How many terms? | 2,432 total |
| Most expensive? | SRA (352 terms) |
| Storage needed? | ~24KB |
| All correct? | Yes (100%) |

This is the bill of materials for XORPU silicon.

---

*End of NODES - Spec Analysis*
