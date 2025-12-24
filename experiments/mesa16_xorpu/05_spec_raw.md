# RAW: XORPU Spec Analysis

> Lincoln Manifold Method - Analyzing Forge Output
> Date: 2025-12-22

---

## The Data

```
15 shapes, 15 passing (100%)
2,432 total polynomial terms
Average 21.2 cycles per operation
Router params: 1,035
```

---

## Shape Breakdown

| Shape | Terms | Cycles | Category |
|-------|-------|--------|----------|
| nop   | 0     | 1      | Control |
| xor   | 32    | 5      | Logic (atom) |
| and   | 32    | 5      | Logic (atom) |
| or    | 64    | 5      | Logic |
| not   | 32    | 5      | Logic (atom) |
| add   | 256   | 37     | Arithmetic |
| sub   | 256   | 37     | Arithmetic |
| sll   | 320   | 41     | Shift |
| srl   | 320   | 41     | Shift |
| sra   | 352   | 41     | Shift |
| slt   | 280   | 42     | Compare |
| sltu  | 264   | 40     | Compare |
| nand  | 64    | 6      | Logic |
| nor   | 96    | 6      | Logic |
| xnor  | 64    | 6      | Logic |

---

## Observations

### 1. The Three Atoms Are Tiny
XOR, AND, NOT: 32 terms each, 5 cycles.
These are the primitives. Everything else composes from them.

### 2. Arithmetic Is Expensive
ADD/SUB: 256 terms, 37 cycles.
That's 8× the atoms. Ripple carry adds up.

### 3. Shifts Are Heaviest
SLL/SRL: 320 terms, 41 cycles.
SRA: 352 terms (needs sign extension).
Barrel shifter is term-heavy.

### 4. Comparisons Are Complex
SLT: 280 terms (signed comparison)
SLTU: 264 terms (unsigned)
Need subtraction + sign logic.

---

## Questions This Raises

1. **Can we reduce arithmetic terms?**
   - Carry-lookahead instead of ripple?
   - But CLA has different polynomial structure

2. **Can shifts be cheaper?**
   - Current: 5-stage barrel shifter
   - Each stage doubles terms
   - Alternative: lookup table? (But that's not polynomial)

3. **What's the term-to-gate ratio?**
   - In hardware, each term ≈ AND gates + accumulator
   - 2,432 terms total
   - Estimate: ~5K LUTs for full XORPU

4. **Memory footprint?**
   - Each term: coefficient (2 bits) + variable indices
   - ~10 bytes per term average
   - 2,432 × 10 = ~24KB for shape storage
   - Fits in BRAM easily

---

## Efficiency Analysis

### Most Efficient (terms per output bit)
- XOR: 32 terms / 32 bits = 1.0 terms/bit
- AND: 32 terms / 32 bits = 1.0 terms/bit
- NOT: 32 terms / 32 bits = 1.0 terms/bit

### Least Efficient
- SRA: 352 terms / 32 bits = 11.0 terms/bit
- SLL/SRL: 320 terms / 32 bits = 10.0 terms/bit
- SLT: 280 terms / 32 bits = 8.75 terms/bit

### Why Shifts Are Expensive
Barrel shifter: 5 stages × ~64 terms per stage = ~320 terms
Each stage is a row of MUXes: MUX = OR(AND(sel, a), AND(NOT(sel), b))

---

## Optimization Opportunities

### 1. Fused Operations
- INC = ADD with b=1 (could be simpler)
- DEC = SUB with b=1
- NEG = SUB with a=0

### 2. Parallel Term Evaluation
- All terms are independent
- 8-wide evaluation: 8× throughput
- Pipeline: hide latency

### 3. Shape Specialization
- 8-bit versions for byte operations
- Simpler shifts when shift amount is constant

### 4. XOR Compression
- Similar shapes share structure
- Store delta from base shape
- 129× compression demonstrated in Mesa 13

---

## What This Tells Us About Hardware

### PEU Design
- Need to handle 352 terms (max for SRA)
- Pipelined: 1 term/cycle = 352 cycles worst case
- 8-wide: ~44 cycles worst case

### Shape Memory
- 24KB for 15 shapes
- Scale: 100 shapes ≈ 160KB
- Fits in FPGA BRAM

### Router
- 1,035 params = ~4KB
- Trivial compared to shapes
- Could be hardwired (opcode → shape direct)

---

*End of RAW - Spec Analysis*
