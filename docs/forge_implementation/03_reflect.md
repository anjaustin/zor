# Reflections: Forge Pipeline Implementation

## Core Insight

**Combinational freezing is not AI. It's algebra.**

We don't need to train anything for combinational logic. The polynomials compose deterministically. Given a netlist of AND/OR/XOR/NOT gates, we can derive the exact polynomial for each output.

This is simpler than I thought. The "neural" part of TriX is for DISCOVERING efficient shapes (like we did with the 6502). For APPLYING known shapes, it's just math.

The Forge Stage 1 is essentially a compiler:
```
Verilog → Netlist → Polynomial → C code
```

No training. No inference. Just transformation.

---

## Resolved Tensions

### Tension 1: Polynomial Explosion vs Tractability

**Resolution: Defer shape recognition to Stage 3.**

For Stage 1, we accept that deeply nested circuits will have large polynomials. This is fine for small circuits (< 100 gates). We'll add pattern recognition in Stage 3 to use optimized shapes.

The key insight: even "exploded" polynomials evaluate fast. The terms are many but each term is just multiplication and addition. We're not trying to minimize polynomial size - we're trying to maximize evaluation speed.

A polynomial with 1000 terms still evaluates in microseconds.

### Tension 2: Yosys Dependency vs Pure Python

**Resolution: Embrace the dependency.**

Yosys is a 10MB install. It's available on every Linux distro. The alternative is months of parser development.

Dependencies are fine when they're:
1. Stable (Yosys is 10+ years old)
2. Focused (it does one thing well)
3. Replaceable (we interface via JSON, could swap later)

We subprocess to Yosys and move on.

### Tension 3: Bit-Level vs Word-Level

**Resolution: Bit-blast everything in Stage 1.**

A 32-bit adder becomes 32 individual bit operations. This is inefficient but CORRECT.

Stage 3 will add word-level recognition: "these 32 bits are a ripple adder, use the ripple_add shape."

Correctness first. Optimization second.

---

## What I Now Understand

### The Three Representations

1. **Verilog**: Human-readable, hierarchical, word-level
2. **Netlist**: Flat, bit-level, graph of primitives
3. **Frozen C**: Evaluation-ready, polynomial operations

The Forge transforms between them:
```
Verilog ──[Yosys]──► Netlist ──[Composer]──► Frozen C
```

### The Composition Algorithm (Detailed)

```python
def compose(netlist):
    # 1. Initialize input polynomials
    for input_port in netlist.inputs:
        poly[input_port] = Variable(input_port.name)

    # 2. Topological sort cells
    cells_sorted = topological_sort(netlist.cells)

    # 3. Compose each cell
    for cell in cells_sorted:
        inputs = [poly[conn] for conn in cell.input_connections]
        poly[cell.output] = cell.type.polynomial(*inputs)

    # 4. Extract output polynomials
    return {port: poly[port] for port in netlist.outputs}
```

### The Polynomial Types

For bit-level operations, inputs are in {0, 1}. Outputs are in {0, 1}.

| Gate | Polynomial | Output Range |
|------|------------|--------------|
| NOT(a) | 1 - a | {0, 1} |
| AND(a,b) | ab | {0, 1} |
| OR(a,b) | a + b - ab | {0, 1} |
| XOR(a,b) | a + b - 2ab | {0, 1} |
| NAND(a,b) | 1 - ab | {0, 1} |
| NOR(a,b) | 1 - a - b + ab | {0, 1} |
| XNOR(a,b) | 1 - a - b + 2ab | {0, 1} |

These are closed under composition for {0,1} inputs. The output is always 0 or 1.

### C Code Generation Strategy

Instead of generating polynomial strings and parsing them, generate C code directly:

```c
// For XOR(a, AND(b, c)):
// = a + (b*c) - 2*a*(b*c)
// = a + bc - 2abc

static inline uint8_t frozen_output_0(uint8_t a, uint8_t b, uint8_t c) {
    return a + b*c - 2*a*b*c;
}
```

Actually, for {0,1} inputs, we can use integer arithmetic directly. No floating point needed.

### Verification Strategy

For N-bit inputs, exhaustive testing is O(2^N).

| Input Bits | Test Cases | Time @ 1M/sec |
|------------|------------|---------------|
| 8 | 256 | <1 ms |
| 12 | 4,096 | <5 ms |
| 16 | 65,536 | <100 ms |
| 20 | 1,048,576 | ~1 sec |
| 24 | 16,777,216 | ~17 sec |
| 32 | 4,294,967,296 | ~72 min |

For Stage 1 (small circuits), exhaustive testing is tractable.

---

## The Implementation Plan

### File Structure

```
foundry/
└── forge/
    ├── __init__.py
    ├── yosys.py      # Yosys interface (subprocess)
    ├── netlist.py    # Netlist parsing and graph
    ├── composer.py   # Polynomial composition
    ├── emitter.py    # C code generation
    └── verifier.py   # Exhaustive testing
```

### Module Responsibilities

**yosys.py**
- `synthesize(verilog_path) -> json_path`
- Shell out to Yosys, return netlist JSON

**netlist.py**
- `parse(json_path) -> Netlist`
- `Netlist.cells`, `Netlist.connections`, `Netlist.ports`
- `topological_sort(cells) -> List[Cell]`

**composer.py**
- `compose(netlist) -> Dict[str, Expression]`
- Builds output expressions from gate primitives

**emitter.py**
- `emit_c(module_name, expressions) -> (header, source)`
- Generates frozen C code

**verifier.py**
- `verify_exhaustive(frozen_func, reference_func, input_bits) -> bool`
- Tests all input combinations

### Test Cases for Stage 1

1. **NOT gate**: 1-bit input, 1-bit output
2. **AND gate**: 2-bit input, 1-bit output
3. **XOR gate**: 2-bit input, 1-bit output
4. **Half adder**: 2-bit input, 2-bit output (sum, carry)
5. **Full adder**: 3-bit input, 2-bit output
6. **4-bit adder**: 8-bit input, 5-bit output
7. **8-bit adder**: 16-bit input, 9-bit output

---

## Remaining Questions

1. How do we handle constants (VCC, GND) in the netlist?
2. What about buffers (BUF cells)?
3. How do we name outputs consistently?

These are implementation details. We'll handle them as they arise.

---

## The One-Sentence Summary

**Stage 1 of the Forge is a deterministic compiler from Verilog combinational logic to frozen C polynomials, verified by exhaustive testing.**
