# Nodes of Interest: Forge Pipeline Implementation

## Node 1: Yosys as Frontend
**Use industrial-strength tooling, don't reinvent parsing.**

Yosys is battle-tested by the open-source FPGA community. It handles Verilog edge cases we'd never think of. It can output JSON netlists.

Why it matters: We focus on freezing, not parsing. Separation of concerns.

Command: `yosys -p "read_verilog X.v; synth -flatten; abc -g AND,OR,XOR,NOT; write_json X.json"`

---

## Node 2: Polynomial Composition is Deterministic
**For combinational logic, no neural network needed.**

We know the polynomials:
- AND(a,b) = ab
- OR(a,b) = a + b - ab
- XOR(a,b) = a + b - 2ab
- NOT(a) = 1 - a

Compose them through the netlist graph. The result is exact.

Why it matters: Combinational freezing is CLOSED FORM. No training, no approximation.

Tension with Node 3: Composition can explode.

---

## Node 3: Polynomial Explosion
**Naive composition creates exponentially many terms.**

A 32-bit adder naively composed: astronomical.

Resolution: Use SHAPES. Recognize patterns (adder, mux, compare) and use pre-built frozen shapes instead of gate-level expansion.

Why it matters: Tractability requires abstraction.

---

## Node 4: Sequential = Combinational + State
**Synchronous circuits decompose cleanly.**

```
inputs + state_now → [combinational logic] → outputs + state_next
```

Freeze the combinational part. Runtime updates state.

Why it matters: This is exactly how the 6502 works. Generalization, not special case.

---

## Node 5: The Netlist Graph
**Yosys JSON gives us cells and connections.**

Structure:
```json
{
  "modules": {
    "adder4": {
      "cells": {
        "cell_1": { "type": "AND", "connections": {...} },
        ...
      },
      "netnames": {...}
    }
  }
}
```

We parse this, build a directed graph, topologically sort, compose.

Why it matters: This is the data structure we operate on.

---

## Node 6: Bit-Blasting
**Multi-bit signals become individual wires.**

`input [3:0] a` becomes `a[0], a[1], a[2], a[3]`.

Yosys handles this. We just see individual bits in the netlist.

Why it matters: Simplifies our logic. Everything is single-bit at the gate level.

---

## Node 7: Stage 1 Scope
**Combinational only. No registers. No memories.**

Input: Pure combinational Verilog
Output: Frozen C function

```c
void frozen_adder4(uint8_t a, uint8_t b, uint8_t *sum);
```

Why it matters: Smallest useful increment. Ship something.

---

## Node 8: Verification by Exhaustive Testing
**Small circuits can be fully verified.**

4-bit adder: 256 × 256 = 65,536 cases. Test all of them.

Compare frozen output to Python/Verilator simulation.

Why it matters: 100% correctness proof for small circuits. Builds confidence.

---

## Node 9: The Cell Type Problem
**Yosys has many cell types. We need to handle them.**

Basic gates: `$_AND_`, `$_OR_`, `$_XOR_`, `$_NOT_`
Also: `$_NAND_`, `$_NOR_`, `$_XNOR_`, `$_MUX_`, `$_DFF_*`

For V1: Handle basic gates. Error on unknown cells.

Why it matters: Explicit scope prevents silent failures.

---

## Node 10: Pure Python vs Subprocess
**Can we avoid shelling out to Yosys?**

Options:
1. Subprocess to yosys binary (simple, requires install)
2. Python bindings to Yosys (exist but fragile)
3. Pure Python Verilog parser (pyverilog - limited)

For V1: Subprocess. It works. Optimize later.

Why it matters: Don't over-engineer the first version.

---

## Node 11: The Composition Algorithm
**Topological sort, then evaluate.**

```
1. Build graph from netlist
2. Identify primary inputs (module ports)
3. Topological sort cells
4. For each cell in order:
   - Look up polynomial for cell type
   - Substitute input polynomials
   - Store result polynomial
5. Output polynomials are the frozen model
```

Why it matters: This is the core algorithm.

---

## Node 12: Polynomial Representation
**How do we store composed polynomials?**

Options:
1. Symbolic (sympy) - flexible but slow
2. Coefficient arrays - fast but complex indexing
3. Evaluation-ready code - generate C directly

For V1: Generate C code directly. Skip intermediate representation.

Why it matters: KISS. Generate what we need.

---

## Node 13: Test Case: 4-bit Adder
**Concrete target for Stage 1.**

```verilog
module adder4(input [3:0] a, input [3:0] b, output [4:0] sum);
    assign sum = a + b;
endmodule
```

Success = frozen C produces identical output for all 65,536 input pairs.

Why it matters: Tangible goal. Either it works or it doesn't.

---

## Node 14: Error Handling Philosophy
**Fail loud, fail early.**

Unknown cell type? Error with message.
Combinational loop? Error with message.
Unsupported construct? Error with message.

Don't silently produce wrong output.

Why it matters: Trust requires predictability.

---

## Summary: The Pipeline

```
Verilog (.v)
    │
    ▼
┌─────────────┐
│    Yosys    │  synth -flatten; abc -g AND,OR,XOR,NOT
└─────────────┘
    │
    ▼
JSON Netlist
    │
    ▼
┌─────────────┐
│   Parser    │  Extract cells, connections, ports
└─────────────┘
    │
    ▼
Cell Graph
    │
    ▼
┌─────────────┐
│  Composer   │  Topological sort, polynomial composition
└─────────────┘
    │
    ▼
┌─────────────┐
│   Emitter   │  Generate C code
└─────────────┘
    │
    ▼
Frozen C (.h, .c)
    │
    ▼
┌─────────────┐
│  Verifier   │  Exhaustive/random testing vs simulation
└─────────────┘
    │
    ▼
✓ Verified Frozen Model
```
