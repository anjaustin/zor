# Raw Thoughts: Forge Pipeline Implementation

## Stream of Consciousness

We need to go from Verilog to frozen C automatically. The 6502 was hand-crafted - we knew the shapes, we knew the routing. For arbitrary Verilog, we need to discover that structure.

What does Verilog give us? A description of logic. Gates, wires, registers. The synthesizable subset is what matters - stuff that can become silicon. That's also what can become polynomials.

First question: how do we parse Verilog? Options:
1. Write our own parser (hard, error-prone, why?)
2. Use pyverilog (Python library, parses to AST)
3. Use Yosys (industrial strength, outputs JSON/RTLIL)
4. Use Verilator (compiles to C++, could extract structure)

Yosys feels right. It's what the open-source FPGA community uses. It can:
- Parse Verilog/SystemVerilog
- Synthesize to gate-level netlist
- Output to various formats (JSON, RTLIL, BLIF)
- Already handles the hard cases

So the pipeline could be:
```
Verilog → Yosys → JSON netlist → Our analysis → Training → Frozen C
```

What's in a netlist? Cells and wires. Each cell is a primitive (AND, OR, XOR, DFF, etc.) with inputs and outputs. Wires connect them.

Wait. We already know the polynomials for primitives:
- AND(a,b) = ab
- OR(a,b) = a + b - ab
- XOR(a,b) = a + b - 2ab
- NOT(a) = 1 - a

If Yosys gives us a netlist of primitives, we can:
1. Topologically sort the cells
2. Compose the polynomials
3. The result IS the frozen model

No neural network needed for combinational logic! The polynomial composition is deterministic.

But wait - that's just polynomial expansion. It could explode exponentially for deep circuits. A 32-bit adder composed naively would have astronomical terms.

That's where frozen SHAPES come in. We don't expand to raw polynomials. We recognize patterns:
- Ripple adder → use ripple_add shape
- Multiplexer → use mux shape
- Comparator → use compare shape

The Forge needs to:
1. Parse to netlist (Yosys)
2. Recognize patterns / extract shapes
3. Compose shapes (not raw gates)
4. Emit frozen C

Pattern recognition is the hard part. How do we know a bunch of gates is a ripple adder? Options:
- Structural matching (look for the pattern)
- Functional matching (simulate and match behavior to known shapes)
- Learning (train a classifier on labeled examples)

For V1, start simple: structural matching for common patterns, fallback to gate-level composition for unknowns.

What about sequential logic? Registers, state machines.

A DFF is: output = input (on clock edge)
Between clock edges, it's just combinational logic.

So a synchronous circuit is:
```
inputs + current_state → combinational logic → outputs + next_state
```

We freeze the combinational part. The state update is just assignment.

For FSMs:
1. Extract state register
2. Freeze next_state logic
3. Freeze output logic
4. Runtime: evaluate frozen logic, update state

This matches how we did the 6502! The registers (A, X, Y, PC, etc.) are state. The instruction execution is frozen combinational logic. We just didn't think of it that way.

What about memories? RAMs, ROMs.

ROM is trivial - it's a lookup table. That's literally what we already do.

RAM is state. A big array of state. The address decoding is combinational. The read/write is state update.

We can handle RAM the same way: freeze the address decode and data path, treat the array as runtime state.

Now: what's the minimal implementation?

Stage 1: Combinational logic only
- Input: Verilog module (no registers, no always blocks with posedge)
- Output: Frozen C function
- Method: Yosys → netlist → compose → emit

Stage 2: Add sequential logic
- Input: Synchronous Verilog (always @(posedge clk))
- Output: Frozen C with state struct
- Method: Extract registers, freeze combinational cones

Stage 3: Pattern recognition
- Recognize adders, muxes, comparators
- Use optimized shapes instead of gate-level

Stage 4: Memories
- Handle RAM/ROM
- Treat as runtime state

Let's start with Stage 1. Combinational only.

Test case: a simple 4-bit adder in Verilog.

```verilog
module adder4(
    input [3:0] a,
    input [3:0] b,
    output [4:0] sum
);
    assign sum = a + b;
endmodule
```

Pipeline:
1. Yosys parses, synthesizes to gates
2. We read the JSON netlist
3. We compose the gate polynomials
4. We emit C code

Actually, Yosys might optimize this to use its internal adder cell. We need to tell it to decompose to basic gates.

Yosys command:
```
read_verilog adder4.v
synth -flatten
abc -g AND,OR,XOR,NOT  # map to basic gates only
write_json adder4.json
```

Then we parse the JSON, build the polynomial, emit C.

For verification:
- Exhaustive testing for small circuits (4-bit adder = 256 * 256 = 65536 cases)
- Random testing for larger circuits
- Compare frozen output to Verilator simulation

Dependencies:
- Yosys (system install or subprocess)
- JSON parsing (stdlib)
- C code generation (we already have this in c_export.py)

What could go wrong:
- Yosys cell types we don't handle
- Combinational loops (illegal but possible to write)
- Tri-state logic (not synthesizable to basic gates)
- X/Z values (unknown/high-impedance)

For V1: fail gracefully on anything we don't handle. Be honest about scope.

## Questions Arising

- Is Yosys the right tool? Alternatives?
- How do we handle multi-bit signals? Bit-blast to individual wires?
- Performance of polynomial composition for large circuits?
- How do we test that our composition is correct?
- Can we do this without shelling out to Yosys? (Pure Python?)

## First Instincts

- Yosys is battle-tested, use it
- Start with 4-bit adder as test case
- Bit-blast everything for V1, optimize later
- Exhaustive testing for small circuits
- Build incrementally: parse → analyze → compose → emit → verify
- Each stage should be independently testable
