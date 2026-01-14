# TriX Fabric CPU - RAW

## The Core Insight

We've been approaching this backwards. We've been asking "how do we run existing CPU designs on TriX" when we should be asking "what does a CPU look like when built natively on TriX principles?"

The compositor work revealed that CPUs are mostly the same patterns repeated:
- Next186 ALU: 55 patterns, 596 instances
- Next186 CPU: 114 patterns, 1,698 instances
- 70-78% compression ratios

This means 70-78% of the silicon is REDUNDANT STRUCTURE. The same operations implemented over and over in different places because wires can only go so far and signals need to be local.

## Traditional CPU Constraints We Don't Have

1. **Physical locality** - In silicon, a signal can only travel so far per clock cycle. So you need multiple copies of the same logic in different places. We don't have this constraint.

2. **Fixed wiring** - Once fabricated, the connections between components are permanent. We can route dynamically.

3. **Instruction format overhead** - CPUs spend enormous resources fetching, decoding, and dispatching instructions. We can pattern-match directly.

4. **Sequential execution model** - The Von Neumann bottleneck. Instructions execute one at a time (pipelining helps but doesn't eliminate). We can execute dataflow.

5. **Register file bottleneck** - Limited read/write ports to a centralized register file. We can have distributed state.

6. **Power proportional to transistor count** - Unused execution units still leak power. Shapes that aren't active don't consume.

## What We Actually Need

The atomic operations:
- Logic: XOR, AND, OR, NOT, NAND, NOR, XNOR
- Arithmetic: ADD, SUB, MUL, DIV, MOD, NEG, ABS
- Shift: SHL, SHR, SAR, ROL, ROR, RCL, RCR
- Compare: EQ, NE, LT, LE, GT, GE
- Memory: LOAD, STORE
- Control: MUX, DEMUX, SELECT

That's maybe 25-30 atomic shapes. Everything in computing is composition of these.

A FullAdder is: XOR(XOR(a,b), cin), OR(AND(a,b), AND(XOR(a,b), cin))
An ALU is: MUX(opcode, ADD, SUB, AND, OR, XOR, ...)
A CPU is: Composition of ALUs, registers, memory interfaces, control logic

## The Routing Fabric

Instead of fixed wiring between components, we have a routing fabric that:
1. Accepts input patterns (data + operation intent)
2. Matches patterns to appropriate shapes
3. Activates those shapes with the data
4. Produces output patterns
5. Routes outputs to next operations or memory

This is dataflow computing with pattern-matched dispatch.

## Key Differences From Traditional Dataflow

Traditional dataflow architectures still had fixed execution units. They just scheduled dynamically to them.

TriX fabric: The shapes themselves are the computation. There's no distinction between "routing" and "executing". The act of matching a pattern to a shape IS the computation.

## The Hollywood Squares Connection

Hollywood Squares is exactly this architecture at the neural network level:
- Positions in the grid
- Patterns route to positions
- Computation happens at positions
- Results flow to next positions

We just need to extend this to general computation, not just neural network operations.

## The Octave Memory Connection

Octave memory provides hierarchical storage with geometric addressing. This solves the "where do results go" problem:
- Results have natural addresses based on their computation path
- Related results cluster together
- Access patterns emerge from computation patterns

## What About Programs?

Traditional programs are sequences of instructions. What's the TriX equivalent?

A program becomes a GRAPH of shape compositions:
- Nodes are shape activations
- Edges are data dependencies
- Execution flows along edges as data becomes available
- No program counter, no instruction fetch

This is similar to:
- Dataflow graphs
- Petri nets
- Kahn process networks

But with the key difference that shapes are mathematical, not physical units.

## Compatibility With Existing Software

Can we run existing x86 binaries? Yes, but through translation:
1. Binary → Instruction stream
2. Instructions → Shape graph
3. Shape graph → Fabric execution

This is like binary translation in emulators, but the target is shapes, not another ISA.

Or we compile directly:
1. Source code → IR
2. IR → Shape graph
3. Shape graph → Fabric execution

This is like compiling to LLVM IR, but the target is shapes.

## Performance Model

Traditional: Instructions per second, limited by pipeline depth and width
TriX: Operations per second, limited by fabric capacity and memory bandwidth

Because shapes activate in parallel:
- 1000 independent ADDs execute in the same "cycle" as 1 ADD
- Parallelism is extracted automatically from data dependencies
- No instruction-level parallelism analysis needed

## The Speculation Question

Traditional CPUs speculate on branches because they need to keep the pipeline fed. With dataflow:
- Both branch paths can execute (lazy or eager)
- Results are selected when condition is known
- No rollback needed, just discard unused results

This trades computation for correctness - we may do "wasted" work, but we never mispredict.

Or with lazy evaluation:
- Neither path executes until condition is known
- Then only the taken path executes
- Zero wasted work, but potential latency

The fabric can do either based on heuristics or hints.

## Memory Ordering

Traditional CPUs have complex memory ordering rules (x86 is mostly TSO). How does TriX handle this?

Memory operations become shapes too:
- LOAD shape: (address) → (value)
- STORE shape: (address, value) → ()

Ordering is enforced by data dependencies:
- If STORE A happens before LOAD A, there's a dependency edge
- The fabric respects this automatically

For unrelated addresses, operations can be reordered freely.

## Interrupts and Exceptions

Traditional: Interrupt controller, privilege levels, context switching
TriX: Events are just patterns that route to handler shapes

An interrupt becomes:
1. Event pattern enters fabric
2. Routes to interrupt handler shape graph
3. Handler executes
4. Returns control pattern

No special hardware, just another shape composition.

## I/O

I/O devices become shape interfaces:
- Device presents a set of shapes (read port, write port, status)
- Programs compose with these shapes
- Fabric routes appropriately

This is similar to memory-mapped I/O but at the shape level.

## The Bootstrap Question

How does the fabric start? What loads the first shapes?

Minimal bootstrap:
1. Reset activates a fixed "boot" shape
2. Boot shape loads initial shape graph from fixed location
3. Initial graph activates, system is running

This is analogous to BIOS/bootloader but much simpler.

## Implementation Targets

1. **Software simulation** - Pure Python/C implementation for development
2. **GPU execution** - Shapes as shaders, fabric as dispatch
3. **FPGA implementation** - Shapes as configurable logic, fabric as routing
4. **ASIC** - Full custom for maximum performance

All from the same shape definitions.

## What This Enables

1. **No ISA lock-in** - Shapes can be updated, extended, optimized
2. **Automatic parallelism** - Dataflow extracts all available parallelism
3. **Natural heterogeneity** - Different shapes can have different implementations
4. **Incremental optimization** - Improve one shape, all compositions benefit
5. **Hardware/software blur** - Same shapes can run on any target

## Questions To Resolve

1. What's the minimal shape set for general computation?
2. How do we represent shape graphs efficiently?
3. What's the memory model precisely?
4. How do we handle real-time constraints?
5. What's the debugging story?
6. How do we profile and optimize?
7. What's the power model?
