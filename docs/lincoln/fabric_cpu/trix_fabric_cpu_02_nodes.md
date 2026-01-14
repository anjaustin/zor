# TriX Fabric CPU - NODES

## Core Concepts Extracted

### N1: Shape-Routed Computation
The fundamental insight: routing IS computation. There's no distinction between "deciding where to send data" and "executing an operation". Pattern matching to a shape IS the operation.

Traditional: Decode → Route → Execute (separate steps)
TriX: Pattern → Shape (one step)

### N2: Atomic Shape Set
The minimal building blocks for all computation:

**Logic (7):** XOR, AND, OR, NOT, NAND, NOR, XNOR
**Arithmetic (7):** ADD, SUB, MUL, DIV, MOD, NEG, ABS
**Shift (7):** SHL, SHR, SAR, ROL, ROR, RCL, RCR
**Compare (6):** EQ, NE, LT, LE, GT, GE
**Memory (2):** LOAD, STORE
**Routing (3):** MUX, DEMUX, SELECT

**Total: ~32 atomic shapes**

Everything else is composition. A computer is a composition of 32 shapes.

### N3: No Instruction Fetch
Traditional CPUs spend 30-40% of their power budget on instruction fetch/decode/dispatch. Pattern-matched routing eliminates this entirely. The pattern IS the instruction.

### N4: Implicit Parallelism
Dataflow execution extracts parallelism automatically:
- Independent operations execute simultaneously
- No ILP analysis needed
- No superscalar complexity
- Parallelism scales with fabric capacity

### N5: No Physical Locality Constraint
Silicon requires multiple copies of logic because signals can only travel so far. Shapes have no physical location - they activate where needed. One ADD shape serves infinite instances.

### N6: Shape Graphs Replace Programs
A program is a directed graph of shape compositions:
- Nodes: Shape activations
- Edges: Data dependencies
- Execution: Data flows along edges
- No program counter

### N7: Memory as Shapes
Memory operations are shapes too:
- LOAD(address) → value
- STORE(address, value) → ()
- Ordering enforced by data dependency edges
- No complex memory ordering rules

### N8: Events as Patterns
Interrupts, exceptions, I/O events are just patterns:
- Event pattern enters fabric
- Routes to handler shape graph
- Handler executes as normal shapes
- No special interrupt controller hardware

### N9: Speculation by Computation
Branch speculation becomes:
- Compute both paths (eager) or neither (lazy)
- Select result when condition known
- No misprediction penalty
- Trade computation for correctness

### N10: Hierarchical Composition
Shapes compose hierarchically:
- Atomic shapes (XOR, AND, ADD)
- Compound shapes (FullAdder, Mux4)
- Functional units (ALU, Shifter)
- Subsystems (Pipeline stage, Memory controller)
- Systems (CPU, GPU, Accelerator)

Same composition rules at every level.

### N11: Target Independence
Shape definitions are mathematical. Execution targets are:
- Software simulation (Python, C)
- GPU (shapes as shaders)
- FPGA (shapes as configurable logic)
- ASIC (shapes as fixed circuits)

Same shapes, different substrates.

### N12: Updateable Architecture
Unlike frozen silicon:
- Shapes can be updated
- New shapes can be added
- Compositions can be optimized
- No re-fabrication needed

### N13: Hollywood Squares as Fabric
The existing Hollywood Squares architecture IS this fabric:
- Positions in grid = shape activation sites
- Routing between positions = data flow
- Position computation = shape execution

Just needs generalization beyond neural network ops.

### N14: Octave Memory as Addressing
Octave memory provides:
- Hierarchical storage
- Geometric addressing
- Natural clustering of related data
- Access patterns emerge from computation

### N15: Bootstrap Minimal
System startup:
1. Reset → fixed boot shape
2. Boot shape → loads initial graph
3. Initial graph → activates
4. System running

Analogous to BIOS but simpler.

## Relationships

```
N1 (Shape-Routed) ─────┬───── N3 (No Fetch)
                       │
N2 (Atomic Shapes) ────┼───── N10 (Hierarchical)
                       │
N4 (Implicit Parallel) ┼───── N6 (Shape Graphs)
                       │
N5 (No Locality) ──────┼───── N11 (Target Independent)
                       │
N7 (Memory Shapes) ────┼───── N14 (Octave Addressing)
                       │
N8 (Events as Patterns)┼───── N9 (Speculation)
                       │
N12 (Updateable) ──────┴───── N13 (Hollywood Fabric)
```

## Key Metrics

| Metric | Traditional CPU | TriX Fabric |
|--------|-----------------|-------------|
| Decode overhead | 30-40% power | 0% |
| Parallelism extraction | ILP hardware | Automatic |
| Shape replication | Physical copies | Single definition |
| ISA modification | Impossible | Trivial |
| Memory ordering | Complex rules | Dependency edges |
| Branch cost | Mispredict penalty | Zero or compute both |

## Critical Dependencies

1. **N1 requires N2**: Shape-routing needs atomic shapes to route to
2. **N4 requires N6**: Implicit parallelism needs graph representation
3. **N7 requires N14**: Memory shapes need Octave addressing
4. **N13 implements N1**: Hollywood Squares is the routing fabric
5. **N11 enables N12**: Target independence enables updates

## Open Questions (from RAW)

1. **Minimal shape set** - Is 32 shapes optimal? Too many? Too few?
2. **Graph representation** - How to store/transmit shape graphs efficiently?
3. **Memory model** - Precise semantics for ordering and consistency?
4. **Real-time** - Deterministic timing in dataflow?
5. **Debugging** - How to debug a graph execution?
6. **Profiling** - What metrics matter in shape-routed compute?
7. **Power model** - How does shape activation map to energy?
