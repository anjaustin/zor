# TriX Fabric CPU - SYNTHESIZE

## The Architecture: Shape-Routed Compute Fabric

### Core Thesis

Build computation from 32 atomic shapes connected by a pattern-matched routing fabric. Let data flow determine execution, not instruction sequence. Composition replaces complexity.

### Architecture Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │              SHAPE FABRIC                    │
                    │                                              │
   Input ──────────►│  ┌─────────────────────────────────────┐    │
   Patterns         │  │         ROUTING MESH                 │    │
                    │  │   (Pattern Matching + Data Flow)     │    │
                    │  └──────────┬──────────────────────────┘    │
                    │             │                                │
                    │  ┌──────────▼──────────────────────────┐    │
                    │  │         SHAPE LAYER                  │    │────► Output
                    │  │                                      │    │      Patterns
                    │  │  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐     │    │
                    │  │  │XOR│ │AND│ │ADD│ │MUL│ │MUX│ ... │    │
                    │  │  └───┘ └───┘ └───┘ └───┘ └───┘     │    │
                    │  │                                      │    │
                    │  │  ┌────────┐ ┌────────┐ ┌────────┐   │    │
                    │  │  │FullAdd │ │Compare │ │Shifter │   │    │
                    │  │  └────────┘ └────────┘ └────────┘   │    │
                    │  │                                      │    │
                    │  └──────────────────────────────────────┘    │
                    │                                              │
                    │  ┌──────────────────────────────────────┐    │
                    │  │         OCTAVE MEMORY                 │    │
                    │  │   (Hierarchical, Geometric Addressing)│    │
                    │  └──────────────────────────────────────┘    │
                    └─────────────────────────────────────────────┘
```

### The 32 Atomic Shapes

```python
ATOMIC_SHAPES = {
    # Logic (7)
    "XOR":  lambda a, b: a ^ b,
    "AND":  lambda a, b: a & b,
    "OR":   lambda a, b: a | b,
    "NOT":  lambda a: ~a,
    "NAND": lambda a, b: ~(a & b),
    "NOR":  lambda a, b: ~(a | b),
    "XNOR": lambda a, b: ~(a ^ b),

    # Arithmetic (7)
    "ADD":  lambda a, b: a + b,
    "SUB":  lambda a, b: a - b,
    "MUL":  lambda a, b: a * b,
    "DIV":  lambda a, b: a // b,
    "MOD":  lambda a, b: a % b,
    "NEG":  lambda a: -a,
    "ABS":  lambda a: abs(a),

    # Shift (7)
    "SHL":  lambda a, n: a << n,
    "SHR":  lambda a, n: a >> n,  # logical
    "SAR":  lambda a, n: a >> n,  # arithmetic (sign extend)
    "ROL":  lambda a, n: rotate_left(a, n),
    "ROR":  lambda a, n: rotate_right(a, n),
    "RCL":  lambda a, n, c: rotate_left_carry(a, n, c),
    "RCR":  lambda a, n, c: rotate_right_carry(a, n, c),

    # Compare (6)
    "EQ":   lambda a, b: a == b,
    "NE":   lambda a, b: a != b,
    "LT":   lambda a, b: a < b,
    "LE":   lambda a, b: a <= b,
    "GT":   lambda a, b: a > b,
    "GE":   lambda a, b: a >= b,

    # Memory (2)
    "LOAD":  lambda addr: memory[addr],
    "STORE": lambda addr, val: memory[addr] = val,

    # Routing (3)
    "MUX":    lambda sel, a, b: b if sel else a,
    "DEMUX":  lambda sel, val: (val, 0) if sel else (0, val),
    "SELECT": lambda idx, *args: args[idx],
}
```

### Composed Shapes (Built from Atomics)

```python
COMPOSED_SHAPES = {
    # Arithmetic
    "FullAdder": compose(
        XOR(XOR(a, b), cin),  # sum
        OR(AND(a, b), AND(XOR(a, b), cin))  # cout
    ),

    "Adder8": chain(FullAdder, 8),
    "Adder16": chain(FullAdder, 16),
    "Adder32": chain(FullAdder, 32),

    # ALU
    "ALU": mux_select(opcode, [
        ADD, SUB, AND, OR, XOR, NOT, SHL, SHR, ...
    ]),

    # Control
    "ConditionalMove": compose(
        MUX(condition, keep_value, new_value)
    ),

    # Memory
    "LoadWord": compose(
        LOAD(addr), LOAD(addr+1), LOAD(addr+2), LOAD(addr+3),
        combine_bytes
    ),
}
```

### Shape Graph Format

```python
@dataclass
class ShapeNode:
    shape: str           # Shape name
    inputs: List[Edge]   # Input edges
    outputs: List[Edge]  # Output edges

@dataclass
class Edge:
    source: ShapeNode
    source_port: int
    target: ShapeNode
    target_port: int

@dataclass
class ShapeGraph:
    nodes: List[ShapeNode]
    inputs: List[Edge]   # External inputs
    outputs: List[Edge]  # External outputs

    def execute(self, input_values):
        """Dataflow execution of graph."""
        ready = set()
        values = {}

        # Initialize with inputs
        for edge, value in zip(self.inputs, input_values):
            values[edge] = value
            mark_ready(edge.target)

        # Execute until all outputs produced
        while not all_outputs_ready():
            for node in ready:
                if all_inputs_available(node):
                    result = execute_shape(node.shape, get_inputs(node))
                    for edge in node.outputs:
                        values[edge] = result
                        mark_ready(edge.target)
                    ready.remove(node)

        return [values[e] for e in self.outputs]
```

### Implementation Plan

#### Phase 1: Atomic Shapes (Week 1-2)

```
1. Define all 32 atomic shapes with:
   - Mathematical specification
   - Python reference implementation
   - Shape signature (inputs, outputs, types)
   - Frozen shape representation

2. Build shape registry:
   - Name → shape lookup
   - Type checking
   - Composition validation

3. Test exhaustively:
   - All atomics against reference
   - Edge cases (overflow, underflow, division by zero)
```

**Deliverable:** `src/trix/fabric/shapes.py` with all atomics

#### Phase 2: Shape Graphs (Week 2-3)

```
1. Define graph format:
   - Node representation
   - Edge representation
   - Serialization (JSON, binary)

2. Build graph operations:
   - Parse/serialize
   - Compose (combine graphs)
   - Validate (type check edges)
   - Optimize (dead code elimination, constant folding)

3. Build interpreter:
   - Dataflow execution engine
   - Value propagation
   - Parallel execution on CPU threads
```

**Deliverable:** `src/trix/fabric/graph.py` with graph engine

#### Phase 3: Routing Fabric (Week 3-4)

```
1. Integrate with Hollywood Squares:
   - Shapes as positions
   - Pattern matching as routing
   - Activation as execution

2. Build pattern matcher:
   - Input pattern → shape selection
   - Multi-pattern dispatch
   - Priority handling

3. Connect to Octave Memory:
   - Memory shapes interface
   - Address translation
   - Caching/hierarchy
```

**Deliverable:** `src/trix/fabric/router.py` with routing fabric

#### Phase 4: Composed Shapes (Week 4-5)

```
1. Build standard library:
   - Arithmetic: Adders, subtractors, multipliers
   - Logic: Comparators, decoders, encoders
   - Control: Muxes, demuxes, selectors
   - Memory: Load/store wrappers

2. Implement compositor integration:
   - Discover patterns in graphs
   - Map to library shapes
   - Optimize execution

3. Add GPU backend:
   - Shapes as CUDA kernels
   - Fabric as kernel dispatch
   - Memory as unified/managed
```

**Deliverable:** `src/trix/fabric/stdlib.py` with composed shapes

#### Phase 5: Benchmark (Week 5-6)

```
1. Implement benchmark programs:
   - Integer arithmetic (Fibonacci, factorial)
   - Bit manipulation (popcount, bit reversal)
   - Array operations (sum, search, sort)
   - Control flow (binary search, loops)

2. Compare against:
   - Native Python
   - NumPy
   - Direct C implementation

3. Profile and optimize:
   - Identify bottlenecks
   - Optimize critical shapes
   - Tune routing fabric
```

**Deliverable:** Benchmark suite with results

### Success Metrics

| Metric | Target |
|--------|--------|
| Atomic shapes | All 32 implemented and tested |
| Graph execution | Correct for all test cases |
| Parallelism | Linear speedup with cores |
| Composition overhead | < 10% vs hand-written |
| Memory bandwidth | > 50% utilization |

### Risks and Mitigations

1. **Risk:** Graph interpretation overhead too high
   **Mitigation:** JIT compilation of hot graphs

2. **Risk:** Pattern matching too slow
   **Mitigation:** Hash-based dispatch (like compositor)

3. **Risk:** Memory shapes become bottleneck
   **Mitigation:** Prefetching, caching, batching

4. **Risk:** Parallelism limited by dependencies
   **Mitigation:** Speculation, lazy evaluation

### File Structure

```
src/trix/fabric/
├── __init__.py         # Exports
├── shapes.py           # Atomic shape definitions
├── composed.py         # Composed shape library
├── graph.py            # Shape graph representation
├── router.py           # Routing fabric
├── executor.py         # Dataflow executor
├── memory.py           # Memory shape interface
├── compiler.py         # Source → graph compilation
└── backends/
    ├── python.py       # Pure Python backend
    ├── numpy.py        # NumPy backend
    ├── cuda.py         # CUDA backend
    └── native.py       # C/Native backend
```

### First Milestone: Fibonacci

```python
# Shape graph for Fibonacci
fib_graph = ShapeGraph([
    # n < 2 ? n : fib(n-1) + fib(n-2)
    Node("LT", inputs=[n, 2]),           # n < 2
    Node("SUB", inputs=[n, 1]),          # n - 1
    Node("SUB", inputs=[n, 2]),          # n - 2
    Node("RECURSE", inputs=[n_minus_1]), # fib(n-1)
    Node("RECURSE", inputs=[n_minus_2]), # fib(n-2)
    Node("ADD", inputs=[fib_1, fib_2]),  # fib(n-1) + fib(n-2)
    Node("MUX", inputs=[lt_2, n, sum]),  # select result
])

# Execute
result = fabric.execute(fib_graph, n=10)
assert result == 55
```

When this runs correctly AND faster than pure Python, we have proof of concept.

### The Vision

This isn't a CPU. It's not a GPU. It's not an FPGA.

It's a **Shape Computer** - a machine where:
- Programs are graphs
- Operations are shapes
- Execution is dataflow
- Parallelism is automatic
- Hardware is optional

The 286 was constrained by 1982. The Ryzen is constrained by legacy.

The Shape Fabric is constrained only by physics.

**Build it.**
