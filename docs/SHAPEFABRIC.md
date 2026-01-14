# Shape Fabric

Shape-routed dataflow compute architecture where **routing IS computation**.

## Overview

The Shape Fabric reimagines computation by replacing traditional instruction-based execution with a dataflow model built on 32 atomic shapes. Instead of a program counter stepping through instructions, data flows through a graph of shape activations, with parallelism extracted automatically from the graph structure.

**Key insight**: The pattern of data determines which shapes activate. No instruction decode. No fetch cycle. The routing itself is the computation.

## Quick Start

```python
from trix.shapefabric import ShapeGraph, execute

# Build a computation graph
graph = ShapeGraph("add_example")
a = graph.input("a")
b = graph.input("b")
result = graph.shape("ADD", [a, b])
graph.output("sum", result)

# Execute with inputs
output = execute(graph, {"a": 5, "b": 3})
print(output["sum"])  # 8
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Shape Fabric                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                │
│   │  Input  │───▶│   XOR   │───▶│  Output │                │
│   └─────────┘    └─────────┘    └─────────┘                │
│        │              ▲                                     │
│        │              │                                     │
│        └──────────────┘                                     │
│                                                             │
│   Tile Pool: [XOR₁, XOR₂, XOR₃, XOR₄, ADD₁, ADD₂, ...]    │
│                                                             │
│   Backend: vulkan (GILLIES) | native (NEON/AVX2) | numpy   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Component | Purpose |
|-----------|---------|
| **ShapeGraph** | DAG representing computation as shape activations |
| **DataflowExecutor** | Executes graphs by propagating values through nodes |
| **ShapeFabric** | Tile pool manager with dynamic allocation |
| **PatternRouter** | Routes data to shapes based on patterns |
| **VectorShapes** | SIMD-accelerated batch operations |

## The 32 Atomic Shapes

All computation reduces to compositions of these primitives:

### Logic (7)
| Shape | Operation | Formula |
|-------|-----------|---------|
| XOR | Exclusive OR | `a ^ b` |
| AND | Logical AND | `a & b` |
| OR | Logical OR | `a \| b` |
| NOT | Logical NOT | `~a` |
| NAND | NOT AND | `~(a & b)` |
| NOR | NOT OR | `~(a \| b)` |
| XNOR | Equivalence | `~(a ^ b)` |

### Arithmetic (7)
| Shape | Operation | Formula |
|-------|-----------|---------|
| ADD | Addition | `a + b` |
| SUB | Subtraction | `a - b` |
| MUL | Multiplication | `a * b` |
| DIV | Division | `a // b` |
| MOD | Modulo | `a % b` |
| NEG | Negation | `-a` |
| ABS | Absolute value | `\|a\|` |

### Shift (7)
| Shape | Operation | Description |
|-------|-----------|-------------|
| SHL | Shift left | Logical left shift |
| SHR | Shift right | Logical right shift |
| SAR | Arithmetic shift right | Sign-extending |
| ROL | Rotate left | Circular left |
| ROR | Rotate right | Circular right |
| RCL | Rotate left through carry | With carry flag |
| RCR | Rotate right through carry | With carry flag |

### Compare (6)
| Shape | Operation | Returns |
|-------|-----------|---------|
| EQ | Equal | 1 if a == b |
| NE | Not equal | 1 if a != b |
| LT | Less than | 1 if a < b |
| LE | Less or equal | 1 if a <= b |
| GT | Greater than | 1 if a > b |
| GE | Greater or equal | 1 if a >= b |

### Memory (2)
| Shape | Operation | Description |
|-------|-----------|-------------|
| LOAD | Memory read | Load from address |
| STORE | Memory write | Store to address |

### Routing (3)
| Shape | Operation | Description |
|-------|-----------|-------------|
| MUX | 2-to-1 multiplexer | Select a or b based on sel |
| DEMUX | 1-to-2 demultiplexer | Route to one of two outputs |
| SELECT | N-to-1 selector | Select from N inputs by index |

## Building Graphs

### Full Adder Example

```python
from trix.shapefabric import ShapeGraph, execute

def build_full_adder():
    graph = ShapeGraph("full_adder")

    # Inputs
    a = graph.input("a")
    b = graph.input("b")
    cin = graph.input("cin")

    # sum = a XOR b XOR cin
    xor1 = graph.shape("XOR", [a, b])
    sum_out = graph.shape("XOR", [xor1, cin])

    # cout = (a AND b) OR ((a XOR b) AND cin)
    and1 = graph.shape("AND", [a, b])
    and2 = graph.shape("AND", [xor1, cin])
    cout = graph.shape("OR", [and1, and2])

    graph.output("sum", sum_out)
    graph.output("cout", cout)

    return graph

# Build and execute
fa = build_full_adder()
result = execute(fa, {"a": 1, "b": 1, "cin": 0})
print(f"sum={result['sum']}, cout={result['cout']}")  # sum=0, cout=1
```

### ALU Example

```python
def build_alu():
    graph = ShapeGraph("alu")

    a = graph.input("a")
    b = graph.input("b")
    op = graph.input("op")

    # Compute all operations in parallel
    add_result = graph.shape("ADD", [a, b])
    sub_result = graph.shape("SUB", [a, b])
    and_result = graph.shape("AND", [a, b])
    or_result = graph.shape("OR", [a, b])
    xor_result = graph.shape("XOR", [a, b])

    # Select based on opcode
    result = graph.shape("SELECT", [op, add_result, sub_result,
                                     and_result, or_result, xor_result])

    graph.output("result", result)
    return graph

alu = build_alu()
result = execute(alu, {"a": 10, "b": 3, "op": 0})  # ADD
print(result["result"])  # 13
```

## Automatic Parallelism

The graph structure explicitly captures parallelism. Nodes with no dependencies can execute simultaneously.

```python
graph = ShapeGraph("parallel_example")
a = graph.input("a")
b = graph.input("b")
c = graph.input("c")
d = graph.input("d")

# Level 1: These can run in parallel
ab = graph.shape("ADD", [a, b])
cd = graph.shape("ADD", [c, d])

# Level 2: Depends on level 1
result = graph.shape("MUL", [ab, cd])

graph.output("result", result)

# Check parallelism
levels = graph.parallel_levels()
print(f"Depth: {len(levels)}")
print(f"Max parallelism: {max(len(l) for l in levels)}")
# Depth: 4 (inputs, two ADDs, MUL, output)
# Max parallelism: 4 (the input nodes)
```

## Execution Modes

### Sequential Execution
```python
from trix.shapefabric import DataflowExecutor

executor = DataflowExecutor(parallel=False)
result = executor.execute(graph, inputs)
```

### Parallel Execution
```python
executor = DataflowExecutor(parallel=True, max_workers=4)
result = executor.execute(graph, inputs)
```

### Fabric-Routed Execution
```python
from trix.shapefabric import create_fabric

fabric, router = create_fabric(tiles_per_shape=8)

# Route through fabric
result = router.route("full_adder", a, b, cin)
```

## Backend Selection

The fabric auto-detects and uses the fastest available backend:

```python
from trix.shapefabric.shapes import VectorShapes

vs = VectorShapes()
print(f"Backend: {vs.backend}")
# Possible outputs:
# - "vulkan (GILLIES)"  - GPU compute, 17B ops/sec
# - "native (NEON)"     - ARM SIMD, 500M ops/sec
# - "native (AVX2)"     - x86 SIMD, 500M ops/sec
# - "numpy"             - Fallback, 50M ops/sec
```

### Force Specific Backend
```python
# CPU only (skip GPU)
vs = VectorShapes(use_gpu=False)

# NumPy only (skip SIMD)
vs = VectorShapes(use_gpu=False, use_native=False)
```

## Vectorized Operations

For batch processing, use VectorShapes:

```python
import numpy as np
from trix.shapefabric.shapes import VectorShapes

vs = VectorShapes()

# Process 1M elements at once
a = np.random.rand(1_000_000).astype(np.float32)
b = np.random.rand(1_000_000).astype(np.float32)

result = vs.xor(a, b)  # GPU/SIMD accelerated
```

## Graph Serialization

Graphs can be serialized for storage or transmission:

```python
# To JSON
json_str = graph.to_json()

# From JSON
restored = ShapeGraph.from_json(json_str)

# To dict
data = graph.to_dict()

# From dict
restored = ShapeGraph.from_dict(data)
```

## Statistics

```python
stats = graph.stats()
print(stats)
# {
#   'name': 'full_adder',
#   'nodes': 9,
#   'edges': 8,
#   'inputs': 3,
#   'outputs': 2,
#   'shapes': 5,
#   'shape_distribution': {'XOR': 2, 'AND': 2, 'OR': 1},
#   'depth': 4,
#   'max_parallelism': 3
# }
```

## Composed Shapes

Common patterns are pre-composed:

```python
from trix.shapefabric.shapes import COMPOSED_SHAPES

# Full adder
sum_bit, carry = COMPOSED_SHAPES["FULL_ADDER"](a, b, cin)

# N-bit adders
result = COMPOSED_SHAPES["ADDER8"](a, b)   # 8-bit
result = COMPOSED_SHAPES["ADDER16"](a, b)  # 16-bit
result = COMPOSED_SHAPES["ADDER32"](a, b)  # 32-bit

# ALU
result = COMPOSED_SHAPES["ALU"](a, b, opcode)
```

## What Dies, What Lives

### Dies
- **Instruction fetch/decode** - patterns ARE the routing
- **Program counter** - execution follows data flow
- **Register file** - values live on edges
- **CPU/GPU distinction** - same shapes, different substrates

### Lives
- **Memory hierarchy** - physics wins
- **Composition** - only abstraction needed
- **Parallelism** - extracted automatically from graph structure

## Performance

| Operation | Throughput |
|-----------|------------|
| XOR (GILLIES Vulkan) | 17B ops/sec |
| XOR (NEON SIMD) | 500M ops/sec |
| XOR (NumPy) | 50M ops/sec |
| Full Adder (fabric) | 200K ops/sec |
| ALU (fabric) | 700K ops/sec |

## API Reference

### ShapeGraph
```python
class ShapeGraph:
    def __init__(name: str = "unnamed")
    def input(name: str) -> str                    # Create input node
    def const(value: Any) -> str                   # Create constant node
    def shape(name: str, inputs: List[str]) -> str # Create shape node
    def output(name: str, source: str) -> str      # Create output node
    def topological_order() -> List[str]           # Get execution order
    def parallel_levels() -> List[List[str]]       # Get parallel groups
    def stats() -> Dict                            # Get statistics
    def to_json() -> str                           # Serialize
    def from_json(json: str) -> ShapeGraph         # Deserialize
```

### DataflowExecutor
```python
class DataflowExecutor:
    def __init__(parallel: bool = False, max_workers: int = 4)
    def execute(graph: ShapeGraph, inputs: Dict) -> Dict
    def get_stats() -> ExecutionStats
```

### ShapeFabric
```python
class ShapeFabric:
    def __init__(tiles_per_shape: int = 4)
    def allocate_tile(shape_name: str) -> Optional[str]
    def free_tile(tile_id: str)
    def execute_tile(tile_id: str, inputs: List) -> Any
    def stats() -> Dict
```

### VectorShapes
```python
class VectorShapes:
    def __init__(use_gpu: bool = True, use_native: bool = True)
    def xor(a: ndarray, b: ndarray) -> ndarray
    def and_(a: ndarray, b: ndarray) -> ndarray
    def or_(a: ndarray, b: ndarray) -> ndarray
    def add(a: ndarray, b: ndarray) -> ndarray
    def sub(a: ndarray, b: ndarray) -> ndarray
    def mul(a: ndarray, b: ndarray) -> ndarray
    def backend -> str
```

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture overview
- [FROZEN_SHAPES.md](FROZEN_SHAPES.md) - Shape mathematical foundations
- [GILLIES_VULKAN.md](GILLIES_VULKAN.md) - GPU acceleration details
- [docs/lincoln/fabric_cpu/](lincoln/fabric_cpu/) - Design documents
