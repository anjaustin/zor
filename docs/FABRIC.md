# Routing Fabric

Dynamic shape routing for dataflow execution - the "magick" layer where data patterns determine computation.

## Overview

The Routing Fabric is the runtime layer that makes shape-routed computation possible. It manages pools of shape tiles, routes data between them, and executes computations based on patterns rather than instruction sequences.

**Key insight**: No instruction decode needed. The data patterns themselves determine routing. Routing IS computation.

## Quick Start

```python
from trix.shapefabric import create_fabric

# Create fabric with tile pools
fabric, router = create_fabric(tiles_per_shape=4)

# Route a full adder computation
sum_bit, carry = router.route("full_adder", a=1, b=1, cin=0)
# sum_bit=0, carry=1

# Route an ALU operation
result = router.route("alu", a=10, b=3, op=0)  # ADD
# result=13

# Check fabric state
print(fabric.stats())
# {'total_tiles': 128, 'tiles_per_shape': 4, 'idle': 128, ...}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Routing Fabric                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────────────────────────────────────────────┐ │
│   │                   Pattern Router                      │ │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐  │ │
│   │  │full_add │ │   alu   │ │ripple_  │ │  custom   │  │ │
│   │  │   er    │ │         │ │ adder   │ │ patterns  │  │ │
│   │  └─────────┘ └─────────┘ └─────────┘ └───────────┘  │ │
│   └──────────────────────────────────────────────────────┘ │
│                          │                                  │
│   ┌──────────────────────▼───────────────────────────────┐ │
│   │                  Routing Mesh                         │ │
│   │      (Crossbar interconnect between tiles)            │ │
│   └──────────────────────────────────────────────────────┘ │
│                          │                                  │
│   ┌──────────────────────▼───────────────────────────────┐ │
│   │                    Tile Pools                         │ │
│   │                                                       │ │
│   │  XOR Pool    AND Pool    OR Pool     ADD Pool   ...  │ │
│   │  ┌──┬──┬──┐ ┌──┬──┬──┐ ┌──┬──┬──┐ ┌──┬──┬──┐        │ │
│   │  │T1│T2│T3│ │T4│T5│T6│ │T7│T8│T9│ │TA│TB│TC│        │ │
│   │  └──┴──┴──┘ └──┴──┴──┘ └──┴──┴──┘ └──┴──┴──┘        │ │
│   │                                                       │ │
│   └───────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Components

### ShapeTile

Individual execution unit for a single shape type.

```python
@dataclass
class ShapeTile:
    id: str              # Unique tile identifier
    shape_name: str      # Shape this tile executes (XOR, AND, etc.)
    state: TileState     # Current state
    inputs: List[Any]    # Input values
    output: Any          # Result after execution
```

**Tile States**:

| State | Description |
|-------|-------------|
| IDLE | Available for allocation |
| ALLOCATED | Reserved for an operation |
| EXECUTING | Currently computing |
| DONE | Result ready |

```python
# Lifecycle
tile = ShapeTile(id="tile_0", shape_name="XOR")
# state: IDLE

tile.state = TileState.ALLOCATED  # Reserved
tile.inputs = [a, b]
result = tile.execute()            # state: EXECUTING -> DONE
tile.reset()                       # state: IDLE
```

### ShapeFabric

Manages tile pools and allocation.

```python
class ShapeFabric:
    def __init__(self, tiles_per_shape: int = 4):
        """
        Create fabric with tile pools.

        Args:
            tiles_per_shape: Instances per shape type

        Creates 32 shapes × tiles_per_shape = total tiles
        """

    def allocate_tile(self, shape_name: str) -> Optional[str]:
        """Allocate an idle tile, returns tile_id or None."""

    def free_tile(self, tile_id: str):
        """Return tile to pool."""

    def execute_tile(self, tile_id: str, inputs: List[Any]) -> Any:
        """Execute tile with inputs, return result."""

    def stats(self) -> Dict:
        """Get fabric statistics."""
```

**Usage**:

```python
fabric = ShapeFabric(tiles_per_shape=4)

# Allocate a tile
tile_id = fabric.allocate_tile("XOR")  # "tile_42"

# Execute
result = fabric.execute_tile(tile_id, [0b1010, 0b1100])  # 0b0110

# Free for reuse
fabric.free_tile(tile_id)
```

### RoutingMesh

Crossbar interconnect for data routing between tiles.

```python
class RoutingMesh:
    def add_route(self, source: str, source_port: int,
                  target: str, target_port: int):
        """Connect source tile output to target tile input."""

    def propagate(self, source_tile: str, source_port: int, value: Any):
        """Push value through mesh to connected targets."""

    def get_inputs(self, tile_id: str) -> Dict[int, Any]:
        """Get pending inputs for a tile."""

    def clear_routes(self):
        """Reset all routes."""
```

**Usage**:

```python
mesh = RoutingMesh()

# Connect XOR output to AND input
mesh.add_route("tile_xor", 0, "tile_and", 0)

# Propagate value
mesh.propagate("tile_xor", 0, 42)

# Target receives it
inputs = mesh.get_inputs("tile_and")  # {0: 42}
```

### PatternRouter

Routes computations based on named patterns.

```python
class PatternRouter:
    def __init__(self, fabric: ShapeFabric):
        """Create router backed by fabric."""

    def register_pattern(self, name: str, router: Callable):
        """Register a pattern router function."""

    def route(self, pattern: str, *args) -> Any:
        """Execute pattern with arguments."""
```

## Built-in Patterns

### full_adder

Single-bit addition with carry.

```python
sum_bit, carry = router.route("full_adder", a, b, cin)
```

Implementation:
```
Level 1: xor1 = a ^ b
         and1 = a & b
Level 2: sum = xor1 ^ cin
         and2 = xor1 & cin
Level 3: cout = and1 | and2
```

### alu

Arithmetic/logic unit with operation selection.

```python
result = router.route("alu", a, b, op)
```

| Op | Operation |
|----|-----------|
| 0 | ADD |
| 1 | SUB |
| 2 | AND |
| 3 | OR |
| 4 | XOR |
| 5 | SHL |
| 6 | SHR |
| 7 | MUL |

### ripple_adder

N-bit ripple carry adder.

```python
result = router.route("ripple_adder", a, b, n_bits)
```

Chains full adders bit-by-bit with carry propagation.

## Creating Custom Patterns

```python
def route_half_adder(fabric: ShapeFabric, a: int, b: int) -> Tuple[int, int]:
    """Custom half adder pattern."""
    xor_tile = fabric.allocate_tile("XOR")
    and_tile = fabric.allocate_tile("AND")

    try:
        sum_bit = fabric.execute_tile(xor_tile, [a, b])
        carry = fabric.execute_tile(and_tile, [a, b])
        return sum_bit, carry
    finally:
        fabric.free_tile(xor_tile)
        fabric.free_tile(and_tile)

# Register
router.register_pattern("half_adder", route_half_adder)

# Use
result = router.route("half_adder", 1, 1)  # (0, 1)
```

## Parallel Execution

Tiles at the same level can execute in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

def route_parallel_example(fabric, a, b, c, d):
    """Example showing parallel tile execution."""
    # Allocate all tiles upfront
    add1 = fabric.allocate_tile("ADD")
    add2 = fabric.allocate_tile("ADD")
    mul = fabric.allocate_tile("MUL")

    try:
        # Level 1: Execute ADD tiles in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(fabric.execute_tile, add1, [a, b])
            future2 = executor.submit(fabric.execute_tile, add2, [c, d])
            ab = future1.result()
            cd = future2.result()

        # Level 2: Multiply results
        result = fabric.execute_tile(mul, [ab, cd])
        return result
    finally:
        for tile in [add1, add2, mul]:
            fabric.free_tile(tile)
```

## Resource Management

### Tile Exhaustion

When all tiles of a shape are in use:

```python
fabric = ShapeFabric(tiles_per_shape=2)

t1 = fabric.allocate_tile("XOR")  # OK
t2 = fabric.allocate_tile("XOR")  # OK
t3 = fabric.allocate_tile("XOR")  # None - exhausted

# Free a tile
fabric.free_tile(t1)
t3 = fabric.allocate_tile("XOR")  # OK now
```

### Sizing Guidelines

| Use Case | tiles_per_shape | Total Tiles |
|----------|-----------------|-------------|
| Testing | 2 | 64 |
| Normal | 4 | 128 |
| High parallelism | 8 | 256 |
| Production | 16+ | 512+ |

## Statistics

```python
stats = fabric.stats()
```

| Field | Description |
|-------|-------------|
| total_tiles | Total tiles in fabric |
| tiles_per_shape | Tiles per shape type |
| idle | Available tiles |
| allocated | Reserved tiles |
| executing | Currently computing |
| done | Completed, awaiting free |
| route_count | Active mesh routes |

## Performance

Benchmarks on typical hardware:

| Pattern | Throughput |
|---------|------------|
| Full Adder | 200K ops/sec |
| ALU | 700K ops/sec |
| 8-bit Ripple Adder | 25K ops/sec |

The overhead is primarily Python function call and allocation. For high performance, use VectorShapes for batch operations.

## Design Philosophy

### What Dies

- **Instruction fetch/decode**: Patterns ARE the routing
- **Program counter**: Execution follows data flow
- **Opcode dispatch**: Shape selection is implicit in pattern

### What Lives

- **Tile pools**: Resource management still needed
- **Routing mesh**: Data must flow somewhere
- **Pattern abstraction**: Composition remains useful

## API Reference

### create_fabric

```python
def create_fabric(tiles_per_shape: int = 4) -> Tuple[ShapeFabric, PatternRouter]:
    """
    Create fabric with built-in pattern routers.

    Args:
        tiles_per_shape: Tiles per shape type

    Returns:
        (fabric, router) tuple with full_adder, alu, ripple_adder registered
    """
```

### ShapeFabric

```python
class ShapeFabric:
    def __init__(tiles_per_shape: int = 4)
    def allocate_tile(shape_name: str) -> Optional[str]
    def free_tile(tile_id: str) -> None
    def execute_tile(tile_id: str, inputs: List[Any]) -> Any
    def stats() -> Dict[str, Any]
```

### PatternRouter

```python
class PatternRouter:
    def __init__(fabric: ShapeFabric)
    def register_pattern(name: str, router: Callable) -> None
    def route(pattern: str, *args) -> Any
```

### ShapeTile

```python
@dataclass
class ShapeTile:
    id: str
    shape_name: str
    state: TileState
    inputs: List[Any]
    output: Any

    def execute() -> Any
    def reset() -> None
```

### TileState

```python
class TileState(Enum):
    IDLE = "idle"
    ALLOCATED = "allocated"
    EXECUTING = "executing"
    DONE = "done"
```

## See Also

- [SHAPEFABRIC.md](SHAPEFABRIC.md) - Shape graph construction and execution
- [COMPOSITOR.md](COMPOSITOR.md) - Automatic pattern discovery
- [FROZEN_SHAPES.md](FROZEN_SHAPES.md) - Mathematical shape foundations
