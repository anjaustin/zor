# Compositor

Automatic structural discovery from flat gate graphs through pattern recognition.

## Overview

The Compositor transforms flat gate-level netlists into hierarchical shape compositions. Given a soup of gates and wires, it discovers repeated patterns, identifies known structures (like full adders), and builds a compositional tree that reveals the underlying architecture.

**Key insight**: Structure exists in the patterns. The Compositor finds it through neighborhood hashing and subgraph isomorphism, turning opaque gate graphs into understandable compositions.

## Quick Start

```python
from trix.compositor import compose, visualize

# Define a gate-level system
system = {
    "gates": {
        "g1": {"type": "XOR", "inputs": ["a", "b"]},
        "g2": {"type": "AND", "inputs": ["a", "b"]},
        "g3": {"type": "XOR", "inputs": ["g1", "cin"]},
        "g4": {"type": "AND", "inputs": ["g1", "cin"]},
        "g5": {"type": "OR", "inputs": ["g2", "g4"]},
    },
    "inputs": ["a", "b", "cin"],
    "outputs": {"sum": "g3", "cout": "g5"}
}

# Discover structure
tree = compose(system)

# Visualize
print(visualize(tree))
# Output:
# FullAdder
# ├── HalfAdder (a, b)
# │   ├── XOR → sum
# │   └── AND → carry
# └── HalfAdder (partial_sum, cin)
#     ├── XOR → sum
#     └── AND → carry
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Compositor Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐ │
│   │ Hasher  │───▶│ Matcher │───▶│Composer │───▶│  Tree   │ │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘ │
│        │              │              │              │       │
│   Neighborhood   Pattern        Build          ShapeTree   │
│   Hashing        Finding       Hierarchy        Output     │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │                   Pattern Library                    │  │
│   │  HalfAdder | FullAdder | Mux2 | XOR3 | Decoder2x4   │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Components

| Component | Purpose |
|-----------|---------|
| **Hasher** | Computes neighborhood hashes to find candidate patterns |
| **Matcher** | Verifies pattern matches via subgraph isomorphism |
| **Composer** | Builds hierarchical ShapeTree from verified patterns |
| **Library** | Known pattern signatures for common circuits |
| **Visualizer** | ASCII and Graphviz output for inspection |

## How It Works

### Step 1: Neighborhood Hashing

The hasher computes a hash for each gate based on its local neighborhood - its type, input sources, and fanout structure. Gates with identical neighborhoods get identical hashes.

```python
from trix.compositor import hash_neighborhood

# Build wire maps and hash all gates
wire_to_gates, gate_to_wires = build_wire_maps(system)
hashes = hash_all_gates(system, wire_to_gates, depth=2)

# Find gates with matching hashes (potential pattern instances)
candidates = find_candidate_patterns(hashes, min_instances=2)
```

The hash captures:
- Gate type (XOR, AND, OR, etc.)
- Input structure (number and types of sources)
- Fanout structure (where outputs go)
- Recursive neighborhood to specified depth

### Step 2: Pattern Matching

Candidate patterns are verified through subgraph extraction and isomorphism checking:

```python
from trix.compositor import find_patterns

patterns = find_patterns(
    system,
    min_instances=2,  # At least 2 occurrences
    min_size=2,       # At least 2 gates
    max_depth=3       # Expansion depth
)

for pattern in patterns:
    print(f"{pattern.name}: {len(pattern.instances)} instances")
    print(f"  Gates: {pattern.gates}")
    print(f"  Inputs: {pattern.boundary_inputs}")
    print(f"  Outputs: {pattern.boundary_outputs}")
```

The matcher:
1. Extracts subgraphs around candidate gates
2. Finds boundary wires (inputs/outputs to the subgraph)
3. Checks isomorphism between candidate instances
4. Expands patterns by adding connected gates

### Step 3: Library Matching

Discovered patterns are compared against known circuit signatures:

```python
from trix.compositor.library import KNOWN_PATTERNS, match_known_pattern

# Check if a pattern matches a known structure
known = match_known_pattern(pattern)
if known:
    print(f"Matched: {known.name}")
    print(f"Description: {known.description}")
```

Known patterns include:

| Pattern | Gates | Description |
|---------|-------|-------------|
| HalfAdder | XOR + AND | Single-bit addition without carry-in |
| FullAdder | 2×XOR + 2×AND + OR | Single-bit addition with carry |
| Mux2 | 2×AND + OR + NOT | 2-to-1 multiplexer |
| XOR3 | 2×XOR | 3-input XOR (parity) |
| Decoder2x4 | 4×AND + 2×NOT | 2-to-4 decoder |

### Step 4: Composition

Patterns are composed into a hierarchical tree:

```python
from trix.compositor import compose

tree = compose(system, min_instances=2, min_size=2, max_depth=3)

# Tree structure
print(f"Root: {tree.root.name}")
for child in tree.root.children:
    print(f"  Child: {child.name}")
```

The composer:
1. Starts with atomic gates
2. Groups gates into discovered patterns
3. Builds hierarchy based on pattern containment
4. Labels with known pattern names when matched

## Data Structures

### Gate System

Input format for composition:

```python
system = {
    "gates": {
        "gate_id": {
            "type": "XOR",           # Gate type
            "inputs": ["a", "b"],    # Input wire names
        },
        # ... more gates
    },
    "inputs": ["a", "b", "cin"],     # Primary inputs
    "outputs": {"sum": "g3", "cout": "g5"}  # Named outputs
}
```

### Pattern

Discovered pattern structure:

```python
@dataclass
class Pattern:
    name: str                    # Pattern name or hash
    gates: Set[str]              # Gate IDs in pattern
    instances: List[Set[str]]    # All instances found
    boundary_inputs: Set[str]    # Input wires
    boundary_outputs: Set[str]   # Output wires
    known_match: Optional[str]   # Matched library pattern
```

### ShapeTree

Hierarchical composition result:

```python
@dataclass
class ComposedShape:
    name: str                           # Shape name
    shape_type: str                     # "atomic" or "composed"
    gates: Set[str]                     # Constituent gates
    children: List['ComposedShape']     # Child shapes
    inputs: List[str]                   # Input wires
    outputs: List[str]                  # Output wires

@dataclass
class ShapeTree:
    root: ComposedShape         # Root of composition tree
    patterns: List[Pattern]     # All discovered patterns
    stats: Dict                 # Composition statistics
```

## Visualization

### ASCII Tree

```python
from trix.compositor import visualize

print(visualize(tree))
```

Output:
```
System: ripple_adder_4bit
├── FullAdder[0] (a0, b0, cin)
│   ├── HalfAdder (a0, b0)
│   │   ├── XOR:g1 → partial_sum
│   │   └── AND:g2 → partial_carry
│   └── HalfAdder (partial_sum, cin)
│       ├── XOR:g3 → sum0
│       └── AND:g4 → carry_prop
│   └── OR:g5 → cout0
├── FullAdder[1] (a1, b1, cout0)
│   └── ...
└── FullAdder[3] (a3, b3, cout2)
    └── ...
```

### Statistics

```python
from trix.compositor import visualize_stats

stats = visualize_stats(tree)
print(stats)
```

Output:
```
Composition Statistics:
  Total gates: 20
  Patterns found: 3
  Pattern instances:
    HalfAdder: 8
    FullAdder: 4
  Compression ratio: 4.0x
  Max depth: 3
```

### Graphviz DOT

```python
from trix.compositor.visualizer import to_dot

dot = to_dot(tree)
with open("composition.dot", "w") as f:
    f.write(dot)

# Render: dot -Tpng composition.dot -o composition.png
```

## Examples

### Discovering a Ripple Adder

```python
from trix.compositor import compose, visualize

# 4-bit ripple carry adder (20 gates)
system = build_ripple_adder(bits=4)

# Discover structure
tree = compose(system)

print(visualize(tree))
# Discovers:
# - 4 FullAdder instances
# - 8 HalfAdder instances within them
# - Carry chain structure
```

### Discovering an ALU

```python
# ALU with multiple operations
system = build_alu_8bit()

tree = compose(system, min_instances=2, max_depth=4)

# Discovers:
# - Repeated arithmetic units
# - Mux structures for operation selection
# - Shared logic patterns
```

### Custom Pattern Library

```python
from trix.compositor.library import register_pattern, PatternSignature

# Register a custom pattern
register_pattern(PatternSignature(
    name="MyPattern",
    gate_types={"XOR": 1, "AND": 2, "OR": 1},
    description="Custom logic block",
    inputs=3,
    outputs=2
))

# Now compose will recognize MyPattern
tree = compose(system)
```

## API Reference

### compose

```python
def compose(
    system: Dict,
    min_instances: int = 2,
    min_size: int = 2,
    max_depth: int = 3
) -> ShapeTree:
    """
    Compose a gate system into a hierarchical shape tree.

    Args:
        system: Gate-level netlist
        min_instances: Minimum pattern occurrences to consider
        min_size: Minimum gates in a pattern
        max_depth: Maximum expansion depth for pattern finding

    Returns:
        ShapeTree with discovered hierarchy
    """
```

### hash_neighborhood

```python
def hash_neighborhood(
    system: Dict,
    gate_id: str,
    depth: int = 2
) -> int:
    """
    Compute neighborhood hash for a gate.

    Args:
        system: Gate-level netlist
        gate_id: Gate to hash
        depth: Neighborhood depth

    Returns:
        Integer hash of neighborhood structure
    """
```

### find_patterns

```python
def find_patterns(
    system: Dict,
    min_instances: int = 2,
    min_size: int = 2,
    max_depth: int = 3
) -> List[Pattern]:
    """
    Find repeated patterns in a gate system.

    Args:
        system: Gate-level netlist
        min_instances: Minimum occurrences
        min_size: Minimum pattern size
        max_depth: Expansion depth

    Returns:
        List of discovered patterns
    """
```

### visualize

```python
def visualize(
    tree: ShapeTree,
    indent: str = "  ",
    show_gates: bool = True
) -> str:
    """
    Generate ASCII tree visualization.

    Args:
        tree: Composition result
        indent: Indentation string
        show_gates: Include gate IDs

    Returns:
        ASCII string representation
    """
```

## Performance

| System Size | Gates | Patterns Found | Time |
|-------------|-------|----------------|------|
| 4-bit adder | 20 | 4 FullAdder, 8 HalfAdder | 2ms |
| 8-bit adder | 40 | 8 FullAdder, 16 HalfAdder | 5ms |
| 16-bit ALU | 200 | 32 patterns | 50ms |
| 6502 CPU | 3,510 | 150+ patterns | 2s |

Complexity: O(n * d * m) where n=gates, d=depth, m=pattern instances

## Use Cases

1. **Reverse Engineering**: Discover structure in synthesized netlists
2. **Verification**: Compare discovered structure against expected architecture
3. **Optimization**: Identify repeated patterns for sharing
4. **Documentation**: Generate hierarchical views of complex circuits
5. **Learning**: Understand how complex circuits decompose

## See Also

- [SHAPEFABRIC.md](SHAPEFABRIC.md) - Shape-routed compute architecture
- [FROZEN_SHAPES.md](FROZEN_SHAPES.md) - Mathematical shape foundations
- [ARCHITECTURE.md](ARCHITECTURE.md) - System overview
