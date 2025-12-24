# MDS: Morphogenic Deterministic System - MVP Specification

## Concept

**The lattice IS the computation. Navigation IS the result.**

## MVP: 2-Bit Adder Lattice

### Inputs
- a: 2-bit number [a1, a0]
- b: 2-bit number [b1, b0]

### Outputs
- s: 3-bit sum [s2, s1, s0]

### Lattice Structure

```
LEVEL 0: INPUTS
┌────┬────┬────┬────┐
│ a0 │ a1 │ b0 │ b1 │
└──┬─┴──┬─┴──┬─┴──┬─┘
   │    │    │    │
   ▼    ▼    ▼    ▼

LEVEL 1: BIT 0 PRIMITIVES
┌─────────────┬─────────────┐
│ xor0        │ c1          │
│ =XOR(a0,b0) │ =AND(a0,b0) │
│ (this is s0)│ (carry)     │
└──────┬──────┴──────┬──────┘
       │             │
       ▼             ▼

LEVEL 2: BIT 1 PRIMITIVES
┌─────────────┬─────────────┬─────────────┐
│ xor1        │ and1        │ xor1_c1     │
│ =XOR(a1,b1) │ =AND(a1,b1) │ =XOR(xor1,  │
│             │             │      c1)    │
│             │             │ (this is s1)│
└──────┬──────┴──────┬──────┴──────┬──────┘
       │             │             │
       ▼             ▼             ▼

LEVEL 3: CARRY LOGIC
┌─────────────────────────────┐
│ c1_and_xor1 = AND(c1, xor1) │
│ c2 = OR(and1, c1_and_xor1)  │
│ (this is s2)                │
└─────────────────────────────┘
```

### Node Count
- Level 0: 4 input nodes
- Level 1: 2 nodes (xor0/s0, c1)
- Level 2: 3 nodes (xor1, and1, xor1_c1/s1)
- Level 3: 2 nodes (c1_and_xor1, c2/s2)
- **Total: 11 nodes**

Compare to lookup table: 16 entries (2^4 inputs)
Savings: 31% for 2-bit, scales to 99.99%+ for larger

## Implementation

### Node Structure
```python
@dataclass
class Node:
    id: str
    op: str          # 'INPUT', 'XOR', 'AND', 'OR', 'NOT'
    src_a: str       # Source node id or None
    src_b: str       # Source node id or None
    value: int = 0   # Current value after navigation
```

### Lattice Class
```python
class MDS_Lattice:
    nodes: Dict[str, Node]
    levels: List[List[str]]  # Node ids by level
    outputs: List[str]       # Output node ids

    def navigate(self, inputs: Dict[str, int]) -> List[int]:
        """Navigate lattice with given inputs. Return outputs."""

    def visualize(self) -> str:
        """Return ASCII visualization of lattice."""
```

### Navigation Algorithm
```python
def navigate(self, inputs):
    # Set input node values
    for name, value in inputs.items():
        self.nodes[name].value = value

    # Propagate through levels (this IS the routing)
    for level in self.levels[1:]:  # Skip input level
        for node_id in level:
            node = self.nodes[node_id]
            a = self.nodes[node.src_a].value
            b = self.nodes[node.src_b].value if node.src_b else 0

            # Route based on operation (no arithmetic!)
            if node.op == 'XOR':
                node.value = a ^ b
            elif node.op == 'AND':
                node.value = a & b
            elif node.op == 'OR':
                node.value = a | b
            elif node.op == 'NOT':
                node.value = 1 - a

    # Read outputs
    return [self.nodes[out].value for out in self.outputs]
```

**Key insight**: The operations `a ^ b`, `a & b`, `a | b` are ROUTING decisions, not polynomial calculations. We're following paths, not computing values.

## Demo Script

```
╔══════════════════════════════════════════════════════════════╗
║         MORPHOGENIC DETERMINISTIC SYSTEM                     ║
║         2-Bit Adder Lattice Demo                             ║
╚══════════════════════════════════════════════════════════════╝

Building lattice...
  Level 0: a0, a1, b0, b1 (inputs)
  Level 1: xor0, c1
  Level 2: xor1, and1, xor1_c1
  Level 3: c1_and_xor1, c2

Lattice complete: 11 nodes

─────────────────────────────────────────────────────────────────

TEST: 2 + 1 = ?

Inputs: a=2 (binary: 10), b=1 (binary: 01)

Navigating...
  a0=0, a1=1, b0=1, b1=0

  Level 1:
    xor0 = XOR(0,1) → path: different → 1 (s0)
    c1   = AND(0,1) → path: not both  → 0

  Level 2:
    xor1     = XOR(1,0) → path: different → 1
    and1     = AND(1,0) → path: not both  → 0
    xor1_c1  = XOR(1,0) → path: different → 1 (s1)

  Level 3:
    c1_and_xor1 = AND(0,1) → path: not both → 0
    c2          = OR(0,0)  → path: neither  → 0 (s2)

RESULT: s2=0, s1=1, s0=1 → binary: 011 → decimal: 3

VERIFICATION: 2 + 1 = 3 ✓

─────────────────────────────────────────────────────────────────

The answer was not computed. It was NAVIGATED.
The lattice contains all 16 possible additions.
We routed to location (a=2, b=1) and read the answer.

╔══════════════════════════════════════════════════════════════╗
║  "The shapes exist. We just find them."                      ║
╚══════════════════════════════════════════════════════════════╝
```

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Lattice builds correctly | 11 nodes |
| All 16 cases correct | 100% |
| Navigation is visually distinct from computation | Clear "path" language |
| Scales linearly demonstrated | Show 4-bit = ~23 nodes |
| Mind = blown | Viewer gets it |

## File Structure

```
/workspace/ZOR/foundry/mds/
├── __init__.py
├── lattice.py      # MDS_Lattice class
├── demo.py         # Interactive demo
└── test_mds.py     # Verify all cases
```

## Next Steps After MVP

1. **Scale to 8-bit** - Show O(n) node growth
2. **Visualize as graph** - Graphviz/networkx output
3. **Hardware mapping** - Each node = wire routing
4. **Freeze integration** - Verilog → Lattice → Navigate

## The Pitch

> "Traditional computers COMPUTE answers. MDS NAVIGATES to answers that already exist. The lattice is the computation. The structure is the program. We're not building calculators. We're building maps to mathematical truth."

Build it. Demo it. Let it speak.
