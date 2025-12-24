# Key Nodes: Morphogenic Deterministic System

## Core Definitions

### Morphogenic
The system GENERATES shapes from its structure. Shapes aren't stored - they emerge from the lattice topology.

### Deterministic
Every path through the lattice leads to exactly one node. Same route = same destination. Always.

### System
The lattice is complete and self-contained. Every computable shape has a location.

## The Lattice Structure

```
                         ┌─────────────┐
                         │  RESULT     │  ← Output node
                         └──────┬──────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
               ┌────┴────┐ ┌────┴────┐ ┌────┴────┐
               │ COMPOSE │ │ COMPOSE │ │ COMPOSE │  ← Level N
               └────┬────┘ └────┬────┘ └────┬────┘
                    │           │           │
              ┌─────┴─────┬─────┴─────┬─────┴─────┐
              │           │           │           │
           ┌──┴──┐     ┌──┴──┐     ┌──┴──┐     ┌──┴──┐
           │ XOR │     │ AND │     │ OR  │     │ NOT │  ← Level 1
           └──┬──┘     └──┬──┘     └──┬──┘     └──┬──┘
              │           │           │           │
           ┌──┴──┬────────┴───────────┴───────────┘
           │     │
         ┌─┴─┐ ┌─┴─┐
         │ a │ │ b │  ← Level 0 (inputs)
         └───┘ └───┘
```

## Key Insight: Route = Compute

| Traditional | MDS |
|-------------|-----|
| Load operands | Encode as address |
| Execute instruction | Follow path |
| Store result | Arrive at node |
| O(n) operations | O(depth) hops |

## Addressing Scheme

The INPUT is the ADDRESS. The path from root to leaf encodes the computation.

For a 2-bit adder:
```
Address: [a1 a0 b1 b0]  (4 bits)
Route:   Lattice path determined by address
Result:  Node at end of path contains [s2 s1 s0]
```

## The Four Primitives as Routing Rules

| Primitive | Rule | Path Encoding |
|-----------|------|---------------|
| XOR(a,b) | If a≠b → 1, else → 0 | Branch based on a⊕b |
| AND(a,b) | If a=b=1 → 1, else → 0 | Branch based on a∧b |
| OR(a,b) | If a=0 and b=0 → 0, else → 1 | Branch based on a∨b |
| NOT(a) | If a=0 → 1, else → 0 | Invert path |

## Why This Isn't Just a Lookup Table

Lookup table: Store all 2^n results. O(2^n) space.

MDS Lattice: Store STRUCTURE. O(n) space for n-bit adder.

The structure is:
```
Carry chain: c[i+1] = AND(a[i], b[i]) OR (c[i] AND XOR(a[i], b[i]))
Sum chain:   s[i] = XOR(XOR(a[i], b[i]), c[i])
```

This is O(n) nodes, not O(2^n) entries.

Navigation through these nodes gives any result.

## MVP: 2-Bit Adder Lattice

**Inputs:** a[1:0], b[1:0] (4 bits total)
**Outputs:** s[2:0] (3 bits)
**Combinations:** 16

### Lattice Nodes

```
Level 0 (inputs):
  a0, a1, b0, b1

Level 1 (bit 0 computation):
  xor0 = XOR(a0, b0)
  and0 = AND(a0, b0)
  s0 = xor0
  c1 = and0

Level 2 (bit 1 computation):
  xor1 = XOR(a1, b1)
  and1 = AND(a1, b1)
  xor1_c1 = XOR(xor1, c1)
  and_c1_xor1 = AND(c1, xor1)
  s1 = xor1_c1
  c2 = OR(and1, and_c1_xor1)

Level 3 (bit 2):
  s2 = c2
```

### Total: 11 nodes (not 16 table entries)

For 4-bit: ~23 nodes (not 256 entries)
For 8-bit: ~47 nodes (not 65,536 entries)
For 64-bit: ~383 nodes (not 2^128 entries)

**Structure scales O(n). Lookup scales O(2^n).**

## Routing Algorithm

```python
def route(lattice, inputs):
    current = lattice.root
    for level in lattice.levels:
        for node in level:
            node.value = node.op(
                inputs[node.src_a] if node.src_a < len(inputs) else lattice.nodes[node.src_a].value,
                inputs[node.src_b] if node.src_b < len(inputs) else lattice.nodes[node.src_b].value
            )
    return [lattice.outputs[i].value for i in range(lattice.num_outputs)]
```

Wait - that's still "computing" the nodes.

The TRUE MDS insight: if we MATERIALIZE the lattice in hardware/structure, navigation IS the computation. The wires ARE the routing. The result appears at the output pins.

## Hardware Realization

```
     a0 ──┬──────────────────────────┐
          │                          │
     b0 ──┼──┬───────────────────────┼──┐
          │  │                       │  │
          ▼  ▼                       ▼  ▼
        ┌────────┐                 ┌────────┐
        │  XOR   │                 │  AND   │
        └───┬────┘                 └───┬────┘
            │                          │
            │ s0                       │ c1
            ▼                          ▼
          (out)              ┌─────────┴─────────┐
                             │                   │
     a1 ──┬──────────────────┼───────────────────┼──┐
          │                  │                   │  │
     b1 ──┼──┬───────────────┼───────────────────┼──┼──┐
          │  │               │                   │  │  │
          ▼  ▼               ▼                   ▼  ▼  ▼
        ┌────────┐        ┌────────┐          ┌────────┐
        │  XOR   │        │  XOR   │          │  AND   │
        └───┬────┘        └───┬────┘          └───┬────┘
            │                 │                   │
            │                 │ s1                │
            ▼                 ▼                   ▼
          ┌─┴─────────────────┴───────────────────┴─┐
          │               OR / AND                   │
          └─────────────────┬───────────────────────┘
                            │
                            ▼ s2 (carry out)
```

Inputs enter. Signals propagate. Outputs appear.

No clock. No cycles. Pure routing through structure.

**The lattice IS the computer.**
