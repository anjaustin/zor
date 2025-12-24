# ZIT-1 Hollywood Edition Specification

```
+===========================================================================+
|                                                                           |
|   ZIT-1 Hollywood Edition                                                 |
|   "The Resonant Transputer"                                               |
|                                                                           |
|   Not a computer. A tuning fork in a mesh.                               |
|                                                                           |
+===========================================================================+
```

## 1. Core Principle

**"Topology is Program. The OS IS the wiring."**

ZIT-1 Hollywood Edition is not just a detector chip - it's a **Resonant Transputer**,
an active mesh node in the Hollywood Squares fabric. The wiring between nodes defines
the algorithm. The same kernel with different wiring produces different behavior.

### Demonstrated Results

```
Initial: [ 42,  17,  93,   8,  55,  71,  23,  64]
         ↓ (9 cycles with bubble sort kernel)
Final:   [  8,  17,  23,  42,  55,  64,  71,  93]
```

The frozen shape (comparator) never changed. Only the wiring defined the algorithm.

---

## 2. Architecture

### Block Diagram

```
                              ZIT-1 NODE
    +----------------------------------------------------------+
    |                                                          |
    |   NORTH ←→ +--------+                                    |
    |            | NEIGH  |                                    |
    |   EAST  ←→ | LATCH  |←──────────────────────────────+   |
    |            | (4x)   |                                |   |
    |   SOUTH ←→ +--------+                                |   |
    |                |                                     |   |
    |   WEST  ←→     ↓                                     |   |
    |            +--------+     +---------+                |   |
    |            |  MUX   |────→| COMPARE |──→ RESONANCE  |   |
    |            +--------+     | (frozen |                |   |
    |                ↑          |  shape) |                |   |
    |                |          +---------+                |   |
    |            +--------+          |                     |   |
    |            |   S    |←─────────┘ (swap if needed)    |   |
    |            | (state)|                                |   |
    |            +--------+                                |   |
    |                |                                     |   |
    |                ↓                                     |   |
    |            +--------+                                |   |
    |            | TX_DATA|────────────────────────────────+   |
    |            +--------+   (broadcast to all neighbors)     |
    |                                                          |
    +----------------------------------------------------------+
```

### The Hollywood Square

Each node is a "Hollywood Square" with:
- **State Register (S)**: The node's current value
- **Neighbor Latches**: Captured values from N/S/E/W neighbors
- **Frozen Shape**: The kernel (comparator for sorting, Zit detector for classification)
- **Resonance Output**: Stability indicator

---

## 3. Protocol: Deterministic Shoving

Each cycle has three phases:

```
    LISTEN          REACT           SHOVE
    +------+       +-------+       +------+
    | Recv |  ──→  | Decide|  ──→  | Send |
    | data |       | action|       | data |
    +------+       +-------+       +------+
       ↓              ↓               ↓
    Capture      Apply frozen     Broadcast
    neighbors    shape kernel     new state
```

### Phase Details

| Phase | Duration | Action |
|-------|----------|--------|
| LISTEN | 4 clocks | Capture neighbor values from previous SHOVE |
| REACT | 4 clocks | Apply frozen shape, compute swap, update state (once) |
| SHOVE | 4 clocks | Broadcast current state to all neighbors |

### Timing

- One complete cycle = 12 clocks
- At 100 MHz = 120 ns per cycle
- 8-element sort in 9 cycles = 1.08 µs

---

## 4. Frozen Shapes

### Shape 1: Bubble Sort Comparator

The first implemented kernel for odd-even transposition sort:

```verilog
// Direction-aware comparison:
//   East: swap if me > neighbor (push larger right)
//   West: swap if neighbor > me (pull larger from right)
wire compare_east = (direction == EAST);
wire should_swap = neighbor_received &&
    (compare_east ? (S > neighbor) : (neighbor > S));
```

### Shape 2: Zit Detector (Future)

The XOR resonance detector for classification:

```verilog
// Hamming distance comparison
wire [9:0] hamming = popcount(S ^ neighbor);
wire resonance = (hamming < theta);
```

---

## 5. Fabric Topologies

### 1D Line (Implemented)

```
[0] ←→ [1] ←→ [2] ←→ [3] ←→ [4] ←→ [5] ←→ [6] ←→ [7]
```

- Odd-even transposition sort
- 8 nodes, O(n) cycles to sort
- Proven in simulation: 9 cycles for 8 elements

### 2D Grid (Designed)

```
[0,0] ←→ [1,0] ←→ [2,0] ←→ [3,0]
  ↕        ↕        ↕        ↕
[0,1] ←→ [1,1] ←→ [2,1] ←→ [3,1]
  ↕        ↕        ↕        ↕
[0,2] ←→ [1,2] ←→ [2,2] ←→ [3,2]
  ↕        ↕        ↕        ↕
[0,3] ←→ [1,3] ←→ [2,3] ←→ [3,3]
```

- Toroidal wrap-around
- 16 nodes, 4x4 grid
- Potential for 2D sorting, constraint propagation

### 3D Hypercube (Future)

Same kernel, 8x8x8 topology = different emergent behavior.

---

## 6. Implementation Files

| File | Description |
|------|-------------|
| `zit_node.v` | The Hollywood Square module |
| `zit_line.v` | 1D line fabric with odd-even controller |
| `zit_line_tb.v` | Testbench proving bubble sort works |
| `zit_fabric.v` | 2D grid fabric (in zit_node.v) |
| `zit_fabric_tb.v` | 2D grid testbench |

---

## 7. Resource Estimates

### Per Node
- State register: 8 flip-flops (8-bit state)
- Neighbor latches: 32 flip-flops (4 × 8-bit)
- Comparator: ~20 gates
- Control logic: ~50 gates
- **Total per node**: ~150 gates

### 8-Node Line
- 8 × 150 = 1,200 gates
- Controller: ~100 gates
- **Total**: ~1,300 gates

### 64-Node Grid (8×8)
- 64 × 150 = 9,600 gates
- Controller: ~150 gates
- Routing: ~500 gates
- **Total**: ~10,250 gates

Fits easily in iCE40UP5K (5,280 LUTs).

---

## 8. Key Insights

1. **The OS isn't software running ON the chip. The OS IS the shape of the connections.**

2. **Same kernel + different wiring = different behavior.**
   - 1D line + comparator = Bubble sort
   - 2D grid + comparator = ???
   - 2D grid + Zit detector = Pattern recognition fabric

3. **The frozen shape doesn't need to know the topology.**
   The comparator works identically whether it's in a 1D line, 2D grid, or 6D hypercube.

4. **Resonance is local stability.**
   A node is "resonant" when it doesn't need to swap with its active neighbor.
   Global convergence = all nodes resonant for two consecutive phases (even + odd).

5. **Fire the Manager. Remove the Warehouse. Build an Assembly Line.**
   No instruction decode. No operand fetch. No register file. Just pure dataflow.

---

## 9. Verification Results

### Bubble Sort Test

```
Initial: [ 42,  17,  93,   8,  55,  71,  23,  64]

Cycle  1 [even]: [ 42,  17,  93,   8,  55,  71,  23,  64]
Cycle  2 [odd ]: [ 42,  17,  93,   8,  55,  23,  71,  64]
Cycle  3 [even]: [ 17,  42,   8,  93,  23,  55,  64,  71]
Cycle  4 [odd ]: [ 17,   8,  42,  23,  93,  55,  64,  71]
Cycle  5 [even]: [  8,  17,  23,  42,  55,  93,  64,  71]
Cycle  6 [odd ]: [  8,  17,  23,  42,  55,  64,  93,  71]
Cycle  7 [even]: [  8,  17,  23,  42,  55,  64,  71,  93]
Cycle  8 [odd ]: [  8,  17,  23,  42,  55,  64,  71,  93]
Cycle  9 [even]: [  8,  17,  23,  42,  55,  64,  71,  93]

[PASS] SORTED in 9 cycles
Final: [  8,  17,  23,  42,  55,  64,  71,  93]
```

---

## 10. The Vision

ZIT-1 Hollywood Edition proves that **topology is program**.

This is the first step toward:
- **ZIT-4**: Multi-class classification in a mesh
- **ZIT-8**: Sensor array with pattern recognition
- **NGP-1**: Full shape fabric with multiple frozen shapes
- **Thor**: The Neural Geometric Processor

The wood cuts itself.

---

*"Not a computer. A tuning fork in a mesh."*

*"The OS IS the wiring."*

*"Topology is Program."*
