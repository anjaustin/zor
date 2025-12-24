# Hollywood Squares: The Complete Record

```
+===========================================================================+
|                                                                           |
|   HOLLYWOOD SQUARES                                                       |
|   A Resonant Transputer Fabric                                           |
|                                                                           |
|   "We're not simulating physics. We ARE physics."                        |
|                                                                           |
|   64 nodes. 6 neighbors. One frozen shape.                               |
|   And it perceives.                                                       |
|                                                                           |
+===========================================================================+
```

---

## Table of Contents

1. [Origin Story](#1-origin-story)
2. [Architecture](#2-architecture)
3. [The Protocol](#3-the-protocol)
4. [Experiments](#4-experiments)
5. [Discoveries](#5-discoveries)
6. [Lincoln Manifold Passes](#6-lincoln-manifold-passes)
7. [The Second Star Constant](#7-the-second-star-constant)
8. [File Inventory](#8-file-inventory)
9. [What We Learned](#9-what-we-learned)
10. [Open Questions](#10-open-questions)

---

## 1. Origin Story

### The Beginning: 1D Line

We built a 1D line of 8 "Zit" nodes to prove a concept:
- Each node has a state (8-bit value)
- Each node compares with its neighbor
- If out of order, swap

**Result:** Bubble sort in 9 cycles.

```
Initial: [ 42,  17,  93,   8,  55,  71,  23,  64]
Final:   [  8,  17,  23,  42,  55,  64,  71,  93]
```

**Key Insight:** The frozen shape (comparator) never changed. Only the wiring.
**TOPOLOGY IS PROGRAM.**

### The Question: What About 3D?

Same frozen shape. 64 nodes. 3D toroidal topology.
What behavior emerges?

We didn't know. That was the point.

---

## 2. Architecture

### 2.1 The Node (zit_cube_node)

```
                    +Z neighbor
                         ↑
                         |
           +Y neighbor ←─┼─→ -Y neighbor
                        /|\
                       / | \
            -X neighbor  |  +X neighbor
                         |
                         ↓
                    -Z neighbor
```

Each node has:
- **State Register (S):** 8-bit value
- **6 Neighbor Latches:** Captured values from ±X, ±Y, ±Z
- **Frozen Shape:** Comparator kernel
- **Resonance Output:** Am I locally consistent?

### 2.2 The Cube (zit_cube)

```
                    Layer 3 (z=3)
                 [48][49][50][51]
                 [52][53][54][55]
                 [56][57][58][59]
                 [60][61][62][63]
                        ↕
                    Layer 2 (z=2)
                 [32][33][34][35]
                 [36][37][38][39]
                 [40][41][42][43]
                 [44][45][46][47]
                        ↕
                    Layer 1 (z=1)
                 [16][17][18][19]
                 [20][21][22][23]
                 [24][25][26][27]
                 [28][29][30][31]
                        ↕
                    Layer 0 (z=0)
                 [ 0][ 1][ 2][ 3]
                 [ 4][ 5][ 6][ 7]
                 [ 8][ 9][10][11]
                 [12][13][14][15]
```

**Addressing:** `index = x + (y * 4) + (z * 16)`

**Topology:** 3D Toroidal (all edges wrap around)

### 2.3 The Frozen Shape: Comparator

```verilog
// Direction-aware comparison:
// Positive directions (+X, +Y, +Z): swap if me > neighbor
// Negative directions (-X, -Y, -Z): swap if neighbor > me
wire positive_dir = (phase[0] == 1'b0);
wire should_swap = active_received &&
    (positive_dir ? (S > active_neighbor) : (active_neighbor > S));
```

The comparator creates a "flow" direction for values.
In 1D, this produces sorting.
In 3D, this produces... something else.

---

## 3. The Protocol

### 3.1 Six-Phase Cycle

```
Phase 0: +X (East)      Compare with +X neighbor
Phase 1: -X (West)      Compare with -X neighbor
Phase 2: +Y (South)     Compare with +Y neighbor
Phase 3: -Y (North)     Compare with -Y neighbor
Phase 4: +Z (Up)        Compare with +Z neighbor
Phase 5: -Z (Down)      Compare with -Z neighbor
```

### 3.2 Three Sub-Phases Per Phase

```
LISTEN (4 clocks) → REACT (4 clocks) → SHOVE (4 clocks)
```

| Sub-Phase | Action |
|-----------|--------|
| LISTEN | Capture neighbor values |
| REACT | Apply frozen shape, compute swap, update state |
| SHOVE | Broadcast current value to all neighbors |

### 3.3 Timing

- One phase: 12 clocks
- One cycle (6 phases): 72 clocks
- At 100 MHz: 720 ns per cycle
- Perception rate: ~1.4 million "frames" per second

---

## 4. Experiments

### Experiment 1: Uniform Seed

**Setup:** All 64 nodes set to value 42.

**Result:** 64/64 resonant. Full convergence.

**Meaning:** No structure → no frustration.

### Experiment 2: Gradient Seed

**Setup:** Node[i] = i * 4 (values 0, 4, 8, ..., 252)

**Result:** 48/64 resonant. 16 frustrated at toroidal wrap.

**Meaning:** The gradient is "correct" ordering, but the wrap creates conflict.

### Experiment 3: Random Sort (3D Bubble Sort)

**Setup:** 64 pseudo-random 8-bit values.

**Result:** 43/64 resonant. 21 frustrated. Stable equilibrium.

**Meaning:** Geometric frustration. Some nodes can never be satisfied.

### Experiment 4: Movie Screen

**Setup:** Run to equilibrium, then inject extreme value (255).

**Result:** Value absorbed in ONE cycle. Fabric returns to equilibrium.

**Meaning:** The fabric reflects, it doesn't store. No memory, only now.

### Experiment 5: Edge Detection

**Setup:** Layers 0-1 = 0 (black), Layers 2-3 = 255 (white).

**Expected:** Frustration at visible edge (z=1 to z=2).

**Actual:** Frustration at toroidal wrap (layer 3 only).

**Meaning:** The fabric detects TOPOLOGICAL CONFLICTS, not geometric edges.

### Experiment 6: Second Star Constant

**Setup:** Seed LFSR with 1122911624 (0x42EB9CE8), generate 64 values.

**Result:** 37/64 resonant. 27 frustrated. Unique asymmetric pattern.

**Meaning:** Each seed produces a unique "frustration fingerprint."

---

## 5. Discoveries

### Discovery 1: Geometric Frustration

The 3D toroidal topology with a comparator kernel creates stable non-convergence.

- ~67% of nodes achieve resonance
- ~33% are frustrated (stable but not satisfied)
- This is physics (analogous to spin glasses)

**The frustrated equilibrium is not a failure. It is the answer.**

### Discovery 2: Movie Screen Effect

Disturbances are absorbed in one cycle.

- Inject any value anywhere
- The fabric diffuses it through neighbor interactions
- Returns to frustrated equilibrium
- No memory, only reflection

**The fabric is a mirror, not a hard drive.**

### Discovery 3: Modeling Perception

The fabric doesn't think. It perceives.

| Thinking | Perceiving |
|----------|------------|
| Sequential | Immediate |
| Symbolic | Direct |
| Stored representations | Present participation |
| Can be wrong | Just IS |

**We built a synthetic sense organ.**

### Discovery 4: Topological Anomaly Detection

The fabric doesn't detect edges. It detects where local order cannot be globally satisfied.

- Compatible edges (consistent with ordering): no frustration
- Incompatible edges (topological conflict): frustration

**The fabric computes topological invariants.**

---

## 6. Lincoln Manifold Passes

### Pass 1: 1D → 3D

| Phase | Finding |
|-------|---------|
| RAW | 1D sorts; what does 3D do? |
| NODES | 14 nodes of interest identified |
| REFLECT | Topology is program; same shape, different behavior |
| SYNTHESIZE | Build 4x4x4, observe emergence |

**Result:** Geometric frustration discovered.

### Pass 2: Physics → Mathematics

| Phase | Finding |
|-------|---------|
| RAW | Frustration is stable; what does it mean? |
| NODES | Frustration as detection; output is not values |
| REFLECT | Fabric perceives; frustration marks features |
| SYNTHESIZE | Test edge detection; discover topological anomaly |

**Result:** Topological anomaly detection discovered.

### Pass 3: ???

The Hat goes deeper. What's next?

---

## 7. The Second Star Constant

**Value:** 1122911624 (decimal) = 0x42EB9CE8 (hex)

**Used as:** LFSR seed to generate 64 initial values.

**Frustration Fingerprint:**

```
Layer 3: 0000 1111 1111 0011  (10 frustrated)
Layer 2: 0000 0000 0100 1100  (3 frustrated)
Layer 1: 0110 0000 0000 0100  (3 frustrated)
Layer 0: 1011 1111 1001 0011  (11 frustrated)

Total: 27 frustrated nodes (37 resonant)
```

**Significance:** Each seed produces a unique topological signature.

---

## 8. File Inventory

### RTL Files

| File | Lines | Description |
|------|-------|-------------|
| `zit_cube.v` | 472 | 4x4x4 fabric with 64 nodes |
| `zit_cube_tb.v` | ~850 | Testbench with 6 experiments |
| `zit_node.v` | 500 | Core node module (2D version) |
| `zit_line.v` | 208 | 1D line fabric (proof of concept) |
| `zit_line_tb.v` | 244 | 1D testbench |
| `zit_fabric_tb.v` | 353 | 2D fabric testbench |
| `zit_detector.v` | 339 | Original Zit detector |

### Documentation Files

| File | Description |
|------|-------------|
| `HOLLYWOOD_SQUARES.md` | This document |
| `ZIT_CUBE_DISCOVERY.md` | Discovery record |
| `ZIT1_HOLLYWOOD_SPEC.md` | Original architecture spec |

### Lincoln Manifold Files (in /tmp/)

| File | Description |
|------|-------------|
| `hollywood_4x4x4_nodes.md` | Pass 1: Nodes |
| `hollywood_4x4x4_reflect.md` | Pass 1: Reflection |
| `hollywood_4x4x4_synth.md` | Pass 1: Synthesis |
| `hollywood_perception_raw.md` | Pass 2: Raw |
| `hollywood_perception_nodes.md` | Pass 2: Nodes |
| `hollywood_perception_reflect.md` | Pass 2: Reflection |
| `hollywood_perception_synth.md` | Pass 2: Synthesis |

---

## 9. What We Learned

### The Progression of Understanding

```
What we built     →  What we thought it was  →  What it actually is
─────────────────────────────────────────────────────────────────────
64 nodes             A sorting network           A perceptual field
Comparator           An ordering function        A consistency detector
Toroidal topology    A boundary-free space       A closed logical manifold
Frustration          A failure to converge       A detection of impossibility
```

### The Ontological Shift

```
Traditional Computing        Resonant Computing
────────────────────        ──────────────────
Symbols                     Shapes
Representation              Participation
Storage                     Reflection
Computation                 Perception
Can be wrong                Just IS
```

### The Core Insight

**The fabric doesn't compute what we tell it to compute.**
**It perceives what its topology makes it perceive.**

We didn't program edge detection.
We didn't program topological analysis.
We created conditions. Behavior emerged.

---

## 10. Open Questions

### Immediate

1. What patterns can the fabric distinguish?
2. What is the "resolution" of topological perception?
3. How does frustration count vary with input structure?

### Architectural

4. What happens with non-toroidal (open boundary) topology?
5. What happens with different frozen shapes (Zit, majority)?
6. What happens when frustration pattern feeds another fabric?

### Deep

7. Is there a homomorphism from input space to frustration space?
8. What are the fixed points of the perception→action loop?
9. Can the fabric detect its own inconsistencies?

### Deepest

10. What would it mean to "dream" - run the fabric backwards?
11. Is there a frustration pattern that encodes "self"?
12. ~~What happens at Pass 3 of the Lincoln Manifold?~~ **ANSWERED: See below**

### Pass 3 Discoveries (Completed)

Pass 3 of Lincoln Manifold revealed:

- **Discovery 5:** Topology IS identity
- **Discovery 6:** Topological plasticity is possible
- **Discovery 7:** The topology LEARNS (experimentally verified)

The plastic fabric (`zit_plastic_fabric.v`) achieved 64/64 resonance by rewiring itself.
The fixed toroidal topology could only achieve 43/64 due to geometric frustration.
The plastic topology eliminated frustration entirely.

See: `EMERGENCE.md`, `TOPOLOGICAL_LEARNING.md`, `lincoln_manifold/pass3_*.md`

---

## Appendix: Key Quotes

*"Topology is program."*

*"The OS IS the wiring."*

*"We're not simulating physics. We ARE physics."*

*"The fabric doesn't think. It perceives."*

*"The frustration map is where logic cannot be self-consistent."*

*"The Hat goes all the way down."*

*"I think... the fabric doesn't."*

*"The topology IS the self."*

*"The self can GROW."*

*"We're not programming intelligence. We're growing it."*

*"The fabric rewired itself to achieve what was impossible with fixed geometry."*

---

## Appendix: Test Results Summary

```
EXPERIMENT              RESONANCE    FRUSTRATED    NOTES
────────────────────────────────────────────────────────────
FIXED TOPOLOGY (zit_cube):
1. Uniform (42)         64/64        0             Baseline
2. Gradient (0-252)     48/64        16            Wrap conflict
3. Random               43/64        21            Geometric frustration
4. Movie Screen         43/64→43/64  (absorbed)    1-cycle reflection
5. Edge (0|255)         48/64        16            Topology, not geometry
6. Second Star          37/64        27            Unique fingerprint

PLASTIC TOPOLOGY (zit_plastic_fabric):
7. Random + Learning    44→64/64     39→0          FULL RESONANCE ACHIEVED
```

---

## Appendix: Resource Estimates

### Per Node
- State register: 8 FF
- 6 neighbor latches: 48 FF
- Comparator: ~20 gates
- 6-direction mux: ~30 gates
- Control logic: ~50 gates
- **Total per node:** ~200 gates

### Full Cube (64 nodes)
- 64 × 200 = 12,800 gates
- Controller: ~200 gates
- Routing: ~1,000 gates
- **Total:** ~14,000 gates

### FPGA Fit
- iCE40UP5K: 5,280 LUTs (~21,000 gates)
- **Fits with room for debug logic**

### Plastic Fabric (64 nodes)

- 64 plastic nodes × 220 gates = 14,080 gates
- All-to-all value broadcast: ~32,768 routing
- Controller: ~100 gates
- **Total:** ~47,000 gates

Larger than fixed fabric, but it **learns**.

---

*Document created: December 2024*
*Updated: December 2024 - Pass 3 Complete*
*Hollywood Squares 4x4x4: The Resonant Transputer Fabric*
*"The wood cuts itself."*
