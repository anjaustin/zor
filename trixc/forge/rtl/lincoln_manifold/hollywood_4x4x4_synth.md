# Synthesis: Hollywood Squares 4x4x4 - The Phase Change

```
+===========================================================================+
|                                                                           |
|   THE 4x4x4 CUBE                                                          |
|   64 Resonant Transputers in 3D Space                                     |
|                                                                           |
|   "We're not simulating physics. We ARE physics."                         |
|                                                                           |
+===========================================================================+
```

---

## 1. Architecture: The Cube

### 1.1 Physical Layout

```
                    Layer 3 (top)
                 [48][49][50][51]
                 [52][53][54][55]
                 [56][57][58][59]
                 [60][61][62][63]
                        ↕
                    Layer 2
                 [32][33][34][35]
                 [36][37][38][39]
                 [40][41][42][43]
                 [44][45][46][47]
                        ↕
                    Layer 1
                 [16][17][18][19]
                 [20][21][22][23]
                 [24][25][26][27]
                 [28][29][30][31]
                        ↕
                    Layer 0 (bottom)
                 [ 0][ 1][ 2][ 3]
                 [ 4][ 5][ 6][ 7]
                 [ 8][ 9][10][11]
                 [12][13][14][15]
```

### 1.2 Addressing

```
Index = x + (y * 4) + (z * 16)

Where:
  x ∈ [0, 3]  (East-West)
  y ∈ [0, 3]  (North-South)
  z ∈ [0, 3]  (Up-Down)
```

### 1.3 Neighbor Connectivity

Each node has 6 neighbors:

| Direction | Index Offset | Wrap Condition |
|-----------|--------------|----------------|
| +X (East) | +1 | Wrap at x=3 → x=0 |
| -X (West) | -1 | Wrap at x=0 → x=3 |
| +Y (South) | +4 | Wrap at y=3 → y=0 |
| -Y (North) | -4 | Wrap at y=0 → y=3 |
| +Z (Up) | +16 | Wrap at z=3 → z=0 |
| -Z (Down) | -16 | Wrap at z=0 → z=3 |

**Topology:** 3D Toroidal (wraparound on all axes)

---

## 2. Protocol: 6-Phase Cycle

### 2.1 Phase Sequence

```
┌─────────────────────────────────────────────────────────────┐
│                    ONE COMPUTE CYCLE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 0: +X (East)      Each node compares with +X        │
│      ↓                                                      │
│  Phase 1: -X (West)      Each node compares with -X        │
│      ↓                                                      │
│  Phase 2: +Y (South)     Each node compares with +Y        │
│      ↓                                                      │
│  Phase 3: -Y (North)     Each node compares with -Y        │
│      ↓                                                      │
│  Phase 4: +Z (Up)        Each node compares with +Z        │
│      ↓                                                      │
│  Phase 5: -Z (Down)      Each node compares with -Z        │
│      ↓                                                      │
│  [Cycle Complete - check convergence]                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Sub-Phase Timing

Each phase has 3 sub-phases (same as 1D):

```
LISTEN (4 clocks) → REACT (4 clocks) → SHOVE (4 clocks)
```

Total per phase: 12 clocks
Total per cycle: 72 clocks
At 100 MHz: 720 ns per cycle

### 2.3 Convergence

Converged when ALL nodes are resonant for a complete 6-phase cycle.
This means: No swaps needed in any direction.

---

## 3. Frozen Shapes

### 3.1 Shape A: 3D Comparator (Primary)

The natural extension of the 1D/2D comparator:

```verilog
// Direction-aware comparison:
// Positive directions (+X, +Y, +Z): swap if me > neighbor
// Negative directions (-X, -Y, -Z): swap if neighbor > me
wire positive_direction = (phase[0] == 1'b0);  // Even phases are positive
wire should_swap = neighbor_received &&
    (positive_direction ? (S > neighbor) : (neighbor > S));
```

**Expected behavior:** Some form of 3D sorting.
Values should "flow" from one corner to the opposite corner.

### 3.2 Shape B: 3D Zit Detector (Secondary)

XOR resonance extended to 6 neighbors:

```verilog
// Combine Hamming distances from all 6 neighbors
wire [11:0] total_hamming =
    popcount(S ^ neighbor_px) +
    popcount(S ^ neighbor_mx) +
    popcount(S ^ neighbor_py) +
    popcount(S ^ neighbor_my) +
    popcount(S ^ neighbor_pz) +
    popcount(S ^ neighbor_mz);

// Resonant if average Hamming distance below threshold
wire resonance = (total_hamming < (theta * 6));
```

**Expected behavior:** Pattern recognition.
Similar values should cluster in 3D space.

### 3.3 Shape C: Majority Gate (Experimental)

Take the majority value among neighbors:

```verilog
// Count how many neighbors have same value as me
wire [2:0] agreement_count =
    (neighbor_px == S) + (neighbor_mx == S) +
    (neighbor_py == S) + (neighbor_my == S) +
    (neighbor_pz == S) + (neighbor_mz == S);

// If majority disagrees, take the mode value
wire should_adopt = (agreement_count < 3);
```

**Expected behavior:** Consensus.
The fabric should converge to uniform regions.

---

## 4. Implementation Specification

### 4.1 Module: zit_cube

```verilog
module zit_cube #(
    parameter CUBE_SIZE = 4,           // 4x4x4
    parameter STATE_WIDTH = 8          // 8-bit values
)(
    input  wire                        clk,
    input  wire                        rst_n,

    // Phase control
    input  wire [2:0]                  phase,          // 0-5 for 6 directions
    input  wire                        phase_strobe,

    // Seed interface
    input  wire [STATE_WIDTH-1:0]      seed_data,
    input  wire [5:0]                  seed_addr,      // 0-63
    input  wire                        seed_write,

    // Status
    output wire                        all_resonant,
    output wire [63:0]                 resonance_map,

    // Debug: all 64 states (64 * 8 = 512 bits)
    output wire [511:0]                all_states
);
```

### 4.2 Module: zit_cube_controller

```verilog
module zit_cube_controller (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         enable,

    output reg  [2:0]   phase,          // 0-5
    output reg  [1:0]   sub_phase,      // 0=LISTEN, 1=REACT, 2=SHOVE
    output reg          phase_strobe,
    output wire         cycle_complete  // After all 6 phases
);
```

### 4.3 Direction Encoding

```verilog
localparam DIR_PX = 3'd0;  // +X (East)
localparam DIR_MX = 3'd1;  // -X (West)
localparam DIR_PY = 3'd2;  // +Y (South)
localparam DIR_MY = 3'd3;  // -Y (North)
localparam DIR_PZ = 3'd4;  // +Z (Up)
localparam DIR_MZ = 3'd5;  // -Z (Down)
```

---

## 5. Experiments to Run

### Experiment 1: 3D Bubble Sort

**Setup:**
- 64 nodes with random 8-bit values
- Comparator kernel
- 6-phase cycle

**Observe:**
- Does it converge?
- How many cycles to convergence?
- What does "sorted" look like in 3D?
  (Values should flow from (0,0,0) to (3,3,3)?)

### Experiment 2: Pattern Recognition

**Setup:**
- Seed a 3D "pattern" (e.g., a diagonal stripe)
- Query with similar and dissimilar patterns
- Zit detector kernel

**Observe:**
- Does the fabric resonate on similar patterns?
- What's the false positive rate?
- How does 3D improve over 2D?

### Experiment 3: Consensus Formation

**Setup:**
- Seed with multiple distinct values
- Majority gate kernel
- Run until convergence

**Observe:**
- What regions form?
- Is there hysteresis?
- How do initial conditions affect final state?

### Experiment 4: Phase Transition

**Setup:**
- Vary the threshold in Zit detector
- Observe fabric behavior as theta changes

**Observe:**
- Is there a critical theta where behavior changes?
- This would be a phase transition.

---

## 6. Success Criteria

### Minimum Viable

- [ ] 4x4x4 fabric compiles in Icarus Verilog
- [ ] 6-phase controller runs correctly
- [ ] Convergence is detected
- [ ] One experiment produces interesting behavior

### Target

- [ ] All 3 frozen shapes implemented
- [ ] All 4 experiments run
- [ ] At least one emergent behavior documented
- [ ] Resource estimate for iCE40

### Stretch

- [ ] Phase transition behavior observed
- [ ] New emergent behavior discovered (not predicted)
- [ ] Synthesis for iCE40 (actual FPGA)
- [ ] Video of running fabric

---

## 7. Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Topology | 3D Toroidal | No edges = uniform behavior |
| Phase order | +X, -X, +Y, -Y, +Z, -Z | Balanced axis coverage |
| First kernel | Comparator | Continuity from 1D/2D |
| State width | 8 bits | Enough for interesting patterns |
| Convergence | Full 6-phase cycle stable | Conservative, ensures global stability |

---

## 8. The Ontological Frame

This is not a product. This is an experiment.

We are asking: **What does 3D resonant computation look like?**

The answer doesn't exist in a textbook. It exists in the fabric.
We're building a window into a new ontology.

### What We Expect
- Some form of 3D sorting (comparator)
- Some form of pattern clustering (Zit)
- Some form of consensus (majority)

### What We Don't Expect
- The unexpected
- Emergent behaviors we haven't named
- Phase transitions we haven't predicted

### What We're Really Doing

We're creating conditions for something to emerge.
The 4x4x4 is a petri dish for resonant computation.

We're not programming. We're gardening.
Plant the seeds (frozen shapes). Design the soil (topology).
Watch what grows.

---

## 9. Resource Estimates

### Per Node
- 8-bit state: 8 FF
- 6 × 8-bit neighbor latches: 48 FF
- Comparator: ~20 gates
- 6-direction mux: ~30 gates
- Control logic: ~50 gates
- **Total per node:** ~200 gates

### Full Cube (64 nodes)
- 64 × 200 = 12,800 gates
- Controller: ~200 gates
- Routing: ~1,000 gates
- **Total:** ~14,000 gates

### iCE40UP5K Fit
- Available: 5,280 LUTs (~21,000 gates)
- Used: ~14,000 gates
- **Fits with room for debug logic**

---

## 10. Files to Create

| File | Purpose |
|------|---------|
| `zit_cube.v` | 4x4x4 fabric instantiation |
| `zit_cube_controller.v` | 6-phase sequencer |
| `zit_cube_tb.v` | Testbench with experiments |
| `ZIT_CUBE_SPEC.md` | This document |

---

## 11. The Clean Cut

The wood is ready.

We've sharpened the axe through:
1. The proof-of-concept (1D bubble sort)
2. The nodes (14 key insights)
3. The reflection (ontological understanding)

Now we cut:
- Build the 4x4x4 fabric
- Run the experiments
- Observe what emerges
- Document the new territory

**This isn't engineering. This is exploration.**

We're not building a faster computer.
We're building a window into what computation becomes
when it stops being symbolic and becomes physical.

---

*"The Bit and the Atom become the same thing again."*

*"We're closing the gap until there is no gap."*

*"This is not AI. This is synthetic physics."*

---

The synthesis is complete. The specification is ready.
The wood will cut itself.
