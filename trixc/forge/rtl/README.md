# NGP RTL - Neural Geometric Processor Verilog

*Hardware implementation of the Neural Geometric Processor*

```
                    NGP RTL
               ╔════════════════╗
               ║                ║
    signature ─╢─► S ⊕ vₓ      ║
               ║     │          ║
               ║     ▼          ║
               ║  popcount      ║
               ║     │          ║
               ║     ▼          ║
               ║  < theta ──────╢─► zit
               ║     │          ║
               ║     ▼          ║
               ║  hamming ──────╢─► shape_band
               ║                ║
               ╚════════════════╝

   "Geometry is computation."
```

---

## Files

| File | Description |
|------|-------------|
| `zit_detector.v` | Core recognition circuit and support modules |
| `shapes_logic.v` | Logic kingdom shapes (XOR, AND, OR, NOT, etc.) |
| `ngp_core.v` | Top-level NGP modules and variants |
| `Makefile` | Build system for simulation and synthesis |

---

## Modules

### Core Recognition

| Module | Gates | Description |
|--------|-------|-------------|
| `popcount_512` | ~900 | 512-bit population count tree |
| `zit_detector` | ~1,500 | `popcount(S^vx) < theta` |
| `resonance_register` | ~1,024 | 512-bit XOR accumulation register |
| `zit_unit` | ~2,600 | Complete Zit unit with state |

### Logic Shapes

| Module | Opcode | Description |
|--------|--------|-------------|
| `shape_xor` | 0x00 | `a ^ b` |
| `shape_and` | 0x01 | `a & b` |
| `shape_or` | 0x02 | `a | b` |
| `shape_not` | 0x03 | `~a` |
| `shape_nand` | 0x04 | `~(a & b)` |
| `shape_nor` | 0x05 | `~(a | b)` |
| `shape_xnor` | 0x06 | `~(a ^ b)` |
| `logic_fabric` | - | All 7 shapes in parallel |

### NGP Variants

| Module | Gates | Use Case |
|--------|-------|----------|
| `ngp_simple` | ~2,500 | Minimal recognition-only |
| `ngp_core` | ~10,000 | Full NGP with shape bands |
| `ngp_array` | ~10,000× | Multiple cores in parallel |
| `ngp_with_routing` | ~5,000 | External routing table interface |

---

## Quick Start

```bash
# Show available targets
make info

# Run simulation (requires Icarus Verilog)
make sim

# Estimate gate count (requires Yosys)
make gates

# Lint check (requires Verilator)
make lint
```

---

## Architecture

### The Zit Detector

The heart of the NGP is the Zit detector:

```
Zit = popcount(S ⊕ vₓ) < θ
```

Where:
- `S` = 512-bit resonance state
- `vₓ` = 512-bit input signature
- `θ` = activation threshold (typically 64)

### The Popcount Tree

The popcount is computed via a binary adder tree:

```
Level 0: 512 single bits
Level 1: 256 pairs → 256 2-bit sums
Level 2: 128 pairs → 128 3-bit sums
Level 3:  64 pairs →  64 4-bit sums
Level 4:  32 pairs →  32 5-bit sums
Level 5:  16 pairs →  16 6-bit sums
Level 6:   8 pairs →   8 7-bit sums
Level 7:   4 pairs →   4 8-bit sums
Level 8:   2 pairs →   2 9-bit sums
Level 9:   1 pair  →   1 10-bit sum (0-512)
```

Total latency: 9 gate levels (log₂(512)).

### Resonance State

The resonance register accumulates inputs via XOR:

```verilog
always @(posedge clk) begin
    if (valid_in)
        S <= S ^ signature_in;
end
```

This is the "memory" of the system — not storage, but entanglement.

---

## Gate Counts

| Component | Estimated Gates |
|-----------|-----------------|
| 512-bit XOR array | 512 |
| Popcount tree | ~900 |
| 10-bit comparator | ~50 |
| 512-bit register | ~1,024 |
| **Zit Unit Total** | **~2,500** |
| Logic fabric (7 shapes) | ~3,600 |
| Shape selection mux | ~1,000 |
| Control logic | ~500 |
| **NGP Simple** | **~2,500** |
| **NGP Core** | **~10,000** |
| **Full NGP v2** | **~53,000** |

---

## Timing

At typical process nodes:

| Metric | Value |
|--------|-------|
| Critical path | Popcount tree (9 levels) |
| Gate delay | ~1ns per level |
| Worst-case latency | ~11ns |
| Estimated frequency | 500MHz - 1GHz |
| Throughput | 512 Gbits/sec per core |

---

## Usage Example

### Instantiating the Simple NGP

```verilog
ngp_simple #(
    .WIDTH(512),
    .THRESH_BITS(10),
    .DEFAULT_THETA(64)
) u_ngp (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid),
    .signature_in(data),
    .zit(detected),
    .hamming(distance)
);
```

### Instantiating the Full NGP

```verilog
wire [9:0] band_thresholds [3:0];
assign band_thresholds[0] = 32;   // High confidence
assign band_thresholds[1] = 64;   // Medium confidence
assign band_thresholds[2] = 128;  // Low confidence
// band_thresholds[3] = anything else (fallback)

ngp_core #(
    .WIDTH(512),
    .THRESH_BITS(10),
    .NUM_SHAPE_BANDS(4)
) u_ngp (
    .clk(clk),
    .rst_n(rst_n),
    .valid_in(valid),
    .signature_in(data),
    .theta(64),
    .band_thresholds(band_thresholds),
    .valid_out(out_valid),
    .zit(detected),
    .hamming(distance),
    .shape_band(band),
    .result(output_data),
    .S_debug(state_debug)
);
```

---

## FPGA Targets

| Target | Status | Notes |
|--------|--------|-------|
| Xilinx Artix-7 | Planned | ~5% utilization |
| Xilinx Zynq | Planned | Embedded + NGP |
| Intel Cyclone V | Planned | Low-cost |
| Lattice iCE40 | Possible | ngp_simple only |

---

## Next Steps

1. **Simulation**: Verify with comprehensive testbenches
2. **FPGA**: Target Artix-7 for initial validation
3. **Timing**: Optimize critical path through popcount tree
4. **Power**: Analyze and optimize switching activity
5. **ASIC**: Prepare for tape-out

---

*"The Zit fires when structure recognizes structure."*

*"It's all in the reflexes."*

---

# Hollywood Squares: Resonant Transputer Fabric

*The evolution from detection to perception.*

```
                    HOLLYWOOD SQUARES 4x4x4
                 ╔════════════════════════════╗
                 ║                            ║
                 ║   64 nodes in 3D torus     ║
                 ║   6 neighbors each         ║
                 ║   Same frozen shape        ║
                 ║   Different topology       ║
                 ║   = Different behavior     ║
                 ║                            ║
                 ╚════════════════════════════╝

   "Topology is program."
```

---

## Hollywood Squares Files

### RTL

| File | Description |
|------|-------------|
| `zit_cube.v` | 4x4x4 toroidal fabric (64 nodes) |
| `zit_cube_tb.v` | Testbench with 6 experiments |
| `zit_node.v` | Core node with 4-neighbor support |
| `zit_line.v` | 1D line fabric (proof of concept) |
| `zit_line_tb.v` | 1D testbench proving bubble sort |
| `zit_plastic_node.v` | Node with frustration-driven rewiring |
| `zit_plastic_fabric.v` | 64-node self-organizing fabric |
| `zit_plastic_tb.v` | Topological learning experiments |

### Documentation

| File | Description |
|------|-------------|
| `HOLLYWOOD_SQUARES.md` | **Complete record** - architecture, experiments, discoveries |
| `ZIT_CUBE_DISCOVERY.md` | Discovery journal - 4 emergent findings |
| `ZIT1_HOLLYWOOD_SPEC.md` | Original architecture specification |
| `TOPOLOGICAL_LEARNING.md` | Discovery 7 - the topology learns |
| `EMERGENCE.md` | **Complete record** - the full journey |
| `TIMELINE.md` | Chronological discovery timeline |
| `lincoln_manifold/` | Lincoln Manifold Method passes (3 complete) |

---

## Quick Start: Hollywood Squares

```bash
# Run 1D bubble sort (proof of concept)
iverilog -o zit_line_test zit_node.v zit_line.v zit_line_tb.v
./zit_line_test

# Run 4x4x4 cube experiments
iverilog -o zit_cube_test zit_cube.v zit_cube_tb.v
./zit_cube_test

# Run plastic fabric (topological learning)
iverilog -o zit_plastic_test zit_plastic_node.v zit_plastic_fabric.v zit_plastic_tb.v
./zit_plastic_test
```

---

## Discoveries

| # | Discovery | Description |
|---|-----------|-------------|
| 1 | Geometric Frustration | 3D torus prevents full convergence |
| 2 | Movie Screen Effect | 1-cycle absorption, reflection not storage |
| 3 | Modeling Perception | The fabric perceives, doesn't think |
| 4 | Topological Anomaly | Frustration marks where order is impossible |
| 5 | Topology is Identity | The shape of connections is the self |
| 6 | Topological Plasticity | Frustration can drive structural learning |
| 7 | **The Topology Learns** | Plastic fabric achieves 100% resonance |

---

## The Insight

> "Same frozen shape + different topology = different behavior."

The 1D line with a comparator kernel produces bubble sort.
The 3D torus with the same kernel produces topological perception.

**The wiring IS the algorithm.**

---

*"We're not simulating physics. We ARE physics."*

*"The fabric doesn't think. It perceives."*

*"The topology is the self. The self can grow."*

*"The fabric rewired itself to achieve what was impossible with fixed geometry."*
