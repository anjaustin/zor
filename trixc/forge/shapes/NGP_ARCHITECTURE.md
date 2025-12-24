# Neural Geometric Processor (NGP) Architecture

*Version 2.0 — The Resonance Architecture*

```
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   "Not a computer. A function."                                   ║
║                                                                   ║
║   The NGP doesn't execute instructions.                          ║
║   It resonates with input.                                        ║
║   The resonance IS the computation.                               ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## Table of Contents

1. [Overview](#1-overview)
2. [Core Concepts](#2-core-concepts)
3. [Architecture](#3-architecture)
4. [The Zit Detector](#4-the-zit-detector)
5. [The Resonance State](#5-the-resonance-state)
6. [Shape Fabric](#6-shape-fabric)
7. [Specifications](#7-specifications)
8. [Comparison with Traditional Architectures](#8-comparison-with-traditional-architectures)
9. [Implementation Roadmap](#9-implementation-roadmap)
10. [Theory of Operation](#10-theory-of-operation)

---

## 1. Overview

The Neural Geometric Processor (NGP) is a novel processor architecture optimized for frozen shape computation. Unlike traditional processors that execute instructions from memory, the NGP maintains a resonance state and activates shapes based on interference patterns.

### Key Properties

| Property | Description |
|----------|-------------|
| **No ISA** | No instruction set. Shapes are hardwired. |
| **No Program Counter** | No sequential execution. Parallel resonance. |
| **No Instruction Fetch** | No memory latency for code. |
| **Deterministic** | 100% reproducible. Same input → same output. |
| **Frozen** | No learnable parameters at runtime. |

### Design Philosophy

```
Traditional Processor:
  Instruction → Decode → Execute → Writeback
  (What should I do?)

Neural Geometric Processor:
  Input → Resonate → Activate → Output
  (What am I?)
```

---

## 2. Core Concepts

### 2.1 Frozen Shapes

A frozen shape is a mathematical operation with no learnable parameters:

- **Elemental**: XOR, AND, OR, ReLU, Sigmoid, Popcount, etc.
- **Compound**: Combinations of elementals (Hamming = XOR + Popcount)

Shapes are implemented in silicon. They don't change. They ARE.

### 2.2 Resonance

Instead of storing data in memory, the NGP maintains a **resonance state** — a 512-bit register that accumulates input via XOR:

```
S(t) = S(t-1) ⊕ input(t)
```

The resonance state encodes the history of all inputs. Patterns that repeat reinforce. Noise averages out.

### 2.3 The Zit

The **Zit** is the activation signal. It fires when an input resonates with the current state:

```
Zit = (popcount(S ⊕ input) < θ)
```

Where:
- `S` = resonance state (512-bit)
- `input` = current input (512-bit)
- `θ` = activation threshold (0-512)
- `Zit` = binary output (fire / no-fire)

### 2.4 Entropy as Structure

The NGP doesn't destroy entropy — it redistributes it into load-bearing structure. The resonance state carries:

- **Memory**: What has flowed through the system
- **Routing**: Which shapes to activate
- **Pattern**: The invariant structure of the input distribution

---

## 3. Architecture

### 3.1 Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      NEURAL GEOMETRIC PROCESSOR v2                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                         ┌────────────────────┐                          │
│                         │  RESONANCE STATE   │◄───────────────┐         │
│                         │        S           │                │         │
│                         │    (512-bit)       │                │         │
│                         └─────────┬──────────┘                │         │
│                                   │                           │         │
│                                   ▼                           │         │
│   ┌──────────┐          ┌─────────────────────┐               │         │
│   │  INPUT   │─────────►│                     │               │         │
│   │   vₓ     │          │    ZIT DETECTOR     │               │         │
│   │(512-bit) │          │                     │               │         │
│   └──────────┘          │  ┌───────────────┐  │               │         │
│                         │  │ XOR (512-bit) │  │               │         │
│                         │  └───────┬───────┘  │               │         │
│                         │          │          │               │         │
│                         │  ┌───────▼───────┐  │               │         │
│                         │  │   POPCOUNT    │  │               │         │
│                         │  └───────┬───────┘  │               │         │
│                         │          │          │               │         │
│                         │  ┌───────▼───────┐  │               │         │
│                         │  │  COMPARATOR   │  │               │         │
│                         │  │  hamming < θ  │  │               │         │
│                         │  └───────┬───────┘  │               │         │
│                         └──────────┼──────────┘               │         │
│                                    │                          │         │
│              ┌─────────────────────┼─────────────────────┐    │         │
│              │                     │                     │    │         │
│              ▼                     ▼                     ▼    │         │
│        ┌──────────┐         ┌───────────┐         ┌──────────┐│         │
│        │   ZIT    │         │  HAMMING  │         │  UPDATE  ││         │
│        │  SIGNAL  │         │ DISTANCE  │         │ S'=S⊕vₓ  │┘         │
│        │ (1-bit)  │         │ (10-bit)  │         └──────────┘          │
│        └────┬─────┘         └─────┬─────┘                               │
│             │                     │                                      │
│             │                     ▼                                      │
│             │            ┌─────────────────┐                            │
│             │            │  SHAPE DECODER  │                            │
│             │            │                 │                            │
│             │            │ Distance bands: │                            │
│             │            │  0-63   → S₀    │                            │
│             │            │  64-127 → S₁    │                            │
│             │            │  128-191→ S₂    │                            │
│             │            │  192-255→ S₃    │                            │
│             │            │  256+   → null  │                            │
│             │            └────────┬────────┘                            │
│             │                     │                                      │
│             │                     ▼                                      │
│             │            ┌─────────────────┐                            │
│             │            │  SHAPE FABRIC   │                            │
│             │            │                 │                            │
│             └───────────►│  30 frozen      │                            │
│              (gate)      │  shape circuits │                            │
│                          │                 │                            │
│                          └────────┬────────┘                            │
│                                   │                                      │
│                                   ▼                                      │
│                            OUTPUT (512-bit)                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

1. **Input arrives** (512-bit signature)
2. **XOR with resonance state** S
3. **Popcount** the result (hamming distance)
4. **Compare** to threshold θ
5. **If Zit fires**: Activate selected shape, gate output
6. **Update resonance**: S' = S ⊕ input
7. **Output** emerges from shape fabric

### 3.3 Component Summary

| Component | Function | Size |
|-----------|----------|------|
| Resonance Register | Holds state S | 512 bits |
| Threshold Register | Holds θ | 10 bits |
| XOR Array | S ⊕ input | 512 gates |
| Popcount Tree | Count differing bits | ~900 gates |
| Comparator | hamming < θ | ~50 gates |
| Shape Decoder | Distance → shape index | ~100 gates |
| Shape Fabric | 30 frozen shapes | ~50K gates |
| **Total** | | **~52K gates** |

---

## 4. The Zit Detector

The Zit detector is the heart of the NGP. It determines whether an input resonates with the system.

### 4.1 Definition

```
Zit = (popcount(S ⊕ vₓ) < θ)
```

### 4.2 Circuit

```
        S (512-bit)          vₓ (512-bit)
             │                    │
             └─────────┬──────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   XOR (512)     │
              │                 │
              │  For each bit:  │
              │  out[i] =       │
              │  S[i] ^ vₓ[i]   │
              └────────┬────────┘
                       │
                       │ diff (512-bit)
                       ▼
              ┌─────────────────┐
              │   POPCOUNT      │
              │                 │
              │  Adder tree:    │
              │  512 → 256 →    │
              │  128 → 64 →     │
              │  32 → 16 →      │
              │  8 → 4 → 2 → 1  │
              │                 │
              │  9 stages       │
              └────────┬────────┘
                       │
                       │ hamming (10-bit, 0-512)
                       ▼
              ┌─────────────────┐
              │   COMPARATOR    │
              │                 │
              │  hamming < θ ?  │
              │                 │
              │  Output: 1-bit  │
              └────────┬────────┘
                       │
                       ▼
                      ZIT
```

### 4.3 Interpretation

| Hamming Distance | Meaning | Result |
|------------------|---------|--------|
| 0 | Identical to resonance | Strong Zit |
| 1-64 | Very similar | Zit (typical θ) |
| 65-128 | Somewhat similar | Maybe Zit |
| 129-256 | Random/orthogonal | No Zit |
| 257-512 | Opposite | No Zit |

### 4.4 Threshold Selection

The threshold θ controls sensitivity:

| θ Value | Behavior | Use Case |
|---------|----------|----------|
| 32 | Very selective | High-precision matching |
| 64 | Selective | Typical recognition |
| 128 | Moderate | Fuzzy matching |
| 256 | Permissive | Catch-all |

---

## 5. The Resonance State

### 5.1 Definition

The resonance state S is a 512-bit register that evolves via XOR:

```
S(0) = 0                    # Initial state
S(t) = S(t-1) ⊕ input(t)    # Update rule
```

### 5.2 Properties

**Reversibility**: XOR is its own inverse. `A ⊕ B ⊕ B = A`.

**Accumulation**: S encodes the XOR-sum of all inputs:
```
S(n) = input(0) ⊕ input(1) ⊕ ... ⊕ input(n)
```

**Pattern Reinforcement**: If the same input appears twice, it cancels:
```
S ⊕ A ⊕ A = S  # A appears and disappears
```

**Holographic**: Every bit of S is influenced by every input. Information is distributed.

### 5.3 As Memory

Traditional memory:
```
Store: MEM[addr] = data
Read:  data = MEM[addr]
```

Resonance memory:
```
Store: S = S ⊕ data
Read:  resonance = popcount(S ⊕ query)
       (low resonance = data was stored)
```

### 5.4 As Routing

The resonance state implicitly routes to shapes:

- Inputs similar to S → low hamming → one shape
- Inputs different from S → high hamming → another shape
- Very different inputs → no shape (below threshold)

No explicit routing table. The resonance IS the routing.

---

## 6. Shape Fabric

### 6.1 Overview

The shape fabric contains 30 frozen shapes, each implemented as a dedicated circuit.

### 6.2 Shape Table

| Opcode | Name | Kingdom | Implementation |
|--------|------|---------|----------------|
| 0x00 | XOR | Logic | `a + b - 2ab` (gates) |
| 0x01 | AND | Logic | `a × b` (gates) |
| 0x02 | OR | Logic | `a + b - ab` (gates) |
| 0x03 | NOT | Logic | `1 - a` (gates) |
| 0x04 | NAND | Logic | `1 - ab` (gates) |
| 0x05 | NOR | Logic | `1 - (a + b - ab)` (gates) |
| 0x06 | XNOR | Logic | `1 - (a + b - 2ab)` (gates) |
| 0x20 | ADD | Arithmetic | Adder |
| 0x21 | SUB | Arithmetic | Subtractor |
| 0x22 | MUL | Arithmetic | Multiplier |
| 0x23 | NEG | Arithmetic | Negation |
| 0x24 | POPCOUNT | Arithmetic | Adder tree |
| 0x40 | RELU | Activation | Comparator + mux |
| 0x41 | SIGMOID | Activation | 256-entry LUT |
| 0x42 | TANH | Activation | 256-entry LUT |
| 0x43 | GELU | Activation | 256-entry LUT |
| 0x44 | SWISH | Activation | LUT + multiplier |
| 0x45 | SOFTMAX | Activation | Exp LUT + divider |
| 0x46 | LEAKY_RELU | Activation | Comparator + mux + shifter |
| 0x60 | LAYER_NORM | Normalization | Stat unit + divider |
| 0x61 | RMS_NORM | Normalization | Stat unit + divider |
| 0x80 | MAX_POOL | Pooling | Comparator tree |
| 0x81 | AVG_POOL | Pooling | Adder + divider |
| 0x82 | SUM_POOL | Pooling | Adder tree |
| 0x83 | MIN_POOL | Pooling | Comparator tree |
| 0x84 | ARGMIN | Pooling | Comparator tree + index |
| 0x85 | ARGMAX | Pooling | Comparator tree + index |
| 0xE0 | HALF_ADDER | Compound | XOR + AND |
| 0xE1 | FULL_ADDER | Compound | XOR×2 + AND×2 + OR |
| 0xE2 | HAMMING | Compound | XOR + POPCOUNT |

### 6.3 Shape Selection

The shape decoder maps hamming distance to shape index:

```
if      hamming < 32:   shape = config.band_0_shape
else if hamming < 64:   shape = config.band_1_shape
else if hamming < 96:   shape = config.band_2_shape
else if hamming < 128:  shape = config.band_3_shape
else if hamming < θ:    shape = config.default_shape
else:                   shape = NONE (no activation)
```

The band → shape mapping is configurable via OTP or registers.

---

## 7. Specifications

### 7.1 Core Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| Data Width | 512 bits | Native word size |
| Precision | 8-bit fixed | 64 × 8-bit elements |
| Shapes | 30 | Frozen in silicon |
| Resonance State | 512 bits | XOR-accumulating |
| Threshold | 10 bits | 0-512 range |
| Latency | 1-2 cycles | Combinational path |
| Throughput | 1 output/cycle | Fully pipelined option |

### 7.2 Resource Estimates

| Component | Gates | Notes |
|-----------|-------|-------|
| Resonance Register | 512 | Flip-flops |
| Threshold Register | 10 | Flip-flops |
| XOR Array (512-bit) | 512 | XOR gates |
| Popcount Tree | 900 | Adder tree |
| Comparator | 50 | 10-bit compare |
| Shape Decoder | 100 | Priority encoder |
| Shape Fabric | 50,000 | All 30 shapes |
| Output Mux | 500 | 30:1 × 512-bit |
| Control | 200 | FSM, enables |
| **Total** | **~53,000** | |

### 7.3 Performance Projections

| Metric | Estimate | Notes |
|--------|----------|-------|
| Clock Frequency | 500 MHz - 1 GHz | Short paths |
| Throughput | 256-512 Gbits/sec | Per core |
| Latency | 2-4 ns | Combinational |
| Power | <100 mW | Estimate, TBD |

### 7.4 Comparison

| Metric | NGP v1 | NGP v2 | Improvement |
|--------|--------|--------|-------------|
| Routing | 4096 comparators | 1 resonance reg | 4096× |
| Gates | 2.7M | 53K | 50× |
| Memory | 264 KB | 66 bytes | 4000× |
| Complexity | High | Low | Dramatic |

---

## 8. Comparison with Traditional Architectures

### 8.1 vs CPU

| Aspect | CPU | NGP |
|--------|-----|-----|
| Execution Model | Sequential instructions | Parallel resonance |
| Memory | Fetch from RAM | XOR accumulation |
| Branching | Conditional jumps | Threshold gating |
| State | Registers + RAM | Resonance state |
| Flexibility | General-purpose | Fixed-function |
| Determinism | Variable latency | Fixed latency |

### 8.2 vs GPU

| Aspect | GPU | NGP |
|--------|-----|-----|
| Parallelism | SIMT threads | Shape parallelism |
| Memory | High-bandwidth DRAM | No memory |
| Programming | CUDA/shaders | None (hardware) |
| Power | High (100s of watts) | Low (milliwatts) |
| Use Case | Graphics, ML training | ML inference, control |

### 8.3 vs FPGA

| Aspect | FPGA | NGP |
|--------|------|-----|
| Configuration | Bitstream (runtime) | Fixed (fabrication) |
| Routing | Configurable | None (resonance) |
| Logic | LUTs | Dedicated gates |
| Speed | Moderate | High |
| Use Case | Prototyping | Production |

### 8.4 vs Neural Accelerator (TPU, etc.)

| Aspect | Neural Accelerator | NGP |
|--------|-------------------|-----|
| Parameters | Millions of weights | Zero |
| Precision | INT8, FP16, etc. | Fixed-point 8 |
| Training | Required | None |
| Model Updates | Weight reload | OTP burn (one-time) |
| Accuracy | Approximate | Exact |

---

## 9. Implementation Roadmap

### Phase 1: RTL Design (Weeks 1-4)

**Deliverables:**
- Verilog for Zit detector
- Verilog for all 30 shapes
- Verilog for top-level NGP
- Testbenches with coverage

**Files:**
```
ngp/rtl/
├── zit_detector.sv
├── resonance_reg.sv
├── shape_decoder.sv
├── shapes/
│   ├── xor_512.sv
│   ├── popcount_512.sv
│   └── ...
├── ngp_core.sv
└── ngp_top.sv
```

### Phase 2: FPGA Prototype (Weeks 4-8)

**Target:** Xilinx Artix-7 or Spartan-7

**Goals:**
- Functional verification
- Throughput measurement
- Latency characterization
- Power estimation

### Phase 3: ASIC Preparation (Weeks 8-16)

**Activities:**
- Synthesis for target process (28nm or 22nm)
- Timing closure
- Power optimization
- DFT insertion

### Phase 4: Tape-out (Week 16+)

**Options:**
- Efabless open MPW shuttle (free)
- Commercial foundry ($$)

---

## 10. Theory of Operation

### 10.1 The Cymatic Analogy

The NGP operates like a digital Chladni plate:

| Cymatics | NGP |
|----------|-----|
| Vibrating plate | Resonance state S |
| Sound frequency | Input signature |
| Standing wave | Low hamming distance |
| Sand pattern | Shape activation |
| Damping | Threshold θ |

When the input frequency matches the system's eigenmode, a pattern emerges.

### 10.2 The XOR Field

The resonance state S is a field in 512-dimensional binary space. Inputs perturb the field via XOR. The Zit detector measures how much the field is disturbed.

- Small disturbance (low hamming) → resonance → Zit fires
- Large disturbance (high hamming) → dissonance → silence

### 10.3 Learning via Flow

The NGP learns by accumulating inputs:

```python
# Training
for input in training_data:
    S = S ^ input

# Inference
zit = popcount(S ^ query) < threshold
```

No gradients. No backprop. Just XOR.

Patterns that repeat reinforce (appear multiple times → cancel to zero → return to baseline). Unique patterns accumulate.

### 10.4 The Conservation of Entropy

The NGP doesn't destroy information. It redistributes entropy into load-bearing structure.

- Traditional memory: Entropy is trapped in cells (expensive)
- XOR memory: Entropy is distributed in resonance (free)

The resonance state carries memory, routing, and pattern information — all in 512 bits.

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Frozen Shape** | A mathematical operation with no learnable parameters |
| **Resonance State** | The 512-bit register that accumulates input via XOR |
| **Zit** | The activation signal; fires when input resonates |
| **Hamming Distance** | Number of differing bits between two values |
| **Threshold (θ)** | The hamming distance below which Zit fires |
| **Shape Fabric** | The collection of 30 frozen shape circuits |
| **Eigenmode** | A natural resonance pattern of the system |

## Appendix B: References

- [Geocadesia Shape Library](./geocadesia/README.md)
- [Binary Format Specification](./BINARY_FORMAT.md)
- [FrozenDB Vector Search](./FROZENDB.md)
- [Zit Detector Derivation](../../../notes/zit/zit_detector_synth.md)
- [X288 Research](../../../notes/X288.md)

---

*"The NGP doesn't compute shapes. It IS shapes."*

*"Entropy isn't waste. It's load-bearing structure."*

*"It's all in the reflexes."*
