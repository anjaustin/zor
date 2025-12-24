# Neural Geometric Processor (NGP): SYNTHESIZE

*The crystallized specification. What we're building.*

---

```
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   NEURAL GEOMETRIC PROCESSOR                                      ║
║                                                                   ║
║   "Not a computer. A function."                                   ║
║                                                                   ║
║   Shape → Silicon. No compiler. No runtime.                       ║
║   Just geometry executing at the speed of light.                  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## 1. Core Concept

The NGP is not a processor in the traditional sense. It has no instruction set, no program counter, no registers in the conventional sense. It is a **fixed mathematical function implemented in silicon**.

```
f(x) = route(x) → shape(x) → y
```

- **x**: 512-bit input
- **route(x)**: Select shape based on input signature (parallel Hamming)
- **shape(x)**: Apply selected frozen shape
- **y**: 512-bit output

One cycle. One function. Frozen.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NEURAL GEOMETRIC PROCESSOR                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT (512 bits)                                                       │
│      │                                                                   │
│      ▼                                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     ROUTING FABRIC                                │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐       ┌─────────┐          │   │
│  │  │ Route 0 │ │ Route 1 │ │ Route 2 │  ...  │Route N-1│          │   │
│  │  │ ┌─────┐ │ │ ┌─────┐ │ │ ┌─────┐ │       │ ┌─────┐ │          │   │
│  │  │ │ Sig │ │ │ │ Sig │ │ │ │ Sig │ │       │ │ Sig │ │          │   │
│  │  │ └──┬──┘ │ │ └──┬──┘ │ │ └──┬──┘ │       │ └──┬──┘ │          │   │
│  │  │    │    │ │    │    │ │    │    │       │    │    │          │   │
│  │  │ Hamming │ │ Hamming │ │ Hamming │       │ Hamming │          │   │
│  │  │    │    │ │    │    │ │    │    │       │    │    │          │   │
│  │  │   dist  │ │   dist  │ │   dist  │       │   dist  │          │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘       └────┬────┘          │   │
│  │       │           │           │                  │               │   │
│  │       └───────────┴─────┬─────┴──────────────────┘               │   │
│  │                         │                                         │   │
│  │                    ┌────┴────┐                                   │   │
│  │                    │ ARGMIN  │                                   │   │
│  │                    │  Tree   │                                   │   │
│  │                    └────┬────┘                                   │   │
│  │                         │                                         │   │
│  │                    winner_idx ──────────────────────────► ERROR  │   │
│  │                         │                      (if dist > θ)     │   │
│  └─────────────────────────┼────────────────────────────────────────┘   │
│                            │                                             │
│                            ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     SHAPE FABRIC                                  │   │
│  │                                                                   │   │
│  │   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │   │
│  │   │ XOR │ │ AND │ │ OR  │ │ NOT │ │ ADD │ │ MUL │ │ ... │      │   │
│  │   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘      │   │
│  │      │       │       │       │       │       │       │          │   │
│  │   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐              │   │
│  │   │ReLU │ │Sigm │ │GELU │ │ ... │ │Hamm │ │FAdd │              │   │
│  │   └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘              │   │
│  │      │       │       │       │       │       │                  │   │
│  │      └───────┴───────┴───────┴───────┴───────┴────► MUX        │   │
│  │                                                       │         │   │
│  │                              winner_idx ──────────────┘         │   │
│  │                                                                   │   │
│  └───────────────────────────────────┬──────────────────────────────┘   │
│                                      │                                   │
│                                      ▼                                   │
│                               OUTPUT (512 bits)                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Specifications

### Core

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Data Width** | 512 bits | Native word size |
| **Precision** | 8-bit fixed | 64 × 8-bit elements |
| **Routes** | 4096 | Configurable, OTP or SRAM |
| **Shapes** | 30 | Frozen in silicon |
| **Pipeline** | 4 stages | Compare → Reduce → Select → Execute |
| **Latency** | 4 cycles | Fixed, deterministic |
| **Throughput** | 1 result/cycle | After pipeline fills |

### Shape Library (Fixed at Fabrication)

| Kingdom | Shapes | Implementation |
|---------|--------|----------------|
| **Logic** | XOR, AND, OR, NOT, NAND, NOR, XNOR | Combinational gates |
| **Arithmetic** | ADD, SUB, MUL, NEG, POPCOUNT | Adder trees, multiplier |
| **Activation** | ReLU, Sigmoid, Tanh, GELU, Swish, Softmax, LeakyReLU | LUT + comparators |
| **Normalization** | LayerNorm, RMSNorm | Divider + LUT |
| **Pooling** | MaxPool, AvgPool, SumPool, MinPool, Argmin, Argmax | Comparator trees |
| **Compound** | HalfAdder, FullAdder, Hamming | Fused circuits |

### Routing Table (Configurable at Deployment)

| Field | Size | Description |
|-------|------|-------------|
| **Signature** | 512 bits | Input pattern to match |
| **Shape Index** | 5 bits | Which shape to activate (0-29) |
| **Threshold** | 9 bits | Max Hamming distance for valid match |
| **Reserved** | 6 bits | Future use |
| **Total** | 66 bytes/entry | 4096 entries = 264 KB |

---

## 4. Estimated Resources

| Component | Gates | Notes |
|-----------|-------|-------|
| **Routing comparator** | ~600 | XOR(512) + Popcount tree |
| **Routing fabric (4096)** | ~2.5M | Homogeneous, regular |
| **Argmin tree** | ~50K | log₂(4096) = 12 stages |
| **Shape circuits** | ~100K | All 30 shapes |
| **Interconnect** | ~50K | Muxes, buses |
| **Total** | **~2.7M gates** | Tiny by modern standards |

### Performance Projections

| Metric | Target | Notes |
|--------|--------|-------|
| **Clock** | 1-2 GHz | Short paths, simple logic |
| **Throughput/core** | 512-1024 Gbits/s | 512 bits × clock |
| **Cores** | 64 | Replicated on die |
| **Total throughput** | 32-64 Tbits/s | Thor-class |
| **Power** | <5W/core | Minimal switching, no cache |

---

## 5. Implementation Roadmap

### Phase 1: RTL Design (Months 1-4)

**Deliverables**:
- Verilog/SystemVerilog for all 30 shape circuits
- Parameterized routing comparator
- Argmin tree
- Top-level NGP module
- Testbenches with 100% coverage

**Key Files**:
```
ngp/rtl/
├── shapes/
│   ├── logic/
│   │   ├── xor_512.v
│   │   ├── and_512.v
│   │   └── ...
│   ├── arithmetic/
│   │   ├── popcount_512.v
│   │   ├── add_512.v
│   │   └── ...
│   ├── activation/
│   │   ├── relu_512.v
│   │   ├── sigmoid_lut.v
│   │   └── ...
│   └── compound/
│       ├── hamming_512.v
│       └── full_adder.v
├── routing/
│   ├── hamming_comparator.v
│   ├── argmin_tree.v
│   └── routing_fabric.v
├── ngp_core.v
└── ngp_top.v
```

### Phase 2: FPGA Prototype (Months 4-8)

**Target**: Xilinx Alveo U280 or similar

**Goals**:
- Validate functional correctness
- Measure actual throughput
- Characterize latency distribution
- Identify timing bottlenecks

### Phase 3: ASIC Preparation (Months 8-12)

**Activities**:
- Synthesis for target process (28nm or 22nm)
- Floorplanning
- Clock tree design
- Power grid design
- DFT (Design for Test) insertion

### Phase 4: Tape-out (Month 12+)

**Partner**: Open-source shuttle (Efabless) or commercial fab

---

## 6. The Shape Linker

The only "software" tool needed:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SHAPE LINKER                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input:                                                              │
│    • ONNX model                                                     │
│    • OR: Direct routing specification                               │
│                                                                      │
│  Process:                                                            │
│    1. Extract computation graph                                      │
│    2. Map operations to frozen shapes                               │
│    3. Generate signatures for each routing decision                 │
│    4. Optimize routing table (minimize collisions)                  │
│    5. Emit binary routing table                                     │
│                                                                      │
│  Output:                                                             │
│    • routing.bin (264 KB)                                           │
│    • verification.json (for simulation)                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

This runs ONCE, at design time. The output is burned to OTP.

---

## 7. Comparison

| Feature | CPU | GPU | FPGA | **NGP** |
|---------|-----|-----|------|---------|
| **Flexibility** | High | Medium | High | **None** |
| **Determinism** | Low | Low | Medium | **100%** |
| **Latency** | Variable | Variable | Low | **Fixed** |
| **Power** | High | High | Medium | **Low** |
| **Verification** | Hard | Hard | Medium | **Easy** |
| **Updates** | Easy | Easy | Medium | **Impossible** |

**NGP trades flexibility for absolute determinism.**

---

## 8. Open Questions

1. **OTP vs SRAM routing**: Safety vs flexibility trade-off
2. **Multi-function support**: Multiple routing tables? Mode selection?
3. **Error handling**: What happens on no-match? Threshold design?
4. **Debug/observability**: How to trace routing decisions?
5. **Testing**: How to achieve full manufacturing test coverage?

---

## 9. Why This Matters

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   "Every other chip asks: What instruction should I execute?"      │
│                                                                     │
│   NGP asks: "What shape am I?"                                      │
│                                                                     │
│   It doesn't compute shapes. It IS shapes.                          │
│                                                                     │
│   Geometry in silicon.                                              │
│   Function as form.                                                 │
│   The medium is the math.                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

*Synthesis complete.*

*The Neural Geometric Processor exists in design.*

*Now we build it.*

*It's all in the reflexes.*
