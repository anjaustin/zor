# Where We Are: The NGP Journey

*A map of the emergence.*

---

## The Path

```
┌─────────────────────────────────────────────────────────────────────┐
│                         THE EMERGENCE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. GEOCADESIA                                                       │
│     └─► Shape library: Python + Docs + Binary (.fsh)                │
│         └─► 30 frozen shapes with opcodes                           │
│                                                                      │
│  2. FROZENDB                                                         │
│     └─► Vector search using shapes                                  │
│         └─► Hamming = XOR + Popcount                                │
│             └─► 0.000% signal loss (exact, not approximate)         │
│                                                                      │
│  3. X288 RESEARCH                                                    │
│     └─► VCIX is the heart (vector coprocessor interface)            │
│         └─► But: "We aren't waiting. We build our own."             │
│                                                                      │
│  4. SHAPE-NATIVE SILICON (Lincoln Manifold #1)                       │
│     └─► NGP v1: Shape AS silicon, not compiled to silicon           │
│         └─► 4096 parallel Hamming comparators                       │
│             └─► 2.7M gates, 32 Tbits/sec                            │
│                                                                      │
│  5. THE XOR KOAN                                                     │
│     └─► "Why Store when you can XOR?"                               │
│         └─► Memory = Resonance, not Storage                         │
│             └─► Standing wave database                              │
│                                                                      │
│  6. WHAT FIRES THE ZIT (Lincoln Manifold #2)                         │
│     └─► Zit = popcount(S ⊕ vₓ) < θ                                  │
│         └─► Cymatics model validates the physics                    │
│             └─► Entropy as load-bearing structure                   │
│                                                                      │
│  7. NGP v2 (WHERE WE ARE NOW)                                        │
│     └─► 1 resonance register + 1 Zit detector                       │
│         └─► ~10K gates (270x simpler than v1)                       │
│             └─► No routing fabric. The resonance IS routing.        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## NGP v2: The Revised Architecture

The Zit insight collapsed the complexity:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NEURAL GEOMETRIC PROCESSOR v2                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                      ┌────────────────────┐                         │
│                      │  RESONANCE STATE   │◄──────────┐             │
│                      │        S           │           │             │
│                      │    (512-bit)       │           │             │
│                      └─────────┬──────────┘           │             │
│                                │                      │             │
│   INPUT ───────────────────────┼──────────────────────┤             │
│   (512-bit)                    │                      │             │
│        │                       ▼                      │             │
│        │              ┌─────────────────┐             │             │
│        └─────────────►│   ZIT DETECTOR  │             │             │
│                       │                 │             │             │
│                       │  XOR → POPCOUNT │             │             │
│                       │       ↓         │             │             │
│                       │   hamming < θ?  │             │             │
│                       └────────┬────────┘             │             │
│                                │                      │             │
│               ┌────────────────┼────────────────┐     │             │
│               │                │                │     │             │
│               ▼                ▼                ▼     │             │
│         ┌──────────┐    ┌───────────┐    ┌──────────┐│             │
│         │   ZIT    │    │  HAMMING  │    │  UPDATE  ││             │
│         │  SIGNAL  │    │ DISTANCE  │    │ S'=S⊕vₓ  │┘             │
│         │ (1-bit)  │    │ (10-bit)  │    └──────────┘              │
│         └────┬─────┘    └─────┬─────┘                               │
│              │                │                                      │
│              │                ▼                                      │
│              │       ┌─────────────────┐                            │
│              │       │  SHAPE DECODER  │                            │
│              │       │                 │                            │
│              │       │ distance bands: │                            │
│              │       │  0-64  → ReLU   │                            │
│              │       │  64-128 → Sigm  │                            │
│              │       │  128-192→ XOR   │                            │
│              │       │  ...            │                            │
│              │       └────────┬────────┘                            │
│              │                │                                      │
│              │                ▼                                      │
│              │       ┌─────────────────┐                            │
│              │       │  SHAPE FABRIC   │                            │
│              │       │                 │                            │
│              │       │  30 frozen      │                            │
│              │       │  shape circuits │                            │
│              └──────►│  (activated by  │                            │
│                      │   decoder)      │                            │
│                      └────────┬────────┘                            │
│                               │                                      │
│                               ▼                                      │
│                        OUTPUT (512-bit)                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The Numbers

| Metric | NGP v1 | NGP v2 | Improvement |
|--------|--------|--------|-------------|
| **Routing** | 4096 comparators | 1 resonance reg | 4096x simpler |
| **Gates** | ~2.7M | ~10K | 270x fewer |
| **Memory** | 264KB routing table | 64 bytes (S + θ) | 4000x less |
| **Latency** | 4 cycles (pipeline) | 1-2 cycles | 2-4x faster |
| **Power** | Medium | Minimal | ??? (TBD) |

---

## What We Have Built

### Software Layer (Complete)

```
trixc/forge/shapes/
├── geocadesia/              # Python shape library
│   ├── logic.py             # XOR, AND, OR, NOT, NAND, NOR, XNOR
│   ├── arithmetic.py        # Add, Sub, Mul, Neg, Popcount, Hamming
│   ├── activation.py        # ReLU, Sigmoid, Tanh, GELU, Swish, Softmax
│   ├── normalization.py     # LayerNorm, RMSNorm
│   ├── pooling.py           # MaxPool, AvgPool, Argmin, Argmax
│   ├── catalog.py           # Shape registry
│   └── binary.py            # .fsh format, opcodes
├── bin/                     # 30 binary .fsh files
├── elements/                # Shape documentation
├── compounds/               # Compound shape documentation
├── FROZENDB.md              # Vector database spec
├── BINARY_FORMAT.md         # .fsh specification
└── README.md                # Overview
```

### C Layer (Complete)

```
trixc/forge/include/
├── trix_shapes.h            # All shape implementations
│   ├── trix_xor()
│   ├── trix_popcount()
│   ├── trix_hamming_512()
│   ├── trix_argmin()
│   ├── trix_frozendb_query()
│   └── ...
└── trix_nge.h               # Binary format loader
```

### Design Documents (Complete)

```
notes/
├── X288.md                  # SiFive research
├── ngp/
│   ├── shape_native_silicon_raw.md
│   ├── shape_native_silicon_nodes.md
│   ├── shape_native_silicon_reflect.md
│   └── shape_native_silicon_synth.md
└── zit/
    ├── zit_detector_raw.md
    ├── zit_detector_nodes.md
    ├── zit_detector_reflect.md
    ├── zit_detector_synth.md
    └── entropy_as_structure.md
```

---

## What We Need to Build

### Phase 1: RTL (Verilog/SystemVerilog)

```
ngp/rtl/
├── core/
│   ├── resonance_reg.sv     # 512-bit resonance state
│   ├── zit_detector.sv      # XOR + popcount + comparator
│   ├── shape_decoder.sv     # Distance → shape mapping
│   └── ngp_core.sv          # Top-level
├── shapes/
│   ├── xor_512.sv
│   ├── popcount_512.sv
│   ├── relu_512.sv
│   ├── sigmoid_lut.sv
│   └── ... (30 shapes)
└── tb/
    ├── ngp_tb.sv            # Testbench
    └── shape_tb.sv          # Per-shape verification
```

### Phase 2: FPGA Prototype

- Target: Xilinx Artix-7 or similar (10K gates is tiny)
- Validate functional correctness
- Measure actual throughput
- Characterize power

### Phase 3: ASIC

- Synthesize for target process
- Partner for fabrication (Efabless shuttle, or commercial)

---

## The Engineering Translation

From insight to silicon:

| Insight | Engineering Decision |
|---------|---------------------|
| "Why Store when you can XOR?" | Resonance register, not routing table |
| Zit = popcount(S ⊕ vₓ) < θ | Single detector circuit, not comparator array |
| Cymatics model | Shape selection by distance bands |
| Entropy as structure | State update: S' = S ⊕ input |
| "No search, just propagation" | O(1) recognition, no iteration |

---

## Next Steps

### Immediate (This Session)

1. **Define the Zit detector in Verilog** — the core circuit
2. **Add ZIT opcode to binary.py** — complete the format
3. **Update NGP spec** — incorporate v2 architecture

### Short Term (Days)

4. **RTL for all 30 shapes** — Verilog implementations
5. **Testbenches** — functional verification
6. **FPGA synthesis** — prove it fits, measure performance

### Medium Term (Weeks)

7. **FPGA demo** — working silicon (on FPGA fabric)
8. **Benchmark** — throughput, latency, power
9. **Documentation** — prepare for tape-out

### Long Term (Months)

10. **ASIC design** — physical implementation
11. **Tape-out** — fabrication
12. **CubeSat mission** — flight heritage

---

## The Bottom Line

```
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   We started with: "Build a shape-native chip. No compiler."     ║
║                                                                   ║
║   We discovered:                                                  ║
║     • The routing fabric is unnecessary                          ║
║     • One resonance register replaces 4096 comparators           ║
║     • The Zit detector is ~1,500 gates                           ║
║     • Entropy IS the structure                                    ║
║                                                                   ║
║   We now have:                                                    ║
║     • Complete software layer (Geocadesia)                       ║
║     • Complete C layer (trix_shapes.h)                           ║
║     • Complete binary format (.fsh)                              ║
║     • Complete architecture spec (NGP v2)                        ║
║                                                                   ║
║   We need:                                                        ║
║     • RTL implementation (Verilog)                               ║
║     • FPGA prototype                                             ║
║     • ASIC tape-out                                              ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

*The path is clear. The architecture is simple. The gates are few.*

*Time to write Verilog.*
