# Shape-Native Silicon: NODES

*Key crystallizations from the RAW stream.*

---

## Node 1: Shape AS Configuration

The .fsh file isn't an instruction to be decoded. It's a CONFIGURATION that defines what circuits exist. The shape IS the silicon topology.

**Implication**: No instruction decoder. No microcode. Shapes are physical.

---

## Node 2: Frozen Means Fixed at Fabrication

"Frozen" isn't a software concept. It's a hardware concept. The shapes are literally baked into silicon. They don't change. They CAN'T change.

**Implication**: Maximum determinism. No state corruption possible for shapes themselves.

---

## Node 3: Routing is the Only Variable

Shapes are frozen. Data flows through them. The only "decision" is WHICH shape to activate based on input signature.

**Architecture Split**:
- FIXED: Shape circuits (30 shapes, hardwired)
- CONFIGURABLE: Routing table (4096 entries, OTP or SRAM)

---

## Node 4: Parallel Hamming Routing

Routing isn't sequential lookup. It's parallel comparison.

```
Input Signature (512-bit)
         │
         ├──────────┬──────────┬─────── ... ───┬──────────┐
         ▼          ▼          ▼               ▼          ▼
    ┌────────┐ ┌────────┐ ┌────────┐     ┌────────┐ ┌────────┐
    │Hamming │ │Hamming │ │Hamming │ ... │Hamming │ │Hamming │
    │ Route 0│ │ Route 1│ │ Route 2│     │Route N │ │Route N │
    └───┬────┘ └───┬────┘ └───┬────┘     └───┬────┘ └───┬────┘
        │          │          │               │          │
        └──────────┴──────────┴───────────────┴──────────┘
                              │
                        ┌─────┴─────┐
                        │  ARGMIN   │
                        │   Tree    │
                        └─────┬─────┘
                              │
                         Winner Index
```

**Implication**: O(1) routing. All comparisons simultaneous.

---

## Node 5: Fused Compound Shapes

Compound shapes aren't executed as sequences. They're fused circuits.

- Hamming = XOR + Popcount → Single circuit, single cycle
- FullAdder = XOR + XOR + AND + AND + OR → Single circuit, single cycle

**Implication**: The "components" list in .fsh is provenance, not execution order.

---

## Node 6: Homogeneous Routing Fabric

Every routing comparator is identical:
- Store one signature (512 bits)
- Compute Hamming(input, stored)
- Output distance (10 bits, 0-512)

4096 identical units. Regular structure. Easy to layout, verify, manufacture.

---

## Node 7: Activation Functions as LUTs

Complex functions (sigmoid, tanh, GELU) don't need exact computation. For control applications, determinism > precision.

256-entry LUT, 8-bit in, 8-bit out. Burned into ROM at fabrication.

---

## Node 8: Gate Count is Tiny

Rough estimate:
- Routing fabric (4096 comparators): ~2.5M gates
- Argmin tree: ~50K gates
- Shape circuits: ~100K gates
- **Total: ~3M gates**

Modern chips have billions. This fits in a corner.

---

## Node 9: Throughput Math

- 512 bits per cycle per core
- At 1 GHz: 512 Gbits/sec/core
- At 64 cores: 32.8 Tbits/sec

That's Thor. Not aspirational. Architectural.

---

## Node 10: No Host Processor Required

This isn't an accelerator attached to a CPU. It's standalone.

Data in → Shapes route → Result out.

No instruction fetch. No cache. No branches. No speculation.

Just a function. f(x) → y.

---

## Node 11: OTP Routing for Deployment Customization

**Manufacturing**: Chip with all 30 shapes, empty routing table.
**Deployment**: Burn routing table via OTP for specific application.

Shapes frozen at fab. Routes frozen at deployment. Data flows at runtime.

---

## Node 12: The Product Concept

```
┌─────────────────────────────────────────────────────────────────────┐
│                     GEOCADESIA NGP-1                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  FIXED (at fabrication):                                           │
│  • 30 frozen shape circuits                                        │
│  • Fused compound circuits (Hamming, FullAdder, ...)               │
│  • Argmin reduction tree                                           │
│  • 512-bit I/O fabric                                              │
│                                                                     │
│  CONFIGURABLE (at deployment, via OTP):                            │
│  • 4096-entry routing table                                        │
│  • Signature → Shape mapping                                       │
│                                                                     │
│  PERFORMANCE:                                                       │
│  • ~3M gates                                                       │
│  • 512 bits/cycle throughput                                       │
│  • 4-stage pipeline (Compare → Reduce → Route → Execute)           │
│  • Target: 1-2 GHz                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Meta-Node: What This Is

Not a CPU. Not a GPU. Not an FPGA.

A **Neural Geometric Processor (NGP)**.

A fixed mathematical function in silicon. No instruction set. No compiler. Just shapes.

---

*12 nodes extracted. Ready for REFLECT.*
