# ZIT-1 Paper Materials

## Document Index

| Document | Purpose |
|----------|---------|
| `ZIT1_HOMEO_ADAPTIVE_FABRIC.md` | Main paper draft |
| `EXPERIMENTAL_DATA.md` | Raw logs and statistical analysis |
| `experiments/` | CUDA source code for all experiments |

## Quick Reference

### The Core Claim

**Learned topology with fixed operations** is a viable alternative to **fixed topology with learned weights**.

### The Key Result

| Metric | Value |
|--------|-------|
| Maximum nodes tested | 56,623,104 |
| Convergence cycles | 570 |
| Scaling efficiency | 884,000x nodes → 3.6x cycles |
| Throughput | 220M node-cycles/sec |
| Speedup vs Verilog | >2,500,000x |

### The Theoretical Framework

**Homeo-Adaptive Model**: Computation as Homeostasis

- The fabric doesn't calculate; it **heals**
- Frustration is not an error; it's **pain**
- Convergence is not a solution; it's **equilibrium**
- The topology IS the learned model

### The Discovery Timeline

1. **Pass 1-2**: Discovered geometric frustration in fixed topology
2. **Pass 3**: Implemented topological plasticity in Verilog
3. **Pass 4**: Ported to CUDA, achieved million-fold speedup
4. **Pass 5**: Scaled to 56M nodes, discovered non-linear sweet spots
5. **Pass 6**: Formulated Homeo-Adaptive theoretical framework

### Files for Publication

```
papers/
├── README.md                          # This file
├── ZIT1_HOMEO_ADAPTIVE_FABRIC.md      # Main paper
├── EXPERIMENTAL_DATA.md               # Raw data appendix
└── experiments/
    ├── fabric_128.cu                  # 2M nodes
    ├── fabric_256.cu                  # 16.7M nodes
    ├── fabric_384.cu                  # 56.6M nodes (flagship)
    ├── sequential_fabric.cu           # 64 nodes (reference)
    ├── sequential_fabric_8x8x8.cu     # 512 nodes
    ├── sequential_fabric_16x16x16.cu  # 4K nodes
    ├── sequential_fabric_32x32x32.cu  # 32K nodes
    ├── turbo_fabric.cu                # Fused kernel version
    ├── turbo_fabric_warmup.cu         # With warmup sequence
    └── turbo_fabric_scaled.cu         # Parameterized version
```

### Verilog Reference (in main repo)

```
trixc/forge/rtl/
├── zit_plastic_node.v                 # Node with plasticity
├── zit_plastic_fabric.v               # Complete fabric
├── zit_plastic_tb.v                   # Testbench
└── FUNKY_CONVERGENCE.md               # Original results doc
```

## Publication Venues (To Consider)

### Conferences

- **NeurIPS** - Novel architectures track
- **ICLR** - Self-organizing systems
- **DAC** - Hardware implementation
- **MICRO** - Microarchitecture innovation
- **ISCA** - Computer architecture

### Journals

- **Nature Machine Intelligence** - Novel computing paradigms
- **IEEE TNNLS** - Neural networks
- **JMLR** - Machine learning theory

## TODO Before Submission

- [x] Complete related work section with citations (`RELATED_WORK.md`)
- [x] Add formal notation for plasticity rules (`FORMAL_NOTATION.md`)
- [x] Create rigorous test suite (`experiments/run_tests.sh`)
- [x] Add Verilog RTL excerpts (Appendix B)
- [x] Document experimental data (Appendix C, `EXPERIMENTAL_DATA.md`)
- [ ] Create figures (topology evolution visualization)
- [ ] Prove convergence bounds (or conjecture them)
- [ ] Get external review
- [ ] Choose target venue
- [ ] Format according to venue requirements

## Document Index (Updated)

| Document | Lines | Purpose |
|----------|-------|---------|
| `ZIT1_HOMEO_ADAPTIVE_FABRIC.md` | ~490 | Main paper draft |
| `RELATED_WORK.md` | ~370 | Theoretical foundations & citations |
| `FORMAL_NOTATION.md` | ~180 | Mathematical formalization |
| `EXPERIMENTAL_DATA.md` | ~330 | Raw logs & statistics |
| `experiments/README.md` | ~140 | Test suite documentation |

## Contact

*[To be added]*

---

*Second Star Constant: 1122911624*
*December 2024*
