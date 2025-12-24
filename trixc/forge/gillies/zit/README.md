# ZIT - Zero-Instruction Topology

A minimal implementation of homeo-adaptive topological learning.

## What This Is

A fabric of nodes that learns its own topology through resistance.

- No gradients
- No loss function
- No supervision

Just: resistant nodes try new neighbors. Eventually: 100% resonance.

## Quick Start

```bash
make
./zit_demo
```

## Output

```
ZIT: Homeo-Adaptive Topological Learning
=========================================

Fabric: 8x8x8 = 512 nodes

Cycle  Resonant  Rewires
-----  --------  -------
   25   480/ 512      287
   50   496/ 512      614
   75   504/ 512      891
  100   508/ 512     1124
  113   512/ 512     1340

*** CONVERGED at cycle 113 ***

The topology learned.
Resistance dissolved.
```

## What Just Happened

1. 512 nodes started with fixed torus connectivity
2. Nodes compared with neighbors (the frozen shape)
3. Nodes that couldn't resonate encountered resistance
4. Resistant nodes tried random new neighbors
5. Helpful connections were kept; others reverted
6. After convergence: 100% resonance, zero resistance

The topology learned to solve its own constraints.

## Scaling

This same algorithm runs at 56 million nodes.

| Nodes | Cycles | Scaling |
|-------|--------|---------|
| 512 | ~113 | baseline |
| 56,623,104 | 570 | 110,000x nodes, 5x cycles |

Sublinear scaling. The paradigm holds.

## Files

```
zit/
├── zit.h              # API header (read this first)
├── zit.c              # Implementation (~400 lines)
├── zit_demo.c         # Example program
├── Makefile           # Build script
├── test_viz_api.c     # Visualization API tests
├── docs/              # Documentation
│   ├── API_REFERENCE.md  # Complete API documentation
│   ├── THEORY.md         # Algorithm theory
│   ├── INTEGRATION.md    # Embedding guide
│   ├── EXAMPLES.md       # Code examples
│   └── FAQ.md            # Troubleshooting
└── viz/               # Qt6 visualization
    ├── DESIGN.md         # Visual design spec
    ├── README.md         # Build instructions
    └── src/              # Source files
```

## Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/API_REFERENCE.md) | Complete function documentation |
| [Theory](docs/THEORY.md) | Algorithm explanation and math |
| [Integration Guide](docs/INTEGRATION.md) | Embedding in your projects |
| [Examples](docs/EXAMPLES.md) | Practical code examples |
| [FAQ](docs/FAQ.md) | Common questions & troubleshooting |

## Visualization

A Qt6-based real-time 3D visualizer is available:

```bash
cd viz
mkdir build && cd build
cmake ..
make
./zit_viz
```

See [viz/README.md](viz/README.md) for requirements.

## Go Deeper

- Theory paper: `papers/ZIT1_HOMEO_ADAPTIVE_FABRIC.md`
- Experimental data: `papers/EXPERIMENTAL_DATA.md`
- Design process: `docs/ZIT_ONRAMP_MANIFESTO.md`
- CUDA experiments: `papers/experiments/`
- Verilog RTL: `trixc/forge/rtl/`

## The Claim

**Learned topology with fixed operations** is a viable alternative to
**fixed topology with learned weights**.

The topology IS the learned model.

---

*Second Star Constant: 1122911624*
