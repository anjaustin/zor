# Getting Started with ZOR

## Choose Your Path

ZOR serves different types of users. Choose the path that matches your goal:

---

## 🔬 Path 1: The Curious (2 minutes)

**"I want to see what this does."**

### Watch Topology Learn

```bash
cd trixc/forge/gillies/zit
make
./zit_demo
```

You'll see:
```
ZIT: Homeo-Adaptive Topological Learning
=========================================

Fabric: 8x8x8 = 512 nodes

Cycle  Resonant  Rewires
-----  --------  -------
   25   234/ 512      623
   50   329/ 512      913
   75   414/ 512     1231
  100   491/ 512     1327
  114   512/ 512     1340

*** CONVERGED at cycle 114 ***

The topology learned.
Resistance dissolved.
```

**What just happened:** 512 nodes in a 3D torus learned their own connectivity. No gradients, no loss function—just resistance and rewiring.

### Next Steps
- Read: `trixc/forge/gillies/zit/README.md`
- Understand: `papers/ZIT1_HOMEO_ADAPTIVE_FABRIC.md`

---

## 🧪 Path 2: The Researcher (30 minutes)

**"I want to understand the theory."**

### Reading Order

1. **Start here:** `docs/THEORY.md`
   - Mathematical foundations
   - Why ternary routing works

2. **The discovery:** `papers/ZIT1_HOMEO_ADAPTIVE_FABRIC.md`
   - Homeo-adaptive computation
   - 56M node experiments
   - Scaling analysis

3. **The formalization:** `papers/FORMAL_NOTATION.md`
   - Rigorous definitions
   - Plasticity rules

4. **The context:** `papers/RELATED_WORK.md`
   - How this relates to existing work
   - Theoretical foundations

### Run the Experiments

```bash
# 512 nodes (instant)
cd trixc/forge/gillies/zit && make && ./zit_demo

# 4096 nodes
./zit_demo 16

# 64 nodes (smaller, still works)
./zit_demo 4
```

### Key Concepts to Understand
- [ ] Frozen shapes (polynomial forms that compute exactly)
- [ ] Resistance as learning signal (not error)
- [ ] Topology as learned model (not weights)
- [ ] Sublinear scaling (more nodes, proportionally fewer cycles)

---

## 💻 Path 3: The Developer (1 hour)

**"I want to use this in my code."**

### Python API (TriX)

TriX has three layers - choose based on your needs:

| Module | Dependencies | Use Case |
|--------|--------------|----------|
| `trix.shapes` | **None** (pure Python) | Simple computation, scripting |
| `trix.native` | NumPy (CuPy for training) | Inference, export to C |
| `trix.nn` | PyTorch | Gradient-based training |

#### Pure Python (Zero Dependencies)

```python
from trix.shapes import add, xor, inc

# Compute with frozen shapes
result = add(42, 13)      # 55
result = xor(0xFF, 0x55)  # 170 (0xAA)
result = inc(255)         # 0 (wraps)

# Show the geometry
add(42, 13, verbose=True)  # Prints 8-bit ripple adder diagram
```

#### Native (CuPy/NumPy)

```python
from trix.native import NativeFrozenHybrid, FrozenALU

# Frozen shapes + learned routing
model = NativeFrozenHybrid()
model.train_supervised()  # Learns opcode->shape mapping
# 0 params in shapes, ~176 in routing, 100% accuracy
```

#### PyTorch (Gradient Training)

```python
from trix import HierarchicalTriXFFN

# Drop-in FFN replacement
ffn = HierarchicalTriXFFN(d_model=512, num_tiles=64)
output, routing_info, aux_losses = ffn(x)

# Train with auxiliary losses for balanced routing
loss = task_loss + aux_losses['total_aux']
```

See: `docs/QUICKSTART.md` for full tutorial

### C API (GILLIES)

```c
#include "gillies.h"

// Execute frozen shape on any substrate
gillies_result_t result;
gillies_invoke(SHAPE_XOR, ports, 2, &result);
```

See: `trixc/forge/gillies/include/gillies.h`

### ZIT API

```c
#include "zit.h"

zit_fabric_t* f = zit_create(8);  // 8x8x8 = 512 nodes
zit_run(f, 0);                     // Run to convergence
printf("Converged in %d cycles\n", zit_cycle(f));
zit_destroy(f);
```

See: `trixc/forge/gillies/zit/zit.h`

---

## ⚡ Path 4: The Hardware Engineer (2 hours)

**"I want to implement this in hardware."**

### Verilog Reference

```
trixc/forge/rtl/
├── zit_plastic_node.v      # Single node with plasticity
├── zit_plastic_fabric.v    # Complete fabric
├── zit_plastic_tb.v        # Testbench
└── FUNKY_CONVERGENCE.md    # Implementation notes
```

### The Key Insight

Frozen shapes are polynomials. Polynomials evaluated on binary inputs produce identical results regardless of how they're computed:

```
XOR(a,b) = a + b - 2ab

This equation is true whether you:
- Evaluate it as CPU arithmetic
- Build it as digital logic gates
- Implement it in analog circuits
```

### Hardware Targets

| Target | Status | Location |
|--------|--------|----------|
| Verilog simulation | ✅ Working | `trixc/forge/rtl/` |
| CUDA (Thor GPU) | ✅ Validated | `papers/experiments/` |
| FPGA | 🔄 In progress | — |
| Custom silicon | 📋 Designed | — |

---

## 🎓 Path 5: The Deep Diver (days)

**"I want to understand everything."**

### Complete Reading List

**Foundations:**
1. `docs/SYSTEM_ARCHITECTURE.md` - How everything connects
2. `docs/THEORY.md` - Mathematical foundations
3. `docs/THE_WAY.md` - Unified philosophy

**ZIT (Topology Learning):**
4. `docs/ZIT_ONRAMP_MANIFESTO.md` - Design process
5. `papers/ZIT1_HOMEO_ADAPTIVE_FABRIC.md` - Main paper
6. `papers/EXPERIMENTAL_DATA.md` - Raw data
7. `papers/experiments/` - All CUDA implementations

**TriX (Neural Architecture):**
8. `docs/ARCHITECTURE.md` - Routing architecture
9. `docs/MESA11_UAT.md` through `MESA16_*.md` - Research evolution

**Frozen Computation:**
10. `docs/FROZEN_SHAPES.md` - Shape mathematics
11. `docs/SHAPE_SUBSTRATE.md` - Hardware implications

**Compiler:**
12. `trixc/README.md` - Complete compiler documentation

### Build Everything

```bash
# Run all tests
python -m pytest tests/ -v

# Run TRIXC tests
cd trixc && make test

# Run ZIT demo
cd trixc/forge/gillies/zit && make && ./zit_demo

# Run CUDA experiments (requires Thor/NVIDIA GPU)
cd papers/experiments && nvcc -o fabric_384 fabric_384.cu && ./fabric_384
```

---

## Quick Reference

| I want to... | Go to... |
|--------------|----------|
| See topology learn | `trixc/forge/gillies/zit/` |
| Understand the theory | `docs/THEORY.md` |
| Use Python API | `docs/QUICKSTART.md` |
| Use C API | `trixc/forge/gillies/include/` |
| See Verilog | `trixc/forge/rtl/` |
| Read the paper | `papers/ZIT1_HOMEO_ADAPTIVE_FABRIC.md` |
| See 56M experiment | `papers/experiments/fabric_384.cu` |
| Understand the method | `docs/LINCOLN_MANIFOLD_METHOD.md` |

---

## Key Concepts Glossary

| Term | Meaning |
|------|---------|
| **Frozen Shape** | A polynomial that computes exactly. Cannot be learned—it IS. |
| **Resistance** | Counter that grows when a node cannot resonate. Triggers rewiring. |
| **Resonance** | When a node's state is stable (didn't change this cycle). |
| **Topology** | The connectivity pattern between nodes. This IS the learned model. |
| **Signature** | Content-addressable identifier for a pattern. Similar patterns → similar signatures. |
| **Routing** | How inputs find their experts. In TriX, this IS learning. |
| **GILLIES** | Substrate-agnostic compute layer. Same operation on CPU/GPU/FPGA. |
| **Second Star Constant** | 1122911624. Reproducible seed for experiments. |

---

## Troubleshooting

### ZIT demo doesn't converge
- Check dimension: `./zit_demo 8` (default)
- Larger dimensions take more cycles
- If truly stuck, try different seed

### Python import fails
```bash
pip install -e .  # From repo root
```

### CUDA experiments fail
- Need NVIDIA GPU with CUDA toolkit
- Thor (Jetson AGX) recommended for 56M experiments
- 512-node demo works without GPU

### Tests fail
```bash
pip install pytest torch  # Ensure dependencies
python -m pytest tests/ -v  # Run with verbosity
```

---

## Getting Help

- **Questions about code:** Open an issue
- **Questions about theory:** Read the papers first, then open an issue
- **Found a bug:** Open an issue with reproduction steps
- **Want to contribute:** See contribution guidelines

---

*The topology IS the learned model.*
*The shape IS the instruction.*
*The routing IS the learning.*

Welcome to ZOR. 🌿
