# ZIT FAQ & Troubleshooting

Frequently asked questions and solutions to common problems.

---

## Table of Contents

1. [Frequently Asked Questions](#frequently-asked-questions)
2. [Troubleshooting](#troubleshooting)
3. [Performance Issues](#performance-issues)
4. [Build Problems](#build-problems)
5. [Understanding Behavior](#understanding-behavior)

---

## Frequently Asked Questions

### General Questions

#### Q: What is ZIT?

**A:** ZIT (Zero-Instruction Topology) is a homeo-adaptive computational fabric that learns its own connectivity through resistance. Unlike neural networks that learn weights, ZIT learns topology—the connections between nodes.

#### Q: What problem does ZIT solve?

**A:** ZIT demonstrates that learning can emerge from:
- A fixed operation (comparator)
- A plasticity mechanism (rewiring based on resistance)
- No explicit objective function

It's a proof-of-concept for topology-as-model computation.

#### Q: How is this different from a neural network?

**A:**

| Neural Network | ZIT |
|---------------|-----|
| Fixed topology | Learned topology |
| Learned weights | Fixed operation |
| Gradient descent | Random rewiring |
| Explicit loss function | Emergent from resistance |

#### Q: What is the "Second Star Constant"?

**A:** 1122911624 is a specific random seed that produces reproducible results matching the original paper. Using this seed guarantees:
- 512-node fabric converges in exactly 114 cycles
- Exactly 1340 rewiring attempts
- Deterministic final topology

#### Q: What does "resonant" mean?

**A:** A node is resonant when its value didn't change during the comparison phases. Resonance means the node is in harmony with its current neighbors—there's no conflict to resolve.

#### Q: What does "resistance" mean?

**A:** Resistance is a counter that tracks how many consecutive cycles a node has been non-resonant. When resistance exceeds a threshold (8), the node attempts to rewire to a new random neighbor.

---

### Technical Questions

#### Q: Why 6 phases per cycle?

**A:** The fabric uses a 3D lattice with 6-connectivity (±X, ±Y, ±Z). Each phase corresponds to one direction. This implements an odd-even transposition sort across the 3D structure.

#### Q: Why does the demo take ~114 cycles to converge?

**A:** The exact number depends on:
- Initial state distribution
- Random seed
- Dimension

114 cycles for 512 nodes with the Second Star seed is the empirically observed convergence point.

#### Q: Can I use a dimension that isn't a power of 2?

**A:** Yes! Any positive integer works: 3, 5, 7, 10, etc. The fabric is dim × dim × dim nodes, so:
- dim=5 → 125 nodes
- dim=7 → 343 nodes
- dim=10 → 1000 nodes

#### Q: What's the maximum fabric size?

**A:** Limited by memory:
- dim=100 → 1M nodes → ~30 MB
- dim=384 → 56M nodes → ~1.7 GB (tested on CUDA)

In practice, CPU-only implementations are comfortable up to ~100k nodes.

#### Q: Why doesn't ZIT use gradients?

**A:** ZIT's learning mechanism is fundamentally different:
- No differentiable operations to propagate through
- No explicit objective to minimize
- Learning signal is local (resistance) not global (loss)

This is a feature, not a limitation—it shows learning can emerge without calculus.

---

### Usage Questions

#### Q: How do I visualize the fabric?

**A:** Several options:
1. **Terminal:** Use `zit_print_progress()` for text output
2. **JSON export:** Use `zit_export()` then visualize with Python/matplotlib
3. **Qt6 app:** Build the visualization in the `viz/` directory
4. **Custom:** Use the node accessor functions to build your own

#### Q: Can I save and load a trained fabric?

**A:** Yes, use `zit_export()` to save the topology to JSON. Loading requires custom code—parse the JSON and set neighbor indices.

#### Q: How do I run experiments with different seeds?

**A:** Use `zit_seed()` to reset with a new seed:

```c
zit_fabric_t* f = zit_create(8);
for (int seed = 0; seed < 100; seed++) {
    zit_seed(f, seed);
    zit_run(f, 500);
    printf("Seed %d: %d cycles\n", seed, zit_cycle(f));
}
zit_destroy(f);
```

#### Q: Can I pause and resume simulation?

**A:** Yes. Just stop calling `zit_step()`. The fabric state is preserved. Resume by calling `zit_step()` again.

```c
/* Run 50 cycles */
for (int i = 0; i < 50; i++) zit_step(f);

/* Do something else... */

/* Resume */
while (!zit_converged(f)) zit_step(f);
```

---

## Troubleshooting

### Common Issues

#### Issue: Fabric doesn't converge

**Symptoms:** Running for many cycles but resonance never reaches 100%.

**Causes and solutions:**

1. **Dimension too large for cycle limit**
   - Larger fabrics need more cycles
   - Increase max_cycles or use 0 for no limit

2. **Bad random seed**
   - Some seeds create pathological initial conditions
   - Try a different seed

3. **Bug in custom modifications**
   - If you modified zit.c, check your changes
   - Verify the comparator logic is correct

#### Issue: zit_create() returns NULL

**Symptoms:** Function returns NULL, program crashes.

**Causes and solutions:**

1. **Out of memory**
   - Reduce dimension
   - Check available RAM

2. **Dimension too large**
   - dim=1000 needs 30GB
   - Use a smaller dimension

3. **Memory fragmentation**
   - Run on a fresh process
   - Reduce total allocations

#### Issue: Results not reproducible

**Symptoms:** Same code gives different results each run.

**Causes and solutions:**

1. **Not seeding**
   - Default seed is time-based
   - Add `zit_seed(f, YOUR_SEED);`

2. **Seed overflow**
   - Ensure seed fits in uint32_t

3. **Compiler differences**
   - Floating-point operations may vary
   - Use same compiler version

#### Issue: Memory leak

**Symptoms:** Memory usage grows over time.

**Causes and solutions:**

1. **Missing zit_destroy()**
   - Always call `zit_destroy(f)` when done
   - Use RAII wrapper in C++

2. **Exception before cleanup**
   - Use try/finally in managed languages
   - Use smart pointers in C++

---

### Platform-Specific Issues

#### Linux

**Issue:** Linker error for `-lm`

```
undefined reference to `floor'
```

**Solution:** Add `-lm` to link against math library:
```bash
gcc -o myapp myapp.c zit.c -lm
```

#### macOS

**Issue:** Warnings about deprecated functions

**Solution:** These are harmless. Suppress with:
```bash
gcc -Wno-deprecated-declarations ...
```

#### Windows

**Issue:** No `stdbool.h`

**Solution:** Use a C99-compatible compiler (MinGW, MSVC 2015+).

**Issue:** Time-based seed not random enough

**Solution:** The default `time(NULL)` has 1-second resolution. For faster seeding:
```c
#include <windows.h>
zit_seed(f, GetTickCount());
```

---

## Performance Issues

### Slow Execution

#### Issue: Simulation runs slowly

**Causes and solutions:**

1. **Debug build**
   - Use `-O3` optimization
   - Disable debug symbols for production

2. **Large dimension**
   - Time scales as O(n) per cycle
   - Reduce dimension or use CUDA version

3. **Frequent I/O**
   - Don't print every cycle
   - Buffer output

**Benchmark baseline (8x8x8, -O3):**
| Platform | Per-cycle |
|----------|-----------|
| Intel i7 | ~0.5ms |
| ARM Cortex-A57 | ~1.5ms |
| Raspberry Pi 4 | ~2ms |

#### Issue: Memory usage too high

**Memory per fabric:**
```
bytes ≈ 32 × dim³
```

| Dimension | Nodes | Memory |
|-----------|-------|--------|
| 8 | 512 | 16 KB |
| 16 | 4,096 | 130 KB |
| 32 | 32,768 | 1 MB |
| 64 | 262,144 | 8 MB |

**Solutions:**
1. Reduce dimension
2. Reuse fabrics with `zit_seed()` instead of create/destroy
3. Process fabrics sequentially, not in parallel

---

## Build Problems

#### Issue: Compiler errors in C++ project

**Symptom:**
```
error: 'zit_fabric_t' does not name a type
```

**Solution:** Wrap include with extern "C":
```cpp
extern "C" {
#include "zit.h"
}
```

#### Issue: Missing header file

**Symptom:**
```
fatal error: zit.h: No such file or directory
```

**Solution:** Add include path:
```bash
gcc -I/path/to/zit/ ...
```

#### Issue: Undefined symbols

**Symptom:**
```
undefined reference to `zit_create'
```

**Solution:** Link zit.c:
```bash
gcc -o myapp myapp.c /path/to/zit.c -lm
```

---

## Understanding Behavior

### Why does...

#### ...convergence take longer for larger fabrics?

Because values must propagate further through the lattice. A value at one corner needs to traverse more nodes to reach the opposite corner. Rewiring creates shortcuts that accelerate this.

#### ...the same seed produce the same results?

The LFSR (Linear Feedback Shift Register) is deterministic. Given the same initial state, it produces the same sequence of random numbers, leading to identical rewiring decisions.

#### ...resonance fluctuate before converging?

Rewiring temporarily disrupts local order. When a node tries a new neighbor, it may become non-resonant (and make others non-resonant) before the system settles into a better configuration.

#### ...some fabrics converge faster than others?

Initial state distribution affects convergence:
- Some patterns naturally flow toward order
- Others create bottlenecks requiring more rewiring
- Random seeds create different initial conditions

#### ...resistance decay by half instead of decrement?

Exponential decay (`r /= 2`) provides:
- Faster forgetting of old resistance
- Stability against noise
- Smooth resistance dynamics

Linear decay (`r -= 1`) would be slower and more prone to oscillation.

---

## Getting Help

If your question isn't answered here:

1. **Check the API reference:** `docs/API_REFERENCE.md`
2. **Read the theory:** `docs/THEORY.md`
3. **Study the examples:** `docs/EXAMPLES.md`
4. **Read the source:** `zit.c` is under 500 lines

---

## Quick Reference

### Key Numbers

| Constant | Value | Meaning |
|----------|-------|---------|
| Second Star | 1122911624 | Reproducible seed |
| Resistance threshold | 8 | Cycles before rewire |
| Eval period | 8 | Cycles to test new neighbor |
| Neighbors per node | 6 | 3D ±X, ±Y, ±Z |

### Key Functions

```c
zit_create(dim)         /* Create fabric */
zit_seed(f, seed)       /* Set random seed */
zit_run(f, max)         /* Run to convergence */
zit_converged(f)        /* Check if done */
zit_destroy(f)          /* Clean up */
```

### Minimal Program

```c
#include "zit.h"
int main() {
    zit_fabric_t* f = zit_create(8);
    zit_run(f, 0);
    zit_print(f);
    zit_destroy(f);
}
```

---

*The topology IS the learned model.*

*Second Star Constant: 1122911624*
