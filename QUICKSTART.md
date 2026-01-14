# TriX Quickstart

**Get to value in under 5 minutes.**

Choose your path:

| You are... | Start here | Time |
|------------|------------|------|
| Hardware engineer | [Import Verilog](#path-a-import-verilog) | 3 min |
| ML researcher | [Build deterministic nets](#path-b-deterministic-neural-nets) | 5 min |
| Embedded developer | [Pure C](#path-c-pure-c) | 2 min |

---

## Path A: Import Verilog

**Goal:** Take any combinational Verilog design, run it through TriX.

### Step 1: Synthesize with Yosys

```bash
# Install Yosys (if needed)
# Ubuntu: sudo apt install yosys
# macOS: brew install yosys

# Synthesize your design to JSON
yosys -p "read_verilog my_design.v; synth; write_json my_design.json"
```

### Step 2: Ingest into TriX

```python
from trix.forge.ingest import ingest_yosys_json, execute, system_summary

# Import the synthesized design
system = ingest_yosys_json("my_design.json")

# See what you got
print(system_summary(system))

# Run it
result = execute(system, {"a": 1, "b": 0, "cin": 1})
print(result)  # {"sum": 0, "cout": 1}
```

### Step 3: Validate exhaustively

```python
from trix.forge.ingest import validate_exhaustive

def expected_behavior(inputs):
    # Your golden reference
    a, b, cin = inputs["a"], inputs["b"], inputs["cin"]
    total = a + b + cin
    return {"sum": total % 2, "cout": total // 2}

passed, failures = validate_exhaustive(system, expected_behavior)
print(f"Validation: {'PASS' if passed else 'FAIL'}")
```

**What just happened:** Your Verilog became a deterministic neural network using frozen polynomial shapes. Every gate is a polynomial (XOR = a + b - 2ab). Execution is exact, not approximated.

**Next:** See [INGEST.md](INGEST.md) for supported cells and advanced usage.

---

## Path B: Deterministic Neural Nets

**Goal:** Build neural networks that never hallucinate, are fully interpretable.

### Step 1: Understand frozen shapes

```python
from trix.native.ops import TrixOps

ops = TrixOps()
print(f"Backend: {ops.simd}")  # NEON / AVX2 / scalar

import numpy as np

# These are polynomials, not approximations
a = np.array([0.0, 1.0, 0.0, 1.0])
b = np.array([0.0, 0.0, 1.0, 1.0])

xor_result = ops.xor(a, b)  # a + b - 2ab
print(xor_result)  # [0, 1, 1, 0] — exact XOR

and_result = ops.multiply(a, b)  # a * b
print(and_result)  # [0, 0, 0, 1] — exact AND
```

### Step 2: Build a glassbox classifier

```python
from trix.nn import GradientTruthFFN

# Shapes are frozen. Only routing learns.
model = GradientTruthFFN(
    d_model=64,
    num_shapes=16,
    use_native=True  # Use C/SIMD backend
)

# Train normally — gradients flow through continuous params only
output = model(x)
loss = criterion(output, target)
loss.backward()
```

### Step 3: Inspect everything

```python
# Every computation is traceable
for name, param in model.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")

# Shapes don't learn — they're algebraic
# Routing learns — it's continuous
# Scales learn — they're continuous
```

**What just happened:** You built a neural network where every operation is a frozen polynomial. No black boxes. Full interpretability. Gradients only flow through continuous parameters.

**Next:** See [GRADIENT_TRUTH.md](docs/GRADIENT_TRUTH.md) for the theory.

---

## Path C: Pure C

**Goal:** Zero dependencies, embedded deployment, maximum control.

### Step 1: Build the library

```bash
cd src/trix/native/ops
make
# Creates libtrix_ops.so (~50KB)
```

### Step 2: Use from C

```c
#include "trix_ops.h"

int main() {
    // Initialize
    TrixOps* ops = trix_ops_create();

    // Ternary matmul (no multiplication!)
    float x[512];
    float y[512];
    int8_t W[512][512];  // Weights are {-1, 0, +1}
    float scale = 1.0f;

    trix_ternary_matvec(ops, y, W, x, 512, 512, scale);

    // Frozen shapes
    float a[4] = {0, 1, 0, 1};
    float b[4] = {0, 0, 1, 1};
    float result[4];

    trix_xor(ops, result, a, b, 4);  // Polynomial XOR
    // result = {0, 1, 1, 0}

    trix_ops_destroy(ops);
    return 0;
}
```

### Step 3: Link and run

```bash
gcc my_program.c -L. -ltrix_ops -o my_program
./my_program
```

**What just happened:** You have a 50KB library that does all TriX operations with SIMD acceleration. No Python. No runtime. Deploy anywhere.

**Next:** See [trixc/](trixc/) for the complete standalone C implementation.

---

## What Makes TriX Different?

| Traditional ML | TriX |
|----------------|------|
| Probabilistic outputs | Deterministic outputs |
| Black box | Glassbox (every op visible) |
| Approximation | Exact computation |
| Requires GPU | Runs on any CPU |
| Weights are learned floats | Weights are {-1, 0, +1} |
| Shapes are learned | Shapes are frozen polynomials |

**The key insight:** Boolean logic has exact polynomial representations.

```
XOR(a, b) = a + b - 2ab

Proof:
  a=0, b=0: 0 + 0 - 0 = 0  ✓
  a=1, b=0: 1 + 0 - 0 = 1  ✓
  a=0, b=1: 0 + 1 - 0 = 1  ✓
  a=1, b=1: 1 + 1 - 2 = 0  ✓
```

This means you can build neural networks that are **algebraically equivalent** to digital circuits. No approximation. No hallucination. Guaranteed correct.

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'trix'"

```bash
# Install from source
cd /path/to/trix
pip install -e .
```

### "ImportError: libtrix_ops.so not found"

```bash
# Build native ops
cd src/trix/native/ops
make
```

### "Permission denied"

```bash
# Fix permissions
chmod -R a+r src/trix/native/
```

### "PyTorch not installed" (but you don't need it)

For inference-only use (Ingest, native ops), you don't need PyTorch:

```python
# Direct import bypasses PyTorch dependency
from trix.forge.ingest import ingest_yosys_json, execute
# or
from trix.native.ops import TrixOps
```

---

## Next Steps

| Goal | Resource |
|------|----------|
| Deep understanding | [ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Verilog import details | [INGEST.md](docs/INGEST.md) |
| Training theory | [GRADIENT_TRUTH.md](docs/GRADIENT_TRUTH.md) |
| Philosophy | [THE_WAY.md](docs/THE_WAY.md) |
| Full examples | [examples/](examples/) |
