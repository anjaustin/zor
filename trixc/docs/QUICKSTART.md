# TRIXC Quickstart

**From zero to running in 5 minutes.**

---

## What You'll Build

In this quickstart, you'll:
1. Compile the TRIXC test suite (30 seconds)
2. Run a 6502 ALU operation (10 seconds)
3. Convert an ONNX model to native code (2 minutes)

By the end, you'll have a working neural network as a standalone executable.

---

## Prerequisites

- GCC (any recent version)
- Python 3.8+ with `pip`
- 5 minutes

```bash
# Check you have what you need
gcc --version
python3 --version
```

---

## Step 1: Clone and Build (30 seconds)

```bash
# Get TRIXC
cd trixc

# Build everything
make clean && make

# Run the test suite
./build/test_rigorous
```

You should see:
```
╔═══════════════════════════════════════════════════════════════╗
║  Total tests:   1329                                          ║
║  Passed:        1329                                          ║
║  Failed:           0                                          ║
╚═══════════════════════════════════════════════════════════════╝

✓ ALL TESTS PASSED
```

**Congratulations!** You just verified 65,536 8-bit additions are mathematically correct.

---

## Step 2: Run the 6502 ALU (10 seconds)

TRIXC includes a complete 6502 ALU built from frozen shapes:

```bash
# Build the ALU demo
make build/alu6502

# Run some operations
./build/alu6502 ADC 100 55      # Add: 100 + 55 = 155
./build/alu6502 AND 0xAA 0x55   # AND: 10101010 & 01010101 = 0
./build/alu6502 EOR 0xFF 0xAA   # XOR: 11111111 ^ 10101010 = 01010101
./build/alu6502 ASL 0x80        # Shift left: 10000000 << 1 = 0, carry=1
```

**That's a working CPU ALU in 6 KB.** No emulation. Pure math.

---

## Step 3: Install Python Dependencies (30 seconds)

```bash
pip install numpy onnx
```

---

## Step 4: Convert ONNX to Native Binary (2 minutes)

Let's convert a real neural network to a standalone executable:

```bash
# Run the end-to-end demo
python examples/demo_onnx2c.py
```

You'll see:
```
Step 1: Creating ONNX model...
Step 2: Converting ONNX to C...
  - Lines: 111
  - Size: 2999 bytes
Step 3: Compiling C code...
  - Binary size: 69.1 KB
Step 5: Running compiled model...
Step 6: Verifying output...
  TRIXC output:    [5.0, 6.0]
  Expected output: [5.0, 6.0]

SUCCESS!
```

**You just compiled a neural network to a native executable.**

---

## Step 5: Try Your Own Model

Create a file called `my_first_model.py`:

```python
#!/usr/bin/env python3
"""My first TRIXC model."""

import sys
from pathlib import Path

# Add TRIXC tools to path
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
from onnx2trix import ONNX2TrixConverter, generate_c_code

# Step 1: Create a simple model
# This one doubles the input: output = input * 2

weight = np.array([[2.0]], dtype=np.float32)  # 1x1 weight = 2
weight_init = numpy_helper.from_array(weight, name="weight")

input_t = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])
output_t = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])

matmul = helper.make_node("MatMul", ["input", "weight"], ["output"])

graph = helper.make_graph([matmul], "doubler", [input_t], [output_t], [weight_init])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

# Save ONNX
onnx.save(model, "doubler.onnx")
print("Created: doubler.onnx")

# Step 2: Convert to C
converter = ONNX2TrixConverter(emit_weights=True)
trix = converter.convert("doubler.onnx")
c_code = generate_c_code(trix, standalone=True)

with open("doubler.c", "w") as f:
    f.write(c_code)
print("Generated: doubler.c")

# Step 3: Show what to do next
print("\nNow compile and run:")
print("  gcc -O3 -DTRIXC_STANDALONE -I./include doubler.c -o doubler -lm")
print("  echo '5.0' | python -c \"import struct; import sys; sys.stdout.buffer.write(struct.pack('f', 5.0))\" > input.bin")
print("  ./doubler input.bin output.bin")
print("  python -c \"import numpy as np; print('Output:', np.fromfile('output.bin', dtype=np.float32))\"")
```

Run it:
```bash
python my_first_model.py
gcc -O3 -DTRIXC_STANDALONE -I./include doubler.c -o doubler -lm
```

**You now have a `doubler` executable that multiplies any input by 2.**

---

## What Just Happened?

1. **ONNX model** → Describes the computation as a graph
2. **onnx2trix.py** → Converts graph to frozen shapes
3. **generate_c_code()** → Emits C source with embedded weights
4. **gcc** → Compiles to native binary
5. **./doubler** → Runs the neural network. No Python. No runtime.

```
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│   ONNX     │ ──▶ │  Octave IR │ ──▶ │   C Code   │ ──▶ │   Binary   │
│  (graph)   │     │   (JSON)   │     │  (source)  │     │   (exe)    │
└────────────┘     └────────────┘     └────────────┘     └────────────┘
     ~1 KB              ~2 KB            ~3 KB             ~70 KB
```

---

## Next Steps

| Want to... | Read this |
|------------|-----------|
| Understand frozen shapes | [FRESHMAN_GUIDE.md](FRESHMAN_GUIDE.md) |
| Learn the theory | [SHAPES.md](SHAPES.md) |
| See all ONNX operations | [ONNX_SHAPES.md](ONNX_SHAPES.md) |
| Understand the APU | [APU.md](APU.md) |
| Run all 1,375 tests | [TEST_SUITE.md](TEST_SUITE.md) |
| Convert your own models | [ONNX2C.md](ONNX2C.md) |

---

## Troubleshooting

### "gcc not found"
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install

# Windows
# Install MinGW or use WSL
```

### "No module named 'onnx'"
```bash
pip install onnx numpy
```

### "trixc/onnx_shapes.h not found"
Make sure you're in the `trixc` directory:
```bash
cd trixc
gcc -I./include ...
```

---

## The Philosophy

> *"Don't learn what you can derive."*

TRIXC doesn't approximate. It doesn't guess. It computes exact mathematical shapes:

- `XOR(a, b) = a + b - 2ab` — Not learned. Derived from Boolean algebra.
- `ReLU(x) = max(0, x)` — Not approximated. Exactly computed.
- `MatMul(A, B)` — Not optimized. Just... matrix multiplication.

The model you just compiled will give the **exact same output** on any platform, forever. Because math doesn't change.

---

*"It's all in the reflexes."*

Welcome to TRIXC.
