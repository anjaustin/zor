# TRIXC Tutorials

**A structured path from "Hello XOR" to "I compiled my own model."**

---

## Learning Paths

Choose your adventure:

| Path | For | Time | Goal |
|------|-----|------|------|
| [Speed Run](#speed-run) | Impatient people | 15 min | Get something working |
| [The Full Journey](#the-full-journey) | Curious learners | 2 hours | Understand everything |
| [Tinkerer's Trail](#tinkerers-trail) | Hackers | 1 hour | Modify and experiment |

---

## Speed Run

**15 minutes to a working neural network binary.**

### Step 1: Build (2 min)

```bash
cd trixc
make clean && make
./build/test_rigorous | tail -10
```

You should see 1,329 tests pass.

### Step 2: Run the 6502 ALU (1 min)

```bash
./build/alu6502 ADC 100 55
./build/alu6502 EOR 0xFF 0xAA
```

A 6 KB CPU that does real math.

### Step 3: Convert ONNX to Binary (5 min)

```bash
pip install onnx numpy
python examples/demo_onnx2c.py
```

You just compiled a neural network to native code.

### Step 4: Run the Examples (7 min)

```bash
cd examples
gcc -o hello_xor 01_hello_xor.c && ./hello_xor
gcc -o logic_gates 02_logic_gates.c && ./logic_gates
gcc -o tiny_mlp 06_tiny_mlp.c -lm && ./tiny_mlp
```

**Done!** You've seen frozen shapes, logic circuits, and neural networks.

---

## The Full Journey

**2 hours to truly understand TRIXC.**

### Chapter 1: The Foundation (20 min)

**Goal:** Understand that Boolean logic is just polynomials.

1. **Read:** [FRESHMAN_GUIDE.md](FRESHMAN_GUIDE.md) — The conceptual introduction
2. **Run:** `examples/01_hello_xor.c` — Your first frozen shape
3. **Run:** `examples/02_logic_gates.c` — All 7 logic gates
4. **Verify:** De Morgan's laws work with these shapes

**Key insight:** `XOR(a, b) = a + b - 2ab` is not an approximation. It's exact.

### Chapter 2: Building Circuits (20 min)

**Goal:** See how simple shapes compose into complex circuits.

1. **Run:** `examples/03_full_adder.c` — Build a 4-bit adder
2. **Try:** Extend it to 8 bits (just add 4 more full adders!)
3. **Read:** How the 6502 ALU is built from these primitives

**Key insight:** Every CPU operation is just logic gates arranged carefully.

```
8-bit ADD = 8 × Full Adder
          = 8 × (XOR + AND + OR)
          = 8 × (polynomials)
          = just math
```

### Chapter 3: Neural Networks (30 min)

**Goal:** Demystify "deep learning."

1. **Run:** `examples/04_activations.c` — See ReLU, GELU, Sigmoid
2. **Run:** `examples/05_matmul.c` — The heart of neural nets
3. **Run:** `examples/06_tiny_mlp.c` — A complete working network

**Key insight:** A neural network is just:
- Matrix multiply (combine inputs with weights)
- Activation function (add non-linearity)
- Repeat

### Chapter 4: The ONNX Pipeline (30 min)

**Goal:** Convert real models to native code.

1. **Read:** [ONNX2C.md](ONNX2C.md) — The conversion pipeline
2. **Run:** `examples/demo_onnx2c.py` — End-to-end conversion
3. **Try:** Modify the weights and see the output change
4. **Read:** The generated C code — it's surprisingly readable

**Key insight:** TRIXC doesn't "run" neural networks. It compiles them to C.

### Chapter 5: Advanced Topics (20 min)

**Goal:** Understand the deeper architecture.

1. **Read:** [APU.md](APU.md) — Mixed precision arithmetic
2. **Read:** [PROVIDENCE.md](PROVIDENCE.md) — Content-addressed memory
3. **Read:** [SPARSE_OCTAVE.md](SPARSE_OCTAVE.md) — Multi-scale lookup
4. **Skim:** [ARCHITECTURE.md](ARCHITECTURE.md) — Compiler internals

**Key insight:** Precision is a shape. Memory is content-addressed. Everything is frozen.

### Final Exam

If you can answer these, you understand TRIXC:

1. What is `XOR(a, b)` as a polynomial?
2. How many full adders do you need for 8-bit addition?
3. What two operations does every neural network layer do?
4. What's the difference between learned weights and frozen shapes?
5. Why is a TRIXC binary smaller than PyTorch?

<details>
<summary>Answers</summary>

1. `XOR(a, b) = a + b - 2ab`
2. 8 (one per bit, chained by carry)
3. Matrix multiply + activation function
4. Weights are trained once then frozen. Shapes are mathematical formulas that never change.
5. No runtime, no framework, no Python—just the math you need.

</details>

---

## Tinkerer's Trail

**1 hour of hands-on experiments.**

### Experiment 1: Verify the Math (10 min)

Open `examples/01_hello_xor.c` and:

1. Add a test for `XOR(0.5, 0.5)` — What happens with non-binary inputs?
2. Implement the challenge: AND, OR, NOT
3. Verify they match the truth tables

### Experiment 2: Build a Bigger Adder (15 min)

Modify `examples/03_full_adder.c`:

1. Extend `add_4bit` to `add_8bit`
2. Test with values > 15
3. Add subtraction using `SBC` semantics

Hint: 6502 subtraction uses "borrow" which is inverted carry.

### Experiment 3: Train Your Own Network (20 min)

The weights in `examples/06_tiny_mlp.c` were trained externally. Try:

1. Change the weights slightly — does it still work?
2. Try making it compute AND instead of XOR
3. Google "XOR neural network weights" and try other solutions

### Experiment 4: Convert Your Own Model (15 min)

Create a custom ONNX model:

```python
# my_model.py
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# Your model: double the input
weight = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
weight_init = numpy_helper.from_array(weight, name="W")

input_t = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
output_t = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])

matmul = helper.make_node("MatMul", ["input", "W"], ["output"])
graph = helper.make_graph([matmul], "doubler", [input_t], [output_t], [weight_init])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

onnx.save(model, "doubler.onnx")
print("Created doubler.onnx")
```

Then convert and run:

```bash
python my_model.py
python tools/onnx2trix.py doubler.onnx doubler.c --emit-c
gcc -O3 -DTRIXC_STANDALONE -I./include doubler.c -o doubler -lm
```

---

## Example Progression

| # | Example | Concepts | Dependencies |
|---|---------|----------|--------------|
| 01 | `hello_xor.c` | Frozen shape, polynomial | None |
| 02 | `logic_gates.c` | All 7 gates, De Morgan | None |
| 03 | `full_adder.c` | Composition, circuits | None |
| 04 | `activations.c` | ReLU, GELU, Sigmoid | `-lm` |
| 05 | `matmul.c` | Matrix ops, neurons | `-lm` |
| 06 | `tiny_mlp.c` | Complete neural network | `-lm` |
| 07 | `demo_onnx2c.py` | ONNX conversion | `onnx`, `numpy` |

Build all C examples:

```bash
cd examples
for f in *.c; do
    echo "Building $f..."
    gcc -o "${f%.c}" "$f" -lm 2>/dev/null || gcc -o "${f%.c}" "$f"
done
```

---

## Common Questions

### "Why not just use PyTorch?"

PyTorch is great for research and training. TRIXC is for deployment:

| | PyTorch | TRIXC |
|---|---------|-------|
| Binary size | 2 GB | 70 KB |
| Startup time | seconds | instant |
| Dependencies | CUDA, cuDNN, Python | C compiler |
| Reproducibility | ~exact | exact |
| Debugging | print statements | read the C code |

### "Can I train models with TRIXC?"

No. TRIXC is a compiler, not a training framework. The workflow is:

1. Train your model in PyTorch/TensorFlow/JAX
2. Export to ONNX
3. Convert with TRIXC
4. Deploy the tiny binary

### "What models are supported?"

See [SUPPORTED_OPS.md](SUPPORTED_OPS.md). Currently:
- MLPs (fully connected networks)
- Transformers (encoder-style)
- Basic CNNs (with some work)

### "How do I debug a TRIXC model?"

Read the generated C code! It's straightforward:

```c
void model_forward(const float* input, float* output) {
    /* Layer 1 */
    trix_onnx_matmul(input, W_fc1, t0, 1, 4, 2);
    for (int i = 0; i < 4; i++) t0[i] = trix_onnx_relu(t0[i]);

    /* Layer 2 */
    trix_onnx_matmul(t0, W_fc2, output, 1, 2, 4);
}
```

Add `printf` statements. Use gdb. It's just C.

---

## Where to Go Next

### If you want to go deeper:
- [APU.md](APU.md) — Mixed precision arithmetic
- [ARCHITECTURE.md](ARCHITECTURE.md) — Compiler internals
- [TEST_SUITE.md](TEST_SUITE.md) — 1,375 tests to study

### If you want to contribute:
- Add support for more ONNX operations
- Implement a CUDA backend
- Optimize the generated code

### If you want to deploy:
- [ONNX2C.md](ONNX2C.md) — Production conversion guide
- Cross-compile for ARM, RISC-V, WebAssembly

---

*"You know what ol' Jack Burton always says at a time like this?"*

*"Who?"*

*"Jack Burton. Me."*

Welcome to TRIXC. Now go build something.
