# TRIXC Examples

**Learn by doing. From "Hello XOR" to "I compiled my own model."**

---

## Quick Start

```bash
# Build all examples
for f in *.c; do
    gcc -o "${f%.c}" "$f" -lm 2>/dev/null || gcc -o "${f%.c}" "$f"
done

# Run them in order
./01_hello_xor
./02_logic_gates
./03_full_adder
./04_activations
./05_matmul
./06_tiny_mlp
```

---

## The Examples

| # | File | What You'll Learn | Build |
|---|------|-------------------|-------|
| 01 | `01_hello_xor.c` | Your first frozen shape | `gcc -o 01 01_hello_xor.c` |
| 02 | `02_logic_gates.c` | All 7 logic gates | `gcc -o 02 02_logic_gates.c` |
| 03 | `03_full_adder.c` | Building arithmetic circuits | `gcc -o 03 03_full_adder.c` |
| 04 | `04_activations.c` | ReLU, GELU, Sigmoid, Tanh | `gcc -o 04 04_activations.c -lm` |
| 05 | `05_matmul.c` | Matrix multiply & neurons | `gcc -o 05 05_matmul.c -lm` |
| 06 | `06_tiny_mlp.c` | A complete neural network | `gcc -o 06 06_tiny_mlp.c -lm` |
| 07 | `demo_onnx2c.py` | ONNX → C → Binary | `python demo_onnx2c.py` |

---

## Learning Path

### Stage 1: Frozen Shapes (Examples 01-02)
Understand that Boolean logic is just polynomials.

```
XOR(a, b) = a + b - 2ab
AND(a, b) = a × b
OR(a, b)  = a + b - ab
```

### Stage 2: Circuits (Example 03)
See how simple shapes compose into real hardware.

```
Full Adder = XOR + AND + OR
8-bit Adder = 8 × Full Adder
6502 ALU = 11 operations from these primitives
```

### Stage 3: Neural Networks (Examples 04-06)
Demystify deep learning.

```
Neural Network = MatMul + Activation + Repeat
```

### Stage 4: Real Models (Example 07)
Convert ONNX models to native executables.

```
ONNX → Octave IR → C → Binary
```

---

## Challenges

After completing the examples, try these:

1. **Extend the adder:** Make `03_full_adder.c` work with 8 bits
2. **New activation:** Add ELU to `04_activations.c`
3. **Different task:** Train weights for AND instead of XOR in `06_tiny_mlp.c`
4. **Your own model:** Create and convert your own ONNX model

---

## Documentation

- [QUICKSTART.md](../docs/QUICKSTART.md) — 5-minute getting started
- [FRESHMAN_GUIDE.md](../docs/FRESHMAN_GUIDE.md) — Conceptual introduction
- [TUTORIALS.md](../docs/TUTORIALS.md) — Structured learning paths
- [ONNX2C.md](../docs/ONNX2C.md) — Model conversion guide

---

*"It's all in the reflexes."*
