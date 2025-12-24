# TRIXC Freshman Guide

**For curious minds who want to understand what's actually happening.**

*No ML background required. No compiler experience needed. Just curiosity.*

---

## What Is TRIXC?

TRIXC is a compiler that turns neural networks into small, fast programs.

But to understand *why* that's interesting, let's start with a question:

**What is 1 + 1?**

You know the answer is 2. You don't need to "learn" it. You don't need a huge runtime to compute it. The answer is a mathematical fact.

TRIXC is built on a simple insight: **most of what neural networks do is also mathematical fact.** We just forgot that somewhere along the way.

---

## The Problem with Traditional ML

Here's how traditional machine learning works:

```
Traditional Approach:
┌─────────────────────────────────────────────────────────────────┐
│  1. Install PyTorch (2 GB)                                      │
│  2. Load your model (500 MB)                                    │
│  3. Run inference                                               │
│  4. Answer: "approximately 2.0000001"                           │
│  5. Time: 50 milliseconds                                       │
│  6. RAM used: 4 GB                                              │
└─────────────────────────────────────────────────────────────────┘
```

That's a lot of machinery to compute something that should be... just math.

---

## The TRIXC Approach

TRIXC does something different:

```
TRIXC Approach:
┌─────────────────────────────────────────────────────────────────┐
│  1. Compile your model to C (3 KB of source code)              │
│  2. Run gcc (produces 70 KB executable)                         │
│  3. Run the executable                                          │
│  4. Answer: "exactly 2.0"                                       │
│  5. Time: 0.1 milliseconds                                      │
│  6. RAM used: whatever your data needs                          │
└─────────────────────────────────────────────────────────────────┘
```

**No runtime. No framework. Just a program that does math.**

---

## What Are "Frozen Shapes"?

This is the core idea. Let's build up to it.

### Boolean Logic: The Simplest Example

You probably know these logic operations:

| A | B | A AND B | A OR B | A XOR B |
|---|---|---------|--------|---------|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 | 1 |
| 1 | 0 | 0 | 1 | 1 |
| 1 | 1 | 1 | 1 | 0 |

These aren't learned. They're defined. XOR means "one or the other, but not both."

### The Mathematical Secret

Here's something beautiful: these operations can be written as simple math formulas:

```
AND(a, b) = a × b
OR(a, b)  = a + b - a×b
XOR(a, b) = a + b - 2×a×b
NOT(a)    = 1 - a
```

Try it! If a=1 and b=0:
- XOR(1, 0) = 1 + 0 - 2×1×0 = 1 + 0 - 0 = 1 ✓

If a=1 and b=1:
- XOR(1, 1) = 1 + 1 - 2×1×1 = 2 - 2 = 0 ✓

**This is a "frozen shape."** It's a mathematical formula that exactly computes a logical operation. It's not learned. It's not approximate. It just *is*.

### Why "Frozen"?

Because it never changes. XOR will always be `a + b - 2ab`. Yesterday. Today. Forever.

Traditional neural networks "learn" their weights during training, and those weights can be different every time. Frozen shapes are mathematical constants—they're the same everywhere, always.

---

## From Logic Gates to Neural Networks

"Okay," you might say, "logic gates are simple. But neural networks are complicated!"

Let's look at what neural networks actually do:

### Activation Functions

Every neural network uses activation functions. Here are the most common ones:

```
ReLU(x)    = max(0, x)           ← If negative, return 0. Otherwise, return x.
Sigmoid(x) = 1 / (1 + e^(-x))    ← Squash to range (0, 1)
GELU(x)    = x × sigmoid(1.702x) ← Smooth approximation of ReLU
```

These are also just math. They're not learned. They're formulas.

### Matrix Multiplication

Neural networks do a lot of this:

```
output = input × weights
```

Matrix multiplication is defined. It's not learned. The formula is:

```
C[i][j] = Σ A[i][k] × B[k][j]
```

That's it. That's the whole thing.

### What's Actually Learned?

The only thing a neural network "learns" is the **values** of the weights. The **operations** are fixed:

```
Fixed (frozen):           Learned (not frozen):
├── Matrix multiply       └── The weight values
├── ReLU activation            (numbers like 0.234, -0.891, etc.)
├── Add bias
├── Softmax
└── etc.
```

TRIXC freezes the operations into exact mathematical shapes. The learned weights become constant arrays in the compiled code.

---

## A Complete Example: The 6502 ALU

The 6502 is a famous microprocessor from 1975. It powered the Apple II, Commodore 64, and Nintendo Entertainment System.

TRIXC includes a complete 6502 ALU (Arithmetic Logic Unit) built entirely from frozen shapes:

```
6502 ALU Operations:
├── ADC: Add with Carry
├── SBC: Subtract with Borrow
├── AND: Bitwise AND
├── ORA: Bitwise OR
├── EOR: Bitwise XOR (Exclusive OR)
├── ASL: Arithmetic Shift Left
├── LSR: Logical Shift Right
├── ROL: Rotate Left
├── ROR: Rotate Right
├── INC: Increment
└── DEC: Decrement
```

Each of these is built from the basic frozen shapes:

```
8-bit ADD is built from:
└── 8 Full Adders, chained together
    └── Each Full Adder is built from:
        ├── XOR (for the sum bit)
        ├── AND (for carry detection)
        └── OR (for carry propagation)
```

The result: a 6 KB executable that performs exact 6502 arithmetic. Not emulation. Not simulation. Actual computation using the same math the original chip used.

---

## How the Compiler Works

TRIXC compilation happens in stages:

```
Stage 1: Parse
┌─────────────────────────────────────────────────────────────────┐
│  Input: ONNX model (industry-standard neural network format)    │
│  Output: Octave IR (intermediate representation)                │
│                                                                 │
│  What happens: Read the graph, identify operations              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Stage 2: Map to Shapes
┌─────────────────────────────────────────────────────────────────┐
│  Input: Octave IR                                               │
│  Output: List of frozen shape calls                             │
│                                                                 │
│  What happens: "MatMul" → trix_onnx_matmul()                   │
│                "ReLU"   → trix_onnx_relu()                     │
│                "Add"    → trix_onnx_add()                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Stage 3: Emit C Code
┌─────────────────────────────────────────────────────────────────┐
│  Input: Shape calls + weights                                   │
│  Output: Complete C source file                                 │
│                                                                 │
│  What happens: Generate model_forward() function                │
│                Embed weights as static const arrays             │
│                Add main() for standalone mode                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
Stage 4: Compile (gcc)
┌─────────────────────────────────────────────────────────────────┐
│  Input: C source                                                │
│  Output: Native executable                                      │
│                                                                 │
│  What happens: Standard C compilation                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## What Makes This Different?

### Traditional ML Runtime

```
Your Code → PyTorch → CUDA → GPU Driver → Hardware
    │           │        │         │
    │           │        │         └── Lots of layers
    │           │        └── More abstraction
    │           └── Framework overhead
    └── Python overhead
```

### TRIXC

```
Your Binary → Hardware
    │
    └── That's it
```

---

## The Precision Story

TRIXC includes an "APU" (Arithmetic Precision Unit) that manages different number formats:

```
FP4  ──▶ 4 bits   (routing, sketches)
FP8  ──▶ 8 bits   (weights, activations)
FP16 ──▶ 16 bits  (computation)
FP32 ──▶ 32 bits  (accumulation)
FP64 ──▶ 64 bits  (when you really need it)
```

Different operations can use different precisions:
- Routing decisions? FP4 is enough.
- Matrix multiply? FP16 works great.
- Accumulating sums? Use FP32 to avoid drift.

The conversions between precisions are also frozen shapes—mathematical transformations, not learned approximations.

---

## Why Would I Use This?

### You might use TRIXC if:

- **You want tiny binaries** — Deploy to embedded systems, IoT devices, or anywhere size matters
- **You want no dependencies** — Just a C compiler, that's it
- **You want exact reproducibility** — Same input = same output, forever
- **You want to understand what's happening** — Read the generated C code
- **You want to learn** — See how neural networks work at the lowest level

### You might NOT use TRIXC if:

- **You need GPU acceleration** — TRIXC generates CPU code (GPU support is planned)
- **You have huge models** — TRIXC embeds weights in the binary
- **You need dynamic shapes** — TRIXC compiles fixed dimensions

---

## Hands-On: Your First Frozen Shape

Let's implement XOR ourselves and verify it works:

```c
// xor_demo.c
#include <stdio.h>

// The frozen XOR shape: a + b - 2ab
float frozen_xor(float a, float b) {
    return a + b - 2.0f * a * b;
}

int main() {
    printf("XOR Truth Table (using frozen shape):\n");
    printf("a | b | XOR(a,b)\n");
    printf("--+---+---------\n");

    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            float result = frozen_xor((float)a, (float)b);
            printf("%d | %d |    %.0f\n", a, b, result);
        }
    }

    return 0;
}
```

Compile and run:
```bash
gcc xor_demo.c -o xor_demo
./xor_demo
```

Output:
```
XOR Truth Table (using frozen shape):
a | b | XOR(a,b)
--+---+---------
0 | 0 |    0
0 | 1 |    1
1 | 0 |    1
1 | 1 |    0
```

**You just implemented a neural network building block.** It's 4 lines of math.

---

## Hands-On: Build a Half Adder

A half adder adds two bits and produces a sum and carry:

```c
// half_adder.c
#include <stdio.h>

// Frozen shapes
float frozen_xor(float a, float b) { return a + b - 2.0f * a * b; }
float frozen_and(float a, float b) { return a * b; }

// Half adder: sum = XOR(a,b), carry = AND(a,b)
void half_adder(float a, float b, float* sum, float* carry) {
    *sum = frozen_xor(a, b);
    *carry = frozen_and(a, b);
}

int main() {
    printf("Half Adder Truth Table:\n");
    printf("a | b | sum | carry\n");
    printf("--+---+-----+------\n");

    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            float sum, carry;
            half_adder((float)a, (float)b, &sum, &carry);
            printf("%d | %d |  %.0f  |   %.0f\n", a, b, sum, carry);
        }
    }

    return 0;
}
```

Output:
```
Half Adder Truth Table:
a | b | sum | carry
--+---+-----+------
0 | 0 |  0  |   0
0 | 1 |  1  |   0
1 | 0 |  1  |   0
1 | 1 |  0  |   1
```

**You just built the fundamental unit of binary arithmetic.** Chain 8 of these together and you have an 8-bit adder—exactly like the 6502 uses.

---

## Where to Go From Here

### Start Here
1. [QUICKSTART.md](QUICKSTART.md) — Get running in 5 minutes
2. [TUTORIALS.md](TUTORIALS.md) — Step-by-step learning path

### Understand the Shapes
3. [SHAPES.md](SHAPES.md) — All 16 core frozen shapes
4. [ONNX_SHAPES.md](ONNX_SHAPES.md) — 40+ ONNX-compatible shapes

### Go Deeper
5. [APU.md](APU.md) — Mixed precision arithmetic
6. [PROVIDENCE.md](PROVIDENCE.md) — Content-addressed memory
7. [ARCHITECTURE.md](ARCHITECTURE.md) — Compiler internals

### Build Something
8. [ONNX2C.md](ONNX2C.md) — Convert your own models
9. [examples/](../examples/) — Runnable code examples

---

## Glossary

| Term | Meaning |
|------|---------|
| **Frozen Shape** | A mathematical formula that computes a fixed operation |
| **ONNX** | Open Neural Network Exchange — industry-standard model format |
| **Octave IR** | TRIXC's intermediate representation |
| **APU** | Arithmetic Precision Unit — manages number formats |
| **Providence** | Content-addressed memory system |
| **6502** | Classic 8-bit microprocessor (1975) |

---

## The Philosophy

> *"Don't learn what you can derive."*

Traditional ML: "Let's train a network to approximate XOR."

TRIXC: "XOR is `a + b - 2ab`. There's nothing to learn."

This isn't a limitation. It's a liberation. When you freeze the shapes, you get:
- **Exactness** — No approximation error
- **Reproducibility** — Same answer everywhere
- **Efficiency** — No runtime overhead
- **Transparency** — Read the code, understand the math

---

*"Everybody relax. I'm here."*

— Jack Burton (and now you)

Welcome to TRIXC.
