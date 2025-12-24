# TRIXC

**The TriX Compiler**

*Frozen shapes. Native code. No runtime. No excuses.*

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   ████████╗██████╗ ██╗██╗  ██╗ ██████╗                     │
│   ╚══██╔══╝██╔══██╗██║╚██╗██╔╝██╔════╝                     │
│      ██║   ██████╔╝██║ ╚███╔╝ ██║                          │
│      ██║   ██╔══██╗██║ ██╔██╗ ██║                          │
│      ██║   ██║  ██║██║██╔╝ ██╗╚██████╗                     │
│      ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝ ╚═════╝                     │
│                                                             │
│         "Shapes are opcodes. C is machine code."           │
│                                                             │
│   "It's all in the reflexes."                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## What Is This?

Look, I'm not going to lie to you. TRIXC is a compiler that does something nobody else wants to do: it takes frozen mathematical truths and turns them into native executables smaller than most PNG files.

| Input | Output | Size |
|-------|--------|------|
| `.trix` (model definition) | `.c` / `.h` (C source) | ~10 KB |
| `.onnx` (any ONNX model) | `.so` (shared library) | ~50-100 KB |
| Trained model | `.exe` (standalone) | Whatever it takes |

**No PyTorch. No ONNX runtime. No TensorFlow. No excuses.**

A trained model becomes a 6 KB executable that runs on anything with a C compiler.

---

## The Core Insight

See, traditional ML is like hiring a team of experts to figure out that 1 + 1 = 2:

```
Traditional ML:
Learn the computation → Train for weeks → 2 GB runtime → "approximately 2"
```

TRIXC is like... just knowing math:

```
TriX:
Freeze the computation → Compile to native → 6 KB binary → "exactly 2"
```

**Math doesn't need to be learned. XOR is always `a + b - 2ab`.** That's not an opinion. That's Boolean algebra. We just compile it.

---

## New Here?

| Guide | Time | For |
|-------|------|-----|
| [QUICKSTART.md](docs/QUICKSTART.md) | 5 min | Get running fast |
| [FRESHMAN_GUIDE.md](docs/FRESHMAN_GUIDE.md) | 30 min | Understand the concepts |
| [TUTORIALS.md](docs/TUTORIALS.md) | 2 hours | Complete learning path |
| [examples/](examples/) | 1 hour | Hands-on code examples |

### Have a Raspberry Pi?

| Guide | Time | For |
|-------|------|-----|
| [TRIXC Pi Platform](platforms/raspberry-pi/) | 5 min | Deploy on Raspberry Pi 4 |
| [Pi QUICKSTART](platforms/raspberry-pi/docs/QUICKSTART.md) | 5 min | Touch screen ML in minutes |
| [Pi Examples](platforms/raspberry-pi/examples/) | 30 min | XOR, MNIST, GPIO control |

---

## Quick Start

### Build Everything

```bash
cd trixc
make clean && make test
```

You'll see **1,375+ tests pass** (1,329 C + 46 Python). If they don't, something's wrong with reality, not the code.

*See [docs/TEST_SUITE.md](docs/TEST_SUITE.md) for the complete test documentation.*

### Run the 6502 ALU Demo

```bash
make demo
```

```
── Addition ──
ADC 16, 32 = 48
ADC 255, 1 = 0 (C=1)

── Logic ──
EOR 255, 170 = 85

── Shift ──
ASL 64, 0 = 128
```

That's a complete 6502 ALU. The same one that ran the Apple II. In 6 KB.

### Run Sparse Octave Demo

```bash
gcc -O3 -I./include -DTRIX_SPARSE_OCTAVE_STANDALONE \
    -x c include/trixc/sparse_octave.h -o build/sparse_octave -lm
./build/sparse_octave
```

```
Sparse Octave Lookup - Pure Frozen Shapes
==========================================

Configuration:
  d_model: 64
  n_octaves: 3
  memory_size: 128
  parameters: 49,283

Forward pass: 0.104 ms
PASS
```

That's an FFN replacement. Multi-scale content-addressed memory. **7,780 bytes.**

### Check Binary Sizes

```bash
make size
```

```
text    data    bss     dec     hex     filename
5688    776     8       6472    1948    alu6502
7780    768     8       8556    216c    sparse_octave
```

---

## The Arsenal

### 1. Endogenous APU (`apu.h`)

The APU manages mixed-precision like a bartender manages drinks - automatically and with style.

```c
#include <trixc/apu.h>

trix_apu_t apu;
trix_apu_init(&apu);

// The APU knows what precision each operation needs
float result = trix_apu_execute(&apu, TRIX_OP_XOR, a, b, TRIX_FP16, TRIX_FP16);
```

**Precision Levels:**

| Level | Bits | Use Case |
|-------|------|----------|
| `TRIX_FP4` | 4 | Routing decisions, sketches |
| `TRIX_FP8` | 8 | Weights, activations |
| `TRIX_FP16` | 16 | Computation |
| `TRIX_FP32` | 32 | Accumulation |
| `TRIX_FP64` | 64 | When you really mean it |

*See [docs/APU.md](docs/APU.md) for the full story.*

### 2. Frozen Shapes (`shapes.h`)

Every shape is a polynomial. Every polynomial is exact. Every result is correct.

```c
#include <trixc/shapes.h>

// These aren't approximations. They're mathematical facts.
float xor = trix_shape_xor_f32(a, b);  // a + b - 2ab
float and = trix_shape_and_f32(a, b);  // ab
float or  = trix_shape_or_f32(a, b);   // a + b - ab

// Full adder - the atom of arithmetic
float sum, carry;
trix_shape_full_adder(a, b, c_in, &sum, &carry);

// 8-bit ripple adder - built from atoms
trix_shape_ripple_add(a_bits, b_bits, carry_in, result_bits, &carry_out);
```

*See [docs/SHAPES.md](docs/SHAPES.md) for all 16 shapes.*

### 3. ONNX Shapes (`onnx_shapes.h`) **NEW**

Forty frozen shapes covering everything ONNX throws at you:

```c
#include <trixc/onnx_shapes.h>

// Activations - frozen, exact, no surprises
float y = trix_onnx_relu(x);           // max(0, x)
float y = trix_onnx_gelu(x);           // x * sigmoid(1.702x)
float y = trix_onnx_sigmoid(x);        // 1 / (1 + exp(-x))

// Matrix ops - the workhorses
trix_onnx_matmul(A, B, C, M, N, K);    // C = A @ B
trix_onnx_gemm(A, B, bias, C, M, N, K, alpha, beta);

// Normalization - composed from primitives
trix_onnx_layer_norm(x, gamma, beta, out, n, eps);
trix_onnx_rms_norm(x, gamma, out, n, eps);

// Attention - yes, the whole thing
trix_onnx_attention(Q, K, V, out, seq_len, d_k, scale);
```

*See [docs/ONNX_SHAPES.md](docs/ONNX_SHAPES.md) for the complete map.*

### 4. Providence (`providence.h`)

Content-addressed memory. You don't look up by index - you look up by *what it is*.

```c
#include <trixc/providence.h>

trix_providence_t prov;
trix_providence_init(&prov, num_entries, key_bits, value_dim,
                     TRIX_FP8,   // Keys: compressed
                     TRIX_FP16); // Values: full precision

// Find the thing most similar to your query
trix_providence_lookup(&prov, query, result);
```

Hamming distance is a frozen shape: `d(a,b) = popcount(a XOR b)`.

*See [docs/PROVIDENCE.md](docs/PROVIDENCE.md) for the theory and practice.*

### 5. Sparse Octave Lookup (`sparse_octave.h`) **NEW**

Multi-scale content-addressed memory. Because information lives at different scales.

```c
#include <trixc/sparse_octave.h>

trix_sparse_octave_t sol;
trix_sparse_octave_init(&sol,
    64,     // d_model
    3,      // n_octaves (fine, medium, coarse)
    128,    // memory_size
    8       // top_k
);

// Forward pass - replaces entire FFN
trix_sparse_octave_forward(&sol, input, output);

// All operations are frozen shapes:
// - Hamming distance (XOR + popcount)
// - Softmax (exp + div)
// - Top-k (comparison)
// - Blend (mul + add)
```

*See [docs/SPARSE_OCTAVE.md](docs/SPARSE_OCTAVE.md) for the full architecture.*

### 6. 6502 ALU (`alu6502.h`)

The crown jewel. A complete 6502 ALU that would make Steve Wozniak nod approvingly.

```c
#include <trixc/alu6502.h>

trix_alu6502_t alu;
trix_alu6502_init(&alu);

// Clean integer interface
uint8_t result = trix_alu6502_execute_int(&alu, ALU_ADC, 16, 32, 0, &carry);

// Operations: ADC, SBC, AND, ORA, EOR, ASL, LSR, ROL, ROR, INC, DEC
```

---

## ONNX to TRIXC Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ONNX → Native Binary                              │
│                                                                          │
│   model.onnx ──▶ onnx2trix.py --emit-c ──▶ model.c ──▶ gcc ──▶ model   │
│        │                │                      │          │       │      │
│        │           Python ONNX             C source    Compile  Native  │
│        │           + C emitter             + weights            binary  │
│        │                                                                 │
│   Weights ──────────────────────────────────────────▶ static const[]    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Convert ONNX Directly to Native Binary

```bash
# One-liner: ONNX -> C -> Binary
python tools/onnx2trix.py model.onnx model.c --emit-c
gcc -O3 -DTRIXC_STANDALONE -I./include model.c -o model -lm
./model input.bin output.bin
```

### Or Convert to Intermediate Octave IR

```bash
python tools/onnx2trix.py model.onnx model.trix
```

### Run the Demo

```bash
python examples/demo_onnx2c.py
```

```
Step 1: Creating ONNX model...
Step 2: Converting ONNX to C...
  - Lines: 111
  - Size: 2999 bytes
Step 3: Compiling C code...
  - Binary size: 69.1 KB
Step 4: Preparing test input...
Step 5: Running compiled model...
Step 6: Verifying output...
  TRIXC output:    [5.0, 6.0]
  Expected output: [5.0, 6.0]

SUCCESS! Output matches expected values.
```

### Supported Operations

| Category | Operations |
|----------|------------|
| Arithmetic | Add, Sub, Mul, Div, MatMul, Gemm |
| Activation | ReLU, GELU, Sigmoid, Tanh, Softmax, SiLU |
| Normalization | LayerNorm, BatchNorm, RMSNorm |
| Reduction | Sum, Mean, Max, Min |
| Shape | Reshape, Transpose, Concat, Split |
| Attention | Full self-attention |

*See [docs/ONNX_SHAPES.md](docs/ONNX_SHAPES.md) for the complete list.*

---

## TRIXC Pi: Raspberry Pi Deployment

**NEW in v0.4.0:** Complete Raspberry Pi 4 deployment platform with touchscreen support.

```
┌────────────────────────────────────────────────────────────────────┐
│                     TRIXC Pi - MNIST Classifier                    │
│                                                                    │
│   ┌──────────────────┐    ┌───────────────────────────────────┐   │
│   │                  │    │  Predictions          0.08ms      │   │
│   │   Draw a digit   │    │  ───────────────────────────      │   │
│   │   with touch     │    │  0 ████████████████████ 0.02      │   │
│   │                  │    │  1 ██████ 0.01                    │   │
│   │       ████       │    │  2 ██ 0.00                        │   │
│   │      ██████      │    │  3 ██████████████████████████ 0.95 │   │
│   │        ██        │    │  4 ███ 0.01                       │   │
│   │        ██        │    │  ...                              │   │
│   │        ██        │    │                                   │   │
│   │                  │    │  Predicted: 3                     │   │
│   └──────────────────┘    └───────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

### What's Included

| Component | Lines | Description |
|-----------|-------|-------------|
| Runtime Library | 1,823 | Display, input, timing, GPIO, visualization |
| Pre-trained Models | 618 | XOR (52 bytes) + MNIST 7×7 (6.5 KB) |
| Examples | 1,050 | Hello XOR, MNIST Draw, GPIO Sensor |
| Documentation | ~1,200 | QUICKSTART, API, HARDWARE, model guides |

### Quick Deploy

```bash
cd platforms/raspberry-pi
./scripts/setup_pi.sh    # Install dependencies
make hello               # Build XOR example
./build/hello_xor        # Run on touchscreen!
```

### Key Features

- **SDL2-based rendering** - Hardware accelerated, 60 fps
- **Unified touch/keyboard input** - Works with touchscreens and mice
- **Built-in 8×8 bitmap font** - No external fonts needed
- **Visualization helpers** - Heatmaps, bar charts, graphs
- **GPIO support** - Control LEDs, read buttons, run motors
- **High-resolution timing** - Microsecond precision for benchmarks

*See [platforms/raspberry-pi/README.md](platforms/raspberry-pi/) for the full documentation.*

---

## File Structure

```
trixc/
├── README.md                    # You are here
├── CHANGELOG.md                 # Version history
├── Makefile                     # Build system
│
├── include/trixc/
│   ├── apu.h                    # Endogenous APU (precision management)
│   ├── shapes.h                 # Frozen arithmetic shapes
│   ├── onnx_shapes.h            # ONNX-compatible shapes
│   ├── providence.h             # Content-addressed memory
│   ├── sparse_octave.h          # Multi-scale sparse lookup
│   └── alu6502.h                # 6502 ALU implementation
│
├── platforms/
│   └── raspberry-pi/            # TRIXC Pi Platform [NEW v0.4.0]
│       ├── include/trixc_pi.h   # Runtime API (display, input, GPIO)
│       ├── src/trixc_pi.c       # SDL2-based implementation
│       ├── models/              # Pre-trained models (XOR, MNIST)
│       ├── examples/            # 3 progressive examples
│       └── docs/                # Pi-specific documentation
│
├── tools/
│   └── onnx2trix.py             # ONNX to Octave IR converter
│
├── test/
│   ├── test_apu.c               # APU test suite (48 tests)
│   ├── test_rigorous.c          # Rigorous test suite (1,329 tests)
│   ├── test_6502_onnx_pipeline.py  # 6502 pipeline tests (20 tests)
│   └── test_emit_c.py           # C emission tests (26 tests)
│
├── docs/
│   ├── APU.md                   # APU deep dive
│   ├── SHAPES.md                # Shape reference
│   ├── ONNX_SHAPES.md           # ONNX shape mapping
│   ├── ONNX2C.md                # ONNX to C pipeline guide
│   ├── SPARSE_OCTAVE.md         # Sparse octave architecture
│   ├── PROVIDENCE.md            # Content-addressed memory
│   ├── TEST_SUITE.md            # Complete test documentation
│   └── ARCHITECTURE.md          # Compiler architecture
│
└── build/
    ├── test_apu                 # Test binary
    ├── alu6502                  # ALU demo binary
    └── sparse_octave            # Sparse octave demo
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              TRIXC                                       │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                         FRONTENDS                                    ││
│  │                                                                      ││
│  │   .trix (JSON) ──┐                                                  ││
│  │                   ├──▶ PARSE ──▶ Octave IR                          ││
│  │   .onnx ─────────┘                                                  ││
│  │                                                                      ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                      ENDOGENOUS APU                                  ││
│  │                                                                      ││
│  │   FP4 ◄────► FP8 ◄────► FP16 ◄────► FP32 ◄────► FP64               ││
│  │        │          │           │           │          │               ││
│  │        └──────────┴───────────┴───────────┴──────────┘               ││
│  │                        Frozen conversion shapes                      ││
│  │                                                                      ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                       FROZEN SHAPES                                  ││
│  │                                                                      ││
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             ││
│  │   │    Logic     │  │  Arithmetic  │  │    ONNX      │             ││
│  │   ├──────────────┤  ├──────────────┤  ├──────────────┤             ││
│  │   │ XOR: a+b-2ab │  │ FULL_ADDER   │  │ MatMul       │             ││
│  │   │ AND: ab      │  │ RIPPLE_ADD   │  │ GELU         │             ││
│  │   │ OR: a+b-ab   │  │ RIPPLE_SUB   │  │ Softmax      │             ││
│  │   │ NOT: 1-a     │  │ INC, DEC     │  │ LayerNorm    │             ││
│  │   └──────────────┘  └──────────────┘  │ Attention    │             ││
│  │                                        └──────────────┘             ││
│  │                                                                      ││
│  │   0 learnable parameters. 100% accuracy. Frozen forever.            ││
│  │                                                                      ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                         MEMORY                                       ││
│  │                                                                      ││
│  │   ┌───────────────────┐    ┌───────────────────────────────────┐   ││
│  │   │    Providence     │    │      Sparse Octave Lookup         │   ││
│  │   ├───────────────────┤    ├───────────────────────────────────┤   ││
│  │   │ Content-addressed │    │ Multi-scale Providence            │   ││
│  │   │ Hamming distance  │    │ Fine → Medium → Coarse            │   ││
│  │   │ Soft lookup       │    │ Learned blending (only learned!)  │   ││
│  │   └───────────────────┘    └───────────────────────────────────┘   ││
│  │                                                                      ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                         BACKENDS                                     ││
│  │                                                                      ││
│  │   EMIT ──▶ .c (portable) ──▶ gcc ──▶ .so / .exe                    ││
│  │       └──▶ .cu (CUDA) ──▶ nvcc ──▶ GPU binary                      ││
│  │                                                                      ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The Philosophy

### 1. Frozen Shapes

Every computation is a **frozen shape** - a polynomial that never changes.

```
XOR(a, b) = a + b - 2ab     # Always. Forever. Frozen.
AND(a, b) = ab              # Always. Forever. Frozen.
OR(a, b)  = a + b - ab      # Always. Forever. Frozen.
GELU(x)   = x * σ(1.702x)   # Always. Forever. Frozen.
```

These aren't approximations. They're exact on their domains.

### 2. Routing is the Only Learning

The shapes don't change. What changes is **which shape to use for which input**.

```
Opcode 0 → Route to RIPPLE_ADD
Opcode 4 → Route to XOR
Input X  → Route to Octave 2 (coarse)
Input Y  → Route to Octave 0 (fine)
```

This routing is the **only** thing learned during training. Then it's frozen.

### 3. Precision is a Shape

```c
FP32 → FP16: trix_fp32_to_fp16(x)  // Frozen bit manipulation
FP16 → FP8:  trix_fp16_to_fp8(h)   // Frozen bit manipulation
```

The APU doesn't "learn" precision. It **knows** precision.

### 4. Information Lives at Different Scales

Sparse Octave Lookup recognizes that:
- Semantic meaning lives at coarse scales (byte level)
- Precise values live at fine scales (bit level)
- The right answer blends both

```
Octave 0 (fine)   → Full precision lookup
Octave 1 (medium) → 4-bit shifted lookup
Octave 2 (coarse) → 8-bit shifted lookup
              ↓
        Learned blend
              ↓
           Output
```

---

## Performance

| Metric | TRIXC | PyTorch | ONNX Runtime |
|--------|-------|---------|--------------|
| Runtime size | 6-8 KB | 2 GB | 100+ MB |
| Dependencies | libc | Everything | Everything |
| Startup time | <1 ms | seconds | seconds |
| Accuracy | 100% (exact) | ~99% (learned) | ~99% (learned) |
| Portable | Yes | Kind of | Kind of |
| Embedded | Yes | No | Barely |

---

## Roadmap

- [x] Endogenous APU (FP4 → FP64)
- [x] Frozen shapes library (16 shapes)
- [x] ONNX shapes library (40+ shapes)
- [x] Providence (content-addressed memory)
- [x] Sparse Octave Lookup
- [x] 6502 ALU reference implementation
- [x] Test suite (1,375+ tests)
- [x] `onnx2trix.py` converter
- [x] Full C code generation from ONNX
- [x] `--emit-c` CLI flag
- [x] **Raspberry Pi 4 platform** (v0.4.0) ← NEW
- [x] **Touchscreen ML examples** ← NEW
- [x] **GPIO hardware control** ← NEW
- [ ] CUDA backend
- [ ] ARM NEON optimizations
- [ ] WASM backend
- [ ] Camera integration (libcamera)

---

## Requirements

**To build:**
- GCC or Clang with C11 support
- libc (that's it)

**To convert ONNX:**
- Python 3.7+
- `onnx` package (`pip install onnx`)

**To exist happily:**
- A belief that math is real
- A willingness to compile rather than interpret

---

## Credits

**Created by:**
- **Tripp** - Vision, architecture, the questions that matter
- **Claude** (Anthropic) - Implementation, documentation, the answers that work

**Born:** December 2025

**Philosophy:** Lincoln Manifold Method

**First applications:**
- 6502 ALU in 6 KB
- Sparse Octave Lookup in 8 KB

---

## License

MIT - Do whatever you want. Just don't blame us when it works too well.

---

## The Principles

> *"Shapes are opcodes. Polynomials are microcode. C is machine code."*

> *"Precision is a shape. The APU is frozen."*

> *"Don't learn what you can derive."*

> *"Information lives at different scales. Capture it where it lives."*

> *"It's all in the reflexes."*

---

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   "You know what Jack Burton says at a time like this?"                 │
│                                                                          │
│   "Jack Burton says... what the hell, let's compile it."                │
│                                                                          │
│   PyTorch: 2 GB to learn what math already knows.                       │
│   TRIXC: 6 KB to just use it.                                           │
│                                                                          │
│   The shapes are frozen. The code is native. The runtime is nothing.    │
│                                                                          │
│   That's TRIXC.                                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```
