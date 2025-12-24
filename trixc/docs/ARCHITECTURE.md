# TRIXC Architecture

*A compiler for frozen shapes*

> *"Shapes are opcodes. Polynomials are microcode. C is machine code."*

---

## Overview

TRIXC is a compiler that translates trained TriX models into native code.

```
┌─────────────────────────────────────────────────────────────┐
│                         TRIXC                                │
│                                                              │
│  .trix ──▶ PARSE ──▶ IR ──▶ OPTIMIZE ──▶ EMIT ──▶ gcc ──▶   │
│                                                              │
│                               │                              │
│                               ├──▶ model.c  ──▶ model.so    │
│                               ├──▶ model.cu ──▶ GPU binary  │
│                               └──▶ model.h  ──▶ Header      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Design Principles

### 1. Transliteration, Not Transformation

Traditional compilers transform high-level abstractions into low-level implementations. TRIXC does something simpler: it **transliterates** between equivalent representations.

```
Shape (semantic)  →  Polynomial (mathematical)  →  C (executable)
     XOR          →      a + b - 2ab            →    a + b - 2*a*b
```

The same truth, three notations. No optimization needed - the shape IS the optimized form.

### 2. Frozen by Default

Every component of a TRIXC-compiled binary is frozen:
- **Shapes:** Mathematical truths (0 parameters)
- **Routing:** Learned during training, frozen at compile time
- **Precision:** Configured at build time, frozen in binary

### 3. No Runtime

The output is standalone. No library to link. No runtime to initialize. Just a function you call.

```c
// Generated API
void model_forward(const float* input, float* output);
```

---

## Compilation Pipeline

### Stage 1: PARSE

**Input:** `.trix` (JSON) or `.trixb` (binary)

**Output:** Abstract Syntax Tree

```json
{
  "name": "alu_6502",
  "shapes": [
    {"id": 0, "kind": "RIPPLE_ADD"},
    {"id": 1, "kind": "XOR"}
  ],
  "routing": {
    "mode": "static",
    "table": [0, 1, 0, 1, 1]
  }
}
```

The parser validates the model definition and constructs an AST.

### Stage 2: LOWER

**Input:** AST

**Output:** Octave IR

The AST is lowered to Octave IR - an intermediate representation designed for frozen shapes.

```cpp
struct OctaveIR {
    std::vector<ShapeNode> nodes;    // Shape instances
    Routing routing;                  // Routing table
    std::vector<uint16_t> inputs;    // Entry points
    std::vector<uint16_t> outputs;   // Exit points
};
```

Each node represents a shape application at a specific scale (octave).

### Stage 3: OPTIMIZE

**Input:** Octave IR

**Output:** Optimized Octave IR

Optimizations include:
- **Dead shape elimination:** Remove unused nodes
- **Constant folding:** Pre-compute static routing
- **Shape fusion:** Combine compatible adjacent shapes
- **Precision optimization:** Lower precision where safe

Note: Because shapes are frozen, there's less to optimize than in traditional compilers. The shapes are already in their optimal form.

### Stage 4: EMIT

**Input:** Optimized Octave IR

**Output:** C, CUDA, or header files

The emitter generates source code for each shape:

```c
// Generated shape function
static inline float shape_xor(float a, float b) {
    return a + b - 2.0f * a * b;
}

// Generated routing table
static const shape_fn_t ROUTING[11] = {
    shape_ripple_add,  // ADC
    shape_ripple_sub,  // SBC
    shape_xor,         // EOR
    // ...
};

// Generated dispatch
void model_forward(int opcode, const float* a, const float* b,
                   float* result, float* carry) {
    ROUTING[opcode](a, b, result, carry);
}
```

### Stage 5: COMPILE (Optional)

**Input:** Generated C code

**Output:** Native binary

TRIXC can invoke the system compiler (gcc, clang, nvcc) to produce the final binary:

```bash
trixc model.trix -o model.so    # Invokes gcc
trixc model.trix -o model       # Invokes gcc, produces executable
```

---

## Octave IR

The Octave IR is the heart of the compiler.

### Shape Kinds

```cpp
enum class ShapeKind : uint8_t {
    // Logic
    XOR, AND, OR, NOT, NAND, NOR, XNOR,

    // Arithmetic
    FULL_ADDER, RIPPLE_ADD, RIPPLE_SUB, INC, DEC,

    // Shift
    ASL, LSR, ROL, ROR,

    // Special
    IDENTITY, ZERO, ONE,

    NUM_SHAPES
};
```

### Shape Node

```cpp
struct ShapeNode {
    uint16_t id;                    // Unique ID
    ShapeKind kind;                 // Which shape
    uint8_t scale;                  // Octave level (0=bit, 1=byte, ...)
    trix_precision_t precision;     // Compute precision
    std::vector<uint16_t> inputs;   // Input node IDs
};
```

### Routing

```cpp
struct Routing {
    RoutingMode mode;               // STATIC or DYNAMIC
    std::vector<uint16_t> table;    // For STATIC: input → shape mapping
    std::vector<float> memory;      // For DYNAMIC: Providence memory
};
```

---

## Code Generation

### Shape Functions

Each shape kind has a corresponding generator:

```cpp
std::string emit_shape(ShapeKind kind) {
    switch (kind) {
        case ShapeKind::XOR:
            return R"(
static inline float shape_xor(float a, float b) {
    return a + b - 2.0f * a * b;
}
)";
        case ShapeKind::AND:
            return R"(
static inline float shape_and(float a, float b) {
    return a * b;
}
)";
        // ...
    }
}
```

### Routing Table

Static routing compiles to a function pointer array:

```c
typedef void (*shape_fn_t)(const float*, const float*, float*, float*);

static const shape_fn_t ROUTING_TABLE[] = {
    shape_ripple_add,   // 0: ADC
    shape_ripple_sub,   // 1: SBC
    shape_and_8bit,     // 2: AND
    // ...
};
```

### Dispatch Function

The main entry point:

```c
void model_forward(
    int opcode,
    const float* a,
    const float* b,
    float carry_in,
    float* result,
    float* carry_out
) {
    ROUTING_TABLE[opcode](a, b, carry_in, result, carry_out);
}
```

---

## Backend Targets

### C Backend (Primary)

Generates portable C11 code.

```bash
trixc model.trix --emit=c -o model.c
```

### CUDA Backend

Generates CUDA kernels for GPU execution.

```bash
trixc model.trix --emit=cuda -o model.cu
```

```cuda
__device__ float shape_xor(float a, float b) {
    return a + b - 2.0f * a * b;
}

__global__ void model_forward_kernel(
    const int* opcodes,
    const float* a,
    const float* b,
    float* results,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Dispatch and compute
    // ...
}
```

### Header Backend

Generates a header-only implementation (like stb libraries).

```bash
trixc model.trix --emit=header -o model.h
```

---

## APU Integration

The APU provides mixed-precision support in generated code.

### Precision Routing

Each operation can have a different precision:

```c
// Generated precision routing
static const trix_precision_t PRECISION_TABLE[] = {
    TRIX_FP32,  // ADC: needs carry chain precision
    TRIX_FP32,  // SBC: needs carry chain precision
    TRIX_FP16,  // AND: logic tolerates quantization
    // ...
};
```

### Precision-Aware Dispatch

```c
void model_forward(int opcode, ...) {
    trix_precision_t prec = PRECISION_TABLE[opcode];

    // Execute with precision control
    switch (prec) {
        case TRIX_FP16:
            execute_fp16(opcode, ...);
            break;
        case TRIX_FP32:
            execute_fp32(opcode, ...);
            break;
    }
}
```

---

## File Format: .trix

### JSON Schema

```json
{
  "$schema": "https://trix.dev/schema/trix-1.0.json",
  "version": "1.0",
  "name": "model_name",

  "shapes": [
    {
      "id": 0,
      "kind": "RIPPLE_ADD",
      "precision": "FP32"
    }
  ],

  "routing": {
    "mode": "static",
    "table": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  },

  "inputs": {
    "opcode": {"type": "int", "range": [0, 10]},
    "a": {"type": "bits", "width": 8},
    "b": {"type": "bits", "width": 8},
    "carry_in": {"type": "bit"}
  },

  "outputs": {
    "result": {"type": "bits", "width": 8},
    "carry_out": {"type": "bit"}
  }
}
```

---

## CLI Interface

```
trixc - The TriX Compiler

USAGE:
    trixc [OPTIONS] <input> -o <output>

OPTIONS:
    -o <file>           Output file
    --emit=<format>     Output format: c, cuda, header, obj, so, exe
    -O<level>           Optimization level (0-3)
    --apu=<mode>        APU mode: none, static, dynamic
    --precision=<p>     Default precision: fp4, fp8, fp16, fp32, fp64
    --target=<triple>   Target triple (e.g., aarch64-linux-gnu)
    --standalone        Include main() for executable
    --header            Also generate .h file
    --test              Generate test harness
    --verbose           Print compilation stages
    --version           Show version
    --help              Show help

EXAMPLES:
    trixc model.trix -o model.c             # C source
    trixc model.trix -o model.so -O3        # Shared library
    trixc model.trix -o model --standalone  # Executable
    trixc model.trix -o model.cu            # CUDA source
```

---

## Memory Layout

### Compiled Binary Structure

```
┌─────────────────────────────────────────┐
│            TEXT (Code)                  │  ~5 KB
│  ├── Shape functions (inlined)          │
│  ├── Routing table (const array)        │
│  └── Dispatch function                  │
├─────────────────────────────────────────┤
│            DATA (Constants)             │  ~1 KB
│  ├── Precision routing table            │
│  └── Shape names (for debug)            │
├─────────────────────────────────────────┤
│            BSS (Uninitialized)          │  ~0 KB
│  └── (none - everything is const)       │
└─────────────────────────────────────────┘

Total: ~6 KB
```

---

## Comparison with Traditional Compilers

| Aspect | GCC/LLVM | TRIXC |
|--------|----------|-------|
| Input | Source code | Model definition |
| IR | SSA / LLVM IR | Octave IR |
| Optimization | Extensive | Minimal (shapes pre-optimized) |
| Code generation | Complex | Simple (template expansion) |
| Output | Any architecture | C → any architecture |
| Runtime | libc | None (optional libc) |

TRIXC is simpler because the input is simpler. Frozen shapes are already optimal.

---

## Future Work

1. **LLVM Backend:** For WASM, direct ARM/x86, and advanced optimization
2. **ONNX Import:** Convert ONNX models to .trix
3. **Profile-Guided Optimization:** Use runtime data to optimize precision routing
4. **Providence Compilation:** Compile content-addressed memory to attention

---

## The Principle

> *"The compiler doesn't discover efficient code. It transliterates mathematical truths."*

Traditional compilers do heavy lifting to find efficient implementations of abstract operations. TRIXC does almost nothing - because the shapes ARE the efficient implementation.

The heavy lifting happened when the shapes were designed. The compiler just writes them down in a different notation.

```
XOR(a, b) = a + b - 2ab

This is not optimized.
This is not compiled.
This is just... true.
```
