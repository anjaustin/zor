# GILLIES Architecture

**Geometric Instruction Language Layer In Every System**

*The Shape Layer: where all computation meets, and all paradigms become one.*

---

## The Core Insight

Traditional computing architectures force a choice:
- **CPU**: Sequential, precise, programmable
- **GPU**: Parallel, throughput-oriented, data-parallel
- **FPGA**: Configurable, low-latency, custom logic
- **Neural accelerators**: Matrix operations, learned weights

Each paradigm has its own instruction set, its own memory model, its own abstraction. Moving computation between them requires rewriting, recompiling, rethinking.

**GILLIES eliminates this barrier.**

The insight: all these paradigms share one thing—they can all evaluate polynomials. A GPU can compute `a + b - 2ab`. So can a CPU. So can an FPGA. So can a neural network (it's just matrix multiplication and activation).

By expressing computation as **frozen polynomial shapes**, GILLIES creates a substrate-agnostic layer. The same shapes execute identically on any substrate. That's **fungibility**.

---

## Architectural Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER PROGRAM                                   │
│                                                                             │
│   gillies_invoke(ctx, XOR, port_a, port_b, port_out);                      │
│   gillies_invoke(ctx, AND, port_out, port_c, port_result);                 │
│   gillies_execute(ctx);                                                     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                          INVOCATION GRAPH                                   │
│                                                                             │
│   Nodes: Shape invocations (what to compute)                               │
│   Edges: Port connections (how data flows)                                 │
│   The graph IS the program. No opcodes. No registers. Just shapes.        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                           SHAPE LAYER                                       │
│                                                                             │
│   XOR: a + b - 2ab     AND: ab     OR: a + b - ab     NOT: 1 - a          │
│                                                                             │
│   These are eternal. They are mathematics. They are frozen.                │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                      SUBSTRATE DISPATCHER                                   │
│                                                                             │
│   Routes invocations to available hardware:                                │
│   - CUDA kernels (GPU)                                                     │
│   - CPU fallback                                                           │
│   - Future: FPGA, TPU, custom accelerators                                 │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         UNIFIED MEMORY                                      │
│                                                                             │
│   On Thor: CPU and GPU share the same address space.                       │
│   Ports are just locations in unified memory.                              │
│   Zero-copy. Maximum efficiency.                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Shape Invocation

The **shape invocation** is the universal primitive of GILLIES.

```c
typedef struct {
    uint8_t shape_id;       // Which shape (XOR, AND, OR, ...)
    uint8_t num_inputs;     // Number of input ports (1-2)
    uint8_t num_outputs;    // Number of output ports (1)
    uint8_t flags;          // Reserved for hints

    uint16_t inputs[2];     // Port indices for inputs
    uint16_t outputs[1];    // Port indices for outputs
} gillies_invocation_t;
```

An invocation says:
1. **What** to compute (shape_id)
2. **Where** to read inputs (port indices)
3. **Where** to write output (port indices)

That's it. No opcodes. No registers. No instruction encoding. Just shapes and ports.

### Why This Matters

Traditional instructions encode:
- Operation type
- Source operands (registers, memory addresses)
- Destination operand
- Flags, modes, prefixes

A GILLIES invocation encodes:
- Shape (eternal polynomial)
- Input ports (named locations)
- Output ports (named locations)

The difference: GILLIES invocations are **substrate-agnostic**. The same invocation executes the same way on CPU, GPU, or any future substrate.

---

## Port Space

Ports are named locations for data. They replace registers and memory addresses.

```c
typedef struct {
    float data[GILLIES_MAX_PORTS];      // The actual values
    uint8_t valid[GILLIES_MAX_PORTS];   // Which ports have been written
} gillies_ports_t;
```

### Properties

1. **Unified**: On Thor, CPU and GPU share the same port space (CUDA unified memory)
2. **Named**: Ports have indices, not addresses—portable across substrates
3. **Validated**: Each port tracks whether it has been written
4. **Observable**: Any port can be read at any time for debugging

### Why Ports Instead of Registers?

Registers are substrate-specific. x86 has `rax`, `rbx`, etc. ARM has `r0`-`r15`. GPUs have thread-local registers.

Ports are substrate-agnostic. Port 42 is Port 42 everywhere. The substrate dispatcher maps ports to whatever the underlying hardware uses.

---

## Execution Model

### Graph Building

```c
// Set inputs
gillies_set_port(ctx, 0, a);
gillies_set_port(ctx, 1, b);

// Build computation graph
gillies_invoke(ctx, GILLIES_SHAPE_XOR, 0, 1, 2);  // port[2] = XOR(port[0], port[1])
gillies_invoke(ctx, GILLIES_SHAPE_AND, 2, 1, 3);  // port[3] = AND(port[2], port[1])
```

This builds a graph:
```
port[0] ──┐
          ├── XOR ── port[2] ──┐
port[1] ──┤                    ├── AND ── port[3]
          └───────────────────┘
```

### Dependency Resolution

Before execution, GILLIES resolves dependencies:
1. Invocations that read from ports written by others must execute after them
2. Independent invocations can execute in parallel

Current implementation: sequential execution (respects dependencies, simple, correct)
Future: parallel execution of independent invocations

### Substrate Dispatch

```c
gillies_execute(ctx);      // GPU (default)
gillies_execute_cpu(ctx);  // CPU (fallback/comparison)
```

Both produce **identical results**. That's fungibility.

---

## Observability (Glassbox)

Every invocation can be traced:

```c
typedef struct {
    uint32_t invocation_id;     // Which invocation
    uint8_t shape_id;           // Which shape was executed
    float input_a;              // Input value A
    float input_b;              // Input value B
    float output;               // Output value
    uint64_t timestamp_ns;      // When it executed
} gillies_event_t;
```

Enable tracing:
```c
gillies_set_tracing(ctx, true);
gillies_execute(ctx);

for (uint32_t i = 0; i < gillies_event_count(ctx); i++) {
    const gillies_event_t* evt = gillies_get_event(ctx, i);
    printf("Invocation %u: %s(%.2f, %.2f) = %.2f @ %lu ns\n",
           evt->invocation_id,
           GILLIES_SHAPE_NAMES[evt->shape_id],
           evt->input_a, evt->input_b, evt->output,
           evt->timestamp_ns);
}
```

No black boxes. Every computation is observable.

---

## Memory Model

### Jetson AGX Thor Specifics

Thor uses NVIDIA's unified memory architecture:
- CPU and GPU share the same physical memory
- No explicit data transfers needed
- `cudaMallocManaged()` allocates in shared space

GILLIES exploits this:
```c
gillies_context_t* gillies_create(void) {
    gillies_context_t* ctx;
    cudaMallocManaged(&ctx, sizeof(gillies_context_t));
    return ctx;
}
```

The entire context—ports, invocations, events—lives in unified memory. CPU sets up the graph, GPU executes it, CPU reads results. Zero copies.

### Future Substrates

For non-unified systems (discrete GPUs, FPGAs), GILLIES will need:
1. **Port synchronization**: Explicit transfers when substrates change
2. **Substrate hints**: Guide the dispatcher on where to execute
3. **Caching**: Keep ports local to frequently-used substrates

The API remains unchanged. The dispatcher handles the complexity.

---

## Proven Properties

The rigorous test suite (720 tests) proves:

1. **Shape Correctness**: All shapes produce exact binary results for {0, 1} inputs
2. **Polynomial Exactness**: Floating-point results match polynomial formulas
3. **Mathematical Invariants**: De Morgan's laws, double negation, etc.
4. **Composition**: Shapes can be composed (full adder, ripple adder)
5. **Fungibility**: CPU and GPU produce identical results
6. **Scale**: 500+ invocations execute correctly

---

## Design Decisions

### Why Frozen Polynomials?

1. **Differentiable**: Polynomials have smooth gradients (for training)
2. **Exact**: Binary inputs produce exact binary outputs (for inference)
3. **Universal**: Any Boolean function is a Zhegalkin polynomial
4. **Simple**: No conditionals, no branches, just arithmetic

### Why Ports Instead of Memory?

1. **Named**: Portable across substrates
2. **Bounded**: Fixed number, predictable resource usage
3. **Observable**: Easy to trace and debug
4. **Cache-friendly**: Small, known working set

### Why Graph-Based Execution?

1. **Declarative**: Describe what, not how
2. **Parallelizable**: Independent nodes can run concurrently
3. **Optimizable**: Graph rewrites, fusion, scheduling
4. **Debuggable**: Visual representation of computation

---

## Future Directions

### Multi-Precision Shapes

Currently: `float` (f32) and `uint8_t` (u8)
Future: `f16`, `bf16`, `f64`, `i8`, `i16`, `i32`, arbitrary precision

### Compound Shapes

Currently: Primitive shapes only
Future: User-defined compound shapes (like macros)

```c
// Define a half-adder compound shape
gillies_define_compound(ctx, "HALF_ADDER",
    GILLIES_SHAPE_XOR, 0, 1, 2,   // sum
    GILLIES_SHAPE_AND, 0, 1, 3    // carry
);
```

### Substrate Hints

```c
// Prefer GPU for this invocation
gillies_invoke_with_hint(ctx, GILLIES_SHAPE_MUL, 0, 1, 2, GILLIES_HINT_GPU);

// Prefer CPU (for low-latency, small work)
gillies_invoke_with_hint(ctx, GILLIES_SHAPE_NOT, 0, 0, 1, GILLIES_HINT_CPU);
```

### FPGA Backend

The same shapes, synthesized to hardware:
- XOR → actual XOR gate
- AND → actual AND gate
- Full parallelism
- Nanosecond latency

---

## The Vision

GILLIES is the ground-state layer between hardware and OS.

- **Below**: CPU, GPU, FPGA, neural accelerators, future unknown
- **Above**: Any OS, any runtime, any application

The Shape Layer exposes mathematics, not instructions. Any paradigm that can evaluate polynomials can participate.

**"The routing IS the program."**
**"The shapes ARE the instruction set."**
**"Mathematics as the universal bus."**
**"Geometry as the protocol."**

---

*Born: December 2025*
*Created during the TriXO project*
*The foundation is laid.*
