# Key Nodes: Shape Processor

## Core Insight
**Polynomial primitives use only coefficients {-2, -1, 0, 1, 2}.**

This means we don't need general multipliers. We need:
- Shift (×2)
- Negate (×-1)
- Zero (×0)
- Pass-through (×1)
- Accumulate (sum)

## The Four Primitives

| Op | Polynomial | Hardware |
|----|-----------|----------|
| XOR | a + b - 2ab | 2 add, 1 shift, 1 mul |
| AND | ab | 1 mul |
| OR | a + b - ab | 2 add, 1 mul |
| NOT | 1 - a | 1 sub |

All reduce to multiply-accumulate with tiny coefficients.

## Architecture Options

### Option A: Cascade Pipeline
```
[PE] → [PE] → [PE] → [PE]
```
- Fixed shape, maximum throughput
- One shape per chip
- Hardwired correctness

### Option B: Reconfigurable Routing (CGRA)
```
[PE] ⟷ [PE] ⟷ [PE]
  ⟷     ⟷     ⟷
[PE] ⟷ [PE] ⟷ [PE]
```
- Flexible, any shape
- Configuration overhead
- Basically an FPGA

### Option C: Time-Multiplexed
```
         ┌────────────────┐
IN ────→ │      PE        │ ────→ OUT
         └───────┬────────┘
                 │ feedback
                 └────────────────┘
```
- Single PE, multiple cycles
- Smallest silicon
- Slowest execution

### Option D: Dataflow Array (Recommended)
```
┌─────────────────────────────────────┐
│  SHAPE MEMORY (coefficients)        │
└──────────────┬──────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼          ▼          ▼
 [PE-0]     [PE-1]     [PE-2]    ...
    │          │          │
    └────→ CARRY NETWORK ←┘
               │
               ▼
           [OUTPUT]
```
- Configurable via shape memory
- Parallel execution
- Carry network for adder optimization

## Key Architectural Decisions

### 1. Polynomial Engine (PE) Design
```
Inputs:  a, b, c_in (3 binary inputs)
Config:  op_select (XOR/AND/OR/NOT)
Output:  result, c_out

Hardwired logic:
  XOR: a + b - 2ab
  AND: ab
  OR:  a + b - ab
  NOT: 1 - a
```

### 2. Carry Network
For adders, ripple carry is O(n) depth. Options:
- Ripple: Simple, slow, O(n)
- Carry-lookahead: Complex, fast, O(log n)
- Carry-select: Middle ground

Recommendation: Start with ripple, optimize later.

### 3. Shape Memory Format
```
struct Shape {
    uint16_t num_outputs;
    uint16_t num_intermediates;
    Term terms[];
};

struct Term {
    int8_t coefficient;    // -2 to 2
    uint8_t operation;     // XOR/AND/OR/NOT
    uint16_t input_a;      // Index
    uint16_t input_b;      // Index (or intermediate)
    uint16_t output;       // Where to store result
};
```

### 4. Execution Model
```
1. LOAD:  Shape coefficients → Shape Memory
2. BIND:  Input bits → Input Registers
3. EVAL:  Parallel polynomial evaluation
4. READ:  Output Registers → Result
```

## Comparison to Existing Architectures

| Architecture | Optimized For | Our Difference |
|--------------|---------------|----------------|
| CPU | General compute | No instruction decode overhead |
| GPU | Dense float SIMD | Sparse binary, tiny coefficients |
| TPU | Matrix multiply | Polynomial structure, not matrices |
| FPGA | Arbitrary logic | Polynomial abstraction, not gates |
| CGRA | Coarse dataflow | Native polynomial primitives |

## Unique Value Propositions

1. **No instruction fetch/decode**: Coefficients ARE the program
2. **Proven correctness**: Hardware built from verified primitives
3. **Optimal for shapes**: Hardwired for exactly our use case
4. **Binary specialization**: Exploits a∈{0,1} constraints

## Killer Application
**Run a frozen CPU as a shape on the shape processor.**

- Freeze 6502 to polynomial form
- Load as shape
- Execute at hardware speed
- CPU running a CPU, all the way down
