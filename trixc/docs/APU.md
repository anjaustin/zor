# The Endogenous APU

*Arithmetic Processing Unit for Mixed-Precision Computation*

> *"Precision is a shape. The APU is frozen."*

---

## Overview

The Endogenous APU manages mixed-precision computation without external libraries. All precision conversions and computations are implemented as **frozen shapes** - pure arithmetic that compiles to native code.

**Endogenous** = originating from within. The APU is not a library you link against. It's code that gets compiled into your binary.

---

## Precision Levels

| Level | Type | Bits | Format | Use Case |
|-------|------|------|--------|----------|
| `TRIX_FP4` | `trix_fp4_t` | 4 | E2M1 | Routing decisions, sketches |
| `TRIX_FP8` | `trix_fp8_t` | 8 | E4M3 | Weights, activations |
| `TRIX_FP16` | `trix_fp16_t` | 16 | IEEE 754 | General computation |
| `TRIX_FP32` | `float` | 32 | IEEE 754 | Accumulation |
| `TRIX_FP64` | `double` | 64 | IEEE 754 | Critical paths |

---

## FP4: Extreme Compression

**Format:** `[sign:1][exp:2][mant:1]`

Only 16 possible values (8 positive, 8 negative):
```
0, 0.5, 1, 1.5, 2, 3, 4, 6  (and negatives)
```

**Use case:** When you only need "big/medium/small/zero" - perfect for routing decisions.

```c
trix_fp4_t val = trix_fp32_to_fp4(1.0f);  // Compresses to 4 bits
float back = trix_fp4_to_fp32(val);        // ~1.0f (quantized)
```

---

## FP8: ML Standard

**Format:** `[sign:1][exp:4][mant:3]`

Range: ±448, precision: ~1%

This is the standard format for ML weights and activations. Good enough for most neural network operations.

```c
trix_fp8_t val = trix_fp32_to_fp8(2.5f);
float back = trix_fp8_to_fp32(val);  // ~2.5f
```

---

## Conversion Shapes

All conversions are **frozen shapes** - pure arithmetic on bit patterns.

### FP32 → FP16

```c
static inline trix_fp16_t trix_fp32_to_fp16(float x) {
    uint32_t bits;
    __builtin_memcpy(&bits, &x, sizeof(bits));

    uint16_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;  // Rebias
    uint32_t mant = (bits >> 13) & 0x03FF;           // Truncate

    // Handle underflow/overflow
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;

    return sign | ((uint16_t)exp << 10) | mant;
}
```

This is not a cast. It's **arithmetic on the bit pattern** - a frozen shape.

### FP16 → FP32

```c
static inline float trix_fp16_to_fp32(trix_fp16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = (h & 0x03FF) << 13;

    if (exp == 0) {
        // Zero or subnormal
        if (mant == 0) return sign_to_float(sign);
        // Normalize subnormal...
    }

    exp = exp - 15 + 127;  // Rebias to FP32

    return bits_to_float(sign | (exp << 23) | mant);
}
```

---

## APU Context

The APU tracks precision usage for profiling and optimization.

```c
typedef struct {
    trix_precision_t routing[TRIX_NUM_OPS];       // Per-op precision
    uint64_t op_counts[TRIX_NUM_OPS];             // Operation counts
    uint64_t precision_counts[TRIX_NUM_PRECISIONS]; // Precision usage
} trix_apu_t;
```

### Initialization

```c
trix_apu_t apu;
trix_apu_init(&apu);

// Optionally customize precision routing
trix_apu_set_routing(&apu, TRIX_OP_XOR, TRIX_FP8);
trix_apu_set_routing(&apu, TRIX_OP_ACCUMULATE, TRIX_FP32);
```

### Execution

```c
float result = trix_apu_execute(&apu, TRIX_OP_XOR, a, b,
                                 TRIX_FP16,   // Input precision
                                 TRIX_FP16);  // Output precision
```

The APU:
1. Looks up the compute precision for XOR in the routing table
2. Executes the frozen XOR shape
3. Applies precision truncation based on compute precision
4. Updates statistics

### Statistics

```c
trix_apu_print_stats(&apu);
```

Output:
```
APU Statistics:
  Operations:
    XOR: 1000 ops
    AND: 500 ops
  Precision usage:
    FP16: 1500 ops
```

---

## Default Precision Routing

The APU ships with sensible defaults:

| Operation | Default Precision | Reason |
|-----------|-------------------|--------|
| XOR | FP16 | Logic tolerates quantization |
| AND | FP16 | Logic tolerates quantization |
| OR | FP16 | Logic tolerates quantization |
| NOT | FP16 | Logic tolerates quantization |
| ADD | FP16 | Standard arithmetic |
| SUB | FP16 | Standard arithmetic |
| MUL | FP16 | Standard arithmetic |
| ACCUMULATE | **FP32** | Needs headroom to avoid overflow |

---

## Integration with Shapes

The APU works with all frozen shapes:

```c
#include <trixc/apu.h>
#include <trixc/shapes.h>

trix_apu_t apu;
trix_apu_init(&apu);

// Full adder with precision control
float sum, carry;
trix_shape_full_adder_p(a, b, c_in, &sum, &carry, TRIX_FP16);

// Ripple add with precision control
trix_shape_ripple_add_p(a_bits, b_bits, carry_in,
                         result_bits, &carry_out, TRIX_FP32);
```

---

## Integration with Providence

Providence uses the APU for multi-precision lookups:

```c
#include <trixc/providence.h>

trix_providence_t prov;
trix_providence_init(&prov, 1000, 64, 32,
                      TRIX_FP8,    // Key comparison at low precision (fast)
                      TRIX_FP16);  // Value retrieval at higher precision

// APU-aware lookup
trix_providence_lookup_apu(&apu, &prov, query, result);
```

The key insight: **Route at low precision, retrieve at high precision.**

- Key comparison: FP8 is enough (you're just finding the nearest neighbor)
- Value retrieval: FP16/FP32 for accuracy

---

## Memory Layout

All precision types are designed for efficient storage:

```c
sizeof(trix_fp4_t)  = 1 byte  (4 bits used)
sizeof(trix_fp8_t)  = 1 byte
sizeof(trix_fp16_t) = 2 bytes
sizeof(float)       = 4 bytes
sizeof(double)      = 8 bytes
```

For bulk storage, FP4 values can be packed:
```c
// 2 FP4 values per byte
uint8_t packed = (fp4_a << 4) | fp4_b;
```

---

## No External Dependencies

The APU is pure C11:
- No CUDA
- No cuDNN
- No hardware-specific intrinsics
- No external libraries

Just `<stdint.h>`, `<stdio.h>`, `<math.h>`.

It compiles with:
```bash
gcc -std=c11 -O3 your_code.c -lm
```

---

## The Principle

The APU is not a runtime. It's not a library. It's a **set of frozen shapes** that get compiled into your binary.

When you call `trix_fp32_to_fp16(x)`, there's no function call overhead. It's inlined arithmetic on bit patterns.

When you call `trix_apu_execute()`, the routing table lookup compiles to a simple array access.

**The entire mixed-precision stack is ~2 KB of compiled code.**

---

## Example: Full Pipeline

```c
#include <trixc/apu.h>
#include <trixc/shapes.h>
#include <trixc/alu6502.h>

int main() {
    // Initialize APU with custom routing
    trix_alu6502_t alu;
    trix_alu6502_init(&alu);

    // Override precision for logic ops (they're exact at FP8)
    trix_alu6502_set_precision(&alu, ALU_AND, TRIX_FP8);
    trix_alu6502_set_precision(&alu, ALU_ORA, TRIX_FP8);
    trix_alu6502_set_precision(&alu, ALU_EOR, TRIX_FP8);

    // Execute operations
    int carry;
    uint8_t result;

    result = trix_alu6502_execute_int(&alu, ALU_ADC, 100, 50, 0, &carry);
    printf("ADC: 100 + 50 = %d\n", result);  // 150

    result = trix_alu6502_execute_int(&alu, ALU_EOR, 0xFF, 0xAA, 0, &carry);
    printf("EOR: 0xFF ^ 0xAA = 0x%02X\n", result);  // 0x55

    // Print statistics
    trix_alu6502_print_stats(&alu);

    return 0;
}
```

Output:
```
ADC: 100 + 50 = 150
EOR: 0xFF ^ 0xAA = 0x55

6502 ALU Statistics:
  Operations:
    ADC: 1 ops @ FP32
    EOR: 1 ops @ FP8
  Precision usage:
    FP8: 1 ops
    FP32: 1 ops
```

---

## Summary

The Endogenous APU provides:

1. **5 precision levels** (FP4 → FP64)
2. **Frozen conversion shapes** (bit manipulation as arithmetic)
3. **Precision routing** (per-operation precision control)
4. **Statistics tracking** (for optimization)
5. **Zero dependencies** (pure C11)
6. **~2 KB compiled** (everything inlines)

> *"Precision is a dimension of the Octave space. The APU navigates it."*
