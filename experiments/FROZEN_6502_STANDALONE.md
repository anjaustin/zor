# Frozen 6502 Standalone C Implementation

A complete 6502 ALU emulator in minimal C. Zero dependencies. 1.5KB executable.

## Quick Start

```bash
# Build with printf (for testing)
gcc -Os -s frozen_6502_standalone.c -o f6502
./f6502

# Build minimal (no libc, 1.5KB)
gcc -Os -s -nostdlib -static frozen_6502_minimal.c -o f6502_tiny
./f6502_tiny
echo $?  # 0 = all tests pass
```

## Files

| File | Description | Size |
|------|-------------|------|
| `frozen_6502_standalone.c` | Full version with tests and output | ~300 lines |
| `frozen_6502_minimal.c` | No-libc version, minimal size | ~100 lines |
| `f6502_tiny` | Compiled minimal executable | 1,544 bytes |

## Binary Size Breakdown

```
Code (text section):     749 bytes
Data (routing table):     16 bytes
─────────────────────────────────
Total logic:             765 bytes

ELF overhead:            779 bytes
─────────────────────────────────
Final executable:      1,544 bytes
```

## The 16 Frozen Shapes

All 6502 ALU operations are composed from these 16 shapes:

| ID | Shape | Operation | C Implementation |
|----|-------|-----------|------------------|
| 0 | RIPPLE_ADD | a + b + carry | `uint16_t s = a + b + c` |
| 1 | RIPPLE_SUB | a - b - borrow | `uint16_t d = a - b - (1-c)` |
| 2 | PARALLEL_AND | a & b | `a & b` |
| 3 | PARALLEL_OR | a \| b | `a \| b` |
| 4 | PARALLEL_XOR | a ^ b | `a ^ b` |
| 5 | SHIFT_LEFT | a << 1 | `a << 1` |
| 6 | SHIFT_RIGHT | a >> 1 | `a >> 1` |
| 7 | ROTATE_LEFT | (a << 1) \| carry | `(a << 1) \| c` |
| 8 | ROTATE_RIGHT | (a >> 1) \| (carry << 7) | `(a >> 1) \| (c << 7)` |
| 9 | INCREMENT | a + 1 | `a + 1` |
| 10 | DECREMENT | a - 1 | `a - 1` |
| 11 | TRANSFER | a | `a` |
| 12 | LOAD | mem | `mem` |
| 13 | STORE | a | `a` |
| 14 | BIT_TEST | a & b (flags only) | `a & b` |
| 15 | IDENTITY | a | `a` |

## Routing Table

The routing table maps opcodes to shapes (16 bytes):

```c
static const uint8_t ROUTE[16] = {
    0,  /* ADC -> RIPPLE_ADD */
    1,  /* SBC -> RIPPLE_SUB */
    2,  /* AND -> PARALLEL_AND */
    3,  /* ORA -> PARALLEL_OR */
    4,  /* EOR -> PARALLEL_XOR */
    5,  /* ASL -> SHIFT_LEFT */
    6,  /* LSR -> SHIFT_RIGHT */
    7,  /* ROL -> ROTATE_LEFT */
    8,  /* ROR -> ROTATE_RIGHT */
    9,  /* INX -> INCREMENT */
    10, /* DEX -> DECREMENT */
    9,  /* INY -> INCREMENT (same shape) */
    10, /* DEY -> DECREMENT (same shape) */
    11, /* TAX -> TRANSFER */
    11, /* TXA -> TRANSFER */
    12, /* LDA -> LOAD */
};
```

## Test Results

```
ADC: 42 + 13 = 55 (expected 55) OK
EOR: 0x55 ^ 0xFF = 0xAA (expected 0xAA) OK
ASL: 0x40 << 1 = 0x80, C=0 (expected 0x80, C=0) OK
ASL: 0x80 << 1 = 0x00, C=1 (expected 0x00, C=1) OK
INX: 254 + 1 = 255 (expected 255) OK
INX: 255 + 1 = 0 (expected 0, wrap) OK
AND: 0xFF & 0x0F = 0x0F (expected 0x0F) OK
TAX: A=0x42 -> X = 0x42 (expected 0x42) OK
```

## The 326-Byte Claim

The documentation claims "326 bytes" for the frozen 6502. This refers to the **minimal data representation**:

- Polynomial coefficients for 16 shapes: ~310 bytes
- Packed routing table: ~16 bytes
- **Total: ~326 bytes**

The 1,544-byte executable includes:
- The model data (~326 bytes conceptually)
- Executor code (749 bytes compiled ARM64)
- ELF headers (779 bytes)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FROZEN 6502                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Opcode ──► Routing Table ──► Shape ID ──► Frozen Shape   │
│              (16 bytes)         (4 bits)    (pure function) │
│                                                             │
│   "Learning is routing. Computation is geometry."           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Insight

> The shapes ARE the computation. They have 0 learnable parameters.
> We don't train math - we discover it.

The polynomial form `XOR(a,b) = a + b - 2ab` is not learned. It's a mathematical identity that happens to equal exclusive-or on binary inputs.

## See Also

- [FROZEN_6502.md](../docs/FROZEN_6502.md) - Full architecture documentation
- [QUICKSTART_6502.md](../docs/QUICKSTART_6502.md) - Python usage guide
- [trix.native.FrozenALU](../src/trix/native/frozen.py) - Native Python implementation
