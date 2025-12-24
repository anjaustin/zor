# Frozen 6502 Quick Start

Get from zero to running in 5 minutes.

---

## Install

```bash
# Clone the repo
git clone https://github.com/your-org/trix.git
cd trix

# Install
pip install -e .
```

---

## 1. Run Your First Shape (30 seconds)

```python
from trix.shapes import add

print(add(42, 13))  # 55
```

You just computed `42 + 13` using a frozen mathematical shape.

---

## 2. See the Geometry (1 minute)

```python
from trix.shapes import add

add(42, 13, verbose=True)
```

Output:
```
┌──────────────────────────────────────────────────┐
│ Shape: RIPPLE_ADD                                │
├──────────────────────────────────────────────────┤
│ 8 chained full-adders propagating carry          │
│ XOR(a,b) = a + b - 2ab                           │
├──────────────────────────────────────────────────┤
│ a: 42 (00101010)                                 │
│ b: 13 (00001101)                                 │
│ carry_in: 0 (00000000)                           │
├──────────────────────────────────────────────────┤
│ result: 55 (00110111)                            │
│ carry_out: 0                                     │
└──────────────────────────────────────────────────┘
```

The addition is computed by 8 chained full-adders, each using the polynomial `XOR(a,b) = a + b - 2ab`.

---

## 3. Try More Shapes (2 minutes)

```python
from trix.shapes import xor, and_op, or_op, asl, inc

# XOR: a + b - 2ab
print(xor(0x55, 0xFF))  # 170 (0xAA)

# AND: ab
print(and_op(0xFF, 0x0F))  # 15 (0x0F)

# OR: a + b - ab
print(or_op(0xF0, 0x0F))  # 255 (0xFF)

# Shift left (returns result, carry)
print(asl(0x80))  # (0, 1) - bit 7 went to carry

# Increment
print(inc(255))  # 0 (wraps around)
```

---

## 4. Use Assembly Syntax (2 minutes)

```python
from trix.asm import run

# Simple addition (one instruction per line)
result = run("""
    LDA #$05
    ADC #$03
""")
print(result['A'])  # 8

# Multiply by 4 using shifts
result = run("""
    LDA #$10    ; Load 16
    ASL A       ; Shift left (x2)
    ASL A       ; Shift left (x2)
""")
print(result['A'])  # 64

# Verbose mode shows each shape
run("""
    LDA #$FF    ; Load 255
    EOR #$55    ; XOR with 0x55
    TAX         ; Transfer to X
""", verbose=True)
```

Output:
```
┌──────────────────────────────────────────────────────────┐
│ FROZEN 6502 ASSEMBLY EXECUTION                          │
├──────────────────────────────────────────────────────────┤
│ LDA #$FF             → A=$FF X=$00 Y=$00 C=0 [TRANSFER] │
│ EOR #$55             → A=$AA X=$00 Y=$00 C=0 [PARALLEL_XOR] │
│ TAX                  → A=$AA X=$AA Y=$00 C=0 [TRANSFER] │
└──────────────────────────────────────────────────────────┘
```

---

## What Just Happened?

You computed with **frozen geometry**.

### The 16 Shapes

The 6502 ALU has only 16 unique operations:

| Shape | Function | Polynomial |
|-------|----------|------------|
| RIPPLE_ADD | 8-bit add | 8 full-adders |
| RIPPLE_SUB | 8-bit sub | inverted add |
| PARALLEL_AND | a & b | ab |
| PARALLEL_OR | a \| b | a + b - ab |
| PARALLEL_XOR | a ^ b | a + b - 2ab |
| SHIFT_LEFT | a << 1 | bit permutation |
| SHIFT_RIGHT | a >> 1 | bit permutation |
| ROTATE_LEFT | rotate through carry | |
| ROTATE_RIGHT | rotate through carry | |
| INCREMENT | a + 1 | |
| DECREMENT | a - 1 | |
| TRANSFER | copy | f(a) = a |
| LOAD | from memory | f(a) = a |
| STORE | to memory | f(a) = a |
| BIT_TEST | flags only | |
| IDENTITY | no-op | f(a) = a |

### Zero Learning

These shapes have **0 learnable parameters**. They are pure mathematical functions that compute exactly.

The XOR formula `a + b - 2ab` is not learned - it's discovered. It IS the exclusive-or operation expressed as a continuous polynomial.

---

## Supported Assembly

```asm
; Load immediate
LDA #$42    ; Load into A
LDX #$10    ; Load into X
LDY #$20    ; Load into Y

; Arithmetic
ADC #$05    ; Add with carry
SBC #$03    ; Subtract with borrow

; Logic
AND #$0F    ; Bitwise AND
ORA #$F0    ; Bitwise OR
EOR #$FF    ; Bitwise XOR

; Shifts (accumulator)
ASL A       ; Shift left
LSR A       ; Shift right
ROL A       ; Rotate left through carry
ROR A       ; Rotate right through carry

; Increment/Decrement
INX         ; Increment X
DEX         ; Decrement X
INY         ; Increment Y
DEY         ; Decrement Y

; Transfers
TAX         ; A -> X
TXA         ; X -> A
TAY         ; A -> Y
TYA         ; Y -> A

; Flags
CLC         ; Clear carry
SEC         ; Set carry

; Compare
CMP #$10    ; Compare A with value
```

---

## Next Steps

1. **Explore the notebook**: `examples/frozen_6502/geometry_journey.ipynb`
2. **Read the theory**: [FROZEN_6502.md](FROZEN_6502.md)
3. **See the neural net**: [FROZEN_6502_NET.md](FROZEN_6502_NET.md)
4. **Export to ONNX**: The geometry can be serialized to a 73KB file

---

## The Core Insight

> *"Computation is geometry. Learning is routing."*

The 6502's ALU operations are not learned - they are mathematical truths expressed as polynomials. The shapes are frozen because they cannot be any other way.

When you call `add(42, 13)`, you're not running a trained model. You're computing with pure geometry.

---

*The shapes ARE the computation.*
