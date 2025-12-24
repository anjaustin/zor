# FrozenFoundry Shape Library

**Complete Reference for Frozen Shapes**

---

## Overview

Frozen shapes are mathematical functions with 0 learnable parameters that implement exact computation. They are the foundation of FrozenFoundry's 100% accuracy guarantee.

---

## Core Primitive: PureMath

All shapes are built from four continuous gate functions:

```python
class PureMath:
    @staticmethod
    def xor_gate(a, b):
        """XOR: Saddle surface - a + b - 2ab"""
        return a + b - 2 * a * b

    @staticmethod
    def and_gate(a, b):
        """AND: Product - ab"""
        return a * b

    @staticmethod
    def or_gate(a, b):
        """OR: Union - a + b - ab"""
        return a + b - a * b

    @staticmethod
    def not_op(a):
        """NOT: Reflection - 1 - a"""
        return 1 - a
```

### Properties

- **Differentiable**: Gradients flow through all operations
- **Exact on {0, 1}**: Perfect accuracy on binary inputs
- **Composable**: Build complex operations from primitives

---

## Built-in Shapes

### Arithmetic

#### `ripple_add` (ADC)
8-bit addition with carry-in and carry-out.

```python
def ripple_add(a, b, c):
    carry = c
    result = []
    for i in range(8):
        s, carry = full_adder(a[:, i], b[:, i], carry)
        result.append(s)
    return torch.stack(result, dim=1), carry
```

**Input**: a[8], b[8], carry_in
**Output**: result[8], carry_out
**Used by**: ADC, XADD

---

#### `ripple_sub` (SBC)
Subtraction using one's complement addition.

```python
def ripple_sub(a, b, c):
    # SBC: a + ~b + c
    b_inv = not_op(b)
    return ripple_add(a, b_inv, c)
```

**Input**: a[8], b[8], borrow_in
**Output**: result[8], borrow_out
**Used by**: SBC, SBB

---

#### `increment` (INC)
Add 1 to the input.

```python
def increment(a, b, c):
    one = zeros(8)
    one[:, 0] = 1  # LSB = 1
    return ripple_add(a, one, 0)
```

**Input**: a[8]
**Output**: result[8], overflow
**Used by**: INC, INX, INY

---

#### `decrement` (DEC)
Subtract 1 from the input.

```python
def decrement(a, b, c):
    # DEC: a + 0xFF + 1 (two's complement -1)
    return ripple_add(a, ones(8), 1)
```

**Input**: a[8]
**Output**: result[8], underflow
**Used by**: DEC, DEX, DEY

---

#### `compare` (CMP)
Subtract for flags without storing result.

```python
def compare(a, b, c):
    result, carry = ripple_sub(a, b, 1)
    return result, carry
```

**Input**: a[8], b[8]
**Output**: result[8], carry (a >= b)
**Used by**: CMP, CPX, CPY, SUB

---

### Logic

#### `parallel_and` (AND)
Bitwise AND across all bits in parallel.

```python
def parallel_and(a, b, c):
    return and_gate(a, b), 0
```

**Input**: a[8], b[8]
**Output**: result[8], 0
**Used by**: AND, TEST

---

#### `parallel_or` (ORA)
Bitwise OR across all bits in parallel.

```python
def parallel_or(a, b, c):
    return or_gate(a, b), 0
```

**Input**: a[8], b[8]
**Output**: result[8], 0
**Used by**: ORA, OR

---

#### `parallel_xor` (EOR)
Bitwise XOR across all bits in parallel.

```python
def parallel_xor(a, b, c):
    return xor_gate(a, b), 0
```

**Input**: a[8], b[8]
**Output**: result[8], 0
**Used by**: EOR, XOR

---

#### `parallel_nand` (NAND)
Bitwise NAND.

```python
def parallel_nand(a, b, c):
    return not_op(and_gate(a, b)), 0
```

---

#### `parallel_nor` (NOR)
Bitwise NOR.

```python
def parallel_nor(a, b, c):
    return not_op(or_gate(a, b)), 0
```

---

#### `parallel_xnor` (XNOR)
Bitwise XNOR (equivalence).

```python
def parallel_xnor(a, b, c):
    return not_op(xor_gate(a, b)), 0
```

---

#### `complement` (NOT)
Bitwise complement (invert all bits).

```python
def complement(a, b, c):
    return not_op(a), c  # Preserves carry
```

**Used by**: NOT

---

### Shifts

#### `shift_left` (ASL)
Arithmetic shift left by 1 bit.

```python
def shift_left(a, b, c):
    # MSB goes to carry, LSB becomes 0
    carry_out = a[:, 7]
    result = cat([zeros(:, 1), a[:, :7]], dim=1)
    return result, carry_out
```

**Input**: a[8]
**Output**: result[8], carry (old MSB)
**Used by**: ASL, SHL

---

#### `shift_right` (LSR)
Logical shift right by 1 bit.

```python
def shift_right(a, b, c):
    # LSB goes to carry, MSB becomes 0
    carry_out = a[:, 0]
    result = cat([a[:, 1:], zeros(:, 1)], dim=1)
    return result, carry_out
```

**Input**: a[8]
**Output**: result[8], carry (old LSB)
**Used by**: LSR, SHR

---

### Rotates

#### `rotate_left` (ROL)
Rotate left through carry.

```python
def rotate_left(a, b, c):
    # MSB -> carry, carry -> LSB
    carry_out = a[:, 7]
    result = cat([c.unsqueeze(-1), a[:, :7]], dim=1)
    return result, carry_out
```

**Input**: a[8], carry_in
**Output**: result[8], carry_out
**Used by**: ROL, RCL

---

#### `rotate_right` (ROR)
Rotate right through carry.

```python
def rotate_right(a, b, c):
    # LSB -> carry, carry -> MSB
    carry_out = a[:, 0]
    result = cat([a[:, 1:], c.unsqueeze(-1)], dim=1)
    return result, carry_out
```

**Input**: a[8], carry_in
**Output**: result[8], carry_out
**Used by**: ROR, RCR

---

### Data Movement

#### `identity` (NOP)
Pass through unchanged.

```python
def identity(a, b, c):
    return a, c
```

**Used by**: MOV, TAX, TXA, LDA, NOP, CLC, STC

---

## x86-Specific Shapes

### Arithmetic

#### `negate`
Two's complement negation: -a = ~a + 1

```python
def negate(a, b, c):
    inverted = not_op(a)
    one = zeros_like(a)
    one[:, 0] = 1
    result, _ = ripple_add(inverted, one, 0)
    # Carry set if a != 0
    cf = 1 - not_op(a.sum(dim=1).clamp(0, 1))
    return result, cf
```

**Used by**: NEG

---

#### `add_no_carry`
Addition without carry input (x86 ADD).

```python
def add_no_carry(a, b, c):
    return ripple_add(a, b, 0)  # Ignore carry_in
```

**Used by**: ADD, XADD

---

### Shifts

#### `shift_right_arith` (SAR)
Arithmetic shift right - preserves sign bit.

```python
def shift_right_arith(a, b, c):
    sign = a[:, 7:8]  # Preserve sign
    shifted = a[:, 1:]
    result = cat([shifted, sign], dim=1)
    return result, a[:, 0]
```

**Used by**: SAR

---

### Rotates (Without Carry)

#### `rotate_left_no_carry`
Rotate left without carry (x86 ROL).

```python
def rotate_left_no_carry(a, b, c):
    # MSB wraps to LSB, ignores carry
    high_bit = a[:, 7:8]
    lower = a[:, :7]
    result = cat([high_bit, lower], dim=1)
    return result, a[:, 7]
```

**Used by**: ROL (x86)

---

#### `rotate_right_no_carry`
Rotate right without carry (x86 ROR).

```python
def rotate_right_no_carry(a, b, c):
    # LSB wraps to MSB, ignores carry
    low_bit = a[:, 0:1]
    upper = a[:, 1:]
    result = cat([upper, low_bit], dim=1)
    return result, a[:, 0]
```

**Used by**: ROR (x86)

---

### Data Movement

#### `byte_swap`
Exchange high and low bytes (XBA, XCHG AL,AH).

```python
def byte_swap(a, b, c):
    low = a[:, :8]
    high = a[:, 8:]
    return cat([high, low], dim=1), 0
```

**Used by**: XBA (65816)

---

#### `return_second`
Return the second operand (for XCHG).

```python
def return_second(a, b, c):
    return b.clone(), 0
```

**Used by**: XCHG

---

### Sign Extension

#### `sign_extend` (CBW)
Sign extend byte to word.

```python
def sign_extend_8_16(a, b, c):
    sign = a[:, 7:8]
    high = sign.expand(-1, 8)
    return cat([a[:, :8], high], dim=1), 0
```

**Used by**: CBW (286)

---

#### `sign_extend_8_32` (MOVSX8)
Sign extend byte to dword.

```python
def sign_extend_8_32(a, b, c):
    sign = a[:, 7:8]
    high = sign.expand(-1, 24)
    return cat([a[:, :8], high], dim=1), 0
```

**Used by**: MOVSX8 (486)

---

#### `sign_extend_16_32` (MOVSX16)
Sign extend word to dword.

```python
def sign_extend_16_32(a, b, c):
    sign = a[:, 15:16]
    high = sign.expand(-1, 16)
    return cat([a[:, :16], high], dim=1), 0
```

**Used by**: MOVSX16 (486)

---

#### `sign_to_word` (CWD)
Create all 0s or all 1s based on sign.

```python
def sign_to_word(a, b, c):
    sign = a[:, 15:16]
    return sign.expand(-1, 16), 0
```

**Used by**: CWD (286)

---

#### `sign_to_dword` (CDQ)
Create all 0s or all 1s based on sign (32-bit).

```python
def sign_to_dword(a, b, c):
    sign = a[:, 31:32]
    return sign.expand(-1, 32), 0
```

**Used by**: CDQ (486)

---

### Zero Extension

#### `zero_extend_8_32` (MOVZX8)
Zero extend byte to dword.

```python
def zero_extend_8_32(a, b, c):
    zeros = torch.zeros(batch, 24)
    return cat([a[:, :8], zeros], dim=1), 0
```

**Used by**: MOVZX8 (486)

---

#### `zero_extend_16_32` (MOVZX16)
Zero extend word to dword.

```python
def zero_extend_16_32(a, b, c):
    zeros = torch.zeros(batch, 16)
    return cat([a[:, :16], zeros], dim=1), 0
```

**Used by**: MOVZX16 (486)

---

### Byte Order

#### `bswap`
Reverse byte order (endianness swap).

```python
def bswap(a, b, c):
    b0 = a[:, 0:8]
    b1 = a[:, 8:16]
    b2 = a[:, 16:24]
    b3 = a[:, 24:32]
    return cat([b3, b2, b1, b0], dim=1), 0
```

**Used by**: BSWAP (486)

---

### Conditional

#### `conditional_select`
MUX: if c then b else a.

```python
def conditional_select(a, b, c):
    c_exp = c.unsqueeze(-1).expand(-1, 32)
    result = a * (1 - c_exp) + b * c_exp
    return result, 0
```

**Used by**: CMPXCHG (486)

---

## Adding Custom Shapes

To add a custom shape:

```python
from trix.foundry.frozen_foundry import FrozenFoundry, PureMath
import torch

foundry = FrozenFoundry(bit_width=8)

def my_custom_shape(a, b, c):
    """
    Args:
        a: [batch, bit_width] - first operand bits
        b: [batch, bit_width] - second operand bits
        c: [batch] - carry in

    Returns:
        result: [batch, bit_width] - result bits
        carry: [batch] - carry out
    """
    # Use PureMath primitives
    xor_result = PureMath.xor_gate(a, b)
    and_result = PureMath.and_gate(a, b)

    # Return result and carry
    return xor_result, and_result[:, 7]  # Example

foundry.register_shape(
    name="my_custom_shape",
    fn=my_custom_shape,
    n_inputs=2,           # Number of operands (1 or 2)
    uses_carry=False,     # Whether it uses carry input
    produces_carry=True,  # Whether it produces carry output
    description="My custom operation"
)
```

---

## Shape Categories

### By Input Count

| 1-Input | 2-Input |
|---------|---------|
| identity | ripple_add |
| complement | ripple_sub |
| increment | parallel_and |
| decrement | parallel_or |
| shift_left | parallel_xor |
| shift_right | compare |
| rotate_left | return_second |
| rotate_right | conditional_select |
| negate | |
| sign_extend_* | |
| zero_extend_* | |
| bswap | |

### By Carry Usage

| Uses Carry | Produces Carry | Neither |
|------------|----------------|---------|
| ripple_add | ripple_add | parallel_and |
| ripple_sub | ripple_sub | parallel_or |
| rotate_left | shift_left | parallel_xor |
| rotate_right | shift_right | identity |
| | rotate_left | complement |
| | rotate_right | byte_swap |
| | negate | sign_extend |

---

## Shape Count by CPU

| CPU | Built-in | Custom | Total |
|-----|----------|--------|-------|
| 6502 | 17 | 0 | 17 |
| 65816 | 17 | 1 | 18 |
| 286 | 17 | 9 | 26 |
| 486 | 17 | 13 | 30 |

---

*"Computation is topology. Learning is routing."*
