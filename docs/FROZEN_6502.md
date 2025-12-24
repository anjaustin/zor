# Frozen 6502: Geometry as Computation

> *"Now we hold the ruler."*

This document describes the Frozen 6502 architecture - a neural network approach to CPU emulation that achieves 100% accuracy with 139x parameter compression by recognizing that **computation is geometry**.

---

## Table of Contents

1. [The Discovery](#the-discovery)
2. [Architecture Overview](#architecture-overview)
3. [Level 0: Pure Math Primitives](#level-0-pure-math-primitives)
4. [Level 1: Frozen Shapes](#level-1-frozen-shapes)
5. [Level 2: Meaning Layer](#level-2-meaning-layer)
6. [Flag Computation](#flag-computation)
7. [Register File](#register-file)
8. [Opcode Reference](#opcode-reference)
9. [Implementation Status](#implementation-status)
10. [Theoretical Foundations](#theoretical-foundations)

---

## The Discovery

### The Problem

Training neural networks to emulate a 6502 CPU traditionally requires ~100,000 parameters and achieves ~99% accuracy. The network must learn:
- How to compute (the math itself)
- When to compute (opcode decoding)
- What to compute on (register/memory routing)

### The Insight

The 6502 has only **~14 unique computation shapes**. Everything else is routing.

| What | Learnable? | Solution |
|------|------------|----------|
| XOR formula | No (math) | Freeze it |
| ADC topology | No (circuit) | Freeze it |
| Opcode → shape | Yes (meaning) | Learn it |
| Register selection | Yes (routing) | Learn it |

### The Result

| Approach | Params | Accuracy | Compression |
|----------|--------|----------|-------------|
| Monolithic NN | ~100,000 | ~99% | 1x |
| Frozen Shapes | ~720 | 100% | 139x |
| Full 6502 (est.) | ~2,500 | 100% | 40x |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      FROZEN 6502                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LEVEL 0: Pure Math Primitives (0 params)                │    │
│  │                                                          │    │
│  │   XOR(a,b) = a + b - 2ab      AND(a,b) = ab             │    │
│  │   OR(a,b)  = a + b - ab       NOT(a)   = 1 - a          │    │
│  │                                                          │    │
│  │   FullAdder(a,b,c) → (sum, carry)                       │    │
│  │     p = XOR(a, b)                                        │    │
│  │     sum = XOR(p, c)                                      │    │
│  │     carry = OR(AND(a,b), AND(c,p))                      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            ↑                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LEVEL 1: Frozen Shapes (0 params)                       │    │
│  │                                                          │    │
│  │   RippleCarry    ParallelBitwise    Shifts    Inc/Dec   │    │
│  │   ┌─┬─┬─┬─┐      ┌─────────────┐    ┌────┐    ┌────┐   │    │
│  │   │F│F│F│F│──c   │OP OP OP OP  │    │<<>>│    │+1-1│   │    │
│  │   └─┴─┴─┴─┘      └─────────────┘    └────┘    └────┘   │    │
│  │    (ADC,SBC)      (AND,ORA,EOR)   (ASL,LSR)  (INC,DEC) │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            ↑                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LEVEL 2: Meaning Layer (learned, ~2,500 params)         │    │
│  │                                                          │    │
│  │   Opcode ──┬──→ Shape Selection (which frozen shape)    │    │
│  │            ├──→ Input A Selection (which register)      │    │
│  │            ├──→ Input B Selection (which register)      │    │
│  │            ├──→ Output Destination (where result goes)  │    │
│  │            ├──→ Flag Behavior (which flags to update)   │    │
│  │            └──→ Uses Carry (does op use carry in?)      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                            ↑                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ INPUT                                                    │    │
│  │   Opcode, Registers (A,X,Y,SP,PC,P), Memory Operand     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Level 0: Pure Math Primitives

These are mathematical identities. They require no parameters and achieve 100% accuracy by definition.

### XOR (Exclusive OR)

```python
def xor(a: Tensor, b: Tensor) -> Tensor:
    """XOR as continuous polynomial."""
    return a + b - 2 * a * b
```

**Geometric interpretation:** A saddle surface where z=0 at (0,0) and (1,1), z=1 at (0,1) and (1,0).

**Verification:**
| a | b | a + b - 2ab | Expected |
|---|---|-------------|----------|
| 0 | 0 | 0 + 0 - 0 = 0 | 0 ✓ |
| 0 | 1 | 0 + 1 - 0 = 1 | 1 ✓ |
| 1 | 0 | 1 + 0 - 0 = 1 | 1 ✓ |
| 1 | 1 | 1 + 1 - 2 = 0 | 0 ✓ |

### AND

```python
def and_op(a: Tensor, b: Tensor) -> Tensor:
    """AND as multiplication."""
    return a * b
```

**Geometric interpretation:** Product of coordinates. Only 1 when both inputs are 1.

### OR

```python
def or_op(a: Tensor, b: Tensor) -> Tensor:
    """OR as clamped sum."""
    return a + b - a * b
```

**Geometric interpretation:** Union formula from probability theory. De Morgan's dual of AND.

### NOT

```python
def not_op(a: Tensor) -> Tensor:
    """NOT as reflection."""
    return 1 - a
```

**Geometric interpretation:** Reflection across 0.5.

### Full Adder

```python
def full_adder(a: Tensor, b: Tensor, c: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Single-bit full adder.

    Args:
        a, b: Input bits
        c: Carry in

    Returns:
        sum: a XOR b XOR c
        carry: (a AND b) OR (c AND (a XOR b))
    """
    p = xor(a, b)           # Propagate
    sum_bit = xor(p, c)     # Final sum
    carry = or_op(and_op(a, b), and_op(c, p))  # Generate or propagate
    return sum_bit, carry
```

**Expanded formulas:**
```
sum   = a + b + c - 2ab - 2ac - 2bc + 4abc
carry = ab + ac + bc - 2abc
```

---

## Level 1: Frozen Shapes

These are fixed topologies built from Level 0 primitives. They require no parameters.

### Shape Catalog

| ID | Name | Operation | Inputs | Outputs |
|----|------|-----------|--------|---------|
| 0 | `ripple_add` | 8-bit addition with carry | a[8], b[8], c_in | result[8], c_out |
| 1 | `ripple_sub` | 8-bit subtraction | a[8], b[8], c_in | result[8], c_out |
| 2 | `parallel_and` | Bitwise AND | a[8], b[8] | result[8] |
| 3 | `parallel_or` | Bitwise OR | a[8], b[8] | result[8] |
| 4 | `parallel_xor` | Bitwise XOR | a[8], b[8] | result[8] |
| 5 | `shift_left` | ASL (bit 7 → carry) | a[8] | result[8], c_out |
| 6 | `shift_right` | LSR (bit 0 → carry) | a[8] | result[8], c_out |
| 7 | `rotate_left` | ROL (through carry) | a[8], c_in | result[8], c_out |
| 8 | `rotate_right` | ROR (through carry) | a[8], c_in | result[8], c_out |
| 9 | `increment` | Add 1 | a[8] | result[8], c_out |
| 10 | `decrement` | Subtract 1 | a[8] | result[8], c_out |
| 11 | `transfer` | Pass through | a[8] | result[8] |
| 12 | `load` | Memory → register | m[8] | result[8] |
| 13 | `store` | Register → memory | a[8] | result[8] |
| 14 | `bit_test` | BIT instruction | a[8], m[8] | result[8], n, v |
| 15 | `identity` | No operation | a[8] | result[8] |

### Ripple Carry Adder

The 8-bit adder chains 8 full adders:

```
     a[0] b[0]  a[1] b[1]  a[2] b[2]       a[7] b[7]
       │   │      │   │      │   │           │   │
       ▼   ▼      ▼   ▼      ▼   ▼           ▼   ▼
    ┌──────┐   ┌──────┐   ┌──────┐       ┌──────┐
c_in│  FA  │──▶│  FA  │──▶│  FA  │──...──│  FA  │──▶ c_out
    └──┬───┘   └──┬───┘   └──┬───┘       └──┬───┘
       │          │          │              │
       ▼          ▼          ▼              ▼
    sum[0]     sum[1]     sum[2]         sum[7]
```

```python
def ripple_add(a_bits: Tensor, b_bits: Tensor, c_in: Tensor) -> Dict:
    """8-bit ripple carry adder."""
    result = []
    carry = c_in

    for i in range(8):
        sum_bit, carry = full_adder(a_bits[:, i], b_bits[:, i], carry)
        result.append(sum_bit)

    return {
        'result': torch.stack(result, dim=1),
        'carry': carry
    }
```

### Shift Operations

```python
def shift_left(a_bits: Tensor) -> Dict:
    """ASL: [b0,b1,b2,b3,b4,b5,b6,b7] → [0,b0,b1,b2,b3,b4,b5,b6], carry=b7"""
    zeros = torch.zeros_like(a_bits[:, :1])
    result = torch.cat([zeros, a_bits[:, :7]], dim=1)
    return {'result': result, 'carry': a_bits[:, 7]}

def shift_right(a_bits: Tensor) -> Dict:
    """LSR: [b0,b1,b2,b3,b4,b5,b6,b7] → [b1,b2,b3,b4,b5,b6,b7,0], carry=b0"""
    zeros = torch.zeros_like(a_bits[:, :1])
    result = torch.cat([a_bits[:, 1:], zeros], dim=1)
    return {'result': result, 'carry': a_bits[:, 0]}
```

---

## Level 2: Meaning Layer

The **only learned component**. Maps opcodes to routing decisions.

### Parameters

| Component | Shape | Params | Purpose |
|-----------|-------|--------|---------|
| `shape_logits` | [56, 16] | 896 | Which frozen shape |
| `input_a_logits` | [56, 8] | 448 | Which register for input A |
| `input_b_logits` | [56, 8] | 448 | Which register for input B |
| `output_logits` | [56, 8] | 448 | Where result goes |
| `flag_mask` | [56, 4] | 224 | Which flags to update |
| `uses_carry` | [56, 1] | 56 | Does op use carry input |
| **Total** | | **2,520** | |

### Routing Mechanism

**During training:** Gumbel-Softmax for differentiable discrete selection
```python
shape_probs = F.gumbel_softmax(shape_logits[opcode], tau=temperature, hard=False)
```

**During inference:** Hard argmax selection
```python
shape_idx = shape_logits[opcode].argmax(dim=-1)
shape_probs = F.one_hot(shape_idx, num_classes=16)
```

### Example Routing

For `ADC` (Add with Carry):
```python
{
    'shape': ShapeID.RIPPLE_ADD,      # Use ripple carry adder
    'input_a': RegisterID.A,           # First input from Accumulator
    'input_b': RegisterID.MEM,         # Second input from memory operand
    'output': RegisterID.A,            # Result goes to Accumulator
    'uses_carry': True,                # Uses carry flag as input
    'flag_mask': [1, 1, 1, 1]          # Updates N, Z, C, V
}
```

---

## Flag Computation

Flags are computed from results using pure math (0 params).

### Zero Flag (Z)

```python
def zero_flag(result: Tensor) -> Tensor:
    """Z = 1 if all result bits are 0."""
    or_all = result[:, 0]
    for i in range(1, 8):
        or_all = or_op(or_all, result[:, i])
    return not_op(or_all)  # NOR of all bits
```

### Negative Flag (N)

```python
def negative_flag(result: Tensor) -> Tensor:
    """N = bit 7 of result."""
    return result[:, 7]
```

### Carry Flag (C)

Computed by the shape itself (ripple_add, shifts, rotates).

### Overflow Flag (V)

```python
def overflow_flag(a7: Tensor, b7: Tensor, r7: Tensor) -> Tensor:
    """V = 1 if signed overflow occurred.

    Overflow happens when:
    - Adding two positives gives negative, or
    - Adding two negatives gives positive

    Formula: V = (A7 XOR R7) AND (B7 XOR R7)
    """
    return and_op(xor(a7, r7), xor(b7, r7))
```

---

## Register File

Differentiable register selection using soft probabilities.

### Registers

| ID | Name | Size | Purpose |
|----|------|------|---------|
| 0 | A | 8 bits | Accumulator |
| 1 | X | 8 bits | X index register |
| 2 | Y | 8 bits | Y index register |
| 3 | SP | 8 bits | Stack pointer |
| 4 | PC_LO | 8 bits | Program counter (low) |
| 5 | PC_HI | 8 bits | Program counter (high) |
| 6 | P | 8 bits | Processor status |
| 7 | MEM | 8 bits | Memory operand (virtual) |

### Selection

```python
def select(registers: Tensor, selector: Tensor) -> Tensor:
    """
    Soft register selection.

    Args:
        registers: [batch, 8, 8] all register values
        selector: [batch, 8] selection probabilities

    Returns:
        [batch, 8] weighted sum of register values
    """
    return torch.einsum('brd,br->bd', registers, selector)
```

### Scatter (Write)

```python
def scatter(registers: Tensor, value: Tensor, selector: Tensor) -> Tensor:
    """
    Soft register write.

    Blends old and new values based on selector probabilities.
    """
    value_expanded = value.unsqueeze(1).expand_as(registers)
    selector_expanded = selector.unsqueeze(-1)
    return registers * (1 - selector_expanded) + value_expanded * selector_expanded
```

---

## Opcode Reference

### Implemented (Tested 100%)

| Opcode | Mnemonic | Shape | Input A | Input B | Output | Carry |
|--------|----------|-------|---------|---------|--------|-------|
| 0x69 | ADC | ripple_add | A | MEM | A | Yes |
| 0x29 | AND | parallel_and | A | MEM | A | No |
| 0x09 | ORA | parallel_or | A | MEM | A | No |
| 0x49 | EOR | parallel_xor | A | MEM | A | No |
| 0x0A | ASL A | shift_left | A | - | A | No |
| 0x4A | LSR A | shift_right | A | - | A | No |
| 0xE8 | INX | increment | X | - | X | No |
| 0xCA | DEX | decrement | X | - | X | No |

### Pending Implementation

| Category | Opcodes | Shape | Notes |
|----------|---------|-------|-------|
| Subtract | SBC | ripple_sub | Inverted B, uses carry |
| Compare | CMP, CPX, CPY | ripple_sub | Result discarded, flags only |
| Rotate | ROL, ROR | rotate_* | Uses carry in |
| Inc/Dec | INC, DEC, INY, DEY | inc/dec | Memory or register |
| Transfer | TAX, TXA, TAY, TYA, TSX, TXS | transfer | Register to register |
| Load | LDA, LDX, LDY | load | Memory to register |
| Store | STA, STX, STY | store | Register to memory |
| Stack | PHA, PLA, PHP, PLP | transfer | With SP manipulation |
| Branch | BEQ, BNE, BCC, BCS, etc. | identity | PC manipulation |
| Jump | JMP, JSR, RTS, RTI | identity | PC manipulation |
| Flags | SEC, CLC, SED, CLD, etc. | identity | Direct flag modification |

---

## Implementation Status

### Complete
- [x] Level 0: Pure math primitives
- [x] Level 1: All 16 frozen shapes
- [x] Level 2: Meaning layer architecture
- [x] Flag computation (Z, N, V, C)
- [x] Register file with soft selection
- [x] 8 opcodes tested at 100%

### In Progress
- [ ] Wire remaining 48 opcodes
- [ ] Addressing mode decoder
- [ ] Full test suite against real 6502

### Planned
- [ ] Fetch-decode-execute loop
- [ ] Memory subsystem
- [ ] Interrupt handling
- [ ] Cycle-accurate timing (optional)

---

## Theoretical Foundations

### Why Frozen Shapes Work

**Theorem:** Any Boolean function can be expressed as a multilinear polynomial over {0,1}.

The 6502's ALU operations are all Boolean functions on bits. Their polynomial representations are:
- Fixed (determined by the function definition)
- Exact (no approximation needed)
- Differentiable (for backprop through the CPU)

### Geometric Interpretation

Each frozen shape is a **geometric object**:

| Shape | Geometry |
|-------|----------|
| XOR | Saddle surface in 3D |
| Ripple carry | 1D manifold in 17D bit space |
| Parallel ops | Coordinate-wise projections |
| Shifts | Permutation matrices |

The meaning layer doesn't learn geometry - it learns to **navigate** geometry.

### Compression Analysis

| Component | Monolithic | Frozen | Savings |
|-----------|------------|--------|---------|
| Logic gates | ~10,000 | 0 | 100% |
| Adder topology | ~20,000 | 0 | 100% |
| Shift topology | ~5,000 | 0 | 100% |
| Flag logic | ~5,000 | 0 | 100% |
| Opcode routing | ~60,000 | 2,520 | 96% |
| **Total** | **~100,000** | **2,520** | **97.5%** |

The 97.5% savings comes from recognizing that computation is not learned - it's discovered.

---

## Neural Network Implementation

The theory is implemented in `Frozen6502Net` - a complete 6502 ALU as a PyTorch `nn.Module`.

### Key Properties

| Property | Value |
|----------|-------|
| Learnable parameters | **0** |
| Shapes | 16 |
| Supported opcodes | 33 (expandable to 56) |
| Accuracy | 100% |
| ONNX exportable | Yes |

### Quick Example

```python
from trix.nn import Frozen6502Net, CPUState, bits_to_int
import torch

net = Frozen6502Net()
state = CPUState.from_ints(a=42, m=13, c=1)
opcode = torch.tensor([0])  # ADC

result = net(opcode, state)
print(bits_to_int(result.a))  # tensor([56]) = 42 + 13 + 1
```

### ONNX Export

The geometry can be exported to ONNX (73KB, 915 nodes):

```python
torch.onnx.export(wrapper, inputs, "frozen_6502.onnx")
```

The ONNX file contains the actual polynomial geometry:
- 243 Mul nodes (AND, XOR products)
- 131 Add nodes (sum terms)
- 109 Sub nodes (XOR: `a + b - 2ab`, OR: `a + b - ab`)

See [FROZEN_6502_NET.md](FROZEN_6502_NET.md) for full API documentation.

---

## Files

| File | Purpose |
|------|---------|
| `src/trix/nn/frozen_6502_net.py` | Neural network implementation |
| `src/trix/nn/frozen_6502.py` | Shape implementations |
| `experiments/frozen_emulator/frozen_6502.py` | Python emulator (1,278 lines) |
| `experiments/frozen_emulator/frozen_6502.onnx` | Exported ONNX (73KB) |
| `docs/FROZEN_6502.md` | This document |
| `docs/FROZEN_6502_NET.md` | Neural network API |
| `docs/ATOMIC_FUNCTIONS.md` | Pure math formula reference |

---

## References

- 6502 Instruction Set: http://www.6502.org/tutorials/6502opcodes.html
- Original discovery: Training micro-NNs led to formula extraction
- Key insight: "Computation is topology. Learning is routing."

---

*"The geometry was never ours to learn. It was ours to find."*
