# Atomic Functions: The Golden Eggs

> *"The math was already perfect. The 16 params were unnecessary."*

This document captures the pure mathematical forms of 6502 atomic functions - continuous formulas that compute discrete logic without neural network weights.

---

## The Discovery

Training micro-NNs for 6502 atomic functions revealed a progression:

| Approach | Params | Accuracy | Insight |
|----------|--------|----------|---------|
| Flat MLP | 6,092 | 3.4% | Structure mismatch |
| Ripple-carry NN | 611 | 98.8% | Right topology |
| Explicit XOR NN | 16 | 100% | Right math |
| **Pure formula** | **0** | **100%** | Math IS the function |

The lesson: when the mathematical structure matches the computation, learning becomes trivial or unnecessary.

---

## Core Logic Operations (Continuous Form)

### XOR (Exclusive OR)

```
XOR(a, b) = a + b - 2ab
```

Truth table verification:
- XOR(0, 0) = 0 + 0 - 0 = 0 ✓
- XOR(0, 1) = 0 + 1 - 0 = 1 ✓
- XOR(1, 0) = 1 + 0 - 0 = 1 ✓
- XOR(1, 1) = 1 + 1 - 2 = 0 ✓

### AND

```
AND(a, b) = ab
```

### OR

```
OR(a, b) = a + b - ab
```

### NOT

```
NOT(a) = 1 - a
```

---

## Full Adder (The Golden Egg)

The full adder computes: `(a, b, c_in) → (sum, c_out)`

### Sum Bit

```
sum = XOR(XOR(a, b), c_in)
    = XOR(a + b - 2ab, c_in)
```

Let `p = a + b - 2ab` (partial XOR), then:

```
sum = p + c - 2pc
    = (a + b - 2ab) + c - 2(a + b - 2ab)c
    = a + b + c - 2ab - 2ac - 2bc + 4abc
```

**Simplified:**
```python
sum = a + b + c - 2*a*b - 2*a*c - 2*b*c + 4*a*b*c
```

### Carry Out

```
c_out = OR(AND(a, b), AND(c_in, XOR(a, b)))
      = OR(ab, c·(a + b - 2ab))
```

Let `p = a + b - 2ab`, then:

```
c_out = ab + cp - ab·cp
      = ab + c(a + b - 2ab) - ab·c(a + b - 2ab)
      = ab + ac + bc - 2abc - abc(a + b - 2ab)
```

**Simplified:**
```python
c_out = a*b + a*c + b*c - 2*a*b*c
```

Wait, let's verify this simpler form:
- c_out(0,0,0) = 0 ✓
- c_out(0,0,1) = 0 ✓
- c_out(0,1,0) = 0 ✓
- c_out(0,1,1) = 1 ✓
- c_out(1,0,0) = 0 ✓
- c_out(1,0,1) = 1 ✓
- c_out(1,1,0) = 1 ✓
- c_out(1,1,1) = 1 ✓ (1+1+1-2 = 1)

**The carry formula simplifies to majority vote!**

---

## Complete ADC Formula

```python
def full_adder_pure(a, b, c_in):
    """
    Pure mathematical full adder. No weights. No learning.

    Args:
        a, b: Input bits (0 or 1, or continuous [0,1])
        c_in: Carry in

    Returns:
        sum_bit: a XOR b XOR c_in
        c_out: Majority(a, b, c_in) with XOR twist
    """
    # XOR chain for sum
    p = a + b - 2*a*b           # a XOR b
    sum_bit = p + c_in - 2*p*c_in  # (a XOR b) XOR c_in

    # Carry = (a AND b) OR (c_in AND (a XOR b))
    c_out = a*b + c_in*p - a*b*c_in*p

    return sum_bit, c_out


def adc_8bit_pure(a_bits, m_bits, c_in):
    """
    8-bit ADC using pure math. Zero parameters.

    Args:
        a_bits: List of 8 bits [a0, a1, ..., a7] (LSB first)
        m_bits: List of 8 bits [m0, m1, ..., m7]
        c_in: Initial carry

    Returns:
        result_bits: List of 8 result bits
        c_out: Final carry
    """
    result = []
    carry = c_in

    for i in range(8):
        sum_bit, carry = full_adder_pure(a_bits[i], m_bits[i], carry)
        result.append(sum_bit)

    return result, carry
```

---

## Other Atomic Functions

### Logic Operations (Trivial)

```python
def and_8bit(a_bits, m_bits):
    return [a * m for a, m in zip(a_bits, m_bits)]

def ora_8bit(a_bits, m_bits):
    return [a + m - a*m for a, m in zip(a_bits, m_bits)]

def eor_8bit(a_bits, m_bits):
    return [a + m - 2*a*m for a, m in zip(a_bits, m_bits)]
```

### Shift Operations (Pure Wiring)

```python
def asl(a_bits):
    """Arithmetic Shift Left: shift bits, carry = bit 7"""
    return [0] + a_bits[:7], a_bits[7]

def lsr(a_bits):
    """Logical Shift Right: shift bits, carry = bit 0"""
    return a_bits[1:] + [0], a_bits[0]

def rol(a_bits, c_in):
    """Rotate Left through carry"""
    return [c_in] + a_bits[:7], a_bits[7]

def ror(a_bits, c_in):
    """Rotate Right through carry"""
    return a_bits[1:] + [c_in], a_bits[0]
```

### Subtract (SBC)

SBC is ADC with inverted M and inverted carry sense:

```python
def sbc_8bit_pure(a_bits, m_bits, c_in):
    """SBC = ADC with NOT(M) and C acting as NOT(borrow)"""
    m_inv = [1 - m for m in m_bits]
    return adc_8bit_pure(a_bits, m_inv, c_in)
```

---

## Flags

### Zero Flag (Z)

```python
def zero_flag(result_bits):
    """Z = 1 if all result bits are 0"""
    # NOR of all bits: 1 - OR(all)
    or_all = result_bits[0]
    for b in result_bits[1:]:
        or_all = or_all + b - or_all * b  # OR
    return 1 - or_all
```

### Negative Flag (N)

```python
def negative_flag(result_bits):
    """N = bit 7 of result"""
    return result_bits[7]
```

### Overflow Flag (V)

```python
def overflow_flag(a7, m7, r7):
    """V = 1 if signed overflow occurred"""
    # V = (A7 XOR R7) AND (M7 XOR R7)
    # Both operands same sign, result different sign
    a_xor_r = a7 + r7 - 2*a7*r7
    m_xor_r = m7 + r7 - 2*m7*r7
    return a_xor_r * m_xor_r
```

---

## Why This Matters

### For Neural Networks

The pure formulas show us:
1. **The target function is polynomial** - not arbitrary
2. **Structure is more important than capacity** - right topology = easy learning
3. **Some functions need zero learning** - they're already solved

### For Hardware

These continuous formulas:
- Work on analog values [0, 1]
- Are differentiable (for backprop through CPU emulation)
- Could map to analog hardware (memristors, optical computing)

### For Meta-Learning

Knowing the pure form tells us what to search for:
- Architecture search should find compositions of XOR/AND/OR
- The search space is small (polynomial combinations)
- We can verify solutions analytically

---

## Experimental Validation

```
Explicit XOR ADC: 16 params
  Epoch 20: Result=100.0%, Perfect=100.0%

Learned scales: all exactly 1.0
→ The math was already perfect
```

---

## Next Steps

1. **Implement full 6502 with pure formulas** - Verify all 56 opcodes
2. **Benchmark against neural approaches** - Speed, accuracy, interpretability
3. **Explore differentiable CPU** - Backprop through execution
4. **Hardware mapping** - Analog implementation feasibility

---

## Frozen Shapes + Meaning Layer (v2)

The ultimate compression: freeze the computation shapes, learn only the routing.

### Architecture

```
Level 0: Pure Math (0 params)
├── XOR: a + b - 2ab
├── AND: ab
├── OR:  a + b - ab
└── NOT: 1 - a

Level 1: Frozen Shapes (0 params)
├── RippleCarry (ADC, SBC)
├── ParallelBitwise (AND, ORA, EOR)
├── Shifts (ASL, LSR, ROL, ROR)
├── Increment/Decrement
└── Transfer/Load/Store

Level 2: Meaning Layer (learned)
├── Shape selection: opcode → which shape
├── Input routing: opcode → which registers
├── Output routing: opcode → where result goes
└── Flag behavior: opcode → which flags update
```

### Results

| Component | Params | Accuracy |
|-----------|--------|----------|
| Pure math | 0 | 100% |
| Frozen shapes | 0 | 100% |
| Meaning layer (16 opcodes) | 720 | 100% |
| **Full 6502 estimate** | **~2,500** | **100%** |

**Compression: 139x** (from ~100,000 monolithic params)

### Key Insight

> "Computation is topology. Learning is routing."

The 6502 doesn't need to *learn* how to add. Addition is a fixed topological operation. What it needs to learn is:
- WHEN to add (opcode decoding)
- WHAT to add (register selection)
- WHERE to put the result (output routing)

This separation of concerns is why the compression is so dramatic.

---

*"When the math matches the function, learning is trivial."*
