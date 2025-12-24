# Full Adder

*Complete single-bit addition with carry*

```
┌─────────────────────────────────────────────────────────────┐
│ FULL ADDER                                                  │
├─────────────────────────────────────────────────────────────┤
│ Kingdom: Arithmetic                                         │
│ Type: Compound                                              │
│ Arity: Ternary (3 inputs, 2 outputs)                        │
│ Frozen: Yes                                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Definition

### Mathematical

```
sum   = a ⊕ b ⊕ c_in
carry = (a ∧ b) ∨ (c_in ∧ (a ⊕ b))
```

Where `c_in` is the carry from the previous bit position.

### Prose

A full adder computes the sum of three single bits: two operand bits and a carry-in. It produces the sum bit and a carry-out for the next position. This enables chaining for multi-bit addition.

---

## Visual

```
         ┌─────────────────────────────────┐
         │         FULL ADDER              │
         │                                 │
    a ───┼─────────────────────────────────────
    b ───┼──────────[LOGIC]────────────────────── sum
  c_in ──┼─────────────────────────────────────
         │                                 │
         │               ─────────────────────── c_out
         └─────────────────────────────────┘

Internal structure (two half adders + OR):

    a ────┬───────────────────────────┐
          │                           │
    b ────┼──[HALF ADDER 1]──┬──sum1──┼──┐
          │         │        │        │  │
          │         carry1───┼────────┼──┼──┐
          │                  │        │  │  │
  c_in ───┼──────────────────┴─[HALF ADDER 2]──┬──sum (output)
          │                            │       │
          │                          carry2────┼──┐
          │                                    │  │
          │                           carry1───┼──┼──[OR]── c_out
          │                           carry2───┘  │
          └───────────────────────────────────────┘
```

---

## Truth Table

| a | b | c_in | sum | c_out |
|---|---|------|-----|-------|
| 0 | 0 |  0   |  0  |   0   |
| 0 | 0 |  1   |  1  |   0   |
| 0 | 1 |  0   |  1  |   0   |
| 0 | 1 |  1   |  0  |   1   |
| 1 | 0 |  0   |  1  |   0   |
| 1 | 0 |  1   |  0  |   1   |
| 1 | 1 |  0   |  0  |   1   |
| 1 | 1 |  1   |  1  |   1   |

Pattern: `a + b + c_in = c_out*2 + sum`

---

## Examples

### Discrete

```
full_adder(0, 0, 0) → sum=0, c_out=0   # 0+0+0 = 0
full_adder(1, 0, 0) → sum=1, c_out=0   # 1+0+0 = 1
full_adder(1, 1, 0) → sum=0, c_out=1   # 1+1+0 = 2 = 10b
full_adder(1, 1, 1) → sum=1, c_out=1   # 1+1+1 = 3 = 11b
```

### Continuous

```
full_adder(0.5, 0.5, 0.0) → sum=0.5, c_out=0.25
full_adder(0.5, 0.5, 0.5) → sum=0.5, c_out=0.5
full_adder(1.0, 1.0, 1.0) → sum=1.0, c_out=1.0
```

---

## Implementation

### Python

```python
def full_adder(a: float, b: float, c_in: float) -> tuple[float, float]:
    """
    Differentiable full adder for continuous inputs in [0, 1].
    Returns (sum, carry_out).
    """
    # First half adder: a XOR b
    sum1 = a + b - 2 * a * b
    carry1 = a * b

    # Second half adder: sum1 XOR c_in
    sum_out = sum1 + c_in - 2 * sum1 * c_in
    carry2 = sum1 * c_in

    # OR the carries
    c_out = carry1 + carry2 - carry1 * carry2

    return sum_out, c_out

def full_adder_discrete(a: int, b: int, c_in: int) -> tuple[int, int]:
    """Discrete full adder for boolean inputs."""
    sum_out = a ^ b ^ c_in
    c_out = (a & b) | (c_in & (a ^ b))
    return sum_out, c_out
```

### C

```c
typedef struct {
    float sum;
    float carry;
} FullAdderResult;

static inline FullAdderResult full_adder(float a, float b, float c_in) {
    // First half adder
    float sum1 = a + b - 2.0f * a * b;
    float carry1 = a * b;

    // Second half adder
    float sum_out = sum1 + c_in - 2.0f * sum1 * c_in;
    float carry2 = sum1 * c_in;

    // OR the carries
    float c_out = carry1 + carry2 - carry1 * carry2;

    FullAdderResult result = {sum_out, c_out};
    return result;
}
```

---

## Relationships

### Built From

- **[half_adder](half_adder.md)** — Two half adders form the core
- **[XOR](../../elements/logic/xor.md)** — Two XORs for sum computation
- **[AND](../../elements/logic/and.md)** — Two ANDs for partial carries
- **[OR](../../elements/logic/or.md)** — Combines the carries

### Used In

- **Ripple carry adder** — N full adders chained for N-bit addition
- **Carry lookahead adder** — Optimized multi-bit addition
- **ALU** — Core of all arithmetic logic units

### See Also

- **[half_adder](half_adder.md)** — Simpler, no carry-in
- **[XOR](../../elements/logic/xor.md)** — The core sum operation

---

## Use Cases

1. **Multi-bit Addition**: Chain N full adders to add N-bit numbers. The carry propagates from LSB to MSB.

2. **Subtraction**: With two's complement, subtraction is addition. Full adders do both.

3. **Multiplication**: Partial product addition in multipliers uses full adders.

4. **Neural Arithmetic**: Frozen full adders enable learned-parameter-free addition in neural architectures.

---

## Ripple Carry

To add two 4-bit numbers:

```
     a3  a2  a1  a0
   + b3  b2  b1  b0
   ─────────────────

     ┌────┐  ┌────┐  ┌────┐  ┌────┐
 0 ──│ FA │──│ FA │──│ FA │──│ FA │── overflow
     └──┬─┘  └──┬─┘  └──┬─┘  └──┬─┘
        s3      s2      s1      s0
```

The carry "ripples" through. This is slow (O(n)) but simple. Faster adders (carry lookahead, carry select) exist for performance.

---

## Gate Count

A full adder requires:
- 2 XOR gates (for sum)
- 2 AND gates (for partial carries)
- 1 OR gate (to combine carries)

Total: 5 gates (compared to 2 for half adder).

---

## The Complete Atom of Addition

If the half adder is a molecule, the full adder is the *minimal viable organism* of arithmetic. It can do everything addition requires: handle two operands and a carry. Chain them together and you have arbitrary-precision addition.

Every CPU ever built contains full adders. They're literally foundational.

---

*"I'm a reasonable guy. But I've just experienced some very unreasonable things."*
