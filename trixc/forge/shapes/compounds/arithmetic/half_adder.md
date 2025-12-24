# Half Adder

*The simplest arithmetic — XOR + AND*

```
┌─────────────────────────────────────────────────────────────┐
│ HALF ADDER                                                  │
├─────────────────────────────────────────────────────────────┤
│ Kingdom: Arithmetic                                         │
│ Type: Compound                                              │
│ Arity: Binary (2 inputs, 2 outputs)                         │
│ Frozen: Yes                                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Definition

### Mathematical

```
sum   = a ⊕ b       (XOR)
carry = a ∧ b       (AND)
```

Two parallel operations on the same inputs.

### Prose

A half adder computes the sum of two single bits. It produces two outputs: the **sum** (1 if exactly one input is 1) and the **carry** (1 if both inputs are 1). It's called "half" because it doesn't accept a carry-in from a previous stage.

---

## Visual

```
         ┌─────────────────────────┐
         │      HALF ADDER         │
         │                         │
    a ───┼───┬───[XOR]───────────────── sum
         │   │                     │
    b ───┼───┴───[AND]───────────────── carry
         │                         │
         └─────────────────────────┘

Internal structure:

    a ──┬──────┐
        │      XOR ──── sum
    b ──┼──┬───┘
        │  │
        │  └───┐
        └──────AND ──── carry
```

---

## Truth Table

| a | b | sum | carry |
|---|---|-----|-------|
| 0 | 0 |  0  |   0   |
| 0 | 1 |  1  |   0   |
| 1 | 0 |  1  |   0   |
| 1 | 1 |  0  |   1   |

Notice: `a + b = carry*2 + sum` in binary.

---

## Examples

### Discrete

```
half_adder(0, 0) → sum=0, carry=0   # 0 + 0 = 0
half_adder(0, 1) → sum=1, carry=0   # 0 + 1 = 1
half_adder(1, 0) → sum=1, carry=0   # 1 + 0 = 1
half_adder(1, 1) → sum=0, carry=1   # 1 + 1 = 10 (binary)
```

### Continuous

```
half_adder(0.5, 0.5) → sum=0.5, carry=0.25
half_adder(0.8, 0.9) → sum=0.26, carry=0.72
half_adder(1.0, 1.0) → sum=0.0, carry=1.0
```

---

## Implementation

### Python

```python
def half_adder(a: float, b: float) -> tuple[float, float]:
    """
    Differentiable half adder for continuous inputs in [0, 1].
    Returns (sum, carry).
    """
    sum_out = a + b - 2 * a * b   # XOR
    carry = a * b                  # AND
    return sum_out, carry

def half_adder_discrete(a: int, b: int) -> tuple[int, int]:
    """Discrete half adder for boolean inputs."""
    return (a ^ b, a & b)
```

### C

```c
typedef struct {
    float sum;
    float carry;
} HalfAdderResult;

static inline HalfAdderResult half_adder(float a, float b) {
    HalfAdderResult result;
    result.sum = a + b - 2.0f * a * b;  // XOR
    result.carry = a * b;                // AND
    return result;
}
```

---

## Relationships

### Built From

- **[XOR](../../elements/logic/xor.md)** — Computes the sum bit
- **[AND](../../elements/logic/and.md)** — Computes the carry bit

### Used In

- **[full_adder](full_adder.md)** — Two half adders + OR make a full adder
- **Ripple carry adder** — Chain of full adders for multi-bit addition
- **ALU design** — Fundamental building block of arithmetic units

### See Also

- **[full_adder](full_adder.md)** — Adds with carry-in
- **[XOR](../../elements/logic/xor.md)** — The sum operation
- **[AND](../../elements/logic/and.md)** — The carry operation

---

## Use Cases

1. **Binary Addition**: The foundation of all digital arithmetic. Every integer add ultimately reduces to half and full adders.

2. **Learning Tool**: The half adder is often the first compound circuit students learn. It demonstrates how logic combines into arithmetic.

3. **Neural Frozen Layers**: In frozen neural networks, half adders can perform addition without learned weights.

4. **XOR Neural Networks**: The half adder is a natural architecture for learning XOR-like patterns.

---

## Why "Half"?

A half adder handles two inputs but cannot accept a carry from a previous addition. When adding multi-bit numbers, each position (except the first) needs to handle three inputs: a, b, and carry-in.

```
  1 0 1 1
+ 0 1 1 0
─────────
        ↑ half adder works here (no carry-in)
      ↑ full adder needed (carry-in possible)
```

The **full adder** extends the half adder to handle all three inputs.

---

## The First Molecule

If XOR and AND are atoms, the half adder is the first molecule. It's the simplest meaningful combination of logic gates — the point where logic becomes arithmetic.

This is what Geocadesia is about: understanding how elements compose into compounds, and compounds into architectures.

---

*"We really shook the pillars of heaven, didn't we, Wang?"*
