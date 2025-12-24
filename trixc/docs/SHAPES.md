# Frozen Shapes Reference

*The mathematical truths that power TRIXC*

> *"XOR is always `a + b - 2ab`. Forever. Frozen."*

---

## What Are Frozen Shapes?

A frozen shape is a mathematical function expressed as a polynomial. These functions:

1. **Never change** - they're mathematical truths
2. **Are exact on binary inputs** - 100% accuracy
3. **Compile to arithmetic** - just add, subtract, multiply
4. **Have zero learnable parameters** - nothing to train

---

## Logic Shapes

### XOR (Exclusive OR)

```
a ⊕ b = a + b - 2ab
```

| a | b | Result |
|---|---|--------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

```c
static inline float trix_shape_xor_f32(float a, float b) {
    return a + b - 2.0f * a * b;
}
```

**Why it works:** When a=1 and b=1, we get 1+1-2(1)(1) = 0. The `-2ab` term cancels the double-counting.

---

### AND (Logical And)

```
a ∧ b = ab
```

| a | b | Result |
|---|---|--------|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

```c
static inline float trix_shape_and_f32(float a, float b) {
    return a * b;
}
```

**Why it works:** Multiplication is AND. Only 1×1=1; everything else is 0.

---

### OR (Logical Or)

```
a ∨ b = a + b - ab
```

| a | b | Result |
|---|---|--------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

```c
static inline float trix_shape_or_f32(float a, float b) {
    return a + b - a * b;
}
```

**Why it works:** `a + b` would give 2 when both are 1. The `-ab` term corrects this.

---

### NOT (Logical Not)

```
¬a = 1 - a
```

| a | Result |
|---|--------|
| 0 | 1 |
| 1 | 0 |

```c
static inline float trix_shape_not_f32(float a) {
    return 1.0f - a;
}
```

**Why it works:** Flipping 0↔1 is just subtracting from 1.

---

### NAND (Not And)

```
¬(a ∧ b) = 1 - ab
```

```c
static inline float trix_shape_nand_f32(float a, float b) {
    return 1.0f - a * b;
}
```

---

### NOR (Not Or)

```
¬(a ∨ b) = 1 - a - b + ab
```

```c
static inline float trix_shape_nor_f32(float a, float b) {
    return 1.0f - a - b + a * b;
}
```

---

### XNOR (Exclusive Nor)

```
¬(a ⊕ b) = 1 - a - b + 2ab
```

```c
static inline float trix_shape_xnor_f32(float a, float b) {
    return 1.0f - a - b + 2.0f * a * b;
}
```

---

## Arithmetic Shapes

### Full Adder

The atomic unit of binary arithmetic. Takes three bits (a, b, carry_in) and produces sum and carry_out.

```
sum   = a ⊕ b ⊕ c
carry = (a ∧ b) ∨ ((a ⊕ b) ∧ c)
```

Expanded:
```
sum   = a + b + c - 2ab - 2ac - 2bc + 4abc
carry = ab + ac + bc - 2abc
```

```c
static inline void trix_shape_full_adder(
    float a, float b, float c,
    float* sum, float* carry
) {
    float ab_xor = a + b - 2.0f * a * b;
    *sum = ab_xor + c - 2.0f * ab_xor * c;

    float ab_and = a * b;
    float ab_xor_c = ab_xor * c;
    *carry = ab_and + ab_xor_c - ab_and * ab_xor_c;
}
```

| a | b | c | sum | carry |
|---|---|---|-----|-------|
| 0 | 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 1 | 0 |
| 0 | 1 | 0 | 1 | 0 |
| 0 | 1 | 1 | 0 | 1 |
| 1 | 0 | 0 | 1 | 0 |
| 1 | 0 | 1 | 0 | 1 |
| 1 | 1 | 0 | 0 | 1 |
| 1 | 1 | 1 | 1 | 1 |

---

### Ripple Carry Adder

8-bit addition using a chain of full adders.

```c
static inline void trix_shape_ripple_add(
    const float* a,    // 8 bits, LSB first
    const float* b,    // 8 bits, LSB first
    float c_in,
    float* result,     // 8 bits, LSB first
    float* c_out
) {
    float carry = c_in;
    for (int i = 0; i < 8; i++) {
        float sum;
        trix_shape_full_adder(a[i], b[i], carry, &sum, &carry);
        result[i] = sum;
    }
    *c_out = carry;
}
```

**Example:** 16 + 32 = 48
```
  a:  0 0 0 0 1 0 0 0  (16)
+ b:  0 0 0 0 0 1 0 0  (32)
= r:  0 0 0 0 1 1 0 0  (48)
```

---

### Ripple Carry Subtractor

8-bit subtraction: `a - b = a + (~b) + 1`

```c
static inline void trix_shape_ripple_sub(
    const float* a,
    const float* b,
    float c_in,
    float* result,
    float* c_out
) {
    float carry = 1.0f - c_in;  // Invert for subtraction
    for (int i = 0; i < 8; i++) {
        float b_inv = 1.0f - b[i];  // NOT b
        float sum;
        trix_shape_full_adder(a[i], b_inv, carry, &sum, &carry);
        result[i] = sum;
    }
    *c_out = 1.0f - carry;  // Invert back
}
```

---

## Shift Shapes

### ASL (Arithmetic Shift Left)

Shift all bits left. LSB becomes 0. MSB goes to carry.

```c
static inline void trix_shape_asl(
    const float* a,
    float* result,
    float* c_out
) {
    *c_out = a[7];  // MSB goes to carry
    result[0] = 0.0f;  // LSB becomes 0
    for (int i = 1; i < 8; i++) {
        result[i] = a[i - 1];
    }
}
```

**Example:** 0x40 << 1 = 0x80
```
Before: 0 1 0 0 0 0 0 0  (0x40)
After:  1 0 0 0 0 0 0 0  (0x80)
Carry:  0
```

---

### LSR (Logical Shift Right)

Shift all bits right. MSB becomes 0. LSB goes to carry.

```c
static inline void trix_shape_lsr(
    const float* a,
    float* result,
    float* c_out
) {
    *c_out = a[0];  // LSB goes to carry
    result[7] = 0.0f;  // MSB becomes 0
    for (int i = 0; i < 7; i++) {
        result[i] = a[i + 1];
    }
}
```

---

### ROL (Rotate Left through Carry)

Shift left, carry goes to LSB, MSB goes to carry.

```c
static inline void trix_shape_rol(
    const float* a,
    float c_in,
    float* result,
    float* c_out
) {
    *c_out = a[7];
    result[0] = c_in;
    for (int i = 1; i < 8; i++) {
        result[i] = a[i - 1];
    }
}
```

---

### ROR (Rotate Right through Carry)

Shift right, carry goes to MSB, LSB goes to carry.

```c
static inline void trix_shape_ror(
    const float* a,
    float c_in,
    float* result,
    float* c_out
) {
    *c_out = a[0];
    result[7] = c_in;
    for (int i = 0; i < 7; i++) {
        result[i] = a[i + 1];
    }
}
```

---

## Increment / Decrement

### INC (Increment)

Add 1 to the value.

```c
static inline void trix_shape_inc(
    const float* a,
    float* result,
    float* c_out
) {
    float one[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    trix_shape_ripple_add(a, one, 0.0f, result, c_out);
}
```

### DEC (Decrement)

Subtract 1 from the value.

```c
static inline void trix_shape_dec(
    const float* a,
    float* result,
    float* c_out
) {
    float one[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    trix_shape_ripple_sub(a, one, 0.0f, result, c_out);
}
```

---

## Distance Shapes

### Hamming Distance

Count the number of differing bits. Used by Providence for content-addressed lookup.

```
hamming(a, b) = popcount(a ⊕ b) = Σ (a[i] ⊕ b[i])
```

```c
static inline float trix_shape_hamming(
    const float* a,
    const float* b,
    int len
) {
    float count = 0.0f;
    for (int i = 0; i < len; i++) {
        float xor_bit = a[i] + b[i] - 2.0f * a[i] * b[i];
        count += xor_bit;
    }
    return count;
}
```

**Example:**
```
a: 0 0 0 0 1 1 1 1  (0x0F)
b: 1 1 1 1 0 0 0 0  (0xF0)
hamming = 8  (all bits differ)
```

---

## 8-Bit Parallel Shapes

For operating on 8 bits at once:

```c
// 8-bit AND
static inline void trix_shape_and_8bit(
    const float* a, const float* b, float* result
) {
    for (int i = 0; i < 8; i++) {
        result[i] = a[i] * b[i];
    }
}

// 8-bit OR
static inline void trix_shape_or_8bit(
    const float* a, const float* b, float* result
) {
    for (int i = 0; i < 8; i++) {
        result[i] = a[i] + b[i] - a[i] * b[i];
    }
}

// 8-bit XOR
static inline void trix_shape_xor_8bit(
    const float* a, const float* b, float* result
) {
    for (int i = 0; i < 8; i++) {
        result[i] = a[i] + b[i] - 2.0f * a[i] * b[i];
    }
}

// 8-bit NOT
static inline void trix_shape_not_8bit(
    const float* a, float* result
) {
    for (int i = 0; i < 8; i++) {
        result[i] = 1.0f - a[i];
    }
}
```

---

## The Shape Table

All 16 shapes used by the 6502 ALU:

| ID | Name | Shape | Parameters |
|----|------|-------|------------|
| 0 | RIPPLE_ADD | 8× full adder chain | 0 |
| 1 | RIPPLE_SUB | 8× full adder with NOT | 0 |
| 2 | AND | 8× parallel AND | 0 |
| 3 | OR | 8× parallel OR | 0 |
| 4 | XOR | 8× parallel XOR | 0 |
| 5 | ASL | Shift left | 0 |
| 6 | LSR | Shift right | 0 |
| 7 | ROL | Rotate left | 0 |
| 8 | ROR | Rotate right | 0 |
| 9 | INC | Add 1 | 0 |
| 10 | DEC | Subtract 1 | 0 |

**Total learnable parameters: 0**

The shapes are mathematical truths. They don't need to be learned.

---

## The Principle

> *"Don't learn what you can derive."*

Every shape in this file is a polynomial. Every polynomial evaluates to the exact correct answer on binary inputs. There's nothing to learn - just math to apply.

The only thing that needs learning is **which shape to use**. That's routing. And once routing is learned, it's frozen too.

```
Traditional ML: Learn the function
TriX: Freeze the function, learn the routing
```

---

## Summary

| Category | Shapes | Total Params |
|----------|--------|--------------|
| Logic | XOR, AND, OR, NOT, NAND, NOR, XNOR | 0 |
| Arithmetic | FULL_ADDER, RIPPLE_ADD, RIPPLE_SUB, INC, DEC | 0 |
| Shift | ASL, LSR, ROL, ROR | 0 |
| Distance | HAMMING | 0 |
| **Total** | **16 shapes** | **0 params** |

Frozen. Exact. Forever.
