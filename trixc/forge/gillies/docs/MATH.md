# GILLIES Mathematical Foundations

*The eternal polynomials that make computation substrate-agnostic.*

---

## Zhegalkin Polynomials

In 1927, Ivan Zhegalkin proved that every Boolean function can be uniquely represented as a polynomial over GF(2) (the field with two elements: 0 and 1).

The key insight: **Boolean algebra IS polynomial arithmetic** when you interpret:
- Addition modulo 2 as XOR
- Multiplication as AND

GILLIES extends this to the real numbers, preserving:
- **Exactness** for binary inputs {0, 1}
- **Differentiability** for continuous inputs [0, 1]

---

## The Frozen Shapes

### XOR (Exclusive OR)

**Boolean**: `a ⊕ b`
**Polynomial**: `a + b - 2ab`

Derivation:
```
In GF(2): a ⊕ b = a + b (mod 2)
For reals: We need f(0,0)=0, f(0,1)=1, f(1,0)=1, f(1,1)=0

The unique polynomial satisfying this:
f(a,b) = a + b - 2ab

Verify:
  f(0,0) = 0 + 0 - 0 = 0 ✓
  f(0,1) = 0 + 1 - 0 = 1 ✓
  f(1,0) = 1 + 0 - 0 = 1 ✓
  f(1,1) = 1 + 1 - 2 = 0 ✓
```

**Gradient**:
```
∂f/∂a = 1 - 2b
∂f/∂b = 1 - 2a
```

At the corners: gradients are ±1 (strong signal).
At the center (0.5, 0.5): gradients are 0 (saddle point).

---

### AND (Logical AND)

**Boolean**: `a ∧ b`
**Polynomial**: `ab`

This is the simplest—multiplication IS conjunction.

```
Verify:
  f(0,0) = 0·0 = 0 ✓
  f(0,1) = 0·1 = 0 ✓
  f(1,0) = 1·0 = 0 ✓
  f(1,1) = 1·1 = 1 ✓
```

**Gradient**:
```
∂f/∂a = b
∂f/∂b = a
```

Gradient magnitude depends on the other input. When both inputs are 1, gradients are strongest.

---

### OR (Logical OR)

**Boolean**: `a ∨ b`
**Polynomial**: `a + b - ab`

Derivation from De Morgan's law:
```
a ∨ b = ¬(¬a ∧ ¬b)
      = 1 - (1-a)(1-b)
      = 1 - (1 - a - b + ab)
      = a + b - ab
```

```
Verify:
  f(0,0) = 0 + 0 - 0 = 0 ✓
  f(0,1) = 0 + 1 - 0 = 1 ✓
  f(1,0) = 1 + 0 - 0 = 1 ✓
  f(1,1) = 1 + 1 - 1 = 1 ✓
```

**Gradient**:
```
∂f/∂a = 1 - b
∂f/∂b = 1 - a
```

---

### NOT (Logical NOT)

**Boolean**: `¬a`
**Polynomial**: `1 - a`

The simplest inversion.

```
Verify:
  f(0) = 1 - 0 = 1 ✓
  f(1) = 1 - 1 = 0 ✓
```

**Gradient**:
```
∂f/∂a = -1
```

Constant negative gradient—always pushing away from the input.

---

### NAND (NOT AND)

**Boolean**: `¬(a ∧ b)`
**Polynomial**: `1 - ab`

```
Verify:
  f(0,0) = 1 - 0 = 1 ✓
  f(0,1) = 1 - 0 = 1 ✓
  f(1,0) = 1 - 0 = 1 ✓
  f(1,1) = 1 - 1 = 0 ✓
```

NAND is universal—any Boolean function can be built from NANDs alone.

---

### NOR (NOT OR)

**Boolean**: `¬(a ∨ b)`
**Polynomial**: `1 - a - b + ab`

```
Derivation:
  ¬(a ∨ b) = 1 - (a + b - ab)
           = 1 - a - b + ab
```

NOR is also universal.

---

### XNOR (NOT XOR)

**Boolean**: `¬(a ⊕ b)` (equivalence)
**Polynomial**: `1 - a - b + 2ab`

```
Derivation:
  ¬(a ⊕ b) = 1 - (a + b - 2ab)
           = 1 - a - b + 2ab
```

Returns 1 when inputs are equal.

---

## Mathematical Properties

### De Morgan's Laws

**First Law**: `¬(a ∧ b) = ¬a ∨ ¬b`

```
Left side (NAND):
  1 - ab

Right side (OR of NOTs):
  (1-a) + (1-b) - (1-a)(1-b)
  = 2 - a - b - (1 - a - b + ab)
  = 2 - a - b - 1 + a + b - ab
  = 1 - ab ✓
```

**Second Law**: `¬(a ∨ b) = ¬a ∧ ¬b`

```
Left side (NOR):
  1 - a - b + ab

Right side (AND of NOTs):
  (1-a)(1-b)
  = 1 - a - b + ab ✓
```

These identities hold **exactly** for all real values, not just binary.

---

### Double Negation

`¬(¬a) = a`

```
Left side:
  1 - (1 - a)
  = 1 - 1 + a
  = a ✓
```

Holds exactly for all real values.

---

### XOR Expansion (Binary Only)

`a ⊕ b = (a ∧ ¬b) ∨ (¬a ∧ b)`

This identity holds for binary {0, 1} inputs but **not** for continuous values.

```
For binary inputs:
  Left:  a + b - 2ab
  Right: a(1-b) + (1-a)b - a(1-b)(1-a)b

At a=0, b=0: Left=0, Right=0 ✓
At a=0, b=1: Left=1, Right=1 ✓
At a=1, b=0: Left=1, Right=1 ✓
At a=1, b=1: Left=0, Right=0 ✓

For a=0.5, b=0.5:
  Left:  0.5 + 0.5 - 2(0.25) = 0.5
  Right: 0.25 + 0.25 - 0.25·0.25 = 0.4375 ≠ 0.5
```

The polynomials are different, so they diverge for non-binary inputs. This is expected—the XOR polynomial is the **minimal** polynomial that matches Boolean XOR.

---

## Composition

Shapes compose by feeding outputs to inputs.

### Half Adder

```
sum   = a ⊕ b
carry = a ∧ b
```

Two shapes, two outputs. For binary inputs:
```
a=0, b=0: sum=0, carry=0  (0+0=00)
a=0, b=1: sum=1, carry=0  (0+1=01)
a=1, b=0: sum=1, carry=0  (1+0=01)
a=1, b=1: sum=0, carry=1  (1+1=10)
```

### Full Adder

```
sum   = a ⊕ b ⊕ cin
carry = (a ∧ b) ∨ ((a ⊕ b) ∧ cin)
```

Three inputs, two outputs. The fundamental building block of arithmetic.

For polynomial representation:
```
Let t = a + b - 2ab  (the a⊕b intermediate)

sum   = t + cin - 2·t·cin
      = (a + b - 2ab) + cin - 2(a + b - 2ab)cin
      = a + b - 2ab + cin - 2cin·a - 2cin·b + 4ab·cin

carry = ab + t·cin - ab·t·cin
      = ab + (a + b - 2ab)cin - ab(a + b - 2ab)cin
```

Complex, but exact for binary inputs.

---

### Ripple Adder

Chain full adders:
```
[FA0] a[0], b[0], 0    → sum[0], c0
[FA1] a[1], b[1], c0   → sum[1], c1
[FA2] a[2], b[2], c1   → sum[2], c2
[FA3] a[3], b[3], c2   → sum[3], c3 (overflow)
```

A 4-bit adder = 4 full adders = 20 shape invocations.
GILLIES proves all 256 possible 4-bit additions are correct.

---

## The Learning Connection

Why polynomials? Because they're **differentiable**.

### Neural Networks as Shape Graphs

A neural network layer:
```
y = σ(Wx + b)
```

Can be expressed as shapes:
```
MUL: weight × input
ADD: sum terms
ADD: bias
// followed by activation (a shape or composition of shapes)
```

The shapes are the same whether training (need gradients) or inference (need speed).

### Gradient Flow

For training, compute gradients via chain rule:

```
If z = f(a, b) and L = loss(z), then:
  ∂L/∂a = ∂L/∂z · ∂z/∂a
  ∂L/∂b = ∂L/∂z · ∂z/∂b
```

Each shape has known gradients:
```
XOR: ∂z/∂a = 1-2b,  ∂z/∂b = 1-2a
AND: ∂z/∂a = b,     ∂z/∂b = a
OR:  ∂z/∂a = 1-b,   ∂z/∂b = 1-a
NOT: ∂z/∂a = -1
```

Backpropagation flows gradients backward through the shape graph.

### The Frozen 6502

TriXO's signature achievement: a 6502 CPU as a neural network.

```
Inputs:  Current CPU state (8192 bits)
Network: Learned routing (ternary weights)
Outputs: Next CPU state

Every instruction, every cycle, learned—not programmed.
98%+ accuracy on exhaustive tests.
```

The shapes that compute the 6502's logic are the same shapes GILLIES uses. The difference: GILLIES freezes them; the 6502 net learns routing between them.

---

## Continuous Semantics

For inputs in [0, 1], the shapes have meaningful continuous semantics:

| Shape | Continuous Interpretation |
|-------|--------------------------|
| AND   | Minimum (geometric mean) |
| OR    | Maximum-ish (not quite max) |
| XOR   | Distance from diagonal |
| NOT   | Complement |

These aren't exact min/max, but they're smooth approximations that:
1. Match exactly at {0, 1}
2. Provide useful gradients everywhere
3. Enable learning of Boolean functions

---

## Why These Specific Polynomials?

### Uniqueness

For each Boolean function f: {0,1}ⁿ → {0,1}, there is exactly ONE multilinear polynomial that matches it.

"Multilinear" means each variable appears at most to the first power (no a², no b³).

The shapes in GILLIES are the unique multilinear polynomials for their Boolean functions.

### Minimality

These polynomials are minimal in degree. XOR is degree 2 (the ab term). AND is degree 2. OR is degree 2. NOT is degree 1.

No simpler polynomials exist that match the Boolean truth tables.

### Universality

From XOR and AND alone (or from NAND alone, or from NOR alone), you can build any Boolean function. GILLIES provides more shapes for convenience, but XOR+AND is sufficient.

---

## Summary

| Shape | Polynomial | Degree | Universal? |
|-------|------------|--------|------------|
| XOR | a + b - 2ab | 2 | Yes (with AND) |
| AND | ab | 2 | Yes (with XOR) |
| OR | a + b - ab | 2 | No |
| NOT | 1 - a | 1 | No |
| NAND | 1 - ab | 2 | Yes (alone) |
| NOR | 1 - a - b + ab | 2 | Yes (alone) |
| XNOR | 1 - a - b + 2ab | 2 | No |
| ADD | a + b | 1 | No |
| SUB | a - b | 1 | No |
| MUL | ab | 2 | No |
| IDENTITY | a | 1 | No |

The mathematical foundation is complete. The shapes are frozen. The substrate doesn't matter.

---

*"Geometry is computation. Polynomials are the bridge."*
