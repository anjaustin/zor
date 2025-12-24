# Frozen Geometry: A Gentle Introduction

Welcome. This tutorial assumes nothing. We'll build understanding step by step.

**Time:** 30 minutes
**Prerequisites:** Basic Python

---

## Part 1: What Are We Doing?

Neural networks are amazing at fuzzy tasks. "Is this a dog?" "What's the sentiment?" They learn patterns and generalize.

But what about exact tasks?

```python
# Neural network: "The answer is probably 55, maybe 56"
# Calculator: "The answer is 55"
```

For arithmetic, logic, and cryptography, "probably" isn't good enough.

**Frozen geometry** lets you embed exact computation into neural networks. The computation isn't learned—it's mathematically guaranteed.

---

## Part 2: The Magic Formula

Here's the XOR truth table you learned in school:

```
a  b  | a XOR b
------+--------
0  0  |   0
0  1  |   1
1  0  |   1
1  1  |   0
```

Now here's a polynomial:

```
f(a, b) = a + b - 2ab
```

Let's test it:

```
f(0, 0) = 0 + 0 - 2(0)(0) = 0  ✓
f(0, 1) = 0 + 1 - 2(0)(1) = 1  ✓
f(1, 0) = 1 + 0 - 2(1)(0) = 1  ✓
f(1, 1) = 1 + 1 - 2(1)(1) = 0  ✓
```

The polynomial gives the same answers as XOR. On binary inputs {0, 1}, they're identical.

**This isn't approximation. It's mathematical identity.**

---

## Part 3: The Building Blocks

Every Boolean function can be built from a few primitives:

| Gate | Formula | Why it works |
|------|---------|--------------|
| AND  | `ab` | Only 1 when both are 1 |
| OR   | `a + b - ab` | 1 unless both are 0 |
| NOT  | `1 - a` | Flips 0↔1 |
| XOR  | `a + b - 2ab` | 1 when different |

Try them yourself:

```python
def AND(a, b): return a * b
def OR(a, b): return a + b - a * b
def NOT(a): return 1 - a
def XOR(a, b): return a + b - 2 * a * b

# Test XOR
for a in [0, 1]:
    for b in [0, 1]:
        print(f"XOR({a}, {b}) = {XOR(a, b)}")
```

Output:
```
XOR(0, 0) = 0
XOR(0, 1) = 1
XOR(1, 0) = 1
XOR(1, 1) = 0
```

---

## Part 4: Building a Full Adder

A full adder takes three bits (a, b, carry_in) and produces two outputs (sum, carry_out).

**Truth table:**
```
a  b  cin | sum  cout
----------+-----------
0  0   0  |  0    0
0  0   1  |  1    0
0  1   0  |  1    0
0  1   1  |  0    1
1  0   0  |  1    0
1  0   1  |  0    1
1  1   0  |  0    1
1  1   1  |  1    1
```

**As polynomials:**
```python
def full_adder(a, b, cin):
    # Sum: XOR of all three inputs
    p = XOR(a, b)
    sum_bit = XOR(p, cin)

    # Carry: majority function
    carry = OR(AND(a, b), AND(cin, p))

    return sum_bit, carry
```

Test it:
```python
for a in [0, 1]:
    for b in [0, 1]:
        for cin in [0, 1]:
            s, cout = full_adder(a, b, cin)
            print(f"add({a}, {b}, {cin}) = sum:{s}, carry:{cout}")
```

Every output matches the truth table. Exactly.

---

## Part 5: Why This Matters

We just built an adder from polynomials. Notice:

1. **No learning.** We didn't train anything.
2. **100% accurate.** Every input gives the right output.
3. **Differentiable.** Polynomials have gradients.
4. **Composable.** Chain 8 adders → 8-bit addition.

This is a **frozen shape**: a polynomial that computes a specific function exactly.

---

## Part 6: Using the Library

The FrozenFoundry library automates this. Here's the simplest example:

```python
from trix.foundry import FrozenFoundry

# Create a foundry for 8-bit operations
foundry = FrozenFoundry(bit_width=8)

# Define your function
def my_add(a, b, carry):
    result = a + b + carry
    return result & 0xFF, int(result > 255)

# Register it
foundry.register("add", my_add)

# Build the frozen model
result = foundry.build()

print(f"Training steps: {result.training_steps}")  # 0
print(f"Accuracy: {result.accuracy}")              # 1.0
```

The output:
```
Training steps: 0
Accuracy: 1.0
```

**Zero training. Perfect accuracy.**

---

## Part 7: The Routing/Execution Split

Here's the key architectural insight:

**Traditional neural networks** blend everything:
- Weights determine both WHAT to compute and HOW

**Frozen architecture** separates concerns:
- **Routing:** Learned. Which operation should I use?
- **Execution:** Frozen. How do I compute that operation?

```
Input → [Learned Router] → selects → [Frozen Shape] → Output
              ↑                           ↑
         Trainable                    Exact, verified
```

The router can be trained with gradient descent. The shapes are fixed polynomials.

---

## Part 8: What "Frozen" Means

A frozen shape is:

1. **A polynomial** that maps inputs to outputs
2. **Exact on binary inputs** (no approximation)
3. **Fixed** (the coefficients don't change during training)
4. **Verifiable** (you can prove it's correct)

The "frozen" metaphor: the shape is crystallized, not fluid. It won't change. It can't be wrong.

---

## Part 9: The Library at a Glance

```python
from trix.foundry import FrozenFoundry

# Create a foundry (specify bit width)
foundry = FrozenFoundry(bit_width=8)

# Register operations (provide truth function)
foundry.register("add", lambda a, b, c: ((a + b + c) & 0xFF, int(a + b + c > 255)))
foundry.register("xor", lambda a, b, c: (a ^ b, 0))

# Build (finds matching shapes, trains routing if needed)
result = foundry.build()

# Validate (test on many samples)
accuracy = foundry.validate(n_samples=10000)

# Export (save the frozen model)
foundry.export("my_model.pt")
foundry.export_onnx("my_model.onnx")
```

---

## Part 10: Putting It Together

You've learned:

1. **Boolean functions are polynomials.** XOR = a + b - 2ab.
2. **Polynomials compose.** Full adder from XOR, AND, OR.
3. **This is exact, not approximate.** 100%, not 99.9%.
4. **The library automates it.** Define function, build, validate.
5. **Routing is learned, execution is frozen.** Best of both worlds.

---

## What's Next?

| If you want to... | Read this |
|-------------------|-----------|
| See the limits | [HONEST_LIMITS.md](HONEST_LIMITS.md) |
| Understand why this matters | [WHY_CARE.md](WHY_CARE.md) |
| Go deeper on theory | [THEORY.md](THEORY.md) |
| See real examples | `examples/` directory |
| Hands-on practice | `notebooks/freshman_tutorial.ipynb` |

---

## Glossary

| Term | Meaning |
|------|---------|
| **Frozen shape** | A polynomial that computes exactly on binary inputs |
| **Router** | The learned component that selects which shape to use |
| **Foundry** | The tool that builds frozen models |
| **Bit width** | Number of bits per operand (8 = 0-255) |
| **Truth function** | The ground truth definition of an operation |

---

## Quick Reference

```python
# The core primitives (memorize these)
XOR(a, b) = a + b - 2*a*b
AND(a, b) = a * b
OR(a, b)  = a + b - a*b
NOT(a)    = 1 - a

# Full adder
sum   = XOR(XOR(a, b), cin)
carry = OR(AND(a, b), AND(cin, XOR(a, b)))

# The library
from trix.foundry import FrozenFoundry
foundry = FrozenFoundry(bit_width=8)
foundry.register("op_name", truth_function)
result = foundry.build()
```

---

*"Computation is topology. Learning is routing."*

*Welcome to frozen geometry.*
