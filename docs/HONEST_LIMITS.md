# Honest Limits: Where Frozen Geometry Fails

Every technology has boundaries. Here are ours.

We believe showing limits builds trust faster than hiding them.

---

## What CAN'T Be Frozen

### 1. Non-Deterministic Functions

If `f(x)` can return different values on different calls, there's no truth table.

```python
import random

def non_deterministic(a, b, c):
    return random.randint(0, 255), 0  # Different each time

# This CANNOT be frozen. There's no single correct answer.
```

**Why:** Frozen shapes encode a fixed mapping. Random functions have no fixed mapping.

### 2. Continuous Functions (Directly)

Frozen shapes work on binary inputs {0, 1}. A function like `sin(x)` over real numbers has infinitely many input/output pairs.

**Workaround:** Quantize the inputs.

```python
# Instead of sin(x) for x ∈ ℝ
# Use sin(x) for x ∈ {0, 1, 2, ..., 255} mapped to [0, 2π]

def quantized_sin(a, b, c):
    import math
    x = a / 256 * 2 * math.pi
    result = int((math.sin(x) + 1) / 2 * 255)
    return result, 0
```

This freezes perfectly, but you accept quantization error between sample points.

### 3. Non-Computable Functions

If a Turing machine can't compute it, neither can frozen geometry.

Examples:
- The halting problem
- Kolmogorov complexity
- Busy beaver function

**Why:** Frozen shapes are polynomials. Polynomials are computable.

---

## When NOT to Use This

### 1. When Approximation Is Fine

Image classification doesn't need 100% accuracy on any single pixel. Use standard neural networks.

```python
# Good for frozen geometry:
is_checksum_valid(data)  # Must be exact

# Not necessary for frozen geometry:
is_this_a_cat(image)  # Approximation is fine
```

### 2. When Input Space Is Huge

32-bit × 32-bit operations have 2^64 possible inputs. The frozen shape still works, but:
- Building takes longer
- Exhaustive verification is impractical

| Bit Width | Possible Inputs | Exhaustive Verify Time |
|-----------|-----------------|------------------------|
| 8-bit     | 65,536          | < 1 second |
| 16-bit    | 4 billion       | Minutes |
| 32-bit    | 2^64            | Years |

**Workaround:** Use statistical validation for large bit widths.

### 3. When Learning IS the Point

Frozen shapes don't generalize. They compute exactly what they're defined to compute.

```python
# Frozen geometry can't:
# - Learn to recognize new patterns
# - Generalize from examples
# - Adapt to new data

# Frozen geometry can:
# - Compute exact functions
# - Execute verified logic
# - Provide guaranteed correctness
```

If you want adaptation, use learned components. If you want guarantees, use frozen.

---

## The Lookup Table Question

**Q: Isn't this just a fancy lookup table?**

**A:** Related, but different in important ways.

| Property | Lookup Table | Frozen Shape |
|----------|--------------|--------------|
| Storage | O(2^n) entries | O(n) polynomial terms |
| Composable | Not naturally | Yes, polynomials compose |
| Differentiable | No | Yes, gradients flow |
| Embeddable in NN | Awkward | Native |
| Formally verifiable | Yes | Yes |

The key insight isn't "use polynomials instead of tables." It's "exact computation can live inside differentiable architectures."

---

## Computational Costs

### Building Time

| Bit Width | Operations | Build Time |
|-----------|------------|------------|
| 8-bit     | 10         | ~500ms |
| 16-bit    | 10         | ~1s |
| 32-bit    | 10         | ~2s |

Building scales linearly with bit width (for ripple-carry operations).

### Memory

Frozen shapes are memory-efficient:

```
8-bit XOR:  ~50 bytes (polynomial coefficients)
8-bit adder: ~200 bytes (8 chained full adders)
Full 6502:  ~326 bytes (16 shapes + routing)
```

### Inference Time

Frozen shapes are fast:
- No weight lookups
- Polynomial evaluation is O(n) for n bits
- GPU-friendly (parallel polynomial evaluation)

---

## Shape Discovery

**Current limitation:** You must define shapes that match your operations.

If your operation doesn't match any built-in shape, you need to:
1. Build a custom shape, OR
2. Accept that routing accuracy will be imperfect

**Future direction:** Automated shape discovery from truth tables.

---

## Floating Point

IEEE 754 floating point has:
- Special values (NaN, Inf, -0)
- Rounding modes
- Denormalized numbers

**Current status:** Not fully supported.

**Workaround:** Use fixed-point representation, which freezes perfectly.

---

## When Routing Fails

If shapes don't match exactly, the router must be trained. Training accuracy depends on:

1. **Shape coverage:** Do you have shapes close to your functions?
2. **Training data:** Did you sample the input space well?
3. **Router capacity:** Is the router expressive enough?

**Symptom:** Accuracy < 100% after training.

**Solution:** Add custom shapes that match your operations exactly.

---

## Known Issues

1. **Error messages could be clearer.** We're working on it.

2. **ONNX export for complex shapes.** Some custom shapes don't export cleanly.

3. **No automated testing for custom shapes.** You must verify them yourself.

4. **Documentation gaps.** This is improving (you're reading one improvement).

---

## Reporting Problems

If you find:
- An operation that should freeze but doesn't
- A bug in the library
- A documentation error

Please open an issue. We want to know.

---

## Still Skeptical?

Good. That's the right attitude.

```bash
python examples/prove_it.py
```

Define the weirdest function you can imagine. Watch it freeze.

If it doesn't freeze, tell us. That's either a bug or a limit we haven't documented.

---

## Summary

| Works Great | Doesn't Work | Works With Caveats |
|-------------|--------------|-------------------|
| Deterministic logic | Random functions | Large bit widths (use sampling) |
| Bit manipulation | Continuous on ℝ | Floating point (use fixed-point) |
| Arithmetic | Non-computable | Operations without matching shapes |
| Cryptographic primitives | Learning/generalization | |

---

*"The honest salesman shows you the dents."*

*We'd rather you know the limits now than discover them in production.*
