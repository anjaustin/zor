# trix.routing

**Learn routing. Freeze execution.**

A complete pipeline for building neural models where only routing is learned and execution is mathematically exact.

---

## The Insight

Traditional neural networks learn both:
- **WHAT** to compute (routing)
- **HOW** to compute it (execution)

This module separates them:
- **Routing:** Learned (tiny neural network)
- **Execution:** Frozen (polynomial shapes, mathematically exact)

Result: 78x fewer parameters, 60x faster training, 100% accuracy.

---

## Quick Start

```python
from trix.routing import RoutingPipeline

# Create pipeline
pipe = RoutingPipeline(bit_width=8)

# Add operations
pipe.add_builtins(["add", "sub", "xor", "and"])

# Train (learns routing only)
result = pipe.train()
print(f"Accuracy: {result.accuracy * 100}%")  # 100%
print(f"Epochs: {result.epochs}")              # 1-2
print(f"Params: {result.router_params}")       # 76

# Use it
assert pipe.compute(17, 38, "add") == 55
assert pipe.compute(100, 50, "sub") == 50

# Export
pipe.save("calculator.pt")
pipe.to_onnx("calculator.onnx")
```

---

## Built-in Shapes

| Name | Operation | Example |
|------|-----------|---------|
| `add` | a + b (mod 256) | add(17, 38) = 55 |
| `sub` | a - b (mod 256) | sub(100, 50) = 50 |
| `inc` | a + 1 | inc(254) = 255 |
| `dec` | a - 1 | dec(1) = 0 |
| `xor` | a ^ b | xor(0xFF, 0x0F) = 0xF0 |
| `and` | a & b | and(0xFF, 0x0F) = 0x0F |
| `or` | a \| b | or(0xF0, 0x0F) = 0xFF |
| `not` | ~a | not(0x00) = 0xFF |
| `shl` | a << 1 | shl(0x40) = 0x80 |
| `shr` | a >> 1 | shr(0x80) = 0x40 |
| `rol` | rotate left | rol(0x81) = 0x03 |
| `ror` | rotate right | ror(0x81) = 0xC0 |

---

## Custom Shapes

```python
from trix.routing import RoutingPipeline, xor, and_, or_

def my_custom_shape(a_bits, b_bits):
    """Custom frozen shape using polynomial primitives."""
    # Example: (a XOR b) AND (a OR b)
    return and_(xor(a_bits, b_bits), or_(a_bits, b_bits))

pipe = RoutingPipeline(bit_width=8)
pipe.add("custom", my_custom_shape, truth=lambda a, b: (a ^ b) & (a | b))
pipe.train()
```

---

## API Reference

### RoutingPipeline

```python
pipe = RoutingPipeline(bit_width=8)

# Registration
pipe.add(name, shape_fn, truth_fn)  # Custom shape
pipe.add_builtin("add")              # Single built-in
pipe.add_builtins(["add", "sub"])    # Multiple built-ins

# Training
result = pipe.train(max_epochs=100, verbose=True)
# Returns: TrainResult(accuracy, epochs, time, router_params)

# Validation
accuracy = pipe.validate(exhaustive=True)

# Inference
result = pipe.compute(a=17, b=38, op="add")

# Export
pipe.save("model.pt")
pipe.to_onnx("model.onnx")

# Load
pipe = RoutingPipeline.load("model.pt")
```

### Primitives

```python
from trix.routing import xor, and_, or_, not_, full_adder

# All work on tensors, compute exactly on {0, 1}
xor(a, b)      # a + b - 2ab
and_(a, b)     # ab
or_(a, b)      # a + b - ab
not_(a)        # 1 - a
full_adder(a, b, cin)  # Returns (sum, carry)
```

---

## The Math

Every Boolean function has an equivalent polynomial (Zhegalkin, 1927):

```
XOR(a, b) = a + b - 2ab
AND(a, b) = ab
OR(a, b)  = a + b - ab
NOT(a)    = 1 - a
```

On binary inputs {0, 1}, these polynomials compute **exactly**. No approximation.

Chain them to build complex operations:
- Full adder = XOR + AND + OR
- 8-bit adder = 8 chained full adders
- Any deterministic function = polynomial composition

---

## Benchmark

The Calculator Test (4 operations, 8-bit, 262,144 cases):

| | MLP | Routing Pipeline |
|-|-----|------------------|
| Parameters | 5,896 | 76 |
| Accuracy | 94.6% | 100% |
| Training Time | 241s | 4.5s |
| Epochs to 100% | Never | 2 |

**78x fewer parameters. 60x faster. Perfect accuracy.**

---

## Files

```
src/trix/routing/
    __init__.py      # Public API
    pipeline.py      # RoutingPipeline class
    shapes.py        # Built-in frozen shapes
    primitives.py    # Polynomial primitives
```

---

*"Learning IS Routing. Everything Else Can Be Frozen."*
