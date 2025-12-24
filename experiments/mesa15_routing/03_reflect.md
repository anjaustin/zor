# REFLECT: Resolving Design Tensions

> Phase 3 of the Lincoln Manifold Method
> Date: 2025-12-22

---

## Tension 1: Flexibility vs Simplicity

**Problem:** Do we support arbitrary router architectures or keep it minimal?

**Resolution:** Default to minimal (single linear layer). Allow override for advanced users.

```python
# Default: minimal
pipe = RoutingPipeline(bit_width=8)

# Advanced: custom router
pipe = RoutingPipeline(bit_width=8, router=MyCustomRouter())
```

Start simple. Complexity is opt-in.

---

## Tension 2: Shape Library vs Custom Shapes

**Problem:** Built-in shapes are convenient but limiting. Custom shapes are powerful but require expertise.

**Resolution:** Support both. Built-ins for common ops, custom for everything else.

```python
# Built-in (90% of use cases)
pipe.use_builtins(["add", "sub", "xor", "and", "or", "not"])

# Custom (power users)
pipe.add_shape("weird", my_weird_polynomial)
```

The library grows over time. Custom shapes become built-ins.

---

## Tension 3: Training Data Generation

**Problem:** User shouldn't have to generate training data manually.

**Resolution:** Pipeline generates data from shape definitions.

```python
# Shapes define truth functions
# Pipeline generates exhaustive or sampled data automatically

pipe.add_shape("add", frozen_add, truth=lambda a,b: (a+b) & 0xFF)
pipe.train()  # Auto-generates data from truth functions
```

---

## Tension 4: Opcode Encoding

**Problem:** How does the input encode which operation to use?

**Resolution:** Explicit opcode bits at the end of input.

```
Input format: [a_bits | b_bits | opcode_bits]
```

Opcode is ceil(log2(num_shapes)) bits:
- 2 shapes: 1 bit
- 4 shapes: 2 bits
- 8 shapes: 3 bits
- 16 shapes: 4 bits

Pipeline handles encoding/decoding.

---

## Tension 5: Batch vs Single Inference

**Problem:** Training needs batches. Inference often single items.

**Resolution:** Support both in API.

```python
# Single
result = pipe.compute(a=17, b=38, op="add")

# Batch
results = pipe.compute_batch(a_array, b_array, op_array)
```

Internally, single wraps to batch of 1.

---

## Tension 6: Where Does This Live?

**Problem:** New module or extend existing FrozenFoundry?

**Resolution:** New module `trix.routing`. Clean separation.

- `trix.foundry`: Shape building, the original paradigm
- `trix.routing`: Routing pipeline, the new paradigm

They share primitives but have different APIs.

---

## The Cleanest Design

```python
from trix.routing import RoutingPipeline, shapes

# 1. Create pipeline
pipe = RoutingPipeline(bit_width=8)

# 2. Add shapes (with truth functions for data generation)
pipe.add("add", shapes.add_8bit, truth=lambda a,b: (a+b) & 0xFF)
pipe.add("sub", shapes.sub_8bit, truth=lambda a,b: (a-b) & 0xFF)
pipe.add("xor", shapes.xor_8bit, truth=lambda a,b: a ^ b)
pipe.add("and", shapes.and_8bit, truth=lambda a,b: a & b)

# 3. Train (generates data, trains router)
result = pipe.train()
print(f"Accuracy: {result.accuracy}")  # 100%
print(f"Epochs: {result.epochs}")      # 1-2
print(f"Time: {result.time}s")         # seconds

# 4. Validate
pipe.validate(exhaustive=True)  # Tests all 262,144 cases

# 5. Export
pipe.save("calculator.pt")
pipe.to_onnx("calculator.onnx")
```

---

## Implementation Order

1. **Core shapes** (add, sub, xor, and, or, not)
2. **RoutingPipeline class** (registration, training, inference)
3. **Validation** (exhaustive + statistical)
4. **Export** (PyTorch first, then ONNX)
5. **Tests** (unit + integration)

---

## What We're NOT Building (Yet)

- Automatic shape discovery from truth tables
- Content-addressable routing (no explicit opcode)
- Multi-output operations (beyond single result + carry)
- Variable bit-width in same pipeline

These are future extensions.

---

*End of Phase 3 - REFLECT*
