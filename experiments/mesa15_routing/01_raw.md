# RAW: Routing Pipeline Design

> Phase 1 of the Lincoln Manifold Method
> Date: 2025-12-22

---

## What We Proved

The Calculator Test showed:
- 76-param router + frozen shapes = 100% accuracy
- 5,896-param MLP learning everything = 94.6% accuracy
- Routing converges in 1-2 epochs

**Learning IS Routing. Everything Else Can Be Frozen.**

---

## What's a "Complete Pipeline"?

From the user's perspective:
1. Define operations I want to compute
2. Get a deployable model that computes them exactly
3. Minimal training, maximum correctness

From the system's perspective:
1. Accept operation definitions (truth functions or shapes)
2. Build frozen shapes for each
3. Train a router to select shapes
4. Export deployable artifact

---

## Stream of Consciousness

The current FrozenFoundry does some of this but doesn't make the routing paradigm explicit.

A clean pipeline would be:

```
RoutingPipeline(
    shapes={
        "add": frozen_add,
        "sub": frozen_sub,
        "xor": frozen_xor,
        ...
    },
    router_input="full" | "opcode_only" | custom,
    bit_width=8
)

pipeline.train(data)  # Learns routing only
pipeline.validate()   # Tests exhaustively
pipeline.export("model.onnx")
```

Key design questions:
1. How does the user define shapes?
2. How does the router know which shape to use?
3. What's the training loop?
4. What gets exported?

---

## Shape Definition Options

### Option A: From Truth Functions
```python
pipeline.register("add", lambda a, b: (a + b) & 0xFF)
# System finds/builds matching frozen shape
```

### Option B: Direct Shape Definition
```python
pipeline.register_shape("add", frozen_add_polynomial)
# User provides the frozen shape directly
```

### Option C: Auto-Discovery
```python
pipeline.register("mystery", truth_function)
# System attempts to discover shape from truth table
```

We proved Option B works. Option A requires shape matching. Option C is research.

---

## Router Design Options

### Option A: Full Input Router
Router sees all input bits, learns to extract opcode.
- More flexible
- Slightly more params
- What we tested

### Option B: Opcode-Only Router
Router only sees opcode bits.
- Minimal params
- Requires explicit opcode in input format
- Trivially learnable

### Option C: Content-Addressable Router
Router computes signature of input, matches to shape.
- Most flexible
- Works when opcode isn't explicit
- More complex

---

## Training Considerations

- Router trains with frozen shapes in the loop
- Shapes provide gradients but don't update
- Loss is on final output
- Should converge very fast (we saw 1-2 epochs)

---

## Export Options

1. **PyTorch (.pt)**: Router weights + shape definitions
2. **ONNX (.onnx)**: Full graph including frozen shapes
3. **Pure Python**: Standalone module, no dependencies
4. **C/Header**: For embedded deployment

---

## What's Missing from Current Repo?

Looking at FrozenFoundry:
- Has shape library
- Has training
- But routing isn't the explicit paradigm

Need to build:
- Explicit RoutingPipeline class
- Clear shape registration API
- Router architecture options
- Validation suite
- Export formats

---

## First Instinct

Build `RoutingPipeline` that:
1. Takes shape definitions (frozen polynomials)
2. Builds minimal router
3. Trains routing only
4. Validates exhaustively
5. Exports to multiple formats

Make it dead simple:
```python
from trix.routing import RoutingPipeline

pipe = RoutingPipeline(bit_width=8)
pipe.add_shape("add", frozen_add)
pipe.add_shape("sub", frozen_sub)
pipe.add_shape("xor", frozen_xor)
pipe.add_shape("and", frozen_and)

pipe.train()  # Learns routing in seconds
pipe.validate()  # 100% on all 262,144 cases

pipe.export("calculator.onnx")
```

---

*End of Phase 1 - RAW*
