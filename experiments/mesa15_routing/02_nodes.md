# NODES: Key Design Decisions

> Phase 2 of the Lincoln Manifold Method
> Date: 2025-12-22

---

## NODE 1: Core Architecture

```
Input → Router → Shape Selection → Frozen Shapes → Output
         ↑              ↓
    (learned)      (weighted sum)
```

- Router: Small neural network, only trainable component
- Shapes: Frozen polynomials, zero trainable parameters
- Selection: Softmax over shapes, differentiable

---

## NODE 2: Shape Registration

Two modes:

### Mode A: Built-in Shapes
```python
pipe.use_builtin("add")  # Uses library shape
pipe.use_builtin("xor")
```

### Mode B: Custom Shapes
```python
def my_shape(a_bits, b_bits):
    # Polynomial computation
    return result_bits

pipe.register("custom", my_shape)
```

---

## NODE 3: Input Format

Standard format: `[a_bits, b_bits, opcode_bits]`

```
a: 8 bits (operand 1)
b: 8 bits (operand 2)
op: log2(num_shapes) bits (operation selector)
```

Router receives full input, learns to use opcode.

---

## NODE 4: Router Architecture

Minimal viable router:
```python
router = nn.Sequential(
    nn.Linear(input_bits, num_shapes),
    nn.Softmax(dim=-1)
)
```

For N shapes with B-bit inputs:
- Parameters: B * N + N
- Example: 18 inputs, 4 shapes = 76 params

---

## NODE 5: Training Loop

```python
for batch in dataloader:
    # 1. Router produces shape weights
    weights = router(batch.input)  # [batch, num_shapes]

    # 2. All shapes compute (frozen, no grad needed for shapes)
    outputs = [shape(batch.a, batch.b) for shape in shapes]
    stacked = stack(outputs)  # [batch, num_shapes, output_bits]

    # 3. Weighted combination
    result = einsum('bn,bno->bo', weights, stacked)

    # 4. Loss on result
    loss = mse(result, batch.target)

    # 5. Only router updates
    loss.backward()
    optimizer.step()
```

---

## NODE 6: Validation

### Exhaustive (small inputs)
```python
for all possible inputs:
    assert model(input) == truth(input)
```

### Statistical (large inputs)
```python
for random sample of inputs:
    assert accuracy >= 0.9999
```

Threshold: exhaustive if < 1M cases, statistical otherwise.

---

## NODE 7: Export Formats

### PyTorch (.pt)
```python
torch.save({
    'router': router.state_dict(),
    'shapes': shape_definitions,
    'config': pipeline_config
}, 'model.pt')
```

### ONNX (.onnx)
```python
torch.onnx.export(full_model, example_input, 'model.onnx')
```

### Standalone Python
```python
# Generated file with no dependencies except numpy
def compute(a, b, op):
    # Inline router weights
    # Inline shape polynomials
    return result
```

---

## NODE 8: API Design

```python
from trix.routing import RoutingPipeline

# Create
pipe = RoutingPipeline(bit_width=8)

# Add shapes
pipe.add_shape("add", frozen_add)
pipe.add_shape("sub", frozen_sub)
pipe.add_shape("xor", frozen_xor)
pipe.add_shape("and", frozen_and)

# Or use builtins
pipe.use_builtins(["add", "sub", "xor", "and"])

# Train
result = pipe.train(max_epochs=100, verbose=True)
# Returns: accuracy, epochs, time

# Validate
accuracy = pipe.validate(exhaustive=True)

# Use
output = pipe(a=17, b=38, op=0)  # 55

# Export
pipe.export_pt("model.pt")
pipe.export_onnx("model.onnx")
pipe.export_python("model.py")
```

---

## NODE 9: Directory Structure

```
src/trix/routing/
    __init__.py
    pipeline.py      # RoutingPipeline class
    router.py        # Router architectures
    shapes.py        # Built-in frozen shapes
    export.py        # Export utilities
    validate.py      # Validation utilities
```

---

## NODE 10: Success Criteria

1. **Simplicity**: 10 lines to go from shapes to deployed model
2. **Speed**: Training completes in seconds
3. **Accuracy**: 100% on deterministic operations
4. **Portability**: Export to PyTorch, ONNX, standalone Python

---

*End of Phase 2 - NODES*
