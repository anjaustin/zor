# SYNTHESIZE: Pipeline Specification

> Phase 4 of the Lincoln Manifold Method
> Date: 2025-12-22

---

## Module: `trix.routing`

A complete pipeline for building routing-based neural models with frozen execution.

---

## File Structure

```
src/trix/routing/
    __init__.py          # Public API
    pipeline.py          # RoutingPipeline class
    router.py            # Router architectures
    shapes.py            # Built-in frozen shapes
    primitives.py        # Polynomial primitives (xor, and, or, not)
    validation.py        # Validation utilities
    export.py            # Export to various formats
```

---

## Core Classes

### RoutingPipeline

```python
class RoutingPipeline:
    def __init__(self, bit_width: int = 8, device: str = "cpu"):
        """Create a routing pipeline."""

    def add(self, name: str, shape: Callable, truth: Callable) -> None:
        """Register a shape with its truth function."""

    def train(self, max_epochs: int = 100, lr: float = 0.001,
              batch_size: int = 256, verbose: bool = True) -> TrainResult:
        """Train the router. Returns accuracy, epochs, time."""

    def validate(self, exhaustive: bool = True, n_samples: int = 100000) -> float:
        """Validate accuracy. Returns fraction correct."""

    def compute(self, a: int, b: int, op: str | int) -> int:
        """Compute single operation."""

    def compute_batch(self, a: Tensor, b: Tensor, op: Tensor) -> Tensor:
        """Compute batch of operations."""

    def save(self, path: str) -> None:
        """Save to PyTorch format."""

    def load(self, path: str) -> None:
        """Load from PyTorch format."""

    def to_onnx(self, path: str) -> None:
        """Export to ONNX format."""

    @property
    def num_shapes(self) -> int:
        """Number of registered shapes."""

    @property
    def router_params(self) -> int:
        """Number of trainable parameters (router only)."""
```

### TrainResult

```python
@dataclass
class TrainResult:
    accuracy: float      # Final accuracy (0.0 to 1.0)
    epochs: int          # Epochs to convergence
    time: float          # Training time in seconds
    router_params: int   # Trainable parameters
```

---

## Built-in Shapes

```python
from trix.routing import shapes

shapes.add_8bit(a_bits, b_bits) -> result_bits
shapes.sub_8bit(a_bits, b_bits) -> result_bits
shapes.xor_8bit(a_bits, b_bits) -> result_bits
shapes.and_8bit(a_bits, b_bits) -> result_bits
shapes.or_8bit(a_bits, b_bits) -> result_bits
shapes.not_8bit(a_bits) -> result_bits
shapes.inc_8bit(a_bits) -> result_bits
shapes.dec_8bit(a_bits) -> result_bits
shapes.shl_8bit(a_bits) -> result_bits  # Shift left by 1
shapes.shr_8bit(a_bits) -> result_bits  # Shift right by 1
```

---

## Polynomial Primitives

```python
from trix.routing import primitives

primitives.xor(a, b)  # a + b - 2ab
primitives.and_(a, b) # ab
primitives.or_(a, b)  # a + b - ab
primitives.not_(a)    # 1 - a
```

---

## Usage Example

```python
from trix.routing import RoutingPipeline, shapes

# Create pipeline
pipe = RoutingPipeline(bit_width=8)

# Register operations
pipe.add("add", shapes.add_8bit, truth=lambda a,b: (a+b) & 0xFF)
pipe.add("sub", shapes.sub_8bit, truth=lambda a,b: (a-b) & 0xFF)
pipe.add("xor", shapes.xor_8bit, truth=lambda a,b: a ^ b)
pipe.add("and", shapes.and_8bit, truth=lambda a,b: a & b)

# Train router
result = pipe.train(verbose=True)
# Output:
#   Training router (76 params)...
#   Epoch 1: 100.00%
#   Done in 4.2s

# Validate
acc = pipe.validate(exhaustive=True)
assert acc == 1.0  # 100%

# Use
assert pipe.compute(17, 38, "add") == 55
assert pipe.compute(100, 50, "sub") == 50
assert pipe.compute(0xFF, 0x0F, "xor") == 0xF0
assert pipe.compute(0xFF, 0x0F, "and") == 0x0F

# Export
pipe.save("calculator.pt")
pipe.to_onnx("calculator.onnx")
```

---

## Implementation Plan

### Phase 1: Core (Today)
- [ ] `primitives.py` - Polynomial primitives
- [ ] `shapes.py` - Built-in shapes (add, sub, xor, and)
- [ ] `pipeline.py` - RoutingPipeline core
- [ ] Basic training loop

### Phase 2: Validation
- [ ] Exhaustive validation
- [ ] Statistical validation
- [ ] Accuracy reporting

### Phase 3: Export
- [ ] PyTorch save/load
- [ ] ONNX export

### Phase 4: Polish
- [ ] More built-in shapes
- [ ] Better error messages
- [ ] Documentation

---

## Tests

```python
def test_pipeline_basic():
    pipe = RoutingPipeline(bit_width=8)
    pipe.add("add", shapes.add_8bit, truth=lambda a,b: (a+b) & 0xFF)
    pipe.add("xor", shapes.xor_8bit, truth=lambda a,b: a ^ b)

    result = pipe.train()
    assert result.accuracy >= 0.9999

def test_pipeline_exhaustive():
    pipe = RoutingPipeline(bit_width=8)
    # ... register shapes ...
    pipe.train()

    acc = pipe.validate(exhaustive=True)
    assert acc == 1.0

def test_pipeline_export():
    pipe = RoutingPipeline(bit_width=8)
    # ... register and train ...

    pipe.save("test.pt")
    pipe2 = RoutingPipeline.load("test.pt")
    assert pipe2.validate() == 1.0
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Lines to basic usage | < 10 |
| Training time (4 shapes) | < 10 seconds |
| Accuracy | 100% |
| Router params (4 shapes, 8-bit) | 76 |

---

*End of Phase 4 - SYNTHESIZE*

*Now: Build it.*
