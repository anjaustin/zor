# API Reference

Complete API documentation for Hollywood Squares Foundry.

## Table of Contents

1. [Training API](#training-api)
2. [Inference API](#inference-api)
3. [Loss Functions](#loss-functions)
4. [Optimizer](#optimizer)
5. [Testing Utilities](#testing-utilities)

---

## Training API

### NativeHollywoodSquares

Complete trainable model with forward and backward passes.

```python
from trix.foundry.native_training import NativeHollywoodSquares
```

#### Constructor

```python
NativeHollywoodSquares(
    d_model: int = 128,
    num_tiles: int = 16,
    grid_size: int = 16,
    position_spread: float = 2.0,
    lr: float = 0.001,
)
```

**Parameters:**
- `d_model`: Model dimension. Must be multiple of 16, max 128.
- `num_tiles`: Number of expert tiles. Max 32.
- `grid_size`: Spline grid resolution per tile.
- `position_spread`: B-spline spreading width for spatial routing.
- `lr`: Learning rate for Adam optimizer.

#### Methods

##### forward

```python
def forward(
    self,
    x: cp.ndarray,
    positions: cp.ndarray = None,
    save_for_backward: bool = True,
) -> cp.ndarray
```

Forward pass through the model.

**Parameters:**
- `x`: Input tensor, shape `[batch_size, d_model]`, dtype `float32`
- `positions`: Position indices, shape `[batch_size]`, dtype `float32`. Default: `[0, 1, 2, ...]`
- `save_for_backward`: Whether to cache intermediates for backward pass.

**Returns:**
- Output tensor, shape `[batch_size, d_model]`, dtype `float32`

##### backward

```python
def backward(self, d_output: cp.ndarray) -> cp.ndarray
```

Backward pass computing gradients.

**Parameters:**
- `d_output`: Gradient of loss w.r.t. output, shape `[batch_size, d_model]`

**Returns:**
- Gradient of loss w.r.t. input, shape `[batch_size, d_model]`

**Side Effects:**
- Accumulates gradients in `self.d_directions` and `self.d_spline_coeffs`

##### step

```python
def step(self) -> None
```

Update weights using accumulated gradients and Adam optimizer.

##### save

```python
def save(self, path: str) -> None
```

Save model weights to NumPy npz file.

**Parameters:**
- `path`: File path (should end in `.npz`)

##### load

```python
def load(self, path: str) -> None
```

Load model weights from NumPy npz file.

**Parameters:**
- `path`: File path to saved model

##### param_count

```python
def param_count(self) -> int
```

Get total number of trainable parameters.

**Returns:**
- Number of parameters (directions + spline_coeffs)

---

### Trainer

Training loop manager.

```python
from trix.foundry.native_training import Trainer
```

#### Constructor

```python
Trainer(
    model: NativeHollywoodSquares,
    loss_fn: str = 'mse',
)
```

**Parameters:**
- `model`: Model to train
- `loss_fn`: Loss function, either `'mse'` or `'cross_entropy'`

#### Methods

##### train_step

```python
def train_step(
    self,
    x: cp.ndarray,
    target: cp.ndarray,
    positions: cp.ndarray = None,
) -> float
```

Single training step (forward, loss, backward, update).

**Parameters:**
- `x`: Input batch, shape `[batch_size, d_model]`
- `target`: Target batch, shape `[batch_size, d_model]` for MSE or `[batch_size]` for cross-entropy
- `positions`: Optional position indices

**Returns:**
- Loss value (float)

##### train

```python
def train(
    self,
    train_x: cp.ndarray,
    train_y: cp.ndarray,
    epochs: int = 100,
    batch_size: int = 256,
    verbose: bool = True,
) -> Dict
```

Full training loop with batching and shuffling.

**Parameters:**
- `train_x`: Training inputs, shape `[n_samples, d_model]`
- `train_y`: Training targets
- `epochs`: Number of training epochs
- `batch_size`: Batch size
- `verbose`: Whether to print progress

**Returns:**
- History dict with `'loss'` and `'time'` lists

---

## Inference API

### HollywoodSquaresEmergence

Optimized inference with weight compression.

```python
from trix.foundry.hollywood_squares_emergence import HollywoodSquaresEmergence
```

#### Constructor

```python
HollywoodSquaresEmergence(
    d_model: int = 128,
    num_tiles: int = 16,
    grid_size: int = 16,
    position_spread: float = 2.0,
)
```

**Parameters:**
- Same as `NativeHollywoodSquares`

**Note:** Weights are randomly initialized. For trained weights, use `NativeHollywoodSquares` and export.

#### Methods

##### forward

```python
def forward(self, x: cp.ndarray, positions: cp.ndarray = None) -> cp.ndarray
```

Standard forward pass.

##### forward_vectorized

```python
def forward_vectorized(self, x: cp.ndarray, positions: cp.ndarray = None) -> cp.ndarray
```

Vectorized forward pass using float4 memory access. Faster for large batches.

##### benchmark

```python
def benchmark(self, batch_size: int = 100000) -> dict
```

Benchmark throughput.

**Returns:**
```python
{
    "batch_size": int,
    "throughput": float,        # tokens/sec
    "throughput_vec": float,    # vectorized tokens/sec
    "time_us": float,           # best time in microseconds
}
```

---

## Loss Functions

### mse_loss

```python
from trix.foundry.native_training import mse_loss

def mse_loss(pred: cp.ndarray, target: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]
```

Mean Squared Error loss.

**Parameters:**
- `pred`: Predictions, shape `[batch_size, d_model]`
- `target`: Targets, shape `[batch_size, d_model]`

**Returns:**
- `loss`: Scalar loss value
- `grad`: Gradient w.r.t. predictions, shape `[batch_size, d_model]`

**Formula:**
```
loss = mean((pred - target)^2)
grad = 2 * (pred - target) / n
```

### cross_entropy_loss

```python
from trix.foundry.native_training import cross_entropy_loss

def cross_entropy_loss(logits: cp.ndarray, targets: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray]
```

Cross-entropy loss for classification.

**Parameters:**
- `logits`: Unnormalized scores, shape `[batch_size, num_classes]`
- `targets`: Class indices, shape `[batch_size]`, dtype `int`

**Returns:**
- `loss`: Scalar loss value
- `grad`: Gradient w.r.t. logits, shape `[batch_size, num_classes]`

**Formula:**
```
probs = softmax(logits)
loss = -mean(log(probs[targets]))
grad = probs - one_hot(targets)
```

---

## Optimizer

### AdamOptimizer

Pure CuPy Adam implementation.

```python
from trix.foundry.native_training import AdamOptimizer
```

#### Constructor

```python
AdamOptimizer(
    params: Dict[str, cp.ndarray],
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
)
```

**Parameters:**
- `params`: Dict of named parameters to optimize
- `lr`: Learning rate
- `beta1`: First moment decay rate
- `beta2`: Second moment decay rate
- `eps`: Numerical stability constant

#### Methods

##### step

```python
def step(self, grads: Dict[str, cp.ndarray]) -> None
```

Update parameters with gradients.

**Parameters:**
- `grads`: Dict of gradients matching `params` keys

**Algorithm:**
```python
t += 1
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
param -= lr * m_hat / (sqrt(v_hat) + eps)
```

---

## Testing Utilities

### ABTestHarness

A/B testing framework.

```python
from trix.foundry.ab_test_harness import ABTestHarness, TestConfig
```

#### TestConfig

```python
@dataclass
class TestConfig:
    d_model: int = 128
    num_tiles: int = 16
    grid_size: int = 8
    learning_rate: float = 0.01
    batch_size: int = 256
    random_seed: int = 42
    convergence_samples: int = 5000
    convergence_epochs: int = 50
    speed_samples: int = 10000
    speed_epochs: int = 20
    inference_samples: int = 100000
    inference_runs: int = 10
```

#### ABTestHarness Methods

```python
harness = ABTestHarness(config)

# Run individual tests
harness.test_convergence()    # Do both converge?
harness.test_training_speed() # Which trains faster?
harness.test_inference_speed() # Which infers faster?
harness.test_memory_usage()   # Which uses less memory?

# Run all tests
report = harness.run_all()    # Returns full report dict
```

### BenchmarkHarness

Reproducible benchmarking.

```python
from trix.foundry.benchmark_harness import BenchmarkHarness

harness = BenchmarkHarness(output_path=Path("results.json"), runs=10)
harness.run_all()
```

---

## Type Annotations

All functions use standard Python type hints:

```python
import cupy as cp
from typing import Dict, Tuple, Optional, List

# Common types
Tensor = cp.ndarray
Shape = Tuple[int, ...]
```

---

## Error Handling

### Common Errors

1. **Dimension mismatch**: `d_model` must match between model and input
2. **Shared memory overflow**: `num_tiles * d_model` must fit in shared memory
3. **CUDA OOM**: Reduce batch size or clear memory pool with `cp.get_default_memory_pool().free_all_blocks()`

### Debug Tips

```python
# Check GPU memory
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

# Verify tensor shapes
print(f"Input shape: {x.shape}, dtype: {x.dtype}")

# Force GPU sync for timing
cp.cuda.Stream.null.synchronize()
```
