# Hollywood Squares Foundry - Quickstart Guide

Get started with native GPU neural networks in 5 minutes.

## Installation

```bash
# Requirements
pip install cupy-cuda12x numpy

# Optional: PyTorch for comparison
pip install torch
```

## 1. Basic Training

```python
from trix.foundry.native_training import NativeHollywoodSquares, Trainer
import cupy as cp

# Create model
model = NativeHollywoodSquares(
    d_model=128,      # Input/output dimension
    num_tiles=16,     # Number of expert tiles
    grid_size=8,      # Spline resolution
    lr=0.01,          # Learning rate
)

# Generate sample data
train_x = cp.random.randn(1000, 128).astype(cp.float32)
train_y = train_x * 0.9  # Simple regression target

# Create trainer and train
trainer = Trainer(model, loss_fn='mse')
for epoch in range(100):
    loss = trainer.train_step(train_x, train_y)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss:.6f}")
```

## 2. Batch Training with Full Loop

```python
# Generate larger dataset
train_x = cp.random.randn(10000, 128).astype(cp.float32)
train_y = train_x * 0.95 + cp.random.randn(10000, 128).astype(cp.float32) * 0.01

# Use batch training
trainer = Trainer(model, loss_fn='mse')
history = trainer.train(
    train_x=train_x,
    train_y=train_y,
    epochs=50,
    batch_size=256,
    verbose=True,
)

print(f"Final loss: {history['loss'][-1]:.6f}")
print(f"Training time: {sum(history['time']):.2f}s")
```

## 3. Save and Load Models

```python
# Save weights
model.save("my_model.npz")

# Load into new model
model2 = NativeHollywoodSquares(d_model=128, num_tiles=16)
model2.load("my_model.npz")

# Verify
output1 = model.forward(train_x[:10], save_for_backward=False)
output2 = model2.forward(train_x[:10], save_for_backward=False)
assert cp.allclose(output1, output2)
```

## 4. Inference Only

```python
from trix.foundry.hollywood_squares_emergence import HollywoodSquaresEmergence

# Create inference model (random weights)
inference = HollywoodSquaresEmergence(
    d_model=128,
    num_tiles=16,
)

# Forward pass
x = cp.random.randn(1000, 128).astype(cp.float32)
output = inference.forward(x)  # Standard kernel
output_vec = inference.forward_vectorized(x)  # Faster for large batches

# Benchmark
results = inference.benchmark(batch_size=100000)
print(f"Throughput: {results['throughput']/1e6:.1f}M tokens/sec")
```

## 5. Classification with Cross-Entropy

```python
from trix.foundry.native_training import cross_entropy_loss

# Create model for 10-class classification
model = NativeHollywoodSquares(d_model=128, num_tiles=16, lr=0.01)

# Generate classification data
train_x = cp.random.randn(1000, 128).astype(cp.float32)
train_y = cp.random.randint(0, 10, (1000,)).astype(cp.int32)

# Training loop (manual)
for epoch in range(100):
    # Forward
    logits = model.forward(train_x)

    # Compute loss
    loss, d_logits = cross_entropy_loss(logits, train_y)

    # Backward
    model.backward(d_logits)

    # Update
    model.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {float(loss):.4f}")
```

## 6. Running Tests

```bash
# Run all tests
PYTHONPATH=src python -m pytest src/trix/foundry/test_hollywood_squares.py -v

# Run A/B tests (Native vs PyTorch)
PYTHONPATH=src python src/trix/foundry/ab_test_harness.py

# Run benchmarks
PYTHONPATH=src python src/trix/foundry/benchmark_harness.py
```

## Common Patterns

### Early Stopping

```python
best_loss = float('inf')
patience = 10
no_improve = 0

for epoch in range(1000):
    loss = trainer.train_step(train_x, train_y)

    if loss < best_loss * 0.99:  # 1% improvement threshold
        best_loss = loss
        no_improve = 0
        model.save("best_model.npz")
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

### Learning Rate Scheduling

```python
initial_lr = 0.01
decay_rate = 0.95

for epoch in range(100):
    # Decay learning rate
    current_lr = initial_lr * (decay_rate ** epoch)
    model.optimizer.lr = current_lr

    loss = trainer.train_step(train_x, train_y)
```

### Memory Management

```python
import cupy as cp

# Clear GPU memory between runs
cp.get_default_memory_pool().free_all_blocks()

# Check memory usage
mempool = cp.get_default_memory_pool()
print(f"GPU memory used: {mempool.used_bytes() / 1e6:.1f} MB")
```

## Performance Tips

1. **Batch Size**: Use 256-1024 for optimal throughput
2. **d_model**: Must be multiple of 16, max 128
3. **num_tiles**: 16-32 works well for most cases
4. **Warmup**: First few iterations are slow (kernel compilation)

## Expected Results

| Metric | Value |
|--------|-------|
| Training Speed | 300K+ samples/sec |
| Inference Speed | 40M+ tokens/sec |
| PyTorch Speedup | 85x training, 8x inference |
| Convergence | Loss reaches near-zero |

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size or clear memory
cp.get_default_memory_pool().free_all_blocks()
```

### Kernel Compilation Slow
First run compiles CUDA kernels. Subsequent runs are faster.

### NaN in Output
Check for:
- Learning rate too high (try 0.001)
- Input not normalized (use mean=0, std=1)
- Extreme input values (clip to reasonable range)

## Next Steps

- Read [API.md](API.md) for complete function reference
- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Read [GAPS.md](GAPS.md) for known limitations
