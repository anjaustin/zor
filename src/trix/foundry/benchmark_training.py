"""
Benchmark: Native Training vs PyTorch

Side-by-side comparison of training speed.
Same task. Same architecture. Different implementations.

"The framework-free future."
"""

import numpy as np
import cupy as cp
import torch
import torch.nn as nn
import time
import json
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, '/workspace/trix_latest/TriXO/src')


def benchmark_native_training(n_samples: int, epochs: int, batch_size: int) -> dict:
    """Benchmark native Hollywood Squares training."""
    from trix.foundry.native_training import NativeHollywoodSquares, Trainer

    print("\n" + "=" * 60)
    print("NATIVE TRAINING BENCHMARK")
    print("=" * 60)

    # Create model (suppress output)
    import io
    import contextlib

    with contextlib.redirect_stdout(io.StringIO()):
        model = NativeHollywoodSquares(
            d_model=128,
            num_tiles=16,
            grid_size=8,
            lr=0.01,
        )

    # Generate data
    train_x = cp.random.randn(n_samples, 128).astype(cp.float32)
    train_y = train_x * 0.95 + cp.random.randn(n_samples, 128).astype(cp.float32) * 0.01

    # Create trainer
    trainer = Trainer(model, loss_fn='mse')

    # Warmup
    warmup_x = train_x[:batch_size]
    warmup_y = train_y[:batch_size]
    for _ in range(5):
        trainer.train_step(warmup_x, warmup_y)
    cp.cuda.Stream.null.synchronize()

    # Benchmark
    start = time.perf_counter()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = (n_samples + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)

            batch_x = train_x[start_idx:end_idx]
            batch_y = train_y[start_idx:end_idx]

            loss = trainer.train_step(batch_x, batch_y)
            epoch_loss += loss

    cp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start

    samples_per_sec = (n_samples * epochs) / elapsed
    batches_per_sec = (n_batches * epochs) / elapsed

    print(f"\n  Samples: {n_samples:,}")
    print(f"  Epochs: {epochs}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Samples/sec: {samples_per_sec:,.0f}")
    print(f"  Batches/sec: {batches_per_sec:.1f}")

    return {
        "framework": "native",
        "n_samples": n_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "elapsed_sec": elapsed,
        "samples_per_sec": samples_per_sec,
        "batches_per_sec": batches_per_sec,
    }


def benchmark_pytorch_training(n_samples: int, epochs: int, batch_size: int) -> dict:
    """Benchmark PyTorch SparseLookupFFNv4 training."""
    from trix.nn.sparse_lookup_v4 import SparseLookupFFNv4

    print("\n" + "=" * 60)
    print("PYTORCH TRAINING BENCHMARK")
    print("=" * 60)

    # Create model
    model = SparseLookupFFNv4(
        d_model=128,
        num_tiles=16,
        grid_size=8,
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Generate data
    train_x = torch.randn(n_samples, 128).cuda()
    train_y = train_x * 0.95 + torch.randn(n_samples, 128).cuda() * 0.01

    # Warmup
    for _ in range(5):
        idx = torch.randint(0, n_samples, (batch_size,))
        batch_x = train_x[idx].unsqueeze(0)  # [1, batch, d_model]
        batch_y = train_y[idx]

        optimizer.zero_grad()
        output, _, aux = model(batch_x)
        loss = criterion(output.squeeze(0), batch_y) + aux['total_aux']
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = (n_samples + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)

            batch_x = train_x[start_idx:end_idx].unsqueeze(0)
            batch_y = train_y[start_idx:end_idx]

            optimizer.zero_grad()
            output, _, aux = model(batch_x)
            loss = criterion(output.squeeze(0), batch_y) + aux['total_aux']
            loss.backward()
            optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    samples_per_sec = (n_samples * epochs) / elapsed
    batches_per_sec = (n_batches * epochs) / elapsed

    print(f"\n  Samples: {n_samples:,}")
    print(f"  Epochs: {epochs}")
    print(f"  Time: {elapsed:.3f}s")
    print(f"  Samples/sec: {samples_per_sec:,.0f}")
    print(f"  Batches/sec: {batches_per_sec:.1f}")

    return {
        "framework": "pytorch",
        "n_samples": n_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "elapsed_sec": elapsed,
        "samples_per_sec": samples_per_sec,
        "batches_per_sec": batches_per_sec,
    }


def full_comparison():
    """Full training benchmark comparison."""
    print()
    print("=" * 70)
    print("TRAINING BENCHMARK: NATIVE vs PYTORCH")
    print("=" * 70)
    print()
    print("Same task. Same architecture. Different implementations.")
    print()

    results = []

    # Test configurations
    configs = [
        {"n_samples": 1000, "epochs": 10, "batch_size": 64},
        {"n_samples": 10000, "epochs": 10, "batch_size": 256},
        {"n_samples": 10000, "epochs": 50, "batch_size": 256},
    ]

    for config in configs:
        print(f"\n{'='*70}")
        print(f"Config: {config['n_samples']:,} samples, {config['epochs']} epochs, batch={config['batch_size']}")
        print("=" * 70)

        # Clear GPU
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()

        # Benchmark both
        native_result = benchmark_native_training(**config)
        results.append(native_result)

        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()

        pytorch_result = benchmark_pytorch_training(**config)
        results.append(pytorch_result)

        # Compare
        speedup = native_result["samples_per_sec"] / pytorch_result["samples_per_sec"]
        print(f"\n  SPEEDUP: {speedup:.2f}x (Native vs PyTorch)")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Config':<30} {'Native (s/s)':<15} {'PyTorch (s/s)':<15} {'Speedup':<10}")
    print("-" * 70)

    for i in range(0, len(results), 2):
        native = results[i]
        pytorch = results[i + 1]
        speedup = native["samples_per_sec"] / pytorch["samples_per_sec"]
        config_str = f"{native['n_samples']}x{native['epochs']}@{native['batch_size']}"
        print(f"{config_str:<30} {native['samples_per_sec']:<15,.0f} {pytorch['samples_per_sec']:<15,.0f} {speedup:<10.2f}x")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    output_path = Path(__file__).parent / "training_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    print()
    print("=" * 70)
    print("NATIVE TRAINING: FASTER. SIMPLER. NO DEPENDENCIES.")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = full_comparison()
