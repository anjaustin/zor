#!/usr/bin/env python3
"""
Benchmarks for Anchored Dual-Mode FFN.

Compares:
1. AnchoredDualModeFFN vs SparseLookupFFN vs Dense FFN
2. Training convergence
3. Inference speed
4. Memory usage
5. Parameter efficiency

Usage:
    python benchmarks/anchored_benchmark.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# TriX imports
from trix.nn.anchored import AnchoredDualModeFFN, get_temperature_schedule
from trix.nn.sparse_lookup import SparseLookupFFN


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    d_model: int = 256
    num_anchors: int = 16
    num_tiles: int = 64
    batch_size: int = 32
    seq_len: int = 64
    num_warmup: int = 10
    num_iterations: int = 100
    train_steps: int = 200
    learning_rate: float = 1e-3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42


# =============================================================================
# BASELINE MODELS
# =============================================================================

class DenseFFN(nn.Module):
    """Standard dense FFN for baseline comparison."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.net(x)), {}


# =============================================================================
# BENCHMARK UTILITIES
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_buffers(model: nn.Module) -> int:
    """Count buffer elements (frozen components)."""
    return sum(b.numel() for b in model.buffers())


def measure_memory(model: nn.Module, x: torch.Tensor) -> Dict[str, float]:
    """Measure memory usage during forward pass."""
    if x.device.type != 'cuda':
        return {'peak_mb': 0, 'allocated_mb': 0}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    model(x)

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024
    allocated = torch.cuda.memory_allocated() / 1024 / 1024

    return {'peak_mb': peak, 'allocated_mb': allocated}


def measure_latency(
    model: nn.Module,
    x: torch.Tensor,
    num_warmup: int = 10,
    num_iterations: int = 100,
    train_mode: bool = False,
) -> Dict[str, float]:
    """Measure forward pass latency."""
    if train_mode:
        model.train()
    else:
        model.eval()

    # Warmup
    for _ in range(num_warmup):
        if train_mode:
            model(x)
        else:
            with torch.no_grad():
                model(x)

    if x.device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        if train_mode:
            result = model(x)
            # Handle different return signatures
            out = result[0] if isinstance(result, tuple) else result
            # Simulate backward
            out.sum().backward()
        else:
            with torch.no_grad():
                model(x)
        if x.device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        'mean_ms': sum(times) / len(times) * 1000,
        'min_ms': min(times) * 1000,
        'max_ms': max(times) * 1000,
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000,
    }


def measure_throughput(
    model: nn.Module,
    x: torch.Tensor,
    num_iterations: int = 100,
) -> float:
    """Measure throughput in samples/second."""
    model.eval()

    if x.device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            model(x)
    if x.device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_samples = x.shape[0] * x.shape[1] * num_iterations
    return total_samples / elapsed


def train_and_measure(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    num_steps: int,
    lr: float,
    temperature_anneal: bool = False,
) -> Dict[str, List[float]]:
    """Train model and measure convergence."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for step in range(num_steps):
        # Temperature annealing for anchored model
        if temperature_anneal and hasattr(model, 'set_temperature'):
            temp = get_temperature_schedule(step, num_steps, 2.0, 0.1, 'cosine')
            model.set_temperature(temp)

        optimizer.zero_grad()
        result = model(x)
        # Handle different return signatures
        output = result[0] if isinstance(result, tuple) else result
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return {
        'losses': losses,
        'initial_loss': losses[0],
        'final_loss': losses[-1],
        'min_loss': min(losses),
        'convergence_rate': (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0,
    }


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_benchmark(config: BenchmarkConfig) -> Dict:
    """Run full benchmark suite."""
    print(f"\n{'='*60}")
    print(f"ANCHORED DUAL-MODE FFN BENCHMARK")
    print(f"{'='*60}")
    print(f"Device: {config.device}")
    print(f"d_model: {config.d_model}")
    print(f"num_anchors: {config.num_anchors}")
    print(f"num_tiles: {config.num_tiles}")
    print(f"batch_size: {config.batch_size}")
    print(f"seq_len: {config.seq_len}")
    print(f"{'='*60}\n")

    torch.manual_seed(config.seed)
    device = torch.device(config.device)

    # Create models
    models = {
        'AnchoredDualMode': AnchoredDualModeFFN(
            d_model=config.d_model,
            num_anchors=config.num_anchors,
            num_tiles=config.num_tiles,
        ).to(device),
        'SparseLookup': SparseLookupFFN(
            d_model=config.d_model,
            num_tiles=config.num_tiles,
        ).to(device),
        'DenseFFN': DenseFFN(
            d_model=config.d_model,
            expansion=4,
        ).to(device),
    }

    # Create test data
    x = torch.randn(config.batch_size, config.seq_len, config.d_model, device=device)
    target = torch.randn(config.batch_size, config.seq_len, config.d_model, device=device)

    results = {
        'config': {
            'd_model': config.d_model,
            'num_anchors': config.num_anchors,
            'num_tiles': config.num_tiles,
            'batch_size': config.batch_size,
            'seq_len': config.seq_len,
            'device': config.device,
        },
        'models': {},
    }

    # =================================================================
    # 1. PARAMETER COUNT
    # =================================================================
    print("1. PARAMETER COUNT")
    print("-" * 40)

    for name, model in models.items():
        params = count_parameters(model)
        buffers = count_buffers(model)
        results['models'][name] = {'parameters': params, 'buffers': buffers}
        print(f"  {name:20s}: {params:,} params, {buffers:,} buffer elements")

    print()

    # =================================================================
    # 2. INFERENCE LATENCY
    # =================================================================
    print("2. INFERENCE LATENCY")
    print("-" * 40)

    for name, model in models.items():
        latency = measure_latency(
            model, x,
            num_warmup=config.num_warmup,
            num_iterations=config.num_iterations,
            train_mode=False,
        )
        results['models'][name]['inference_latency'] = latency
        print(f"  {name:20s}: {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms")

    print()

    # =================================================================
    # 3. TRAINING LATENCY
    # =================================================================
    print("3. TRAINING LATENCY (forward + backward)")
    print("-" * 40)

    for name, model in models.items():
        # Re-create model to avoid state issues
        if name == 'AnchoredDualMode':
            model = AnchoredDualModeFFN(
                d_model=config.d_model,
                num_anchors=config.num_anchors,
                num_tiles=config.num_tiles,
            ).to(device)
        elif name == 'SparseLookup':
            model = SparseLookupFFN(
                d_model=config.d_model,
                num_tiles=config.num_tiles,
            ).to(device)
        else:
            model = DenseFFN(d_model=config.d_model).to(device)

        latency = measure_latency(
            model, x,
            num_warmup=config.num_warmup,
            num_iterations=min(config.num_iterations, 50),  # Fewer for training
            train_mode=True,
        )
        results['models'][name]['training_latency'] = latency
        print(f"  {name:20s}: {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms")

    print()

    # =================================================================
    # 4. THROUGHPUT
    # =================================================================
    print("4. THROUGHPUT (samples/sec)")
    print("-" * 40)

    for name, model in models.items():
        if name == 'AnchoredDualMode':
            model = AnchoredDualModeFFN(
                d_model=config.d_model,
                num_anchors=config.num_anchors,
                num_tiles=config.num_tiles,
            ).to(device)
        elif name == 'SparseLookup':
            model = SparseLookupFFN(
                d_model=config.d_model,
                num_tiles=config.num_tiles,
            ).to(device)
        else:
            model = DenseFFN(d_model=config.d_model).to(device)

        throughput = measure_throughput(model, x, config.num_iterations)
        results['models'][name]['throughput'] = throughput
        print(f"  {name:20s}: {throughput:,.0f} samples/sec")

    print()

    # =================================================================
    # 5. MEMORY USAGE
    # =================================================================
    if config.device == 'cuda':
        print("5. MEMORY USAGE")
        print("-" * 40)

        for name, model in models.items():
            if name == 'AnchoredDualMode':
                model = AnchoredDualModeFFN(
                    d_model=config.d_model,
                    num_anchors=config.num_anchors,
                    num_tiles=config.num_tiles,
                ).to(device)
            elif name == 'SparseLookup':
                model = SparseLookupFFN(
                    d_model=config.d_model,
                    num_tiles=config.num_tiles,
                ).to(device)
            else:
                model = DenseFFN(d_model=config.d_model).to(device)

            memory = measure_memory(model, x)
            results['models'][name]['memory'] = memory
            print(f"  {name:20s}: {memory['peak_mb']:.1f} MB peak")

        print()

    # =================================================================
    # 6. TRAINING CONVERGENCE
    # =================================================================
    print("6. TRAINING CONVERGENCE")
    print("-" * 40)

    for name, _ in models.items():
        # Fresh model for training
        if name == 'AnchoredDualMode':
            model = AnchoredDualModeFFN(
                d_model=config.d_model,
                num_anchors=config.num_anchors,
                num_tiles=config.num_tiles,
            ).to(device)
            use_anneal = True
        elif name == 'SparseLookup':
            model = SparseLookupFFN(
                d_model=config.d_model,
                num_tiles=config.num_tiles,
            ).to(device)
            use_anneal = False
        else:
            model = DenseFFN(d_model=config.d_model).to(device)
            use_anneal = False

        torch.manual_seed(config.seed)  # Same init for fair comparison
        x_train = torch.randn(config.batch_size, config.seq_len, config.d_model, device=device)
        target_train = torch.randn(config.batch_size, config.seq_len, config.d_model, device=device)

        convergence = train_and_measure(
            model, x_train, target_train,
            num_steps=config.train_steps,
            lr=config.learning_rate,
            temperature_anneal=use_anneal,
        )

        results['models'][name]['convergence'] = {
            'initial_loss': convergence['initial_loss'],
            'final_loss': convergence['final_loss'],
            'min_loss': convergence['min_loss'],
            'convergence_rate': convergence['convergence_rate'],
        }

        print(f"  {name:20s}: {convergence['initial_loss']:.4f} -> {convergence['final_loss']:.4f} "
              f"(rate: {convergence['convergence_rate']:.1%})")

    print()

    # =================================================================
    # 7. SUMMARY
    # =================================================================
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    anchored = results['models']['AnchoredDualMode']
    sparse = results['models']['SparseLookup']
    dense = results['models']['DenseFFN']

    print(f"\nParameter Efficiency:")
    print(f"  AnchoredDualMode: {anchored['parameters']:,} params")
    print(f"  SparseLookup:     {sparse['parameters']:,} params")
    print(f"  DenseFFN:         {dense['parameters']:,} params")
    print(f"  Anchored vs Dense: {anchored['parameters'] / dense['parameters']:.1%} of dense params")

    print(f"\nInference Speed (vs Dense):")
    anchored_speedup = dense['inference_latency']['mean_ms'] / anchored['inference_latency']['mean_ms']
    sparse_speedup = dense['inference_latency']['mean_ms'] / sparse['inference_latency']['mean_ms']
    print(f"  AnchoredDualMode: {anchored_speedup:.2f}x")
    print(f"  SparseLookup:     {sparse_speedup:.2f}x")

    print(f"\nConvergence (loss reduction):")
    print(f"  AnchoredDualMode: {anchored['convergence']['convergence_rate']:.1%}")
    print(f"  SparseLookup:     {sparse['convergence']['convergence_rate']:.1%}")
    print(f"  DenseFFN:         {dense['convergence']['convergence_rate']:.1%}")

    return results


def run_scale_benchmark(base_config: BenchmarkConfig) -> Dict:
    """Run benchmark at different scales."""
    print(f"\n{'='*60}")
    print("SCALE BENCHMARK")
    print(f"{'='*60}\n")

    scales = [
        {'d_model': 128, 'num_anchors': 8, 'num_tiles': 32},
        {'d_model': 256, 'num_anchors': 16, 'num_tiles': 64},
        {'d_model': 512, 'num_anchors': 32, 'num_tiles': 128},
    ]

    results = {'scales': []}

    for scale in scales:
        print(f"\n--- Scale: d_model={scale['d_model']}, anchors={scale['num_anchors']}, tiles={scale['num_tiles']} ---")

        config = BenchmarkConfig(
            d_model=scale['d_model'],
            num_anchors=scale['num_anchors'],
            num_tiles=scale['num_tiles'],
            batch_size=base_config.batch_size,
            seq_len=base_config.seq_len,
            num_warmup=5,
            num_iterations=50,
            train_steps=100,
            device=base_config.device,
        )

        torch.manual_seed(config.seed)
        device = torch.device(config.device)

        # Create models
        anchored = AnchoredDualModeFFN(
            d_model=config.d_model,
            num_anchors=config.num_anchors,
            num_tiles=config.num_tiles,
        ).to(device)

        x = torch.randn(config.batch_size, config.seq_len, config.d_model, device=device)

        # Measure
        params = count_parameters(anchored)
        latency = measure_latency(anchored, x, config.num_warmup, config.num_iterations)
        throughput = measure_throughput(anchored, x, config.num_iterations)

        scale_result = {
            **scale,
            'parameters': params,
            'latency_ms': latency['mean_ms'],
            'throughput': throughput,
        }
        results['scales'].append(scale_result)

        print(f"  Parameters: {params:,}")
        print(f"  Latency: {latency['mean_ms']:.2f} ms")
        print(f"  Throughput: {throughput:,.0f} samples/sec")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Anchored Dual-Mode FFN Benchmark')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--num_anchors', type=int, default=16)
    parser.add_argument('--num_tiles', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--train_steps', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    parser.add_argument('--scale', action='store_true', help='Run scale benchmark')

    args = parser.parse_args()

    config = BenchmarkConfig(
        d_model=args.d_model,
        num_anchors=args.num_anchors,
        num_tiles=args.num_tiles,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        train_steps=args.train_steps,
        device=args.device,
    )

    # Run main benchmark
    results = run_benchmark(config)

    # Run scale benchmark if requested
    if args.scale:
        scale_results = run_scale_benchmark(config)
        results['scale_benchmark'] = scale_results

    # Save results
    if args.save:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'benchmarks/results/anchored_benchmark_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {filename}")

    return results


if __name__ == '__main__':
    main()
