#!/usr/bin/env python3
"""
Level 4 Benchmarks: Production-Scale Validation

This benchmark suite validates TriX architectures at production scale:

1. Language Modeling - Perplexity on synthetic and real text
2. Training Speed - Tokens per second, convergence rate
3. Inference Speed - Samples per second, latency
4. Memory Usage - Peak memory, scaling behavior
5. Comparison - vs standard transformer baseline

Usage:
    python benchmarks/level4_production.py [--quick] [--full] [--gpu]

"The proof is in the pudding at scale."
"""

import sys
import time
import gc
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from trix.nn.anchored import AnchoredDualModeFFN, AnchoredDualModeBlock, get_temperature_schedule
from trix.nn.octave import TrueOctaveFFN, TrueOctaveBlock


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""
    # Model sizes
    small: Dict = field(default_factory=lambda: {
        'd_model': 128,
        'num_layers': 4,
        'num_heads': 4,
        'vocab_size': 1000,
    })
    medium: Dict = field(default_factory=lambda: {
        'd_model': 256,
        'num_layers': 6,
        'num_heads': 8,
        'vocab_size': 5000,
    })
    large: Dict = field(default_factory=lambda: {
        'd_model': 512,
        'num_layers': 8,
        'num_heads': 8,
        'vocab_size': 10000,
    })

    # Training
    batch_size: int = 32
    seq_len: int = 128
    train_steps: int = 500
    warmup_steps: int = 50
    lr: float = 1e-3

    # Evaluation
    eval_samples: int = 1000
    inference_iterations: int = 100

    # Device
    device: str = 'cpu'


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class AnchoredLM(nn.Module):
    """Language model using AnchoredDualModeBlocks."""

    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)  # Max 512 positions

        self.blocks = nn.ModuleList([
            AnchoredDualModeBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_anchors=max(4, d_model // 32),
                num_tiles=max(8, d_model // 16),
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        h = self.embed(x) + self.pos_embed(positions)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        for block in self.blocks:
            h, _ = block(h, attn_mask=mask)

        h = self.norm(h)
        return self.head(h)

    def set_temperature(self, temp: float):
        for block in self.blocks:
            block.set_temperature(temp)


class OctaveLM(nn.Module):
    """Language model using TrueOctaveBlocks."""

    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)

        self.blocks = nn.ModuleList([
            TrueOctaveBlock(
                d_model=d_model,
                n_heads=num_heads,
                num_fine_tiles=max(16, d_model // 8),
                pool_factor=4,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        h = self.embed(x) + self.pos_embed(positions)

        for block in self.blocks:
            h, _ = block(h, is_causal=True)

        h = self.norm(h)
        return self.head(h)

    def set_mode(self, mode: str):
        for block in self.blocks:
            block.set_mode(mode)


class TransformerLM(nn.Module):
    """Standard transformer baseline."""

    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        h = self.embed(x) + self.pos_embed(positions)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.transformer(h, mask=mask, is_causal=True)

        h = self.norm(h)
        return self.head(h)


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_lm_data(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
    pattern_strength: float = 0.3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate language modeling data with learnable patterns.

    Patterns:
    - Bigram transitions (token i followed by i+1)
    - Repetition patterns
    - Rare token suppression
    """
    # Transition matrix with patterns
    transition = torch.ones(vocab_size, vocab_size) / vocab_size

    for i in range(vocab_size):
        next_tok = (i + 1) % vocab_size
        transition[i, next_tok] += pattern_strength
        transition[i] /= transition[i].sum()

    # Generate sequences
    sequences = []
    for _ in range(n_samples):
        seq = [torch.randint(1, vocab_size, (1,)).item()]
        for _ in range(seq_len - 1):
            probs = transition[seq[-1]].numpy()
            next_tok = np.random.choice(vocab_size, p=probs)
            seq.append(next_tok)
        sequences.append(seq)

    x = torch.tensor(sequences, dtype=torch.long)
    y = torch.cat([x[:, 1:], torch.zeros(n_samples, 1, dtype=torch.long)], dim=1)

    return x, y


# =============================================================================
# BENCHMARK RESULTS
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model_name: str
    model_size: str
    task: str

    # Quality metrics
    final_loss: float
    final_perplexity: float

    # Speed metrics
    train_tokens_per_sec: float
    inference_samples_per_sec: float
    inference_latency_ms: float

    # Resource metrics
    param_count: int
    peak_memory_mb: float

    # Convergence
    convergence_step: int  # Step where loss < threshold

    # Extra info
    extra: Dict = field(default_factory=dict)


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_peak_memory(model: nn.Module, x: torch.Tensor) -> float:
    """Measure peak memory during forward + backward."""
    gc.collect()

    if x.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Forward
    output = model(x)
    loss = F.cross_entropy(output.view(-1, output.size(-1)), x.view(-1))

    # Backward
    loss.backward()
    model.zero_grad()

    if x.device.type == 'cuda':
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        # Rough estimate for CPU
        peak_mb = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024 * 3

    return peak_mb


def train_lm(
    model: nn.Module,
    config: BenchmarkConfig,
    model_config: Dict,
    device: torch.device,
) -> Tuple[List[float], float, int]:
    """
    Train language model and return losses, tokens/sec, convergence step.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    vocab_size = model_config['vocab_size']
    losses = []
    tokens_processed = 0
    convergence_step = config.train_steps
    threshold = 0.8 * np.log(vocab_size)  # 80% of random perplexity

    start_time = time.time()

    for step in range(config.train_steps):
        # Generate batch
        x, y = generate_lm_data(
            config.batch_size,
            config.seq_len,
            vocab_size,
        )
        x, y = x.to(device), y.to(device)

        # Learning rate warmup
        if step < config.warmup_steps:
            lr = config.lr * (step + 1) / config.warmup_steps
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        # Temperature annealing for Anchored
        if hasattr(model, 'set_temperature'):
            temp = get_temperature_schedule(step, config.train_steps, 2.0, 0.1)
            model.set_temperature(temp)

        # Forward
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size),
            y[:, :-1].reshape(-1)
        )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        tokens_processed += config.batch_size * config.seq_len

        # Check convergence
        if loss.item() < threshold and convergence_step == config.train_steps:
            convergence_step = step

    elapsed = time.time() - start_time
    tokens_per_sec = tokens_processed / elapsed

    return losses, tokens_per_sec, convergence_step


def measure_inference_speed(
    model: nn.Module,
    config: BenchmarkConfig,
    model_config: Dict,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Measure inference speed.

    Returns: (samples_per_sec, latency_ms)
    """
    model.eval()

    # Set deterministic mode if available
    if hasattr(model, 'set_mode'):
        model.set_mode('deterministic')

    vocab_size = model_config['vocab_size']
    x = torch.randint(0, vocab_size, (config.batch_size, config.seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(config.inference_iterations):
            _ = model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.time() - start

    samples_per_sec = (config.batch_size * config.inference_iterations) / elapsed
    latency_ms = elapsed / config.inference_iterations * 1000

    return samples_per_sec, latency_ms


def evaluate_perplexity(
    model: nn.Module,
    config: BenchmarkConfig,
    model_config: Dict,
    device: torch.device,
) -> float:
    """Evaluate perplexity on held-out data."""
    model.eval()

    vocab_size = model_config['vocab_size']
    x, y = generate_lm_data(config.eval_samples, config.seq_len, vocab_size)
    x, y = x.to(device), y.to(device)

    total_loss = 0.0
    n_batches = config.eval_samples // config.batch_size

    with torch.no_grad():
        for i in range(n_batches):
            start = i * config.batch_size
            end = start + config.batch_size

            logits = model(x[start:end])
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                y[start:end, :-1].reshape(-1)
            )
            total_loss += loss.item()

    avg_loss = total_loss / n_batches
    perplexity = np.exp(avg_loss)

    return perplexity


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================

def run_benchmark(
    model_class,
    model_name: str,
    model_size: str,
    config: BenchmarkConfig,
    model_config: Dict,
) -> BenchmarkResult:
    """Run complete benchmark for a single model configuration."""
    device = torch.device(config.device)

    print(f"\n  Creating {model_name} ({model_size})...")
    model = model_class(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
    ).to(device)

    param_count = count_parameters(model)
    print(f"    Parameters: {param_count:,}")

    # Measure memory
    x = torch.randint(0, model_config['vocab_size'],
                      (config.batch_size, config.seq_len), device=device)
    peak_memory = measure_peak_memory(model, x)
    print(f"    Peak memory: {peak_memory:.1f} MB")

    # Train
    print(f"    Training for {config.train_steps} steps...")
    losses, tokens_per_sec, convergence_step = train_lm(
        model, config, model_config, device
    )
    print(f"    Tokens/sec: {tokens_per_sec:,.0f}")
    print(f"    Convergence: step {convergence_step}")

    # Evaluate
    print(f"    Evaluating perplexity...")
    perplexity = evaluate_perplexity(model, config, model_config, device)
    print(f"    Perplexity: {perplexity:.2f}")

    # Inference speed
    print(f"    Measuring inference speed...")
    samples_per_sec, latency_ms = measure_inference_speed(
        model, config, model_config, device
    )
    print(f"    Samples/sec: {samples_per_sec:,.0f}")
    print(f"    Latency: {latency_ms:.2f} ms")

    return BenchmarkResult(
        model_name=model_name,
        model_size=model_size,
        task="Language Modeling",
        final_loss=losses[-1],
        final_perplexity=perplexity,
        train_tokens_per_sec=tokens_per_sec,
        inference_samples_per_sec=samples_per_sec,
        inference_latency_ms=latency_ms,
        param_count=param_count,
        peak_memory_mb=peak_memory,
        convergence_step=convergence_step,
    )


def run_all_benchmarks(config: BenchmarkConfig, sizes: List[str] = None) -> List[BenchmarkResult]:
    """Run benchmarks for all model types and sizes."""
    sizes = sizes or ['small']

    models = [
        (AnchoredLM, "Anchored"),
        (OctaveLM, "Octave"),
        (TransformerLM, "Transformer"),
    ]

    results = []

    for size in sizes:
        print(f"\n{'='*70}")
        print(f"BENCHMARKING SIZE: {size.upper()}")
        print(f"{'='*70}")

        model_config = getattr(config, size)

        for model_class, model_name in models:
            try:
                result = run_benchmark(
                    model_class, model_name, size, config, model_config
                )
                results.append(result)
            except Exception as e:
                print(f"  ERROR: {model_name} failed: {e}")

    return results


def print_results_table(results: List[BenchmarkResult]):
    """Print formatted results table."""
    print(f"\n{'='*90}")
    print("BENCHMARK RESULTS")
    print(f"{'='*90}")

    # Group by size
    sizes = sorted(set(r.model_size for r in results))

    for size in sizes:
        size_results = [r for r in results if r.model_size == size]

        print(f"\n--- {size.upper()} ---")
        print(f"{'Model':<15} {'Params':>12} {'PPL':>8} {'Train T/s':>12} {'Inf S/s':>10} {'Mem MB':>8}")
        print("-" * 70)

        for r in size_results:
            print(f"{r.model_name:<15} {r.param_count:>12,} {r.final_perplexity:>8.2f} "
                  f"{r.train_tokens_per_sec:>12,.0f} {r.inference_samples_per_sec:>10,.0f} "
                  f"{r.peak_memory_mb:>8.1f}")

    # Comparison summary
    print(f"\n{'='*90}")
    print("COMPARISON SUMMARY")
    print(f"{'='*90}")

    for size in sizes:
        size_results = [r for r in results if r.model_size == size]
        transformer = next((r for r in size_results if r.model_name == "Transformer"), None)

        if transformer:
            print(f"\n{size.upper()} vs Transformer baseline:")
            for r in size_results:
                if r.model_name != "Transformer":
                    ppl_ratio = r.final_perplexity / transformer.final_perplexity
                    speed_ratio = r.inference_samples_per_sec / transformer.inference_samples_per_sec
                    param_ratio = r.param_count / transformer.param_count

                    print(f"  {r.model_name}:")
                    print(f"    Perplexity: {ppl_ratio:.2f}x transformer ({r.final_perplexity:.2f} vs {transformer.final_perplexity:.2f})")
                    print(f"    Inference:  {speed_ratio:.2f}x speed")
                    print(f"    Parameters: {param_ratio:.2f}x count")


def save_results(results: List[BenchmarkResult], path: str):
    """Save results to JSON."""
    data = []
    for r in results:
        d = {
            'model_name': r.model_name,
            'model_size': r.model_size,
            'task': r.task,
            'final_loss': r.final_loss,
            'final_perplexity': r.final_perplexity,
            'train_tokens_per_sec': r.train_tokens_per_sec,
            'inference_samples_per_sec': r.inference_samples_per_sec,
            'inference_latency_ms': r.inference_latency_ms,
            'param_count': r.param_count,
            'peak_memory_mb': r.peak_memory_mb,
            'convergence_step': r.convergence_step,
        }
        data.append(d)

    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Level 4 Production Benchmarks")
    parser.add_argument('--quick', action='store_true', help="Quick benchmark (small only, fewer steps)")
    parser.add_argument('--full', action='store_true', help="Full benchmark (all sizes)")
    parser.add_argument('--gpu', action='store_true', help="Use GPU if available")
    parser.add_argument('--output', type=str, default='benchmark_results.json', help="Output file")
    args = parser.parse_args()

    print("="*70)
    print("LEVEL 4 PRODUCTION BENCHMARKS")
    print("="*70)

    config = BenchmarkConfig()

    # Device
    if args.gpu and torch.cuda.is_available():
        config.device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        config.device = 'cpu'
        print("Using CPU")

    # Size selection
    if args.quick:
        sizes = ['small']
        config.train_steps = 100
        config.eval_samples = 200
        config.inference_iterations = 20
        print("Quick mode: small model, 100 steps")
    elif args.full:
        sizes = ['small', 'medium', 'large']
        print("Full mode: all sizes")
    else:
        sizes = ['small']
        config.train_steps = 300
        config.eval_samples = 500
        config.inference_iterations = 50
        print("Default mode: small, 300 steps")

    print(f"Batch size: {config.batch_size}")
    print(f"Sequence length: {config.seq_len}")
    print(f"Training steps: {config.train_steps}")

    # Run benchmarks
    results = run_all_benchmarks(config, sizes)

    # Print results
    print_results_table(results)

    # Save results
    save_results(results, args.output)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
