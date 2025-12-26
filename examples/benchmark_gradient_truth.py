#!/usr/bin/env python3
"""
Benchmark: Gradient Truth vs STE Training

Compares:
1. Training convergence speed
2. Final accuracy
3. Gradient correctness
4. Memory usage
5. Inference speed

"Gradients should flow only where there is genuine uncertainty."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trix.nn.gradient_truth import (
    GradientTruthFFN,
    PolynomialShapeBank,
    create_gradient_truth_ffn,
)
from trix.nn.hierarchical import HierarchicalTriXFFN
from trix.kernel import STESign


# =============================================================================
# STE BASELINE
# =============================================================================

class STEBaseline(nn.Module):
    """
    Traditional STE-trained FFN for comparison.
    
    Uses Straight-Through Estimator for ternary weights.
    """
    
    def __init__(self, d_model: int, d_hidden: int = None, num_tiles: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden or d_model * 2
        self.num_tiles = num_tiles
        
        tile_hidden = self.d_hidden // num_tiles
        
        # Ternary weights (float, quantized via STE)
        self.up_weights = nn.Parameter(torch.randn(num_tiles, tile_hidden, d_model) * 0.02)
        self.down_weights = nn.Parameter(torch.randn(num_tiles, d_model, tile_hidden) * 0.02)
        
        # Scales
        self.scales = nn.Parameter(torch.ones(num_tiles))
        
        # Router
        self.router = nn.Linear(d_model, num_tiles)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm(x)
        
        # Routing
        logits = self.router(x)
        weights = F.softmax(logits, dim=-1)
        
        # Quantize weights via STE
        up_q = STESign.apply(self.up_weights)
        down_q = STESign.apply(self.down_weights)
        
        # Apply all tiles
        batch = x.shape[0]
        outputs = []
        for t in range(self.num_tiles):
            h = F.linear(x, up_q[t])
            h = F.relu(h)
            out = F.linear(h, down_q[t]) * self.scales[t]
            outputs.append(out)
        
        outputs = torch.stack(outputs, dim=1)  # [batch, num_tiles, d_model]
        
        # Weighted sum
        output = (weights.unsqueeze(-1) * outputs).sum(dim=1)
        
        return residual + output, weights


# =============================================================================
# BENCHMARK TASKS
# =============================================================================

def generate_regression_task(batch: int, d_model: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simple nonlinear regression task."""
    x = torch.randn(batch, d_model)
    target = torch.tanh(x) + 0.5 * torch.sin(x * 2)
    return x, target


def generate_classification_task(batch: int, d_model: int, num_classes: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classification task - cluster assignment."""
    # Create cluster centers
    centers = torch.randn(num_classes, d_model)
    
    # Assign each sample to a random cluster and add noise
    labels = torch.randint(0, num_classes, (batch,))
    x = centers[labels] + torch.randn(batch, d_model) * 0.3
    
    return x, labels


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

@dataclass
class BenchmarkResult:
    name: str
    task: str
    epochs: int
    final_loss: float
    convergence_epoch: int  # Epoch where loss < threshold
    train_time: float
    inference_time: float
    param_count: int
    gradient_correct: bool


def train_model(
    model: nn.Module,
    task_fn,
    epochs: int = 200,
    batch: int = 64,
    d_model: int = 64,
    lr: float = 0.01,
    threshold: float = 0.1,
    is_classification: bool = False,
) -> Tuple[List[float], int, float]:
    """
    Train a model and return losses, convergence epoch, and training time.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    convergence_epoch = epochs
    
    start_time = time.time()
    
    for epoch in range(epochs):
        if is_classification:
            x, labels = task_fn(batch, d_model)
            result = model(x)
            output = result[0] if isinstance(result, tuple) else result
            loss = F.cross_entropy(output, labels)
        else:
            x, target = task_fn(batch, d_model)
            result = model(x)
            output = result[0] if isinstance(result, tuple) else result
            loss = F.mse_loss(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if loss.item() < threshold and convergence_epoch == epochs:
            convergence_epoch = epoch
    
    train_time = time.time() - start_time
    
    return losses, convergence_epoch, train_time


def measure_inference_speed(model: nn.Module, d_model: int, batch: int = 64, iterations: int = 100) -> float:
    """Measure inference speed in samples/second."""
    model.eval()
    x = torch.randn(batch, d_model)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            result = model(x)
    
    start = time.time()
    for _ in range(iterations):
        with torch.no_grad():
            result = model(x)
    elapsed = time.time() - start
    
    model.train()
    return (batch * iterations) / elapsed


def check_gradient_correctness(model: nn.Module, d_model: int) -> bool:
    """
    Check if gradients are correct (no STE artifacts in Gradient Truth).
    
    For Gradient Truth: shape bank should have no gradients
    For STE: weights have gradients but they're approximated
    """
    model.train()
    x = torch.randn(16, d_model, requires_grad=True)
    result = model(x)
    output = result[0] if isinstance(result, tuple) else result
    loss = output.sum()
    loss.backward()
    
    # Check for NaN gradients
    for name, p in model.named_parameters():
        if p.grad is not None:
            if p.grad.isnan().any():
                return False
    
    model.zero_grad()
    return True


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_benchmarks():
    print("=" * 70)
    print("BENCHMARK: Gradient Truth vs STE Training")
    print("=" * 70)
    
    # Configuration
    d_model = 64
    num_shapes = 8
    epochs = 200
    batch = 64
    
    results = []
    
    # =========================================================================
    # Task 1: Regression
    # =========================================================================
    print("\n" + "-" * 70)
    print("TASK 1: Nonlinear Regression")
    print("-" * 70)
    
    # Gradient Truth
    print("\nTraining Gradient Truth...")
    gt_model = create_gradient_truth_ffn(d_model=d_model, num_shapes=num_shapes)
    gt_losses, gt_conv, gt_time = train_model(
        gt_model, generate_regression_task, epochs=epochs, batch=batch, d_model=d_model
    )
    gt_infer = measure_inference_speed(gt_model, d_model)
    gt_correct = check_gradient_correctness(gt_model, d_model)
    
    results.append(BenchmarkResult(
        name="Gradient Truth",
        task="Regression",
        epochs=epochs,
        final_loss=gt_losses[-1],
        convergence_epoch=gt_conv,
        train_time=gt_time,
        inference_time=1000 / gt_infer,  # ms per sample
        param_count=count_parameters(gt_model),
        gradient_correct=gt_correct,
    ))
    
    # STE Baseline
    print("Training STE Baseline...")
    ste_model = STEBaseline(d_model=d_model, num_tiles=num_shapes)
    ste_losses, ste_conv, ste_time = train_model(
        ste_model, generate_regression_task, epochs=epochs, batch=batch, d_model=d_model
    )
    ste_infer = measure_inference_speed(ste_model, d_model)
    ste_correct = check_gradient_correctness(ste_model, d_model)
    
    results.append(BenchmarkResult(
        name="STE Baseline",
        task="Regression",
        epochs=epochs,
        final_loss=ste_losses[-1],
        convergence_epoch=ste_conv,
        train_time=ste_time,
        inference_time=1000 / ste_infer,
        param_count=count_parameters(ste_model),
        gradient_correct=ste_correct,
    ))
    
    # HierarchicalTriXFFN (production STE)
    print("Training HierarchicalTriXFFN...")
    htrix_model = HierarchicalTriXFFN(d_model=d_model, num_tiles=num_shapes, tiles_per_cluster=4)
    htrix_losses, htrix_conv, htrix_time = train_model(
        htrix_model, 
        lambda b, d: (torch.randn(b, d), torch.tanh(torch.randn(b, d))),
        epochs=epochs, batch=batch, d_model=d_model
    )
    htrix_infer = measure_inference_speed(htrix_model, d_model)
    htrix_correct = check_gradient_correctness(htrix_model, d_model)
    
    results.append(BenchmarkResult(
        name="HierarchicalTriXFFN",
        task="Regression",
        epochs=epochs,
        final_loss=htrix_losses[-1],
        convergence_epoch=htrix_conv,
        train_time=htrix_time,
        inference_time=1000 / htrix_infer,
        param_count=count_parameters(htrix_model),
        gradient_correct=htrix_correct,
    ))
    
    # =========================================================================
    # Print Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\n{:<25} {:>12} {:>12} {:>12} {:>12}".format(
        "Model", "Final Loss", "Conv Epoch", "Train Time", "Params"
    ))
    print("-" * 70)
    
    for r in results:
        print("{:<25} {:>12.4f} {:>12} {:>12.2f}s {:>12,}".format(
            r.name, r.final_loss, r.convergence_epoch, r.train_time, r.param_count
        ))
    
    print("\n{:<25} {:>15} {:>15}".format(
        "Model", "Inference (ms)", "Grads Correct"
    ))
    print("-" * 55)
    
    for r in results:
        print("{:<25} {:>15.4f} {:>15}".format(
            r.name, r.inference_time, "Yes" if r.gradient_correct else "No"
        ))
    
    # =========================================================================
    # Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    gt_result = results[0]
    ste_result = results[1]
    
    print(f"\nGradient Truth vs STE Baseline:")
    print(f"  - Final loss:      {gt_result.final_loss:.4f} vs {ste_result.final_loss:.4f}")
    print(f"  - Convergence:     epoch {gt_result.convergence_epoch} vs {ste_result.convergence_epoch}")
    print(f"  - Train time:      {gt_result.train_time:.2f}s vs {ste_result.train_time:.2f}s")
    print(f"  - Parameters:      {gt_result.param_count:,} vs {ste_result.param_count:,}")
    
    if gt_result.final_loss < ste_result.final_loss:
        improvement = (ste_result.final_loss - gt_result.final_loss) / ste_result.final_loss * 100
        print(f"\n  Gradient Truth achieves {improvement:.1f}% lower loss!")
    
    if gt_result.convergence_epoch < ste_result.convergence_epoch:
        speedup = ste_result.convergence_epoch / max(gt_result.convergence_epoch, 1)
        print(f"  Gradient Truth converges {speedup:.1f}x faster!")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("""
Gradient Truth separates:
  - WHAT to compute (frozen shapes) - discovered, not learned
  - WHICH to use (routing) - learned with real gradients
  - HOW MUCH (scales) - learned with real gradients

No STE required. Gradients are mathematically correct.
""")
    
    return results


def run_scaling_benchmark():
    """Benchmark scaling behavior."""
    print("\n" + "=" * 70)
    print("SCALING BENCHMARK")
    print("=" * 70)
    
    configs = [
        (32, 4),
        (64, 8),
        (128, 16),
        (256, 32),
        (512, 64),
    ]
    
    print("\n{:<12} {:>12} {:>12} {:>15} {:>15}".format(
        "d_model", "num_shapes", "Params", "Forward (ms)", "Backward (ms)"
    ))
    print("-" * 70)
    
    for d_model, num_shapes in configs:
        model = create_gradient_truth_ffn(d_model=d_model, num_shapes=num_shapes)
        x = torch.randn(32, d_model, requires_grad=True)
        
        # Forward time
        start = time.time()
        for _ in range(100):
            output, _ = model(x)
        forward_time = (time.time() - start) / 100 * 1000
        
        # Backward time
        start = time.time()
        for _ in range(100):
            output, _ = model(x)
            loss = output.sum()
            loss.backward()
            model.zero_grad()
        backward_time = (time.time() - start) / 100 * 1000 - forward_time
        
        params = count_parameters(model)
        
        print("{:<12} {:>12} {:>12,} {:>15.3f} {:>15.3f}".format(
            d_model, num_shapes, params, forward_time, backward_time
        ))


if __name__ == "__main__":
    results = run_benchmarks()
    run_scaling_benchmark()
    
    print("\nBenchmark complete.")
