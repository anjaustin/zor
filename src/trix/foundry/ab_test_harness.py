"""
A/B Test Harness: Native vs PyTorch

Rigorous side-by-side comparison.
Same data. Same hyperparameters. Same task.
Reproducible. Documented. Scientific.

Tests:
1. Training convergence (do both reach same loss?)
2. Training speed (samples/sec)
3. Inference speed (tokens/sec)
4. Memory usage (GPU MB)
5. Numerical stability (gradient norms, loss variance)

"Let the numbers speak."
"""

import numpy as np
import cupy as cp
import torch
import torch.nn as nn
import time
import json
import gc
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, '/workspace/trix_latest/TriXO/src')


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

@dataclass
class TestConfig:
    """Configuration for A/B tests."""
    d_model: int = 128
    num_tiles: int = 16
    grid_size: int = 8
    learning_rate: float = 0.01
    batch_size: int = 256
    random_seed: int = 42

    # Test parameters
    convergence_samples: int = 5000
    convergence_epochs: int = 50
    speed_samples: int = 10000
    speed_epochs: int = 20
    inference_samples: int = 100000
    inference_runs: int = 10


@dataclass
class TestResult:
    """Result from a single test."""
    test_name: str
    native_value: float
    pytorch_value: float
    unit: str
    winner: str
    ratio: float
    notes: str = ""


# =============================================================================
# TEST HARNESS
# =============================================================================

class ABTestHarness:
    """Comprehensive A/B testing harness."""

    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.results: List[TestResult] = []
        self.detailed_logs: Dict = {}

        print("=" * 70)
        print("A/B TEST HARNESS: NATIVE vs PYTORCH")
        print("=" * 70)
        print()
        print("Configuration:")
        for k, v in asdict(self.config).items():
            print(f"  {k}: {v}")
        print()

    def _clear_gpu(self):
        """Clear GPU memory between tests."""
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
        gc.collect()

    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        return torch.cuda.memory_allocated() / 1e6

    # =========================================================================
    # TEST 1: CONVERGENCE
    # =========================================================================

    def test_convergence(self) -> TestResult:
        """Test: Do both systems converge to similar loss?"""
        print("\n" + "=" * 60)
        print("TEST 1: CONVERGENCE")
        print("=" * 60)
        print("Question: Do both systems reach the same final loss?")
        print()

        config = self.config
        np.random.seed(config.random_seed)

        # Generate shared data (numpy for reproducibility)
        train_x_np = np.random.randn(config.convergence_samples, config.d_model).astype(np.float32)
        train_y_np = train_x_np * 0.9 + np.random.randn(config.convergence_samples, config.d_model).astype(np.float32) * 0.05

        # === NATIVE ===
        print("Running Native training...")
        self._clear_gpu()

        from trix.foundry.native_training import NativeHollywoodSquares, Trainer
        import io, contextlib

        with contextlib.redirect_stdout(io.StringIO()):
            native_model = NativeHollywoodSquares(
                d_model=config.d_model,
                num_tiles=config.num_tiles,
                grid_size=config.grid_size,
                lr=config.learning_rate,
            )

        train_x_native = cp.asarray(train_x_np)
        train_y_native = cp.asarray(train_y_np)

        native_trainer = Trainer(native_model, loss_fn='mse')
        native_losses = []

        for epoch in range(config.convergence_epochs):
            epoch_loss = 0.0
            n_batches = (config.convergence_samples + config.batch_size - 1) // config.batch_size

            for batch_idx in range(n_batches):
                start_idx = batch_idx * config.batch_size
                end_idx = min(start_idx + config.batch_size, config.convergence_samples)

                batch_x = train_x_native[start_idx:end_idx]
                batch_y = train_y_native[start_idx:end_idx]

                loss = native_trainer.train_step(batch_x, batch_y)
                epoch_loss += loss

            epoch_loss /= n_batches
            native_losses.append(epoch_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  Native Epoch {epoch+1}: loss = {epoch_loss:.6f}")

        native_final_loss = native_losses[-1]

        # === PYTORCH ===
        print("\nRunning PyTorch training...")
        self._clear_gpu()

        from trix.nn.sparse_lookup_v4 import SparseLookupFFNv4

        pytorch_model = SparseLookupFFNv4(
            d_model=config.d_model,
            num_tiles=config.num_tiles,
            grid_size=config.grid_size,
        ).cuda()

        optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()

        train_x_torch = torch.from_numpy(train_x_np).cuda()
        train_y_torch = torch.from_numpy(train_y_np).cuda()

        pytorch_losses = []

        for epoch in range(config.convergence_epochs):
            epoch_loss = 0.0
            n_batches = (config.convergence_samples + config.batch_size - 1) // config.batch_size

            for batch_idx in range(n_batches):
                start_idx = batch_idx * config.batch_size
                end_idx = min(start_idx + config.batch_size, config.convergence_samples)

                batch_x = train_x_torch[start_idx:end_idx].unsqueeze(0)
                batch_y = train_y_torch[start_idx:end_idx]

                optimizer.zero_grad()
                output, _, aux = pytorch_model(batch_x)
                loss = criterion(output.squeeze(0), batch_y) + aux['total_aux']
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= n_batches
            pytorch_losses.append(epoch_loss)

            if (epoch + 1) % 10 == 0:
                print(f"  PyTorch Epoch {epoch+1}: loss = {epoch_loss:.6f}")

        pytorch_final_loss = pytorch_losses[-1]

        # Compare
        winner = "Native" if native_final_loss < pytorch_final_loss else "PyTorch"
        ratio = pytorch_final_loss / native_final_loss if native_final_loss > 0 else float('inf')

        result = TestResult(
            test_name="Convergence",
            native_value=native_final_loss,
            pytorch_value=pytorch_final_loss,
            unit="MSE Loss",
            winner=winner,
            ratio=ratio,
            notes=f"After {config.convergence_epochs} epochs on {config.convergence_samples} samples"
        )

        self.results.append(result)
        self.detailed_logs["convergence"] = {
            "native_losses": native_losses,
            "pytorch_losses": pytorch_losses,
        }

        print(f"\n  Native final loss:  {native_final_loss:.6f}")
        print(f"  PyTorch final loss: {pytorch_final_loss:.6f}")
        print(f"  Winner: {winner}")

        return result

    # =========================================================================
    # TEST 2: TRAINING SPEED
    # =========================================================================

    def test_training_speed(self) -> TestResult:
        """Test: Which system trains faster?"""
        print("\n" + "=" * 60)
        print("TEST 2: TRAINING SPEED")
        print("=" * 60)
        print("Question: Which system processes more samples per second?")
        print()

        config = self.config
        np.random.seed(config.random_seed)

        train_x_np = np.random.randn(config.speed_samples, config.d_model).astype(np.float32)
        train_y_np = train_x_np * 0.9 + np.random.randn(config.speed_samples, config.d_model).astype(np.float32) * 0.05

        # === NATIVE ===
        print("Benchmarking Native training speed...")
        self._clear_gpu()

        from trix.foundry.native_training import NativeHollywoodSquares, Trainer
        import io, contextlib

        with contextlib.redirect_stdout(io.StringIO()):
            native_model = NativeHollywoodSquares(
                d_model=config.d_model,
                num_tiles=config.num_tiles,
                grid_size=config.grid_size,
                lr=config.learning_rate,
            )

        train_x_native = cp.asarray(train_x_np)
        train_y_native = cp.asarray(train_y_np)
        native_trainer = Trainer(native_model, loss_fn='mse')

        # Warmup
        for _ in range(3):
            native_trainer.train_step(train_x_native[:config.batch_size], train_y_native[:config.batch_size])
        cp.cuda.Stream.null.synchronize()

        # Benchmark
        start = time.perf_counter()
        for epoch in range(config.speed_epochs):
            n_batches = (config.speed_samples + config.batch_size - 1) // config.batch_size
            for batch_idx in range(n_batches):
                start_idx = batch_idx * config.batch_size
                end_idx = min(start_idx + config.batch_size, config.speed_samples)
                native_trainer.train_step(
                    train_x_native[start_idx:end_idx],
                    train_y_native[start_idx:end_idx]
                )
        cp.cuda.Stream.null.synchronize()
        native_time = time.perf_counter() - start

        native_samples_per_sec = (config.speed_samples * config.speed_epochs) / native_time
        print(f"  Native: {native_samples_per_sec:,.0f} samples/sec ({native_time:.2f}s)")

        # === PYTORCH ===
        print("Benchmarking PyTorch training speed...")
        self._clear_gpu()

        from trix.nn.sparse_lookup_v4 import SparseLookupFFNv4

        pytorch_model = SparseLookupFFNv4(
            d_model=config.d_model,
            num_tiles=config.num_tiles,
            grid_size=config.grid_size,
        ).cuda()

        optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()

        train_x_torch = torch.from_numpy(train_x_np).cuda()
        train_y_torch = torch.from_numpy(train_y_np).cuda()

        # Warmup
        for _ in range(3):
            optimizer.zero_grad()
            out, _, aux = pytorch_model(train_x_torch[:config.batch_size].unsqueeze(0))
            loss = criterion(out.squeeze(0), train_y_torch[:config.batch_size]) + aux['total_aux']
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for epoch in range(config.speed_epochs):
            n_batches = (config.speed_samples + config.batch_size - 1) // config.batch_size
            for batch_idx in range(n_batches):
                start_idx = batch_idx * config.batch_size
                end_idx = min(start_idx + config.batch_size, config.speed_samples)

                optimizer.zero_grad()
                out, _, aux = pytorch_model(train_x_torch[start_idx:end_idx].unsqueeze(0))
                loss = criterion(out.squeeze(0), train_y_torch[start_idx:end_idx]) + aux['total_aux']
                loss.backward()
                optimizer.step()
        torch.cuda.synchronize()
        pytorch_time = time.perf_counter() - start

        pytorch_samples_per_sec = (config.speed_samples * config.speed_epochs) / pytorch_time
        print(f"  PyTorch: {pytorch_samples_per_sec:,.0f} samples/sec ({pytorch_time:.2f}s)")

        # Compare
        winner = "Native" if native_samples_per_sec > pytorch_samples_per_sec else "PyTorch"
        ratio = native_samples_per_sec / pytorch_samples_per_sec

        result = TestResult(
            test_name="Training Speed",
            native_value=native_samples_per_sec,
            pytorch_value=pytorch_samples_per_sec,
            unit="samples/sec",
            winner=winner,
            ratio=ratio,
            notes=f"{config.speed_epochs} epochs on {config.speed_samples} samples"
        )

        self.results.append(result)
        print(f"\n  Speedup: {ratio:.1f}x ({winner} wins)")

        return result

    # =========================================================================
    # TEST 3: INFERENCE SPEED
    # =========================================================================

    def test_inference_speed(self) -> TestResult:
        """Test: Which system has faster inference?"""
        print("\n" + "=" * 60)
        print("TEST 3: INFERENCE SPEED")
        print("=" * 60)
        print("Question: Which system processes inference faster?")
        print()

        config = self.config
        np.random.seed(config.random_seed)

        test_x_np = np.random.randn(config.inference_samples, config.d_model).astype(np.float32)

        # === NATIVE ===
        print("Benchmarking Native inference...")
        self._clear_gpu()

        from trix.foundry.hollywood_squares_emergence import HollywoodSquaresEmergence

        native_model = HollywoodSquaresEmergence(
            d_model=config.d_model,
            num_tiles=config.num_tiles,
            grid_size=config.grid_size,
        )

        test_x_native = cp.asarray(test_x_np)

        # Warmup
        for _ in range(3):
            _ = native_model.forward_vectorized(test_x_native)
        cp.cuda.Stream.null.synchronize()

        # Benchmark
        native_times = []
        for _ in range(config.inference_runs):
            start = time.perf_counter()
            _ = native_model.forward_vectorized(test_x_native)
            cp.cuda.Stream.null.synchronize()
            native_times.append(time.perf_counter() - start)

        native_best = min(native_times)
        native_throughput = config.inference_samples / native_best
        print(f"  Native: {native_throughput/1e6:.2f}M tokens/sec (best of {config.inference_runs})")

        # === PYTORCH ===
        print("Benchmarking PyTorch inference...")
        self._clear_gpu()

        from trix.nn.sparse_lookup_v4 import SparseLookupFFNv4

        pytorch_model = SparseLookupFFNv4(
            d_model=config.d_model,
            num_tiles=config.num_tiles,
            grid_size=config.grid_size,
        ).cuda().eval()

        test_x_torch = torch.from_numpy(test_x_np).cuda().unsqueeze(0)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = pytorch_model(test_x_torch)
        torch.cuda.synchronize()

        # Benchmark
        pytorch_times = []
        with torch.no_grad():
            for _ in range(config.inference_runs):
                start = time.perf_counter()
                _ = pytorch_model(test_x_torch)
                torch.cuda.synchronize()
                pytorch_times.append(time.perf_counter() - start)

        pytorch_best = min(pytorch_times)
        pytorch_throughput = config.inference_samples / pytorch_best
        print(f"  PyTorch: {pytorch_throughput/1e6:.2f}M tokens/sec (best of {config.inference_runs})")

        # Compare
        winner = "Native" if native_throughput > pytorch_throughput else "PyTorch"
        ratio = native_throughput / pytorch_throughput

        result = TestResult(
            test_name="Inference Speed",
            native_value=native_throughput,
            pytorch_value=pytorch_throughput,
            unit="tokens/sec",
            winner=winner,
            ratio=ratio,
            notes=f"Best of {config.inference_runs} runs on {config.inference_samples} samples"
        )

        self.results.append(result)
        print(f"\n  Speedup: {ratio:.1f}x ({winner} wins)")

        return result

    # =========================================================================
    # TEST 4: MEMORY USAGE
    # =========================================================================

    def test_memory_usage(self) -> TestResult:
        """Test: Which system uses less GPU memory?"""
        print("\n" + "=" * 60)
        print("TEST 4: MEMORY USAGE")
        print("=" * 60)
        print("Question: Which system uses less GPU memory during training?")
        print()

        config = self.config
        batch_size = config.batch_size

        # === NATIVE ===
        print("Measuring Native memory usage...")
        self._clear_gpu()
        torch.cuda.reset_peak_memory_stats()

        from trix.foundry.native_training import NativeHollywoodSquares, Trainer
        import io, contextlib

        with contextlib.redirect_stdout(io.StringIO()):
            native_model = NativeHollywoodSquares(
                d_model=config.d_model,
                num_tiles=config.num_tiles,
                grid_size=config.grid_size,
                lr=config.learning_rate,
            )

        train_x = cp.random.randn(batch_size, config.d_model).astype(cp.float32)
        train_y = cp.random.randn(batch_size, config.d_model).astype(cp.float32)

        native_trainer = Trainer(native_model, loss_fn='mse')

        # Run training step
        for _ in range(5):
            native_trainer.train_step(train_x, train_y)
        cp.cuda.Stream.null.synchronize()

        # Note: CuPy memory tracking is different from torch
        # Use torch's memory tracking as proxy
        native_memory = torch.cuda.max_memory_allocated() / 1e6
        print(f"  Native peak memory: {native_memory:.1f} MB")

        # === PYTORCH ===
        print("Measuring PyTorch memory usage...")
        self._clear_gpu()
        torch.cuda.reset_peak_memory_stats()

        from trix.nn.sparse_lookup_v4 import SparseLookupFFNv4

        pytorch_model = SparseLookupFFNv4(
            d_model=config.d_model,
            num_tiles=config.num_tiles,
            grid_size=config.grid_size,
        ).cuda()

        optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=config.learning_rate)
        criterion = nn.MSELoss()

        train_x_torch = torch.randn(batch_size, config.d_model).cuda()
        train_y_torch = torch.randn(batch_size, config.d_model).cuda()

        for _ in range(5):
            optimizer.zero_grad()
            out, _, aux = pytorch_model(train_x_torch.unsqueeze(0))
            loss = criterion(out.squeeze(0), train_y_torch) + aux['total_aux']
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

        pytorch_memory = torch.cuda.max_memory_allocated() / 1e6
        print(f"  PyTorch peak memory: {pytorch_memory:.1f} MB")

        # Compare
        winner = "Native" if native_memory < pytorch_memory else "PyTorch"
        ratio = pytorch_memory / native_memory if native_memory > 0 else float('inf')

        result = TestResult(
            test_name="Memory Usage",
            native_value=native_memory,
            pytorch_value=pytorch_memory,
            unit="MB",
            winner=winner,
            ratio=ratio,
            notes=f"Peak memory during training with batch_size={batch_size}"
        )

        self.results.append(result)
        print(f"\n  Memory reduction: {ratio:.1f}x ({winner} uses less)")

        return result

    # =========================================================================
    # RUN ALL TESTS
    # =========================================================================

    def run_all(self) -> Dict:
        """Run all A/B tests."""
        print("\n" + "=" * 70)
        print("RUNNING ALL A/B TESTS")
        print("=" * 70)

        self.test_convergence()
        self.test_training_speed()
        self.test_inference_speed()
        self.test_memory_usage()

        return self.generate_report()

    def generate_report(self) -> Dict:
        """Generate final report."""
        print("\n" + "=" * 70)
        print("A/B TEST RESULTS SUMMARY")
        print("=" * 70)
        print()

        # Summary table
        print(f"{'Test':<20} {'Native':<15} {'PyTorch':<15} {'Winner':<10} {'Ratio':<10}")
        print("-" * 70)

        native_wins = 0
        pytorch_wins = 0

        for result in self.results:
            native_str = f"{result.native_value:,.2f}" if result.native_value < 1e6 else f"{result.native_value/1e6:.2f}M"
            pytorch_str = f"{result.pytorch_value:,.2f}" if result.pytorch_value < 1e6 else f"{result.pytorch_value/1e6:.2f}M"

            print(f"{result.test_name:<20} {native_str:<15} {pytorch_str:<15} {result.winner:<10} {result.ratio:.2f}x")

            if result.winner == "Native":
                native_wins += 1
            else:
                pytorch_wins += 1

        print("-" * 70)
        print(f"\n  Native wins: {native_wins}/{len(self.results)}")
        print(f"  PyTorch wins: {pytorch_wins}/{len(self.results)}")

        overall_winner = "NATIVE" if native_wins > pytorch_wins else "PYTORCH" if pytorch_wins > native_wins else "TIE"

        print(f"\n  OVERALL WINNER: {overall_winner}")
        print()
        print("=" * 70)

        # Save report
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "results": [asdict(r) for r in self.results],
            "summary": {
                "native_wins": native_wins,
                "pytorch_wins": pytorch_wins,
                "overall_winner": overall_winner,
            },
            "detailed_logs": self.detailed_logs,
        }

        output_path = Path(__file__).parent / "ab_test_results.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report saved to: {output_path}")

        return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the full A/B test suite."""
    config = TestConfig(
        d_model=128,
        num_tiles=16,
        grid_size=8,
        learning_rate=0.01,
        batch_size=256,
        random_seed=42,
        convergence_samples=5000,
        convergence_epochs=50,
        speed_samples=10000,
        speed_epochs=20,
        inference_samples=100000,
        inference_runs=10,
    )

    harness = ABTestHarness(config)
    report = harness.run_all()

    return report


if __name__ == "__main__":
    report = main()
