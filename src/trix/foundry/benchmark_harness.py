"""
Hollywood Squares Foundry - Benchmark Harness

Reproducible benchmarks with JSON logging.
All measurements recorded with timestamps and hardware info.

Usage:
    PYTHONPATH=src python src/trix/foundry/benchmark_harness.py
    PYTHONPATH=src python src/trix/foundry/benchmark_harness.py --output results.json
    PYTHONPATH=src python src/trix/foundry/benchmark_harness.py --runs 20
"""

import argparse
import json
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# GPU imports
import cupy as cp
import torch


def get_hardware_info() -> Dict[str, Any]:
    """Collect hardware information for reproducibility."""
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)

    return {
        "gpu": {
            "name": props['name'].decode(),
            "compute_capability": f"{props['major']}.{props['minor']}",
            "sm_count": props['multiProcessorCount'],
            "total_memory_gb": round(props['totalGlobalMem'] / 1e9, 2),
            "max_threads_per_block": props['maxThreadsPerBlock'],
        },
        "cuda": {
            "version": cp.cuda.runtime.runtimeGetVersion(),
            "driver_version": cp.cuda.runtime.driverGetVersion(),
        },
        "cupy_version": cp.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }


def get_software_info() -> Dict[str, str]:
    """Collect software version information."""
    return {
        "python": sys.version,
        "platform": sys.platform,
    }


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name: str, batch_size: int):
        self.name = name
        self.batch_size = batch_size
        self.times: List[float] = []
        self.error: str = None

    def add_time(self, t: float):
        self.times.append(t)

    def to_dict(self) -> Dict[str, Any]:
        if self.error:
            return {
                "name": self.name,
                "batch_size": self.batch_size,
                "error": self.error,
            }

        times_ms = [t * 1000 for t in self.times]
        throughput = self.batch_size / min(self.times) if self.times else 0

        return {
            "name": self.name,
            "batch_size": self.batch_size,
            "runs": len(self.times),
            "times_ms": {
                "min": round(min(times_ms), 4),
                "max": round(max(times_ms), 4),
                "mean": round(np.mean(times_ms), 4),
                "std": round(np.std(times_ms), 4),
                "median": round(np.median(times_ms), 4),
            },
            "throughput": {
                "tokens_per_sec": round(throughput, 0),
                "millions_per_sec": round(throughput / 1e6, 4),
            },
        }


class BenchmarkHarness:
    """Main benchmark harness with logging."""

    def __init__(self, output_path: Path = None, runs: int = 10):
        self.output_path = output_path or Path(__file__).parent / "benchmark_results.json"
        self.runs = runs
        self.results: Dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "runs_per_benchmark": runs,
            },
            "hardware": get_hardware_info(),
            "software": get_software_info(),
            "benchmarks": {},
        }

    def log(self, msg: str):
        """Print with timestamp."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def run_benchmark(
        self,
        name: str,
        fn: callable,
        batch_sizes: List[int],
        warmup_runs: int = 3,
    ) -> Dict[str, List[BenchmarkResult]]:
        """Run a benchmark across multiple batch sizes."""
        self.log(f"Starting benchmark: {name}")
        results = []

        for batch_size in batch_sizes:
            result = BenchmarkResult(name, batch_size)

            try:
                cp.get_default_memory_pool().free_all_blocks()
                torch.cuda.empty_cache()

                # Warmup
                for _ in range(warmup_runs):
                    fn(batch_size)
                    cp.cuda.Stream.null.synchronize()

                # Benchmark
                for run in range(self.runs):
                    start = time.perf_counter()
                    fn(batch_size)
                    cp.cuda.Stream.null.synchronize()
                    elapsed = time.perf_counter() - start
                    result.add_time(elapsed)

                throughput = result.batch_size / min(result.times) / 1e6
                self.log(f"  {batch_size:>10,}: {throughput:.2f} M/s")

            except Exception as e:
                result.error = str(e)
                self.log(f"  {batch_size:>10,}: ERROR - {e}")

            results.append(result)

        return results

    def benchmark_emergence(self) -> List[BenchmarkResult]:
        """Benchmark Emergence kernel."""
        from trix.foundry.hollywood_squares_emergence import HollywoodSquaresEmergence

        model = HollywoodSquaresEmergence(d_model=128, num_tiles=16)

        def run(batch_size):
            x = cp.random.randn(batch_size, 128).astype(cp.float32)
            positions = cp.arange(batch_size, dtype=cp.float32) % 64
            return model.forward_vectorized(x, positions)

        batch_sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        return self.run_benchmark("emergence_vectorized", run, batch_sizes)

    def benchmark_emergence_standard(self) -> List[BenchmarkResult]:
        """Benchmark Emergence standard kernel."""
        from trix.foundry.hollywood_squares_emergence import HollywoodSquaresEmergence

        model = HollywoodSquaresEmergence(d_model=128, num_tiles=16)

        def run(batch_size):
            x = cp.random.randn(batch_size, 128).astype(cp.float32)
            positions = cp.arange(batch_size, dtype=cp.float32) % 64
            return model.forward(x, positions)

        batch_sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        return self.run_benchmark("emergence_standard", run, batch_sizes)

    def benchmark_pytorch(self) -> List[BenchmarkResult]:
        """Benchmark PyTorch SparseLookupFFNv4."""
        from trix.nn.sparse_lookup_v4 import SparseLookupFFNv4

        model = SparseLookupFFNv4(d_model=128, num_tiles=16).cuda().eval()

        def run(batch_size):
            x = torch.randn(1, batch_size, 128).cuda()
            with torch.no_grad():
                return model(x)

        batch_sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        return self.run_benchmark("pytorch_sparse_lookup_v4", run, batch_sizes)

    def benchmark_fast_kernel(self) -> List[BenchmarkResult]:
        """Benchmark Fast kernel (shared memory, float32)."""
        from trix.foundry.hollywood_squares_ffn_fast import HollywoodSquaresFFNFast

        model = HollywoodSquaresFFNFast(d_model=128, num_tiles=16)

        def run(batch_size):
            x = cp.random.randn(batch_size, 128).astype(cp.float32)
            positions = cp.arange(batch_size, dtype=cp.float32) % 64
            return model.forward_fast(x, positions)

        batch_sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        return self.run_benchmark("fast_shared_memory", run, batch_sizes)

    def compute_speedups(self):
        """Compute speedups between implementations."""
        benchmarks = self.results["benchmarks"]

        if "pytorch_sparse_lookup_v4" not in benchmarks:
            return

        pytorch_results = {
            r["batch_size"]: r for r in benchmarks["pytorch_sparse_lookup_v4"]
            if "error" not in r
        }

        speedups = {}
        for name, results in benchmarks.items():
            if name == "pytorch_sparse_lookup_v4":
                continue

            speedups[name] = []
            for r in results:
                if "error" in r:
                    continue
                bs = r["batch_size"]
                if bs in pytorch_results:
                    pt_time = pytorch_results[bs]["times_ms"]["min"]
                    our_time = r["times_ms"]["min"]
                    speedup = pt_time / our_time
                    speedups[name].append({
                        "batch_size": bs,
                        "speedup": round(speedup, 2),
                        "pytorch_ms": pt_time,
                        "ours_ms": our_time,
                    })

        self.results["speedups"] = speedups

    def save_results(self):
        """Save results to JSON file."""
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        self.log(f"Results saved to: {self.output_path}")

    def print_summary(self):
        """Print summary of results."""
        print()
        print("=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print()
        print(f"Hardware: {self.results['hardware']['gpu']['name']}")
        print(f"SMs: {self.results['hardware']['gpu']['sm_count']}")
        print()

        # Peak throughputs
        print("Peak Throughputs (M tokens/sec):")
        print("-" * 40)
        for name, results in self.results["benchmarks"].items():
            valid_results = [r for r in results if "error" not in r]
            if valid_results:
                peak = max(r["throughput"]["millions_per_sec"] for r in valid_results)
                print(f"  {name:30}: {peak:.2f}")

        # Speedups
        if "speedups" in self.results:
            print()
            print("Peak Speedups vs PyTorch:")
            print("-" * 40)
            for name, speedups in self.results["speedups"].items():
                if speedups:
                    peak = max(s["speedup"] for s in speedups)
                    print(f"  {name:30}: {peak:.1f}x")

        print()
        print("=" * 70)

    def run_all(self):
        """Run all benchmarks."""
        self.log("Starting full benchmark suite")
        print()

        # Run all benchmarks
        self.results["benchmarks"]["emergence_vectorized"] = [
            r.to_dict() for r in self.benchmark_emergence()
        ]

        self.results["benchmarks"]["emergence_standard"] = [
            r.to_dict() for r in self.benchmark_emergence_standard()
        ]

        self.results["benchmarks"]["fast_shared_memory"] = [
            r.to_dict() for r in self.benchmark_fast_kernel()
        ]

        self.results["benchmarks"]["pytorch_sparse_lookup_v4"] = [
            r.to_dict() for r in self.benchmark_pytorch()
        ]

        # Compute speedups
        self.compute_speedups()

        # Save and summarize
        self.save_results()
        self.print_summary()


def main():
    parser = argparse.ArgumentParser(description="Hollywood Squares Benchmark Harness")
    parser.add_argument("--output", type=Path, help="Output JSON file path")
    parser.add_argument("--runs", type=int, default=10, help="Runs per benchmark")
    args = parser.parse_args()

    harness = BenchmarkHarness(output_path=args.output, runs=args.runs)
    harness.run_all()


if __name__ == "__main__":
    main()
