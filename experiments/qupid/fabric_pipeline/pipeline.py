"""
Main Pipeline - Orchestrates the complete TriX → XOR Fabric flow

The pipeline:
1. Configure - Set fabric and task parameters
2. (Optional) Train - Train TriX model to learn computation
3. Freeze - Export frozen shapes and signatures
4. Compile - Generate optimized CUDA kernel
5. Verify - Ensure TriX output = CUDA output
6. Benchmark - Measure performance

Usage:
    pipeline = Pipeline(config, task)
    pipeline.freeze()
    pipeline.compile("fabric.cu")
    pipeline.verify()
    result = pipeline.benchmark()
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/TriXO/src')

import time
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import tempfile

from .config import FabricConfig, TaskConfig, get_preset
from .shapes import ShapeRegistry, get_registry
from .compiler import FabricCompiler, CompiledFabric
from .verifier import FabricVerifier, VerificationResult


@dataclass
class BenchmarkResult:
    """Result of fabric benchmark."""
    blocks_per_sec: float
    bytes_per_sec: float
    ops_per_sec: float
    bits_per_sec: float
    time_ms: float
    num_blocks: int
    iterations: int

    def __str__(self) -> str:
        return (
            f"Benchmark Results:\n"
            f"  Throughput: {self.bytes_per_sec / 1e9:.2f} GB/s\n"
            f"  Blocks/sec: {self.blocks_per_sec / 1e6:.2f} M\n"
            f"  Shape ops/sec: {self.ops_per_sec / 1e9:.2f} G\n"
            f"  Bits/sec: {self.bits_per_sec / 1e12:.2f} T"
        )


class Pipeline:
    """
    Complete TriX → XOR Fabric pipeline.

    Orchestrates configuration, compilation, verification, and benchmarking.
    """

    def __init__(
        self,
        config: Optional[FabricConfig] = None,
        task: Optional[TaskConfig] = None,
        preset: Optional[str] = None,
    ):
        """
        Initialize pipeline.

        Args:
            config: Fabric configuration (or use preset)
            task: Task configuration
            preset: Use a preset configuration ("tiny", "small", "default", "large", "huge")
        """
        if preset:
            self.config = get_preset(preset)
        else:
            self.config = config or FabricConfig()

        self.task = task or TaskConfig()
        self.registry = get_registry()

        # State
        self.fabric: Optional[CompiledFabric] = None
        self.verification: Optional[VerificationResult] = None
        self.benchmark_result: Optional[BenchmarkResult] = None

        # Paths
        self.output_dir: Optional[Path] = None

    def configure(
        self,
        num_tiles: Optional[int] = None,
        signature_bits: Optional[int] = None,
        block_size_bits: Optional[int] = None,
        rounds: Optional[int] = None,
        shape_pattern: Optional[List[str]] = None,
    ) -> "Pipeline":
        """Update configuration."""
        if num_tiles:
            self.config.num_tiles = num_tiles
        if signature_bits:
            self.config.signature_bits = signature_bits
        if block_size_bits:
            self.config.block_size_bits = block_size_bits
        if rounds:
            self.task.rounds = rounds
        if shape_pattern:
            self.task.shape_pattern = shape_pattern
        return self

    def freeze(self) -> "Pipeline":
        """
        Freeze shapes from registry.

        In a full implementation, this would:
        1. Take a trained TriX model
        2. Extract learned tile signatures
        3. Identify frozen shapes for each tile

        Currently uses the configured shape pattern.
        """
        print("Freezing shapes...")

        # Validate shapes exist
        for shape_name in self.task.shape_pattern:
            if shape_name not in self.registry:
                raise ValueError(f"Unknown shape: {shape_name}")
            shape = self.registry.get(shape_name)
            if not shape.verified:
                print(f"  Warning: Shape '{shape_name}' not verified (poly ≠ hw)")

        # Count verified shapes
        verified = sum(
            1 for s in self.task.shape_pattern
            if self.registry.get(s).verified
        )
        total = len(self.task.shape_pattern)
        print(f"  Pattern: {self.task.shape_pattern}")
        print(f"  Verified: {verified}/{total} shapes")

        return self

    def compile(
        self,
        output_path: Optional[str] = None,
        compile_binary: bool = True,
    ) -> "Pipeline":
        """
        Compile frozen shapes to CUDA.

        Args:
            output_path: Path for output files (default: temp directory)
            compile_binary: Whether to compile to binary (requires nvcc)
        """
        print("Compiling to CUDA...")

        if output_path:
            self.output_dir = Path(output_path).parent
            cuda_path = Path(output_path)
        else:
            self.output_dir = Path(tempfile.mkdtemp(prefix="fabric_"))
            cuda_path = self.output_dir / "fabric.cu"

        compiler = FabricCompiler(self.config, self.task)

        if compile_binary:
            binary_path = cuda_path.with_suffix("")
            self.fabric = compiler.compile_to_binary(binary_path)
            print(f"  CUDA source: {cuda_path}")
            print(f"  Binary: {binary_path}")
        else:
            self.fabric = compiler.compile_to_file(cuda_path)
            print(f"  CUDA source: {cuda_path}")

        print(f"  Tiles: {len(self.fabric.tiles)}")
        print(f"  Code size: {len(self.fabric.cuda_code)} bytes")

        return self

    def verify(self) -> "Pipeline":
        """
        Verify compiled fabric.

        Checks that:
        1. All shape polynomials match hardware on binary inputs
        2. Tile dispatch is correct
        """
        if self.fabric is None:
            raise RuntimeError("Must compile before verifying")

        print("Verifying fabric...")

        verifier = FabricVerifier(self.fabric)
        self.verification = verifier.verify_shapes()

        if self.verification.passed:
            print(f"  ✓ All {self.verification.total_tests} tiles verified")
        else:
            print(f"  ✗ {self.verification.failed_tests} tiles failed verification")
            for error in self.verification.details.get("errors", [])[:5]:
                print(f"    - {error}")

        return self

    def benchmark(
        self,
        num_blocks: int = 1 << 20,
        iterations: int = 100,
    ) -> BenchmarkResult:
        """
        Benchmark compiled fabric.

        Args:
            num_blocks: Number of blocks to process
            iterations: Number of benchmark iterations

        Returns:
            BenchmarkResult with performance metrics
        """
        if self.fabric is None or self.fabric.binary_path is None:
            raise RuntimeError("Must compile binary before benchmarking")

        print(f"Benchmarking ({num_blocks} blocks, {iterations} iterations)...")

        # Use absolute path for binary
        binary_path = self.fabric.binary_path.resolve()

        # Run benchmark
        result = subprocess.run(
            [str(binary_path), str(num_blocks), str(iterations)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Benchmark failed: {result.stderr}")

        # Parse output
        output = result.stdout
        print(output)

        # Extract metrics from output
        self.benchmark_result = self._parse_benchmark_output(
            output, num_blocks, iterations
        )

        return self.benchmark_result

    def _parse_benchmark_output(
        self,
        output: str,
        num_blocks: int,
        iterations: int,
    ) -> BenchmarkResult:
        """Parse benchmark output to extract metrics."""
        lines = output.split("\n")

        throughput = 0.0
        blocks_per_sec = 0.0
        ops_per_sec = 0.0
        bits_per_sec = 0.0
        time_ms = 0.0

        for line in lines:
            if "Throughput:" in line and "GB/s" in line:
                throughput = float(line.split(":")[1].split()[0])
            elif "Blocks/sec:" in line:
                blocks_per_sec = float(line.split(":")[1].split()[0]) * 1e6
            elif "Shape ops/sec:" in line:
                ops_per_sec = float(line.split(":")[1].split()[0]) * 1e9
            elif "Bits/sec:" in line:
                bits_per_sec = float(line.split(":")[1].split()[0]) * 1e12
            elif "Time:" in line and "ms" in line:
                time_ms = float(line.split(":")[1].split()[0])

        return BenchmarkResult(
            blocks_per_sec=blocks_per_sec,
            bytes_per_sec=throughput * 1e9,
            ops_per_sec=ops_per_sec,
            bits_per_sec=bits_per_sec,
            time_ms=time_ms,
            num_blocks=num_blocks,
            iterations=iterations,
        )

    def run(
        self,
        output_path: str = "fabric.cu",
        num_blocks: int = 1 << 20,
        iterations: int = 100,
    ) -> Tuple[VerificationResult, BenchmarkResult]:
        """
        Run complete pipeline: freeze → compile → verify → benchmark

        Args:
            output_path: Path for CUDA output
            num_blocks: Blocks for benchmark
            iterations: Benchmark iterations

        Returns:
            Tuple of (verification result, benchmark result)
        """
        print("=" * 70)
        print("TRIX → XOR FABRIC PIPELINE")
        print("=" * 70)
        print()

        self.freeze()
        print()

        self.compile(output_path)
        print()

        self.verify()
        print()

        result = self.benchmark(num_blocks, iterations)
        print()

        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print()
        print(f"Verification: {'PASSED' if self.verification.passed else 'FAILED'}")
        print(f"Throughput: {result.bytes_per_sec / 1e9:.2f} GB/s")
        print(f"Shape ops: {result.ops_per_sec / 1e9:.2f} G/s")
        print()

        return self.verification, result

    def save_config(self, path: str):
        """Save configuration to JSON."""
        config = {
            "fabric": self.config.to_dict(),
            "task": self.task.to_dict(),
        }
        Path(path).write_text(json.dumps(config, indent=2))

    @classmethod
    def load_config(cls, path: str) -> "Pipeline":
        """Load pipeline from saved configuration."""
        config = json.loads(Path(path).read_text())
        return cls(
            config=FabricConfig.from_dict(config["fabric"]),
            task=TaskConfig.from_dict(config["task"]),
        )

    def summary(self) -> str:
        """Generate summary of pipeline state."""
        lines = [
            "Pipeline Summary",
            "=" * 40,
            f"Tiles: {self.config.num_tiles}",
            f"Signature bits: {self.config.signature_bits}",
            f"Block size: {self.config.block_size_bits} bits",
            f"Rounds: {self.task.rounds}",
            f"Shape pattern: {self.task.shape_pattern}",
            "",
        ]

        if self.fabric:
            lines.append(f"Compiled: Yes ({len(self.fabric.cuda_code)} bytes)")
            if self.fabric.binary_path:
                lines.append(f"Binary: {self.fabric.binary_path}")

        if self.verification:
            status = "PASSED" if self.verification.passed else "FAILED"
            lines.append(f"Verification: {status}")

        if self.benchmark_result:
            lines.append(f"Throughput: {self.benchmark_result.bytes_per_sec / 1e9:.2f} GB/s")

        return "\n".join(lines)
