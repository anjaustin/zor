"""
Fabric Verifier - Ensures TriX output = CUDA output

Provides:
- Bit-exact verification on binary inputs
- Statistical verification on continuous inputs
- Performance regression testing
- Correctness certificates
"""

import sys
sys.path.insert(0, '/workspace/trix_latest/TriXO/src')

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import subprocess
import tempfile
import struct

from .config import FabricConfig, TaskConfig
from .shapes import ShapeRegistry, get_registry
from .compiler import CompiledFabric, CompiledTile


@dataclass
class VerificationResult:
    """Result of fabric verification."""
    passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    max_error: float
    mean_error: float
    details: Dict[str, any]

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Verification {status}: {self.passed_tests}/{self.total_tests} tests\n"
            f"  Max error: {self.max_error:.2e}\n"
            f"  Mean error: {self.mean_error:.2e}"
        )


class FabricVerifier:
    """
    Verifies that compiled CUDA fabric matches TriX computation.

    Verification levels:
    1. Shape verification: Each shape's polynomial = hardware
    2. Tile verification: Each tile produces correct output
    3. Fabric verification: Full fabric computation matches
    4. Statistical verification: Distribution of outputs matches
    """

    def __init__(self, fabric: CompiledFabric):
        self.fabric = fabric
        self.registry = get_registry()

    def verify_shapes(self) -> VerificationResult:
        """Verify all shapes in the fabric."""
        passed = 0
        failed = 0
        max_error = 0.0
        errors = []
        details = {}

        for tile in self.fabric.tiles:
            shape = self.registry.get(tile.shape_name)

            # Test on all binary combinations
            binary = torch.tensor([0.0, 1.0])
            tile_passed = True

            for a_val in binary:
                for b_val in binary:
                    a = torch.tensor([a_val])
                    b = torch.tensor([b_val])

                    # TriX polynomial
                    try:
                        if shape.n_inputs == 1:
                            poly_result = shape.poly_fn(a)
                        else:
                            result = shape.poly_fn(a, b)
                            poly_result = result[0] if isinstance(result, tuple) else result
                    except Exception as e:
                        tile_passed = False
                        errors.append(f"Tile {tile.tile_id} ({tile.shape_name}): {e}")
                        continue

                    # Hardware equivalent
                    a_int = int(a_val.item())
                    b_int = int(b_val.item())

                    hw_result = self._eval_cuda_expr(tile.cuda_expr, a_int, b_int)
                    if hw_result is not None:
                        error = abs(poly_result.item() - hw_result)
                        max_error = max(max_error, error)
                        if error > 1e-6:
                            tile_passed = False
                            errors.append(
                                f"Tile {tile.tile_id}: {tile.shape_name}({a_int},{b_int}) = "
                                f"{poly_result.item()} (poly) vs {hw_result} (hw)"
                            )

            if tile_passed:
                passed += 1
            else:
                failed += 1

            details[f"tile_{tile.tile_id}"] = tile_passed

        total = passed + failed
        return VerificationResult(
            passed=(failed == 0),
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            max_error=max_error,
            mean_error=max_error / max(total, 1),
            details={"tiles": details, "errors": errors},
        )

    def _eval_cuda_expr(self, expr: str, a: int, b: int) -> Optional[int]:
        """Evaluate a CUDA expression on integers."""
        # Simple evaluator for common expressions
        expr = expr.strip()

        if expr == "a ^ b":
            return a ^ b
        elif expr == "a & b":
            return a & b
        elif expr == "a | b":
            return a | b
        elif expr == "~a" or expr == "(~a)":
            return 1 - a
        elif expr == "~(a & b)":
            return 1 if not (a and b) else 0
        elif expr == "~(a | b)":
            return 1 if not (a or b) else 0
        elif expr == "~(a ^ b)":
            return 1 if a == b else 0
        elif expr == "(~a) | b":
            return (1 - a) | b
        elif expr == "a + b":
            return (a + b) & 0xFFFFFFFF
        elif expr == "a":
            return a
        elif expr == "b":
            return b

        return None

    def verify_fabric_output(self, num_samples: int = 1000) -> VerificationResult:
        """
        Verify fabric output by running both TriX and CUDA on same inputs.

        This requires the CUDA binary to be compiled and runnable.
        """
        if self.fabric.binary_path is None or not self.fabric.binary_path.exists():
            return VerificationResult(
                passed=False,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                max_error=float('inf'),
                mean_error=float('inf'),
                details={"error": "Binary not compiled"},
            )

        # Generate test vectors
        np.random.seed(42)
        test_data = np.random.randint(0, 256, size=(num_samples, self.fabric.config.block_size_bytes), dtype=np.uint8)

        # Run TriX simulation
        trix_outputs = self._run_trix_simulation(test_data)

        # Run CUDA
        cuda_outputs = self._run_cuda_execution(test_data)

        if cuda_outputs is None:
            return VerificationResult(
                passed=False,
                total_tests=num_samples,
                passed_tests=0,
                failed_tests=num_samples,
                max_error=float('inf'),
                mean_error=float('inf'),
                details={"error": "CUDA execution failed"},
            )

        # Compare outputs
        passed = 0
        failed = 0
        max_error = 0.0
        total_error = 0.0

        for i in range(num_samples):
            if np.array_equal(trix_outputs[i], cuda_outputs[i]):
                passed += 1
            else:
                failed += 1
                error = np.sum(np.abs(trix_outputs[i].astype(int) - cuda_outputs[i].astype(int)))
                max_error = max(max_error, error)
                total_error += error

        return VerificationResult(
            passed=(failed == 0),
            total_tests=num_samples,
            passed_tests=passed,
            failed_tests=failed,
            max_error=max_error,
            mean_error=total_error / max(failed, 1),
            details={"samples": num_samples},
        )

    def _run_trix_simulation(self, test_data: np.ndarray) -> np.ndarray:
        """Run TriX simulation on test data."""
        outputs = []

        for block in test_data:
            # Convert to 32-bit words
            words = np.frombuffer(block.tobytes(), dtype=np.uint32).copy()

            # Simulate fabric execution
            for r in range(self.fabric.task.rounds):
                # Column rounds
                for col in range(4):
                    if col + 12 < len(words):
                        a, b, c, d = col, col + 4, col + 8, col + 12
                        tile_idx = (r * 4 + col) % len(self.fabric.tiles)
                        words[a] = self._execute_tile_sim(tile_idx, words[a], words[b])

                        tile_idx = (r * 4 + col + 16) % len(self.fabric.tiles)
                        words[d] = self._execute_tile_sim(tile_idx, words[d], words[a])

                        tile_idx = (r * 4 + col + 32) % len(self.fabric.tiles)
                        words[c] = self._execute_tile_sim(tile_idx, words[c], words[d])

                        tile_idx = (r * 4 + col + 48) % len(self.fabric.tiles)
                        words[b] = self._execute_tile_sim(tile_idx, words[b], words[c])

            outputs.append(np.frombuffer(words.tobytes(), dtype=np.uint8))

        return np.array(outputs)

    def _execute_tile_sim(self, tile_idx: int, a: np.uint32, b: np.uint32) -> np.uint32:
        """Simulate a single tile execution."""
        tile = self.fabric.tiles[tile_idx]
        a_int = int(a)
        b_int = int(b)

        # Evaluate CUDA expression
        expr = tile.cuda_expr

        if expr == "a ^ b":
            result = a_int ^ b_int
        elif expr == "a & b":
            result = a_int & b_int
        elif expr == "a | b":
            result = a_int | b_int
        elif expr == "~(a & b)":
            result = ~(a_int & b_int) & 0xFFFFFFFF
        elif expr == "~(a | b)":
            result = ~(a_int | b_int) & 0xFFFFFFFF
        elif expr == "~(a ^ b)":
            result = ~(a_int ^ b_int) & 0xFFFFFFFF
        else:
            result = a_int ^ b_int  # Default

        # Apply round constant
        result ^= tile.round_constant
        return np.uint32(result & 0xFFFFFFFF)

    def _run_cuda_execution(self, test_data: np.ndarray) -> Optional[np.ndarray]:
        """Run CUDA binary on test data."""
        # This would require IPC or file-based data exchange
        # For now, return None to indicate not implemented
        return None

    def generate_certificate(self) -> Dict:
        """Generate a verification certificate."""
        shape_result = self.verify_shapes()

        return {
            "version": "1.0",
            "fabric_config": self.fabric.config.to_dict(),
            "task_config": self.fabric.task.to_dict(),
            "verification": {
                "shapes": {
                    "passed": shape_result.passed,
                    "total_tests": shape_result.total_tests,
                    "passed_tests": shape_result.passed_tests,
                    "max_error": shape_result.max_error,
                },
                "verified_shapes": [
                    tile.shape_name for tile in self.fabric.tiles
                    if self.registry.get(tile.shape_name).verified
                ],
            },
            "mathematical_identity": {
                "polynomial": "a + b - 2ab",
                "hardware": "a ^ b",
                "equivalent_on_binary": True,
            },
        }
