"""
System: The executable artifact built by the Foundry.

A System is a compiled, validated, exportable deterministic system.

Usage:
    from trix.forge import Foundry

    foundry = Foundry(bits=8)
    foundry.atom("xor", lambda a, b: a ^ b)
    foundry.atom("and", lambda a, b: a & b)

    system = foundry.build()

    # Execute
    result = system.execute(42, 13, "xor")

    # Validate (100% or fail)
    validation = system.validate(exhaustive=True)
    assert validation.all_passed()

    # Export
    system.export_cuda("output/")
    system.export_verilog("output/")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any
from pathlib import Path

from .term import (
    ShapeTerms, ValidationResult,
    validate_quick, validate_edge, validate_exhaustive_8bit,
    evaluate_batch, compute_fast, is_fast_path,
    _int_to_bits, _bits_to_int,
)
from .backend import Backend, get_backend

import torch


# =============================================================================
# VALIDATION RESULT AGGREGATOR
# =============================================================================

@dataclass
class SystemValidation:
    """Aggregated validation results for a system."""

    results: Dict[str, ValidationResult] = field(default_factory=dict)
    bits: int = 8
    mode: str = "unknown"

    def all_passed(self) -> bool:
        """True if all shapes passed validation."""
        return all(r.passed for r in self.results.values())

    def accuracy(self) -> float:
        """Average accuracy across all shapes."""
        if not self.results:
            return 0.0
        return sum(r.accuracy for r in self.results.values()) / len(self.results)

    def failures(self) -> Dict[str, List]:
        """Get failures by shape name."""
        return {name: r.failures for name, r in self.results.items() if r.failures}

    def summary(self) -> str:
        """Human-readable validation summary."""
        lines = [
            f"System Validation ({self.mode}, {self.bits}-bit)",
            f"  Shapes: {len(self.results)}",
            f"  All passed: {self.all_passed()}",
            f"  Average accuracy: {self.accuracy() * 100:.2f}%",
        ]

        for name, result in self.results.items():
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"    {name}: {status} ({result.accuracy * 100:.2f}%)")
            if result.failures:
                lines.append(f"      First failure: {result.failures[0]}")

        return "\n".join(lines)


# =============================================================================
# SYSTEM CLASS
# =============================================================================

@dataclass
class System:
    """
    An executable deterministic system.

    The System is the artifact produced by Foundry.build(). It contains:
    - All registered shapes (atoms + composites)
    - Execution engine (via backend)
    - Validation engine
    - Export pipelines (CUDA, Verilog)
    """

    shapes: Dict[str, ShapeTerms]
    bits: int
    truth_fns: Dict[str, Callable] = field(default_factory=dict)
    _backend: Optional[Backend] = None
    _signatures: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize backend and derive signatures."""
        self._backend = get_backend()
        self._signatures = self._derive_signatures()

    def _derive_signatures(self) -> Dict[str, Any]:
        """
        Derive ternary signatures from shape terms.

        Signatures are computed from term structure, not sampled from behavior.
        This is the key insight: addresses come from the geometry.
        """
        from .signature import signature_from_terms

        signatures = {}
        for name, shape in self.shapes.items():
            signatures[name] = signature_from_terms(shape)
        return signatures

    def execute(self, a: int, b: int, op: str) -> int:
        """
        Execute operation on inputs.

        Args:
            a: First operand
            b: Second operand
            op: Operation name (shape name)

        Returns:
            Result of operation
        """
        # Find matching shape (case-insensitive)
        op_lower = op.lower()
        matching_op = None
        for shape_name in self.shapes:
            if shape_name.lower() == op_lower:
                matching_op = shape_name
                break

        if matching_op is None:
            raise ValueError(f"Unknown operation: {op}. Available: {list(self.shapes.keys())}")

        # Fast path for simple shapes
        if is_fast_path(op_lower):
            return compute_fast(op_lower, a, b, self.bits)

        # Polynomial evaluation
        shape = self.shapes[op_lower]
        results = evaluate_batch(shape, [a], [b])
        return results[0]

    def execute_batch(self, a_batch: List[int], b_batch: List[int], op: str) -> List[int]:
        """
        Execute operation on batched inputs.

        Args:
            a_batch: List of first operands
            b_batch: List of second operands
            op: Operation name

        Returns:
            List of results
        """
        op_lower = op.lower()

        if op_lower not in self.shapes:
            raise ValueError(f"Unknown operation: {op}. Available: {list(self.shapes.keys())}")

        # Fast path for simple shapes
        if is_fast_path(op_lower):
            mask = (1 << self.bits) - 1
            return [compute_fast(op_lower, a, b, self.bits) for a, b in zip(a_batch, b_batch)]

        # Polynomial evaluation
        shape = self.shapes[op_lower]
        return evaluate_batch(shape, a_batch, b_batch)

    def validate(self, exhaustive: bool = True, mode: str = None) -> SystemValidation:
        """
        Validate all shapes against their truth functions.

        Args:
            exhaustive: If True and bits <= 8, test all 2^16 combinations
            mode: Override validation mode ("quick", "edge", "exhaustive")

        Returns:
            SystemValidation with per-shape results
        """
        if mode is None:
            if exhaustive and self.bits <= 8:
                mode = "exhaustive"
            else:
                mode = "quick"

        results = {}

        for name, shape in self.shapes.items():
            # Get truth function
            truth_fn = self.truth_fns.get(name)

            if truth_fn is None:
                # Try to infer from shape name
                truth_fn = self._get_default_truth_fn(name)

            if truth_fn is None:
                # Skip validation for shapes without truth functions
                continue

            if mode == "exhaustive":
                # Need 8-bit version for exhaustive
                if shape.output_bits != 8:
                    from .term import generate_shape
                    shape_8bit = generate_shape(name, bits=8)
                    if shape_8bit is not None:
                        result = validate_exhaustive_8bit(shape_8bit, truth_fn)
                    else:
                        result = validate_quick(shape, truth_fn, bits=self.bits)
                else:
                    result = validate_exhaustive_8bit(shape, truth_fn)
            elif mode == "edge":
                result = validate_edge(shape, truth_fn, bits=self.bits)
            else:
                result = validate_quick(shape, truth_fn, bits=self.bits)

            results[name] = result

        return SystemValidation(results=results, bits=self.bits, mode=mode)

    def _get_default_truth_fn(self, name: str) -> Optional[Callable]:
        """Get default truth function for known operations."""
        mask = (1 << self.bits) - 1
        defaults = {
            "xor": lambda a, b: a ^ b,
            "and": lambda a, b: a & b,
            "or": lambda a, b: a | b,
            "not": lambda a, b: (~a) & mask,
            "nand": lambda a, b: (~(a & b)) & mask,
            "nor": lambda a, b: (~(a | b)) & mask,
            "xnor": lambda a, b: (~(a ^ b)) & mask,
            "nop": lambda a, b: a,
            "add": lambda a, b: (a + b) & mask,
            "sub": lambda a, b: (a - b) & mask,
        }
        return defaults.get(name.lower())

    def export_cuda(self, path: str):
        """
        Export system to CUDA.

        Creates:
            - CUDA kernel file for each shape
            - Header with declarations
            - Makefile for compilation
            - Test program for validation
        """
        from .cuda import export_cuda
        export_cuda(self.shapes, path)

    def export_verilog(self, path: str):
        """
        Export system to Verilog.

        Creates:
            - Verilog module for each shape
            - Top-level wrapper with shape selection
            - Testbench for simulation
        """
        from .verilog import export_verilog
        export_verilog(self.shapes, path)

    def get_signature(self, op: str) -> Any:
        """Get ternary signature for operation."""
        return self._signatures.get(op.lower())

    def list_operations(self) -> List[str]:
        """List available operations."""
        return list(self.shapes.keys())

    def summary(self) -> str:
        """Human-readable system summary."""
        lines = [
            f"System: {len(self.shapes)} shapes, {self.bits}-bit",
            f"Shapes: {list(self.shapes.keys())}",
            f"Backend: {self._backend.info().name if self._backend else 'None'}",
        ]

        total_terms = sum(s.total_terms() for s in self.shapes.values())
        lines.append(f"Total terms: {total_terms}")

        return "\n".join(lines)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "System",
    "SystemValidation",
]
