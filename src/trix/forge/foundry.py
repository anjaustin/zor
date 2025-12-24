"""
Unified Neural-Geometric Deterministic Systems Foundry.

The Foundry is a compiler: Spec -> IR (ShapeTerms) -> Target (CUDA/Verilog).

Usage:
    from trix.forge import Foundry

    # Create foundry
    foundry = Foundry(bits=8)

    # Register atomic shapes from truth functions
    foundry.atom("xor", lambda a, b: a ^ b)
    foundry.atom("and", lambda a, b: a & b)
    foundry.atom("or",  lambda a, b: a | b)
    foundry.atom("add", lambda a, b: (a + b) & 0xFF)

    # Build executable system
    system = foundry.build()

    # Validate (100% or fail)
    result = system.validate(exhaustive=True)
    assert result.all_passed()

    # Execute
    print(system.execute(42, 13, "xor"))  # 39

    # Export
    system.export_cuda("output/alu_cuda/")
    system.export_verilog("output/alu_verilog/")

The Three Pillars:
    1. Polynomial computation - ShapeTerms is the IR
    2. Ternary signatures - Derived from terms, not sampled
    3. Routing-only learning - Shapes are frozen geometry
"""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Union, Any
import inspect

from .term import (
    Term, BitTerms, ShapeTerms,
    generate_shape, generate_all_shapes, SHAPE_GENERATORS,
    _int_to_bits, _bits_to_int,
)

import torch


# =============================================================================
# TRUTH TABLE GENERATION
# =============================================================================

def generate_from_truth(fn: Callable, bits: int) -> ShapeTerms:
    """
    Generate ShapeTerms from truth function.

    Determines arity from function signature and generates appropriate shape.

    Args:
        fn: Truth function (unary, binary, or ternary)
        bits: Bit width

    Returns:
        ShapeTerms representing the function
    """
    # Determine arity
    sig = inspect.signature(fn)
    arity = len(sig.parameters)

    if arity == 1:
        return _generate_unary_shape(fn, bits)
    elif arity == 2:
        return _generate_binary_shape(fn, bits)
    elif arity == 3:
        return _generate_ternary_shape(fn, bits)
    else:
        raise ValueError(f"Unsupported arity: {arity}")


def _generate_unary_shape(fn: Callable, bits: int) -> ShapeTerms:
    """Generate shape from unary truth function."""
    # Use Lagrange interpolation on each output bit
    bit_terms = []

    for out_bit in range(bits):
        bt = BitTerms(out_bit)

        # For each output bit, find polynomial terms
        # Sample all possible input bit patterns and output bit values
        # Then derive minimal polynomial representation

        # Simplified approach: enumerate minterm contributions
        for a_val in range(1 << bits):
            result = fn(a_val) & ((1 << bits) - 1)
            result_bit = (result >> out_bit) & 1

            if result_bit:
                # This input produces a 1 in this output bit
                # Add minterm (product of input bits)
                vars_present = []
                for i in range(bits):
                    if (a_val >> i) & 1:
                        vars_present.append(i)

                # For unary, each minterm is a product of input bits
                # This creates the canonical sum-of-products form
                if vars_present:
                    bt.add_term(Term(1, tuple(vars_present)))
                else:
                    bt.add_term(Term.constant(1))

        bit_terms.append(bt)

    return ShapeTerms(
        name=fn.__name__ if hasattr(fn, '__name__') else "unary",
        input_bits=bits,
        output_bits=bits,
        bit_terms=bit_terms,
    )


def _generate_binary_shape(fn: Callable, bits: int) -> ShapeTerms:
    """
    Generate shape from binary truth function.

    Uses algebraic decomposition: for each output bit, determine
    the polynomial in terms of input bits.
    """
    from .term import xor_terms, and_terms, or_terms, not_terms

    # Check if this is a known shape
    # Test characteristic values
    test_vals = [
        (0, 0, fn(0, 0)),
        (0xFF, 0, fn(0xFF, 0) if bits >= 8 else fn((1 << bits) - 1, 0)),
        (0, 0xFF, fn(0, 0xFF) if bits >= 8 else fn(0, (1 << bits) - 1)),
        (0xFF, 0xFF, fn(0xFF, 0xFF) if bits >= 8 else fn((1 << bits) - 1, (1 << bits) - 1)),
    ]

    mask = (1 << bits) - 1

    # Pattern matching for common operations
    if _matches_xor(fn, bits):
        return _make_bitwise_shape("xor", xor_terms, bits)
    elif _matches_and(fn, bits):
        return _make_bitwise_shape("and", and_terms, bits)
    elif _matches_or(fn, bits):
        return _make_bitwise_shape("or", or_terms, bits)
    elif _matches_add(fn, bits):
        # ADD is structural - use existing generator
        shape = generate_shape("add", bits)
        return shape
    elif _matches_sub(fn, bits):
        shape = generate_shape("sub", bits)
        return shape
    else:
        # Fall back to truth table enumeration (expensive for large bits)
        return _generate_binary_from_enumeration(fn, bits)


def _matches_xor(fn: Callable, bits: int) -> bool:
    """Check if function is XOR."""
    mask = (1 << bits) - 1
    test_cases = [(0, 0), (mask, 0), (0, mask), (mask, mask), (0xAA & mask, 0x55 & mask)]
    for a, b in test_cases:
        if fn(a, b) != (a ^ b):
            return False
    return True


def _matches_and(fn: Callable, bits: int) -> bool:
    """Check if function is AND."""
    mask = (1 << bits) - 1
    test_cases = [(0, 0), (mask, 0), (0, mask), (mask, mask), (0xAA & mask, 0x55 & mask)]
    for a, b in test_cases:
        if fn(a, b) != (a & b):
            return False
    return True


def _matches_or(fn: Callable, bits: int) -> bool:
    """Check if function is OR."""
    mask = (1 << bits) - 1
    test_cases = [(0, 0), (mask, 0), (0, mask), (mask, mask), (0xAA & mask, 0x55 & mask)]
    for a, b in test_cases:
        if fn(a, b) != (a | b):
            return False
    return True


def _matches_add(fn: Callable, bits: int) -> bool:
    """Check if function is ADD."""
    mask = (1 << bits) - 1
    test_cases = [(0, 0), (1, 1), (mask, 1), (0x55 & mask, 0x55 & mask)]
    for a, b in test_cases:
        if fn(a, b) != ((a + b) & mask):
            return False
    return True


def _matches_sub(fn: Callable, bits: int) -> bool:
    """Check if function is SUB."""
    mask = (1 << bits) - 1
    test_cases = [(0, 0), (1, 1), (mask, 1), (0x55 & mask, 0x22 & mask)]
    for a, b in test_cases:
        if fn(a, b) != ((a - b) & mask):
            return False
    return True


def _make_bitwise_shape(name: str, term_fn: Callable, bits: int) -> ShapeTerms:
    """Create bitwise shape from term generator."""
    bit_terms = []
    for i in range(bits):
        a_idx, b_idx = i, i + bits
        bt = BitTerms(i)
        bt.add_terms(term_fn(a_idx, b_idx))
        bit_terms.append(bt)

    return ShapeTerms(name, bits * 2, bits, bit_terms)


def _generate_binary_from_enumeration(fn: Callable, bits: int) -> ShapeTerms:
    """
    Generate shape by enumerating truth table.

    Warning: O(2^(2*bits)) - only practical for small bit widths.
    """
    if bits > 8:
        raise ValueError(f"Enumeration too expensive for {bits} bits. Use pattern matching.")

    mask = (1 << bits) - 1
    bit_terms = []

    for out_bit in range(bits):
        bt = BitTerms(out_bit)

        # Build truth table for this output bit
        for a_val in range(1 << bits):
            for b_val in range(1 << bits):
                result = fn(a_val, b_val) & mask
                result_bit = (result >> out_bit) & 1

                if result_bit:
                    # This combination produces a 1
                    # Create minterm
                    vars_present = []
                    for i in range(bits):
                        if (a_val >> i) & 1:
                            vars_present.append(i)
                        if (b_val >> i) & 1:
                            vars_present.append(i + bits)

                    if vars_present:
                        bt.add_term(Term(1, tuple(sorted(vars_present))))
                    else:
                        bt.add_term(Term.constant(1))

        bit_terms.append(bt)

    return ShapeTerms(
        name=fn.__name__ if hasattr(fn, '__name__') else "binary",
        input_bits=bits * 2,
        output_bits=bits,
        bit_terms=bit_terms,
    )


def _generate_ternary_shape(fn: Callable, bits: int) -> ShapeTerms:
    """
    Generate shape from ternary truth function (e.g., full adder).

    Returns a shape with 2 outputs per bit: (sum, carry).
    """
    if bits > 4:
        raise ValueError(f"Ternary enumeration too expensive for {bits} bits.")

    mask = (1 << bits) - 1

    # Ternary functions like full adder have 3 inputs and 2 outputs per bit
    # For now, treat as single-bit with 3 inputs

    bit_terms = []
    for out_bit in range(2):  # sum and carry
        bt = BitTerms(out_bit)

        for a in range(2):
            for b in range(2):
                for c in range(2):
                    result = fn(a, b, c)
                    if isinstance(result, tuple):
                        result_bit = result[out_bit]
                    else:
                        result_bit = (result >> out_bit) & 1

                    if result_bit:
                        vars_present = []
                        if a: vars_present.append(0)
                        if b: vars_present.append(1)
                        if c: vars_present.append(2)

                        if vars_present:
                            bt.add_term(Term(1, tuple(vars_present)))
                        else:
                            bt.add_term(Term.constant(1))

        bit_terms.append(bt)

    return ShapeTerms(
        name=fn.__name__ if hasattr(fn, '__name__') else "ternary",
        input_bits=3,
        output_bits=2,
        bit_terms=bit_terms,
    )


# =============================================================================
# FOUNDRY CLASS
# =============================================================================

@dataclass
class Foundry:
    """
    Unified Neural-Geometric Deterministic Systems Foundry.

    The Foundry is a compiler:
        - Frontend: Truth tables, composition operators
        - Middle (IR): ShapeTerms (polynomial representation)
        - Backend: CUDA, Verilog, validation

    Attributes:
        bits: Default bit width for shapes
        atoms: Registered atomic shapes (truth table -> ShapeTerms)
        composites: Registered composite shapes (from composition operators)
    """

    bits: int = 8
    atoms: Dict[str, ShapeTerms] = field(default_factory=dict)
    composites: Dict[str, ShapeTerms] = field(default_factory=dict)
    _truth_fns: Dict[str, Callable] = field(default_factory=dict)

    def atom(self, name: str, truth_fn: Callable) -> "Foundry":
        """
        Register an atomic shape from truth function.

        Args:
            name: Shape name (e.g., "xor", "add")
            truth_fn: Truth function (a, b) -> result

        Returns:
            self for chaining

        Example:
            foundry.atom("xor", lambda a, b: a ^ b)
            foundry.atom("add", lambda a, b: (a + b) & 0xFF)
        """
        # Try to use existing generator for known shapes
        name_lower = name.lower()
        if name_lower in SHAPE_GENERATORS:
            shape = generate_shape(name_lower, self.bits)
        else:
            shape = generate_from_truth(truth_fn, self.bits)
            shape.name = name

        self.atoms[name] = shape
        self._truth_fns[name] = truth_fn
        return self

    def compose(self, name: str, composition: "Composition") -> "Foundry":
        """
        Register a composite shape from composition operators.

        Args:
            name: Composite shape name
            composition: Composition expression (seq, par, sel, rep)

        Returns:
            self for chaining

        Example:
            foundry.compose("ripple_add", rep("full_adder", 8))
            foundry.compose("alu", sel("xor", "and", "or", "add"))
        """
        from .composition import Composition

        if isinstance(composition, Composition):
            # Evaluate composition against atom registry
            shape = composition.evaluate(self.atoms)
            self.composites[name] = shape
        else:
            raise TypeError(f"Expected Composition, got {type(composition)}")

        return self

    def build(self) -> "System":
        """
        Build executable system from registered shapes.

        Returns:
            System ready for execution, validation, and export
        """
        from .system import System

        all_shapes = {**self.atoms, **self.composites}
        return System(
            shapes=all_shapes,
            bits=self.bits,
            truth_fns=self._truth_fns,
        )

    def list_atoms(self) -> List[str]:
        """List registered atomic shapes."""
        return list(self.atoms.keys())

    def list_composites(self) -> List[str]:
        """List registered composite shapes."""
        return list(self.composites.keys())

    def get_shape(self, name: str) -> Optional[ShapeTerms]:
        """Get shape by name (atom or composite)."""
        if name in self.atoms:
            return self.atoms[name]
        if name in self.composites:
            return self.composites[name]
        return None

    def summary(self) -> str:
        """Human-readable summary of foundry state."""
        lines = [
            f"Foundry ({self.bits}-bit)",
            f"  Atoms: {len(self.atoms)}",
        ]
        for name, shape in self.atoms.items():
            lines.append(f"    - {name}: {shape.total_terms()} terms")

        if self.composites:
            lines.append(f"  Composites: {len(self.composites)}")
            for name, shape in self.composites.items():
                lines.append(f"    - {name}: {shape.total_terms()} terms")

        total_terms = sum(s.total_terms() for s in self.atoms.values())
        total_terms += sum(s.total_terms() for s in self.composites.values())
        lines.append(f"  Total terms: {total_terms}")

        return "\n".join(lines)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Foundry",
    "generate_from_truth",
]
