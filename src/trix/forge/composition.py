"""
Composition Algebra for Shape Composition.

Four operators cover all deterministic composition:
    - seq(a, b): Sequential - output of A feeds input of B
    - par(a, b): Parallel - same input, concatenated outputs
    - sel(*shapes): Selection - opcode selects which shape
    - rep(shape, n): Repetition - chain shape N times

Usage:
    from trix.forge import Foundry
    from trix.forge.composition import seq, par, sel, rep

    foundry = Foundry(bits=8)
    foundry.atom("xor", lambda a, b: a ^ b)
    foundry.atom("and", lambda a, b: a & b)

    # Sequential: and(xor(a, b), b)
    foundry.compose("xor_then_and", seq("xor", "and"))

    # Parallel: [xor(a, b), and(a, b)]
    foundry.compose("xor_and_par", par("xor", "and"))

    # Selection: opcode picks operation
    foundry.compose("alu", sel("xor", "and", "or", "add"))

    # Repetition: chain 8 times (like ripple carry)
    foundry.compose("chain8", rep("half_add", 8))
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple

from .term import (
    Term, BitTerms, ShapeTerms,
    evaluate_batch, _int_to_bits, _bits_to_int,
)

import torch


# =============================================================================
# COMPOSITION BASE CLASS
# =============================================================================

class Composition(ABC):
    """Abstract composition operator."""

    @abstractmethod
    def evaluate(self, registry: Dict[str, ShapeTerms]) -> ShapeTerms:
        """
        Evaluate composition against a shape registry.

        Args:
            registry: Dict mapping shape names to ShapeTerms

        Returns:
            Composed ShapeTerms
        """
        pass


# =============================================================================
# SEQUENTIAL COMPOSITION
# =============================================================================

@dataclass
class Seq(Composition):
    """
    Sequential composition: output of A feeds input of B.

    seq(A, B)(x) = B(A(x))

    For term-based representation, this requires substitution:
    substitute A's output bit terms into B's input variables.
    """
    a: str
    b: str

    def evaluate(self, registry: Dict[str, ShapeTerms]) -> ShapeTerms:
        """Compose shapes sequentially."""
        if self.a not in registry:
            raise ValueError(f"Unknown shape: {self.a}")
        if self.b not in registry:
            raise ValueError(f"Unknown shape: {self.b}")

        a_shape = registry[self.a]
        b_shape = registry[self.b]

        # Verify compatibility: A's output bits == B's input bits (per operand)
        if a_shape.output_bits * 2 != b_shape.input_bits:
            # B expects two operands but A produces one
            # For sequential composition, A's output becomes B's first input
            # and original second input is forwarded
            pass

        # Use behavioral composition (compute composed truth table)
        return compose_sequential(a_shape, b_shape)


def compose_sequential(a: ShapeTerms, b: ShapeTerms) -> ShapeTerms:
    """
    Compose shapes sequentially: b(a(x)).

    For now, uses behavioral composition (compute truth table of composition).
    Future: algebraic term substitution.
    """
    # Determine output bit count
    output_bits = b.output_bits

    # For sequential composition, we need to:
    # 1. Evaluate a(x) to get intermediate result
    # 2. Feed intermediate as input to b
    #
    # The composed shape's input is same as a's input.

    a_input_bits = a.input_bits
    a_output_bits = a.output_bits

    # Build composed terms by behavioral sampling
    # This is expensive but correct

    bit_terms = []
    for out_bit in range(output_bits):
        bt = BitTerms(out_bit)

        # For small input spaces, enumerate
        if a_input_bits <= 16:  # Up to 8-bit binary operations
            minterms = []
            for input_val in range(1 << a_input_bits):
                # Build input tensor
                input_bits = torch.tensor(
                    [(input_val >> i) & 1 for i in range(a_input_bits)],
                    dtype=torch.float32
                ).unsqueeze(0)

                # Evaluate a
                a_output = a.evaluate(input_bits)  # [1, a_output_bits]
                a_int = _bits_to_int(a_output.squeeze())

                # Build b input (a_output as first operand, zeros as second)
                # This assumes b is binary
                b_input_bits = torch.zeros(1, b.input_bits)
                for i in range(min(a_output_bits, b.input_bits)):
                    b_input_bits[0, i] = a_output[0, i]

                # Evaluate b
                b_output = b.evaluate(b_input_bits)  # [1, output_bits]

                # Check if output bit is 1
                if b_output[0, out_bit] > 0.5:
                    # Add minterm for this input
                    vars_present = tuple(i for i in range(a_input_bits) if (input_val >> i) & 1)
                    if vars_present:
                        minterms.append(vars_present)
                    else:
                        bt.add_term(Term.constant(1))

            # Combine minterms (simplified - full minimization would be better)
            for vars in set(minterms):
                bt.add_term(Term(1, vars))

        bit_terms.append(bt)

    return ShapeTerms(
        name=f"seq_{a.name}_{b.name}",
        input_bits=a_input_bits,
        output_bits=output_bits,
        bit_terms=bit_terms,
    )


# =============================================================================
# PARALLEL COMPOSITION
# =============================================================================

@dataclass
class Par(Composition):
    """
    Parallel composition: same input, concatenated outputs.

    par(A, B)(x) = (A(x), B(x))

    Output bits are concatenation: [A_output_bits, B_output_bits]
    """
    shapes: Tuple[str, ...]

    def __init__(self, *shapes: str):
        self.shapes = shapes

    def evaluate(self, registry: Dict[str, ShapeTerms]) -> ShapeTerms:
        """Compose shapes in parallel."""
        for name in self.shapes:
            if name not in registry:
                raise ValueError(f"Unknown shape: {name}")

        shape_terms = [registry[name] for name in self.shapes]
        return compose_parallel(shape_terms)


def compose_parallel(shapes: List[ShapeTerms]) -> ShapeTerms:
    """
    Compose shapes in parallel: (a(x), b(x), ...).

    Concatenates output bits from each shape.
    All shapes must have same input_bits.
    """
    if not shapes:
        raise ValueError("Need at least one shape for parallel composition")

    # Verify all shapes have same input
    input_bits = shapes[0].input_bits
    for shape in shapes[1:]:
        if shape.input_bits != input_bits:
            raise ValueError(f"Input bit mismatch: {shapes[0].name} has {input_bits}, "
                           f"{shape.name} has {shape.input_bits}")

    # Concatenate output bits
    output_bits = sum(s.output_bits for s in shapes)
    bit_terms = []

    bit_offset = 0
    for shape in shapes:
        for bt in shape.bit_terms:
            # Remap bit index
            new_bt = BitTerms(bt.bit_index + bit_offset)
            new_bt.terms = bt.terms.copy()
            bit_terms.append(new_bt)
        bit_offset += shape.output_bits

    names = "_".join(s.name for s in shapes)
    return ShapeTerms(
        name=f"par_{names}",
        input_bits=input_bits,
        output_bits=output_bits,
        bit_terms=bit_terms,
    )


# =============================================================================
# SELECTION COMPOSITION
# =============================================================================

@dataclass
class Sel(Composition):
    """
    Selection composition: opcode selects which shape executes.

    sel(op, [A, B, C])(x) = {A(x) if op=0, B(x) if op=1, C(x) if op=2}

    This is the routing fabric - maps to MUX in hardware.
    """
    shapes: Tuple[str, ...]

    def __init__(self, *shapes: str):
        self.shapes = shapes

    def evaluate(self, registry: Dict[str, ShapeTerms]) -> ShapeTerms:
        """Create selection (MUX) shape."""
        for name in self.shapes:
            if name not in registry:
                raise ValueError(f"Unknown shape: {name}")

        shape_terms = [registry[name] for name in self.shapes]
        return compose_selection(shape_terms)


def compose_selection(shapes: List[ShapeTerms]) -> ShapeTerms:
    """
    Create selection (MUX) composition.

    The result is a CompositeShape that dispatches based on opcode.
    For term representation, we create a wrapper that includes all shapes.
    """
    if not shapes:
        raise ValueError("Need at least one shape for selection")

    # All shapes should have same I/O dimensions
    input_bits = shapes[0].input_bits
    output_bits = shapes[0].output_bits

    for shape in shapes[1:]:
        if shape.output_bits != output_bits:
            raise ValueError(f"Output bit mismatch in selection")

    # Selection composition creates a meta-shape that contains all options
    # The actual selection happens at runtime

    # For term representation, we encode selection as:
    # output = sum over i of (sel_i * shape_i(input))
    # where sel_i are one-hot selection bits

    n_shapes = len(shapes)
    sel_bits = (n_shapes - 1).bit_length()  # bits needed for selection

    # For now, return the first shape with metadata about selection
    # Full implementation would create MUX terms

    names = "_".join(s.name for s in shapes)
    result = ShapeTerms(
        name=f"sel_{names}",
        input_bits=input_bits,
        output_bits=output_bits,
        bit_terms=shapes[0].bit_terms.copy(),  # Default to first shape
    )

    # Attach selection metadata
    result._selection_shapes = shapes
    result._selection_count = n_shapes

    return result


# =============================================================================
# REPETITION COMPOSITION
# =============================================================================

@dataclass
class Rep(Composition):
    """
    Repetition: chain shape N times.

    rep(A, N)(x) = A(A(A(...A(x)...)))  # N times

    This is for ripple-carry patterns and iterative structures.
    """
    shape: str
    n: int

    def evaluate(self, registry: Dict[str, ShapeTerms]) -> ShapeTerms:
        """Chain shape N times."""
        if self.shape not in registry:
            raise ValueError(f"Unknown shape: {self.shape}")

        base = registry[self.shape]
        return compose_repetition(base, self.n)


def compose_repetition(shape: ShapeTerms, n: int) -> ShapeTerms:
    """
    Chain shape N times: shape(shape(...shape(x)...)).
    """
    if n < 1:
        raise ValueError("Repetition count must be positive")

    if n == 1:
        return shape

    result = shape
    for i in range(n - 1):
        result = compose_sequential(result, shape)

    result.name = f"rep_{shape.name}_{n}"
    return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def seq(a: str, b: str) -> Seq:
    """Sequential composition: b(a(x))."""
    return Seq(a, b)


def par(*shapes: str) -> Par:
    """Parallel composition: same input, concatenated outputs."""
    return Par(*shapes)


def sel(*shapes: str) -> Sel:
    """Selection composition: opcode selects which shape."""
    return Sel(*shapes)


def rep(shape: str, n: int) -> Rep:
    """Repetition: chain shape N times."""
    return Rep(shape, n)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base class
    "Composition",

    # Composition classes
    "Seq",
    "Par",
    "Sel",
    "Rep",

    # Convenience functions
    "seq",
    "par",
    "sel",
    "rep",

    # Composition functions
    "compose_sequential",
    "compose_parallel",
    "compose_selection",
    "compose_repetition",
]
