"""
Hardware estimation for XORPU shapes.

Provides LUT, FF, cycle, and power estimates for FPGA synthesis planning.

Usage:
    from trix.forge.hardware import estimate_shape, estimate_xorpu

    # Single shape estimate
    xor_shape = generate_xor_shape(bits=32)
    est = estimate_shape(xor_shape)
    print(f"XOR: {est.luts} LUTs, {est.cycles} cycles")

    # Full XORPU estimate
    shapes = generate_all_shapes(bits=32)
    total = estimate_xorpu(shapes)
    print(f"Total: {total.luts} LUTs")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math

from .term import ShapeTerms, BitTerms, Term


# =============================================================================
# HARDWARE ESTIMATE DATACLASS
# =============================================================================

@dataclass
class HardwareEstimate:
    """
    Hardware resource and timing estimate for a shape or XORPU.

    Attributes:
        luts: Number of lookup tables (6-LUT architecture assumed)
        ffs: Number of flip-flops (0 for combinational)
        bram_kb: Block RAM in KB (0 for pure logic)
        cycles: Execution cycles (1 for combinational)
        power_mw: Estimated power in milliwatts
    """
    luts: int
    ffs: int
    bram_kb: float
    cycles: int
    power_mw: float

    def __add__(self, other: "HardwareEstimate") -> "HardwareEstimate":
        """Sum two estimates (for combining shapes)."""
        return HardwareEstimate(
            luts=self.luts + other.luts,
            ffs=self.ffs + other.ffs,
            bram_kb=self.bram_kb + other.bram_kb,
            cycles=max(self.cycles, other.cycles),  # Parallel execution
            power_mw=self.power_mw + other.power_mw,
        )

    def __repr__(self) -> str:
        return (
            f"HardwareEstimate(luts={self.luts}, ffs={self.ffs}, "
            f"bram_kb={self.bram_kb:.2f}, cycles={self.cycles}, "
            f"power_mw={self.power_mw:.2f})"
        )

    def summary(self) -> str:
        """Human-readable summary."""
        return f"{self.luts} LUTs, {self.ffs} FFs, {self.cycles} cyc, {self.power_mw:.1f} mW"


# =============================================================================
# LUT ESTIMATION
# =============================================================================

# Simple shapes that map directly to primitives
SIMPLE_SHAPES = {"xor", "and", "or", "not", "nand", "nor", "xnor", "nop"}


def _is_simple_shape(name: str) -> bool:
    """Check if shape is a simple primitive."""
    return name.lower() in SIMPLE_SHAPES


def _estimate_term_luts(term: Term) -> int:
    """
    Estimate LUTs for a single term.

    Model (6-LUT architecture):
    - Constant (degree 0): 0 LUTs (hardwired)
    - Single variable (degree 1): 0 LUTs (wire)
    - Product of 2-6 vars: 1 LUT
    - Product of 7-12 vars: 2 LUTs
    - etc.
    """
    degree = term.degree()

    if degree == 0:
        return 0  # Constant
    elif degree == 1:
        return 0  # Just a wire
    else:
        # AND tree: ceil(degree / 6) LUTs for 6-input LUT
        return max(1, math.ceil(degree / 6))


def _estimate_xor_reduction_luts(num_terms: int) -> int:
    """
    Estimate LUTs for XOR reduction tree.

    Model:
    - 2-input XOR: 1 LUT
    - Tree depth: ceil(log2(num_terms))
    - Each level needs ceil(inputs/2) LUTs
    """
    if num_terms <= 1:
        return 0
    elif num_terms <= 2:
        return 1
    else:
        # XOR tree: approximately num_terms - 1 LUTs
        # (each XOR reduces 2 inputs to 1)
        return num_terms - 1


def _estimate_bit_luts(bit_terms: BitTerms) -> int:
    """Estimate LUTs for a single output bit."""
    # LUTs for computing each term (AND products)
    term_luts = sum(_estimate_term_luts(t) for t in bit_terms.terms)

    # LUTs for XOR reduction
    xor_luts = _estimate_xor_reduction_luts(len(bit_terms.terms))

    return term_luts + xor_luts


def estimate_luts(shape: ShapeTerms) -> int:
    """
    Estimate total LUTs for a shape.

    For simple shapes (XOR, AND, etc.), uses 1 LUT per bit.
    For complex shapes, sums term and reduction LUTs.
    """
    if _is_simple_shape(shape.name):
        # Simple primitives: 1 LUT per output bit
        return shape.output_bits
    else:
        # Complex polynomial: sum bit estimates
        return sum(_estimate_bit_luts(bt) for bt in shape.bit_terms)


# =============================================================================
# FF ESTIMATION
# =============================================================================

def estimate_ffs(shape: ShapeTerms, pipelined: bool = False, stages: int = 0) -> int:
    """
    Estimate flip-flops for a shape.

    Args:
        shape: The shape specification
        pipelined: If True, add pipeline registers
        stages: Number of pipeline stages

    For combinational logic (default), FFs = 0.
    For pipelined, FFs = output_bits * stages.
    """
    if pipelined and stages > 0:
        return shape.output_bits * stages
    return 0


# =============================================================================
# CYCLE ESTIMATION
# =============================================================================

def estimate_cycles(shape: ShapeTerms) -> int:
    """
    Estimate execution cycles for a shape.

    Combinational logic executes in 1 cycle.
    Complex shapes might need more cycles if deeply pipelined.
    """
    # All shapes are combinational: 1 cycle
    return 1


# =============================================================================
# POWER ESTIMATION
# =============================================================================

# Power model constants
POWER_PER_LUT_MW = 0.1  # Rough estimate: 0.1 mW per LUT
STATIC_POWER_MW = 10.0   # Base static power


def estimate_power(shape: ShapeTerms) -> float:
    """
    Estimate power consumption in milliwatts.

    Rough model:
    - Dynamic power proportional to LUT count
    - Uses activity factor of 0.5 (50% switching)
    """
    luts = estimate_luts(shape)
    dynamic_power = luts * POWER_PER_LUT_MW
    return dynamic_power


# =============================================================================
# MAIN ESTIMATION FUNCTIONS
# =============================================================================

def estimate_shape(
    shape: ShapeTerms,
    pipelined: bool = False,
    pipeline_stages: int = 0,
) -> HardwareEstimate:
    """
    Estimate hardware resources for a single shape.

    Args:
        shape: The shape specification
        pipelined: If True, add pipeline registers
        pipeline_stages: Number of pipeline stages

    Returns:
        HardwareEstimate with LUT, FF, cycle, and power estimates
    """
    return HardwareEstimate(
        luts=estimate_luts(shape),
        ffs=estimate_ffs(shape, pipelined, pipeline_stages),
        bram_kb=0.0,  # Pure logic, no BRAM
        cycles=estimate_cycles(shape),
        power_mw=estimate_power(shape),
    )


def estimate_xorpu(
    shapes: Dict[str, ShapeTerms],
    shared_inputs: bool = True,
) -> HardwareEstimate:
    """
    Estimate hardware resources for complete XORPU.

    Args:
        shapes: Dictionary of shape name -> ShapeTerms
        shared_inputs: If True, assume inputs are shared (reduces routing)

    Returns:
        HardwareEstimate for the entire XORPU
    """
    total = HardwareEstimate(
        luts=0,
        ffs=0,
        bram_kb=0.0,
        cycles=1,
        power_mw=0.0,
    )

    for name, shape in shapes.items():
        est = estimate_shape(shape)
        total = total + est

    # Add MUX overhead for shape selection
    if len(shapes) > 1:
        # Assume 4-bit shape selector, outputs = bits per shape
        bits = next(iter(shapes.values())).output_bits if shapes else 32
        mux_luts = bits * len(shapes) // 4  # Rough MUX estimate
        total.luts += mux_luts
        total.power_mw += mux_luts * POWER_PER_LUT_MW

    return total


def estimate_summary(shapes: Dict[str, ShapeTerms]) -> str:
    """
    Generate summary table of hardware estimates.

    Returns formatted string with per-shape and total estimates.
    """
    lines = []
    lines.append("XORPU Hardware Estimates")
    lines.append("=" * 60)
    lines.append(f"{'Shape':<15} {'LUTs':>8} {'FFs':>6} {'Cycles':>8} {'Power':>10}")
    lines.append("-" * 60)

    total = HardwareEstimate(luts=0, ffs=0, bram_kb=0, cycles=1, power_mw=0)

    for name in sorted(shapes.keys()):
        shape = shapes[name]
        est = estimate_shape(shape)
        total = total + est
        lines.append(
            f"{name:<15} {est.luts:>8} {est.ffs:>6} {est.cycles:>8} {est.power_mw:>9.1f}mW"
        )

    lines.append("-" * 60)
    lines.append(
        f"{'TOTAL':<15} {total.luts:>8} {total.ffs:>6} {total.cycles:>8} {total.power_mw:>9.1f}mW"
    )

    return "\n".join(lines)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "HardwareEstimate",
    "estimate_shape",
    "estimate_xorpu",
    "estimate_luts",
    "estimate_ffs",
    "estimate_cycles",
    "estimate_power",
    "estimate_summary",
]
