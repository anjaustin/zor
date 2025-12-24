"""
Main Forge pipeline: Verilog → Frozen C

Three modes:
- freeze(): Expanded polynomials (small circuits, up to ~16 bits)
- freeze_factored(): Factored intermediates (any size, linear growth)
- freeze_compact(): Loop-based O(1) code (any size, smallest output)
"""

from pathlib import Path
from typing import Optional, Tuple

from .parser import parse, ParseError, Module
from .composer import compose, compose_factored
from .emitter import emit_c, emit_c_factored, emit_c_compact


class ForgeError(Exception):
    """Forge pipeline error."""
    pass


def freeze(
    verilog_source: str,
    module_name: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Tuple[str, str]:
    """
    Freeze Verilog combinational logic to C code.

    This is the main entry point for the Forge.

    Args:
        verilog_source: Verilog source code as string
        module_name: Module to freeze (uses first module if None)
        output_dir: Directory to write files (returns strings if None)

    Returns:
        (header_content, source_content)

    Raises:
        ForgeError: If freezing fails

    Example:
        >>> verilog = '''
        ... module adder4(
        ...     input [3:0] a,
        ...     input [3:0] b,
        ...     output [4:0] sum
        ... );
        ...     assign sum = a + b;
        ... endmodule
        ... '''
        >>> header, source = freeze(verilog)
        >>> # header contains frozen_adder4.h
        >>> # source contains frozen_adder4.c
    """
    try:
        # Parse Verilog
        module = parse(verilog_source)

        # Check module name matches if specified
        if module_name and module.name != module_name:
            raise ForgeError(
                f"Module '{module_name}' not found. "
                f"Found: '{module.name}'"
            )

        # Validate: only combinational logic
        _validate_combinational(module)

        # Compose polynomials
        bit_exprs = compose(module)

        if not bit_exprs:
            raise ForgeError("No output expressions generated")

        # Emit C code
        header, source = emit_c(module, bit_exprs)

        # Write files if output_dir specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            header_path = output_dir / f"frozen_{module.name}.h"
            source_path = output_dir / f"frozen_{module.name}.c"

            header_path.write_text(header)
            source_path.write_text(source)

        return header, source

    except ParseError as e:
        raise ForgeError(f"Parse error: {e}") from e
    except Exception as e:
        raise ForgeError(f"Freeze failed: {e}") from e


def _validate_combinational(module: Module) -> None:
    """Validate module is purely combinational."""
    # Check we have at least one input and output
    inputs = [p for p in module.ports.values() if p.direction == "input"]
    outputs = [p for p in module.ports.values() if p.direction == "output"]

    if not inputs:
        raise ForgeError("Module has no inputs")

    if not outputs:
        raise ForgeError("Module has no outputs")

    if not module.assignments:
        raise ForgeError("Module has no assignments (not combinational)")


def freeze_factored(
    verilog_source: str,
    module_name: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Tuple[str, str]:
    """
    Freeze Verilog to C with factored intermediate variables.

    Uses O(n) code size instead of O(2^n) by keeping carries as
    intermediate variables instead of expanding them inline.

    This enables 32-bit, 64-bit, and arbitrary-width adders.

    Args:
        verilog_source: Verilog source code as string
        module_name: Module to freeze (uses first module if None)
        output_dir: Directory to write files (returns strings if None)

    Returns:
        (header_content, source_content)

    Example:
        >>> verilog = '''
        ... module adder64(
        ...     input [63:0] a,
        ...     input [63:0] b,
        ...     output [64:0] sum
        ... );
        ...     assign sum = a + b;
        ... endmodule
        ... '''
        >>> header, source = freeze_factored(verilog)
        >>> # Works! O(n) code size, not O(2^n)
    """
    try:
        # Parse Verilog
        module = parse(verilog_source)

        # Check module name matches if specified
        if module_name and module.name != module_name:
            raise ForgeError(
                f"Module '{module_name}' not found. "
                f"Found: '{module.name}'"
            )

        # Validate: only combinational logic
        _validate_combinational(module)

        # Compose with factored intermediates
        factored = compose_factored(module)

        if not factored.outputs:
            raise ForgeError("No output expressions generated")

        # Emit C code
        header, source = emit_c_factored(module, factored)

        # Write files if output_dir specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            header_path = output_dir / f"frozen_{module.name}.h"
            source_path = output_dir / f"frozen_{module.name}.c"

            header_path.write_text(header)
            source_path.write_text(source)

        return header, source

    except ParseError as e:
        raise ForgeError(f"Parse error: {e}") from e
    except Exception as e:
        raise ForgeError(f"Freeze failed: {e}") from e


def freeze_compact(
    verilog_source: str,
    module_name: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Tuple[str, str]:
    """
    Freeze Verilog to compact loop-based C code.

    O(1) code size regardless of bit width. A 1-million-bit adder
    generates the same ~1KB of code as a 4-bit adder.

    Same polynomial math as expanded/factored modes, but executed
    sequentially in a loop instead of unrolled.

    Tradeoffs:
    - Pro: Tiny code size (~1KB for any width)
    - Pro: No regeneration needed for different widths
    - Con: Sequential execution (not parallelizable)
    - Con: Variable timing (not constant-time, not crypto-safe)

    Args:
        verilog_source: Verilog source code as string
        module_name: Module to freeze (uses first module if None)
        output_dir: Directory to write files (returns strings if None)

    Returns:
        (header_content, source_content)

    Example:
        >>> verilog = '''
        ... module adder1M(
        ...     input [1048575:0] a,
        ...     input [1048575:0] b,
        ...     output [1048576:0] sum
        ... );
        ...     assign sum = a + b;
        ... endmodule
        ... '''
        >>> header, source = freeze_compact(verilog)
        >>> len(source)  # ~1KB, not 500MB!
        1200
    """
    try:
        # Parse Verilog
        module = parse(verilog_source)

        # Check module name matches if specified
        if module_name and module.name != module_name:
            raise ForgeError(
                f"Module '{module_name}' not found. "
                f"Found: '{module.name}'"
            )

        # Validate: only combinational logic
        _validate_combinational(module)

        # Emit compact C code directly from module
        header, source = emit_c_compact(module)

        # Write files if output_dir specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            header_path = output_dir / f"frozen_{module.name}.h"
            source_path = output_dir / f"frozen_{module.name}.c"

            header_path.write_text(header)
            source_path.write_text(source)

        return header, source

    except ParseError as e:
        raise ForgeError(f"Parse error: {e}") from e
    except Exception as e:
        raise ForgeError(f"Freeze failed: {e}") from e
