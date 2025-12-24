"""
TriX Forge - Verilog to Frozen C Pipeline

Stage 1: Combinational logic only.

Usage:
    from foundry.forge import freeze, freeze_factored, freeze_compact, certify

    verilog = '''
    module adder4(
        input [3:0] a,
        input [3:0] b,
        output [4:0] sum
    );
        assign sum = a + b;
    endmodule
    '''

    # For small circuits (up to ~16 bits) - fully expanded
    header, source = freeze(verilog)

    # For large circuits (32-bit, 64-bit, any size) - O(n) code
    header, source = freeze_factored(verilog)

    # For any size - O(1) code, loop-based (~1KB for any width)
    header, source = freeze_compact(verilog)

    # Get formal proof of correctness (no exhaustive testing needed)
    from foundry.forge.certifier import certify, quick_certify
"""

from .pipeline import freeze, freeze_factored, freeze_compact, ForgeError
from .certifier import certify, quick_certify, Certificate

__all__ = [
    'freeze', 'freeze_factored', 'freeze_compact', 'ForgeError',
    'certify', 'quick_certify', 'Certificate'
]
