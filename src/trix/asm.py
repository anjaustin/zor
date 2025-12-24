"""
trix.asm - Minimal 6502 assembler for frozen shapes.

Assemble and run 6502 assembly code using frozen geometry.

Usage:
    >>> from trix.asm import run
    >>> result = run("LDA #$05\\nADC #$03")
    >>> print(result['A'])  # 8

    >>> result = run('''
    ...     LDA #$10    ; Load 16
    ...     ASL A       ; Shift left (multiply by 2)
    ...     ASL A       ; Shift left (multiply by 2)
    ... ''')
    >>> print(result['A'])  # 64 (16 * 4)

Supported instructions (immediate mode only):
    LDA, ADC, SBC, AND, ORA, EOR #$XX
    ASL, LSR, ROL, ROR A
    INX, DEX, INY, DEY
    TAX, TXA, TAY, TYA
    CLC, SEC

The assembler uses frozen shapes - computation via geometry, not learning.
"""

import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from trix.shapes import (
    add, sub, and_op, or_op, xor,
    asl, lsr, rol, ror,
    inc, dec,
)


# =============================================================================
# INSTRUCTION DEFINITIONS
# =============================================================================

@dataclass
class CPUState:
    """Simple CPU state for assembly execution."""
    A: int = 0      # Accumulator
    X: int = 0      # X register
    Y: int = 0      # Y register
    C: int = 0      # Carry flag
    Z: int = 0      # Zero flag
    N: int = 0      # Negative flag

    def update_nz(self, value: int):
        """Update N and Z flags from result."""
        self.Z = 1 if value == 0 else 0
        self.N = 1 if value & 0x80 else 0


# Instruction patterns
PATTERNS = {
    # Load/Store immediate
    'LDA': r'LDA\s+#\$?([0-9A-Fa-f]+)',
    'LDX': r'LDX\s+#\$?([0-9A-Fa-f]+)',
    'LDY': r'LDY\s+#\$?([0-9A-Fa-f]+)',

    # Arithmetic immediate
    'ADC': r'ADC\s+#\$?([0-9A-Fa-f]+)',
    'SBC': r'SBC\s+#\$?([0-9A-Fa-f]+)',

    # Logic immediate
    'AND': r'AND\s+#\$?([0-9A-Fa-f]+)',
    'ORA': r'ORA\s+#\$?([0-9A-Fa-f]+)',
    'EOR': r'EOR\s+#\$?([0-9A-Fa-f]+)',

    # Shifts (accumulator)
    'ASL_A': r'ASL\s+A',
    'LSR_A': r'LSR\s+A',
    'ROL_A': r'ROL\s+A',
    'ROR_A': r'ROR\s+A',

    # Inc/Dec registers
    'INX': r'INX',
    'DEX': r'DEX',
    'INY': r'INY',
    'DEY': r'DEY',

    # Transfers
    'TAX': r'TAX',
    'TXA': r'TXA',
    'TAY': r'TAY',
    'TYA': r'TYA',

    # Flags
    'CLC': r'CLC',
    'SEC': r'SEC',

    # Compare (sets flags only)
    'CMP': r'CMP\s+#\$?([0-9A-Fa-f]+)',
}


# =============================================================================
# PARSER
# =============================================================================

def parse_line(line: str) -> Optional[Tuple[str, Optional[int]]]:
    """
    Parse a single assembly line.

    Returns:
        (instruction, operand) or None if empty/comment
    """
    # Remove comments
    line = line.split(';')[0].strip().upper()
    if not line:
        return None

    # Try each pattern
    for instr, pattern in PATTERNS.items():
        match = re.match(pattern, line, re.IGNORECASE)
        if match:
            # Extract operand if present
            if match.lastindex:
                operand_str = match.group(1)
                # Handle hex vs decimal
                if line.find('$') >= 0:
                    operand = int(operand_str, 16)
                else:
                    operand = int(operand_str, 16) if all(c in '0123456789ABCDEFabcdef' for c in operand_str) else int(operand_str)
                return (instr, operand & 0xFF)
            else:
                return (instr, None)

    raise SyntaxError(f"Unknown instruction: {line}")


def assemble(code: str) -> List[Tuple[str, Optional[int]]]:
    """
    Assemble source code into instruction list.

    Args:
        code: Assembly source (can be multiline)

    Returns:
        List of (instruction, operand) tuples
    """
    instructions = []

    # Split by newlines only (semicolons are comments, like real 6502 asm)
    lines = code.split('\n')

    for line in lines:
        parsed = parse_line(line)
        if parsed:
            instructions.append(parsed)

    return instructions


# =============================================================================
# EXECUTOR
# =============================================================================

def execute_instruction(
    state: CPUState,
    instr: str,
    operand: Optional[int],
    verbose: bool = False,
) -> str:
    """
    Execute a single instruction on state.

    Returns:
        Shape name used (for verbose output)
    """
    shape_used = "NONE"

    if instr == 'LDA':
        state.A = operand
        state.update_nz(state.A)
        shape_used = "TRANSFER"

    elif instr == 'LDX':
        state.X = operand
        state.update_nz(state.X)
        shape_used = "TRANSFER"

    elif instr == 'LDY':
        state.Y = operand
        state.update_nz(state.Y)
        shape_used = "TRANSFER"

    elif instr == 'ADC':
        result, carry = add(state.A, operand, state.C), None
        # Get carry too
        from trix.shapes import add_with_carry
        result, carry = add_with_carry(state.A, operand, state.C)
        state.A = result
        state.C = carry
        state.update_nz(state.A)
        shape_used = "RIPPLE_ADD"

    elif instr == 'SBC':
        # SBC uses inverted borrow: borrow = 1 - carry
        result = sub(state.A, operand, 1 - state.C)
        # Determine new carry (no borrow if result >= 0)
        full = state.A - operand - (1 - state.C)
        state.C = 0 if full < 0 else 1
        state.A = result
        state.update_nz(state.A)
        shape_used = "RIPPLE_SUB"

    elif instr == 'AND':
        state.A = and_op(state.A, operand)
        state.update_nz(state.A)
        shape_used = "PARALLEL_AND"

    elif instr == 'ORA':
        state.A = or_op(state.A, operand)
        state.update_nz(state.A)
        shape_used = "PARALLEL_OR"

    elif instr == 'EOR':
        state.A = xor(state.A, operand)
        state.update_nz(state.A)
        shape_used = "PARALLEL_XOR"

    elif instr == 'ASL_A':
        state.A, state.C = asl(state.A)
        state.update_nz(state.A)
        shape_used = "SHIFT_LEFT"

    elif instr == 'LSR_A':
        state.A, state.C = lsr(state.A)
        state.update_nz(state.A)
        shape_used = "SHIFT_RIGHT"

    elif instr == 'ROL_A':
        state.A, state.C = rol(state.A, state.C)
        state.update_nz(state.A)
        shape_used = "ROTATE_LEFT"

    elif instr == 'ROR_A':
        state.A, state.C = ror(state.A, state.C)
        state.update_nz(state.A)
        shape_used = "ROTATE_RIGHT"

    elif instr == 'INX':
        state.X = inc(state.X)
        state.update_nz(state.X)
        shape_used = "INCREMENT"

    elif instr == 'DEX':
        state.X = dec(state.X)
        state.update_nz(state.X)
        shape_used = "DECREMENT"

    elif instr == 'INY':
        state.Y = inc(state.Y)
        state.update_nz(state.Y)
        shape_used = "INCREMENT"

    elif instr == 'DEY':
        state.Y = dec(state.Y)
        state.update_nz(state.Y)
        shape_used = "DECREMENT"

    elif instr == 'TAX':
        state.X = state.A
        state.update_nz(state.X)
        shape_used = "TRANSFER"

    elif instr == 'TXA':
        state.A = state.X
        state.update_nz(state.A)
        shape_used = "TRANSFER"

    elif instr == 'TAY':
        state.Y = state.A
        state.update_nz(state.Y)
        shape_used = "TRANSFER"

    elif instr == 'TYA':
        state.A = state.Y
        state.update_nz(state.A)
        shape_used = "TRANSFER"

    elif instr == 'CLC':
        state.C = 0
        shape_used = "IDENTITY"

    elif instr == 'SEC':
        state.C = 1
        shape_used = "IDENTITY"

    elif instr == 'CMP':
        # Compare: subtract but don't store result
        result = sub(state.A, operand, 0)
        state.C = 1 if state.A >= operand else 0
        state.Z = 1 if state.A == operand else 0
        state.N = 1 if result & 0x80 else 0
        shape_used = "RIPPLE_SUB"

    else:
        raise ValueError(f"Unknown instruction: {instr}")

    return shape_used


# =============================================================================
# MAIN API
# =============================================================================

def run(
    code: str,
    initial_state: Optional[Dict] = None,
    verbose: bool = False,
) -> Dict:
    """
    Assemble and run 6502 code.

    Args:
        code: Assembly source code
        initial_state: Optional dict with initial A, X, Y, C values
        verbose: If True, print each instruction and shape used

    Returns:
        Dict with final state: A, X, Y, C, Z, N, instructions

    Example:
        >>> result = run("LDA #$05\\nADC #$03")
        >>> result['A']
        8

        >>> result = run("LDA #$FF\\nEOR #$55", verbose=True)
        # Prints each step
    """
    # Initialize state
    state = CPUState()
    if initial_state:
        state.A = initial_state.get('A', 0) & 0xFF
        state.X = initial_state.get('X', 0) & 0xFF
        state.Y = initial_state.get('Y', 0) & 0xFF
        state.C = initial_state.get('C', 0) & 1

    # Assemble
    instructions = assemble(code)

    if verbose:
        print("┌" + "─" * 58 + "┐")
        print("│ FROZEN 6502 ASSEMBLY EXECUTION" + " " * 26 + "│")
        print("├" + "─" * 58 + "┤")

    # Execute
    for instr, operand in instructions:
        if verbose:
            if operand is not None:
                instr_str = f"{instr} #${operand:02X}"
            else:
                instr_str = instr.replace('_A', ' A')
            print(f"│ {instr_str:<20}", end="")

        shape_used = execute_instruction(state, instr, operand, verbose)

        if verbose:
            print(f" → A=${state.A:02X} X=${state.X:02X} Y=${state.Y:02X} C={state.C}", end="")
            print(f" [{shape_used}]".ljust(20) + "│")

    if verbose:
        print("└" + "─" * 58 + "┘")

    return {
        'A': state.A,
        'X': state.X,
        'Y': state.Y,
        'C': state.C,
        'Z': state.Z,
        'N': state.N,
        'instructions': len(instructions),
    }


def disassemble(instructions: List[Tuple[str, Optional[int]]]) -> str:
    """
    Convert instruction list back to assembly text.

    Args:
        instructions: List of (instruction, operand) tuples

    Returns:
        Assembly source code
    """
    lines = []
    for instr, operand in instructions:
        if operand is not None:
            lines.append(f"{instr} #${operand:02X}")
        else:
            # Handle ASL_A -> ASL A
            lines.append(instr.replace('_A', ' A'))
    return '\n'.join(lines)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = ['run', 'assemble', 'disassemble', 'CPUState']
