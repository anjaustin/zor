"""
Minimal Verilog parser for combinational logic.

Handles:
- module declarations
- input/output ports with bit widths
- assign statements with basic operators

Does NOT handle:
- always blocks
- registers
- complex expressions
- hierarchical modules
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ParseError(Exception):
    """Verilog parsing failed."""
    pass


class OpType(Enum):
    """Expression operators."""
    VAR = "VAR"       # Variable reference
    CONST = "CONST"   # Constant value
    NOT = "NOT"       # ~a
    AND = "AND"       # a & b
    OR = "OR"         # a | b
    XOR = "XOR"       # a ^ b
    ADD = "ADD"       # a + b
    SUB = "SUB"       # a - b
    INDEX = "INDEX"   # a[i]
    CONCAT = "CONCAT" # {a, b}


@dataclass
class Expr:
    """Expression tree node."""
    op: OpType
    value: Optional[str] = None  # For VAR/CONST
    children: List['Expr'] = field(default_factory=list)
    bit_index: Optional[int] = None  # For INDEX


@dataclass
class Port:
    """Module port."""
    name: str
    direction: str  # "input" or "output"
    width: int      # Bit width (1 for single bit)
    msb: int = 0    # Most significant bit index
    lsb: int = 0    # Least significant bit index


@dataclass
class Assignment:
    """Continuous assignment."""
    target: str           # Output signal name
    target_bits: Optional[Tuple[int, int]] = None  # (msb, lsb) or None for full
    expr: Expr = None     # Expression tree


@dataclass
class Module:
    """Parsed Verilog module."""
    name: str
    ports: Dict[str, Port]
    assignments: List[Assignment]


def parse(verilog_source: str) -> Module:
    """
    Parse Verilog source to Module structure.

    Args:
        verilog_source: Verilog source code

    Returns:
        Parsed Module

    Raises:
        ParseError: If parsing fails
    """
    # Remove comments
    source = _remove_comments(verilog_source)

    # Parse module declaration
    module_match = re.search(
        r'module\s+(\w+)\s*\((.*?)\)\s*;',
        source,
        re.DOTALL
    )
    if not module_match:
        raise ParseError("No module declaration found")

    module_name = module_match.group(1)
    port_list = module_match.group(2)

    # Parse port declarations
    ports = _parse_ports(source)

    # Parse assignments
    assignments = _parse_assignments(source, ports)

    return Module(
        name=module_name,
        ports=ports,
        assignments=assignments
    )


def _remove_comments(source: str) -> str:
    """Remove Verilog comments."""
    # Remove // comments
    source = re.sub(r'//.*$', '', source, flags=re.MULTILINE)
    # Remove /* */ comments
    source = re.sub(r'/\*.*?\*/', '', source, flags=re.DOTALL)
    return source


def _parse_ports(source: str) -> Dict[str, Port]:
    """Parse port declarations."""
    ports = {}

    # Pattern for: input [7:0] name or input name
    port_pattern = re.compile(
        r'(input|output)\s+'
        r'(?:\[(\d+):(\d+)\]\s+)?'  # Optional [msb:lsb]
        r'(\w+)',
        re.MULTILINE
    )

    for match in port_pattern.finditer(source):
        direction = match.group(1)
        msb = int(match.group(2)) if match.group(2) else 0
        lsb = int(match.group(3)) if match.group(3) else 0
        name = match.group(4)
        width = msb - lsb + 1

        ports[name] = Port(
            name=name,
            direction=direction,
            width=width,
            msb=msb,
            lsb=lsb
        )

    return ports


def _parse_assignments(source: str, ports: Dict[str, Port]) -> List[Assignment]:
    """Parse assign statements."""
    assignments = []

    # Pattern for: assign target = expr;
    assign_pattern = re.compile(
        r'assign\s+'
        r'(\{[^}]+\}|\w+(?:\[\d+(?::\d+)?\])?)\s*'  # Target (possibly concat or indexed)
        r'=\s*'
        r'([^;]+);',
        re.MULTILINE
    )

    for match in assign_pattern.finditer(source):
        target_str = match.group(1).strip()
        expr_str = match.group(2).strip()

        # Parse target
        target, target_bits = _parse_target(target_str)

        # Parse expression
        expr = _parse_expr(expr_str, ports)

        assignments.append(Assignment(
            target=target,
            target_bits=target_bits,
            expr=expr
        ))

    return assignments


def _parse_target(target_str: str) -> Tuple[str, Optional[Tuple[int, int]]]:
    """Parse assignment target."""
    # Check for bit select: name[i] or name[msb:lsb]
    bit_match = re.match(r'(\w+)\[(\d+)(?::(\d+))?\]', target_str)
    if bit_match:
        name = bit_match.group(1)
        msb = int(bit_match.group(2))
        lsb = int(bit_match.group(3)) if bit_match.group(3) else msb
        return name, (msb, lsb)

    # Check for concatenation: {a, b}
    if target_str.startswith('{'):
        # For now, treat as single target
        return target_str, None

    return target_str, None


def _parse_expr(expr_str: str, ports: Dict[str, Port]) -> Expr:
    """Parse expression string to expression tree."""
    expr_str = expr_str.strip()

    # Remove outer parentheses
    while expr_str.startswith('(') and expr_str.endswith(')'):
        # Check if they match
        depth = 0
        matched = True
        for i, c in enumerate(expr_str):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth == 0 and i < len(expr_str) - 1:
                matched = False
                break
        if matched:
            expr_str = expr_str[1:-1].strip()
        else:
            break

    # Check for unary NOT: ~expr
    if expr_str.startswith('~'):
        return Expr(
            op=OpType.NOT,
            children=[_parse_expr(expr_str[1:], ports)]
        )

    # Find lowest precedence binary operator (outside parentheses)
    # Precedence (low to high): |, ^, &, +-, ~
    for ops, op_type in [
        (['|'], OpType.OR),
        (['^'], OpType.XOR),
        (['&'], OpType.AND),
        (['+'], OpType.ADD),
        (['-'], OpType.SUB),
    ]:
        pos = _find_operator(expr_str, ops)
        if pos >= 0:
            left = expr_str[:pos].strip()
            right = expr_str[pos+1:].strip()
            return Expr(
                op=op_type,
                children=[
                    _parse_expr(left, ports),
                    _parse_expr(right, ports)
                ]
            )

    # Check for bit index: name[i]
    index_match = re.match(r'(\w+)\[(\d+)\]$', expr_str)
    if index_match:
        return Expr(
            op=OpType.INDEX,
            value=index_match.group(1),
            bit_index=int(index_match.group(2))
        )

    # Check for constant
    if expr_str.isdigit():
        return Expr(op=OpType.CONST, value=expr_str)

    # Check for sized constant: N'bXXX or N'hXX
    const_match = re.match(r"(\d+)'([bhd])([0-9a-fA-F_]+)$", expr_str)
    if const_match:
        width = int(const_match.group(1))
        base = const_match.group(2)
        value_str = const_match.group(3).replace('_', '')
        if base == 'b':
            value = int(value_str, 2)
        elif base == 'h':
            value = int(value_str, 16)
        else:
            value = int(value_str)
        return Expr(op=OpType.CONST, value=str(value))

    # Must be a variable
    if re.match(r'^\w+$', expr_str):
        return Expr(op=OpType.VAR, value=expr_str)

    raise ParseError(f"Cannot parse expression: {expr_str}")


def _find_operator(expr: str, ops: List[str]) -> int:
    """Find operator position outside parentheses."""
    depth = 0
    # Search from right to left for left-associativity
    for i in range(len(expr) - 1, -1, -1):
        c = expr[i]
        if c == ')':
            depth += 1
        elif c == '(':
            depth -= 1
        elif depth == 0 and c in ops:
            # Make sure it's not part of another operator
            # e.g., don't match & in &&
            if i > 0 and expr[i-1] == c:
                continue
            if i < len(expr) - 1 and expr[i+1] == c:
                continue
            return i
    return -1
