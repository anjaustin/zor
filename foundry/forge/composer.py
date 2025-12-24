"""
Polynomial composition from expression trees.

Transforms parsed Verilog expressions into polynomial form
suitable for C code generation.

Supports two modes:
- Expanded: Full polynomial expansion (small circuits)
- Factored: Intermediate variables for carries (large circuits)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
from .parser import Module, Expr, OpType, Port, Assignment


@dataclass
class BitExpr:
    """Expression for a single output bit as polynomial string."""
    name: str           # Output bit name (e.g., "sum_0")
    polynomial: str     # C expression string
    inputs_used: Set[str]  # Input bits referenced


@dataclass
class FactoredResult:
    """Result of factored composition with intermediate variables."""
    intermediates: List[Tuple[str, str]]  # (var_name, expression) in order
    outputs: List[Tuple[str, str]]        # (bit_name, expression) for output bits
    input_bits: List[str]                 # Input bit names needed


def compose(module: Module) -> List[BitExpr]:
    """
    Compose module expressions into bit-level polynomials.

    For multi-bit operations like addition, we expand to individual
    bit operations using polynomial arithmetic.

    Returns:
        List of BitExpr for each output bit
    """
    results = []

    # Get output ports
    outputs = {name: port for name, port in module.ports.items()
               if port.direction == "output"}

    for assign in module.assignments:
        target_port = outputs.get(assign.target)
        if not target_port:
            continue

        # Compose expression for each output bit
        bit_exprs = _compose_expr(assign.expr, module.ports, target_port.width)

        for i, poly in enumerate(bit_exprs):
            # Always use _i suffix for consistency
            bit_name = f"{assign.target}_{i}"
            results.append(BitExpr(
                name=bit_name,
                polynomial=poly,
                inputs_used=_extract_inputs(poly)
            ))

    return results


def _compose_expr(expr: Expr, ports: Dict[str, Port], result_width: int) -> List[str]:
    """
    Compose expression to list of polynomial strings (one per bit).

    For an N-bit result, returns [bit_0_poly, bit_1_poly, ..., bit_N-1_poly].
    """
    if expr.op == OpType.CONST:
        value = int(expr.value)
        return [str((value >> i) & 1) for i in range(result_width)]

    elif expr.op == OpType.VAR:
        port = ports.get(expr.value)
        if port:
            width = min(port.width, result_width)
            # Always use _i suffix for consistency
            polys = [f"{expr.value}_{i}" for i in range(width)]
            # Zero-extend if needed
            polys.extend(["0"] * (result_width - width))
            return polys
        else:
            # Unknown variable, treat as single bit with _0 suffix
            return [f"{expr.value}_0"] + ["0"] * (result_width - 1)

    elif expr.op == OpType.INDEX:
        bit_name = f"{expr.value}_{expr.bit_index}"
        return [bit_name] + ["0"] * (result_width - 1)

    elif expr.op == OpType.NOT:
        child_polys = _compose_expr(expr.children[0], ports, result_width)
        return [f"(1 - {p})" if p != "0" else "1" for p in child_polys]

    elif expr.op == OpType.AND:
        left = _compose_expr(expr.children[0], ports, result_width)
        right = _compose_expr(expr.children[1], ports, result_width)
        return [_poly_and(l, r) for l, r in zip(left, right)]

    elif expr.op == OpType.OR:
        left = _compose_expr(expr.children[0], ports, result_width)
        right = _compose_expr(expr.children[1], ports, result_width)
        return [_poly_or(l, r) for l, r in zip(left, right)]

    elif expr.op == OpType.XOR:
        left = _compose_expr(expr.children[0], ports, result_width)
        right = _compose_expr(expr.children[1], ports, result_width)
        return [_poly_xor(l, r) for l, r in zip(left, right)]

    elif expr.op == OpType.ADD:
        # Multi-bit addition with carry propagation
        left_width = _infer_width(expr.children[0], ports)
        right_width = _infer_width(expr.children[1], ports)
        operand_width = max(left_width, right_width)

        left = _compose_expr(expr.children[0], ports, operand_width)
        right = _compose_expr(expr.children[1], ports, operand_width)

        return _ripple_add(left, right, result_width)

    elif expr.op == OpType.SUB:
        # Subtraction: a - b = a + (~b) + 1
        left_width = _infer_width(expr.children[0], ports)
        right_width = _infer_width(expr.children[1], ports)
        operand_width = max(left_width, right_width)

        left = _compose_expr(expr.children[0], ports, operand_width)
        right = _compose_expr(expr.children[1], ports, operand_width)

        # Invert right operand
        right_inv = [f"(1 - {r})" if r != "0" else "1" for r in right]

        # Add with carry-in of 1
        return _ripple_add(left, right_inv, result_width, carry_in="1")

    else:
        raise ValueError(f"Unsupported operation: {expr.op}")


def _infer_width(expr: Expr, ports: Dict[str, Port]) -> int:
    """Infer bit width of expression."""
    if expr.op == OpType.CONST:
        value = int(expr.value)
        if value == 0:
            return 1
        return value.bit_length()

    elif expr.op == OpType.VAR:
        port = ports.get(expr.value)
        return port.width if port else 1

    elif expr.op == OpType.INDEX:
        return 1

    elif expr.op in (OpType.NOT, OpType.AND, OpType.OR, OpType.XOR):
        widths = [_infer_width(c, ports) for c in expr.children]
        return max(widths) if widths else 1

    elif expr.op in (OpType.ADD, OpType.SUB):
        widths = [_infer_width(c, ports) for c in expr.children]
        return max(widths) + 1 if widths else 1

    return 1


def _poly_and(a: str, b: str) -> str:
    """AND polynomial: a * b"""
    if a == "0" or b == "0":
        return "0"
    if a == "1":
        return b
    if b == "1":
        return a
    return f"({a} * {b})"


def _poly_or(a: str, b: str) -> str:
    """OR polynomial: a + b - a*b"""
    if a == "0":
        return b
    if b == "0":
        return a
    if a == "1" or b == "1":
        return "1"
    return f"({a} + {b} - {a} * {b})"


def _poly_xor(a: str, b: str) -> str:
    """XOR polynomial: a + b - 2*a*b"""
    if a == "0":
        return b
    if b == "0":
        return a
    if a == b:
        return "0"
    return f"({a} + {b} - 2 * {a} * {b})"


def _poly_not(a: str) -> str:
    """NOT polynomial: 1 - a"""
    if a == "0":
        return "1"
    if a == "1":
        return "0"
    return f"(1 - {a})"


def _ripple_add(left: List[str], right: List[str], result_width: int,
                carry_in: str = "0") -> List[str]:
    """
    Ripple carry addition using polynomial arithmetic.

    For each bit position:
        sum = a XOR b XOR carry
        carry_out = (a AND b) OR (carry AND (a XOR b))
    """
    results = []
    carry = carry_in

    # Pad to equal length
    max_len = max(len(left), len(right))
    left = left + ["0"] * (max_len - len(left))
    right = right + ["0"] * (max_len - len(right))

    for i in range(result_width):
        if i < len(left):
            a = left[i]
            b = right[i]

            # sum = a XOR b XOR carry
            ab_xor = _poly_xor(a, b)
            sum_bit = _poly_xor(ab_xor, carry)

            # carry_out = (a AND b) OR (carry AND (a XOR b))
            ab_and = _poly_and(a, b)
            carry_ab_xor = _poly_and(carry, ab_xor)
            carry = _poly_or(ab_and, carry_ab_xor)

            results.append(sum_bit)
        else:
            # Just the carry
            results.append(carry)
            carry = "0"

    return results


def _extract_inputs(polynomial: str) -> Set[str]:
    """Extract input variable names from polynomial."""
    import re
    # Find all word tokens that look like variable names
    tokens = re.findall(r'\b([a-zA-Z_]\w*)\b', polynomial)
    # Filter out numbers
    return {t for t in tokens if not t.isdigit()}


def compose_factored(module: Module) -> FactoredResult:
    """
    Compose module with factored intermediate variables.

    Instead of expanding carries inline (exponential growth),
    we generate intermediate carry variables (linear growth).

    This enables 32-bit, 64-bit, and arbitrary-width adders.
    """
    intermediates = []
    outputs = []
    input_bits = []

    # Collect input bits
    for port in module.ports.values():
        if port.direction == "input":
            for i in range(port.width):
                input_bits.append(f"{port.name}_{i}")

    # Get output ports
    output_ports = {name: port for name, port in module.ports.items()
                    if port.direction == "output"}

    for assign in module.assignments:
        target_port = output_ports.get(assign.target)
        if not target_port:
            continue

        # Check if this is an addition (needs factoring)
        if assign.expr.op == OpType.ADD:
            _compose_add_factored(
                assign.expr, module.ports, target_port.width,
                assign.target, intermediates, outputs
            )
        elif assign.expr.op == OpType.SUB:
            _compose_sub_factored(
                assign.expr, module.ports, target_port.width,
                assign.target, intermediates, outputs
            )
        else:
            # Fall back to expanded composition for non-adders
            bit_exprs = _compose_expr(assign.expr, module.ports, target_port.width)
            for i, poly in enumerate(bit_exprs):
                outputs.append((f"{assign.target}_{i}", poly))

    return FactoredResult(
        intermediates=intermediates,
        outputs=outputs,
        input_bits=input_bits
    )


def _compose_add_factored(expr: Expr, ports: Dict[str, Port], result_width: int,
                          target: str, intermediates: List, outputs: List,
                          carry_in: str = "0") -> None:
    """
    Compose addition with factored carry variables.

    Generates:
        c_0 = 0  (or carry_in for subtraction)
        c_1 = (a_0 * b_0) + (c_0 * (a_0 + b_0 - 2*a_0*b_0)) - (a_0 * b_0) * (c_0 * (a_0 + b_0 - 2*a_0*b_0))
        c_2 = ...
        sum_0 = a_0 + b_0 - 2*a_0*b_0 + c_0 - 2*(a_0 + b_0 - 2*a_0*b_0)*c_0  (XOR of XOR)
        ...
    """
    # Get operand widths
    left_width = _infer_width(expr.children[0], ports)
    right_width = _infer_width(expr.children[1], ports)
    operand_width = max(left_width, right_width)

    # Build input references
    left_var = expr.children[0].value if expr.children[0].op == OpType.VAR else None
    right_var = expr.children[1].value if expr.children[1].op == OpType.VAR else None

    # Generate carry chain with intermediate variables
    prev_carry = carry_in

    for i in range(result_width):
        if i < operand_width:
            a = f"{left_var}_{i}" if left_var else "0"
            b = f"{right_var}_{i}" if right_var else "0"

            # XOR for this bit: a ^ b
            ab_xor = _poly_xor(a, b)

            # Sum bit: (a XOR b) XOR carry
            sum_expr = _poly_xor(ab_xor, prev_carry)
            outputs.append((f"{target}_{i}", sum_expr))

            # Carry out: (a AND b) OR (carry AND (a XOR b))
            ab_and = _poly_and(a, b)
            carry_and_xor = _poly_and(prev_carry, ab_xor)
            carry_expr = _poly_or(ab_and, carry_and_xor)

            # Store carry as intermediate (except last iteration if not needed)
            if i < result_width - 1:
                carry_name = f"_c_{i+1}"
                intermediates.append((carry_name, carry_expr))
                prev_carry = carry_name
            else:
                # Final carry becomes MSB if result is wider than operands
                prev_carry = carry_expr
        else:
            # Beyond operand width, output is just the carry
            outputs.append((f"{target}_{i}", prev_carry))
            prev_carry = "0"


def _compose_sub_factored(expr: Expr, ports: Dict[str, Port], result_width: int,
                          target: str, intermediates: List, outputs: List) -> None:
    """
    Compose subtraction with factored variables.

    Subtraction: a - b = a + (~b) + 1
    We invert b and use carry_in = 1.
    """
    left_width = _infer_width(expr.children[0], ports)
    right_width = _infer_width(expr.children[1], ports)
    operand_width = max(left_width, right_width)

    left_var = expr.children[0].value if expr.children[0].op == OpType.VAR else None
    right_var = expr.children[1].value if expr.children[1].op == OpType.VAR else None

    # Generate carry chain with b inverted and carry_in = 1
    prev_carry = "1"  # +1 for two's complement

    for i in range(result_width):
        if i < operand_width:
            a = f"{left_var}_{i}" if left_var else "0"
            b_raw = f"{right_var}_{i}" if right_var else "0"
            b = _poly_not(b_raw)  # Invert b

            ab_xor = _poly_xor(a, b)
            sum_expr = _poly_xor(ab_xor, prev_carry)
            outputs.append((f"{target}_{i}", sum_expr))

            ab_and = _poly_and(a, b)
            carry_and_xor = _poly_and(prev_carry, ab_xor)
            carry_expr = _poly_or(ab_and, carry_and_xor)

            if i < result_width - 1:
                carry_name = f"_c_{i+1}"
                intermediates.append((carry_name, carry_expr))
                prev_carry = carry_name
            else:
                prev_carry = carry_expr
        else:
            outputs.append((f"{target}_{i}", prev_carry))
            prev_carry = "0"
