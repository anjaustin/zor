"""
TriX Ingest Core - Yosys JSON Netlist to executable System

This is a standalone version for PyPI distribution.
"""

import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional, Set
from collections import defaultdict
import numpy as np

# =============================================================================
# Backend Selection
# =============================================================================

_USE_NATIVE = False
_native_ops = None

def _try_load_native():
    """Attempt to load native ops from trix-core package."""
    global _USE_NATIVE, _native_ops

    # Try 1: trix-core package (PyPI)
    try:
        from trix_core import TrixOps
        _native_ops = TrixOps()
        _USE_NATIVE = True
        return True
    except ImportError:
        pass

    # Try 2: Full trix package (direct import of native module)
    try:
        import importlib.util
        import os

        # Common locations
        search_paths = [
            os.environ.get('TRIX_NATIVE_PATH', ''),
            '/workspace/ZOR/src/trix/native/ops/__init__.py',
            os.path.expanduser('~/trix/src/trix/native/ops/__init__.py'),
        ]

        for path in search_paths:
            if path and os.path.exists(path):
                spec = importlib.util.spec_from_file_location('trix_native_ops', path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _native_ops = module.TrixOps()
                _USE_NATIVE = True
                return True
    except Exception:
        pass

    return False

# Try to load native on import
_try_load_native()

if _USE_NATIVE:
    print(f"[trix-ingest] Using native backend: {_native_ops.simd}")


# =============================================================================
# Frozen Shapes - Polynomial implementations
# =============================================================================

if _USE_NATIVE and _native_ops is not None:
    def shape_and(a: float, b: float) -> float:
        return float(_native_ops.and_(np.array([a]), np.array([b]))[0])

    def shape_or(a: float, b: float) -> float:
        return float(_native_ops.or_(np.array([a]), np.array([b]))[0])

    def shape_xor(a: float, b: float) -> float:
        return float(_native_ops.xor(np.array([a]), np.array([b]))[0])

    def shape_not(a: float) -> float:
        return float(_native_ops.not_(np.array([a]))[0])

    def shape_nand(a: float, b: float) -> float:
        return float(_native_ops.not_(_native_ops.and_(np.array([a]), np.array([b])))[0])

    def shape_nor(a: float, b: float) -> float:
        return float(_native_ops.not_(_native_ops.or_(np.array([a]), np.array([b])))[0])

    def shape_xnor(a: float, b: float) -> float:
        return float(_native_ops.not_(_native_ops.xor(np.array([a]), np.array([b])))[0])

    def shape_mux(s: float, a: float, b: float) -> float:
        not_s = _native_ops.not_(np.array([s]))
        left = _native_ops.and_(not_s, np.array([a]))
        right = _native_ops.and_(np.array([s]), np.array([b]))
        return float(_native_ops.or_(left, right)[0])

    def shape_buf(a: float) -> float:
        return a

    def shape_andnot(a: float, b: float) -> float:
        not_b = _native_ops.not_(np.array([b]))
        return float(_native_ops.and_(np.array([a]), not_b)[0])

    def shape_ornot(a: float, b: float) -> float:
        not_b = _native_ops.not_(np.array([b]))
        return float(_native_ops.or_(np.array([a]), not_b)[0])

else:
    # Pure Python fallback - same polynomial algebra
    def shape_and(a: float, b: float) -> float:
        """AND: a * b"""
        return a * b

    def shape_or(a: float, b: float) -> float:
        """OR: a + b - ab"""
        return a + b - a * b

    def shape_xor(a: float, b: float) -> float:
        """XOR: a + b - 2ab"""
        return a + b - 2 * a * b

    def shape_not(a: float) -> float:
        """NOT: 1 - a"""
        return 1.0 - a

    def shape_nand(a: float, b: float) -> float:
        """NAND: 1 - ab"""
        return 1.0 - a * b

    def shape_nor(a: float, b: float) -> float:
        """NOR: 1 - (a + b - ab)"""
        return 1.0 - (a + b - a * b)

    def shape_xnor(a: float, b: float) -> float:
        """XNOR: 1 - (a + b - 2ab)"""
        return 1.0 - (a + b - 2 * a * b)

    def shape_mux(s: float, a: float, b: float) -> float:
        """MUX: (1-s)*a + s*b"""
        return (1.0 - s) * a + s * b

    def shape_buf(a: float) -> float:
        """Buffer: identity"""
        return a

    def shape_andnot(a: float, b: float) -> float:
        """ANDNOT: a * (1-b) = a - ab"""
        return a - a * b

    def shape_ornot(a: float, b: float) -> float:
        """ORNOT: 1 - b + ab"""
        return 1.0 - b + a * b


# =============================================================================
# Cell Mapping
# =============================================================================

CELL_MAP: Dict[str, Tuple[Callable, List[str], str]] = {
    "$_AND_": (shape_and, ["A", "B"], "Y"),
    "$_OR_": (shape_or, ["A", "B"], "Y"),
    "$_XOR_": (shape_xor, ["A", "B"], "Y"),
    "$_NOT_": (shape_not, ["A"], "Y"),
    "$_NAND_": (shape_nand, ["A", "B"], "Y"),
    "$_NOR_": (shape_nor, ["A", "B"], "Y"),
    "$_XNOR_": (shape_xnor, ["A", "B"], "Y"),
    "$_MUX_": (shape_mux, ["S", "A", "B"], "Y"),
    "$_BUF_": (shape_buf, ["A"], "Y"),
    "$_TBUF_": (shape_buf, ["A"], "Y"),
    "$_ANDNOT_": (shape_andnot, ["A", "B"], "Y"),
    "$_ORNOT_": (shape_ornot, ["A", "B"], "Y"),
}

SEQUENTIAL_CELLS = {
    "$_DFF_P_", "$_DFF_N_", "$_DFF_PP0_", "$_DFF_PP1_",
    "$_DFF_PN0_", "$_DFF_PN1_", "$_DFF_NP0_", "$_DFF_NP1_",
    "$_DFF_NN0_", "$_DFF_NN1_", "$_DFFSR_PPP_", "$_DFFSR_PPN_",
    "$_SDFF_PP0_", "$_SDFF_PP1_", "$_DLATCH_P_", "$_DLATCH_N_",
    "$_SR_PP_", "$_SR_PN_", "$_SR_NP_", "$_SR_NN_",
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Wire:
    """A wire (signal) in the circuit"""
    name: str
    bits: List[int]

    @property
    def width(self) -> int:
        return len(self.bits)


@dataclass
class Cell:
    """A cell (gate) in the circuit"""
    name: str
    cell_type: str
    connections: Dict[str, List[int]]
    shape_fn: Callable
    input_ports: List[str]
    output_port: str


@dataclass
class System:
    """A compiled system from a netlist"""
    name: str
    inputs: Dict[str, Wire]
    outputs: Dict[str, Wire]
    cells: Dict[str, Cell]
    execution_order: List[str]
    bit_to_wire: Dict[int, str]
    using_native: bool = _USE_NATIVE

    def __repr__(self):
        backend = "native" if self.using_native else "python"
        return (f"System({self.name}, "
                f"inputs={list(self.inputs.keys())}, "
                f"outputs={list(self.outputs.keys())}, "
                f"cells={len(self.cells)}, "
                f"backend={backend})")


class UnsupportedCellError(Exception):
    """Raised when a cell type is not supported"""
    pass


class CyclicGraphError(Exception):
    """Raised when the netlist contains a combinational loop"""
    pass


# =============================================================================
# Parser
# =============================================================================

def ingest_yosys_json(path: str) -> System:
    """
    Parse a Yosys JSON netlist and return an executable System.

    Args:
        path: Path to the Yosys JSON output file

    Returns:
        System object ready for execution

    Raises:
        UnsupportedCellError: If sequential or unknown cells are found
        CyclicGraphError: If combinational loop detected
        FileNotFoundError: If file doesn't exist
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return _parse_netlist(data)


def _parse_netlist(data: dict) -> System:
    """Parse Yosys JSON data structure into a System"""
    modules = data.get("modules", {})
    if not modules:
        raise ValueError("No modules found in netlist")

    module_name = list(modules.keys())[0]
    module = modules[module_name]

    # Parse ports
    inputs = {}
    outputs = {}
    bit_to_wire = {}

    ports = module.get("ports", {})
    for port_name, port_info in ports.items():
        direction = port_info.get("direction")
        bits = port_info.get("bits", [])
        wire = Wire(name=port_name, bits=bits)

        if direction == "input":
            inputs[port_name] = wire
        elif direction == "output":
            outputs[port_name] = wire

        for i, bit in enumerate(bits):
            if isinstance(bit, int):
                if wire.width == 1:
                    bit_to_wire[bit] = port_name
                else:
                    bit_to_wire[bit] = f"{port_name}[{i}]"

    # Parse cells
    cells = {}
    yosys_cells = module.get("cells", {})

    for cell_name, cell_info in yosys_cells.items():
        cell_type = cell_info.get("type", "")
        connections = cell_info.get("connections", {})

        if cell_type in SEQUENTIAL_CELLS:
            raise UnsupportedCellError(
                f"Sequential cell '{cell_type}' not supported. "
                f"trix-ingest only handles combinational logic."
            )

        if cell_type not in CELL_MAP:
            raise UnsupportedCellError(
                f"Unknown cell type '{cell_type}'. "
                f"Supported: {list(CELL_MAP.keys())}"
            )

        shape_fn, input_ports, output_port = CELL_MAP[cell_type]

        cell = Cell(
            name=cell_name,
            cell_type=cell_type,
            connections=connections,
            shape_fn=shape_fn,
            input_ports=input_ports,
            output_port=output_port,
        )
        cells[cell_name] = cell

        out_bits = connections.get(output_port, [])
        for i, bit in enumerate(out_bits):
            if isinstance(bit, int) and bit not in bit_to_wire:
                bit_to_wire[bit] = f"__{cell_name}_{output_port}"

    execution_order = _topological_sort(cells, inputs, bit_to_wire)

    return System(
        name=module_name,
        inputs=inputs,
        outputs=outputs,
        cells=cells,
        execution_order=execution_order,
        bit_to_wire=bit_to_wire,
        using_native=_USE_NATIVE,
    )


def _topological_sort(
    cells: Dict[str, Cell],
    inputs: Dict[str, Wire],
    bit_to_wire: Dict[int, str]
) -> List[str]:
    """Topologically sort cells by dependency."""
    dependencies: Dict[str, Set[str]] = defaultdict(set)

    input_bits = set()
    for wire in inputs.values():
        input_bits.update(wire.bits)

    for cell_name, cell in cells.items():
        for port in cell.input_ports:
            port_bits = cell.connections.get(port, [])
            for bit in port_bits:
                if isinstance(bit, int) and bit not in input_bits:
                    for other_name, other_cell in cells.items():
                        if other_name == cell_name:
                            continue
                        other_out_bits = other_cell.connections.get(
                            other_cell.output_port, []
                        )
                        if bit in other_out_bits:
                            dependencies[cell_name].add(other_name)

    in_degree = {name: 0 for name in cells}
    for name, deps in dependencies.items():
        in_degree[name] = len(deps)

    queue = [name for name, deg in in_degree.items() if deg == 0]
    result = []

    while queue:
        current = queue.pop(0)
        result.append(current)

        for name, deps in dependencies.items():
            if current in deps:
                in_degree[name] -= 1
                if in_degree[name] == 0:
                    queue.append(name)

    if len(result) != len(cells):
        remaining = set(cells.keys()) - set(result)
        raise CyclicGraphError(
            f"Combinational loop detected involving: {remaining}"
        )

    return result


# =============================================================================
# Execution
# =============================================================================

def execute(system: System, inputs: Dict[str, int]) -> Dict[str, int]:
    """
    Execute the system on given input values.

    Args:
        system: A System from ingest_yosys_json
        inputs: Dict mapping port names to values (int for multi-bit, 0/1 for single-bit)

    Returns:
        Dict mapping output port names to values
    """
    wire_values: Dict[int, float] = {}

    # Set input values
    for port_name, wire in system.inputs.items():
        if port_name in inputs:
            value = inputs[port_name]
            for i, bit in enumerate(wire.bits):
                if isinstance(bit, int):
                    if wire.width == 1:
                        wire_values[bit] = float(value)
                    else:
                        wire_values[bit] = float((value >> i) & 1)
        else:
            for i, bit in enumerate(wire.bits):
                if isinstance(bit, int):
                    bit_name = f"{port_name}[{i}]" if wire.width > 1 else port_name
                    if bit_name in inputs:
                        wire_values[bit] = float(inputs[bit_name])

    # Execute cells in topological order
    for cell_name in system.execution_order:
        cell = system.cells[cell_name]

        input_values = []
        for port in cell.input_ports:
            port_bits = cell.connections.get(port, [])
            if len(port_bits) != 1:
                raise ValueError(
                    f"Expected single-bit port {port} in cell {cell_name}"
                )

            bit = port_bits[0]
            if bit == "0":
                input_values.append(0.0)
            elif bit == "1":
                input_values.append(1.0)
            elif isinstance(bit, int):
                input_values.append(wire_values.get(bit, 0.0))
            else:
                raise ValueError(f"Unknown bit value: {bit}")

        # Execute the frozen shape
        output_value = cell.shape_fn(*input_values)

        out_bits = cell.connections.get(cell.output_port, [])
        for bit in out_bits:
            if isinstance(bit, int):
                wire_values[bit] = output_value

    # Collect outputs
    result = {}
    for port_name, wire in system.outputs.items():
        if wire.width == 1:
            bit = wire.bits[0]
            if isinstance(bit, int):
                result[port_name] = int(round(wire_values.get(bit, 0.0)))
        else:
            value = 0
            for i, bit in enumerate(wire.bits):
                if isinstance(bit, int):
                    bit_val = int(round(wire_values.get(bit, 0.0)))
                    value |= (bit_val << i)
            result[port_name] = value

    return result


# =============================================================================
# Validation
# =============================================================================

def validate_exhaustive(
    system: System,
    truth_fn: Optional[Callable] = None
) -> Tuple[bool, List[Tuple]]:
    """
    Exhaustively validate the system against expected behavior.

    Args:
        system: The System to validate
        truth_fn: Optional function(inputs) -> expected_outputs

    Returns:
        Tuple of (passed: bool, failures: list)
    """
    total_input_bits = sum(w.width for w in system.inputs.values())

    if total_input_bits > 20:
        raise ValueError(
            f"Exhaustive validation requires 2^{total_input_bits} cases. "
            f"Use validate_sample for larger circuits."
        )

    failures = []
    n_cases = 2 ** total_input_bits

    for i in range(n_cases):
        inputs = {}
        bit_idx = 0
        for port_name, wire in system.inputs.items():
            if wire.width == 1:
                inputs[port_name] = (i >> bit_idx) & 1
                bit_idx += 1
            else:
                value = 0
                for j in range(wire.width):
                    value |= ((i >> bit_idx) & 1) << j
                    bit_idx += 1
                inputs[port_name] = value

        try:
            result = execute(system, inputs)
        except Exception as e:
            failures.append((inputs, "no error", str(e)))
            continue

        if truth_fn is not None:
            expected = truth_fn(inputs)
            if result != expected:
                failures.append((inputs, expected, result))

    return len(failures) == 0, failures


def validate_sample(
    system: System,
    truth_fn: Callable,
    n_samples: int = 10000
) -> Tuple[bool, List[Tuple]]:
    """
    Validate with random sampling for large circuits.

    Args:
        system: The System to validate
        truth_fn: Function(inputs) -> expected_outputs
        n_samples: Number of random test cases

    Returns:
        Tuple of (passed: bool, failures: list)
    """
    import random

    failures = []

    for _ in range(n_samples):
        inputs = {}
        for port_name, wire in system.inputs.items():
            if wire.width == 1:
                inputs[port_name] = random.randint(0, 1)
            else:
                inputs[port_name] = random.randint(0, 2**wire.width - 1)

        result = execute(system, inputs)
        expected = truth_fn(inputs)

        if result != expected:
            failures.append((inputs, expected, result))

    return len(failures) == 0, failures


# =============================================================================
# Utilities
# =============================================================================

def system_summary(system: System) -> str:
    """Generate a human-readable summary of the system."""
    backend = "NATIVE (C/SIMD)" if system.using_native else "Python fallback"
    lines = [
        f"System: {system.name}",
        f"Backend: {backend}",
        "=" * 50,
        "",
        "Inputs:",
    ]

    for name, wire in system.inputs.items():
        lines.append(f"  {name}: {wire.width} bit(s)")

    lines.append("")
    lines.append("Outputs:")
    for name, wire in system.outputs.items():
        lines.append(f"  {name}: {wire.width} bit(s)")

    lines.append("")
    lines.append(f"Cells: {len(system.cells)}")

    type_counts = defaultdict(int)
    for cell in system.cells.values():
        type_counts[cell.cell_type] += 1

    for cell_type, count in sorted(type_counts.items()):
        lines.append(f"  {cell_type}: {count}")

    lines.append("")
    lines.append(f"Execution order: {len(system.execution_order)} steps")

    return "\n".join(lines)


def system_to_truth_table(system: System, max_bits: int = 8) -> str:
    """Generate truth table for small systems."""
    total_input_bits = sum(w.width for w in system.inputs.values())

    if total_input_bits > max_bits:
        return f"Truth table too large ({2**total_input_bits} rows)"

    input_names = list(system.inputs.keys())
    output_names = list(system.outputs.keys())

    lines = [" | ".join(input_names + output_names)]
    lines.append("-" * len(lines[0]))

    for i in range(2 ** total_input_bits):
        inputs = {}
        bit_idx = 0
        row = []

        for port_name, wire in system.inputs.items():
            if wire.width == 1:
                val = (i >> bit_idx) & 1
                inputs[port_name] = val
                row.append(str(val))
                bit_idx += 1
            else:
                value = 0
                for j in range(wire.width):
                    value |= ((i >> bit_idx) & 1) << j
                    bit_idx += 1
                inputs[port_name] = value
                row.append(str(value))

        result = execute(system, inputs)

        for port_name in output_names:
            row.append(str(result.get(port_name, "?")))

        lines.append(" | ".join(row))

    return "\n".join(lines)
