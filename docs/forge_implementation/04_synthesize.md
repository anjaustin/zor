# Synthesis: Forge Stage 1 Implementation

## Objective

Build a working pipeline that takes combinational Verilog and produces verified frozen C code.

```
Verilog → Yosys → Netlist → Compose → C Code → Verify → ✓
```

---

## Architecture

```
foundry/forge/
├── __init__.py          # Public API: freeze()
├── yosys.py             # Yosys subprocess interface
├── netlist.py           # JSON parsing, graph construction
├── composer.py          # Polynomial composition
├── emitter.py           # C code generation
└── verifier.py          # Exhaustive testing

foundry/forge/tests/
├── __init__.py
├── test_yosys.py
├── test_netlist.py
├── test_composer.py
├── test_emitter.py
├── test_verifier.py
└── test_integration.py
```

---

## Module Specifications

### 1. yosys.py

```python
"""Yosys synthesis interface."""

from pathlib import Path
import subprocess
import tempfile
import json

class YosysError(Exception):
    """Yosys synthesis failed."""
    pass

def synthesize(verilog_source: str, top_module: str = None) -> dict:
    """
    Synthesize Verilog to gate-level netlist.

    Args:
        verilog_source: Verilog source code as string
        top_module: Top module name (auto-detected if None)

    Returns:
        Parsed JSON netlist as dict

    Raises:
        YosysError: If synthesis fails
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        v_path = Path(tmpdir) / "input.v"
        j_path = Path(tmpdir) / "output.json"

        v_path.write_text(verilog_source)

        script = f"""
            read_verilog {v_path}
            synth -flatten -noalumacc
            abc -g AND,OR,XOR,NOT
            write_json {j_path}
        """

        result = subprocess.run(
            ["yosys", "-Q", "-p", script],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise YosysError(f"Yosys failed: {result.stderr}")

        return json.loads(j_path.read_text())
```

### 2. netlist.py

```python
"""Netlist parsing and graph construction."""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from enum import Enum

class GateType(Enum):
    NOT = "NOT"
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    NAND = "NAND"
    NOR = "NOR"
    XNOR = "XNOR"
    CONST0 = "CONST0"
    CONST1 = "CONST1"
    BUF = "BUF"

@dataclass
class Cell:
    name: str
    gate_type: GateType
    inputs: List[int]      # Wire IDs
    output: int            # Wire ID

@dataclass
class Port:
    name: str
    direction: str         # "input" or "output"
    bits: List[int]        # Wire IDs

@dataclass
class Netlist:
    module_name: str
    cells: Dict[str, Cell]
    ports: Dict[str, Port]
    wire_names: Dict[int, str]

def parse(yosys_json: dict, module_name: str = None) -> Netlist:
    """Parse Yosys JSON to Netlist."""
    # Auto-detect module name if not provided
    if module_name is None:
        module_name = list(yosys_json["modules"].keys())[0]

    module = yosys_json["modules"][module_name]

    # Parse ports
    ports = {}
    for port_name, port_data in module["ports"].items():
        ports[port_name] = Port(
            name=port_name,
            direction=port_data["direction"],
            bits=port_data["bits"]
        )

    # Parse cells
    cells = {}
    for cell_name, cell_data in module.get("cells", {}).items():
        gate_type = _parse_gate_type(cell_data["type"])
        inputs, output = _parse_connections(cell_data, gate_type)
        cells[cell_name] = Cell(
            name=cell_name,
            gate_type=gate_type,
            inputs=inputs,
            output=output
        )

    # Build wire names
    wire_names = {}
    for name, data in module.get("netnames", {}).items():
        for i, bit in enumerate(data["bits"]):
            if isinstance(bit, int):
                wire_names[bit] = f"{name}[{i}]" if len(data["bits"]) > 1 else name

    return Netlist(
        module_name=module_name,
        cells=cells,
        ports=ports,
        wire_names=wire_names
    )

def topological_sort(netlist: Netlist) -> List[Cell]:
    """Sort cells so each cell's inputs are computed before it."""
    # Build dependency graph
    wire_to_cell = {}  # wire_id -> cell that produces it
    for cell in netlist.cells.values():
        wire_to_cell[cell.output] = cell

    # Find input wires (produced by ports, not cells)
    input_wires = set()
    for port in netlist.ports.values():
        if port.direction == "input":
            input_wires.update(port.bits)

    # Kahn's algorithm
    in_degree = {name: 0 for name in netlist.cells}
    for cell in netlist.cells.values():
        for inp in cell.inputs:
            if inp in wire_to_cell:
                in_degree[cell.name] += 1

    queue = [name for name, deg in in_degree.items() if deg == 0]
    result = []

    while queue:
        name = queue.pop(0)
        cell = netlist.cells[name]
        result.append(cell)

        # Find cells that depend on this cell's output
        for other in netlist.cells.values():
            if cell.output in other.inputs:
                in_degree[other.name] -= 1
                if in_degree[other.name] == 0:
                    queue.append(other.name)

    if len(result) != len(netlist.cells):
        raise ValueError("Combinational loop detected")

    return result

# Helper functions
def _parse_gate_type(yosys_type: str) -> GateType:
    mapping = {
        "$_NOT_": GateType.NOT,
        "$_AND_": GateType.AND,
        "$_OR_": GateType.OR,
        "$_XOR_": GateType.XOR,
        "$_NAND_": GateType.NAND,
        "$_NOR_": GateType.NOR,
        "$_XNOR_": GateType.XNOR,
        "$_BUF_": GateType.BUF,
    }
    if yosys_type not in mapping:
        raise ValueError(f"Unsupported gate type: {yosys_type}")
    return mapping[yosys_type]

def _parse_connections(cell_data: dict, gate_type: GateType) -> Tuple[List[int], int]:
    conns = cell_data["connections"]
    if gate_type == GateType.NOT or gate_type == GateType.BUF:
        return [conns["A"][0]], conns["Y"][0]
    else:
        return [conns["A"][0], conns["B"][0]], conns["Y"][0]
```

### 3. composer.py

```python
"""Polynomial composition for gate netlists."""

from dataclasses import dataclass
from typing import Dict, List, Union
from .netlist import Netlist, Cell, GateType, topological_sort

@dataclass
class Expr:
    """Polynomial expression in terms of inputs."""
    # For simplicity, represent as string that can be evaluated
    # More sophisticated: use symbolic representation
    code: str

def compose(netlist: Netlist) -> Dict[str, Expr]:
    """
    Compose gate polynomials to get output expressions.

    Returns:
        Dict mapping output port bit names to expressions
    """
    # Map wire IDs to expressions
    wire_expr: Dict[int, str] = {}

    # Initialize input wires as variables
    for port in netlist.ports.values():
        if port.direction == "input":
            for i, bit in enumerate(port.bits):
                var_name = f"{port.name}_{i}" if len(port.bits) > 1 else port.name
                wire_expr[bit] = var_name

    # Handle constants
    wire_expr["0"] = "0"
    wire_expr["1"] = "1"
    wire_expr[0] = "0"  # Yosys uses 0 for constant 0
    wire_expr[1] = "1"  # Yosys uses 1 for constant 1

    # Process cells in topological order
    for cell in topological_sort(netlist):
        inputs = [wire_expr.get(i, str(i)) for i in cell.inputs]
        wire_expr[cell.output] = _gate_polynomial(cell.gate_type, inputs)

    # Extract output expressions
    outputs = {}
    for port in netlist.ports.values():
        if port.direction == "output":
            for i, bit in enumerate(port.bits):
                bit_name = f"{port.name}_{i}" if len(port.bits) > 1 else port.name
                outputs[bit_name] = Expr(code=wire_expr.get(bit, "0"))

    return outputs

def _gate_polynomial(gate_type: GateType, inputs: List[str]) -> str:
    """Return polynomial expression for gate."""
    if gate_type == GateType.NOT:
        a = inputs[0]
        return f"(1 - {a})"

    elif gate_type == GateType.BUF:
        return inputs[0]

    elif gate_type == GateType.AND:
        a, b = inputs
        return f"({a} * {b})"

    elif gate_type == GateType.OR:
        a, b = inputs
        return f"({a} + {b} - {a} * {b})"

    elif gate_type == GateType.XOR:
        a, b = inputs
        return f"({a} + {b} - 2 * {a} * {b})"

    elif gate_type == GateType.NAND:
        a, b = inputs
        return f"(1 - {a} * {b})"

    elif gate_type == GateType.NOR:
        a, b = inputs
        return f"(1 - {a} - {b} + {a} * {b})"

    elif gate_type == GateType.XNOR:
        a, b = inputs
        return f"(1 - {a} - {b} + 2 * {a} * {b})"

    else:
        raise ValueError(f"Unknown gate type: {gate_type}")
```

### 4. emitter.py

```python
"""C code generation for frozen polynomials."""

from pathlib import Path
from typing import Dict, List
from .netlist import Netlist, Port
from .composer import Expr

def emit_c(
    netlist: Netlist,
    expressions: Dict[str, Expr]
) -> tuple[str, str]:
    """
    Generate C header and source for frozen module.

    Returns:
        (header_content, source_content)
    """
    name = netlist.module_name

    # Collect input and output info
    inputs = []
    outputs = []
    for port in netlist.ports.values():
        if port.direction == "input":
            inputs.append((port.name, len(port.bits)))
        else:
            outputs.append((port.name, len(port.bits)))

    header = _generate_header(name, inputs, outputs)
    source = _generate_source(name, inputs, outputs, expressions)

    return header, source

def _generate_header(name: str, inputs: List, outputs: List) -> str:
    guard = f"FROZEN_{name.upper()}_H"

    # Determine types based on bit widths
    def c_type(bits):
        if bits <= 8: return "uint8_t"
        if bits <= 16: return "uint16_t"
        if bits <= 32: return "uint32_t"
        return "uint64_t"

    params = ", ".join(f"{c_type(bits)} {n}" for n, bits in inputs)
    out_params = ", ".join(f"{c_type(bits)} *{n}" for n, bits in outputs)
    if params and out_params:
        params = params + ", " + out_params
    elif out_params:
        params = out_params

    return f'''/*
 * Frozen {name} - Generated by TriX Forge
 *
 * Pure polynomial computation. No inference.
 */

#ifndef {guard}
#define {guard}

#include <stdint.h>

void frozen_{name}({params});

#endif /* {guard} */
'''

def _generate_source(name: str, inputs: List, outputs: List, expressions: Dict[str, Expr]) -> str:
    def c_type(bits):
        if bits <= 8: return "uint8_t"
        if bits <= 16: return "uint16_t"
        if bits <= 32: return "uint32_t"
        return "uint64_t"

    params = ", ".join(f"{c_type(bits)} {n}" for n, bits in inputs)
    out_params = ", ".join(f"{c_type(bits)} *{n}" for n, bits in outputs)
    if params and out_params:
        params = params + ", " + out_params
    elif out_params:
        params = out_params

    # Generate bit extraction for multi-bit inputs
    bit_extracts = []
    for in_name, bits in inputs:
        for i in range(bits):
            var = f"{in_name}_{i}" if bits > 1 else in_name
            if bits > 1:
                bit_extracts.append(f"    uint8_t {var} = ({in_name} >> {i}) & 1;")
            else:
                bit_extracts.append(f"    uint8_t {var} = {in_name} & 1;")

    # Generate output bit calculations
    output_calcs = []
    for out_name, bits in outputs:
        for i in range(bits):
            var = f"{out_name}_{i}" if bits > 1 else out_name
            if var in expressions:
                output_calcs.append(f"    uint8_t {var} = {expressions[var].code};")

    # Generate output packing
    output_packs = []
    for out_name, bits in outputs:
        if bits > 1:
            terms = " | ".join(f"({out_name}_{i} << {i})" for i in range(bits))
            output_packs.append(f"    *{out_name} = {terms};")
        else:
            output_packs.append(f"    *{out_name} = {out_name};")

    return f'''/*
 * Frozen {name} - Generated by TriX Forge
 */

#include "frozen_{name}.h"

void frozen_{name}({params}) {{
{chr(10).join(bit_extracts)}

{chr(10).join(output_calcs)}

{chr(10).join(output_packs)}
}}
'''
```

### 5. verifier.py

```python
"""Verification by exhaustive testing."""

import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Tuple

def verify_exhaustive(
    verilog_source: str,
    frozen_header: str,
    frozen_source: str,
    input_bits: int,
    output_bits: int,
    module_name: str
) -> Tuple[bool, int, int]:
    """
    Verify frozen model against Verilog by exhaustive testing.

    Returns:
        (passed, tested, failed)
    """
    # Generate test harness
    test_code = _generate_test_harness(
        module_name, input_bits, output_bits
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Write files
        (tmp / f"frozen_{module_name}.h").write_text(frozen_header)
        (tmp / f"frozen_{module_name}.c").write_text(frozen_source)
        (tmp / "test_harness.c").write_text(test_code)
        (tmp / "input.v").write_text(verilog_source)

        # Compile frozen model
        subprocess.run([
            "gcc", "-O2", "-c",
            str(tmp / f"frozen_{module_name}.c"),
            "-o", str(tmp / "frozen.o")
        ], check=True)

        # Compile test harness
        subprocess.run([
            "gcc", "-O2",
            str(tmp / "test_harness.c"),
            str(tmp / "frozen.o"),
            "-o", str(tmp / "test")
        ], check=True)

        # Run test
        result = subprocess.run(
            [str(tmp / "test")],
            capture_output=True,
            text=True
        )

        # Parse result
        lines = result.stdout.strip().split("\n")
        for line in lines:
            if line.startswith("RESULT:"):
                parts = line.split()
                passed = parts[1] == "PASS"
                tested = int(parts[2])
                failed = int(parts[3])
                return passed, tested, failed

        return False, 0, 0

def _generate_test_harness(module_name: str, input_bits: int, output_bits: int) -> str:
    """Generate C test harness that compares frozen vs expected."""
    # For now, generate expected values inline
    # In full implementation, would use Verilator or Icarus
    return f'''
#include <stdio.h>
#include <stdint.h>
#include "frozen_{module_name}.h"

int main() {{
    int tested = 0;
    int failed = 0;

    for (uint64_t i = 0; i < (1ULL << {input_bits}); i++) {{
        // TODO: Compare with reference implementation
        tested++;
    }}

    printf("RESULT: %s %d %d\\n", failed == 0 ? "PASS" : "FAIL", tested, failed);
    return failed > 0 ? 1 : 0;
}}
'''
```

### 6. __init__.py (Public API)

```python
"""
TriX Forge - Verilog to Frozen C Pipeline

Usage:
    from foundry.forge import freeze

    verilog = '''
    module adder4(input [3:0] a, input [3:0] b, output [4:0] sum);
        assign sum = a + b;
    endmodule
    '''

    header, source = freeze(verilog, "adder4")
"""

from pathlib import Path
from typing import Tuple, Optional

from .yosys import synthesize, YosysError
from .netlist import parse, Netlist
from .composer import compose
from .emitter import emit_c

def freeze(
    verilog_source: str,
    module_name: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Tuple[str, str]:
    """
    Freeze Verilog combinational logic to C code.

    Args:
        verilog_source: Verilog source code
        module_name: Module to freeze (auto-detect if None)
        output_dir: Write files here (return strings if None)

    Returns:
        (header_content, source_content)
    """
    # Synthesize to netlist
    yosys_json = synthesize(verilog_source, module_name)

    # Parse netlist
    netlist = parse(yosys_json, module_name)

    # Compose polynomials
    expressions = compose(netlist)

    # Emit C code
    header, source = emit_c(netlist, expressions)

    # Optionally write files
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"frozen_{netlist.module_name}.h").write_text(header)
        (output_dir / f"frozen_{netlist.module_name}.c").write_text(source)

    return header, source
```

---

## Test Plan

### Unit Tests

| Test | Input | Expected |
|------|-------|----------|
| yosys_not_gate | NOT gate Verilog | JSON with $_NOT_ cell |
| yosys_and_gate | AND gate Verilog | JSON with $_AND_ cell |
| netlist_parse_simple | Yosys JSON | Netlist with correct cells/ports |
| netlist_topo_sort | Multi-gate netlist | Correct dependency order |
| composer_not | NOT gate netlist | `(1 - a)` |
| composer_and | AND gate netlist | `(a * b)` |
| composer_xor | XOR gate netlist | `(a + b - 2 * a * b)` |
| emitter_simple | Expressions | Valid C code |

### Integration Tests

| Test | Input | Verification |
|------|-------|--------------|
| not_gate | `assign y = ~a;` | Exhaustive (2 cases) |
| and_gate | `assign y = a & b;` | Exhaustive (4 cases) |
| xor_gate | `assign y = a ^ b;` | Exhaustive (4 cases) |
| half_adder | `assign {c,s} = a + b;` | Exhaustive (4 cases) |
| full_adder | `assign {c,s} = a + b + cin;` | Exhaustive (8 cases) |
| adder4 | `assign sum = a + b;` | Exhaustive (65,536 cases) |
| adder8 | `assign sum = a + b;` | Exhaustive (16M cases) |

---

## Success Criteria

- [ ] `freeze()` function works for combinational Verilog
- [ ] 4-bit adder freezes correctly (65,536 test cases pass)
- [ ] 8-bit adder freezes correctly (16M test cases pass)
- [ ] Generated C compiles with `-Wall -Werror`
- [ ] All unit tests pass
- [ ] All integration tests pass

---

## Implementation Order

1. **yosys.py** - Get Yosys integration working
2. **netlist.py** - Parse JSON to usable structure
3. **composer.py** - Build polynomial expressions
4. **emitter.py** - Generate C code
5. **tests** - Unit tests for each module
6. **Integration** - End-to-end freeze() function
7. **verifier.py** - Exhaustive testing framework
8. **Integration tests** - Full pipeline verification

---

*The wood cuts itself. Let's build.*
