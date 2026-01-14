# trix-ingest

**Import any combinational Verilog design as a deterministic neural network.**

```
Verilog RTL -> [Yosys] -> JSON Netlist -> [trix-ingest] -> Executable System
```

## Installation

```bash
pip install trix-ingest
```

## Quick Start

```bash
# 1. Synthesize your Verilog with Yosys
yosys -p "read_verilog adder.v; synth; write_json adder.json"

# 2. Import and run
trix-ingest adder.json --execute '{"a": 5, "b": 3, "cin": 0}'
```

Or from Python:

```python
from trix_ingest import ingest_yosys_json, execute

system = ingest_yosys_json("adder.json")
result = execute(system, {"a": 5, "b": 3, "cin": 0})
print(result)  # {"sum": 8, "cout": 0}
```

## What This Does

trix-ingest compiles combinational Verilog to a **deterministic neural net** that:

1. **Uses frozen shapes** - Pure polynomial math, not approximation
2. **Guarantees correctness** - Algebraic equivalence, not statistical
3. **Runs anywhere** - No GPU required, pure CPU execution
4. **Is fully interpretable** - Every operation is traceable

This is **compilation**, not learning. The "weights" are derived algebraically from the circuit structure.

## Supported Cells

| Yosys Cell | Frozen Shape | Polynomial |
|------------|--------------|------------|
| `$_AND_` | AND | `a * b` |
| `$_OR_` | OR | `a + b - ab` |
| `$_XOR_` | XOR | `a + b - 2ab` |
| `$_NOT_` | NOT | `1 - a` |
| `$_NAND_` | NAND | `1 - ab` |
| `$_NOR_` | NOR | `1 - (a + b - ab)` |
| `$_XNOR_` | XNOR | `1 - (a + b - 2ab)` |
| `$_MUX_` | MUX | `(1-s)*a + s*b` |
| `$_ANDNOT_` | AND-NOT | `a - ab` |
| `$_ORNOT_` | OR-NOT | `1 - b + ab` |

Sequential cells (flip-flops, latches) are not supported - trix-ingest handles combinational logic only.

## API Reference

### `ingest_yosys_json(path) -> System`

Parse a Yosys JSON netlist file.

```python
system = ingest_yosys_json("design.json")
```

### `execute(system, inputs) -> outputs`

Execute the system on input values.

```python
result = execute(system, {"a": 1, "b": 0})
```

### `validate_exhaustive(system, truth_fn) -> (passed, failures)`

Exhaustively validate against a reference function.

```python
def my_adder(inputs):
    total = inputs["a"] + inputs["b"]
    return {"sum": total % 2, "cout": total // 2}

passed, failures = validate_exhaustive(system, my_adder)
```

### `system_summary(system) -> str`

Generate human-readable summary.

### `system_to_truth_table(system) -> str`

Generate truth table (max 8 input bits).

## CLI

```bash
# Show system info
trix-ingest design.json

# Show truth table
trix-ingest design.json --truth-table

# Execute with inputs
trix-ingest design.json --execute '{"a": 1, "b": 0}'
```

## Native Acceleration

For best performance, install `trix-core` (C/SIMD backend):

```bash
pip install trix-core
```

Without `trix-core`, trix-ingest uses a pure Python fallback (same correctness, slower execution).

## License

MIT

## Links

- [Documentation](https://github.com/anjaustin/zor/blob/main/docs/INGEST.md)
- [GitHub](https://github.com/anjaustin/zor)
- [Issues](https://github.com/anjaustin/zor/issues)
