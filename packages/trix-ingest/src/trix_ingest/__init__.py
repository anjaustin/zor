"""
TriX Ingest - Import Verilog designs as deterministic neural networks.

Usage:
    from trix_ingest import ingest_yosys_json, execute, validate_exhaustive

    system = ingest_yosys_json("design.json")
    result = execute(system, {"a": 1, "b": 0})
"""

__version__ = "0.1.0"

from .core import (
    ingest_yosys_json,
    execute,
    validate_exhaustive,
    validate_sample,
    system_summary,
    system_to_truth_table,
    UnsupportedCellError,
    CyclicGraphError,
)

__all__ = [
    "ingest_yosys_json",
    "execute",
    "validate_exhaustive",
    "validate_sample",
    "system_summary",
    "system_to_truth_table",
    "UnsupportedCellError",
    "CyclicGraphError",
    "__version__",
]
