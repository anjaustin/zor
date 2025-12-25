#!/usr/bin/env python3
"""
TriX Command Line Interface

Unified interface for export, test, and benchmark operations.

Usage:
    python -m trix.cli export --dispatch table.json --name alu --output ./out
    python -m trix.cli --help
"""

import argparse
import json
import sys
from pathlib import Path


def load_dispatch(path: str) -> tuple:
    """Load dispatch table from JSON.

    Expected format:
    {
        "table": {"0": 0, "1": 1, ...},
        "shapes": ["add", "sub", "xor", ...]
    }
    """
    with open(path) as f:
        data = json.load(f)

    table = {int(k): v for k, v in data["table"].items()}
    shapes = data["shapes"]
    return table, shapes


def cmd_export(args):
    """Handle 'export' subcommand."""
    # Lazy import to keep CLI fast
    from foundry.export.dispatch_export import DispatchExporter

    try:
        table, shapes = load_dispatch(args.dispatch)
    except FileNotFoundError:
        print(f"Error: Dispatch file not found: {args.dispatch}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {args.dispatch}: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing required field in dispatch file: {e}", file=sys.stderr)
        sys.exit(1)

    exporter = DispatchExporter(
        dispatch_table=table,
        shapes=shapes,
        name=args.name
    )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    try:
        header_path, source_path = exporter.export(output)
        print(f"Exported: {header_path}")
        print(f"Exported: {source_path}")
    except Exception as e:
        print(f"Error during export: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_test(args):
    """Handle 'test' subcommand (placeholder)."""
    print("Test subcommand not yet implemented.")
    print("Use pytest to run tests: python -m pytest tests/")


def cmd_benchmark(args):
    """Handle 'benchmark' subcommand (placeholder)."""
    print("Benchmark subcommand not yet implemented.")
    print("See examples/07_deploy.py for benchmark examples.")


def main():
    parser = argparse.ArgumentParser(
        prog="trix",
        description="TriX CLI - Export, test, and benchmark frozen models"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Export subcommand
    export_parser = subparsers.add_parser(
        "export",
        help="Export dispatch table to C"
    )
    export_parser.add_argument(
        "--dispatch", "-d",
        required=True,
        help="Path to dispatch table JSON"
    )
    export_parser.add_argument(
        "--name", "-n",
        default="trix_model",
        help="Name for generated files (default: trix_model)"
    )
    export_parser.add_argument(
        "--output", "-o",
        default="./out",
        help="Output directory (default: ./out)"
    )
    export_parser.add_argument(
        "--format", "-f",
        choices=["c", "cuda", "verilog"],
        default="c",
        help="Output format (default: c)"
    )
    export_parser.set_defaults(func=cmd_export)

    # Test subcommand (placeholder)
    test_parser = subparsers.add_parser(
        "test",
        help="Validate exported code against reference"
    )
    test_parser.set_defaults(func=cmd_test)

    # Benchmark subcommand (placeholder)
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Measure performance of exported code"
    )
    bench_parser.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
