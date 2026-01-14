"""
Command-line interface for trix-ingest.

Usage:
    trix-ingest design.json
    trix-ingest design.json --truth-table
    trix-ingest design.json --execute '{"a": 1, "b": 0}'
"""

import argparse
import json
import sys

from .core import (
    ingest_yosys_json,
    execute,
    system_summary,
    system_to_truth_table,
    UnsupportedCellError,
    CyclicGraphError,
)


def main():
    parser = argparse.ArgumentParser(
        description="Import Yosys JSON netlists as deterministic neural networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show system summary
  trix-ingest design.json

  # Show truth table (small circuits only)
  trix-ingest design.json --truth-table

  # Execute with specific inputs
  trix-ingest design.json --execute '{"a": 1, "b": 0}'

First, synthesize your Verilog with Yosys:
  yosys -p "read_verilog design.v; synth; write_json design.json"
        """
    )

    parser.add_argument(
        "json_file",
        help="Path to Yosys JSON netlist file"
    )
    parser.add_argument(
        "--truth-table", "-t",
        action="store_true",
        help="Print truth table (max 8 input bits)"
    )
    parser.add_argument(
        "--execute", "-e",
        metavar="JSON",
        help="Execute with given inputs (JSON dict)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress informational messages"
    )

    args = parser.parse_args()

    try:
        system = ingest_yosys_json(args.json_file)

        if not args.quiet:
            print(system_summary(system))
            print()

        if args.truth_table:
            print("Truth Table:")
            print(system_to_truth_table(system))
            print()

        if args.execute:
            inputs = json.loads(args.execute)
            result = execute(system, inputs)
            print(f"Inputs:  {inputs}")
            print(f"Outputs: {result}")

    except FileNotFoundError:
        print(f"Error: File not found: {args.json_file}", file=sys.stderr)
        sys.exit(1)
    except UnsupportedCellError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except CyclicGraphError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
