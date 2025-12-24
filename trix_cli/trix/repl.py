"""
The REPL - interactive frozen shape exploration.
"""
import sys
import re
import traceback
from typing import Optional, Dict, Any

from .display import (
    BOLD, DIM, GREEN, YELLOW, BLUE, CYAN, RED, RESET,
    print_banner, print_shape_info, print_truth_table,
    print_error, print_success, print_code, print_section,
    ProgressDisplay
)
from .foundry import get_foundry, Shape
from .demo import run_demo


HELP_TEXT = f"""
{BOLD}TriX REPL Commands{RESET}
{DIM}──────────────────{RESET}

  {GREEN}demo{RESET}                    Run the full demo
  {GREEN}help{RESET}                    Show this help

{BOLD}Shapes{RESET}
  {GREEN}xor{RESET}, {GREEN}and{RESET}, {GREEN}or{RESET}, {GREEN}add{RESET}, ...   Show shape info
  {GREEN}xor(5, 3){RESET}               Compute with shape
  {GREEN}shapes{RESET}                  List all available shapes

{BOLD}Custom Shapes{RESET}
  {GREEN}shape("lambda a, b: ...")   {RESET}Create custom shape
  {GREEN}show <name>{RESET}             Show shape details

{BOLD}Validation{RESET}
  {GREEN}validate <name>{RESET}         Exhaustive validation (8-bit = 65536 cases)
  {GREEN}validate <name> --bits 4{RESET} Validate with fewer bits (faster)

{BOLD}Export{RESET}
  {GREEN}export <name>{RESET}           Export to C code
  {GREEN}export <name> --show{RESET}    Show generated code

{BOLD}Other{RESET}
  {GREEN}quit{RESET} / {GREEN}exit{RESET}            Exit the REPL
  {GREEN}clear{RESET}                   Clear screen

"""


class REPL:
    """The TriX REPL."""

    def __init__(self):
        self.foundry = get_foundry()
        self.running = True
        self.last_result: Optional[Any] = None

        # Import shapes to register them
        from . import shapes

    def run(self):
        """Run the REPL loop."""
        print_banner()

        while self.running:
            try:
                line = input(f"{CYAN}trix>{RESET} ").strip()
                if not line:
                    continue

                self.execute(line)

            except EOFError:
                print()
                self.running = False
            except KeyboardInterrupt:
                print("\n  (Use 'quit' to exit)")

    def execute(self, line: str):
        """Execute a REPL command."""
        # Handle special commands first
        cmd = line.lower().split()[0] if line else ""

        if cmd in ('quit', 'exit', 'q'):
            self.running = False
            print(f"\n  {DIM}Geometry is computation. Until next time.{RESET}\n")
            return

        if cmd == 'demo':
            run_demo()
            return

        if cmd == 'help':
            print(HELP_TEXT)
            return

        if cmd == 'clear':
            print("\033[2J\033[H")
            print_banner()
            return

        if cmd == 'shapes':
            self._list_shapes()
            return

        if cmd == 'show':
            self._show_shape(line)
            return

        if cmd == 'validate':
            self._validate_shape(line)
            return

        if cmd == 'export':
            self._export_shape(line)
            return

        if line.startswith('shape('):
            self._create_shape(line)
            return

        # Try to evaluate as shape call or shape name
        self._evaluate(line)

    def _list_shapes(self):
        """List all available shapes."""
        print(f"\n{BOLD}Available Shapes{RESET}")
        print(f"{DIM}────────────────{RESET}")

        shapes = self.foundry.list_shapes()
        logic = [s for s in shapes if s in ('xor', 'and', 'or', 'not', 'nand', 'nor', 'xnor')]
        arith = [s for s in shapes if s in ('add', 'sub', 'mul')]
        custom = [s for s in shapes if s.startswith('custom_')]

        if logic:
            print(f"\n  {CYAN}Logic:{RESET} {', '.join(logic)}")
        if arith:
            print(f"  {CYAN}Arithmetic:{RESET} {', '.join(arith)}")
        if custom:
            print(f"  {CYAN}Custom:{RESET} {', '.join(custom)}")

        print(f"\n  Type a shape name to see details, or {GREEN}<name>(a, b){RESET} to compute.\n")

    def _show_shape(self, line: str):
        """Show shape details."""
        parts = line.split()
        if len(parts) < 2:
            print_error("Usage: show <shape_name>")
            return

        name = parts[1]
        shape = self.foundry.get(name)
        if not shape:
            print_error(f"Shape '{name}' not found.", f"Try 'shapes' to list available shapes.")
            return

        print(print_shape_info(shape))
        print(print_truth_table(shape))

    def _validate_shape(self, line: str):
        """Validate a shape exhaustively."""
        parts = line.split()
        if len(parts) < 2:
            print_error("Usage: validate <shape_name>")
            return

        name = parts[1]
        shape = self.foundry.get(name)
        if not shape:
            print_error(f"Shape '{name}' not found.")
            return

        # Parse optional --bits flag
        bits = shape.bits
        if '--bits' in parts:
            try:
                idx = parts.index('--bits')
                bits = int(parts[idx + 1])
            except (IndexError, ValueError):
                print_error("Usage: validate <name> --bits <n>")
                return

        # Calculate total cases
        if shape.arity == 1:
            total = 1 << bits
        else:
            total = (1 << bits) ** 2

        print(f"\n  Validating {BOLD}{name}{RESET} ({total:,} cases)...\n")

        progress = ProgressDisplay(total, "Validating")

        # Temporarily override bits for validation
        old_bits = shape.bits
        shape.bits = bits

        try:
            passed, total = shape.validate(callback=progress.update)
        finally:
            shape.bits = old_bits

        color = GREEN if passed == total else RED
        pct = (passed / total) * 100

        print(f"\n  Result: {color}{passed:,}/{total:,} ({pct:.0f}%){RESET}")

        if passed == total:
            print(f"  {GREEN}✓{RESET} Shape is {BOLD}100% correct{RESET}\n")
        else:
            print(f"  {RED}✗{RESET} Shape has {total - passed} failures\n")

    def _export_shape(self, line: str):
        """Export shape to target language."""
        parts = line.split()
        if len(parts) < 2:
            print_error("Usage: export <shape_name> [--show]")
            return

        name = parts[1]
        shape = self.foundry.get(name)
        if not shape:
            print_error(f"Shape '{name}' not found.")
            return

        show = '--show' in parts

        # Validate first if not already
        if not shape._validated:
            print(f"  Validating first...")
            passed, total = shape.validate()
            if passed != total:
                print_error(f"Shape failed validation ({passed}/{total}). Cannot export.")
                return

        # Generate C code
        c_code = shape.export_c()

        # Write files
        c_file = f"{name}.c"
        h_file = f"{name}.h"

        with open(c_file, 'w') as f:
            f.write(c_code)

        with open(h_file, 'w') as f:
            type_str = "uint8_t"
            if shape.arity == 1:
                f.write(f"#pragma once\n#include <stdint.h>\n{type_str} trix_{name}({type_str} a);\n")
            else:
                f.write(f"#pragma once\n#include <stdint.h>\n{type_str} trix_{name}({type_str} a, {type_str} b);\n")

        print_success(f"Exported to {GREEN}{c_file}{RESET} and {GREEN}{h_file}{RESET}")

        if show:
            print_code(c_code, "c")

    def _create_shape(self, line: str):
        """Create a custom shape from lambda expression."""
        # Extract the expression from shape("...")
        match = re.match(r'shape\s*\(\s*["\'](.+?)["\']\s*\)', line)
        if not match:
            print_error(
                "Invalid shape syntax.",
                'Use: shape("lambda a, b: a ^ b")'
            )
            return

        expr = match.group(1)

        try:
            shape = self.foundry.custom(expr)
            print(f"\n  {GREEN}✓{RESET} Created shape '{CYAN}{shape.name}{RESET}'")
            print(f"  Type '{GREEN}show {shape.name}{RESET}' to see details")
            print(f"  Type '{GREEN}validate {shape.name}{RESET}' to verify\n")
        except ValueError as e:
            print_error(str(e))
        except Exception as e:
            print_error(f"Failed to create shape: {e}")

    def _evaluate(self, line: str):
        """Evaluate a line as a shape call or name."""
        # Check if it's a shape name (info request)
        shape = self.foundry.get(line)
        if shape:
            print(print_shape_info(shape))
            print(print_truth_table(shape))
            return

        # Try to evaluate as function call
        # e.g., xor(5, 3)
        match = re.match(r'(\w+)\s*\(([^)]+)\)', line)
        if match:
            name = match.group(1)
            args_str = match.group(2)

            shape = self.foundry.get(name)
            if not shape:
                print_error(f"Unknown shape '{name}'.", "Try 'shapes' to list available shapes.")
                return

            try:
                args = [int(x.strip()) for x in args_str.split(',')]
                result = shape(*args)
                self.last_result = result
                print(f"{result}")
            except ValueError as e:
                print_error(f"Invalid arguments: {e}")
            except Exception as e:
                print_error(f"Execution error: {e}")
            return

        # Unknown command
        print_error(
            f"Unknown command: '{line}'",
            "Type 'help' for available commands."
        )


def run_repl():
    """Start the REPL."""
    repl = REPL()
    repl.run()


if __name__ == "__main__":
    run_repl()
