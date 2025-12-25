#!/usr/bin/env python3
"""
08_explore.py - Your Canvas

You've learned the fundamentals.
Now the real work begins.

This is your blank canvas. No tutorial. Just possibilities.
"""

import sys
from pathlib import Path

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print()
    print("=" * 60)
    print("  Your Canvas")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------
    # What you've learned
    # -----------------------------------------------------------------

    print("  What you've learned:")
    print()
    print("    00_witness  - A 6502 CPU runs on frozen polynomials")
    print("    01_touch    - XOR is a + b - 2ab. Always.")
    print("    02_truth    - All logic gates are polynomials")
    print("    03_build    - You can construct any gate from math")
    print("    04_compose  - Combine shapes into useful units")
    print("    05_route    - Learning is routing, not computation")
    print("    06_export   - Freeze routing to integers")
    print("    07_deploy   - Run on Thor at machine speed")
    print()
    print("  That's the core of TriX: computation is geometry,")
    print("  learning is routing, deployment is compilation.")
    print()

    # -----------------------------------------------------------------
    # What's in the codebase
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  What's in the codebase:")
    print()

    # Find interesting directories
    root = Path(__file__).parent.parent

    areas = [
        ("foundry/", "The Foundry - synthesis and export tools"),
        ("foundry/export/", "Export: Python -> C, Verilog -> C"),
        ("foundry/forge/", "Forge: Verilog -> Polynomial synthesis"),
        ("foundry/mds/", "MDS: Morphogenic Deterministic System"),
        ("src/trix/", "Core TriX architecture"),
        ("experiments/", "Research experiments"),
        ("tests/", "Test suite"),
        ("rtl_reference/", "Verilog reference designs"),
    ]

    for path, desc in areas:
        full_path = root / path
        if full_path.exists():
            if full_path.is_dir():
                n_files = len(list(full_path.glob("*.py")))
                print(f"    {path:25s} - {desc}")
            else:
                print(f"    {path:25s} - {desc}")
    print()

    # -----------------------------------------------------------------
    # A few things to try
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  Things to try:")
    print()

    experiments = [
        ("See the 6502 shapes", "python -c \"from foundry.export.trix_6502 import "
         "TriX6502Exporter; print(f'{len(TriX6502Exporter.SHAPES)} shapes: ' + ', '.join(TriX6502Exporter.SHAPES[:5]) + '...')\""),
        ("See all polynomials", "python -c \"from foundry.export.dispatch_export import "
         "SHAPE_POLYNOMIALS; [print(f'{k}: {v}') for k,v in SHAPE_POLYNOMIALS.items()]\""),
        ("Run all tests", "python -m pytest tests/ -v --ignore=tests/test_viz"),
    ]

    for name, cmd in experiments:
        print(f"    {name}:")
        print(f"      $ {cmd}")
        print()

    # -----------------------------------------------------------------
    # Questions to explore
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  Questions to explore:")
    print()
    print("    - Can you build a full adder as polynomials?")
    print("    - How does the 6502 represent memory operations?")
    print("    - What if you trained a router on MNIST?")
    print("    - How would you add GPU acceleration?")
    print("    - Can shapes be learned, not just routing?")
    print()

    # -----------------------------------------------------------------
    # The philosophy
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  The philosophy:")
    print()
    print("    \"Computation is geometry.\"")
    print()
    print("    Shapes don't change. They're mathematical truths.")
    print("    XOR will always be a + b - 2ab. That's not learned.")
    print()
    print("    What changes is which shapes you use.")
    print("    That's what neural networks actually learn:")
    print("    routing, not computation.")
    print()
    print("    TriX makes that separation explicit.")
    print("    Freeze the shapes. Learn the routes.")
    print("    Deploy the result.")
    print()

    # -----------------------------------------------------------------
    # Closing thought
    # -----------------------------------------------------------------

    print("=" * 60)
    print()
    print("  The onramp ends here.")
    print()
    print("  But the road continues.")
    print()
    print("  Welcome to TriX.")
    print()
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
