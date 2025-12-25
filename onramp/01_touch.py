#!/usr/bin/env python3
"""
01_touch.py - First Contact

Run a single XOR. See that it works.
Then see what it really is.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    print()
    print("=" * 60)
    print("  First Contact: XOR")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------
    # Step 1: Just run it
    # -----------------------------------------------------------------

    print("  Let's run XOR:")
    print()

    # The simplest possible call
    def xor(a, b):
        """XOR as a polynomial."""
        return a + b - 2 * a * b

    # Try it
    result = xor(1, 1)
    print(f"    xor(1, 1) = {result}")
    print()

    # Check against Python's built-in XOR
    expected = 1 ^ 1
    print(f"    1 ^ 1     = {expected}")
    print()

    if result == expected:
        print("    They match.")
    print()

    # -----------------------------------------------------------------
    # Step 2: What is it really?
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  But what IS that function?")
    print()
    print("  Look at the code:")
    print()
    print("    def xor(a, b):")
    print("        return a + b - 2 * a * b")
    print()
    print("  That's not bit manipulation.")
    print("  That's a polynomial.")
    print()
    print("  XOR(a, b) = a + b - 2ab")
    print()

    # -----------------------------------------------------------------
    # Step 3: Does it always work?
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  Does it always work? Let's check all 4 cases:")
    print()

    all_correct = True
    for a in [0, 1]:
        for b in [0, 1]:
            poly_result = xor(a, b)
            bit_result = a ^ b
            match = "✓" if poly_result == bit_result else "✗"
            print(f"    xor({a}, {b}) = {poly_result}   (expected {bit_result}) {match}")
            if poly_result != bit_result:
                all_correct = False

    print()
    if all_correct:
        print("  All 4 cases match. Exactly.")
    print()

    # -----------------------------------------------------------------
    # Step 4: The insight
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  This is the core insight of TriX:")
    print()
    print("    Boolean logic IS polynomial arithmetic.")
    print()
    print("  XOR isn't 'like' a polynomial.")
    print("  XOR IS a polynomial.")
    print()
    print("  When a and b are in {0, 1}:")
    print()
    print("    a + b - 2ab  =  a ⊕ b")
    print()
    print("  Every time. Not approximately. Exactly.")
    print()

    # -----------------------------------------------------------------
    # Step 5: What's next
    # -----------------------------------------------------------------

    print("=" * 60)
    print()
    print("  That was just XOR.")
    print("  AND, OR, NOT - they're all polynomials too.")
    print()
    print("  Want to see the proof?")
    print()
    print("  Run:  python onramp/02_truth.py")
    print()


if __name__ == "__main__":
    main()
