#!/usr/bin/env python3
"""
02_truth.py - Verify the Math

We claimed XOR is a polynomial.
Let's prove it. For XOR. For AND. For OR. For all of them.
"""


def main():
    print()
    print("=" * 60)
    print("  Verify the Math")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------
    # The four fundamental shapes
    # -----------------------------------------------------------------

    # These are the polynomials. No imports. No libraries.
    # Just math you can verify by hand.

    def xor_poly(a, b):
        return a + b - 2 * a * b

    def and_poly(a, b):
        return a * b

    def or_poly(a, b):
        return a + b - a * b

    def not_poly(a):
        return 1 - a

    # -----------------------------------------------------------------
    # Verify XOR
    # -----------------------------------------------------------------

    print("  XOR(a, b) = a + b - 2ab")
    print()

    for a in [0, 1]:
        for b in [0, 1]:
            poly = xor_poly(a, b)
            bits = a ^ b
            check = "✓" if poly == bits else "✗"
            # Show the arithmetic
            calc = f"{a} + {b} - 2×{a}×{b} = {a + b} - {2*a*b} = {poly}"
            print(f"    {a} XOR {b}:  {calc}  (= {bits}) {check}")

    print()

    # -----------------------------------------------------------------
    # Verify AND
    # -----------------------------------------------------------------

    print("  AND(a, b) = ab")
    print()

    for a in [0, 1]:
        for b in [0, 1]:
            poly = and_poly(a, b)
            bits = a & b
            check = "✓" if poly == bits else "✗"
            calc = f"{a} × {b} = {poly}"
            print(f"    {a} AND {b}:  {calc}  (= {bits}) {check}")

    print()

    # -----------------------------------------------------------------
    # Verify OR
    # -----------------------------------------------------------------

    print("  OR(a, b) = a + b - ab")
    print()

    for a in [0, 1]:
        for b in [0, 1]:
            poly = or_poly(a, b)
            bits = a | b
            check = "✓" if poly == bits else "✗"
            calc = f"{a} + {b} - {a}×{b} = {a + b} - {a*b} = {poly}"
            print(f"    {a} OR {b}:   {calc}  (= {bits}) {check}")

    print()

    # -----------------------------------------------------------------
    # Verify NOT
    # -----------------------------------------------------------------

    print("  NOT(a) = 1 - a")
    print()

    for a in [0, 1]:
        poly = not_poly(a)
        bits = 1 - a  # Same as ~a & 1 for single bits
        check = "✓" if poly == bits else "✗"
        calc = f"1 - {a} = {poly}"
        print(f"    NOT {a}:    {calc}  (= {bits}) {check}")

    print()

    # -----------------------------------------------------------------
    # The insight
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  Every case matches. This isn't coincidence.")
    print()
    print("  Boolean algebra over {0, 1} is isomorphic to")
    print("  polynomial arithmetic over the same domain.")
    print()
    print("  The mapping:")
    print()
    print("    XOR  →  a + b - 2ab")
    print("    AND  →  ab")
    print("    OR   →  a + b - ab")
    print("    NOT  →  1 - a")
    print()
    print("  This is mathematics. It's always true.")
    print()

    # -----------------------------------------------------------------
    # Derived operations
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  From these four, we can derive everything:")
    print()
    print("    NAND(a, b) = NOT(AND(a, b)) = 1 - ab")
    print("    NOR(a, b)  = NOT(OR(a, b))  = 1 - a - b + ab")
    print("    XNOR(a, b) = NOT(XOR(a, b)) = 1 - a - b + 2ab")
    print()
    print("  Every logic gate is a polynomial.")
    print("  Every digital circuit is polynomial composition.")
    print()

    # -----------------------------------------------------------------
    # What's next
    # -----------------------------------------------------------------

    print("=" * 60)
    print()
    print("  You've verified the foundation.")
    print("  Now let's build something.")
    print()
    print("  Run:  python onramp/03_build.py")
    print()


if __name__ == "__main__":
    main()
