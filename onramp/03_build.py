#!/usr/bin/env python3
"""
03_build.py - Create a Shape

You've seen XOR, AND, OR, NOT.
Now build NAND from first principles.
"""
import sys


def wait_for_user():
    """Wait for user input in interactive mode, skip otherwise."""
    if sys.stdin.isatty():
        wait_for_user()
    else:
        print("  (Answers below)")


def main():
    print()
    print("=" * 60)
    print("  Create a Shape: NAND")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------
    # What is NAND?
    # -----------------------------------------------------------------

    print("  NAND is 'NOT AND'.")
    print()
    print("    NAND(a, b) = NOT(AND(a, b))")
    print()
    print("  We know:")
    print("    AND(a, b) = ab")
    print("    NOT(x)    = 1 - x")
    print()
    print("  So:")
    print("    NAND(a, b) = 1 - AND(a, b)")
    print("              = 1 - ab")
    print()

    # -----------------------------------------------------------------
    # Build it
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  Let's build it:")
    print()
    print("    def nand(a, b):")
    print("        return 1 - a * b")
    print()

    def nand(a, b):
        return 1 - a * b

    # -----------------------------------------------------------------
    # Verify it
    # -----------------------------------------------------------------

    print("  Verify against truth table:")
    print()
    print("    a  b  │ NAND │ 1-ab │ ✓?")
    print("   ───────┼──────┼──────┼────")

    all_correct = True
    for a in [0, 1]:
        for b in [0, 1]:
            # Python doesn't have NAND, so we compute it
            expected = 1 if not (a and b) else 0
            result = nand(a, b)
            check = "✓" if result == expected else "✗"
            print(f"    {a}  {b}  │   {expected}  │   {result}  │  {check}")
            if result != expected:
                all_correct = False

    print()
    if all_correct:
        print("  All cases match. You built NAND.")
    print()

    # -----------------------------------------------------------------
    # Why NAND matters
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  NAND is special.")
    print()
    print("  It's 'functionally complete' - you can build ANY logic")
    print("  gate using only NAND:")
    print()
    print("    NOT(a)    = NAND(a, a)")
    print("    AND(a, b) = NOT(NAND(a, b)) = NAND(NAND(a,b), NAND(a,b))")
    print("    OR(a, b)  = NAND(NOT(a), NOT(b))")
    print()
    print("  Every digital computer is NAND gates, all the way down.")
    print("  Now you have NAND as a polynomial: 1 - ab")
    print()

    # -----------------------------------------------------------------
    # Your turn
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  Try building these yourself (answers below):")
    print()
    print("    NOR(a, b)  = NOT(OR(a, b)) = ?")
    print("    XNOR(a, b) = NOT(XOR(a, b)) = ?")
    print()
    wait_for_user()
    print()
    print("    NOR(a, b)  = 1 - (a + b - ab) = 1 - a - b + ab")
    print("    XNOR(a, b) = 1 - (a + b - 2ab) = 1 - a - b + 2ab")
    print()

    # Verify NOR
    def nor(a, b):
        return 1 - a - b + a * b

    print("  Let's verify NOR:")
    print()
    for a in [0, 1]:
        for b in [0, 1]:
            expected = 1 if not (a or b) else 0
            result = nor(a, b)
            check = "✓" if result == expected else "✗"
            print(f"    NOR({a}, {b}) = {result}  (expected {expected}) {check}")

    print()

    # -----------------------------------------------------------------
    # What's next
    # -----------------------------------------------------------------

    print("=" * 60)
    print()
    print("  You can build any logic gate as a polynomial.")
    print("  Now let's combine them into something useful.")
    print()
    print("  Run:  python onramp/04_compose.py")
    print()


if __name__ == "__main__":
    main()
