#!/usr/bin/env python3
"""
04_compose.py - Build an ALU

Combine shapes into an Arithmetic Logic Unit.
Four operations. One dispatch.
"""

import numpy as np


def main():
    print()
    print("=" * 60)
    print("  Build an ALU")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------
    # Define the shapes
    # -----------------------------------------------------------------

    print("  Our toolkit - four frozen shapes:")
    print()
    print("    add(a, b) = (a + b) & 0xFF    # 8-bit addition")
    print("    sub(a, b) = (a - b) & 0xFF    # 8-bit subtraction")
    print("    xor(a, b) = a ^ b             # bitwise XOR")
    print("    and_(a, b) = a & b            # bitwise AND")
    print()

    def add(a, b):
        return (a + b) & 0xFF

    def sub(a, b):
        return (a - b) & 0xFF

    def xor(a, b):
        return a ^ b

    def and_(a, b):
        return a & b

    # -----------------------------------------------------------------
    # Map opcodes to shapes
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  An ALU is just: opcode → shape")
    print()
    print("    opcode 0 → add")
    print("    opcode 1 → sub")
    print("    opcode 2 → xor")
    print("    opcode 3 → and")
    print()

    shapes = [add, sub, xor, and_]
    shape_names = ['add', 'sub', 'xor', 'and']

    def alu(opcode, a, b):
        """Execute the shape selected by opcode."""
        return shapes[opcode](a, b)

    # -----------------------------------------------------------------
    # Test it
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  Let's test it:")
    print()

    tests = [
        (0, 10, 20, "add(10, 20)"),
        (0, 200, 100, "add(200, 100) [overflow]"),
        (1, 50, 20, "sub(50, 20)"),
        (1, 10, 20, "sub(10, 20) [underflow]"),
        (2, 0b10101010, 0b01010101, "xor(0xAA, 0x55)"),
        (3, 0b11111111, 0b00001111, "and(0xFF, 0x0F)"),
    ]

    all_correct = True
    for opcode, a, b, desc in tests:
        result = alu(opcode, a, b)

        # Expected
        if opcode == 0:
            expected = (a + b) & 0xFF
        elif opcode == 1:
            expected = (a - b) & 0xFF
        elif opcode == 2:
            expected = a ^ b
        else:
            expected = a & b

        check = "✓" if result == expected else "✗"
        print(f"    alu({opcode}, ...) → {desc:25s} = {result:3d} (0x{result:02X}) {check}")

        if result != expected:
            all_correct = False

    print()
    if all_correct:
        print("  All operations correct.")
    print()

    # -----------------------------------------------------------------
    # What we built
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  What we just built:")
    print()
    print("    ┌─────────┐")
    print("    │ opcode  │───┐")
    print("    └─────────┘   │")
    print("                  ▼")
    print("    ┌─────────┐  ┌─────────────────────┐")
    print("    │    a    │──│                     │")
    print("    └─────────┘  │      DISPATCH       │──▶ result")
    print("    ┌─────────┐  │   ┌───┬───┬───┬───┐ │")
    print("    │    b    │──│   │add│sub│xor│and│ │")
    print("    └─────────┘  │   └───┴───┴───┴───┘ │")
    print("                  └─────────────────────┘")
    print()
    print("  The opcode selects the shape. The shape computes.")
    print("  This is 'fixed routing' - the mapping is hardcoded.")
    print()

    # -----------------------------------------------------------------
    # The question
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  But what if we didn't know the opcode?")
    print()
    print("  What if the INPUT determined which shape to use?")
    print()
    print("  That's 'learned routing' - the neural part of TriX.")
    print()

    # -----------------------------------------------------------------
    # What's next
    # -----------------------------------------------------------------

    print("=" * 60)
    print()
    print("  So far, everything has been fixed.")
    print("  Frozen shapes. Fixed dispatch.")
    print()
    print("  Now let's add learning.")
    print()
    print("  Run:  python onramp/05_route.py")
    print()


if __name__ == "__main__":
    main()
