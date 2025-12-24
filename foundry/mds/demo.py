#!/usr/bin/env python3
"""
MDS Demo - Morphogenic Deterministic System

Demonstrates that computation is navigation, not calculation.
The lattice contains all answers. We route to them.
"""

from lattice import MDS_Lattice, build_adder_lattice


def demo_2bit_adder():
    """Demonstrate 2-bit adder as MDS lattice."""

    print("╔" + "═" * 62 + "╗")
    print("║" + "  MORPHOGENIC DETERMINISTIC SYSTEM".center(62) + "║")
    print("║" + "  2-Bit Adder Lattice Demo".center(62) + "║")
    print("╚" + "═" * 62 + "╝")
    print()

    # Build the lattice
    print("Building lattice...")
    lattice = build_adder_lattice(2)

    print(lattice.visualize())
    print()

    # Test case: 2 + 1 = 3
    print("─" * 64)
    print("\nTEST: 2 + 1 = ?")
    print()

    a_val = 2  # binary: 10
    b_val = 1  # binary: 01

    print(f"Inputs: a={a_val} (binary: {a_val:02b}), b={b_val} (binary: {b_val:02b})")

    # Convert to individual bits
    inputs = {
        'a0': (a_val >> 0) & 1,
        'a1': (a_val >> 1) & 1,
        'b0': (b_val >> 0) & 1,
        'b1': (b_val >> 1) & 1,
    }

    # Navigate (not compute!)
    result = lattice.navigate(inputs, verbose=True)

    # Collect result
    s0 = result.get('s0', 0)
    s1 = result.get('s1', 0)
    s2 = result.get('s2', 0)
    sum_val = s0 + (s1 << 1) + (s2 << 2)

    print(f"\n  RESULT: s2={s2}, s1={s1}, s0={s0} → binary: {sum_val:03b} → decimal: {sum_val}")

    expected = a_val + b_val
    status = "✓" if sum_val == expected else "✗"
    print(f"\n  VERIFICATION: {a_val} + {b_val} = {expected} {status}")

    print()
    print("─" * 64)
    print()
    print("The answer was not COMPUTED. It was NAVIGATED.")
    print("The lattice contains all 16 possible additions.")
    print(f"We routed to location (a={a_val}, b={b_val}) and read the answer.")
    print()

    return sum_val == expected


def verify_all_cases(bits: int):
    """Verify all possible input combinations."""

    print(f"\n{'═' * 64}")
    print(f"VERIFYING ALL {2**(2*bits)} CASES FOR {bits}-BIT ADDER")
    print('═' * 64)

    lattice = build_adder_lattice(bits)
    print(f"Lattice nodes: {lattice.node_count()}")
    print(f"Lookup table would need: {2**(2*bits)} entries")
    print(f"Compression: {100 * (1 - lattice.node_count() / 2**(2*bits)):.1f}%")
    print()

    passed = 0
    failed = 0

    max_val = 2 ** bits

    for a_val in range(max_val):
        for b_val in range(max_val):
            # Build inputs
            inputs = {}
            for i in range(bits):
                inputs[f'a{i}'] = (a_val >> i) & 1
                inputs[f'b{i}'] = (b_val >> i) & 1

            # Navigate
            result = lattice.navigate(inputs, verbose=False)

            # Collect result
            sum_val = 0
            for i in range(bits + 1):
                sum_val += result.get(f's{i}', 0) << i

            expected = a_val + b_val

            if sum_val == expected:
                passed += 1
            else:
                failed += 1
                print(f"  FAIL: {a_val} + {b_val} = {sum_val} (expected {expected})")

    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print(f"\n✓ All {passed} cases verified by NAVIGATION, not computation.")

    return failed == 0


def show_scaling():
    """Show how node count scales with bit width."""

    print(f"\n{'═' * 64}")
    print("SCALING: Lattice Nodes vs Lookup Table Entries")
    print('═' * 64)
    print()
    print(f"{'Bits':>6} │ {'Nodes':>10} │ {'Lookup':>15} │ {'Ratio':>10}")
    print("─" * 50)

    for bits in [2, 4, 8, 16, 32, 64]:
        lattice = build_adder_lattice(bits)
        nodes = lattice.node_count()
        lookup = 2 ** (2 * bits)

        if lookup > 10**15:
            lookup_str = f"2^{2*bits}"
        else:
            lookup_str = f"{lookup:,}"

        if lookup > nodes:
            ratio = lookup / nodes
            if ratio > 10**6:
                ratio_str = f"{ratio:.0e}x"
            else:
                ratio_str = f"{ratio:,.0f}x"
        else:
            ratio_str = "N/A"

        print(f"{bits:>6} │ {nodes:>10,} │ {lookup_str:>15} │ {ratio_str:>10}")

    print()
    print("The lattice scales O(n). Lookup scales O(2^n).")
    print("At 64 bits, the lattice is 10^35x smaller than a lookup table.")


def main():
    """Run the full demo."""

    # Main demo
    demo_2bit_adder()

    # Verify all cases
    verify_all_cases(2)
    verify_all_cases(4)

    # Show scaling
    show_scaling()

    print()
    print("╔" + "═" * 62 + "╗")
    print("║" + "  \"The shapes exist. We just find them.\"".center(62) + "║")
    print("╚" + "═" * 62 + "╝")


if __name__ == "__main__":
    main()
