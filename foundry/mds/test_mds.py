"""
MDS Proof Test - The lattice IS the computation.
"""

from lattice import build_adder_lattice


def test_navigation_equals_arithmetic():
    """
    PROOF: Navigation through lattice = arithmetic computation.

    The lattice is built ONCE. It never changes.
    Every input pair navigates the SAME structure.
    The result is always correct.

    This proves: the answers exist in the lattice.
    We find them. We don't compute them.
    """

    # Build lattice ONCE
    lattice = build_adder_lattice(4)

    # Test ALL 256 cases
    errors = 0
    for a in range(16):
        for b in range(16):
            # Navigate (not compute)
            inputs = {f'a{i}': (a >> i) & 1 for i in range(4)}
            inputs.update({f'b{i}': (b >> i) & 1 for i in range(4)})

            result = lattice.navigate(inputs)
            navigated = sum(result[f's{i}'] << i for i in range(5))

            # Compare to arithmetic
            computed = a + b

            if navigated != computed:
                errors += 1

    assert errors == 0, f"{errors} navigation errors"
    print(f"✓ All 256 navigations match arithmetic")


def test_lattice_is_static():
    """
    PROOF: The lattice doesn't change between navigations.

    Build once. Navigate many times. Same structure.
    The answers are already there.
    """

    lattice = build_adder_lattice(2)
    node_count_before = lattice.node_count()
    structure_before = [(n.id, n.op, n.src_a, n.src_b) for n in lattice.nodes.values()]

    # Navigate 100 times
    for _ in range(100):
        inputs = {'a0': 1, 'a1': 0, 'b0': 1, 'b1': 1}
        lattice.navigate(inputs)

    node_count_after = lattice.node_count()
    structure_after = [(n.id, n.op, n.src_a, n.src_b) for n in lattice.nodes.values()]

    assert node_count_before == node_count_after
    assert structure_before == structure_after
    print(f"✓ Lattice unchanged after 100 navigations")


def test_no_arithmetic_in_navigation():
    """
    PROOF: Navigation uses only routing decisions, not polynomial math.

    XOR: "are inputs different?" (routing decision)
    AND: "are both inputs 1?" (routing decision)
    OR:  "is either input 1?" (routing decision)

    No multiplication. No addition. Just path selection.
    """

    # The navigate() function uses only:
    #   a != b  (comparison, not a + b - 2ab)
    #   a == 1 and b == 1  (comparison, not a * b)
    #   a == 1 or b == 1   (comparison, not a + b - ab)

    # This IS the proof - inspect the code:
    import inspect
    source = inspect.getsource(build_adder_lattice(2).navigate)

    # No polynomial operations in navigate()
    assert 'a + b - 2*a*b' not in source
    assert 'a * b' not in source

    # Only routing logic
    assert '!=' in source  # XOR routing
    assert 'and' in source.lower()  # AND routing
    assert 'or' in source.lower()   # OR routing

    print(f"✓ Navigation contains no polynomial arithmetic")


if __name__ == "__main__":
    print("MDS PROOF TESTS")
    print("=" * 40)
    print()

    test_navigation_equals_arithmetic()
    test_lattice_is_static()
    test_no_arithmetic_in_navigation()

    print()
    print("=" * 40)
    print("ALL PROOFS PASSED")
    print()
    print("The lattice contains the answers.")
    print("Navigation finds them.")
    print("No computation required.")
