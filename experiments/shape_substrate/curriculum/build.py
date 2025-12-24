"""
BUILD

You're going to build an 8-bit adder. The thing inside every CPU.
Using only +, -, *
"""

# Your building blocks (from discover.py)
def XOR(a, b): return a + b - 2*a*b
def AND(a, b): return a * b
def OR(a, b):  return a + b - a*b


# ============================================================
# STEP 1: Build a half adder
# ============================================================
#
# A half adder takes 2 bits, returns (sum, carry)
#   0 + 0 = 0, carry 0
#   0 + 1 = 1, carry 0
#   1 + 0 = 1, carry 0
#   1 + 1 = 0, carry 1  <-- this is the tricky one

def half_adder(a, b):
    sum_bit = XOR(a, b)
    carry = AND(a, b)
    return int(sum_bit), int(carry)

# Test it
print("Half Adder:")
for a in [0, 1]:
    for b in [0, 1]:
        s, c = half_adder(a, b)
        print(f"  {a} + {b} = {s}, carry {c}")
print()


# ============================================================
# STEP 2: Build a full adder
# ============================================================
#
# A full adder takes 3 bits (a, b, carry_in), returns (sum, carry_out)
# This is YOUR job. Fill in the function.

def full_adder(a, b, carry_in):
    # HINT: Use two half adders and an OR
    #
    # First half adder: add a and b
    # Second half adder: add that result to carry_in
    # carry_out = OR of both carries
    #
    # YOUR CODE HERE:
    pass  # Replace this


# Test harness - run this to check your work
def test_full_adder():
    print("Full Adder Test:")
    all_pass = True
    for a in [0, 1]:
        for b in [0, 1]:
            for cin in [0, 1]:
                result = full_adder(a, b, cin)
                if result is None:
                    print("  (implement full_adder first)")
                    return
                s, cout = result
                expected_s = (a + b + cin) % 2
                expected_c = (a + b + cin) // 2
                ok = (s == expected_s and cout == expected_c)
                all_pass = all_pass and ok
                status = "✓" if ok else "✗"
                print(f"  {a}+{b}+{cin} = {s}, carry {cout} {status}")
    if all_pass:
        print("  ALL TESTS PASSED!")
    print()

test_full_adder()


# ============================================================
# STEP 3: Chain into 8-bit adder
# ============================================================
#
# Once full_adder works, chain 8 of them together.
# Bit 0's carry_out becomes bit 1's carry_in, etc.

def add_8bit(a_bits, b_bits):
    """
    Add two 8-bit numbers.
    a_bits and b_bits are lists of 8 bits, LSB first.
    Returns (result_bits, overflow)
    """
    # YOUR CODE HERE:
    pass  # Replace this


# Test it
def test_8bit():
    print("8-bit Adder Test:")

    # Helper: convert int to bits (LSB first)
    def to_bits(n):
        return [(n >> i) & 1 for i in range(8)]

    # Helper: convert bits to int
    def from_bits(bits):
        return sum(b << i for i, b in enumerate(bits))

    tests = [(42, 17), (100, 55), (255, 1), (0, 0), (127, 128)]

    for a, b in tests:
        result = add_8bit(to_bits(a), to_bits(b))
        if result is None:
            print("  (implement add_8bit first)")
            return
        sum_bits, overflow = result
        got = from_bits(sum_bits)
        expected = (a + b) % 256
        status = "✓" if got == expected else "✗"
        print(f"  {a:3d} + {b:3d} = {got:3d} (expected {expected:3d}) {status}")
    print()

test_8bit()


# ============================================================
# CHALLENGE
# ============================================================
#
# 1. Make a 16-bit adder
# 2. Add a subtract mode (hint: invert b and set carry_in = 1)
# 3. Measure: how many operations per addition?


# ============================================================
# YOU GOT IT WHEN
# ============================================================
#
# You can explain how a CPU adds numbers using only AND, OR, XOR.
# You built the same circuit, but with polynomials.


# ============================================================
# SOLUTION (don't peek until you've tried)
# ============================================================
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# def full_adder(a, b, carry_in):
#     s1, c1 = half_adder(a, b)
#     s2, c2 = half_adder(s1, carry_in)
#     return s2, int(OR(c1, c2))
#
#
# def add_8bit(a_bits, b_bits):
#     result = []
#     carry = 0
#     for i in range(8):
#         s, carry = full_adder(a_bits[i], b_bits[i], carry)
#         result.append(s)
#     return result, carry
#
# Now go to: python compose.py
