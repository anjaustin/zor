"""
Let's build something real.

A full adder. The thing inside every CPU that adds numbers.

    python 03_build_something.py
"""

# Our polynomial building blocks
def XOR(a, b): return a + b - 2*a*b
def AND(a, b): return a * b
def OR(a, b):  return a + b - a*b

# A full adder: adds three bits (a, b, carry_in) -> (sum, carry_out)
def full_adder(a, b, cin):
    # This is the same circuit as hardware, but with polynomials
    p = XOR(a, b)
    sum_bit = XOR(p, cin)
    carry = OR(AND(a, b), AND(p, cin))
    return sum_bit, carry

# Test it
print("Full Adder (built from polynomial shapes):")
print()
print("  a  b cin | sum cout")
print("  ---------|----------")

for a in [0, 1]:
    for b in [0, 1]:
        for cin in [0, 1]:
            s, cout = full_adder(a, b, cin)
            print(f"  {a}  {b}  {cin}  |  {int(s)}   {int(cout)}")

print()
print("That's a real full adder. No tricks.")
print()

# Now chain 8 of them to make an 8-bit adder
def add_8bit(a_bits, b_bits):
    """Add two 8-bit numbers (LSB first)"""
    result = []
    carry = 0
    for i in range(8):
        s, carry = full_adder(a_bits[i], b_bits[i], carry)
        result.append(int(s))
    return result, int(carry)

# Convert number to bits (LSB first)
def to_bits(n, width=8):
    return [(n >> i) & 1 for i in range(width)]

def from_bits(bits):
    return sum(b << i for i, b in enumerate(bits))

# Test: 42 + 17 = 59
a, b = 42, 17
a_bits = to_bits(a)
b_bits = to_bits(b)
sum_bits, overflow = add_8bit(a_bits, b_bits)
result = from_bits(sum_bits)

print(f"8-bit addition: {a} + {b} = {result}")
print(f"Correct? {result == a + b}")
print()

# Try a few more
tests = [(100, 55), (255, 1), (0, 0), (127, 128)]
print("More tests:")
for a, b in tests:
    sum_bits, overflow = add_8bit(to_bits(a), to_bits(b))
    result = from_bits(sum_bits)
    expected = (a + b) % 256
    status = "✓" if result == expected else "✗"
    print(f"  {a:3d} + {b:3d} = {result:3d} {status}")

print()
print("A complete 8-bit adder, built from polynomials.")
print("No binary operators. Just +, -, *.")
print()
print("Next: python 04_why_fast.py")
