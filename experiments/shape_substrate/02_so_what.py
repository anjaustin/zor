"""
So what? Why does XOR-as-math matter?

Run this:
    python 02_so_what.py
"""

# Here's the thing: math has gradients. XOR doesn't.

def xor_polynomial(a, b):
    """XOR as a polynomial: a + b - 2ab"""
    return a + b - 2*a*b

# Watch what happens with values BETWEEN 0 and 1
print("XOR with fuzzy inputs (values between 0 and 1):")
print()

inputs = [
    (0.0, 0.0),
    (0.0, 1.0),
    (0.5, 0.5),  # <-- what happens here?
    (0.3, 0.7),
    (0.9, 0.1),
    (1.0, 1.0),
]

for a, b in inputs:
    result = xor_polynomial(a, b)
    print(f"  xor({a:.1f}, {b:.1f}) = {result:.2f}")

print()
print("Notice: 0.5 XOR 0.5 = 0.5")
print("        0.3 XOR 0.7 = 0.58")
print()
print("The polynomial INTERPOLATES between the corners.")
print("It's smooth. It has derivatives. You can train it with gradient descent.")
print()
print("This is how you teach a neural network to XOR:")
print("  - Use the polynomial during training (gradients flow)")
print("  - Use native XOR at runtime (fast)")
print()
print("Same result. Different path. Next: python 03_build_something.py")
