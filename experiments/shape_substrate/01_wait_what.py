"""
Wait, what?

Run this. Don't read the theory first. Just run it.

    python 01_wait_what.py

Then ask yourself: where's the XOR instruction?
"""

# XOR without XOR
def xor(a, b):
    return a + b - 2*a*b

# Test it
print("XOR without using XOR:")
print(f"  0 XOR 0 = {int(xor(0, 0))}")
print(f"  0 XOR 1 = {int(xor(0, 1))}")
print(f"  1 XOR 0 = {int(xor(1, 0))}")
print(f"  1 XOR 1 = {int(xor(1, 1))}")

print()
print("Check: does it match?")
print(f"  0 ^ 0 = {0 ^ 0}")
print(f"  0 ^ 1 = {0 ^ 1}")
print(f"  1 ^ 0 = {1 ^ 0}")
print(f"  1 ^ 1 = {1 ^ 1}")

print()
print("The function a + b - 2ab computes XOR.")
print("No bitwise operations. Just add, subtract, multiply.")
print()
print("Why does this matter? Keep going: python 02_so_what.py")
