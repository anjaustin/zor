"""
Why is this fast?

The polynomial is for TRAINING (gradients).
Once trained, you EXPORT to native operations.

    python 04_why_fast.py
"""

import time

# Polynomial XOR (for training)
def xor_poly(a, b):
    return a + b - 2*a*b

# Native XOR (for inference)
def xor_native(a, b):
    return a ^ b

# Let's race them
N = 1_000_000

print(f"Racing {N:,} XOR operations...")
print()

# Polynomial version
start = time.perf_counter()
for _ in range(N):
    _ = xor_poly(1, 0)
poly_time = time.perf_counter() - start

# Native version
start = time.perf_counter()
for _ in range(N):
    _ = xor_native(1, 0)
native_time = time.perf_counter() - start

print(f"Polynomial XOR: {poly_time*1000:.1f} ms")
print(f"Native XOR:     {native_time*1000:.1f} ms")
print(f"Speedup:        {poly_time/native_time:.1f}x")
print()

print("The polynomial is slower. That's fine.")
print("You use it during TRAINING when you need gradients.")
print("You use native XOR during INFERENCE when you need speed.")
print()
print("The insight:")
print("  TRAIN: polynomial (slow, but gradients work)")
print("  INFER: native (fast, no gradients needed)")
print()
print("Both compute the same thing on binary inputs.")
print("You get the best of both worlds.")
print()
print("Next: python 05_the_trick.py")
