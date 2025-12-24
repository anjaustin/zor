"""
What is a "shape"?

It's not philosophy. It's a function that looks like a surface.

    python 06_what_is_a_shape.py
"""

# XOR as a polynomial
def xor(a, b):
    return a + b - 2*a*b

# Let's visualize it as a surface
print("XOR as a surface (z = a + b - 2ab):")
print()
print("      b=0.0  b=0.5  b=1.0")
print("     ┌──────┬──────┬──────┐")

for a in [0.0, 0.5, 1.0]:
    row = f"a={a:.1f}│"
    for b in [0.0, 0.5, 1.0]:
        z = xor(a, b)
        row += f" {z:4.2f} │"
    print(row)
    if a < 1.0:
        print("     ├──────┼──────┼──────┤")
    else:
        print("     └──────┴──────┴──────┘")

print()
print("The corners (0,0), (0,1), (1,0), (1,1) are the XOR truth table.")
print("The middle is where the 'shape' interpolates.")
print()

# Compare AND and OR
def and_shape(a, b): return a * b
def or_shape(a, b): return a + b - a*b

print("AND as a surface (z = ab):")
print()
print("      b=0.0  b=0.5  b=1.0")
print("     ┌──────┬──────┬──────┐")
for a in [0.0, 0.5, 1.0]:
    row = f"a={a:.1f}│"
    for b in [0.0, 0.5, 1.0]:
        z = and_shape(a, b)
        row += f" {z:4.2f} │"
    print(row)
    if a < 1.0:
        print("     ├──────┼──────┼──────┤")
    else:
        print("     └──────┴──────┴──────┘")

print()
print("Notice: AND(0.5, 0.5) = 0.25")
print("        XOR(0.5, 0.5) = 0.50")
print()
print("Each function has a different SHAPE.")
print("The shape determines the computation.")
print("The shape IS the computation.")
print()
print("This isn't metaphor. It's literally how it works:")
print("  - The polynomial defines a surface")
print("  - The surface has a shape")
print("  - Evaluating the polynomial = sampling the shape")
print()
print("Next: python 07_shapes_compose.py")
