"""
Shapes compose like LEGO.

Build small shapes. Snap them together. Get bigger shapes.

    python 07_shapes_compose.py
"""

# Basic shapes
def XOR(a, b): return a + b - 2*a*b
def AND(a, b): return a * b

# Compose: what if we XOR something, then AND it with something else?
def shape_1(a, b, c):
    """XOR a,b then AND with c"""
    return AND(XOR(a, b), c)

# Compose differently: AND first, then XOR
def shape_2(a, b, c):
    """AND a,b then XOR with c"""
    return XOR(AND(a, b), c)

# Test all combinations
print("Two composed shapes with different behaviors:")
print()
print("Shape 1: AND(XOR(a,b), c)")
print("Shape 2: XOR(AND(a,b), c)")
print()
print("  a  b  c | Shape1  Shape2")
print("  --------|---------------")

for a in [0, 1]:
    for b in [0, 1]:
        for c in [0, 1]:
            s1 = shape_1(a, b, c)
            s2 = shape_2(a, b, c)
            print(f"  {a}  {b}  {c} |   {int(s1)}       {int(s2)}")

print()
print("Same ingredients (XOR, AND). Different arrangement. Different behavior.")
print()

# The analogy
print("Think of it like chemistry:")
print()
print("  ATOMS:")
print("    XOR = Hydrogen")
print("    AND = Oxygen")
print()
print("  MOLECULES:")
print("    AND(XOR(a,b), c) = H2O (water)")
print("    XOR(AND(a,b), c) = H2O2 (hydrogen peroxide)")
print()
print("Same atoms. Different structure. Different properties.")
print()
print("This is why we call them 'shapes':")
print("  - The STRUCTURE determines the FUNCTION")
print("  - Compose shapes → get new shapes")
print("  - No retraining needed")
print()
print("Next: python 08_routing.py")
