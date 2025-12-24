"""
The trick: shapes compose.

Once you have XOR, AND, OR as polynomials,
you can build ANYTHING by composing them.

    python 05_the_trick.py
"""

# The basic shapes
def XOR(a, b): return a + b - 2*a*b
def AND(a, b): return a * b
def OR(a, b):  return a + b - a*b
def NOT(a):    return 1 - a

# Compose them into new shapes
def NAND(a, b): return NOT(AND(a, b))
def NOR(a, b):  return NOT(OR(a, b))
def XNOR(a, b): return NOT(XOR(a, b))

# A multiplexer: if sel=0, output a; if sel=1, output b
def MUX(a, b, sel):
    return OR(AND(a, NOT(sel)), AND(b, sel))

# Test the mux
print("Multiplexer (MUX):")
print("  sel=0: output = a")
print("  sel=1: output = b")
print()
print("  a  b  sel | out")
print("  ----------|----")
for a in [0, 1]:
    for b in [0, 1]:
        for sel in [0, 1]:
            out = MUX(a, b, sel)
            print(f"  {a}  {b}   {sel}  |  {int(out)}")

print()

# A 4-to-1 selector
def MUX4(inputs, sel0, sel1):
    """Select one of 4 inputs based on 2 select bits"""
    a, b, c, d = inputs
    low = MUX(a, b, sel0)   # sel0 picks a or b
    high = MUX(c, d, sel0)  # sel0 picks c or d
    return MUX(low, high, sel1)  # sel1 picks between results

print("4-to-1 Multiplexer:")
print("  inputs = [A, B, C, D]")
print("  sel = 0,0 -> A; 0,1 -> B; 1,0 -> C; 1,1 -> D")
print()

inputs = [1, 0, 1, 0]  # A=1, B=0, C=1, D=0
for sel1 in [0, 1]:
    for sel0 in [0, 1]:
        out = MUX4(inputs, sel0, sel1)
        idx = sel1 * 2 + sel0
        label = ['A', 'B', 'C', 'D'][idx]
        print(f"  sel={sel1},{sel0} -> {label} = {int(out)}")

print()
print("You just built a 4-to-1 multiplexer from polynomials.")
print("No conditionals. No branches. Just math.")
print()
print("This is the key insight:")
print("  - ANY digital circuit can be built from these shapes")
print("  - The shapes are polynomials (trainable)")
print("  - The shapes compose (no limits)")
print()
print("Next: python 06_what_is_a_shape.py")
