"""
COMPOSE

Shapes snap together like LEGO.
Same pieces, different structures, different behaviors.
"""

def XOR(a, b): return a + b - 2*a*b
def AND(a, b): return a * b
def OR(a, b):  return a + b - a*b
def NOT(a):    return 1 - a


# ============================================================
# DISCOVER: What does this do?
# ============================================================

def mystery(a, b, sel):
    return OR(AND(a, NOT(sel)), AND(b, sel))

# Run these and figure out what mystery() does:
print("mystery(0, 1, 0) =", int(mystery(0, 1, 0)))
print("mystery(0, 1, 1) =", int(mystery(0, 1, 1)))
print("mystery(7, 3, 0) =", int(mystery(7, 3, 0)))  # works with any values!
print("mystery(7, 3, 1) =", int(mystery(7, 3, 1)))
print()

# What is this function called? (Answer before scrolling)


# ============================================================
# TRY THESE
# ============================================================
#
# 1. Print the full truth table for mystery(a, b, sel)
#    with a, b, sel each in [0, 1]
#
# 2. Swap a and b in the formula. Run again. What changes?
#
# 3. What happens if sel = 0.5? (Try it!)


# ============================================================
# CHALLENGE 1: Build a 4-to-1 multiplexer
# ============================================================
#
# mux4(inputs, sel0, sel1) should return:
#   inputs[0] if sel1=0, sel0=0
#   inputs[1] if sel1=0, sel0=1
#   inputs[2] if sel1=1, sel0=0
#   inputs[3] if sel1=1, sel0=1
#
# Use mystery() as your building block.

def mux4(inputs, sel0, sel1):
    # YOUR CODE HERE
    pass


def test_mux4():
    print("4-to-1 MUX Test:")
    inputs = [10, 20, 30, 40]  # Test with recognizable values

    for s1 in [0, 1]:
        for s0 in [0, 1]:
            result = mux4(inputs, s0, s1)
            if result is None:
                print("  (implement mux4 first)")
                return
            idx = s1 * 2 + s0
            expected = inputs[idx]
            status = "✓" if result == expected else "✗"
            print(f"  sel={s1},{s0} -> {result} (expected {expected}) {status}")
    print()

test_mux4()


# ============================================================
# CHALLENGE 2: Build the other gates
# ============================================================
#
# Using XOR, AND, OR, NOT, build:

def NAND(a, b):
    # YOUR CODE HERE
    pass

def NOR(a, b):
    # YOUR CODE HERE
    pass

def XNOR(a, b):
    # YOUR CODE HERE
    pass


def test_gates():
    print("Gate Tests:")
    gates = [
        ("NAND", NAND, [(0,0,1), (0,1,1), (1,0,1), (1,1,0)]),
        ("NOR",  NOR,  [(0,0,1), (0,1,0), (1,0,0), (1,1,0)]),
        ("XNOR", XNOR, [(0,0,1), (0,1,0), (1,0,0), (1,1,1)]),
    ]

    for name, fn, cases in gates:
        if fn(0, 0) is None:
            print(f"  {name}: (not implemented)")
            continue
        all_ok = True
        for a, b, expected in cases:
            got = int(fn(a, b))
            if got != expected:
                all_ok = False
        status = "✓" if all_ok else "✗"
        print(f"  {name}: {status}")
    print()

test_gates()


# ============================================================
# THE AHA MOMENT
# ============================================================
#
# You now have 7 gates: XOR, AND, OR, NOT, NAND, NOR, XNOR
# Plus a multiplexer.
#
# With these, you can build ANY digital circuit.
# A CPU. A GPU. Anything.
#
# And they're all just polynomials: +, -, *


# ============================================================
# YOU GOT IT WHEN
# ============================================================
#
# You can build any boolean function from these building blocks.
# You understand: the structure IS the computation.


# ============================================================
# SOLUTIONS
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
# mystery() is a MULTIPLEXER (MUX)
# sel=0 picks a, sel=1 picks b
#
#
# def mux4(inputs, sel0, sel1):
#     low = mystery(inputs[0], inputs[1], sel0)
#     high = mystery(inputs[2], inputs[3], sel0)
#     return mystery(low, high, sel1)
#
#
# def NAND(a, b): return NOT(AND(a, b))
# def NOR(a, b):  return NOT(OR(a, b))
# def XNOR(a, b): return NOT(XOR(a, b))
#
# Now go to: python route.py
