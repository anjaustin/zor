# Synthesize: Freshman Onboarding

## The Five-File Curriculum

Replace 10 shallow files with 5 deep ones.

---

## File 1: `discover.py` (The Hook)

Structure:
```python
# DISCOVERY: Run this file. Don't read ahead.

[10 lines that show XOR = a + b - 2ab]

# ^^^^^ That just happened. XOR from addition and multiplication.

# === TRY THESE ===
# 1. Change the 2 to a 3. What breaks?
# 2. Change the 2 to a 1. What do you get?
# 3. What happens with inputs between 0 and 1?

# === CHALLENGE ===
# Write AND(a, b) using only +, -, *
# Hint: It's simpler than XOR.
# Test it with the truth table.

# === YOU GOT IT WHEN ===
# You can explain why the formula works on a napkin.

# === SOLUTION (don't peek) ===
# Scroll down...
#
#
#
# AND(a, b) = a * b
# That's it. Multiplication IS AND for binary inputs.
```

---

## File 2: `build.py` (The Adder)

Structure:
```python
# BUILD: An 8-bit adder from scratch

def XOR(a, b): return a + b - 2*a*b
def AND(a, b): return a * b
def OR(a, b):  return a + b - a*b

# A full adder (you'll build this)
def full_adder(a, b, carry_in):
    # YOUR CODE HERE
    # Return (sum_bit, carry_out)
    pass

# Test harness (run this to check your work)
def test_full_adder():
    cases = [(0,0,0), (0,0,1), (0,1,0), (0,1,1),
             (1,0,0), (1,0,1), (1,1,0), (1,1,1)]
    for a, b, cin in cases:
        s, cout = full_adder(a, b, cin)
        expected_s = (a + b + cin) % 2
        expected_c = (a + b + cin) // 2
        status = "✓" if (s == expected_s and cout == expected_c) else "✗"
        print(f"  {a}+{b}+{cin} = {s},{cout} {status}")

# === TRY THESE ===
# 1. Implement full_adder using XOR, AND, OR
# 2. Run test_full_adder() to verify
# 3. Chain 8 full adders to make add_8bit(a_bits, b_bits)

# === CHALLENGE ===
# Make it work: add_8bit([1,0,1,0,1,0,0,0], [1,1,0,0,0,0,0,0])
# Should return: [0,0,0,1,1,0,0,0] (21 + 3 = 24... wait, check the bit order)

# === SOLUTION ===
# [Hidden at bottom of file]
```

---

## File 3: `compose.py` (The LEGO)

Structure:
```python
# COMPOSE: Shapes snap together

def XOR(a, b): return a + b - 2*a*b
def AND(a, b): return a * b
def OR(a, b):  return a + b - a*b
def NOT(a):    return 1 - a

# Here's a mystery function:
def mystery(a, b, sel):
    return OR(AND(a, NOT(sel)), AND(b, sel))

# === DISCOVER ===
# Run mystery(0, 1, 0) and mystery(0, 1, 1)
# What does this function do?

# === TRY THESE ===
# 1. Build truth table for mystery(a, b, sel)
# 2. What common circuit is this?
# 3. Swap the order of a and b in the formula. What changes?

# === CHALLENGE 1 ===
# Build a 4-to-1 multiplexer using mystery as a building block
# mux4(inputs, sel0, sel1) should select inputs[sel1*2 + sel0]

# === CHALLENGE 2 ===
# Build NAND, NOR, and XNOR using only the basic gates
# Verify each with its truth table

# === YOU GOT IT WHEN ===
# You can build any boolean function from XOR, AND, OR, NOT
```

---

## File 4: `route.py` (The Selector)

Structure:
```python
# ROUTE: How does input find the right shape?

shapes = {
    'XOR': lambda a, b: a + b - 2*a*b,
    'AND': lambda a, b: a * b,
    'OR':  lambda a, b: a + b - a*b,
}

# Each shape has a "signature" - what inputs it prefers
signatures = {
    'XOR': [1, -1],   # Likes difference
    'AND': [1, 1],    # Likes agreement
    'OR':  [-1, -1],  # Try changing this - what happens?
}

def route(a, b):
    scores = {}
    for name, sig in signatures.items():
        scores[name] = a * sig[0] + b * sig[1]
    return max(scores, key=scores.get)

# === DISCOVER ===
# Run: route(0, 1), route(1, 1), route(0, 0)
# Which shape wins each time? Why?

# === TRY THESE ===
# 1. Change OR's signature to [1, 1]. Now what wins for (1,1)?
# 2. Add a new shape 'NAND' with signature [-1, -1]. Does it ever win?
# 3. What signature would make a shape win for input (0, 0)?

# === CHALLENGE ===
# Train the signatures!
# Goal: make XOR win for (0,1) and (1,0), AND win for (1,1), OR win for (0,0)
# Write a loop that adjusts signatures based on desired routing.
# Hint: If wrong shape wins, make the right shape's signature more like the input.

# === YOU GOT IT WHEN ===
# You can predict which shape wins before running the code.
```

---

## File 5: `system.py` (The Whole Thing)

Structure:
```python
# SYSTEM: Train it, run it, see it work

# All your tools from previous files
def XOR(a, b): return a + b - 2*a*b
def AND(a, b): return a * b
def OR(a, b):  return a + b - a*b

class ShapeRouter:
    def __init__(self):
        self.shapes = {'XOR': XOR, 'AND': AND, 'OR': OR}
        self.signatures = {
            'XOR': [0.0, 0.0],
            'AND': [0.0, 0.0],
            'OR':  [0.0, 0.0],
        }

    def route(self, a, b):
        # YOUR CODE: return winning shape name

    def compute(self, a, b):
        winner = self.route(a, b)
        return self.shapes[winner](a, b)

    def train(self, examples, epochs=10):
        # YOUR CODE: adjust signatures so each input routes to target shape
        # examples = [((a, b), 'target_shape'), ...]

# === BUILD THIS ===
# 1. Implement route() using signatures
# 2. Implement train() using gradient-like updates
# 3. Train on: (0,0)->OR, (0,1)->XOR, (1,0)->XOR, (1,1)->AND
# 4. Verify routing works after training

# === CHALLENGE ===
# Add a new shape of your own invention.
# Give it a formula and a name.
# Train the router to use it for specific inputs.
# Prove it works.

# === YOU GOT IT WHEN ===
# You've added your own shape and it routes correctly.
# You understand: the shapes are frozen, only routing is learned.

# === THE INSIGHT ===
# [Left blank - you write it after completing the challenge]
```

---

## Success Criteria

| File | Takes | Output | Challenge |
|------|-------|--------|-----------|
| discover.py | 15 min | Understands polynomial XOR | Write AND |
| build.py | 25 min | Working 8-bit adder | 16-bit version |
| compose.py | 20 min | Multiplexer working | 4-to-1 mux |
| route.py | 20 min | Understands routing | Train signatures |
| system.py | 30 min | Full trained system | Add new shape |

Total: ~2 hours for complete understanding.

---

## Files to Delete

- 01_wait_what.py (replaced by discover.py)
- 02_so_what.py (deleted - too preachy)
- 03_build_something.py (replaced by build.py)
- 04_why_fast.py (merged into build.py)
- 05_the_trick.py (replaced by compose.py)
- 06_what_is_a_shape.py (deleted - too abstract)
- 07_shapes_compose.py (merged into compose.py)
- 08_routing.py (replaced by route.py)
- 09_put_it_together.py (replaced by system.py)
- 10_protein_version.py (moved to advanced/ or deleted)

---

## Implementation Priority

1. Write discover.py (the hook - must be perfect)
2. Write build.py (the satisfaction - builds real thing)
3. Write system.py (the payoff - whole system works)
4. Write compose.py (the aha - composition clicks)
5. Write route.py (the mechanics - routing understood)

Test with actual freshman before shipping.
