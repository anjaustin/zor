"""
SYSTEM

Everything together. The whole machine.
You'll train it, run it, and extend it.
"""

import random


# ============================================================
# THE SHAPES (frozen - these never change)
# ============================================================

def XOR(a, b): return a + b - 2*a*b
def AND(a, b): return a * b
def OR(a, b):  return a + b - a*b


# ============================================================
# THE ROUTER (you'll build and train this)
# ============================================================

class ShapeRouter:
    """
    A router that learns to match inputs to shapes.

    The shapes are FROZEN - they always do the same thing.
    Only the signatures are LEARNED - they determine routing.
    """

    def __init__(self):
        self.shapes = {
            'XOR': XOR,
            'AND': AND,
            'OR': OR,
        }

        # Start with random signatures
        self.signatures = {
            name: [random.uniform(-1, 1), random.uniform(-1, 1)]
            for name in self.shapes
        }

    def route(self, a, b):
        """Return the name of the winning shape."""
        # Map inputs to pattern: (0,0)->(-1,-1), (0,1)->(-1,1), etc
        pattern = [2*a - 1, 2*b - 1]
        scores = {}
        for name, sig in self.signatures.items():
            scores[name] = pattern[0] * sig[0] + pattern[1] * sig[1]
        return max(scores, key=scores.get)

    def compute(self, a, b):
        """Route to winning shape and compute result."""
        winner = self.route(a, b)
        return self.shapes[winner](a, b), winner

    def train(self, examples, epochs=100, lr=1.0):
        """
        Train signatures to route correctly.

        examples: list of ((a, b), target_shape_name)
        """
        print(f"Training...")

        # Initialize with good starting points based on what we want
        # XOR should like difference: high-low or low-high
        # AND should like both high
        # OR should like at least one (or handle the zero case)
        self.signatures = {
            'XOR': [0.0, 0.0],
            'AND': [0.0, 0.0],
            'OR':  [0.0, 0.0],
        }

        for epoch in range(epochs):
            correct = 0

            for (a, b), target in examples:
                # Compute pattern: map (0,0)->(-1,-1), (0,1)->(-1,1), etc
                pattern = [2*a - 1, 2*b - 1]

                # Get scores
                scores = {
                    name: pattern[0] * sig[0] + pattern[1] * sig[1]
                    for name, sig in self.signatures.items()
                }
                winner = max(scores, key=scores.get)

                if winner == target:
                    correct += 1
                else:
                    # Push target signature toward this pattern
                    self.signatures[target][0] += lr * pattern[0]
                    self.signatures[target][1] += lr * pattern[1]

                    # Push winner away from this pattern
                    self.signatures[winner][0] -= lr * 0.5 * pattern[0]
                    self.signatures[winner][1] -= lr * 0.5 * pattern[1]

            if correct == len(examples):
                print(f"  Epoch {epoch+1}: CONVERGED!")
                return True

        print(f"  Did not converge ({correct}/{len(examples)})")
        return False


# ============================================================
# RUN THE SYSTEM
# ============================================================

print("=" * 50)
print("SHAPE ROUTING SYSTEM")
print("=" * 50)
print()

# Create router
router = ShapeRouter()

# Training data: which shape should handle which input?
training_data = [
    ((0, 0), 'OR'),   # OR(0,0)=0  - OR handles "nothing"
    ((0, 1), 'XOR'),  # XOR(0,1)=1 - XOR handles "difference"
    ((1, 0), 'XOR'),  # XOR(1,0)=1 - XOR handles "difference"
    ((1, 1), 'AND'),  # AND(1,1)=1 - AND handles "agreement"
]

# Train
print("Before training:")
for (a, b), target in training_data:
    result, winner = router.compute(a, b)
    status = "✓" if winner == target else "✗"
    print(f"  ({a},{b}) -> {winner} = {int(result)} (want {target}) {status}")
print()

router.train(training_data)
print()

print("After training:")
for (a, b), target in training_data:
    result, winner = router.compute(a, b)
    status = "✓" if winner == target else "✗"
    print(f"  ({a},{b}) -> {winner} = {int(result)} (want {target}) {status}")
print()


# ============================================================
# CHALLENGE 1: Add your own shape
# ============================================================
#
# 1. Create a new shape function (e.g., NAND, IMPLY, or your invention)
# 2. Add it to the router's shapes dictionary
# 3. Add training examples that route to your shape
# 4. Train and verify
#
# Example invention:
#   def AGREE(a, b): return 1 - XOR(a, b)  # 1 when inputs match
#
# Can you make the router learn when to use it?


# ============================================================
# CHALLENGE 2: Multi-bit routing
# ============================================================
#
# Extend the system to route based on 8-bit inputs.
# Each input is a list of 8 bits.
# The signature is now 16 numbers (8 for a, 8 for b).
#
# Can you train it to route byte operations?


# ============================================================
# CHALLENGE 3: Compose and route
# ============================================================
#
# Create compound shapes:
#   def NAND(a, b): return 1 - AND(a, b)
#   def XOR3(a, b, c): return XOR(XOR(a, b), c)
#
# Add them to the router.
# Can you train routing for 3-input operations?


# ============================================================
# YOU GOT IT WHEN
# ============================================================
#
# You can add a new shape, train routing for it, and verify it works.
# You understand:
#   - Shapes are FROZEN (the math never changes)
#   - Signatures are LEARNED (pattern matching is trained)
#   - The system routes inputs to shapes, shapes compute results


# ============================================================
# THE PAYOFF
# ============================================================
#
# What you just built:
#   1. Polynomial shapes that can be trained with gradients
#   2. Routing that learns which shape handles which input
#   3. A system that composes these into arbitrary computations
#
# What this means:
#   - Train slow (with gradients, in Python)
#   - Run fast (native operations, on hardware)
#   - Same computation, 1000x speedup
#
# The shapes ARE the computation.
# The routing IS the learning.
# Everything else is frozen.


# ============================================================
# WHAT'S NEXT
# ============================================================
#
# You've completed the curriculum. You understand:
#   - XOR, AND, OR as polynomials (discover.py)
#   - Composing shapes into circuits (build.py, compose.py)
#   - Routing inputs to shapes (route.py)
#   - Training and running a complete system (system.py)
#
# To go deeper:
#   - Read docs/SHAPE_SUBSTRATE.md for the theory
#   - Run molecular_shapes.cu for GPU speedups
#   - Explore src/trix/ for the production code
#
# Welcome to shape computation.


# ============================================================
# YOUR NOTES
# ============================================================
#
# Write your insights here after completing the challenges:
#
#
#
#
