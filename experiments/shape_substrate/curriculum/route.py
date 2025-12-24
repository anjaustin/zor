"""
ROUTE

You have multiple shapes. How does an input find the right one?
Pattern matching. Like a key finding its lock.
"""

# Our shapes
def XOR(a, b): return a + b - 2*a*b
def AND(a, b): return a * b
def OR(a, b):  return a + b - a*b

shapes = {'XOR': XOR, 'AND': AND, 'OR': OR}

# Each shape has a "signature" - a pattern it likes
signatures = {
    'XOR': [1, -1],   # Likes when inputs are different
    'AND': [1, 1],    # Likes when both inputs are high
    'OR':  [-1, -1],  # Likes when both inputs are low (so it can "create" something)
}


# ============================================================
# DISCOVER: Which shape wins?
# ============================================================

def score(input_a, input_b, signature):
    """How well does input match signature? (dot product)"""
    return input_a * signature[0] + input_b * signature[1]


def route(a, b):
    """Find the shape that best matches this input."""
    scores = {}
    for name, sig in signatures.items():
        scores[name] = score(a, b, sig)

    winner = max(scores, key=scores.get)
    return winner, scores


# Test routing
print("Routing decisions:")
print()
for a in [0, 1]:
    for b in [0, 1]:
        winner, scores = route(a, b)
        score_str = ", ".join(f"{k}:{v:+d}" for k, v in scores.items())
        print(f"  Input ({a},{b}): {score_str}")
        print(f"              Winner: {winner}")
        print()


# ============================================================
# TRY THESE
# ============================================================
#
# 1. Change OR's signature to [1, 1]. Run again.
#    Now what wins for input (1, 1)?
#
# 2. Add a new shape 'NAND' with signature [-1, -1].
#    (Add it to both `shapes` and `signatures`)
#    Does NAND ever win? For which inputs?
#
# 3. What signature would guarantee a shape wins for (0, 0)?


# ============================================================
# CHALLENGE: Train the signatures
# ============================================================
#
# Goal: Learn signatures so that:
#   - XOR wins for (0,1) and (1,0)
#   - AND wins for (1,1)
#   - OR wins for (0,0)
#
# Start with random signatures. Adjust them until routing works.

import random

def train_signatures(epochs=20):
    """
    Train signatures to route correctly.

    Training rule (simple version):
    - If wrong shape wins, make target signature more like input
    - And make wrong winner's signature less like input
    """

    # Start random
    sigs = {
        'XOR': [random.uniform(-1, 1), random.uniform(-1, 1)],
        'AND': [random.uniform(-1, 1), random.uniform(-1, 1)],
        'OR':  [random.uniform(-1, 1), random.uniform(-1, 1)],
    }

    # What we want
    targets = [
        ((0, 0), 'OR'),
        ((0, 1), 'XOR'),
        ((1, 0), 'XOR'),
        ((1, 1), 'AND'),
    ]

    lr = 0.5  # Learning rate

    for epoch in range(epochs):
        correct = 0

        for (a, b), target in targets:
            # Find current winner
            scores = {name: a * sig[0] + b * sig[1] for name, sig in sigs.items()}
            winner = max(scores, key=scores.get)

            if winner == target:
                correct += 1
            else:
                # YOUR CODE HERE:
                # 1. Move target's signature toward (a, b)
                # 2. Move winner's signature away from (a, b)
                #
                # Hint:
                #   sigs[target][0] += lr * (a - sigs[target][0])
                #   sigs[target][1] += lr * (b - sigs[target][1])
                #
                pass  # Replace with your training code

        if correct == len(targets):
            print(f"Epoch {epoch+1}: {correct}/{len(targets)} - CONVERGED!")
            break
        else:
            print(f"Epoch {epoch+1}: {correct}/{len(targets)}")

    # Show final signatures
    print("\nFinal signatures:")
    for name, sig in sigs.items():
        print(f"  {name}: [{sig[0]:+.2f}, {sig[1]:+.2f}]")

    # Test
    print("\nFinal routing:")
    for (a, b), target in targets:
        scores = {name: a * sig[0] + b * sig[1] for name, sig in sigs.items()}
        winner = max(scores, key=scores.get)
        status = "✓" if winner == target else "✗"
        print(f"  ({a},{b}) -> {winner} (want {target}) {status}")

    return sigs


# Uncomment to run training:
# train_signatures()


# ============================================================
# YOU GOT IT WHEN
# ============================================================
#
# 1. You can predict which shape wins before running the code
# 2. You implemented the training loop and it converges
# 3. You understand: signatures are learned, shapes are frozen


# ============================================================
# THE INSIGHT
# ============================================================
#
# The shapes never change. XOR is always a + b - 2ab.
# Only the ROUTING changes - which shape handles which input.
#
# This is the core idea:
#   - Shapes = frozen computation (polynomials)
#   - Signatures = learned routing (pattern matching)
#
# Train the routing. Freeze the computation.


# ============================================================
# SOLUTION
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
# In the training loop, replace `pass` with:
#
#   # Move target toward input
#   sigs[target][0] += lr * (a - sigs[target][0])
#   sigs[target][1] += lr * (b - sigs[target][1])
#
#   # Move wrong winner away from input
#   sigs[winner][0] -= lr * 0.5 * a
#   sigs[winner][1] -= lr * 0.5 * b
#
# Now go to: python system.py
