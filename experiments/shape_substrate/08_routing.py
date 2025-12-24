"""
Routing: how does an input find the right shape?

Like a key finding its lock. Pattern matching.

    python 08_routing.py
"""

# We have multiple shapes
shapes = {
    'XOR': lambda a, b: a + b - 2*a*b,
    'AND': lambda a, b: a * b,
    'OR':  lambda a, b: a + b - a*b,
}

# Each shape has a "signature" - what inputs it likes
# (In real TriX, these are learned. Here we'll set them manually.)
signatures = {
    'XOR': [1, -1],   # Likes difference (one high, one low)
    'AND': [1, 1],    # Likes both high
    'OR':  [-1, -1],  # Likes neither low (so it can make something from nothing)
}

def route(a, b):
    """Find the best-matching shape for this input."""
    input_pattern = [a, b]

    # Compute match score for each shape
    scores = {}
    for name, sig in signatures.items():
        # Dot product: how well does input match signature?
        score = sum(i * s for i, s in zip(input_pattern, sig))
        scores[name] = score

    # Winner is highest score
    winner = max(scores, key=scores.get)
    return winner, scores

# Test routing
print("Routing: which shape handles each input?")
print()
print("Signatures (what each shape 'likes'):")
for name, sig in signatures.items():
    print(f"  {name}: {sig}")
print()

print("  a  b | Scores              | Winner")
print("  -----|---------------------|-------")

for a in [0, 1]:
    for b in [0, 1]:
        winner, scores = route(a, b)
        score_str = ", ".join(f"{k}:{v:+d}" for k, v in scores.items())
        print(f"  {a}  {b} | {score_str:20s}| {winner}")

print()
print("The input (a,b) gets routed to the shape that 'likes' it most.")
print()
print("This is content-addressable routing:")
print("  - No if/else statements")
print("  - No explicit routing logic")
print("  - The signature IS the address")
print("  - The match IS the routing decision")
print()
print("In TriX, the signatures are LEARNED during training.")
print("The network discovers which inputs each shape should handle.")
print()
print("Next: python 09_put_it_together.py")
