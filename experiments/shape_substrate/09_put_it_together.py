"""
Put it all together: a tiny "neural network" that learns to route.

    python 09_put_it_together.py
"""

import random

# Our shapes (frozen - no learning here)
def XOR(a, b): return a + b - 2*a*b
def AND(a, b): return a * b
def OR(a, b):  return a + b - a*b

shapes = {'XOR': XOR, 'AND': AND, 'OR': OR}

# The signatures START random. Training will improve them.
def random_sig():
    return [random.uniform(-1, 1), random.uniform(-1, 1)]

signatures = {name: random_sig() for name in shapes}

def route_and_compute(a, b, signatures):
    """Route input to best shape, compute result."""
    input_pattern = [a, b]
    scores = {}
    for name, sig in signatures.items():
        score = sum(i * s for i, s in zip(input_pattern, sig))
        scores[name] = score
    winner = max(scores, key=scores.get)
    result = shapes[winner](a, b)
    return winner, result, scores

# Training data: we WANT specific shapes for specific inputs
# (0,0)->OR gives 0, (0,1)->XOR gives 1, (1,0)->XOR gives 1, (1,1)->AND gives 1
training_data = [
    ((0, 0), 'OR'),   # OR(0,0)=0
    ((0, 1), 'XOR'),  # XOR(0,1)=1
    ((1, 0), 'XOR'),  # XOR(1,0)=1
    ((1, 1), 'AND'),  # AND(1,1)=1
]

print("=" * 50)
print("TRAINING A ROUTER")
print("=" * 50)
print()
print("Goal: learn signatures so inputs route to the right shapes.")
print()
print("Before training (random signatures):")
for name, sig in signatures.items():
    print(f"  {name}: [{sig[0]:+.2f}, {sig[1]:+.2f}]")
print()

# Super simple training: nudge signatures toward inputs that should match
learning_rate = 0.5
epochs = 10

for epoch in range(epochs):
    correct = 0
    for (a, b), target_shape in training_data:
        winner, result, scores = route_and_compute(a, b, signatures)

        if winner == target_shape:
            correct += 1
        else:
            # Nudge target signature toward this input
            signatures[target_shape][0] += learning_rate * (a - signatures[target_shape][0])
            signatures[target_shape][1] += learning_rate * (b - signatures[target_shape][1])
            # Nudge wrong winner away from this input
            signatures[winner][0] -= learning_rate * 0.5 * (a - signatures[winner][0])
            signatures[winner][1] -= learning_rate * 0.5 * (b - signatures[winner][1])

    if correct == len(training_data):
        print(f"Epoch {epoch + 1}: {correct}/{len(training_data)} correct - CONVERGED!")
        break
    else:
        print(f"Epoch {epoch + 1}: {correct}/{len(training_data)} correct")

print()
print("After training:")
for name, sig in signatures.items():
    print(f"  {name}: [{sig[0]:+.2f}, {sig[1]:+.2f}]")

print()
print("=" * 50)
print("TESTING")
print("=" * 50)
print()

print("  Input | Routed to | Output | Expected")
print("  ------|-----------|--------|----------")
expected_outputs = {(0,0): 0, (0,1): 1, (1,0): 1, (1,1): 1}
for (a, b), target in training_data:
    winner, result, _ = route_and_compute(a, b, signatures)
    expected = expected_outputs[(a, b)]
    status = "✓" if int(result) == expected else "✗"
    print(f"  ({a},{b})  |    {winner:3s}    |   {int(result)}    |    {expected}     {status}")

print()
print("=" * 50)
print("WHAT JUST HAPPENED")
print("=" * 50)
print()
print("1. We have three FROZEN shapes (XOR, AND, OR)")
print("   - These never change. They're just polynomials.")
print()
print("2. We have LEARNABLE signatures")
print("   - These tell us which inputs each shape should handle.")
print()
print("3. Training adjusts the signatures")
print("   - The shapes stay frozen.")
print("   - Only the routing changes.")
print()
print("4. Result: a simple 'neural network'")
print("   - Input -> Route -> Shape -> Output")
print("   - The shapes do the compute.")
print("   - The router just picks which shape.")
print()
print("This is the core idea of TriX.")
print("Learning IS routing. Computation IS frozen shapes.")
print()
print("Explore more: python 10_protein_version.py")
