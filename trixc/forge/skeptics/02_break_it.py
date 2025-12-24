#!/usr/bin/env python3
"""
02_break_it.py - Find Where XOR Resonance Fails

Every technique has limits. Let's find them honestly.

    python3 02_break_it.py

"""

import numpy as np
np.random.seed(42)

def popcount(x):
    return bin(x).count('1')

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║                    XOR RESONANCE - Find the Limits                        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# LIMIT 1: Too Many Patterns
# =============================================================================

print("LIMIT 1: What happens with too many patterns?")
print("=" * 70)

for n_patterns in [3, 10, 50, 100, 500, 1000]:
    # Generate random patterns
    patterns = [np.random.randint(0, 2**64) for _ in range(n_patterns)]

    # Accumulate
    S = 0
    for p in patterns:
        S ^= p

    # Test: can we still recognize the first pattern?
    hamming_first = popcount(S ^ patterns[0])
    hamming_random = popcount(S ^ np.random.randint(0, 2**64))

    # Classification: is first pattern closer than random?
    first_wins = hamming_first < hamming_random

    print(f"  {n_patterns:>4} patterns: first={hamming_first:>2}, random={hamming_random:>2}  → {'✓ First wins' if first_wins else '✗ Random wins'}")

print("""
INSIGHT: With many patterns, S approaches random noise.
         Individual patterns become indistinguishable.
         This is the information capacity limit.
""")

# =============================================================================
# LIMIT 2: Similar Patterns (Collision)
# =============================================================================

print("LIMIT 2: What happens with very similar patterns?")
print("=" * 70)

base = 0xDEADBEEFCAFEBABE

for noise_bits in [0, 1, 2, 4, 8, 16, 32]:
    # Create patterns that differ from base by noise_bits
    patterns = []
    for _ in range(10):
        noise_mask = 0
        positions = np.random.choice(64, noise_bits, replace=False)
        for pos in positions:
            noise_mask |= (1 << pos)
        patterns.append(base ^ noise_mask)

    # Accumulate
    S = 0
    for p in patterns:
        S ^= p

    # How far is S from base?
    hamming_to_base = popcount(S ^ base)

    print(f"  {noise_bits:>2} noise bits: S differs from base by {hamming_to_base:>2} bits")

print("""
INSIGHT: Very similar patterns cancel out via XOR.
         S converges toward patterns with even overlap counts.
         This is useful for finding the "consensus" pattern.
""")

# =============================================================================
# LIMIT 3: The Retrieval Problem
# =============================================================================

print("LIMIT 3: Can we RETRIEVE which pattern matched?")
print("=" * 70)

patterns = [
    0xAAAAAAAAAAAAAAAA,
    0xBBBBBBBBBBBBBBBB,
    0xCCCCCCCCCCCCCCCC,
]

S = 0
for p in patterns:
    S ^= p

query = 0xAAAAAAAAAAAAAAAA  # This is pattern 0

hamming = popcount(S ^ query)
resonates = hamming < 32

print(f"  Query: {query:#018x}")
print(f"  Hamming distance: {hamming}")
print(f"  Resonates: {resonates}")
print(f"  BUT: Which pattern was it? We don't know!")

print("""
INSIGHT: XOR resonance answers "DOES this belong?" not "WHICH one is it?"
         For routing/classification: that's enough.
         For retrieval: you need a different approach.

         This is the fundamental trade-off:
           Traditional: O(n) memory, tells you WHICH
           XOR:         O(1) memory, tells you WHETHER
""")

# =============================================================================
# LIMIT 4: False Positives
# =============================================================================

print("LIMIT 4: False positives (firing when it shouldn't)")
print("=" * 70)

# Train on structured patterns
train_patterns = [0xFF00FF00FF00FF00 ^ (i << 8) for i in range(20)]

S = 0
for p in train_patterns:
    S ^= p

# Test on random patterns
n_test = 1000
false_positives = 0
for theta in [16, 24, 32, 40, 48]:
    fp = 0
    for _ in range(n_test):
        random_query = np.random.randint(0, 2**64)
        if popcount(S ^ random_query) < theta:
            fp += 1

    fp_rate = fp / n_test
    print(f"  θ = {theta:>2}: False positive rate = {fp_rate:.1%}")

print("""
INSIGHT: Lower theta = fewer false positives, but also fewer true positives.
         There's a precision/recall trade-off, just like any classifier.
         The optimal theta depends on your tolerance for errors.
""")

# =============================================================================
# LIMIT 5: Adversarial Patterns
# =============================================================================

print("LIMIT 5: Can we craft an adversarial pattern?")
print("=" * 70)

# Build resonance state
honest_patterns = [np.random.randint(0, 2**64) for _ in range(50)]
S = 0
for p in honest_patterns:
    S ^= p

# Adversarial: find a pattern that resonates but wasn't trained
# Strategy: just use S itself (or close to it)
adversarial = S ^ 0x0000000F  # Flip 4 bits

hamming_adversarial = popcount(S ^ adversarial)

print(f"  Resonance state S = {S:#018x}")
print(f"  Adversarial query  = {adversarial:#018x}")
print(f"  Hamming distance   = {hamming_adversarial}")
print(f"  Resonates (θ=32):  {hamming_adversarial < 32}")

print("""
INSIGHT: If you know S, you can craft inputs that resonate.
         This matters for security applications.
         Mitigation: keep S secret, or use cryptographic hashing.
""")

# =============================================================================
# SUMMARY
# =============================================================================

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUMMARY OF LIMITS:

  1. CAPACITY: Too many patterns → S becomes noise
  2. COLLISION: Similar patterns → cancel out
  3. RETRIEVAL: Can't tell WHICH pattern matched, only WHETHER
  4. FALSE POSITIVES: Threshold trade-off exists
  5. ADVERSARIAL: If S is known, can be gamed

WHEN TO USE XOR RESONANCE:

  ✓ Routing/classification (which shape to use?)
  ✓ Anomaly detection (is this in-distribution?)
  ✓ Resource-constrained (512 bits >> 1MB index)
  ✓ Speed-critical (O(1) >> O(log n))

WHEN NOT TO USE:

  ✗ Retrieval (need to return the matching item)
  ✗ High-precision recall (need exact matches)
  ✗ Adversarial settings without S protection

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

print("""
TRY IT:

  1. Find the EXACT number of patterns where recognition breaks down
  2. Craft a pattern that resonates but is maximally different from all training patterns
  3. Build a simple adversarial defense (hint: salt the patterns)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Next: python3 03_measure_it.py  (get the numbers)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
