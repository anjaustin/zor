#!/usr/bin/env python3
"""
01_see_it.py - Watch XOR Resonance Work

Run this first. Don't read ahead. Just run it.

    python3 01_see_it.py

"""

import numpy as np
np.random.seed(42)

# =============================================================================
# THE DEMO
# =============================================================================

print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║                        XOR RESONANCE - See It Work                        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

# Step 1: Create some patterns
print("STEP 1: Create patterns")
print("-" * 60)

pattern_a = 0xDEADBEEFCAFEBABE
pattern_b = 0x123456789ABCDEF0
pattern_c = 0xFEEDFACEDEADC0DE

print(f"  Pattern A: {pattern_a:#018x}")
print(f"  Pattern B: {pattern_b:#018x}")
print(f"  Pattern C: {pattern_c:#018x}")

# Step 2: Accumulate via XOR
print("\nSTEP 2: Accumulate via XOR")
print("-" * 60)

S = 0
print(f"  S = {S:#018x}  (start empty)")

S ^= pattern_a
print(f"  S ^= A  →  S = {S:#018x}")

S ^= pattern_b
print(f"  S ^= B  →  S = {S:#018x}")

S ^= pattern_c
print(f"  S ^= C  →  S = {S:#018x}")

print(f"\n  Final resonance state S = {S:#018x}")

# Step 3: Query - does a pattern resonate?
print("\nSTEP 3: Query - does a pattern resonate?")
print("-" * 60)

def popcount(x):
    """Count set bits."""
    return bin(x).count('1')

def bar(hamming, max_bits=64):
    """Visual bar."""
    filled = int((1 - hamming/max_bits) * 20)
    return "█" * filled + "░" * (20 - filled)

print(f"\n  {'Query':<20} {'Hamming':>8}  {'Visual':<22} Resonates?")
print(f"  {'-'*20} {'-'*8}  {'-'*22} {'-'*10}")

theta = 40  # Threshold

for name, query in [("Pattern A", pattern_a),
                    ("Pattern B", pattern_b),
                    ("Pattern C", pattern_c),
                    ("A ^ B", pattern_a ^ pattern_b),
                    ("A ^ C", pattern_a ^ pattern_c),
                    ("Random", np.random.randint(0, 2**64)),
                    ("All 1s", 0xFFFFFFFFFFFFFFFF),
                    ("All 0s", 0x0000000000000000)]:
    hamming = popcount(S ^ query)
    resonates = "YES" if hamming < theta else "NO"
    print(f"  {name:<20} {hamming:>8}  {bar(hamming):<22} {resonates}")

print(f"\n  Threshold θ = {theta}")
print(f"  Memory used: 64 bits (8 bytes)")

# =============================================================================
# THE EXPLANATION
# =============================================================================

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT JUST HAPPENED?

1. We created 3 patterns (A, B, C)
2. We XOR'd them all into one state S
3. We queried: "How close is this query to S?"

The magic: S = A ^ B ^ C encodes information about ALL THREE patterns
in just 64 bits. Not copies. Not hashes. The XOR interference pattern.

Why does A resonate?  Because A is "in" S.
Why does Random fail? Because it wasn't part of the accumulation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# =============================================================================
# TRY IT
# =============================================================================

print("""
TRY IT (modify the code above):

  1. Add more patterns. What happens to the Hamming distances?

  2. Change theta. What's the sweet spot?

  3. Try XOR'ing the SAME pattern twice:
        S ^= pattern_a
        S ^= pattern_a   # XOR is its own inverse!
     What happens?

  4. What if you XOR 1000 random patterns? Do the originals still resonate?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# Bonus: Show the XOR inverse property
print("BONUS: XOR is its own inverse")
print("-" * 60)

S2 = pattern_a
print(f"  Start:     S = {S2:#018x}")
S2 ^= pattern_b
print(f"  S ^= B:    S = {S2:#018x}")
S2 ^= pattern_b  # XOR again!
print(f"  S ^= B:    S = {S2:#018x}  ← Back to original!")

print(f"\n  A ^ B ^ B = A. Nothing is lost. Everything is reversible.")

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Next: python3 02_break_it.py  (find the limits)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
