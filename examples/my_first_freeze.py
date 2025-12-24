#!/usr/bin/env python3
"""
My First Freeze
===============

The simplest possible frozen geometry example.
Under 30 lines of actual code.

Usage:
    python examples/my_first_freeze.py
"""

from trix.foundry import FrozenFoundry

# 1. Create a foundry for 8-bit operations
foundry = FrozenFoundry(bit_width=8)

# 2. Define deterministic functions - these match built-in shapes
def add_with_carry(a: int, b: int, carry: int) -> tuple:
    """Add two 8-bit numbers with carry."""
    result = a + b + carry
    return result & 0xFF, int(result > 255)

def bitwise_xor(a: int, b: int, carry: int) -> tuple:
    """XOR two 8-bit numbers."""
    return a ^ b, 0

def increment(a: int, b: int, carry: int) -> tuple:
    """Increment by 1."""
    result = a + 1
    return result & 0xFF, int(result > 255)

# 3. Register them with the foundry
foundry.register("add", add_with_carry)
foundry.register("xor", bitwise_xor)
foundry.register("inc", increment)

# 4. Build the frozen model
print("Building frozen model...")
print()
result = foundry.build(verbose=True)

# 5. See the results
print()
print(f"Training steps: {result.training_steps}")
print(f"Final accuracy: {result.accuracy * 100:.2f}%")
print()

# 6. Validate on many samples
accuracy = foundry.validate(n_samples=50000)
print(f"Validation on 50,000 samples: {accuracy * 100:.4f}%")
print()

# 7. That's it. Your functions are now frozen geometry.
if accuracy >= 0.9999:
    print("All functions froze perfectly.")
    print("They will compute exactly on all binary inputs, forever.")
    print()
    print("Next steps:")
    print("  - docs/WHY_CARE.md      Why this matters")
    print("  - docs/FRESHMAN.md      Learn the concepts")
    print("  - docs/HONEST_LIMITS.md What CAN'T this do?")
