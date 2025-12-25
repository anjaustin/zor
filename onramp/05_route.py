#!/usr/bin/env python3
"""
05_route.py - Learn to Route

The shapes are frozen. But how do we choose which one?

This is where learning enters. Not learning the computation.
Learning the routing.

Using TriX Native - no PyTorch, no TensorFlow.
"""

import sys
from pathlib import Path

# Add the project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# TriX Native - 0 PyTorch
from trix.native import FrozenShapes, FrozenALU, NativeRouter

# CuPy if available, else NumPy
try:
    import cupy as cp
    BACKEND = "CuPy (GPU)"
except ImportError:
    import numpy as cp
    BACKEND = "NumPy (CPU)"


def main():
    print()
    print("=" * 60)
    print("  Learn to Route")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------
    # The architecture
    # -----------------------------------------------------------------

    print(f"  Backend: {BACKEND}")
    print()
    print("  TriX Native provides:")
    print()
    print("    FrozenShapes  - The polynomials (xor, and, or, not)")
    print("    FrozenALU     - 16 shapes for 6502 operations")
    print("    NativeRouter  - Learnable routing (the ONLY learnable part)")
    print()

    # -----------------------------------------------------------------
    # The problem: opcode -> shape
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  The 6502 CPU has 11 ALU operations:")
    print()

    opcode_names = ['ADC', 'SBC', 'AND', 'ORA', 'EOR',
                    'ASL', 'LSR', 'ROL', 'ROR', 'INC', 'DEC']
    shape_names = FrozenALU.SHAPE_NAMES

    for i, name in enumerate(opcode_names):
        print(f"    Opcode {i:2d}: {name:4s} -> Shape {i}: {shape_names[i]}")

    print()
    print("  The shapes are frozen. The question is:")
    print("  Can the router LEARN this mapping?")
    print()

    # -----------------------------------------------------------------
    # Create the router
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  Create the router:")
    print()

    router = NativeRouter(
        num_inputs=11,   # 11 opcodes
        num_shapes=16,   # 16 shapes
        lr=0.5
    )

    print(f"    num_inputs:  {router.num_inputs}")
    print(f"    num_shapes:  {router.num_shapes}")
    print(f"    parameters:  {router.param_count()}")
    print()
    print("    The ONLY learnable parameters are in routing.")
    print("    The shapes have 0 parameters.")
    print()

    # -----------------------------------------------------------------
    # Before training
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  Before training:")
    print()

    hard_routing = router.get_hard_routing()
    correct = sum(1 for i in range(11) if int(hard_routing[i]) == i)

    print(f"    Routing accuracy: {correct}/11")
    print()
    print("    Current routing:")
    for i in range(11):
        shape_id = int(hard_routing[i])
        marker = "✓" if shape_id == i else "✗"
        print(f"      {opcode_names[i]:4s} -> {shape_names[shape_id]:15s} {marker}")
    print()

    # -----------------------------------------------------------------
    # Train the router
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  Training the router:")
    print()
    print("    Using supervised updates: tell it the correct answer.")
    print()

    # Ground truth: opcode i -> shape i
    for epoch in range(20):
        for opcode in range(11):
            router.supervised_update(opcode, correct_shape=opcode)

        if (epoch + 1) % 5 == 0:
            hard_routing = router.get_hard_routing()
            correct = sum(1 for i in range(11) if int(hard_routing[i]) == i)
            entropy = router.routing_entropy()
            print(f"    Epoch {epoch+1:2d}: accuracy={correct}/11, entropy={entropy:.3f}")

    print()

    # -----------------------------------------------------------------
    # After training
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  After training:")
    print()

    hard_routing = router.get_hard_routing()
    correct = sum(1 for i in range(11) if int(hard_routing[i]) == i)

    print(f"    Routing accuracy: {correct}/11")
    print()
    print("    Final routing:")
    for i in range(11):
        shape_id = int(hard_routing[i])
        marker = "✓" if shape_id == i else "✗"
        print(f"      {opcode_names[i]:4s} -> {shape_names[shape_id]:15s} {marker}")
    print()

    # -----------------------------------------------------------------
    # The insight
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  What just happened?")
    print()
    print("    The shapes never changed. They're frozen polynomials.")
    print()
    print("    The router learned a mapping:")
    print(f"      logits: [{router.num_inputs} x {router.num_shapes}] = {router.param_count()} parameters")
    print()
    print("    That's it. 176 numbers encode the entire routing logic.")
    print()

    # -----------------------------------------------------------------
    # The TriX insight
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  This is the core of TriX:")
    print()
    print("    COMPUTATION is frozen shapes (polynomials).")
    print("    LEARNING is routing (which shape for which input).")
    print()
    print("    The shapes are the 'what'.")
    print("    The routing is the 'when'.")
    print()
    print("    Neural networks usually learn both.")
    print("    TriX separates them.")
    print()
    print("    Why? Because:")
    print()
    print("      - Frozen shapes can be optimized to hardware")
    print("      - Routing can be compiled to lookup tables")
    print("      - The result is fast, deterministic, deployable")
    print()

    # -----------------------------------------------------------------
    # What's next
    # -----------------------------------------------------------------

    print("=" * 60)
    print()
    print("  You've learned the routing with TriX Native.")
    print("  No PyTorch. No TensorFlow. Just CuPy/NumPy.")
    print()
    print("  Next: freeze the routing into a dispatch table.")
    print("  Then export it to C code.")
    print()
    print("  Run:  python onramp/06_export.py")
    print()


if __name__ == "__main__":
    main()
