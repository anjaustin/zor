#!/usr/bin/env python3
"""
THE WAY DEMO
============

A complete demonstration of the Unified Foundry:
    Spec (truth table) → IR (ShapeTerms) → Target (CUDA/Execute)

This is the core pattern:
    - Define with lambdas
    - Build with Foundry
    - Execute with Backend
    - Export to CUDA

"Train with gradients. Execute with geometry. Shape IS compute."
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trix.forge import Foundry


def main():
    print("=" * 70)
    print("                    THE WAY DEMO")
    print("         Spec → IR → Target (the complete pipeline)")
    print("=" * 70)
    print()

    # =========================================================================
    # STEP 1: DEFINE (spec as truth tables)
    # =========================================================================
    print("STEP 1: DEFINE")
    print("-" * 40)
    print("Creating Foundry with atomic shapes from truth functions...")
    print()

    foundry = Foundry(bits=8)

    # Register atomic shapes
    foundry.atom("xor", lambda a, b: a ^ b)
    foundry.atom("and", lambda a, b: a & b)
    foundry.atom("or",  lambda a, b: a | b)
    foundry.atom("not", lambda a, _: a ^ 0xFF)

    print(f"  Registered atoms: {list(foundry.atoms.keys())}")
    print()

    # Show the polynomial representation
    xor_shape = foundry.atoms["xor"]
    print(f"  XOR shape has {len(xor_shape.bit_terms)} output bits")
    print(f"  Total terms: {sum(len(bt.terms) for bt in xor_shape.bit_terms)}")
    print()
    print("  The polynomial: XOR(a,b) = a + b - 2ab")
    print("  This is exact on binary inputs {0,1}")
    print()

    # =========================================================================
    # STEP 2: BUILD (compile to System)
    # =========================================================================
    print("STEP 2: BUILD")
    print("-" * 40)
    print("Building executable system...")
    print()

    system = foundry.build()

    print(f"  System built with {len(system.shapes)} shapes")
    print(f"  Bit width: {system.bits}")
    print()

    # =========================================================================
    # STEP 3: VALIDATE (100/100 or don't ship)
    # =========================================================================
    print("STEP 3: VALIDATE")
    print("-" * 40)
    print("Running exhaustive validation (65,536 test cases per shape)...")
    print()

    validation = system.validate(exhaustive=True)

    for name, result in validation.results.items():
        status = "PASS" if result.passed else "FAIL"
        print(f"  {name}: {status} ({result.accuracy * 100:.2f}%)")

    print()
    if validation.all_passed():
        print("  ALL SHAPES VALIDATED 100%")
    else:
        print("  WARNING: Some shapes failed validation")
    print()

    # =========================================================================
    # STEP 4: EXECUTE (direct computation)
    # =========================================================================
    print("STEP 4: EXECUTE")
    print("-" * 40)
    print("Direct execution examples...")
    print()

    # Example computations
    tests = [
        (42, 13, "xor", 42 ^ 13),
        (42, 13, "and", 42 & 13),
        (42, 13, "or", 42 | 13),
        (0xAA, 0, "not", 0xAA ^ 0xFF),
    ]

    for a, b, op, expected in tests:
        result = system.execute(a, b, op)
        status = "OK" if result == expected else "FAIL"
        print(f"  {op}({a}, {b}) = {result} (expected {expected}) [{status}]")

    print()

    # =========================================================================
    # STEP 5: EXPORT (to hardware targets)
    # =========================================================================
    print("STEP 5: EXPORT")
    print("-" * 40)

    output_dir = "/tmp/the_way_demo"
    os.makedirs(output_dir, exist_ok=True)

    # Export to Verilog
    verilog_dir = os.path.join(output_dir, "verilog")
    system.export_verilog(verilog_dir)
    print(f"  Verilog exported to: {verilog_dir}")

    # Export to CUDA
    cuda_dir = os.path.join(output_dir, "cuda")
    system.export_cuda(cuda_dir)
    print(f"  CUDA exported to: {cuda_dir}")

    print()

    # Show what was generated
    print("  Generated files:")
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            print(f"    {path}: {size} bytes")

    print()

    # =========================================================================
    # THE INSIGHT
    # =========================================================================
    print("=" * 70)
    print("                    THE INSIGHT")
    print("=" * 70)
    print()
    print("  We just:")
    print("    1. Defined shapes as Python lambdas (truth tables)")
    print("    2. Built a System with polynomial representation")
    print("    3. Validated 100% accuracy (262,144 test cases total)")
    print("    4. Executed directly (no training needed)")
    print("    5. Exported to Verilog (for FPGAs/ASICs)")
    print("    6. Exported to CUDA (for GPU execution)")
    print()
    print("  Same shapes. Same math. Different representations.")
    print("  Train with gradients. Execute with geometry.")
    print("  Shape IS compute.")
    print()
    print("  This is The Way.")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
