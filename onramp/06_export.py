#!/usr/bin/env python3
"""
06_export.py - Freeze the Routing

The router learned which shapes to use.
Now freeze that knowledge into a dispatch table.
Then export it to C.

Using TriX Native + Foundry Export.
"""

import sys
from pathlib import Path

# Add the project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# TriX Native
from trix.native import FrozenALU, NativeRouter

# TriX Foundry Export
from foundry.export.dispatch_export import DispatchExporter

# CuPy if available, else NumPy
try:
    import cupy as cp
    BACKEND = "CuPy"
except ImportError:
    import numpy as cp
    BACKEND = "NumPy"


def main():
    print()
    print("=" * 60)
    print("  Freeze the Routing")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------
    # Train a router (like in 05_route.py)
    # -----------------------------------------------------------------

    print("  Training a router with TriX Native...")
    print()

    router = NativeRouter(num_inputs=11, num_shapes=16, lr=0.5)

    # Train: opcode i -> shape i
    for _ in range(20):
        for opcode in range(11):
            router.supervised_update(opcode, correct_shape=opcode)

    hard_routing = router.get_hard_routing()
    correct = sum(1 for i in range(11) if int(hard_routing[i]) == i)
    print(f"    Router trained: {correct}/11 accuracy")
    print()

    # -----------------------------------------------------------------
    # Freeze the routing
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  Freeze the routing:")
    print()
    print("    For each opcode, ask the router:")
    print("    'Which shape would you use?'")
    print()
    print("    Store the answer in a dispatch table.")
    print()

    # Extract dispatch table from router
    dispatch_table = {}
    for opcode in range(11):
        shape_id = int(hard_routing[opcode])
        dispatch_table[opcode] = shape_id

    print(f"    Created dispatch table with {len(dispatch_table)} entries.")
    print()

    # Show the mapping
    opcode_names = ['ADC', 'SBC', 'AND', 'ORA', 'EOR',
                    'ASL', 'LSR', 'ROL', 'ROR', 'INC', 'DEC']
    shape_names = FrozenALU.SHAPE_NAMES

    print("    Dispatch table:")
    for opcode in range(11):
        shape_id = dispatch_table[opcode]
        print(f"      {opcode_names[opcode]:4s} (opcode {opcode:2d}) -> {shape_names[shape_id]}")
    print()

    # -----------------------------------------------------------------
    # The insight: no more floating point
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  What we just did:")
    print()
    print("    Before:")
    print(f"      router.logits: [{router.num_inputs} x {router.num_shapes}] floats")
    print(f"      = {router.param_count()} floating-point parameters")
    print()
    print("    After:")
    print("      dispatch_table: opcode -> shape_id")
    print("      = 11 integers")
    print()
    print("    The floating-point router is GONE.")
    print("    It's now a lookup table.")
    print()

    # -----------------------------------------------------------------
    # Export to C
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  Export to C using TriX Foundry:")
    print()

    # Use the first 11 shapes (matching the 11 opcodes)
    export_shapes = [name.lower() for name in shape_names[:11]]

    # Create the exporter
    exporter = DispatchExporter(
        dispatch_table=dispatch_table,
        shapes=['add', 'sub', 'and', 'or', 'xor',  # Map to known shapes
                'identity', 'identity', 'identity', 'identity',
                'add', 'sub'],  # inc/dec as add/sub for simplicity
        name="alu_6502"
    )

    # Generate the code
    header, source = exporter.generate()

    # Show snippets
    print("    Generated: alu_6502.h")
    print()
    header_lines = header.split('\n')
    for line in header_lines[12:20]:
        print(f"      {line}")
    print("      ...")
    print()

    print("    Generated: alu_6502.c")
    print()
    source_lines = source.split('\n')
    for line in source_lines[6:12]:
        print(f"      {line}")
    print("      ...")
    print()

    # -----------------------------------------------------------------
    # Write the files
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  Write to disk:")
    print()

    output_dir = Path(__file__).parent / "export_output"
    header_path, source_path = exporter.export(output_dir)

    print(f"    {header_path}")
    print(f"    {source_path}")
    print()

    # -----------------------------------------------------------------
    # The complete picture
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  What we have now:")
    print()
    print("    ┌─────────────┐")
    print("    │ Training    │  (TriX Native - CuPy/NumPy)")
    print("    │ NativeRouter│")
    print("    └──────┬──────┘")
    print("           │ freeze routing")
    print("           ▼")
    print("    ┌─────────────┐")
    print("    │ Dispatch    │  (integers, O(1) lookup)")
    print("    │ Table       │")
    print("    └──────┬──────┘")
    print("           │ DispatchExporter")
    print("           ▼")
    print("    ┌─────────────┐")
    print("    │ C Code      │  (standalone, no dependencies)")
    print("    │ (alu_6502)  │")
    print("    └─────────────┘")
    print()
    print("  The shapes are polynomials in C.")
    print("  The routing is a lookup table in C.")
    print("  No Python. No frameworks. Just C.")
    print()

    # -----------------------------------------------------------------
    # What's next
    # -----------------------------------------------------------------

    print("=" * 60)
    print()
    print("  You've exported to C.")
    print("  Now let's compile it and run it on Thor.")
    print()
    print("  Run:  python onramp/07_deploy.py")
    print()


if __name__ == "__main__":
    main()
