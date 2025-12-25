#!/usr/bin/env python3
"""
E2E Verification: Train -> Export -> Verify

Proves the TriX pipeline is correct by comparing
Python NativeRouter output to exported C output.

This is the trust anchor for the entire pipeline.

Usage:
    python examples/e2e_verify.py           # Quick verification (1000 samples)
    python examples/e2e_verify.py --exhaustive  # Full verification (262K tests)
"""

import sys
import subprocess
import tempfile
import random
from pathlib import Path

# Add src and foundry to path
_root = Path(__file__).parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from trix.native import FrozenShapes, NativeRouter
from foundry.export.dispatch_export import DispatchExporter

# The 4 operations we'll train
SHAPES = ["add", "sub", "xor", "and"]


def train_router():
    """Train a 4-op ALU router."""
    router = NativeRouter(num_inputs=4, num_shapes=4, lr=1.0)

    # Supervised training: op 0 -> shape 0, etc.
    for epoch in range(50):
        for op in range(4):
            router.supervised_update(op, correct_shape=op)

    # Extract dispatch table from hard routing
    hard_routing = router.get_hard_routing()
    dispatch = {i: int(hard_routing[i]) for i in range(4)}
    return router, dispatch


def export_to_c(dispatch, output_dir):
    """Export dispatch table to C."""
    exporter = DispatchExporter(
        dispatch_table=dispatch,
        shapes=SHAPES,
        name="alu_verify"
    )
    header, source = exporter.export(Path(output_dir))
    return source


def compile_c(source_path, output_dir):
    """Compile C to binary with test harness."""
    binary = Path(output_dir) / "alu_verify"

    # Generate test harness
    harness = Path(output_dir) / "harness.c"
    harness.write_text(f'''
#include "alu_verify.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {{
    if (argc != 4) {{
        fprintf(stderr, "Usage: %s a b op\\n", argv[0]);
        return 1;
    }}
    uint8_t a = (uint8_t)atoi(argv[1]);
    uint8_t b = (uint8_t)atoi(argv[2]);
    int op = atoi(argv[3]);
    uint8_t result = alu_verify_forward(op, a, b);
    printf("%d\\n", result);
    return 0;
}}
''')

    result = subprocess.run(
        ["gcc", "-O2", "-o", str(binary), str(harness), str(source_path), "-I", str(output_dir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed: {result.stderr}")

    return binary


def generate_test_inputs(n=1000, seed=42):
    """Generate random test inputs."""
    random.seed(seed)
    inputs = []
    for _ in range(n):
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        op = random.randint(0, 3)
        inputs.append((a, b, op))
    return inputs


def run_python(router, inputs):
    """Compute outputs using Python NativeRouter."""
    # Map shape names to operations
    SHAPE_OPS = {
        'add': lambda a, b: (a + b) & 0xFF,
        'sub': lambda a, b: (a - b) & 0xFF,
        'xor': lambda a, b: a ^ b,
        'and': lambda a, b: a & b,
    }

    results = {}

    for a, b, op in inputs:
        shape_idx = int(router.route(op))
        shape_name = SHAPES[shape_idx]
        result = SHAPE_OPS[shape_name](a, b)
        results[(a, b, op)] = result

    return results


def run_c(binary, inputs):
    """Compute outputs using compiled C."""
    results = {}

    for a, b, op in inputs:
        result = subprocess.run(
            [str(binary), str(a), str(b), str(op)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"C execution failed: {result.stderr}")
        results[(a, b, op)] = int(result.stdout.strip())

    return results


def verify(exhaustive=False):
    """Main verification routine."""
    print("=" * 60)
    print("  E2E Verification: Train -> Export -> Verify")
    print("=" * 60)

    # Step 1: Train
    print("\n[1/5] Training NativeRouter...")
    router, dispatch = train_router()
    print(f"      Dispatch table: {dispatch}")

    # Step 2: Export
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n[2/5] Exporting to C...")
        source = export_to_c(dispatch, tmpdir)
        print(f"      Generated: {source}")

        # Step 3: Compile
        print("\n[3/5] Compiling C...")
        binary = compile_c(source, tmpdir)
        print(f"      Binary: {binary}")

        # Step 4: Generate inputs
        if exhaustive:
            n = 256 * 256 * 4
            print(f"\n[4/5] Generating {n:,} exhaustive test inputs...")
            inputs = [(a, b, op) for a in range(256) for b in range(256) for op in range(4)]
        else:
            n = 1000
            print(f"\n[4/5] Generating {n:,} random test inputs...")
            inputs = generate_test_inputs(n)

        # Step 5: Run and compare
        print("\n[5/5] Running verification...")
        print("      Computing Python outputs...", end=" ", flush=True)
        py_results = run_python(router, inputs)
        print("done")

        print("      Computing C outputs...", end=" ", flush=True)
        c_results = run_c(binary, inputs)
        print("done")

        # Compare
        mismatches = []
        for key in inputs:
            if py_results[key] != c_results[key]:
                mismatches.append((key, py_results[key], c_results[key]))

        # Report
        print("\n" + "=" * 60)
        if not mismatches:
            print(f"  SUCCESS: {len(inputs):,} tests passed!")
            print("  Pipeline verified: Train -> Export -> C matches Python")
        else:
            print(f"  FAILURE: {len(mismatches):,} mismatches found")
            for (a, b, op), py, c in mismatches[:10]:
                print(f"    ({a}, {b}, op={op}): Python={py}, C={c}")
            if len(mismatches) > 10:
                print(f"    ... and {len(mismatches) - 10} more")
        print("=" * 60)

        return len(mismatches) == 0


if __name__ == "__main__":
    exhaustive = "--exhaustive" in sys.argv
    success = verify(exhaustive=exhaustive)
    sys.exit(0 if success else 1)
