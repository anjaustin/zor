#!/usr/bin/env python3
"""
07_deploy.py - Deploy to Thor

Compile the exported C code.
Run it on your Thor hardware.
See the polynomials execute at machine speed.

Using TriX Native + Foundry Export.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

# Add the project root and src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# TriX Native
from trix.native import FrozenALU, NativeRouter

# TriX Foundry Export
from foundry.export.dispatch_export import DispatchExporter


def main():
    print()
    print("=" * 60)
    print("  Deploy to Thor")
    print("=" * 60)
    print()

    # -----------------------------------------------------------------
    # Train and export (like 05 + 06)
    # -----------------------------------------------------------------

    print("  Training router and exporting to C...")
    print()

    # Train router
    router = NativeRouter(num_inputs=11, num_shapes=16, lr=0.5)
    for _ in range(20):
        for opcode in range(11):
            router.supervised_update(opcode, correct_shape=opcode)

    # Freeze routing
    hard_routing = router.get_hard_routing()
    dispatch_table = {i: int(hard_routing[i]) for i in range(11)}

    # Export
    exporter = DispatchExporter(
        dispatch_table=dispatch_table,
        shapes=['add', 'sub', 'and', 'or', 'xor',
                'identity', 'identity', 'identity', 'identity',
                'add', 'sub'],
        name="thor_alu"
    )

    # Write to temp directory
    export_dir = Path(tempfile.mkdtemp(prefix="trix_deploy_"))
    header_path, source_path = exporter.export(export_dir)

    print(f"    Generated: {header_path}")
    print(f"    Generated: {source_path}")
    print()

    # -----------------------------------------------------------------
    # Create a test harness
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  Creating test harness...")
    print()

    test_code = '''
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "thor_alu.h"

int main(void) {
    printf("\\n");
    printf("  TriX on Thor\\n");
    printf("\\n");

    // -----------------------------------------------------------------
    // Test the frozen shapes
    // -----------------------------------------------------------------

    printf("  Testing frozen shapes:\\n");
    printf("\\n");

    // Test ADD (opcode 0 -> shape 0)
    uint8_t a = 10, b = 20;
    uint8_t result = thor_alu_forward(0, a, b);
    printf("    ADD: %d + %d = %d (expected %d) %s\\n",
           a, b, result, (a + b) & 0xFF,
           result == ((a + b) & 0xFF) ? "OK" : "FAIL");

    // Test SUB (opcode 1 -> shape 1)
    a = 50; b = 20;
    result = thor_alu_forward(1, a, b);
    printf("    SUB: %d - %d = %d (expected %d) %s\\n",
           a, b, result, (a - b) & 0xFF,
           result == ((a - b) & 0xFF) ? "OK" : "FAIL");

    // Test AND (opcode 2 -> shape 2)
    a = 0xFF; b = 0x0F;
    result = thor_alu_forward(2, a, b);
    printf("    AND: 0x%02X & 0x%02X = 0x%02X (expected 0x%02X) %s\\n",
           a, b, result, a & b,
           result == (a & b) ? "OK" : "FAIL");

    // Test OR (opcode 3 -> shape 3)
    a = 0xF0; b = 0x0F;
    result = thor_alu_forward(3, a, b);
    printf("    OR:  0x%02X | 0x%02X = 0x%02X (expected 0x%02X) %s\\n",
           a, b, result, a | b,
           result == (a | b) ? "OK" : "FAIL");

    // Test XOR (opcode 4 -> shape 4)
    a = 0xAA; b = 0x55;
    result = thor_alu_forward(4, a, b);
    printf("    XOR: 0x%02X ^ 0x%02X = 0x%02X (expected 0x%02X) %s\\n",
           a, b, result, a ^ b,
           result == (a ^ b) ? "OK" : "FAIL");

    printf("\\n");

    // -----------------------------------------------------------------
    // Benchmark
    // -----------------------------------------------------------------

    printf("  Benchmarking on Thor:\\n");
    printf("\\n");

    int n_ops = 10000000;  // 10 million operations

    // Warm up
    volatile uint8_t sink = 0;
    for (int i = 0; i < 10000; i++) {
        sink ^= thor_alu_forward(i % 5, i % 256, (i >> 8) % 256);
    }

    // Timed run
    clock_t start = clock();

    for (int i = 0; i < n_ops; i++) {
        sink ^= thor_alu_forward(i % 5, i % 256, (i >> 8) % 256);
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    double ops_per_sec = n_ops / elapsed;
    double ns_per_op = (elapsed / n_ops) * 1e9;

    printf("    Operations: %d\\n", n_ops);
    printf("    Time: %.3f seconds\\n", elapsed);
    printf("    Speed: %.0f ops/sec\\n", ops_per_sec);
    printf("    Latency: %.1f ns/op\\n", ns_per_op);
    printf("\\n");

    // Prevent optimization
    printf("    (checksum: %d)\\n", sink);
    printf("\\n");

    return 0;
}
'''

    test_path = export_dir / "test_thor.c"
    test_path.write_text(test_code)
    print(f"    Written: {test_path}")
    print()

    # -----------------------------------------------------------------
    # Compile
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  Compiling for Thor (aarch64)...")
    print()

    binary_path = export_dir / "test_thor"

    compile_cmd = [
        "gcc", "-O3", "-march=native",
        "-o", str(binary_path),
        str(source_path), str(test_path),
        "-I", str(export_dir)
    ]

    print(f"    $ {' '.join(compile_cmd)}")
    print()

    try:
        result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print("    Compilation failed:")
            print(result.stderr)
            return

        print("    Compilation successful!")
        print()

    except Exception as e:
        print(f"    Compilation error: {e}")
        return

    # -----------------------------------------------------------------
    # Run on Thor
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  Running on Thor hardware...")
    print()

    try:
        result = subprocess.run(
            [str(binary_path)],
            capture_output=True,
            text=True,
            timeout=60
        )

        for line in result.stdout.split('\n'):
            print(f"  {line}")

    except Exception as e:
        print(f"    Execution error: {e}")
        return

    # -----------------------------------------------------------------
    # The TriX pipeline
    # -----------------------------------------------------------------

    print("-" * 60)
    print()
    print("  The TriX Pipeline (0 PyTorch):")
    print()
    print("    ╔═════════════════════════════════════════════════════════╗")
    print("    ║  TRAINING (TriX Native)                                 ║")
    print("    ║    NativeRouter: CuPy/NumPy softmax routing             ║")
    print("    ║    FrozenShapes: 0 learnable parameters                 ║")
    print("    ╚═══════════════════════╤═════════════════════════════════╝")
    print("                            │")
    print("                            ▼ freeze")
    print("    ╔═════════════════════════════════════════════════════════╗")
    print("    ║  DISPATCH TABLE                                         ║")
    print("    ║    opcode -> shape_id (11 integers)                     ║")
    print("    ╚═══════════════════════╤═════════════════════════════════╝")
    print("                            │")
    print("                            ▼ DispatchExporter")
    print("    ╔═════════════════════════════════════════════════════════╗")
    print("    ║  C CODE                                                 ║")
    print("    ║    Frozen shapes as inline polynomials                  ║")
    print("    ║    Dispatch as static lookup table                      ║")
    print("    ╚═══════════════════════╤═════════════════════════════════╝")
    print("                            │")
    print("                            ▼ gcc -O3")
    print("    ╔═════════════════════════════════════════════════════════╗")
    print("    ║  THOR (aarch64)                                         ║")
    print("    ║    Native execution, hundreds of millions ops/sec       ║")
    print("    ╚═════════════════════════════════════════════════════════╝")
    print()

    # -----------------------------------------------------------------
    # What's next
    # -----------------------------------------------------------------

    print("=" * 60)
    print()
    print("  You've deployed TriX to Thor.")
    print()
    print("  The entire pipeline:")
    print("    - No PyTorch")
    print("    - No TensorFlow")
    print("    - No ONNX")
    print("    - No inference runtime")
    print()
    print("  Just frozen polynomials and lookup tables.")
    print("  Compiled to native code.")
    print()
    print("  Run:  python onramp/08_explore.py")
    print()


if __name__ == "__main__":
    main()
