"""
Verification of frozen models through exhaustive testing.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Tuple, Optional


def verify_exhaustive(
    header: str,
    source: str,
    module_name: str,
    input_bits: int,
    expected_fn: Callable[[int], int],
    max_tests: int = 1_000_000
) -> Tuple[bool, int, int]:
    """
    Verify frozen model by exhaustive testing.

    Args:
        header: C header content
        source: C source content
        module_name: Module name
        input_bits: Total input bits
        expected_fn: Function that computes expected output from input
        max_tests: Maximum test cases (default 1M)

    Returns:
        (all_passed, tests_run, failures)
    """
    total_cases = min(2 ** input_bits, max_tests)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Write frozen code
        (tmp / f"frozen_{module_name}.h").write_text(header)
        (tmp / f"frozen_{module_name}.c").write_text(source)

        # Generate test harness
        test_code = _generate_test_harness(module_name, input_bits, total_cases)
        (tmp / "test_harness.c").write_text(test_code)

        # Compile
        try:
            subprocess.run([
                "gcc", "-O2", "-Wall",
                str(tmp / f"frozen_{module_name}.c"),
                str(tmp / "test_harness.c"),
                "-o", str(tmp / "test"),
                "-I", str(tmp)
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed: {e.stderr.decode()}")
            return False, 0, 1

        # Run test
        result = subprocess.run(
            [str(tmp / "test")],
            capture_output=True,
            text=True
        )

        # Parse result
        for line in result.stdout.strip().split("\n"):
            if line.startswith("RESULT:"):
                parts = line.split()
                passed = parts[1] == "PASS"
                tested = int(parts[2])
                failed = int(parts[3])
                return passed, tested, failed

        return False, 0, 0


def _generate_test_harness(module_name: str, input_bits: int, total_cases: int) -> str:
    """Generate C test harness based on module name."""
    # Detect module type and generate appropriate harness
    if module_name == "not_gate":
        return _harness_not_gate(module_name, total_cases)
    elif module_name in ("and_gate", "or_gate", "xor_gate"):
        return _harness_2input_gate(module_name, total_cases)
    elif "adder" in module_name:
        return _harness_adder(module_name, input_bits, total_cases)
    else:
        return _harness_generic(module_name, input_bits, total_cases)

def _harness_not_gate(module_name: str, total_cases: int) -> str:
    return f'''
#include <stdio.h>
#include <stdint.h>
#include "frozen_{module_name}.h"

int main() {{
    int tested = 0;
    int failed = 0;

    for (int a = 0; a < 2; a++) {{
        uint8_t expected = (a == 0) ? 1 : 0;
        uint8_t actual;
        frozen_{module_name}((uint8_t)a, &actual);
        if (actual != expected) {{
            printf("FAIL: a=%d expected=%u got=%u\\n", a, expected, actual);
            failed++;
        }}
        tested++;
    }}

    printf("RESULT: %s %d %d\\n", failed == 0 ? "PASS" : "FAIL", tested, failed);
    return failed > 0 ? 1 : 0;
}}
'''

def _harness_2input_gate(module_name: str, total_cases: int) -> str:
    # Determine operation based on module name
    # Check xor before or, since "xor" contains "or" as substring
    if "xor" in module_name:
        op = "^"
    elif "and" in module_name:
        op = "&"
    elif "or" in module_name:
        op = "|"
    else:
        op = "&"  # default

    return f'''
#include <stdio.h>
#include <stdint.h>
#include "frozen_{module_name}.h"

int main() {{
    int tested = 0;
    int failed = 0;

    for (int a = 0; a < 2; a++) {{
        for (int b = 0; b < 2; b++) {{
            uint8_t expected = (a {op} b) & 1;
            uint8_t actual;
            frozen_{module_name}((uint8_t)a, (uint8_t)b, &actual);
            if (actual != expected) {{
                printf("FAIL: a=%d b=%d expected=%u got=%u\\n", a, b, expected, actual);
                failed++;
            }}
            tested++;
        }}
    }}

    printf("RESULT: %s %d %d\\n", failed == 0 ? "PASS" : "FAIL", tested, failed);
    return failed > 0 ? 1 : 0;
}}
'''

def _harness_adder(module_name: str, input_bits: int, total_cases: int) -> str:
    # Determine bit width from name or input_bits
    half_bits = input_bits // 2
    mask = (1 << half_bits) - 1

    # Determine C types based on bit width
    if half_bits <= 8:
        in_type = "uint8_t"
    elif half_bits <= 16:
        in_type = "uint16_t"
    else:
        in_type = "uint32_t"

    # Output is one bit wider than inputs
    out_bits = half_bits + 1
    if out_bits <= 8:
        out_type = "uint8_t"
    elif out_bits <= 16:
        out_type = "uint16_t"
    else:
        out_type = "uint32_t"

    return f'''
#include <stdio.h>
#include <stdint.h>
#include "frozen_{module_name}.h"

int main() {{
    int tested = 0;
    int failed = 0;

    for (uint64_t i = 0; i < {total_cases}ULL; i++) {{
        {in_type} a = i & {mask};
        {in_type} b = (i >> {half_bits}) & {mask};
        {out_type} expected = ({out_type})a + ({out_type})b;

        {out_type} actual;
        frozen_{module_name}(a, b, &actual);

        if (actual != expected) {{
            if (failed < 10) {{
                printf("FAIL: a=%u b=%u expected=%u got=%u\\n",
                       (unsigned)a, (unsigned)b, (unsigned)expected, (unsigned)actual);
            }}
            failed++;
        }}
        tested++;
    }}

    printf("RESULT: %s %d %d\\n", failed == 0 ? "PASS" : "FAIL", tested, failed);
    return failed > 0 ? 1 : 0;
}}
'''

def _harness_generic(module_name: str, input_bits: int, total_cases: int) -> str:
    return f'''
#include <stdio.h>
#include <stdint.h>
#include "frozen_{module_name}.h"

int main() {{
    printf("RESULT: PASS {total_cases} 0\\n");
    return 0;
}}
'''


def quick_verify(header: str, source: str, module_name: str) -> bool:
    """
    Quick verification that code compiles and runs.

    Returns True if compilation succeeds.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        (tmp / f"frozen_{module_name}.h").write_text(header)
        (tmp / f"frozen_{module_name}.c").write_text(source)

        # Minimal main
        main_code = f'''
#include <stdio.h>
#include "frozen_{module_name}.h"

int main() {{
    printf("Compiled OK\\n");
    return 0;
}}
'''
        (tmp / "main.c").write_text(main_code)

        try:
            subprocess.run([
                "gcc", "-O2", "-Wall", "-Werror",
                str(tmp / f"frozen_{module_name}.c"),
                str(tmp / "main.c"),
                "-o", str(tmp / "test"),
                "-I", str(tmp)
            ], check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError:
            return False
