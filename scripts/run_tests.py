#!/usr/bin/env python3
"""
TriX Test Runner - Easy test execution for all users.

Usage:
    python scripts/run_tests.py              # Run all tests
    python scripts/run_tests.py quick        # Quick smoke test (~30 seconds)
    python scripts/run_tests.py providence   # Providence architecture tests
    python scripts/run_tests.py rigorous     # Rigorous stress tests
    python scripts/run_tests.py core         # Core TriX tests only
    python scripts/run_tests.py help         # Show this help
"""

import subprocess
import sys
import os
from pathlib import Path

# Ensure we're in the project root
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# Test suites organized by category
TEST_SUITES = {
    "quick": {
        "description": "Quick smoke test (~30 seconds)",
        "tests": [
            "tests/test_nn.py",
            "tests/test_hierarchical.py::TestHierarchicalTriXFFN::test_forward_shape_2d",
            "tests/test_providence_ffn.py::TestProvidenceFFNBasic::test_forward_2d",
            "tests/test_providence_rigorous.py::TestEdgeCases::test_single_sample",
        ],
    },
    "core": {
        "description": "Core TriX architecture tests",
        "tests": [
            "tests/test_nn.py",
            "tests/test_hierarchical.py",
            "tests/test_sparse.py",
            "tests/test_sparse_lookup.py",
            "tests/test_kernel.py",
        ],
    },
    "providence": {
        "description": "Providence unified architecture tests (237 tests)",
        "tests": [
            "tests/test_xor_ffn.py",
            "tests/test_frozen_shapes.py",
            "tests/test_hierarchical_temporal.py",
            "tests/test_providence_ffn.py",
            "tests/test_providence_rigorous.py",
        ],
    },
    "rigorous": {
        "description": "Rigorous stress tests for Providence (67 tests)",
        "tests": [
            "tests/test_providence_rigorous.py",
        ],
    },
    "frozen": {
        "description": "Frozen shapes and 6502 tests",
        "tests": [
            "tests/test_frozen.py",
            "tests/test_frozen_6502.py",
            "tests/test_frozen_shapes.py",
        ],
    },
    "memory": {
        "description": "Memory architecture tests (Octave, Providence)",
        "tests": [
            "tests/test_octave_memory.py",
            "tests/test_providence_memory.py",
        ],
    },
    "all": {
        "description": "All tests (may take several minutes)",
        "tests": ["tests/"],
    },
}


def print_header():
    """Print a nice header."""
    print("\n" + "=" * 60)
    print("  TriX Test Suite Runner")
    print("  The architecture that provides itself.")
    print("=" * 60 + "\n")


def print_help():
    """Print usage help."""
    print_header()
    print("Usage: python scripts/run_tests.py [suite]\n")
    print("Available test suites:\n")
    for name, info in TEST_SUITES.items():
        print(f"  {name:12} - {info['description']}")
    print(f"\n  help         - Show this help message")
    print("\nExamples:")
    print("  python scripts/run_tests.py quick      # Fast smoke test")
    print("  python scripts/run_tests.py providence # Full Providence suite")
    print("  python scripts/run_tests.py            # Run all tests")
    print()


def run_tests(suite_name: str) -> int:
    """Run a test suite and return exit code."""
    if suite_name not in TEST_SUITES:
        print(f"Unknown suite: {suite_name}")
        print_help()
        return 1

    suite = TEST_SUITES[suite_name]
    print_header()
    print(f"Running: {suite['description']}\n")

    # Build pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short",
    ]
    cmd.extend(suite["tests"])

    # Set PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    # Run tests
    print(f"Command: {' '.join(cmd)}\n")
    print("-" * 60)

    result = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT)

    print("-" * 60)
    if result.returncode == 0:
        print("\n  ALL TESTS PASSED")
    else:
        print(f"\n  SOME TESTS FAILED (exit code: {result.returncode})")
    print("=" * 60 + "\n")

    return result.returncode


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        # Default to all tests
        suite = "all"
    else:
        suite = sys.argv[1].lower()

    if suite in ("help", "-h", "--help"):
        print_help()
        return 0

    return run_tests(suite)


if __name__ == "__main__":
    sys.exit(main())
