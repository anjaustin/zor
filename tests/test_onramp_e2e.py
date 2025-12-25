#!/usr/bin/env python3
"""
Onramp E2E Tests

Verifies that all onramp scripts (00-08) execute successfully.
Each script runs in subprocess isolation to catch import errors,
runtime failures, and verify expected outputs.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ONRAMP_DIR = Path(__file__).parent.parent / "onramp"


def run_script(script_name: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run an onramp script and return the result."""
    script_path = ONRAMP_DIR / script_name
    assert script_path.exists(), f"Script not found: {script_path}"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(ONRAMP_DIR.parent)  # Run from project root
    )
    return result


class TestOnrampScripts:
    """Test each onramp script runs successfully."""

    def test_00_witness(self):
        """00_witness.py - Witness the geometry."""
        result = run_script("00_witness.py")
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "XOR" in result.stdout or "witness" in result.stdout.lower()

    def test_01_touch(self):
        """01_touch.py - Touch a frozen shape."""
        result = run_script("01_touch.py")
        assert result.returncode == 0, f"Failed:\n{result.stderr}"

    def test_02_truth(self):
        """02_truth.py - Truth tables and verification."""
        result = run_script("02_truth.py")
        assert result.returncode == 0, f"Failed:\n{result.stderr}"

    def test_03_build(self):
        """03_build.py - Build computation graphs."""
        result = run_script("03_build.py")
        assert result.returncode == 0, f"Failed:\n{result.stderr}"

    def test_04_compose(self):
        """04_compose.py - Compose shapes together."""
        result = run_script("04_compose.py")
        assert result.returncode == 0, f"Failed:\n{result.stderr}"

    def test_05_route(self):
        """05_route.py - Learn to route with NativeRouter."""
        result = run_script("05_route.py")
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        # Should show training progress
        assert "epoch" in result.stdout.lower() or "routing" in result.stdout.lower() or "confidence" in result.stdout.lower()

    def test_06_export_native(self):
        """06_export.py - Freeze and export native router."""
        result = run_script("06_export.py")
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        assert "dispatch" in result.stdout.lower() or "frozen" in result.stdout.lower()

    def test_07_deploy(self):
        """07_deploy.py - Full pipeline to Thor."""
        result = run_script("07_deploy.py", timeout=120)
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        # Should show benchmark results
        assert "ops/sec" in result.stdout.lower() or "latency" in result.stdout.lower() or "benchmark" in result.stdout.lower()

    def test_08_explore(self):
        """08_explore.py - Blank canvas exploration."""
        result = run_script("08_explore.py")
        assert result.returncode == 0, f"Failed:\n{result.stderr}"
        # Minimal script, just needs to run


class TestOnrampSequence:
    """Test that onramp scripts work in sequence."""

    def test_full_sequence(self):
        """Run all scripts in order, verify none fail."""
        scripts = [
            "00_witness.py",
            "01_touch.py",
            "02_truth.py",
            "03_build.py",
            "04_compose.py",
            "05_route.py",
            "06_export.py",
            "07_deploy.py",
            "08_explore.py",
        ]

        for script in scripts:
            result = run_script(script, timeout=120)
            assert result.returncode == 0, f"{script} failed:\n{result.stderr}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
