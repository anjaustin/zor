"""
Tests for C Code Exporter
==========================

The C exporter generates portable C99 code from chip specifications.
"""

import pytest
from pathlib import Path
import tempfile
import subprocess
import os

from foundry.export.c_export import CExporter, ExportConfig
from foundry.export.trixc import ChipSpec, ChipType


class TestCExporter:
    """Tests for C code generation."""

    def test_generate_returns_header_and_source(self):
        """Generate should return tuple of (header, source)."""
        spec = ChipSpec(
            name="XOR",
            chip_type=ChipType.ATOM,
            bits=1,
            description="XOR gate",
            polynomial="a + b - 2*a*b",
            coefficients=[1, 1, -2],
            inputs=["a", "b"],
        )

        header, source = CExporter.generate(spec)

        assert isinstance(header, str)
        assert isinstance(source, str)
        assert len(header) > 0
        assert len(source) > 0

    def test_header_has_include_guard(self):
        """Generated header should have include guard."""
        spec = ChipSpec(
            name="test",
            chip_type=ChipType.ATOM,
            bits=8,
            description="",
            polynomial="a",
            coefficients=[1],
            inputs=["a"],
        )

        header, _ = CExporter.generate(spec)

        assert "#ifndef" in header
        assert "#define" in header
        assert "#endif" in header

    def test_header_includes_stdint(self):
        """Header should include stdint.h for uint8_t etc."""
        spec = ChipSpec(
            name="test",
            chip_type=ChipType.ATOM,
            bits=8,
            description="",
            polynomial="a",
            coefficients=[1],
            inputs=["a"],
        )

        header, _ = CExporter.generate(spec)

        assert "#include <stdint.h>" in header

    def test_source_includes_header(self):
        """Source should include its own header."""
        spec = ChipSpec(
            name="XOR",
            chip_type=ChipType.ATOM,
            bits=1,
            description="",
            polynomial="a + b - 2*a*b",
            coefficients=[1, 1, -2],
            inputs=["a", "b"],
        )

        _, source = CExporter.generate(spec)

        assert '#include "trix_xor.h"' in source

    def test_function_name_prefix(self):
        """Function names should use configured prefix."""
        spec = ChipSpec(
            name="test",
            chip_type=ChipType.ATOM,
            bits=8,
            description="",
            polynomial="a",
            coefficients=[1],
            inputs=["a"],
        )

        config = ExportConfig(prefix="myprefix")
        header, source = CExporter.generate(spec, config)

        assert "myprefix_test" in header
        assert "myprefix_test" in source

    def test_export_creates_files(self):
        """Export should create .h and .c files."""
        spec = ChipSpec(
            name="AND",
            chip_type=ChipType.ATOM,
            bits=1,
            description="AND gate",
            polynomial="a * b",
            coefficients=[0, 0, 1],
            inputs=["a", "b"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            header_path, source_path = CExporter.export(spec, output_dir)

            assert header_path.exists()
            assert source_path.exists()
            assert header_path.suffix == ".h"
            assert source_path.suffix == ".c"

    def test_polynomial_in_source(self):
        """Source should contain polynomial comment."""
        spec = ChipSpec(
            name="XOR",
            chip_type=ChipType.ATOM,
            bits=1,
            description="",
            polynomial="a + b - 2*a*b",
            coefficients=[1, 1, -2],
            inputs=["a", "b"],
        )

        _, source = CExporter.generate(spec)

        assert "a + b - 2*a*b" in source

    def test_coefficients_in_source(self):
        """Source should contain coefficient array."""
        spec = ChipSpec(
            name="test",
            chip_type=ChipType.ATOM,
            bits=8,
            description="",
            polynomial="",
            coefficients=[1, -2, 3],
            inputs=["a"],
        )

        _, source = CExporter.generate(spec)

        assert "COEFFICIENTS" in source
        assert "1" in source
        assert "-2" in source
        assert "3" in source

    def test_name_sanitization(self):
        """Names with spaces/hyphens should be sanitized."""
        spec = ChipSpec(
            name="8-bit ADD",
            chip_type=ChipType.ATOM,
            bits=8,
            description="",
            polynomial="a + b",
            coefficients=[1, 1],
            inputs=["a", "b"],
        )

        header, source = CExporter.generate(spec)

        # Function and type names should be sanitized (no hyphens)
        # But comments can contain original name
        assert "trix_8_bit_add" in header  # Function name sanitized
        assert "trix_8_bit_add_t" in header  # Type name sanitized
        assert "void trix_8-bit" not in header  # No hyphens in declarations


class TestCExporterPolynomials:
    """Tests for known polynomial mappings."""

    @pytest.mark.parametrize("gate,poly", [
        ("xor", "a + b - 2*a*b"),
        ("and", "a * b"),
        ("or", "a + b - a*b"),
        ("not", "1 - a"),
        ("nand", "1 - a*b"),
        ("nor", "1 - a - b + a*b"),
        ("xnor", "1 - a - b + 2*a*b"),
    ])
    def test_known_gate_polynomials(self, gate, poly):
        """Known gates should have correct polynomials."""
        spec = ChipSpec(
            name=gate,
            chip_type=ChipType.ATOM,
            bits=1,
            description="",
            polynomial="",  # Empty - should use known mapping
            coefficients=[],
            inputs=["a", "b"] if gate != "not" else ["a"],
        )

        _, source = CExporter.generate(spec)

        assert poly in source


class TestCExporterCompilation:
    """Tests that verify generated C code compiles."""

    @pytest.fixture
    def compile_test(self):
        """Fixture to compile and optionally run generated code."""
        def _compile(spec, test_code=None):
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                header_path, source_path = CExporter.export(spec, output_dir)

                # Try to compile
                obj_path = output_dir / "test.o"
                result = subprocess.run(
                    ["gcc", "-c", "-o", str(obj_path), str(source_path)],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    pytest.fail(f"Compilation failed:\n{result.stderr}")

                # If test code provided, compile and run full test
                if test_code:
                    test_path = output_dir / "test_main.c"
                    test_path.write_text(test_code)

                    exe_path = output_dir / "test_exe"
                    result = subprocess.run(
                        ["gcc", "-o", str(exe_path), str(test_path), str(source_path)],
                        capture_output=True,
                        text=True
                    )

                    if result.returncode != 0:
                        pytest.fail(f"Test compilation failed:\n{result.stderr}")

                    result = subprocess.run(
                        [str(exe_path)],
                        capture_output=True,
                        text=True
                    )

                    return result.returncode, result.stdout

                return 0, ""

        return _compile

    def test_xor_compiles(self, compile_test):
        """XOR gate should compile."""
        spec = ChipSpec(
            name="XOR",
            chip_type=ChipType.ATOM,
            bits=1,
            description="",
            polynomial="a + b - 2*a*b",
            coefficients=[1, 1, -2],
            inputs=["a", "b"],
        )
        compile_test(spec)

    def test_xor_computes_correctly(self, compile_test):
        """XOR gate should compute correct values."""
        spec = ChipSpec(
            name="XOR",
            chip_type=ChipType.ATOM,
            bits=1,
            description="",
            polynomial="a + b - 2*a*b",
            coefficients=[1, 1, -2],
            inputs=["a", "b"],
        )

        test_code = '''
#include <stdio.h>
#include "trix_xor.h"

int main() {
    int pass = 1;
    pass &= (trix_xor(0, 0) == 0);
    pass &= (trix_xor(0, 1) == 1);
    pass &= (trix_xor(1, 0) == 1);
    pass &= (trix_xor(1, 1) == 0);
    return pass ? 0 : 1;
}
'''
        retcode, _ = compile_test(spec, test_code)
        assert retcode == 0

    def test_and_computes_correctly(self, compile_test):
        """AND gate should compute correct values."""
        spec = ChipSpec(
            name="AND",
            chip_type=ChipType.ATOM,
            bits=1,
            description="",
            polynomial="a * b",
            coefficients=[0, 0, 1],
            inputs=["a", "b"],
        )

        test_code = '''
#include <stdio.h>
#include "trix_and.h"

int main() {
    int pass = 1;
    pass &= (trix_and(0, 0) == 0);
    pass &= (trix_and(0, 1) == 0);
    pass &= (trix_and(1, 0) == 0);
    pass &= (trix_and(1, 1) == 1);
    return pass ? 0 : 1;
}
'''
        retcode, _ = compile_test(spec, test_code)
        assert retcode == 0

    def test_or_computes_correctly(self, compile_test):
        """OR gate should compute correct values."""
        spec = ChipSpec(
            name="OR",
            chip_type=ChipType.ATOM,
            bits=1,
            description="",
            polynomial="a + b - a*b",
            coefficients=[1, 1, -1],
            inputs=["a", "b"],
        )

        test_code = '''
#include <stdio.h>
#include "trix_or.h"

int main() {
    int pass = 1;
    pass &= (trix_or(0, 0) == 0);
    pass &= (trix_or(0, 1) == 1);
    pass &= (trix_or(1, 0) == 1);
    pass &= (trix_or(1, 1) == 1);
    return pass ? 0 : 1;
}
'''
        retcode, _ = compile_test(spec, test_code)
        assert retcode == 0

    def test_not_computes_correctly(self, compile_test):
        """NOT gate should compute correct values."""
        spec = ChipSpec(
            name="NOT",
            chip_type=ChipType.ATOM,
            bits=1,
            description="",
            polynomial="1 - a",
            coefficients=[-1],
            inputs=["a"],
        )

        test_code = '''
#include <stdio.h>
#include "trix_not.h"

int main() {
    int pass = 1;
    pass &= (trix_not(0) == 1);
    pass &= (trix_not(1) == 0);
    return pass ? 0 : 1;
}
'''
        retcode, _ = compile_test(spec, test_code)
        assert retcode == 0

    def test_8bit_add_computes_correctly(self, compile_test):
        """8-bit ADD should compute correct values."""
        spec = ChipSpec(
            name="add",
            chip_type=ChipType.ATOM,
            bits=8,
            description="",
            polynomial="a + b",
            coefficients=[1, 1],
            inputs=["a", "b"],
        )

        test_code = '''
#include <stdio.h>
#include "trix_add.h"

int main() {
    int pass = 1;
    pass &= (trix_add(0, 0) == 0);
    pass &= (trix_add(1, 1) == 2);
    pass &= (trix_add(10, 20) == 30);
    pass &= (trix_add(100, 55) == 155);
    return pass ? 0 : 1;
}
'''
        retcode, _ = compile_test(spec, test_code)
        assert retcode == 0
