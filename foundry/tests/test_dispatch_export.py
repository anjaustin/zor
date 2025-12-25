"""Tests for Dispatch Table Export."""

import pytest
import subprocess
import tempfile
from pathlib import Path

from export.dispatch_export import (
    DispatchExporter,
    ExportConfig,
    SHAPE_POLYNOMIALS,
)


class TestDispatchExporter:
    """Test DispatchExporter class."""

    def test_create_exporter(self):
        """Test basic exporter creation."""
        dispatch = {0: 0, 1: 1, 2: 0}
        shapes = ['xor', 'and']
        exporter = DispatchExporter(dispatch, shapes, name="test_model")
        assert exporter.name == "test_model"
        assert len(exporter.shapes) == 2

    def test_validate_missing_shape(self):
        """Test validation catches unknown shapes."""
        dispatch = {0: 0}
        shapes = ['unknown_shape']
        with pytest.raises(ValueError, match="Unknown shape"):
            DispatchExporter(dispatch, shapes)

    def test_validate_tile_out_of_range(self):
        """Test validation catches tile index out of range."""
        dispatch = {0: 5}  # tile 5 doesn't exist
        shapes = ['xor', 'and']  # only tiles 0, 1
        with pytest.raises(ValueError, match="references tile 5"):
            DispatchExporter(dispatch, shapes)

    def test_generate_returns_tuple(self):
        """Test generate returns header and source."""
        dispatch = {0: 0, 1: 1}
        shapes = ['xor', 'and']
        exporter = DispatchExporter(dispatch, shapes, name="test")

        header, source = exporter.generate()
        assert isinstance(header, str)
        assert isinstance(source, str)
        assert "#ifndef TEST_H" in header
        assert "#include" in source


class TestGeneratedCode:
    """Test the generated C code content."""

    def test_header_has_guard(self):
        """Test header has include guard."""
        dispatch = {0: 0}
        shapes = ['xor']
        exporter = DispatchExporter(dispatch, shapes, name="my_model")

        header, _ = exporter.generate()
        assert "#ifndef MY_MODEL_H" in header
        assert "#define MY_MODEL_H" in header
        assert "#endif" in header

    def test_header_has_forward_decl(self):
        """Test header declares forward function."""
        dispatch = {0: 0}
        shapes = ['xor']
        exporter = DispatchExporter(dispatch, shapes, name="my_model")

        header, _ = exporter.generate()
        assert "my_model_forward" in header
        assert "uint8_t" in header

    def test_source_has_shapes(self):
        """Test source contains shape functions."""
        dispatch = {0: 0, 1: 1}
        shapes = ['xor', 'and']
        exporter = DispatchExporter(dispatch, shapes)

        _, source = exporter.generate()
        assert "shape_xor" in source
        assert "shape_and" in source
        assert "a ^ b" in source  # XOR bitwise
        assert "a & b" in source  # AND bitwise

    def test_source_has_dispatch_table(self):
        """Test source contains dispatch table."""
        dispatch = {0: 0, 1: 1, 2: 0, 3: 1}
        shapes = ['xor', 'and']
        exporter = DispatchExporter(dispatch, shapes)

        _, source = exporter.generate()
        assert "DISPATCH_TABLE" in source
        assert "int8_t" in source

    def test_source_has_tile_switch(self):
        """Test source contains tile switch."""
        dispatch = {0: 0, 1: 1}
        shapes = ['xor', 'and']
        exporter = DispatchExporter(dispatch, shapes, name="test")

        _, source = exporter.generate()
        assert "test_tile" in source
        assert "switch" in source
        assert "case 0:" in source
        assert "case 1:" in source

    def test_batch_function_included(self):
        """Test batch function is included by default."""
        dispatch = {0: 0}
        shapes = ['xor']
        config = ExportConfig(include_batch=True)
        exporter = DispatchExporter(dispatch, shapes, config=config)

        header, source = exporter.generate()
        assert "forward_batch" in header
        assert "forward_batch" in source
        assert "batch_size" in source

    def test_batch_function_excluded(self):
        """Test batch function can be excluded."""
        dispatch = {0: 0}
        shapes = ['xor']
        config = ExportConfig(include_batch=False)
        exporter = DispatchExporter(dispatch, shapes, config=config)

        header, source = exporter.generate()
        assert "forward_batch" not in header
        assert "batch_size" not in source


class TestExportFiles:
    """Test file export functionality."""

    def test_export_creates_files(self):
        """Test export creates .h and .c files."""
        dispatch = {0: 0, 1: 1}
        shapes = ['xor', 'and']
        exporter = DispatchExporter(dispatch, shapes, name="test_alu")

        with tempfile.TemporaryDirectory() as tmpdir:
            h_path, c_path = exporter.export(Path(tmpdir))

            assert h_path.exists()
            assert c_path.exists()
            assert h_path.name == "test_alu.h"
            assert c_path.name == "test_alu.c"

    def test_export_file_contents(self):
        """Test exported files have correct content."""
        dispatch = {0: 0}
        shapes = ['xor']
        exporter = DispatchExporter(dispatch, shapes, name="simple")

        with tempfile.TemporaryDirectory() as tmpdir:
            h_path, c_path = exporter.export(Path(tmpdir))

            header = h_path.read_text()
            source = c_path.read_text()

            assert "SIMPLE_H" in header
            assert '#include "simple.h"' in source


class TestCompilation:
    """Test that generated code compiles."""

    def test_compiles_with_gcc(self):
        """Test generated code compiles with gcc."""
        dispatch = {i: i % 4 for i in range(16)}
        shapes = ['xor', 'and', 'or', 'add']
        exporter = DispatchExporter(dispatch, shapes, name="compile_test")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            exporter.export(tmpdir)

            # Try to compile (object file only, no main)
            result = subprocess.run(
                ['gcc', '-c', '-O3', '-Wall', '-Werror',
                 str(tmpdir / "compile_test.c"),
                 '-I', str(tmpdir),
                 '-o', str(tmpdir / "compile_test.o")],
                capture_output=True,
                text=True
            )

            assert result.returncode == 0, f"Compilation failed: {result.stderr}"

    def test_compiles_all_shapes(self):
        """Test code with all shapes compiles."""
        all_shapes = list(SHAPE_POLYNOMIALS.keys())
        dispatch = {i: i % len(all_shapes) for i in range(len(all_shapes))}
        exporter = DispatchExporter(dispatch, all_shapes, name="all_shapes")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            exporter.export(tmpdir)

            result = subprocess.run(
                ['gcc', '-c', '-O3', '-Wall',
                 str(tmpdir / "all_shapes.c"),
                 '-I', str(tmpdir),
                 '-o', str(tmpdir / "all_shapes.o")],
                capture_output=True,
                text=True
            )

            assert result.returncode == 0, f"Compilation failed: {result.stderr}"


class TestExecution:
    """Test that generated code executes correctly."""

    def test_xor_correctness(self):
        """Test XOR shape produces correct results."""
        dispatch = {0: 0}  # class 0 → tile 0 (xor)
        shapes = ['xor']
        exporter = DispatchExporter(dispatch, shapes, name="xor_test")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            exporter.export(tmpdir)

            # Write test program
            test_prog = tmpdir / "test.c"
            test_prog.write_text('''
#include <stdio.h>
#include "xor_test.h"

int main() {
    int pass = 1;
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            uint8_t result = xor_test_forward(0, a, b);
            uint8_t expected = a ^ b;
            if (result != expected) {
                printf("FAIL: %d XOR %d = %d (expected %d)\\n",
                       a, b, result, expected);
                pass = 0;
            }
        }
    }
    printf(pass ? "PASS\\n" : "FAIL\\n");
    return pass ? 0 : 1;
}
''')

            # Compile
            result = subprocess.run(
                ['gcc', '-O3',
                 str(test_prog),
                 str(tmpdir / "xor_test.c"),
                 '-I', str(tmpdir),
                 '-o', str(tmpdir / "test")],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0, f"Compile failed: {result.stderr}"

            # Run
            result = subprocess.run(
                [str(tmpdir / "test")],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0
            assert "PASS" in result.stdout

    def test_multi_shape_dispatch(self):
        """Test dispatch to multiple shapes works."""
        # class 0 → xor, class 1 → and, class 2 → or
        dispatch = {0: 0, 1: 1, 2: 2}
        shapes = ['xor', 'and', 'or']
        exporter = DispatchExporter(dispatch, shapes, name="multi_test")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            exporter.export(tmpdir)

            test_prog = tmpdir / "test.c"
            test_prog.write_text('''
#include <stdio.h>
#include "multi_test.h"

int main() {
    uint8_t a = 1, b = 1;

    // class 0 → XOR: 1 ^ 1 = 0
    uint8_t r0 = multi_test_forward(0, a, b);

    // class 1 → AND: 1 & 1 = 1
    uint8_t r1 = multi_test_forward(1, a, b);

    // class 2 → OR: 1 | 1 = 1
    uint8_t r2 = multi_test_forward(2, a, b);

    int pass = (r0 == 0) && (r1 == 1) && (r2 == 1);
    printf("XOR=%d AND=%d OR=%d : %s\\n", r0, r1, r2, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
''')

            result = subprocess.run(
                ['gcc', '-O3',
                 str(test_prog),
                 str(tmpdir / "multi_test.c"),
                 '-I', str(tmpdir),
                 '-o', str(tmpdir / "test")],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0, f"Compile failed: {result.stderr}"

            result = subprocess.run(
                [str(tmpdir / "test")],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0
            assert "PASS" in result.stdout

    def test_batch_execution(self):
        """Test batch forward works."""
        dispatch = {0: 0, 1: 1}
        shapes = ['xor', 'and']
        exporter = DispatchExporter(dispatch, shapes, name="batch_test")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            exporter.export(tmpdir)

            test_prog = tmpdir / "test.c"
            test_prog.write_text('''
#include <stdio.h>
#include "batch_test.h"

int main() {
    int classes[] = {0, 0, 1, 1};  // xor, xor, and, and
    uint8_t a[] = {0, 1, 0, 1};
    uint8_t b[] = {1, 1, 1, 1};
    uint8_t out[4];

    batch_test_forward_batch(classes, a, b, out, 4);

    // Expected: 0^1=1, 1^1=0, 0&1=0, 1&1=1
    int pass = (out[0] == 1) && (out[1] == 0) && (out[2] == 0) && (out[3] == 1);
    printf("Results: %d %d %d %d : %s\\n",
           out[0], out[1], out[2], out[3],
           pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
''')

            result = subprocess.run(
                ['gcc', '-O3',
                 str(test_prog),
                 str(tmpdir / "batch_test.c"),
                 '-I', str(tmpdir),
                 '-o', str(tmpdir / "test")],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0, f"Compile failed: {result.stderr}"

            result = subprocess.run(
                [str(tmpdir / "test")],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0
            assert "PASS" in result.stdout


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_dispatch(self):
        """Test empty dispatch table."""
        dispatch = {}
        shapes = ['xor']
        exporter = DispatchExporter(dispatch, shapes)

        header, source = exporter.generate()
        assert "DISPATCH_TABLE" in source

    def test_single_shape(self):
        """Test single shape export."""
        dispatch = {i: 0 for i in range(256)}
        shapes = ['xor']
        exporter = DispatchExporter(dispatch, shapes, name="single")

        _, source = exporter.generate()
        assert "shape_xor" in source
        assert "case 0:" in source

    def test_max_classes(self):
        """Test with 256 classes."""
        dispatch = {i: i % 4 for i in range(256)}
        shapes = ['xor', 'and', 'or', 'add']
        config = ExportConfig(max_classes=256)
        exporter = DispatchExporter(dispatch, shapes, config=config)

        header, source = exporter.generate()
        assert "256" in header  # NUM_CLASSES defined in header
        assert "DISPATCH_TABLE" in source

    def test_different_bit_widths(self):
        """Test different bit width configurations."""
        dispatch = {0: 0}
        shapes = ['add']

        for bits, expected_type in [(8, 'uint8_t'), (16, 'uint16_t'), (32, 'uint32_t')]:
            config = ExportConfig(bits=bits)
            exporter = DispatchExporter(dispatch, shapes, config=config)
            header, _ = exporter.generate()
            assert expected_type in header
