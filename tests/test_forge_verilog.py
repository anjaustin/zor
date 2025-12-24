"""
Tests for trix.forge.verilog module.

Tests Verilog RTL generation from ShapeTerms specifications.
"""

import pytest
import tempfile
import os
from pathlib import Path

from trix.forge.verilog import (
    shape_to_verilog,
    export_verilog,
    generate_testbench,
    _term_to_verilog,
    _is_simple_shape,
    _generate_simple_bit,
    _generate_polynomial_bit,
    _generate_wrapper,
)
from trix.forge.term import (
    Term, BitTerms, ShapeTerms,
    generate_xor_shape, generate_and_shape, generate_or_shape, generate_not_shape,
    generate_nand_shape, generate_nor_shape, generate_xnor_shape, generate_nop_shape,
    generate_add_shape,
)


# =============================================================================
# TERM TO VERILOG
# =============================================================================

class TestTermToVerilog:
    """Test _term_to_verilog helper."""

    def test_constant_term_one(self):
        """Constant 1 becomes 1'b1."""
        term = Term(1, ())
        result = _term_to_verilog(term, bits=8)
        assert result == "1'b1"

    def test_constant_term_other(self):
        """Other constants become their value."""
        term = Term(5, ())
        result = _term_to_verilog(term, bits=8)
        assert result == "5"

    def test_single_variable_a(self):
        """Variable in first half maps to a[i]."""
        term = Term(1, (3,))
        result = _term_to_verilog(term, bits=8)
        assert result == "a[3]"

    def test_single_variable_b(self):
        """Variable in second half maps to b[i-bits]."""
        term = Term(1, (10,))
        result = _term_to_verilog(term, bits=8)
        assert result == "b[2]"

    def test_product_of_variables(self):
        """Multiple variables become AND expression."""
        term = Term(1, (0, 8))  # a[0] * b[0]
        result = _term_to_verilog(term, bits=8)
        assert result == "a[0] & b[0]"

    def test_three_variable_product(self):
        """Three variables become chained AND."""
        term = Term(1, (0, 1, 8))
        result = _term_to_verilog(term, bits=8)
        assert result == "a[0] & a[1] & b[0]"


# =============================================================================
# SIMPLE SHAPE DETECTION
# =============================================================================

class TestSimpleShapeDetection:
    """Test _is_simple_shape helper."""

    def test_recognizes_xor(self):
        shape = generate_xor_shape(bits=8)
        assert _is_simple_shape(shape) == "xor"

    def test_recognizes_and(self):
        shape = generate_and_shape(bits=8)
        assert _is_simple_shape(shape) == "and"

    def test_recognizes_or(self):
        shape = generate_or_shape(bits=8)
        assert _is_simple_shape(shape) == "or"

    def test_recognizes_not(self):
        shape = generate_not_shape(bits=8)
        assert _is_simple_shape(shape) == "not"

    def test_recognizes_nand(self):
        shape = generate_nand_shape(bits=8)
        assert _is_simple_shape(shape) == "nand"

    def test_recognizes_nor(self):
        shape = generate_nor_shape(bits=8)
        assert _is_simple_shape(shape) == "nor"

    def test_recognizes_xnor(self):
        shape = generate_xnor_shape(bits=8)
        assert _is_simple_shape(shape) == "xnor"

    def test_recognizes_nop(self):
        shape = generate_nop_shape(bits=8)
        assert _is_simple_shape(shape) == "nop"

    def test_complex_shape_not_simple(self):
        shape = generate_add_shape(bits=8)
        assert _is_simple_shape(shape) is None


# =============================================================================
# SIMPLE BIT GENERATION
# =============================================================================

class TestSimpleBitGeneration:
    """Test _generate_simple_bit helper."""

    def test_xor_bit(self):
        result = _generate_simple_bit("xor", 5)
        assert result == "a[5] ^ b[5]"

    def test_and_bit(self):
        result = _generate_simple_bit("and", 3)
        assert result == "a[3] & b[3]"

    def test_or_bit(self):
        result = _generate_simple_bit("or", 7)
        assert result == "a[7] | b[7]"

    def test_not_bit(self):
        result = _generate_simple_bit("not", 0)
        assert result == "~a[0]"

    def test_nand_bit(self):
        result = _generate_simple_bit("nand", 2)
        assert result == "~(a[2] & b[2])"

    def test_nor_bit(self):
        result = _generate_simple_bit("nor", 4)
        assert result == "~(a[4] | b[4])"

    def test_xnor_bit(self):
        result = _generate_simple_bit("xnor", 6)
        assert result == "~(a[6] ^ b[6])"

    def test_nop_bit(self):
        result = _generate_simple_bit("nop", 1)
        assert result == "a[1]"


# =============================================================================
# SHAPE TO VERILOG
# =============================================================================

class TestShapeToVerilog:
    """Test shape_to_verilog function."""

    def test_xor_module_structure(self):
        """XOR generates valid module with correct ports."""
        shape = generate_xor_shape(bits=8)
        verilog = shape_to_verilog(shape, include_header=False)

        assert "module xorpu_xor" in verilog
        assert "input  wire [7:0] a" in verilog
        assert "input  wire [7:0] b" in verilog
        assert "output wire [7:0] result" in verilog
        assert "endmodule" in verilog

    def test_xor_optimized_assigns(self):
        """Optimized XOR uses ^ operator."""
        shape = generate_xor_shape(bits=4)
        verilog = shape_to_verilog(shape, optimize=True, include_header=False)

        for i in range(4):
            assert f"assign result[{i}] = a[{i}] ^ b[{i}];" in verilog

    def test_not_single_operand(self):
        """NOT shape has only input a (not b)."""
        shape = generate_not_shape(bits=8)
        verilog = shape_to_verilog(shape, include_header=False)

        assert "input  wire [7:0] a" in verilog
        assert "input  wire [7:0] b" not in verilog

    def test_with_header(self):
        """Header includes metadata."""
        shape = generate_xor_shape(bits=8)
        verilog = shape_to_verilog(shape, include_header=True)

        assert "Auto-generated by trix.forge" in verilog
        assert "Shape: xor" in verilog
        assert "Total terms:" in verilog

    def test_polynomial_mode(self):
        """Non-optimized generates polynomial with wires."""
        shape = generate_xor_shape(bits=4)
        verilog = shape_to_verilog(shape, optimize=False, include_header=False)

        # Should have wire declarations for terms
        assert "wire t0_0" in verilog or "assign result[0]" in verilog


# =============================================================================
# TESTBENCH GENERATION
# =============================================================================

class TestTestbenchGeneration:
    """Test generate_testbench function."""

    def test_basic_testbench_structure(self):
        """Testbench has correct module and signals."""
        shape = generate_xor_shape(bits=8)
        tb = generate_testbench(shape)

        assert "module tb_xor" in tb
        assert "reg [7:0] a" in tb
        assert "reg [7:0] b" in tb
        assert "wire [7:0] result" in tb
        assert "xorpu_xor dut" in tb
        assert "$finish" in tb

    def test_testbench_with_test_cases(self):
        """Test cases generate assertions."""
        shape = generate_xor_shape(bits=8)
        test_cases = [(0xAA, 0x55, 0xFF), (0x00, 0xFF, 0xFF)]
        tb = generate_testbench(shape, test_cases=test_cases)

        assert "a = 8'hAA" in tb
        assert "b = 8'h55" in tb
        assert "8'hFF" in tb

    def test_single_operand_testbench(self):
        """NOT testbench has no b signal."""
        shape = generate_not_shape(bits=8)
        tb = generate_testbench(shape)

        # Should have a but not b in stimulus
        assert "reg [7:0] a" in tb
        # b is not in the testbench for single operand
        lines = tb.split('\n')
        reg_lines = [l for l in lines if 'reg' in l and '[7:0]' in l]
        assert len(reg_lines) == 1  # Only 'a'


# =============================================================================
# EXPORT VERILOG
# =============================================================================

class TestExportVerilog:
    """Test export_verilog function."""

    def test_exports_all_files(self):
        """All shapes get exported plus top wrapper."""
        shapes = {
            'xor': generate_xor_shape(bits=8),
            'and': generate_and_shape(bits=8),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            files = export_verilog(shapes, tmpdir)

            assert len(files) == 3  # xor, and, top
            assert any('xorpu_xor.v' in f for f in files)
            assert any('xorpu_and.v' in f for f in files)
            assert any('xorpu_top.v' in f for f in files)

            # Files exist
            for f in files:
                assert os.path.exists(f)

    def test_creates_output_directory(self):
        """Creates output directory if needed."""
        shapes = {'xor': generate_xor_shape(bits=8)}

        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, 'new', 'nested', 'dir')
            files = export_verilog(shapes, new_dir)

            assert os.path.exists(new_dir)
            assert len(files) == 2


# =============================================================================
# WRAPPER GENERATION
# =============================================================================

class TestWrapperGeneration:
    """Test _generate_wrapper function."""

    def test_wrapper_structure(self):
        """Wrapper has correct module structure."""
        shapes = {
            'xor': generate_xor_shape(bits=8),
            'and': generate_and_shape(bits=8),
        }
        wrapper = _generate_wrapper(shapes)

        assert "module xorpu_top" in wrapper
        assert "input  wire [7:0] a" in wrapper
        assert "input  wire [7:0] b" in wrapper
        assert "input  wire [3:0] shape_id" in wrapper
        assert "output reg  [7:0] result" in wrapper

    def test_wrapper_instantiates_shapes(self):
        """Wrapper instantiates all shapes."""
        shapes = {
            'xor': generate_xor_shape(bits=8),
            'and': generate_and_shape(bits=8),
        }
        wrapper = _generate_wrapper(shapes)

        assert "xorpu_xor u_xor" in wrapper
        assert "xorpu_and u_and" in wrapper

    def test_wrapper_has_mux(self):
        """Wrapper has shape selection MUX."""
        shapes = {
            'xor': generate_xor_shape(bits=8),
            'and': generate_and_shape(bits=8),
        }
        wrapper = _generate_wrapper(shapes)

        assert "case (shape_id)" in wrapper
        assert "4'd0:" in wrapper
        assert "4'd1:" in wrapper
        assert "default:" in wrapper

    def test_wrapper_shape_id_comments(self):
        """Wrapper documents shape IDs."""
        shapes = {
            'xor': generate_xor_shape(bits=8),
            'and': generate_and_shape(bits=8),
        }
        wrapper = _generate_wrapper(shapes)

        assert "// Shape IDs:" in wrapper


# =============================================================================
# INTEGRATION
# =============================================================================

class TestVerilogIntegration:
    """Integration tests for full Verilog generation flow."""

    def test_all_simple_shapes_generate(self):
        """All simple shapes generate valid Verilog."""
        generators = [
            generate_xor_shape,
            generate_and_shape,
            generate_or_shape,
            generate_not_shape,
            generate_nand_shape,
            generate_nor_shape,
            generate_xnor_shape,
            generate_nop_shape,
        ]

        for gen in generators:
            shape = gen(bits=8)
            verilog = shape_to_verilog(shape)

            # Should have valid structure
            assert "module xorpu_" in verilog
            assert "endmodule" in verilog

    def test_different_bit_widths(self):
        """Works for various bit widths."""
        for bits in [4, 8, 16, 32]:
            shape = generate_xor_shape(bits=bits)
            verilog = shape_to_verilog(shape, include_header=False)

            assert f"[{bits-1}:0]" in verilog

    def test_exported_files_are_valid(self):
        """Exported files contain valid Verilog."""
        shapes = {
            'xor': generate_xor_shape(bits=8),
            'and': generate_and_shape(bits=8),
            'or': generate_or_shape(bits=8),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            files = export_verilog(shapes, tmpdir)

            for f in files:
                with open(f, 'r') as file:
                    content = file.read()
                    assert "module" in content
                    assert "endmodule" in content


# =============================================================================
# IMPORT FROM FORGE
# =============================================================================

class TestForgeExports:
    """Test that verilog functions are exported from forge."""

    def test_shape_to_verilog_import(self):
        from trix.forge import shape_to_verilog
        assert callable(shape_to_verilog)

    def test_export_verilog_import(self):
        from trix.forge import export_verilog
        assert callable(export_verilog)

    def test_generate_testbench_import(self):
        from trix.forge import generate_testbench
        assert callable(generate_testbench)
