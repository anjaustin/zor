"""
Tests for TriX Forge - Verilog to Frozen C pipeline.
"""

import pytest
from foundry.forge import freeze, ForgeError
from foundry.forge.parser import parse, ParseError
from foundry.forge.composer import compose
from foundry.forge.verifier import verify_exhaustive, quick_verify


class TestParser:
    """Test Verilog parser."""

    def test_parse_simple_module(self):
        """Parse simple module declaration."""
        verilog = """
        module test(input a, output y);
            assign y = a;
        endmodule
        """
        module = parse(verilog)
        assert module.name == "test"
        assert "a" in module.ports
        assert "y" in module.ports

    def test_parse_multi_bit_ports(self):
        """Parse multi-bit port declarations."""
        verilog = """
        module adder(input [3:0] a, input [3:0] b, output [4:0] sum);
            assign sum = a + b;
        endmodule
        """
        module = parse(verilog)
        assert module.ports["a"].width == 4
        assert module.ports["b"].width == 4
        assert module.ports["sum"].width == 5

    def test_parse_and_gate(self):
        """Parse AND gate."""
        verilog = """
        module and_gate(input a, input b, output y);
            assign y = a & b;
        endmodule
        """
        module = parse(verilog)
        assert len(module.assignments) == 1
        assert module.assignments[0].target == "y"

    def test_parse_not_gate(self):
        """Parse NOT gate."""
        verilog = """
        module not_gate(input a, output y);
            assign y = ~a;
        endmodule
        """
        module = parse(verilog)
        assert len(module.assignments) == 1


class TestComposer:
    """Test polynomial composer."""

    def test_compose_not_gate(self):
        """Compose NOT gate polynomial."""
        verilog = """
        module not_gate(input a, output y);
            assign y = ~a;
        endmodule
        """
        module = parse(verilog)
        exprs = compose(module)
        assert len(exprs) == 1
        # NOT(a) = 1 - a
        assert "1 -" in exprs[0].polynomial or "(1 - " in exprs[0].polynomial

    def test_compose_and_gate(self):
        """Compose AND gate polynomial."""
        verilog = """
        module and_gate(input a, input b, output y);
            assign y = a & b;
        endmodule
        """
        module = parse(verilog)
        exprs = compose(module)
        assert len(exprs) == 1
        # AND(a, b) = a * b
        assert "*" in exprs[0].polynomial

    def test_compose_xor_gate(self):
        """Compose XOR gate polynomial."""
        verilog = """
        module xor_gate(input a, input b, output y);
            assign y = a ^ b;
        endmodule
        """
        module = parse(verilog)
        exprs = compose(module)
        assert len(exprs) == 1
        # XOR(a, b) = a + b - 2*a*b
        poly = exprs[0].polynomial
        assert "+" in poly and "-" in poly and "*" in poly


class TestFreeze:
    """Test complete freeze pipeline."""

    def test_freeze_not_gate(self):
        """Freeze NOT gate."""
        verilog = """
        module not_gate(input a, output y);
            assign y = ~a;
        endmodule
        """
        header, source = freeze(verilog)
        assert "frozen_not_gate" in header
        assert "frozen_not_gate" in source
        assert quick_verify(header, source, "not_gate")

    def test_freeze_and_gate(self):
        """Freeze AND gate."""
        verilog = """
        module and_gate(input a, input b, output y);
            assign y = a & b;
        endmodule
        """
        header, source = freeze(verilog)
        assert quick_verify(header, source, "and_gate")

    def test_freeze_4bit_adder(self):
        """Freeze 4-bit adder."""
        verilog = """
        module adder4(input [3:0] a, input [3:0] b, output [4:0] sum);
            assign sum = a + b;
        endmodule
        """
        header, source = freeze(verilog)
        assert quick_verify(header, source, "adder4")

    def test_freeze_error_no_module(self):
        """Error on missing module."""
        with pytest.raises(ForgeError):
            freeze("// no module here")

    def test_freeze_error_no_outputs(self):
        """Error on module with no outputs."""
        verilog = """
        module bad(input a, input b);
        endmodule
        """
        with pytest.raises(ForgeError):
            freeze(verilog)


class TestVerification:
    """Test exhaustive verification."""

    def test_verify_not_gate(self):
        """Verify NOT gate exhaustively."""
        verilog = """
        module not_gate(input a, output y);
            assign y = ~a;
        endmodule
        """
        header, source = freeze(verilog)
        passed, tested, failed = verify_exhaustive(
            header, source, "not_gate", 1, None
        )
        assert passed
        assert tested == 2
        assert failed == 0

    def test_verify_and_gate(self):
        """Verify AND gate exhaustively."""
        verilog = """
        module and_gate(input a, input b, output y);
            assign y = a & b;
        endmodule
        """
        header, source = freeze(verilog)
        passed, tested, failed = verify_exhaustive(
            header, source, "and_gate", 2, None
        )
        assert passed
        assert tested == 4
        assert failed == 0

    def test_verify_4bit_adder(self):
        """Verify 4-bit adder exhaustively (256 cases)."""
        verilog = """
        module adder4(input [3:0] a, input [3:0] b, output [4:0] sum);
            assign sum = a + b;
        endmodule
        """
        header, source = freeze(verilog)
        passed, tested, failed = verify_exhaustive(
            header, source, "adder4", 8, None
        )
        assert passed
        assert tested == 256
        assert failed == 0
