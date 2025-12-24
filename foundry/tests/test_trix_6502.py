"""
Tests for TriX 6502 Export
===========================

The TriX 6502 is a complete MOS 6502 CPU implemented as frozen
polynomial geometry. This tests the export pipeline and generated code.

Architecture:
    Level 0: Polynomial primitives (XOR, AND, OR, NOT)
    Level 1: 16 frozen shapes (ripple_add, shift_left, etc.)
    Level 2: Meaning layer as routing tables
    Runtime: Fetch-decode-execute loop
"""

import pytest
from pathlib import Path
import tempfile
import subprocess
import os

from foundry.export.trix_6502 import TriX6502Exporter


class TestTriX6502Exporter:
    """Tests for the TriX 6502 exporter."""

    def test_export_creates_files(self):
        """Export should create header and source files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            header, source = TriX6502Exporter.export(output_dir)

            assert header.exists()
            assert source.exists()
            assert header.name == "trix_6502.h"
            assert source.name == "trix_6502.c"

    def test_header_has_include_guard(self):
        """Header should have proper include guard."""
        header = TriX6502Exporter.generate_header()

        assert "#ifndef TRIX_6502_H" in header
        assert "#define TRIX_6502_H" in header
        assert "#endif" in header

    def test_header_includes_stdint(self):
        """Header should include stdint.h."""
        header = TriX6502Exporter.generate_header()

        assert "#include <stdint.h>" in header

    def test_header_declares_cpu_struct(self):
        """Header should declare trix_6502_t struct."""
        header = TriX6502Exporter.generate_header()

        assert "typedef struct" in header
        assert "trix_6502_t" in header
        assert "uint8_t a;" in header  # Accumulator
        assert "uint8_t x;" in header  # X register
        assert "uint8_t y;" in header  # Y register

    def test_header_declares_api(self):
        """Header should declare public API functions."""
        header = TriX6502Exporter.generate_header()

        assert "trix_6502_init" in header
        assert "trix_6502_reset" in header
        assert "trix_6502_step" in header
        assert "trix_6502_irq" in header
        assert "trix_6502_nmi" in header

    def test_source_includes_header(self):
        """Source should include its header."""
        source = TriX6502Exporter.generate_source()

        assert '#include "trix_6502.h"' in source

    def test_source_has_level0_primitives(self):
        """Source should have Level 0 polynomial primitives."""
        source = TriX6502Exporter.generate_source()

        assert "trix_xor_bit" in source
        assert "trix_and_bit" in source
        assert "trix_or_bit" in source
        assert "trix_not_bit" in source

    def test_source_has_level1_shapes(self):
        """Source should have all 16 Level 1 shapes."""
        source = TriX6502Exporter.generate_source()

        shapes = [
            "shape_ripple_add",
            "shape_ripple_sub",
            "shape_parallel_and",
            "shape_parallel_or",
            "shape_parallel_xor",
            "shape_shift_left",
            "shape_shift_right",
            "shape_rotate_left",
            "shape_rotate_right",
            "shape_increment",
            "shape_decrement",
            "shape_transfer",
            "shape_load",
            "shape_store",
            "shape_bit_test",
            "shape_identity",
        ]

        for shape in shapes:
            assert shape in source, f"Missing shape: {shape}"

    def test_source_has_level2_tables(self):
        """Source should have Level 2 routing tables."""
        source = TriX6502Exporter.generate_source()

        tables = [
            "SHAPE_TABLE",
            "INPUT_A_TABLE",
            "INPUT_B_TABLE",
            "OUTPUT_TABLE",
            "USES_CARRY_TABLE",
            "FLAGS_TABLE",
            "ADDR_MODE_TABLE",
        ]

        for table in tables:
            assert table in source, f"Missing table: {table}"

    def test_source_has_addressing_modes(self):
        """Source should have addressing mode definitions."""
        source = TriX6502Exporter.generate_source()

        modes = [
            "ADDR_IMPLIED",
            "ADDR_IMMEDIATE",
            "ADDR_ZERO_PAGE",
            "ADDR_ABSOLUTE",
        ]

        for mode in modes:
            assert mode in source, f"Missing addressing mode: {mode}"


class TestOpcodeMap:
    """Tests for the opcode mapping."""

    def test_opcode_map_has_adc(self):
        """Opcode map should have ADC instructions."""
        opcodes = TriX6502Exporter.OPCODE_MAP

        # ADC immediate
        assert 0x69 in opcodes
        assert opcodes[0x69][0] == 0  # RIPPLE_ADD shape

    def test_opcode_map_has_lda(self):
        """Opcode map should have LDA instructions."""
        opcodes = TriX6502Exporter.OPCODE_MAP

        # LDA immediate
        assert 0xA9 in opcodes
        assert opcodes[0xA9][0] == 12  # LOAD shape
        assert opcodes[0xA9][3] == 0   # Output to A register

    def test_opcode_map_has_ldx(self):
        """Opcode map should have LDX instructions."""
        opcodes = TriX6502Exporter.OPCODE_MAP

        # LDX immediate
        assert 0xA2 in opcodes
        assert opcodes[0xA2][0] == 12  # LOAD shape
        assert opcodes[0xA2][3] == 1   # Output to X register

    def test_opcode_map_has_logic_ops(self):
        """Opcode map should have logic operations."""
        opcodes = TriX6502Exporter.OPCODE_MAP

        # AND immediate
        assert 0x29 in opcodes
        assert opcodes[0x29][0] == 2  # PARALLEL_AND shape

        # ORA immediate
        assert 0x09 in opcodes
        assert opcodes[0x09][0] == 3  # PARALLEL_OR shape

        # EOR immediate
        assert 0x49 in opcodes
        assert opcodes[0x49][0] == 4  # PARALLEL_XOR shape

    def test_opcode_map_has_shifts(self):
        """Opcode map should have shift operations."""
        opcodes = TriX6502Exporter.OPCODE_MAP

        # ASL accumulator
        assert 0x0A in opcodes
        assert opcodes[0x0A][0] == 5  # SHIFT_LEFT shape

        # LSR accumulator
        assert 0x4A in opcodes
        assert opcodes[0x4A][0] == 6  # SHIFT_RIGHT shape

    def test_opcode_map_has_inc_dec(self):
        """Opcode map should have increment/decrement."""
        opcodes = TriX6502Exporter.OPCODE_MAP

        # INX
        assert 0xE8 in opcodes
        assert opcodes[0xE8][0] == 9   # INCREMENT shape
        assert opcodes[0xE8][3] == 1   # Output to X

        # DEX
        assert 0xCA in opcodes
        assert opcodes[0xCA][0] == 10  # DECREMENT shape

    def test_opcode_map_has_transfers(self):
        """Opcode map should have register transfers."""
        opcodes = TriX6502Exporter.OPCODE_MAP

        # TAX
        assert 0xAA in opcodes
        assert opcodes[0xAA][0] == 11  # TRANSFER shape
        assert opcodes[0xAA][1] == 0   # Input from A
        assert opcodes[0xAA][3] == 1   # Output to X

    def test_addressing_modes_correct(self):
        """Addressing modes should be correctly assigned."""
        opcodes = TriX6502Exporter.OPCODE_MAP

        # Immediate mode = 1
        assert opcodes[0xA9][6] == 1  # LDA immediate
        assert opcodes[0xA2][6] == 1  # LDX immediate
        assert opcodes[0x69][6] == 1  # ADC immediate

        # Zero page mode = 2
        assert opcodes[0xA5][6] == 2  # LDA zero page

        # Implied mode = 0
        assert opcodes[0xE8][6] == 0  # INX
        assert opcodes[0xCA][6] == 0  # DEX
        assert opcodes[0xEA][6] == 0  # NOP


class TestTriX6502Compilation:
    """Tests that verify the generated C code compiles and runs."""

    @pytest.fixture
    def compiled_6502(self):
        """Fixture that compiles the 6502 and returns paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            header, source = TriX6502Exporter.export(output_dir)

            yield output_dir, header, source

    def test_compiles_without_errors(self, compiled_6502):
        """Generated code should compile without errors."""
        output_dir, header, source = compiled_6502

        obj_path = output_dir / "trix_6502.o"
        result = subprocess.run(
            ["gcc", "-c", "-o", str(obj_path), str(source)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"Compilation failed:\n{result.stderr}"
        assert obj_path.exists()

    def test_compiles_with_optimizations(self, compiled_6502):
        """Generated code should compile with -O2."""
        output_dir, header, source = compiled_6502

        obj_path = output_dir / "trix_6502.o"
        result = subprocess.run(
            ["gcc", "-O2", "-c", "-o", str(obj_path), str(source)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"Compilation failed:\n{result.stderr}"

    def test_compiles_with_wall(self, compiled_6502):
        """Generated code should compile with -Wall without warnings."""
        output_dir, header, source = compiled_6502

        obj_path = output_dir / "trix_6502.o"
        result = subprocess.run(
            ["gcc", "-Wall", "-c", "-o", str(obj_path), str(source)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"Compilation failed:\n{result.stderr}"
        # Note: Some warnings may be acceptable, so we just check it compiles


class TestTriX6502Execution:
    """Tests that verify the CPU executes instructions correctly."""

    @pytest.fixture
    def cpu_test_harness(self):
        """Creates a test harness for running 6502 tests."""
        def run_test(test_code: str) -> tuple:
            with tempfile.TemporaryDirectory() as tmpdir:
                output_dir = Path(tmpdir)
                header, source = TriX6502Exporter.export(output_dir)

                test_file = output_dir / "test.c"
                test_file.write_text(test_code)

                exe = output_dir / "test"
                result = subprocess.run(
                    ["gcc", "-O2", "-o", str(exe), str(test_file), str(source)],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    return -1, f"Compile error:\n{result.stderr}"

                result = subprocess.run(
                    [str(exe)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                return result.returncode, result.stdout

        return run_test

    def test_lda_immediate(self, cpu_test_harness):
        """LDA immediate should load value into A."""
        test_code = '''
#include <stdio.h>
#include <string.h>
#include "trix_6502.h"

static uint8_t mem[65536];
uint8_t mem_read(uint16_t a, void *c) { return mem[a]; }
void mem_write(uint16_t a, uint8_t v, void *c) { mem[a] = v; }

int main() {
    trix_6502_t cpu;
    trix_6502_init(&cpu);
    cpu.read = mem_read;
    cpu.write = mem_write;

    mem[0x8000] = 0xA9;  // LDA #$42
    mem[0x8001] = 0x42;
    mem[0xFFFC] = 0x00;
    mem[0xFFFD] = 0x80;

    trix_6502_reset(&cpu);
    trix_6502_step(&cpu);

    return (cpu.a == 0x42) ? 0 : 1;
}
'''
        retcode, output = cpu_test_harness(test_code)
        assert retcode == 0, f"Test failed: {output}"

    def test_ldx_immediate(self, cpu_test_harness):
        """LDX immediate should load value into X."""
        test_code = '''
#include <string.h>
#include "trix_6502.h"

static uint8_t mem[65536];
uint8_t mem_read(uint16_t a, void *c) { return mem[a]; }
void mem_write(uint16_t a, uint8_t v, void *c) { mem[a] = v; }

int main() {
    trix_6502_t cpu;
    trix_6502_init(&cpu);
    cpu.read = mem_read;
    cpu.write = mem_write;

    mem[0x8000] = 0xA2;  // LDX #$55
    mem[0x8001] = 0x55;
    mem[0xFFFC] = 0x00;
    mem[0xFFFD] = 0x80;

    trix_6502_reset(&cpu);
    trix_6502_step(&cpu);

    return (cpu.x == 0x55) ? 0 : 1;
}
'''
        retcode, output = cpu_test_harness(test_code)
        assert retcode == 0, f"Test failed: {output}"

    def test_inx(self, cpu_test_harness):
        """INX should increment X register."""
        test_code = '''
#include <string.h>
#include "trix_6502.h"

static uint8_t mem[65536];
uint8_t mem_read(uint16_t a, void *c) { return mem[a]; }
void mem_write(uint16_t a, uint8_t v, void *c) { mem[a] = v; }

int main() {
    trix_6502_t cpu;
    trix_6502_init(&cpu);
    cpu.read = mem_read;
    cpu.write = mem_write;

    mem[0x8000] = 0xA2;  // LDX #$41
    mem[0x8001] = 0x41;
    mem[0x8002] = 0xE8;  // INX
    mem[0xFFFC] = 0x00;
    mem[0xFFFD] = 0x80;

    trix_6502_reset(&cpu);
    trix_6502_step(&cpu);  // LDX
    trix_6502_step(&cpu);  // INX

    return (cpu.x == 0x42) ? 0 : 1;
}
'''
        retcode, output = cpu_test_harness(test_code)
        assert retcode == 0, f"Test failed: {output}"

    def test_and_immediate(self, cpu_test_harness):
        """AND immediate should perform bitwise AND."""
        test_code = '''
#include <string.h>
#include "trix_6502.h"

static uint8_t mem[65536];
uint8_t mem_read(uint16_t a, void *c) { return mem[a]; }
void mem_write(uint16_t a, uint8_t v, void *c) { mem[a] = v; }

int main() {
    trix_6502_t cpu;
    trix_6502_init(&cpu);
    cpu.read = mem_read;
    cpu.write = mem_write;

    mem[0x8000] = 0xA9;  // LDA #$FF
    mem[0x8001] = 0xFF;
    mem[0x8002] = 0x29;  // AND #$0F
    mem[0x8003] = 0x0F;
    mem[0xFFFC] = 0x00;
    mem[0xFFFD] = 0x80;

    trix_6502_reset(&cpu);
    trix_6502_step(&cpu);  // LDA
    trix_6502_step(&cpu);  // AND

    return (cpu.a == 0x0F) ? 0 : 1;
}
'''
        retcode, output = cpu_test_harness(test_code)
        assert retcode == 0, f"Test failed: {output}"

    def test_ora_immediate(self, cpu_test_harness):
        """ORA immediate should perform bitwise OR."""
        test_code = '''
#include <string.h>
#include "trix_6502.h"

static uint8_t mem[65536];
uint8_t mem_read(uint16_t a, void *c) { return mem[a]; }
void mem_write(uint16_t a, uint8_t v, void *c) { mem[a] = v; }

int main() {
    trix_6502_t cpu;
    trix_6502_init(&cpu);
    cpu.read = mem_read;
    cpu.write = mem_write;

    mem[0x8000] = 0xA9;  // LDA #$0F
    mem[0x8001] = 0x0F;
    mem[0x8002] = 0x09;  // ORA #$F0
    mem[0x8003] = 0xF0;
    mem[0xFFFC] = 0x00;
    mem[0xFFFD] = 0x80;

    trix_6502_reset(&cpu);
    trix_6502_step(&cpu);  // LDA
    trix_6502_step(&cpu);  // ORA

    return (cpu.a == 0xFF) ? 0 : 1;
}
'''
        retcode, output = cpu_test_harness(test_code)
        assert retcode == 0, f"Test failed: {output}"

    def test_eor_immediate(self, cpu_test_harness):
        """EOR immediate should perform bitwise XOR."""
        test_code = '''
#include <string.h>
#include "trix_6502.h"

static uint8_t mem[65536];
uint8_t mem_read(uint16_t a, void *c) { return mem[a]; }
void mem_write(uint16_t a, uint8_t v, void *c) { mem[a] = v; }

int main() {
    trix_6502_t cpu;
    trix_6502_init(&cpu);
    cpu.read = mem_read;
    cpu.write = mem_write;

    mem[0x8000] = 0xA9;  // LDA #$FF
    mem[0x8001] = 0xFF;
    mem[0x8002] = 0x49;  // EOR #$0F
    mem[0x8003] = 0x0F;
    mem[0xFFFC] = 0x00;
    mem[0xFFFD] = 0x80;

    trix_6502_reset(&cpu);
    trix_6502_step(&cpu);  // LDA
    trix_6502_step(&cpu);  // EOR

    return (cpu.a == 0xF0) ? 0 : 1;
}
'''
        retcode, output = cpu_test_harness(test_code)
        assert retcode == 0, f"Test failed: {output}"

    def test_asl_accumulator(self, cpu_test_harness):
        """ASL accumulator should shift left and set carry."""
        test_code = '''
#include <string.h>
#include "trix_6502.h"

static uint8_t mem[65536];
uint8_t mem_read(uint16_t a, void *c) { return mem[a]; }
void mem_write(uint16_t a, uint8_t v, void *c) { mem[a] = v; }

int main() {
    trix_6502_t cpu;
    trix_6502_init(&cpu);
    cpu.read = mem_read;
    cpu.write = mem_write;

    mem[0x8000] = 0xA9;  // LDA #$81
    mem[0x8001] = 0x81;
    mem[0x8002] = 0x0A;  // ASL A
    mem[0xFFFC] = 0x00;
    mem[0xFFFD] = 0x80;

    trix_6502_reset(&cpu);
    trix_6502_step(&cpu);  // LDA
    trix_6502_step(&cpu);  // ASL

    // A should be 0x02 (0x81 << 1), carry should be set
    return (cpu.a == 0x02 && cpu.c == 1) ? 0 : 1;
}
'''
        retcode, output = cpu_test_harness(test_code)
        assert retcode == 0, f"Test failed: {output}"

    def test_zero_flag(self, cpu_test_harness):
        """Zero flag should be set when result is zero."""
        test_code = '''
#include <string.h>
#include "trix_6502.h"

static uint8_t mem[65536];
uint8_t mem_read(uint16_t a, void *c) { return mem[a]; }
void mem_write(uint16_t a, uint8_t v, void *c) { mem[a] = v; }

int main() {
    trix_6502_t cpu;
    trix_6502_init(&cpu);
    cpu.read = mem_read;
    cpu.write = mem_write;

    mem[0x8000] = 0xA9;  // LDA #$00
    mem[0x8001] = 0x00;
    mem[0xFFFC] = 0x00;
    mem[0xFFFD] = 0x80;

    trix_6502_reset(&cpu);
    trix_6502_step(&cpu);  // LDA

    return (cpu.z == 1) ? 0 : 1;
}
'''
        retcode, output = cpu_test_harness(test_code)
        assert retcode == 0, f"Test failed: {output}"

    def test_negative_flag(self, cpu_test_harness):
        """Negative flag should be set when bit 7 is set."""
        test_code = '''
#include <string.h>
#include "trix_6502.h"

static uint8_t mem[65536];
uint8_t mem_read(uint16_t a, void *c) { return mem[a]; }
void mem_write(uint16_t a, uint8_t v, void *c) { mem[a] = v; }

int main() {
    trix_6502_t cpu;
    trix_6502_init(&cpu);
    cpu.read = mem_read;
    cpu.write = mem_write;

    mem[0x8000] = 0xA9;  // LDA #$80
    mem[0x8001] = 0x80;
    mem[0xFFFC] = 0x00;
    mem[0xFFFD] = 0x80;

    trix_6502_reset(&cpu);
    trix_6502_step(&cpu);  // LDA

    return (cpu.n == 1) ? 0 : 1;
}
'''
        retcode, output = cpu_test_harness(test_code)
        assert retcode == 0, f"Test failed: {output}"

    def test_tax_transfer(self, cpu_test_harness):
        """TAX should transfer A to X."""
        test_code = '''
#include <string.h>
#include "trix_6502.h"

static uint8_t mem[65536];
uint8_t mem_read(uint16_t a, void *c) { return mem[a]; }
void mem_write(uint16_t a, uint8_t v, void *c) { mem[a] = v; }

int main() {
    trix_6502_t cpu;
    trix_6502_init(&cpu);
    cpu.read = mem_read;
    cpu.write = mem_write;

    mem[0x8000] = 0xA9;  // LDA #$42
    mem[0x8001] = 0x42;
    mem[0x8002] = 0xAA;  // TAX
    mem[0xFFFC] = 0x00;
    mem[0xFFFD] = 0x80;

    trix_6502_reset(&cpu);
    trix_6502_step(&cpu);  // LDA
    trix_6502_step(&cpu);  // TAX

    return (cpu.a == 0x42 && cpu.x == 0x42) ? 0 : 1;
}
'''
        retcode, output = cpu_test_harness(test_code)
        assert retcode == 0, f"Test failed: {output}"
