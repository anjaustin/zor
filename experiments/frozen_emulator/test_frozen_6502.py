#!/usr/bin/env python3
"""
Comprehensive tests for the Frozen 6502 Emulator.

Tests verify that frozen shapes compute correctly and that
the emulator behaves as a true 6502.
"""

import pytest
from frozen_6502 import (
    CPU6502, OPCODES,
    frozen_add, frozen_sub, frozen_and, frozen_or, frozen_xor,
    frozen_asl, frozen_lsr, frozen_rol, frozen_ror,
    frozen_inc, frozen_dec,
    disassemble, dump_state, trace,
)


# =============================================================================
# FROZEN SHAPE TESTS
# =============================================================================

class TestFrozenShapes:
    """Test the frozen mathematical operations."""

    def test_frozen_add_basic(self):
        """Test basic addition without carry."""
        assert frozen_add(0x00, 0x00, 0) == (0x00, 0)
        assert frozen_add(0x10, 0x20, 0) == (0x30, 0)
        assert frozen_add(0x7F, 0x01, 0) == (0x80, 0)

    def test_frozen_add_carry_out(self):
        """Test addition producing carry."""
        assert frozen_add(0xFF, 0x01, 0) == (0x00, 1)
        assert frozen_add(0x80, 0x80, 0) == (0x00, 1)
        assert frozen_add(0xFF, 0xFF, 0) == (0xFE, 1)

    def test_frozen_add_carry_in(self):
        """Test addition with carry input."""
        assert frozen_add(0x00, 0x00, 1) == (0x01, 0)
        assert frozen_add(0xFF, 0x00, 1) == (0x00, 1)
        assert frozen_add(0xFF, 0xFF, 1) == (0xFF, 1)

    def test_frozen_sub_basic(self):
        """Test basic subtraction."""
        assert frozen_sub(0x50, 0x10, 1) == (0x40, 1)  # No borrow
        assert frozen_sub(0x50, 0x50, 1) == (0x00, 1)
        assert frozen_sub(0x00, 0x00, 1) == (0x00, 1)

    def test_frozen_sub_borrow(self):
        """Test subtraction with borrow."""
        assert frozen_sub(0x10, 0x20, 1) == (0xF0, 0)  # Borrow occurred
        assert frozen_sub(0x00, 0x01, 1) == (0xFF, 0)

    def test_frozen_sub_with_borrow_in(self):
        """Test subtraction with incoming borrow."""
        assert frozen_sub(0x50, 0x10, 0) == (0x3F, 1)  # C=0 means borrow
        assert frozen_sub(0x00, 0x00, 0) == (0xFF, 0)

    def test_frozen_and(self):
        """Test bitwise AND."""
        assert frozen_and(0xFF, 0xFF) == 0xFF
        assert frozen_and(0xFF, 0x00) == 0x00
        assert frozen_and(0x0F, 0xF0) == 0x00
        assert frozen_and(0xAA, 0xFF) == 0xAA

    def test_frozen_or(self):
        """Test bitwise OR."""
        assert frozen_or(0x00, 0x00) == 0x00
        assert frozen_or(0xFF, 0x00) == 0xFF
        assert frozen_or(0x0F, 0xF0) == 0xFF
        assert frozen_or(0xAA, 0x55) == 0xFF

    def test_frozen_xor(self):
        """Test bitwise XOR."""
        assert frozen_xor(0x00, 0x00) == 0x00
        assert frozen_xor(0xFF, 0xFF) == 0x00
        assert frozen_xor(0xFF, 0x00) == 0xFF
        assert frozen_xor(0xAA, 0x55) == 0xFF

    def test_frozen_asl(self):
        """Test arithmetic shift left."""
        assert frozen_asl(0x00) == (0x00, 0)
        assert frozen_asl(0x01) == (0x02, 0)
        assert frozen_asl(0x80) == (0x00, 1)
        assert frozen_asl(0x81) == (0x02, 1)
        assert frozen_asl(0x40) == (0x80, 0)

    def test_frozen_lsr(self):
        """Test logical shift right."""
        assert frozen_lsr(0x00) == (0x00, 0)
        assert frozen_lsr(0x02) == (0x01, 0)
        assert frozen_lsr(0x01) == (0x00, 1)
        assert frozen_lsr(0x81) == (0x40, 1)
        assert frozen_lsr(0x80) == (0x40, 0)

    def test_frozen_rol(self):
        """Test rotate left through carry."""
        assert frozen_rol(0x00, 0) == (0x00, 0)
        assert frozen_rol(0x00, 1) == (0x01, 0)
        assert frozen_rol(0x80, 0) == (0x00, 1)
        assert frozen_rol(0x80, 1) == (0x01, 1)
        assert frozen_rol(0x55, 0) == (0xAA, 0)

    def test_frozen_ror(self):
        """Test rotate right through carry."""
        assert frozen_ror(0x00, 0) == (0x00, 0)
        assert frozen_ror(0x00, 1) == (0x80, 0)
        assert frozen_ror(0x01, 0) == (0x00, 1)
        assert frozen_ror(0x01, 1) == (0x80, 1)
        assert frozen_ror(0xAA, 0) == (0x55, 0)

    def test_frozen_inc(self):
        """Test increment."""
        assert frozen_inc(0x00) == 0x01
        assert frozen_inc(0xFE) == 0xFF
        assert frozen_inc(0xFF) == 0x00  # Wrap

    def test_frozen_dec(self):
        """Test decrement."""
        assert frozen_dec(0x01) == 0x00
        assert frozen_dec(0xFF) == 0xFE
        assert frozen_dec(0x00) == 0xFF  # Wrap


# =============================================================================
# CPU INSTRUCTION TESTS
# =============================================================================

class TestCPUInstructions:
    """Test individual CPU instructions."""

    @pytest.fixture
    def cpu(self):
        return CPU6502()

    # --- Load/Store ---

    def test_lda_immediate(self, cpu):
        cpu.memory[0x0000] = 0xA9  # LDA #$42
        cpu.memory[0x0001] = 0x42
        cpu.step()
        assert cpu.a == 0x42
        assert cpu.pc == 0x02

    def test_lda_zero_page(self, cpu):
        cpu.memory[0x0000] = 0xA5  # LDA $10
        cpu.memory[0x0001] = 0x10
        cpu.memory[0x0010] = 0x77
        cpu.step()
        assert cpu.a == 0x77

    def test_lda_absolute(self, cpu):
        cpu.memory[0x0000] = 0xAD  # LDA $1234
        cpu.memory[0x0001] = 0x34
        cpu.memory[0x0002] = 0x12
        cpu.memory[0x1234] = 0x88
        cpu.step()
        assert cpu.a == 0x88

    def test_lda_sets_zero_flag(self, cpu):
        cpu.memory[0x0000] = 0xA9  # LDA #$00
        cpu.memory[0x0001] = 0x00
        cpu.step()
        assert cpu.flag_z == 1

    def test_lda_sets_negative_flag(self, cpu):
        cpu.memory[0x0000] = 0xA9  # LDA #$80
        cpu.memory[0x0001] = 0x80
        cpu.step()
        assert cpu.flag_n == 1

    def test_sta_zero_page(self, cpu):
        cpu.a = 0x42
        cpu.memory[0x0000] = 0x85  # STA $10
        cpu.memory[0x0001] = 0x10
        cpu.step()
        assert cpu.memory[0x0010] == 0x42

    def test_ldx_immediate(self, cpu):
        cpu.memory[0x0000] = 0xA2  # LDX #$55
        cpu.memory[0x0001] = 0x55
        cpu.step()
        assert cpu.x == 0x55

    def test_ldy_immediate(self, cpu):
        cpu.memory[0x0000] = 0xA0  # LDY #$66
        cpu.memory[0x0001] = 0x66
        cpu.step()
        assert cpu.y == 0x66

    # --- Arithmetic ---

    def test_adc_immediate(self, cpu):
        cpu.a = 0x10
        cpu.set_flag(0, False)
        cpu.memory[0x0000] = 0x69  # ADC #$20
        cpu.memory[0x0001] = 0x20
        cpu.step()
        assert cpu.a == 0x30
        assert cpu.flag_c == 0

    def test_adc_with_carry_in(self, cpu):
        cpu.a = 0x10
        cpu.set_flag(0, True)
        cpu.memory[0x0000] = 0x69  # ADC #$20
        cpu.memory[0x0001] = 0x20
        cpu.step()
        assert cpu.a == 0x31

    def test_adc_overflow(self, cpu):
        cpu.a = 0x7F
        cpu.set_flag(0, False)
        cpu.memory[0x0000] = 0x69  # ADC #$01
        cpu.memory[0x0001] = 0x01
        cpu.step()
        assert cpu.a == 0x80
        assert cpu.flag_v == 1  # Overflow: positive + positive = negative

    def test_sbc_immediate(self, cpu):
        cpu.a = 0x50
        cpu.set_flag(0, True)  # No borrow
        cpu.memory[0x0000] = 0xE9  # SBC #$10
        cpu.memory[0x0001] = 0x10
        cpu.step()
        assert cpu.a == 0x40
        assert cpu.flag_c == 1

    def test_sbc_with_borrow(self, cpu):
        cpu.a = 0x50
        cpu.set_flag(0, False)  # Borrow
        cpu.memory[0x0000] = 0xE9  # SBC #$10
        cpu.memory[0x0001] = 0x10
        cpu.step()
        assert cpu.a == 0x3F

    # --- Logic ---

    def test_and_immediate(self, cpu):
        cpu.a = 0xFF
        cpu.memory[0x0000] = 0x29  # AND #$0F
        cpu.memory[0x0001] = 0x0F
        cpu.step()
        assert cpu.a == 0x0F

    def test_ora_immediate(self, cpu):
        cpu.a = 0x0F
        cpu.memory[0x0000] = 0x09  # ORA #$F0
        cpu.memory[0x0001] = 0xF0
        cpu.step()
        assert cpu.a == 0xFF

    def test_eor_immediate(self, cpu):
        cpu.a = 0xFF
        cpu.memory[0x0000] = 0x49  # EOR #$0F
        cpu.memory[0x0001] = 0x0F
        cpu.step()
        assert cpu.a == 0xF0

    # --- Shifts ---

    def test_asl_accumulator(self, cpu):
        cpu.a = 0x81
        cpu.memory[0x0000] = 0x0A  # ASL A
        cpu.step()
        assert cpu.a == 0x02
        assert cpu.flag_c == 1

    def test_lsr_accumulator(self, cpu):
        cpu.a = 0x81
        cpu.memory[0x0000] = 0x4A  # LSR A
        cpu.step()
        assert cpu.a == 0x40
        assert cpu.flag_c == 1

    def test_rol_accumulator(self, cpu):
        cpu.a = 0x81
        cpu.set_flag(0, True)
        cpu.memory[0x0000] = 0x2A  # ROL A
        cpu.step()
        assert cpu.a == 0x03
        assert cpu.flag_c == 1

    def test_ror_accumulator(self, cpu):
        cpu.a = 0x81
        cpu.set_flag(0, True)
        cpu.memory[0x0000] = 0x6A  # ROR A
        cpu.step()
        assert cpu.a == 0xC0
        assert cpu.flag_c == 1

    # --- Increment/Decrement ---

    def test_inx(self, cpu):
        cpu.x = 0x10
        cpu.memory[0x0000] = 0xE8  # INX
        cpu.step()
        assert cpu.x == 0x11

    def test_dex(self, cpu):
        cpu.x = 0x10
        cpu.memory[0x0000] = 0xCA  # DEX
        cpu.step()
        assert cpu.x == 0x0F

    def test_iny(self, cpu):
        cpu.y = 0x10
        cpu.memory[0x0000] = 0xC8  # INY
        cpu.step()
        assert cpu.y == 0x11

    def test_dey(self, cpu):
        cpu.y = 0x10
        cpu.memory[0x0000] = 0x88  # DEY
        cpu.step()
        assert cpu.y == 0x0F

    def test_inc_memory(self, cpu):
        cpu.memory[0x0000] = 0xE6  # INC $10
        cpu.memory[0x0001] = 0x10
        cpu.memory[0x0010] = 0x42
        cpu.step()
        assert cpu.memory[0x0010] == 0x43

    def test_dec_memory(self, cpu):
        cpu.memory[0x0000] = 0xC6  # DEC $10
        cpu.memory[0x0001] = 0x10
        cpu.memory[0x0010] = 0x42
        cpu.step()
        assert cpu.memory[0x0010] == 0x41

    # --- Compare ---

    def test_cmp_equal(self, cpu):
        cpu.a = 0x50
        cpu.memory[0x0000] = 0xC9  # CMP #$50
        cpu.memory[0x0001] = 0x50
        cpu.step()
        assert cpu.flag_z == 1
        assert cpu.flag_c == 1

    def test_cmp_greater(self, cpu):
        cpu.a = 0x50
        cpu.memory[0x0000] = 0xC9  # CMP #$30
        cpu.memory[0x0001] = 0x30
        cpu.step()
        assert cpu.flag_z == 0
        assert cpu.flag_c == 1

    def test_cmp_less(self, cpu):
        cpu.a = 0x30
        cpu.memory[0x0000] = 0xC9  # CMP #$50
        cpu.memory[0x0001] = 0x50
        cpu.step()
        assert cpu.flag_z == 0
        assert cpu.flag_c == 0

    # --- Branches ---

    def test_beq_taken(self, cpu):
        cpu.set_flag(1, True)  # Z = 1
        cpu.memory[0x0000] = 0xF0  # BEQ
        cpu.memory[0x0001] = 0x05
        cpu.step()
        assert cpu.pc == 0x07

    def test_beq_not_taken(self, cpu):
        cpu.set_flag(1, False)  # Z = 0
        cpu.memory[0x0000] = 0xF0  # BEQ
        cpu.memory[0x0001] = 0x05
        cpu.step()
        assert cpu.pc == 0x02

    def test_bne_taken(self, cpu):
        cpu.set_flag(1, False)  # Z = 0
        cpu.memory[0x0000] = 0xD0  # BNE
        cpu.memory[0x0001] = 0x05
        cpu.step()
        assert cpu.pc == 0x07

    def test_bcc_taken(self, cpu):
        cpu.set_flag(0, False)  # C = 0
        cpu.memory[0x0000] = 0x90  # BCC
        cpu.memory[0x0001] = 0x10
        cpu.step()
        assert cpu.pc == 0x12

    def test_bcs_taken(self, cpu):
        cpu.set_flag(0, True)  # C = 1
        cpu.memory[0x0000] = 0xB0  # BCS
        cpu.memory[0x0001] = 0x10
        cpu.step()
        assert cpu.pc == 0x12

    def test_branch_backward(self, cpu):
        cpu.pc = 0x1000
        cpu.set_flag(1, True)  # Z = 1
        cpu.memory[0x1000] = 0xF0  # BEQ
        cpu.memory[0x1001] = 0xFC  # -4
        cpu.step()
        assert cpu.pc == 0x0FFE

    # --- Jump/Subroutine ---

    def test_jmp_absolute(self, cpu):
        cpu.memory[0x0000] = 0x4C  # JMP $1234
        cpu.memory[0x0001] = 0x34
        cpu.memory[0x0002] = 0x12
        cpu.step()
        assert cpu.pc == 0x1234

    def test_jsr_rts(self, cpu):
        cpu.pc = 0x1000
        cpu.sp = 0xFF
        cpu.memory[0x1000] = 0x20  # JSR $2000
        cpu.memory[0x1001] = 0x00
        cpu.memory[0x1002] = 0x20
        cpu.memory[0x2000] = 0x60  # RTS
        cpu.step()  # JSR
        assert cpu.pc == 0x2000
        assert cpu.sp == 0xFD
        cpu.step()  # RTS
        assert cpu.pc == 0x1003
        assert cpu.sp == 0xFF

    # --- Stack ---

    def test_pha_pla(self, cpu):
        cpu.sp = 0xFF
        cpu.a = 0x42
        cpu.memory[0x0000] = 0x48  # PHA
        cpu.memory[0x0001] = 0xA9  # LDA #$00
        cpu.memory[0x0002] = 0x00
        cpu.memory[0x0003] = 0x68  # PLA
        cpu.step()  # PHA
        assert cpu.sp == 0xFE
        cpu.step()  # LDA #$00
        assert cpu.a == 0x00
        cpu.step()  # PLA
        assert cpu.a == 0x42
        assert cpu.sp == 0xFF

    def test_php_plp(self, cpu):
        cpu.p = 0xFF
        cpu.sp = 0xFF
        cpu.memory[0x0000] = 0x08  # PHP
        cpu.memory[0x0001] = 0x28  # PLP
        cpu.step()  # PHP
        cpu.p = 0x00
        cpu.step()  # PLP
        assert cpu.p & 0xCF == 0xCF  # N, V, D, I, Z, C

    # --- Flags ---

    def test_clc_sec(self, cpu):
        cpu.set_flag(0, True)
        cpu.memory[0x0000] = 0x18  # CLC
        cpu.step()
        assert cpu.flag_c == 0

        cpu.memory[0x0001] = 0x38  # SEC
        cpu.step()
        assert cpu.flag_c == 1

    def test_cli_sei(self, cpu):
        cpu.set_flag(2, True)
        cpu.memory[0x0000] = 0x58  # CLI
        cpu.step()
        assert cpu.flag_i == 0

        cpu.memory[0x0001] = 0x78  # SEI
        cpu.step()
        assert cpu.flag_i == 1

    # --- Transfers ---

    def test_tax_txa(self, cpu):
        cpu.a = 0x42
        cpu.memory[0x0000] = 0xAA  # TAX
        cpu.step()
        assert cpu.x == 0x42

        cpu.x = 0x55
        cpu.memory[0x0001] = 0x8A  # TXA
        cpu.step()
        assert cpu.a == 0x55

    def test_tay_tya(self, cpu):
        cpu.a = 0x42
        cpu.memory[0x0000] = 0xA8  # TAY
        cpu.step()
        assert cpu.y == 0x42

        cpu.y = 0x55
        cpu.memory[0x0001] = 0x98  # TYA
        cpu.step()
        assert cpu.a == 0x55


# =============================================================================
# ADDRESSING MODE TESTS
# =============================================================================

class TestAddressingModes:
    """Test all addressing modes."""

    @pytest.fixture
    def cpu(self):
        return CPU6502()

    def test_zero_page_x(self, cpu):
        cpu.x = 0x10
        cpu.memory[0x0000] = 0xB5  # LDA $20,X
        cpu.memory[0x0001] = 0x20
        cpu.memory[0x0030] = 0x42
        cpu.step()
        assert cpu.a == 0x42

    def test_zero_page_x_wrap(self, cpu):
        cpu.x = 0x10
        cpu.memory[0x0000] = 0xB5  # LDA $F0,X
        cpu.memory[0x0001] = 0xF0
        cpu.memory[0x0000] = 0xB5  # Wrap to $00
        cpu.memory[0x0001] = 0xF0
        # $F0 + $10 = $100, wraps to $00 in zero page
        cpu.memory[0x0000] = 0x99  # Put value there (conflicts, so use different addr)

        cpu2 = CPU6502()
        cpu2.x = 0x20
        cpu2.memory[0x0000] = 0xB5  # LDA $F0,X
        cpu2.memory[0x0001] = 0xF0
        cpu2.memory[0x0010] = 0x77  # $F0 + $20 = $110 -> $10
        cpu2.step()
        assert cpu2.a == 0x77

    def test_absolute_x(self, cpu):
        cpu.x = 0x10
        cpu.memory[0x0000] = 0xBD  # LDA $1000,X
        cpu.memory[0x0001] = 0x00
        cpu.memory[0x0002] = 0x10
        cpu.memory[0x1010] = 0x42
        cpu.step()
        assert cpu.a == 0x42

    def test_absolute_y(self, cpu):
        cpu.y = 0x20
        cpu.memory[0x0000] = 0xB9  # LDA $1000,Y
        cpu.memory[0x0001] = 0x00
        cpu.memory[0x0002] = 0x10
        cpu.memory[0x1020] = 0x55
        cpu.step()
        assert cpu.a == 0x55

    def test_indirect_x(self, cpu):
        cpu.x = 0x04
        cpu.memory[0x0000] = 0xA1  # LDA ($10,X)
        cpu.memory[0x0001] = 0x10
        cpu.memory[0x0014] = 0x00  # Low byte
        cpu.memory[0x0015] = 0x20  # High byte -> $2000
        cpu.memory[0x2000] = 0xAB
        cpu.step()
        assert cpu.a == 0xAB

    def test_indirect_y(self, cpu):
        cpu.y = 0x10
        cpu.memory[0x0000] = 0xB1  # LDA ($20),Y
        cpu.memory[0x0001] = 0x20
        cpu.memory[0x0020] = 0x00  # Low byte
        cpu.memory[0x0021] = 0x30  # High byte -> $3000
        cpu.memory[0x3010] = 0xCD  # $3000 + $10
        cpu.step()
        assert cpu.a == 0xCD

    def test_jmp_indirect(self, cpu):
        cpu.memory[0x0000] = 0x6C  # JMP ($1234)
        cpu.memory[0x0001] = 0x34
        cpu.memory[0x0002] = 0x12
        cpu.memory[0x1234] = 0x00  # Low byte
        cpu.memory[0x1235] = 0x80  # High byte -> $8000
        cpu.step()
        assert cpu.pc == 0x8000

    def test_jmp_indirect_page_bug(self, cpu):
        """Test 6502 JMP indirect page boundary bug."""
        cpu.memory[0x0000] = 0x6C  # JMP ($10FF)
        cpu.memory[0x0001] = 0xFF
        cpu.memory[0x0002] = 0x10
        cpu.memory[0x10FF] = 0x34  # Low byte
        cpu.memory[0x1000] = 0x12  # High byte (bug: wraps within page!)
        cpu.step()
        assert cpu.pc == 0x1234  # Not $1134!


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Test complete programs."""

    @pytest.fixture
    def cpu(self):
        return CPU6502()

    def test_count_to_ten(self, cpu):
        """Simple loop counting X from 0 to 10."""
        program = bytes([
            0xA2, 0x00,       # LDX #$00
            0xE8,             # INX
            0xE0, 0x0A,       # CPX #$0A
            0xD0, 0xFB,       # BNE -5 (back to INX)
            0x02,             # Illegal opcode (halt)
        ])
        cpu.load_binary(program, 0x0000)
        cpu.run(0x0000)
        assert cpu.x == 0x0A

    def test_multiply_8x8(self, cpu):
        """Multiply 5 * 6 = 30 using shift-and-add."""
        # Standard 8-bit multiply: result in A
        # Uses $20 for multiplicand (shifted), $21 for multiplier
        program = bytes([
            0xA9, 0x00,       # 0000: LDA #$00 (result = 0)
            0xA2, 0x08,       # 0002: LDX #$08 (8 bits to process)
            # loop (0004):
            0x46, 0x21,       # 0004: LSR $21 (shift multiplier right, bit to C)
            0x90, 0x03,       # 0006: BCC +3 (skip 3 bytes to ASL at 000B)
            0x18,             # 0008: CLC
            0x65, 0x20,       # 0009: ADC $20 (add multiplicand to result)
            # skip (000B):
            0x06, 0x20,       # 000B: ASL $20 (double multiplicand)
            0xCA,             # 000D: DEX
            0xD0, 0xF4,       # 000E: BNE loop (offset -12 = 0xF4, back to 0004)
            0x02,             # 0010: Illegal (halt)
        ])
        cpu.load_binary(program, 0x0000)
        cpu.memory[0x0020] = 5   # multiplicand
        cpu.memory[0x0021] = 6   # multiplier
        cpu.run(0x0000)
        assert cpu.a == 30

    def test_fibonacci(self, cpu):
        """Compute Fibonacci numbers: 1, 1, 2, 3, 5, 8, 13..."""
        # Result stored at $20, $21, $22...
        program = bytes([
            0xA9, 0x01,       # LDA #$01
            0x85, 0x20,       # STA $20 (fib[0] = 1)
            0x85, 0x21,       # STA $21 (fib[1] = 1)
            0xA2, 0x02,       # LDX #$02 (index)
            0xA0, 0x06,       # LDY #$06 (count)
            # loop:
            0xB5, 0x1E,       # LDA $1E,X (fib[n-2])
            0x18,             # CLC
            0x75, 0x1F,       # ADC $1F,X (fib[n-1])
            0x95, 0x20,       # STA $20,X (fib[n])
            0xE8,             # INX
            0x88,             # DEY
            0xD0, 0xF5,       # BNE loop
            0x00,             # BRK
        ])
        cpu.load_binary(program, 0x0000)
        cpu.run(0x0000)
        # Check: 1, 1, 2, 3, 5, 8, 13, 21
        assert cpu.memory[0x20] == 1
        assert cpu.memory[0x21] == 1
        assert cpu.memory[0x22] == 2
        assert cpu.memory[0x23] == 3
        assert cpu.memory[0x24] == 5
        assert cpu.memory[0x25] == 8
        assert cpu.memory[0x26] == 13
        assert cpu.memory[0x27] == 21


# =============================================================================
# UTILITY TESTS
# =============================================================================

class TestUtilities:
    """Test disassembler and state dump."""

    @pytest.fixture
    def cpu(self):
        return CPU6502()

    def test_disassemble_implied(self, cpu):
        cpu.memory[0x0000] = 0xEA  # NOP
        text, length = disassemble(cpu, 0x0000)
        assert text == "NOP"
        assert length == 1

    def test_disassemble_immediate(self, cpu):
        cpu.memory[0x0000] = 0xA9  # LDA #$42
        cpu.memory[0x0001] = 0x42
        text, length = disassemble(cpu, 0x0000)
        assert text == "LDA #$42"
        assert length == 2

    def test_disassemble_absolute(self, cpu):
        cpu.memory[0x0000] = 0xAD  # LDA $1234
        cpu.memory[0x0001] = 0x34
        cpu.memory[0x0002] = 0x12
        text, length = disassemble(cpu, 0x0000)
        assert text == "LDA $1234"
        assert length == 3

    def test_dump_state(self, cpu):
        cpu.a = 0x42
        cpu.x = 0x10
        cpu.y = 0x20
        cpu.pc = 0x1234
        state = dump_state(cpu)
        assert "PC=$1234" in state
        assert "A=$42" in state
        assert "X=$10" in state
        assert "Y=$20" in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
