#!/usr/bin/env python3
"""
Frozen 6502: A Complete Emulator Built on Geometric Computation

The ALU operations are frozen shapes - pure mathematical functions
that compute via geometry, not learned weights.

This emulator proves: Computation is geometry. The shapes are natural.
The routing relaxes into harmony.

Usage:
    python frozen_6502.py program.bin [start_addr]
    python frozen_6502.py --test

Author: TriX Project
Date: 2025-12-20
"""

from typing import Dict, Tuple, Optional, Callable
import sys

# =============================================================================
# SECTION 1: OPCODE TABLE
# =============================================================================
# Format: opcode -> (mnemonic, addressing_mode, bytes, cycles)
# All 151 valid 6502 opcodes

OPCODES: Dict[int, Tuple[str, str, int, int]] = {
    # BRK, ORA, ASL block (0x00-0x1F)
    0x00: ('BRK', 'implied', 1, 7),
    0x01: ('ORA', 'indirect_x', 2, 6),
    0x05: ('ORA', 'zero_page', 2, 3),
    0x06: ('ASL', 'zero_page', 2, 5),
    0x08: ('PHP', 'implied', 1, 3),
    0x09: ('ORA', 'immediate', 2, 2),
    0x0A: ('ASL', 'accumulator', 1, 2),
    0x0D: ('ORA', 'absolute', 3, 4),
    0x0E: ('ASL', 'absolute', 3, 6),
    0x10: ('BPL', 'relative', 2, 2),
    0x11: ('ORA', 'indirect_y', 2, 5),
    0x15: ('ORA', 'zero_page_x', 2, 4),
    0x16: ('ASL', 'zero_page_x', 2, 6),
    0x18: ('CLC', 'implied', 1, 2),
    0x19: ('ORA', 'absolute_y', 3, 4),
    0x1D: ('ORA', 'absolute_x', 3, 4),
    0x1E: ('ASL', 'absolute_x', 3, 7),

    # JSR, AND, ROL block (0x20-0x3F)
    0x20: ('JSR', 'absolute', 3, 6),
    0x21: ('AND', 'indirect_x', 2, 6),
    0x24: ('BIT', 'zero_page', 2, 3),
    0x25: ('AND', 'zero_page', 2, 3),
    0x26: ('ROL', 'zero_page', 2, 5),
    0x28: ('PLP', 'implied', 1, 4),
    0x29: ('AND', 'immediate', 2, 2),
    0x2A: ('ROL', 'accumulator', 1, 2),
    0x2C: ('BIT', 'absolute', 3, 4),
    0x2D: ('AND', 'absolute', 3, 4),
    0x2E: ('ROL', 'absolute', 3, 6),
    0x30: ('BMI', 'relative', 2, 2),
    0x31: ('AND', 'indirect_y', 2, 5),
    0x35: ('AND', 'zero_page_x', 2, 4),
    0x36: ('ROL', 'zero_page_x', 2, 6),
    0x38: ('SEC', 'implied', 1, 2),
    0x39: ('AND', 'absolute_y', 3, 4),
    0x3D: ('AND', 'absolute_x', 3, 4),
    0x3E: ('ROL', 'absolute_x', 3, 7),

    # RTI, EOR, LSR block (0x40-0x5F)
    0x40: ('RTI', 'implied', 1, 6),
    0x41: ('EOR', 'indirect_x', 2, 6),
    0x45: ('EOR', 'zero_page', 2, 3),
    0x46: ('LSR', 'zero_page', 2, 5),
    0x48: ('PHA', 'implied', 1, 3),
    0x49: ('EOR', 'immediate', 2, 2),
    0x4A: ('LSR', 'accumulator', 1, 2),
    0x4C: ('JMP', 'absolute', 3, 3),
    0x4D: ('EOR', 'absolute', 3, 4),
    0x4E: ('LSR', 'absolute', 3, 6),
    0x50: ('BVC', 'relative', 2, 2),
    0x51: ('EOR', 'indirect_y', 2, 5),
    0x55: ('EOR', 'zero_page_x', 2, 4),
    0x56: ('LSR', 'zero_page_x', 2, 6),
    0x58: ('CLI', 'implied', 1, 2),
    0x59: ('EOR', 'absolute_y', 3, 4),
    0x5D: ('EOR', 'absolute_x', 3, 4),
    0x5E: ('LSR', 'absolute_x', 3, 7),

    # RTS, ADC, ROR block (0x60-0x7F)
    0x60: ('RTS', 'implied', 1, 6),
    0x61: ('ADC', 'indirect_x', 2, 6),
    0x65: ('ADC', 'zero_page', 2, 3),
    0x66: ('ROR', 'zero_page', 2, 5),
    0x68: ('PLA', 'implied', 1, 4),
    0x69: ('ADC', 'immediate', 2, 2),
    0x6A: ('ROR', 'accumulator', 1, 2),
    0x6C: ('JMP', 'indirect', 3, 5),
    0x6D: ('ADC', 'absolute', 3, 4),
    0x6E: ('ROR', 'absolute', 3, 6),
    0x70: ('BVS', 'relative', 2, 2),
    0x71: ('ADC', 'indirect_y', 2, 5),
    0x75: ('ADC', 'zero_page_x', 2, 4),
    0x76: ('ROR', 'zero_page_x', 2, 6),
    0x78: ('SEI', 'implied', 1, 2),
    0x79: ('ADC', 'absolute_y', 3, 4),
    0x7D: ('ADC', 'absolute_x', 3, 4),
    0x7E: ('ROR', 'absolute_x', 3, 7),

    # STA, STX, STY block (0x80-0x9F)
    0x81: ('STA', 'indirect_x', 2, 6),
    0x84: ('STY', 'zero_page', 2, 3),
    0x85: ('STA', 'zero_page', 2, 3),
    0x86: ('STX', 'zero_page', 2, 3),
    0x88: ('DEY', 'implied', 1, 2),
    0x8A: ('TXA', 'implied', 1, 2),
    0x8C: ('STY', 'absolute', 3, 4),
    0x8D: ('STA', 'absolute', 3, 4),
    0x8E: ('STX', 'absolute', 3, 4),
    0x90: ('BCC', 'relative', 2, 2),
    0x91: ('STA', 'indirect_y', 2, 6),
    0x94: ('STY', 'zero_page_x', 2, 4),
    0x95: ('STA', 'zero_page_x', 2, 4),
    0x96: ('STX', 'zero_page_y', 2, 4),
    0x98: ('TYA', 'implied', 1, 2),
    0x99: ('STA', 'absolute_y', 3, 5),
    0x9A: ('TXS', 'implied', 1, 2),
    0x9D: ('STA', 'absolute_x', 3, 5),

    # LDA, LDX, LDY block (0xA0-0xBF)
    0xA0: ('LDY', 'immediate', 2, 2),
    0xA1: ('LDA', 'indirect_x', 2, 6),
    0xA2: ('LDX', 'immediate', 2, 2),
    0xA4: ('LDY', 'zero_page', 2, 3),
    0xA5: ('LDA', 'zero_page', 2, 3),
    0xA6: ('LDX', 'zero_page', 2, 3),
    0xA8: ('TAY', 'implied', 1, 2),
    0xA9: ('LDA', 'immediate', 2, 2),
    0xAA: ('TAX', 'implied', 1, 2),
    0xAC: ('LDY', 'absolute', 3, 4),
    0xAD: ('LDA', 'absolute', 3, 4),
    0xAE: ('LDX', 'absolute', 3, 4),
    0xB0: ('BCS', 'relative', 2, 2),
    0xB1: ('LDA', 'indirect_y', 2, 5),
    0xB4: ('LDY', 'zero_page_x', 2, 4),
    0xB5: ('LDA', 'zero_page_x', 2, 4),
    0xB6: ('LDX', 'zero_page_y', 2, 4),
    0xB8: ('CLV', 'implied', 1, 2),
    0xB9: ('LDA', 'absolute_y', 3, 4),
    0xBA: ('TSX', 'implied', 1, 2),
    0xBC: ('LDY', 'absolute_x', 3, 4),
    0xBD: ('LDA', 'absolute_x', 3, 4),
    0xBE: ('LDX', 'absolute_y', 3, 4),

    # CPY, CMP, DEC block (0xC0-0xDF)
    0xC0: ('CPY', 'immediate', 2, 2),
    0xC1: ('CMP', 'indirect_x', 2, 6),
    0xC4: ('CPY', 'zero_page', 2, 3),
    0xC5: ('CMP', 'zero_page', 2, 3),
    0xC6: ('DEC', 'zero_page', 2, 5),
    0xC8: ('INY', 'implied', 1, 2),
    0xC9: ('CMP', 'immediate', 2, 2),
    0xCA: ('DEX', 'implied', 1, 2),
    0xCC: ('CPY', 'absolute', 3, 4),
    0xCD: ('CMP', 'absolute', 3, 4),
    0xCE: ('DEC', 'absolute', 3, 6),
    0xD0: ('BNE', 'relative', 2, 2),
    0xD1: ('CMP', 'indirect_y', 2, 5),
    0xD5: ('CMP', 'zero_page_x', 2, 4),
    0xD6: ('DEC', 'zero_page_x', 2, 6),
    0xD8: ('CLD', 'implied', 1, 2),
    0xD9: ('CMP', 'absolute_y', 3, 4),
    0xDD: ('CMP', 'absolute_x', 3, 4),
    0xDE: ('DEC', 'absolute_x', 3, 7),

    # CPX, SBC, INC block (0xE0-0xFF)
    0xE0: ('CPX', 'immediate', 2, 2),
    0xE1: ('SBC', 'indirect_x', 2, 6),
    0xE4: ('CPX', 'zero_page', 2, 3),
    0xE5: ('SBC', 'zero_page', 2, 3),
    0xE6: ('INC', 'zero_page', 2, 5),
    0xE8: ('INX', 'implied', 1, 2),
    0xE9: ('SBC', 'immediate', 2, 2),
    0xEA: ('NOP', 'implied', 1, 2),
    0xEC: ('CPX', 'absolute', 3, 4),
    0xED: ('SBC', 'absolute', 3, 4),
    0xEE: ('INC', 'absolute', 3, 6),
    0xF0: ('BEQ', 'relative', 2, 2),
    0xF1: ('SBC', 'indirect_y', 2, 5),
    0xF5: ('SBC', 'zero_page_x', 2, 4),
    0xF6: ('INC', 'zero_page_x', 2, 6),
    0xF8: ('SED', 'implied', 1, 2),
    0xF9: ('SBC', 'absolute_y', 3, 4),
    0xFD: ('SBC', 'absolute_x', 3, 4),
    0xFE: ('INC', 'absolute_x', 3, 7),
}

# Instructions that modify PC themselves
PC_MODIFYING = {'JMP', 'JSR', 'RTS', 'RTI', 'BRK',
                'BCC', 'BCS', 'BEQ', 'BNE', 'BMI', 'BPL', 'BVC', 'BVS'}


# =============================================================================
# SECTION 2: FROZEN SHAPES
# =============================================================================
# Integer implementations of the geometric operations.
# These are the DUALS of the bit-level frozen shapes.
# Python's native integer operations implement the same geometry.

def frozen_add(a: int, b: int, c: int) -> Tuple[int, int]:
    """
    Ripple add: 8-bit addition with carry.

    This is the integer dual of 8 chained full adders.
    The geometry: sum_i = a_i XOR b_i XOR c_i
                  c_{i+1} = (a_i AND b_i) OR (c_i AND (a_i XOR b_i))

    Returns: (result, carry_out)
    """
    total = a + b + c
    return total & 0xFF, (total >> 8) & 1


def frozen_sub(a: int, b: int, c: int) -> Tuple[int, int]:
    """
    Ripple subtract: a - b - !c (borrow semantics).

    6502 uses inverted carry for subtraction:
    - C=1 means no borrow
    - C=0 means borrow

    Returns: (result, carry_out) where carry=1 means no borrow
    """
    total = a - b - (1 - c)
    return total & 0xFF, 0 if total < 0 else 1


def frozen_and(a: int, b: int) -> int:
    """Parallel AND: 8 independent AND gates."""
    return a & b


def frozen_or(a: int, b: int) -> int:
    """Parallel OR: 8 independent OR gates."""
    return a | b


def frozen_xor(a: int, b: int) -> int:
    """Parallel XOR: 8 independent XOR gates. XOR(a,b) = a + b - 2ab in {0,1}."""
    return a ^ b


def frozen_asl(a: int) -> Tuple[int, int]:
    """
    Arithmetic shift left.
    Bit 7 goes to carry, 0 enters bit 0.

    Returns: (result, carry_out)
    """
    return (a << 1) & 0xFF, (a >> 7) & 1


def frozen_lsr(a: int) -> Tuple[int, int]:
    """
    Logical shift right.
    Bit 0 goes to carry, 0 enters bit 7.

    Returns: (result, carry_out)
    """
    return a >> 1, a & 1


def frozen_rol(a: int, c: int) -> Tuple[int, int]:
    """
    Rotate left through carry.
    Carry enters bit 0, bit 7 goes to carry.

    Returns: (result, carry_out)
    """
    result = ((a << 1) | c) & 0xFF
    return result, (a >> 7) & 1


def frozen_ror(a: int, c: int) -> Tuple[int, int]:
    """
    Rotate right through carry.
    Carry enters bit 7, bit 0 goes to carry.

    Returns: (result, carry_out)
    """
    result = (a >> 1) | (c << 7)
    return result, a & 1


def frozen_inc(a: int) -> int:
    """Increment by 1."""
    return (a + 1) & 0xFF


def frozen_dec(a: int) -> int:
    """Decrement by 1."""
    return (a - 1) & 0xFF


# =============================================================================
# SECTION 3: CPU CLASS
# =============================================================================

class CPU6502:
    """
    Complete 6502 processor emulator.

    State:
        a, x, y: 8-bit registers
        sp: 8-bit stack pointer (stack at $0100-$01FF)
        pc: 16-bit program counter
        p: 8-bit status register (NV-BDIZC)
        memory: 64KB byte array
    """

    def __init__(self):
        self.a: int = 0          # Accumulator
        self.x: int = 0          # X index register
        self.y: int = 0          # Y index register
        self.sp: int = 0xFD      # Stack pointer
        self.pc: int = 0         # Program counter
        self.p: int = 0x24       # Status: NV-BDIZC (I set, unused bit set)
        self.memory = bytearray(65536)
        self.halted: bool = False
        self.cycles: int = 0

    # -------------------------------------------------------------------------
    # Flag accessors (bit positions: N=7, V=6, B=4, D=3, I=2, Z=1, C=0)
    # -------------------------------------------------------------------------

    @property
    def flag_n(self) -> int:
        """Negative flag (bit 7)."""
        return (self.p >> 7) & 1

    @property
    def flag_v(self) -> int:
        """Overflow flag (bit 6)."""
        return (self.p >> 6) & 1

    @property
    def flag_b(self) -> int:
        """Break flag (bit 4)."""
        return (self.p >> 4) & 1

    @property
    def flag_d(self) -> int:
        """Decimal flag (bit 3)."""
        return (self.p >> 3) & 1

    @property
    def flag_i(self) -> int:
        """Interrupt disable flag (bit 2)."""
        return (self.p >> 2) & 1

    @property
    def flag_z(self) -> int:
        """Zero flag (bit 1)."""
        return (self.p >> 1) & 1

    @property
    def flag_c(self) -> int:
        """Carry flag (bit 0)."""
        return self.p & 1

    def set_flag(self, bit: int, value: bool):
        """Set or clear a flag bit."""
        if value:
            self.p |= (1 << bit)
        else:
            self.p &= ~(1 << bit)

    def update_nz(self, value: int):
        """Update N and Z flags based on value."""
        self.set_flag(7, bool(value & 0x80))  # N = bit 7
        self.set_flag(1, value == 0)           # Z = value is zero

    # -------------------------------------------------------------------------
    # Memory access
    # -------------------------------------------------------------------------

    def read(self, addr: int) -> int:
        """Read byte from memory."""
        return self.memory[addr & 0xFFFF]

    def write(self, addr: int, value: int):
        """Write byte to memory."""
        self.memory[addr & 0xFFFF] = value & 0xFF

    def read16(self, addr: int) -> int:
        """Read 16-bit word (little-endian)."""
        lo = self.read(addr)
        hi = self.read((addr + 1) & 0xFFFF)
        return lo | (hi << 8)

    def read16_zp(self, addr: int) -> int:
        """Read 16-bit word from zero page (wraps within ZP)."""
        lo = self.read(addr & 0xFF)
        hi = self.read((addr + 1) & 0xFF)
        return lo | (hi << 8)

    # -------------------------------------------------------------------------
    # Stack operations (stack at $0100-$01FF, grows downward)
    # -------------------------------------------------------------------------

    def push(self, value: int):
        """Push byte to stack."""
        self.write(0x100 + self.sp, value)
        self.sp = (self.sp - 1) & 0xFF

    def pull(self) -> int:
        """Pull byte from stack."""
        self.sp = (self.sp + 1) & 0xFF
        return self.read(0x100 + self.sp)

    def push16(self, value: int):
        """Push 16-bit word to stack (high byte first)."""
        self.push((value >> 8) & 0xFF)
        self.push(value & 0xFF)

    def pull16(self) -> int:
        """Pull 16-bit word from stack."""
        lo = self.pull()
        hi = self.pull()
        return lo | (hi << 8)

    # -------------------------------------------------------------------------
    # Addressing modes
    # -------------------------------------------------------------------------

    def addr_implied(self) -> Optional[int]:
        """Implied: no operand."""
        return None

    def addr_accumulator(self) -> Optional[int]:
        """Accumulator: operand is A register."""
        return None

    def addr_immediate(self) -> int:
        """Immediate: operand is next byte."""
        return self.pc + 1

    def addr_zero_page(self) -> int:
        """Zero page: address is in zero page."""
        return self.read(self.pc + 1)

    def addr_zero_page_x(self) -> int:
        """Zero page,X: (zp + X) wrapped in zero page."""
        return (self.read(self.pc + 1) + self.x) & 0xFF

    def addr_zero_page_y(self) -> int:
        """Zero page,Y: (zp + Y) wrapped in zero page."""
        return (self.read(self.pc + 1) + self.y) & 0xFF

    def addr_absolute(self) -> int:
        """Absolute: full 16-bit address."""
        return self.read16(self.pc + 1)

    def addr_absolute_x(self) -> int:
        """Absolute,X: absolute + X."""
        return (self.read16(self.pc + 1) + self.x) & 0xFFFF

    def addr_absolute_y(self) -> int:
        """Absolute,Y: absolute + Y."""
        return (self.read16(self.pc + 1) + self.y) & 0xFFFF

    def addr_indirect(self) -> int:
        """Indirect: pointer to address (JMP only)."""
        ptr = self.read16(self.pc + 1)
        # 6502 bug: if pointer is at page boundary, high byte wraps within page
        if (ptr & 0xFF) == 0xFF:
            return self.read(ptr) | (self.read(ptr & 0xFF00) << 8)
        return self.read16(ptr)

    def addr_indirect_x(self) -> int:
        """Indexed indirect: (zp,X) - pointer in zero page."""
        ptr = (self.read(self.pc + 1) + self.x) & 0xFF
        return self.read16_zp(ptr)

    def addr_indirect_y(self) -> int:
        """Indirect indexed: (zp),Y - indirect then add Y."""
        ptr = self.read(self.pc + 1)
        base = self.read16_zp(ptr)
        return (base + self.y) & 0xFFFF

    def addr_relative(self) -> int:
        """Relative: signed offset for branches."""
        offset = self.read(self.pc + 1)
        if offset & 0x80:
            offset -= 256
        return (self.pc + 2 + offset) & 0xFFFF

    def get_address(self, mode: str) -> Optional[int]:
        """Dispatch to addressing mode handler."""
        handlers = {
            'implied': self.addr_implied,
            'accumulator': self.addr_accumulator,
            'immediate': self.addr_immediate,
            'zero_page': self.addr_zero_page,
            'zero_page_x': self.addr_zero_page_x,
            'zero_page_y': self.addr_zero_page_y,
            'absolute': self.addr_absolute,
            'absolute_x': self.addr_absolute_x,
            'absolute_y': self.addr_absolute_y,
            'indirect': self.addr_indirect,
            'indirect_x': self.addr_indirect_x,
            'indirect_y': self.addr_indirect_y,
            'relative': self.addr_relative,
        }
        return handlers[mode]()

    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------

    def op_adc(self, addr: int, mode: str):
        """Add with carry: A = A + M + C"""
        m = self.read(addr)
        result, carry = frozen_add(self.a, m, self.flag_c)
        # Overflow: sign of result differs from both operands
        overflow = (~(self.a ^ m) & (self.a ^ result)) & 0x80
        self.a = result
        self.set_flag(0, bool(carry))
        self.set_flag(6, bool(overflow))
        self.update_nz(result)

    def op_sbc(self, addr: int, mode: str):
        """Subtract with carry: A = A - M - !C"""
        m = self.read(addr)
        result, carry = frozen_sub(self.a, m, self.flag_c)
        # Overflow: operands have different signs, result sign differs from A
        overflow = ((self.a ^ m) & (self.a ^ result)) & 0x80
        self.a = result
        self.set_flag(0, bool(carry))
        self.set_flag(6, bool(overflow))
        self.update_nz(result)

    def op_and(self, addr: int, mode: str):
        """Logical AND: A = A & M"""
        self.a = frozen_and(self.a, self.read(addr))
        self.update_nz(self.a)

    def op_ora(self, addr: int, mode: str):
        """Logical OR: A = A | M"""
        self.a = frozen_or(self.a, self.read(addr))
        self.update_nz(self.a)

    def op_eor(self, addr: int, mode: str):
        """Logical XOR: A = A ^ M"""
        self.a = frozen_xor(self.a, self.read(addr))
        self.update_nz(self.a)

    def op_asl(self, addr: Optional[int], mode: str):
        """Arithmetic shift left."""
        if mode == 'accumulator':
            self.a, carry = frozen_asl(self.a)
            self.update_nz(self.a)
        else:
            val = self.read(addr)
            result, carry = frozen_asl(val)
            self.write(addr, result)
            self.update_nz(result)
        self.set_flag(0, bool(carry))

    def op_lsr(self, addr: Optional[int], mode: str):
        """Logical shift right."""
        if mode == 'accumulator':
            self.a, carry = frozen_lsr(self.a)
            self.update_nz(self.a)
        else:
            val = self.read(addr)
            result, carry = frozen_lsr(val)
            self.write(addr, result)
            self.update_nz(result)
        self.set_flag(0, bool(carry))

    def op_rol(self, addr: Optional[int], mode: str):
        """Rotate left through carry."""
        if mode == 'accumulator':
            self.a, carry = frozen_rol(self.a, self.flag_c)
            self.update_nz(self.a)
        else:
            val = self.read(addr)
            result, carry = frozen_rol(val, self.flag_c)
            self.write(addr, result)
            self.update_nz(result)
        self.set_flag(0, bool(carry))

    def op_ror(self, addr: Optional[int], mode: str):
        """Rotate right through carry."""
        if mode == 'accumulator':
            self.a, carry = frozen_ror(self.a, self.flag_c)
            self.update_nz(self.a)
        else:
            val = self.read(addr)
            result, carry = frozen_ror(val, self.flag_c)
            self.write(addr, result)
            self.update_nz(result)
        self.set_flag(0, bool(carry))

    def op_inc(self, addr: int, mode: str):
        """Increment memory."""
        result = frozen_inc(self.read(addr))
        self.write(addr, result)
        self.update_nz(result)

    def op_dec(self, addr: int, mode: str):
        """Decrement memory."""
        result = frozen_dec(self.read(addr))
        self.write(addr, result)
        self.update_nz(result)

    def op_inx(self, addr: Optional[int], mode: str):
        """Increment X."""
        self.x = frozen_inc(self.x)
        self.update_nz(self.x)

    def op_dex(self, addr: Optional[int], mode: str):
        """Decrement X."""
        self.x = frozen_dec(self.x)
        self.update_nz(self.x)

    def op_iny(self, addr: Optional[int], mode: str):
        """Increment Y."""
        self.y = frozen_inc(self.y)
        self.update_nz(self.y)

    def op_dey(self, addr: Optional[int], mode: str):
        """Decrement Y."""
        self.y = frozen_dec(self.y)
        self.update_nz(self.y)

    def op_cmp(self, addr: int, mode: str):
        """Compare A with memory."""
        m = self.read(addr)
        result, carry = frozen_sub(self.a, m, 1)  # No borrow
        self.set_flag(0, bool(carry))
        self.update_nz(result)

    def op_cpx(self, addr: int, mode: str):
        """Compare X with memory."""
        m = self.read(addr)
        result, carry = frozen_sub(self.x, m, 1)
        self.set_flag(0, bool(carry))
        self.update_nz(result)

    def op_cpy(self, addr: int, mode: str):
        """Compare Y with memory."""
        m = self.read(addr)
        result, carry = frozen_sub(self.y, m, 1)
        self.set_flag(0, bool(carry))
        self.update_nz(result)

    def op_bit(self, addr: int, mode: str):
        """Test bits: N=M7, V=M6, Z=(A&M)==0"""
        m = self.read(addr)
        self.set_flag(7, bool(m & 0x80))  # N = bit 7 of memory
        self.set_flag(6, bool(m & 0x40))  # V = bit 6 of memory
        self.set_flag(1, (self.a & m) == 0)  # Z = A AND M

    def op_lda(self, addr: int, mode: str):
        """Load accumulator."""
        self.a = self.read(addr)
        self.update_nz(self.a)

    def op_ldx(self, addr: int, mode: str):
        """Load X register."""
        self.x = self.read(addr)
        self.update_nz(self.x)

    def op_ldy(self, addr: int, mode: str):
        """Load Y register."""
        self.y = self.read(addr)
        self.update_nz(self.y)

    def op_sta(self, addr: int, mode: str):
        """Store accumulator."""
        self.write(addr, self.a)

    def op_stx(self, addr: int, mode: str):
        """Store X register."""
        self.write(addr, self.x)

    def op_sty(self, addr: int, mode: str):
        """Store Y register."""
        self.write(addr, self.y)

    def op_tax(self, addr: Optional[int], mode: str):
        """Transfer A to X."""
        self.x = self.a
        self.update_nz(self.x)

    def op_txa(self, addr: Optional[int], mode: str):
        """Transfer X to A."""
        self.a = self.x
        self.update_nz(self.a)

    def op_tay(self, addr: Optional[int], mode: str):
        """Transfer A to Y."""
        self.y = self.a
        self.update_nz(self.y)

    def op_tya(self, addr: Optional[int], mode: str):
        """Transfer Y to A."""
        self.a = self.y
        self.update_nz(self.a)

    def op_tsx(self, addr: Optional[int], mode: str):
        """Transfer SP to X."""
        self.x = self.sp
        self.update_nz(self.x)

    def op_txs(self, addr: Optional[int], mode: str):
        """Transfer X to SP."""
        self.sp = self.x

    def op_pha(self, addr: Optional[int], mode: str):
        """Push accumulator."""
        self.push(self.a)

    def op_php(self, addr: Optional[int], mode: str):
        """Push processor status (with B and unused bits set)."""
        self.push(self.p | 0x30)  # B and unused always pushed as 1

    def op_pla(self, addr: Optional[int], mode: str):
        """Pull accumulator."""
        self.a = self.pull()
        self.update_nz(self.a)

    def op_plp(self, addr: Optional[int], mode: str):
        """Pull processor status."""
        self.p = (self.pull() & 0xEF) | 0x20  # Clear B, set unused

    def op_jmp(self, addr: int, mode: str):
        """Jump to address."""
        self.pc = addr

    def op_jsr(self, addr: int, mode: str):
        """Jump to subroutine."""
        self.push16(self.pc + 2)  # Push return address - 1
        self.pc = addr

    def op_rts(self, addr: Optional[int], mode: str):
        """Return from subroutine."""
        self.pc = self.pull16() + 1

    def op_rti(self, addr: Optional[int], mode: str):
        """Return from interrupt."""
        self.p = (self.pull() & 0xEF) | 0x20
        self.pc = self.pull16()

    def _branch(self, condition: bool, addr: int):
        """Helper for branch instructions."""
        if condition:
            self.pc = addr
        else:
            self.pc += 2  # Skip branch instruction

    def op_bcc(self, addr: int, mode: str):
        """Branch if carry clear."""
        self._branch(self.flag_c == 0, addr)

    def op_bcs(self, addr: int, mode: str):
        """Branch if carry set."""
        self._branch(self.flag_c == 1, addr)

    def op_beq(self, addr: int, mode: str):
        """Branch if equal (Z=1)."""
        self._branch(self.flag_z == 1, addr)

    def op_bne(self, addr: int, mode: str):
        """Branch if not equal (Z=0)."""
        self._branch(self.flag_z == 0, addr)

    def op_bmi(self, addr: int, mode: str):
        """Branch if minus (N=1)."""
        self._branch(self.flag_n == 1, addr)

    def op_bpl(self, addr: int, mode: str):
        """Branch if plus (N=0)."""
        self._branch(self.flag_n == 0, addr)

    def op_bvc(self, addr: int, mode: str):
        """Branch if overflow clear."""
        self._branch(self.flag_v == 0, addr)

    def op_bvs(self, addr: int, mode: str):
        """Branch if overflow set."""
        self._branch(self.flag_v == 1, addr)

    def op_clc(self, addr: Optional[int], mode: str):
        """Clear carry flag."""
        self.set_flag(0, False)

    def op_sec(self, addr: Optional[int], mode: str):
        """Set carry flag."""
        self.set_flag(0, True)

    def op_cld(self, addr: Optional[int], mode: str):
        """Clear decimal flag."""
        self.set_flag(3, False)

    def op_sed(self, addr: Optional[int], mode: str):
        """Set decimal flag."""
        self.set_flag(3, True)

    def op_cli(self, addr: Optional[int], mode: str):
        """Clear interrupt disable."""
        self.set_flag(2, False)

    def op_sei(self, addr: Optional[int], mode: str):
        """Set interrupt disable."""
        self.set_flag(2, True)

    def op_clv(self, addr: Optional[int], mode: str):
        """Clear overflow flag."""
        self.set_flag(6, False)

    def op_nop(self, addr: Optional[int], mode: str):
        """No operation."""
        pass

    def op_brk(self, addr: Optional[int], mode: str):
        """Software interrupt."""
        self.push16(self.pc + 2)
        self.push(self.p | 0x30)  # B flag set
        self.set_flag(2, True)  # Set I
        self.pc = self.read16(0xFFFE)

    # Operation dispatch table
    OPERATIONS: Dict[str, Callable] = {}

    def _init_operations(self):
        """Initialize operation dispatch table."""
        self.OPERATIONS = {
            'ADC': self.op_adc, 'SBC': self.op_sbc,
            'AND': self.op_and, 'ORA': self.op_ora, 'EOR': self.op_eor,
            'ASL': self.op_asl, 'LSR': self.op_lsr,
            'ROL': self.op_rol, 'ROR': self.op_ror,
            'INC': self.op_inc, 'DEC': self.op_dec,
            'INX': self.op_inx, 'DEX': self.op_dex,
            'INY': self.op_iny, 'DEY': self.op_dey,
            'CMP': self.op_cmp, 'CPX': self.op_cpx, 'CPY': self.op_cpy,
            'BIT': self.op_bit,
            'LDA': self.op_lda, 'LDX': self.op_ldx, 'LDY': self.op_ldy,
            'STA': self.op_sta, 'STX': self.op_stx, 'STY': self.op_sty,
            'TAX': self.op_tax, 'TXA': self.op_txa,
            'TAY': self.op_tay, 'TYA': self.op_tya,
            'TSX': self.op_tsx, 'TXS': self.op_txs,
            'PHA': self.op_pha, 'PHP': self.op_php,
            'PLA': self.op_pla, 'PLP': self.op_plp,
            'JMP': self.op_jmp, 'JSR': self.op_jsr,
            'RTS': self.op_rts, 'RTI': self.op_rti,
            'BCC': self.op_bcc, 'BCS': self.op_bcs,
            'BEQ': self.op_beq, 'BNE': self.op_bne,
            'BMI': self.op_bmi, 'BPL': self.op_bpl,
            'BVC': self.op_bvc, 'BVS': self.op_bvs,
            'CLC': self.op_clc, 'SEC': self.op_sec,
            'CLD': self.op_cld, 'SED': self.op_sed,
            'CLI': self.op_cli, 'SEI': self.op_sei,
            'CLV': self.op_clv,
            'NOP': self.op_nop, 'BRK': self.op_brk,
        }

    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def step(self) -> int:
        """Execute one instruction. Returns cycle count."""
        if self.halted:
            return 0

        # Initialize operations table if needed
        if not self.OPERATIONS:
            self._init_operations()

        opcode = self.read(self.pc)

        if opcode not in OPCODES:
            # Illegal opcode - halt
            self.halted = True
            return 0

        mnemonic, mode, length, cycles = OPCODES[opcode]

        # Get effective address
        addr = self.get_address(mode)

        # Execute operation
        self.OPERATIONS[mnemonic](addr, mode)

        # Advance PC (branches/jumps handle their own)
        if mnemonic not in PC_MODIFYING:
            self.pc += length

        self.cycles += cycles
        return cycles

    def run(self, start_addr: int, max_cycles: int = 10_000_000) -> int:
        """Run from address until halt or max cycles. Returns cycles executed."""
        self.pc = start_addr
        start_cycles = self.cycles
        while not self.halted and self.cycles - start_cycles < max_cycles:
            self.step()
        return self.cycles - start_cycles

    def reset(self):
        """Hardware reset: load PC from reset vector."""
        self.a = 0
        self.x = 0
        self.y = 0
        self.sp = 0xFD
        self.p = 0x24
        self.pc = self.read16(0xFFFC)
        self.halted = False
        self.cycles = 0

    def load_binary(self, data: bytes, addr: int):
        """Load binary data into memory."""
        for i, byte in enumerate(data):
            self.memory[addr + i] = byte

    def irq(self):
        """Trigger IRQ interrupt (if interrupts enabled)."""
        if not self.flag_i:
            self.push16(self.pc)
            self.push(self.p & 0xEF)  # B clear
            self.set_flag(2, True)
            self.pc = self.read16(0xFFFE)

    def nmi(self):
        """Trigger NMI interrupt."""
        self.push16(self.pc)
        self.push(self.p & 0xEF)  # B clear
        self.set_flag(2, True)
        self.pc = self.read16(0xFFFA)


# =============================================================================
# SECTION 4: UTILITIES
# =============================================================================

def disassemble(cpu: CPU6502, addr: int) -> Tuple[str, int]:
    """Disassemble instruction at address. Returns (text, length)."""
    opcode = cpu.read(addr)

    if opcode not in OPCODES:
        return f"???  (${opcode:02X})", 1

    mnemonic, mode, length, _ = OPCODES[opcode]

    if mode == 'implied':
        return mnemonic, length
    elif mode == 'accumulator':
        return f"{mnemonic} A", length
    elif mode == 'immediate':
        return f"{mnemonic} #${cpu.read(addr + 1):02X}", length
    elif mode == 'zero_page':
        return f"{mnemonic} ${cpu.read(addr + 1):02X}", length
    elif mode == 'zero_page_x':
        return f"{mnemonic} ${cpu.read(addr + 1):02X},X", length
    elif mode == 'zero_page_y':
        return f"{mnemonic} ${cpu.read(addr + 1):02X},Y", length
    elif mode == 'absolute':
        return f"{mnemonic} ${cpu.read16(addr + 1):04X}", length
    elif mode == 'absolute_x':
        return f"{mnemonic} ${cpu.read16(addr + 1):04X},X", length
    elif mode == 'absolute_y':
        return f"{mnemonic} ${cpu.read16(addr + 1):04X},Y", length
    elif mode == 'indirect':
        return f"{mnemonic} (${cpu.read16(addr + 1):04X})", length
    elif mode == 'indirect_x':
        return f"{mnemonic} (${cpu.read(addr + 1):02X},X)", length
    elif mode == 'indirect_y':
        return f"{mnemonic} (${cpu.read(addr + 1):02X}),Y", length
    elif mode == 'relative':
        offset = cpu.read(addr + 1)
        if offset & 0x80:
            offset -= 256
        target = (addr + 2 + offset) & 0xFFFF
        return f"{mnemonic} ${target:04X}", length

    return f"{mnemonic} ???", length


def dump_state(cpu: CPU6502) -> str:
    """Return formatted CPU state string."""
    flags = ""
    flags += "N" if cpu.flag_n else "n"
    flags += "V" if cpu.flag_v else "v"
    flags += "-"
    flags += "B" if cpu.flag_b else "b"
    flags += "D" if cpu.flag_d else "d"
    flags += "I" if cpu.flag_i else "i"
    flags += "Z" if cpu.flag_z else "z"
    flags += "C" if cpu.flag_c else "c"

    return (f"PC=${cpu.pc:04X} A=${cpu.a:02X} X=${cpu.x:02X} Y=${cpu.y:02X} "
            f"SP=${cpu.sp:02X} P={flags}")


def trace(cpu: CPU6502) -> str:
    """Return trace line for current instruction."""
    dis, _ = disassemble(cpu, cpu.pc)
    return f"{cpu.pc:04X}: {dis:20s}  {dump_state(cpu)}"


# =============================================================================
# SECTION 5: TESTS
# =============================================================================

def run_tests():
    """Run unit tests for the emulator."""
    print("Running frozen 6502 tests...")
    passed = 0
    failed = 0

    def test(name: str, condition: bool):
        nonlocal passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {name}")

    # Test frozen shapes
    test("frozen_add basic", frozen_add(0x50, 0x10, 0) == (0x60, 0))
    test("frozen_add carry", frozen_add(0xFF, 0x01, 0) == (0x00, 1))
    test("frozen_add with carry in", frozen_add(0x50, 0x10, 1) == (0x61, 0))
    test("frozen_sub basic", frozen_sub(0x50, 0x10, 1) == (0x40, 1))
    test("frozen_sub borrow", frozen_sub(0x10, 0x50, 1) == (0xC0, 0))
    test("frozen_and", frozen_and(0x0F, 0xF0) == 0x00)
    test("frozen_or", frozen_or(0x0F, 0xF0) == 0xFF)
    test("frozen_xor", frozen_xor(0xFF, 0x0F) == 0xF0)
    test("frozen_asl", frozen_asl(0x81) == (0x02, 1))
    test("frozen_lsr", frozen_lsr(0x81) == (0x40, 1))
    test("frozen_rol", frozen_rol(0x81, 1) == (0x03, 1))
    test("frozen_ror", frozen_ror(0x81, 1) == (0xC0, 1))
    test("frozen_inc", frozen_inc(0xFF) == 0x00)
    test("frozen_dec", frozen_dec(0x00) == 0xFF)

    # Test CPU instructions
    cpu = CPU6502()

    # LDA immediate
    cpu.memory[0x0000] = 0xA9  # LDA #$42
    cpu.memory[0x0001] = 0x42
    cpu.pc = 0
    cpu.step()
    test("LDA immediate value", cpu.a == 0x42)
    test("LDA immediate PC", cpu.pc == 0x02)

    # ADC immediate
    cpu.pc = 0
    cpu.a = 0x10
    cpu.set_flag(0, False)  # Clear carry
    cpu.memory[0x0000] = 0x69  # ADC #$20
    cpu.memory[0x0001] = 0x20
    cpu.step()
    test("ADC immediate value", cpu.a == 0x30)
    test("ADC immediate carry clear", cpu.flag_c == 0)

    # ADC with carry out
    cpu.pc = 0
    cpu.a = 0xF0
    cpu.set_flag(0, False)
    cpu.memory[0x0001] = 0x20
    cpu.step()
    test("ADC carry out value", cpu.a == 0x10)
    test("ADC carry out flag", cpu.flag_c == 1)

    # SBC immediate
    cpu.pc = 0
    cpu.a = 0x50
    cpu.set_flag(0, True)  # No borrow
    cpu.memory[0x0000] = 0xE9  # SBC #$10
    cpu.memory[0x0001] = 0x10
    cpu.step()
    test("SBC immediate value", cpu.a == 0x40)
    test("SBC no borrow", cpu.flag_c == 1)

    # INX
    cpu.pc = 0
    cpu.x = 0x05
    cpu.memory[0x0000] = 0xE8  # INX
    cpu.step()
    test("INX value", cpu.x == 0x06)

    # DEX wrap
    cpu.pc = 0
    cpu.x = 0x00
    cpu.memory[0x0000] = 0xCA  # DEX
    cpu.step()
    test("DEX wrap", cpu.x == 0xFF)
    test("DEX negative flag", cpu.flag_n == 1)

    # BEQ taken
    cpu.pc = 0
    cpu.set_flag(1, True)  # Z = 1
    cpu.memory[0x0000] = 0xF0  # BEQ
    cpu.memory[0x0001] = 0x05  # +5
    cpu.step()
    test("BEQ taken", cpu.pc == 0x07)

    # BEQ not taken
    cpu.pc = 0
    cpu.set_flag(1, False)  # Z = 0
    cpu.step()
    test("BEQ not taken", cpu.pc == 0x02)

    # JSR/RTS
    cpu.pc = 0x1000
    cpu.sp = 0xFF
    cpu.memory[0x1000] = 0x20  # JSR $2000
    cpu.memory[0x1001] = 0x00
    cpu.memory[0x1002] = 0x20
    cpu.memory[0x2000] = 0x60  # RTS
    cpu.step()  # JSR
    test("JSR PC", cpu.pc == 0x2000)
    test("JSR SP", cpu.sp == 0xFD)
    cpu.step()  # RTS
    test("RTS PC", cpu.pc == 0x1003)
    test("RTS SP", cpu.sp == 0xFF)

    # Shifts
    cpu.pc = 0
    cpu.a = 0x81
    cpu.memory[0x0000] = 0x0A  # ASL A
    cpu.step()
    test("ASL A value", cpu.a == 0x02)
    test("ASL A carry", cpu.flag_c == 1)

    cpu.pc = 0
    cpu.a = 0x81
    cpu.memory[0x0000] = 0x4A  # LSR A
    cpu.step()
    test("LSR A value", cpu.a == 0x40)
    test("LSR A carry", cpu.flag_c == 1)

    # Zero page addressing
    cpu.pc = 0
    cpu.memory[0x0000] = 0xA5  # LDA $10
    cpu.memory[0x0001] = 0x10
    cpu.memory[0x0010] = 0x77
    cpu.step()
    test("LDA zero page", cpu.a == 0x77)

    # Absolute addressing
    cpu.pc = 0
    cpu.memory[0x0000] = 0xAD  # LDA $1234
    cpu.memory[0x0001] = 0x34
    cpu.memory[0x0002] = 0x12
    cpu.memory[0x1234] = 0x88
    cpu.step()
    test("LDA absolute", cpu.a == 0x88)

    # Indexed addressing
    cpu.pc = 0
    cpu.x = 0x05
    cpu.memory[0x0000] = 0xB5  # LDA $10,X
    cpu.memory[0x0001] = 0x10
    cpu.memory[0x0015] = 0x99
    cpu.step()
    test("LDA zero page,X", cpu.a == 0x99)

    # CMP
    cpu.pc = 0
    cpu.a = 0x50
    cpu.memory[0x0000] = 0xC9  # CMP #$30
    cpu.memory[0x0001] = 0x30
    cpu.step()
    test("CMP greater carry", cpu.flag_c == 1)
    test("CMP greater zero", cpu.flag_z == 0)

    cpu.pc = 0
    cpu.memory[0x0001] = 0x50  # CMP #$50
    cpu.step()
    test("CMP equal carry", cpu.flag_c == 1)
    test("CMP equal zero", cpu.flag_z == 1)

    # PHA/PLA
    cpu.pc = 0
    cpu.sp = 0xFF
    cpu.a = 0x42
    cpu.memory[0x0000] = 0x48  # PHA
    cpu.memory[0x0001] = 0xA9  # LDA #$00
    cpu.memory[0x0002] = 0x00
    cpu.memory[0x0003] = 0x68  # PLA
    cpu.step()  # PHA
    test("PHA SP", cpu.sp == 0xFE)
    cpu.step()  # LDA #$00
    test("PHA clear A", cpu.a == 0x00)
    cpu.step()  # PLA
    test("PLA value", cpu.a == 0x42)
    test("PLA SP", cpu.sp == 0xFF)

    # Indirect addressing
    cpu.pc = 0
    cpu.x = 0x04
    cpu.memory[0x0000] = 0xA1  # LDA ($10,X)
    cpu.memory[0x0001] = 0x10
    cpu.memory[0x0014] = 0x00  # Low byte of pointer
    cpu.memory[0x0015] = 0x20  # High byte of pointer
    cpu.memory[0x2000] = 0xAB
    cpu.step()
    test("LDA (zp,X)", cpu.a == 0xAB)

    cpu.pc = 0
    cpu.y = 0x10
    cpu.memory[0x0000] = 0xB1  # LDA ($20),Y
    cpu.memory[0x0001] = 0x20
    cpu.memory[0x0020] = 0x00  # Low byte of pointer
    cpu.memory[0x0021] = 0x30  # High byte of pointer
    cpu.memory[0x3010] = 0xCD  # $3000 + $10
    cpu.step()
    test("LDA (zp),Y", cpu.a == 0xCD)

    print(f"\nTests: {passed} passed, {failed} failed")
    return failed == 0


# =============================================================================
# SECTION 6: MAIN
# =============================================================================

def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Frozen 6502 Emulator")
        print("Usage: python frozen_6502.py <program.bin> [start_addr]")
        print("       python frozen_6502.py --test")
        return

    if sys.argv[1] == '--test':
        success = run_tests()
        sys.exit(0 if success else 1)

    # Load and run program
    filename = sys.argv[1]
    start_addr = int(sys.argv[2], 0) if len(sys.argv) > 2 else 0x0000

    cpu = CPU6502()

    try:
        with open(filename, 'rb') as f:
            data = f.read()
        cpu.load_binary(data, start_addr)
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        sys.exit(1)

    print(f"Loaded {len(data)} bytes at ${start_addr:04X}")
    print(f"Running from ${start_addr:04X}...")
    print()

    # Run with tracing for first 100 instructions
    for i in range(100):
        if cpu.halted:
            break
        print(trace(cpu))
        cpu.step()

    if not cpu.halted:
        print("...")
        cpu.run(cpu.pc, max_cycles=1_000_000)

    print()
    print(f"Final state: {dump_state(cpu)}")
    print(f"Total cycles: {cpu.cycles:,}")


if __name__ == "__main__":
    main()
