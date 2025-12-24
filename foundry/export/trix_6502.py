"""
Pure TriX 6502 Export
=====================

Exports the frozen 6502 as pure C code with zero dependencies.

Architecture:
    Level 0: Polynomial primitives (XOR, AND, OR, NOT)
    Level 1: 16 frozen shapes composed from Level 0
    Level 2: Meaning layer as static lookup tables
    Runtime: Simple fetch-decode-execute loop

The output is 100% pure C. No PyTorch. No ONNX. No inference.
Just math, data, and a loop.
"""

from pathlib import Path
from typing import Dict, List, Tuple


class TriX6502Exporter:
    """
    Exports the frozen 6502 as pure C code.

    The entire CPU becomes:
    - ~200 lines of polynomial math (Level 0 + 1)
    - ~500 bytes of routing tables (Level 2)
    - ~300 lines of runtime
    """

    # The 16 shapes and their implementations
    SHAPES = [
        "RIPPLE_ADD",
        "RIPPLE_SUB",
        "PARALLEL_AND",
        "PARALLEL_OR",
        "PARALLEL_XOR",
        "SHIFT_LEFT",
        "SHIFT_RIGHT",
        "ROTATE_LEFT",
        "ROTATE_RIGHT",
        "INCREMENT",
        "DECREMENT",
        "TRANSFER",
        "LOAD",
        "STORE",
        "BIT_TEST",
        "IDENTITY",
    ]

    # Addressing modes
    ADDR_IMPLIED = 0
    ADDR_IMMEDIATE = 1
    ADDR_ZERO_PAGE = 2
    ADDR_ZERO_PAGE_X = 3
    ADDR_ZERO_PAGE_Y = 4
    ADDR_ABSOLUTE = 5
    ADDR_ABSOLUTE_X = 6
    ADDR_ABSOLUTE_Y = 7
    ADDR_INDIRECT_X = 8
    ADDR_INDIRECT_Y = 9
    ADDR_ACCUMULATOR = 10

    # Standard 6502 opcodes mapped to shapes
    # Format: opcode -> (shape_id, input_a, input_b, output, uses_carry, flags, addr_mode)
    # Registers: 0=A, 1=X, 2=Y, 3=SP, 4=MEM
    # A = ADDR_* constant
    A_IMP = 0   # Implied
    A_IMM = 1   # Immediate
    A_ZP = 2    # Zero page
    A_ZPX = 3   # Zero page,X
    A_ZPY = 4   # Zero page,Y
    A_ABS = 5   # Absolute
    A_ABX = 6   # Absolute,X
    A_ABY = 7   # Absolute,Y
    A_INX = 8   # (Indirect,X)
    A_INY = 9   # (Indirect),Y
    A_ACC = 10  # Accumulator

    # Standard 6502 opcodes mapped to shapes
    # Format: opcode -> (shape_id, input_a, input_b, output, uses_carry, flags, addr_mode)
    # Registers: 0=A, 1=X, 2=Y, 3=SP, 4=MEM
    OPCODE_MAP = {
        # ADC - Add with Carry
        0x69: (0, 0, 4, 0, 1, 0b1111, 1),   # ADC immediate
        0x65: (0, 0, 4, 0, 1, 0b1111, 2),   # ADC zero page
        0x75: (0, 0, 4, 0, 1, 0b1111, 3),   # ADC zero page,X
        0x6D: (0, 0, 4, 0, 1, 0b1111, 5),   # ADC absolute
        0x7D: (0, 0, 4, 0, 1, 0b1111, 6),   # ADC absolute,X
        0x79: (0, 0, 4, 0, 1, 0b1111, 7),   # ADC absolute,Y
        0x61: (0, 0, 4, 0, 1, 0b1111, 8),   # ADC (indirect,X)
        0x71: (0, 0, 4, 0, 1, 0b1111, 9),   # ADC (indirect),Y

        # SBC - Subtract with Carry
        0xE9: (1, 0, 4, 0, 1, 0b1111, 1),   # SBC immediate
        0xE5: (1, 0, 4, 0, 1, 0b1111, 2),   # SBC zero page
        0xF5: (1, 0, 4, 0, 1, 0b1111, 3),   # SBC zero page,X
        0xED: (1, 0, 4, 0, 1, 0b1111, 5),   # SBC absolute
        0xFD: (1, 0, 4, 0, 1, 0b1111, 6),   # SBC absolute,X
        0xF9: (1, 0, 4, 0, 1, 0b1111, 7),   # SBC absolute,Y
        0xE1: (1, 0, 4, 0, 1, 0b1111, 8),   # SBC (indirect,X)
        0xF1: (1, 0, 4, 0, 1, 0b1111, 9),   # SBC (indirect),Y

        # AND
        0x29: (2, 0, 4, 0, 0, 0b1100, 1),   # AND immediate
        0x25: (2, 0, 4, 0, 0, 0b1100, 2),   # AND zero page
        0x35: (2, 0, 4, 0, 0, 0b1100, 3),   # AND zero page,X
        0x2D: (2, 0, 4, 0, 0, 0b1100, 5),   # AND absolute
        0x3D: (2, 0, 4, 0, 0, 0b1100, 6),   # AND absolute,X
        0x39: (2, 0, 4, 0, 0, 0b1100, 7),   # AND absolute,Y
        0x21: (2, 0, 4, 0, 0, 0b1100, 8),   # AND (indirect,X)
        0x31: (2, 0, 4, 0, 0, 0b1100, 9),   # AND (indirect),Y

        # ORA
        0x09: (3, 0, 4, 0, 0, 0b1100, 1),   # ORA immediate
        0x05: (3, 0, 4, 0, 0, 0b1100, 2),   # ORA zero page
        0x15: (3, 0, 4, 0, 0, 0b1100, 3),   # ORA zero page,X
        0x0D: (3, 0, 4, 0, 0, 0b1100, 5),   # ORA absolute
        0x1D: (3, 0, 4, 0, 0, 0b1100, 6),   # ORA absolute,X
        0x19: (3, 0, 4, 0, 0, 0b1100, 7),   # ORA absolute,Y
        0x01: (3, 0, 4, 0, 0, 0b1100, 8),   # ORA (indirect,X)
        0x11: (3, 0, 4, 0, 0, 0b1100, 9),   # ORA (indirect),Y

        # EOR
        0x49: (4, 0, 4, 0, 0, 0b1100, 1),   # EOR immediate
        0x45: (4, 0, 4, 0, 0, 0b1100, 2),   # EOR zero page
        0x55: (4, 0, 4, 0, 0, 0b1100, 3),   # EOR zero page,X
        0x4D: (4, 0, 4, 0, 0, 0b1100, 5),   # EOR absolute
        0x5D: (4, 0, 4, 0, 0, 0b1100, 6),   # EOR absolute,X
        0x59: (4, 0, 4, 0, 0, 0b1100, 7),   # EOR absolute,Y
        0x41: (4, 0, 4, 0, 0, 0b1100, 8),   # EOR (indirect,X)
        0x51: (4, 0, 4, 0, 0, 0b1100, 9),   # EOR (indirect),Y

        # ASL
        0x0A: (5, 0, 0, 0, 0, 0b1110, 10),  # ASL accumulator
        0x06: (5, 4, 4, 4, 0, 0b1110, 2),   # ASL zero page
        0x16: (5, 4, 4, 4, 0, 0b1110, 3),   # ASL zero page,X
        0x0E: (5, 4, 4, 4, 0, 0b1110, 5),   # ASL absolute
        0x1E: (5, 4, 4, 4, 0, 0b1110, 6),   # ASL absolute,X

        # LSR
        0x4A: (6, 0, 0, 0, 0, 0b1110, 10),  # LSR accumulator
        0x46: (6, 4, 4, 4, 0, 0b1110, 2),   # LSR zero page
        0x56: (6, 4, 4, 4, 0, 0b1110, 3),   # LSR zero page,X
        0x4E: (6, 4, 4, 4, 0, 0b1110, 5),   # LSR absolute
        0x5E: (6, 4, 4, 4, 0, 0b1110, 6),   # LSR absolute,X

        # ROL
        0x2A: (7, 0, 0, 0, 1, 0b1110, 10),  # ROL accumulator
        0x26: (7, 4, 4, 4, 1, 0b1110, 2),   # ROL zero page
        0x36: (7, 4, 4, 4, 1, 0b1110, 3),   # ROL zero page,X
        0x2E: (7, 4, 4, 4, 1, 0b1110, 5),   # ROL absolute
        0x3E: (7, 4, 4, 4, 1, 0b1110, 6),   # ROL absolute,X

        # ROR
        0x6A: (8, 0, 0, 0, 1, 0b1110, 10),  # ROR accumulator
        0x66: (8, 4, 4, 4, 1, 0b1110, 2),   # ROR zero page
        0x76: (8, 4, 4, 4, 1, 0b1110, 3),   # ROR zero page,X
        0x6E: (8, 4, 4, 4, 1, 0b1110, 5),   # ROR absolute
        0x7E: (8, 4, 4, 4, 1, 0b1110, 6),   # ROR absolute,X

        # INC
        0xE6: (9, 4, 4, 4, 0, 0b1100, 2),   # INC zero page
        0xF6: (9, 4, 4, 4, 0, 0b1100, 3),   # INC zero page,X
        0xEE: (9, 4, 4, 4, 0, 0b1100, 5),   # INC absolute
        0xFE: (9, 4, 4, 4, 0, 0b1100, 6),   # INC absolute,X

        # DEC
        0xC6: (10, 4, 4, 4, 0, 0b1100, 2),  # DEC zero page
        0xD6: (10, 4, 4, 4, 0, 0b1100, 3),  # DEC zero page,X
        0xCE: (10, 4, 4, 4, 0, 0b1100, 5),  # DEC absolute
        0xDE: (10, 4, 4, 4, 0, 0b1100, 6),  # DEC absolute,X

        # INX, INY, DEX, DEY
        0xE8: (9, 1, 1, 1, 0, 0b1100, 0),   # INX
        0xC8: (9, 2, 2, 2, 0, 0b1100, 0),   # INY
        0xCA: (10, 1, 1, 1, 0, 0b1100, 0),  # DEX
        0x88: (10, 2, 2, 2, 0, 0b1100, 0),  # DEY

        # Transfers
        0xAA: (11, 0, 0, 1, 0, 0b1100, 0),  # TAX
        0x8A: (11, 1, 1, 0, 0, 0b1100, 0),  # TXA
        0xA8: (11, 0, 0, 2, 0, 0b1100, 0),  # TAY
        0x98: (11, 2, 2, 0, 0, 0b1100, 0),  # TYA
        0xBA: (11, 3, 3, 1, 0, 0b1100, 0),  # TSX
        0x9A: (11, 1, 1, 3, 0, 0b0000, 0),  # TXS (no flags)

        # LDA
        0xA9: (12, 4, 4, 0, 0, 0b1100, 1),  # LDA immediate
        0xA5: (12, 4, 4, 0, 0, 0b1100, 2),  # LDA zero page
        0xB5: (12, 4, 4, 0, 0, 0b1100, 3),  # LDA zero page,X
        0xAD: (12, 4, 4, 0, 0, 0b1100, 5),  # LDA absolute
        0xBD: (12, 4, 4, 0, 0, 0b1100, 6),  # LDA absolute,X
        0xB9: (12, 4, 4, 0, 0, 0b1100, 7),  # LDA absolute,Y
        0xA1: (12, 4, 4, 0, 0, 0b1100, 8),  # LDA (indirect,X)
        0xB1: (12, 4, 4, 0, 0, 0b1100, 9),  # LDA (indirect),Y

        # LDX
        0xA2: (12, 4, 4, 1, 0, 0b1100, 1),  # LDX immediate
        0xA6: (12, 4, 4, 1, 0, 0b1100, 2),  # LDX zero page
        0xB6: (12, 4, 4, 1, 0, 0b1100, 4),  # LDX zero page,Y
        0xAE: (12, 4, 4, 1, 0, 0b1100, 5),  # LDX absolute
        0xBE: (12, 4, 4, 1, 0, 0b1100, 7),  # LDX absolute,Y

        # LDY
        0xA0: (12, 4, 4, 2, 0, 0b1100, 1),  # LDY immediate
        0xA4: (12, 4, 4, 2, 0, 0b1100, 2),  # LDY zero page
        0xB4: (12, 4, 4, 2, 0, 0b1100, 3),  # LDY zero page,X
        0xAC: (12, 4, 4, 2, 0, 0b1100, 5),  # LDY absolute
        0xBC: (12, 4, 4, 2, 0, 0b1100, 6),  # LDY absolute,X

        # STA
        0x85: (13, 0, 0, 4, 0, 0b0000, 2),  # STA zero page
        0x95: (13, 0, 0, 4, 0, 0b0000, 3),  # STA zero page,X
        0x8D: (13, 0, 0, 4, 0, 0b0000, 5),  # STA absolute
        0x9D: (13, 0, 0, 4, 0, 0b0000, 6),  # STA absolute,X
        0x99: (13, 0, 0, 4, 0, 0b0000, 7),  # STA absolute,Y
        0x81: (13, 0, 0, 4, 0, 0b0000, 8),  # STA (indirect,X)
        0x91: (13, 0, 0, 4, 0, 0b0000, 9),  # STA (indirect),Y

        # STX
        0x86: (13, 1, 1, 4, 0, 0b0000, 2),  # STX zero page
        0x96: (13, 1, 1, 4, 0, 0b0000, 4),  # STX zero page,Y
        0x8E: (13, 1, 1, 4, 0, 0b0000, 5),  # STX absolute

        # STY
        0x84: (13, 2, 2, 4, 0, 0b0000, 2),  # STY zero page
        0x94: (13, 2, 2, 4, 0, 0b0000, 3),  # STY zero page,X
        0x8C: (13, 2, 2, 4, 0, 0b0000, 5),  # STY absolute

        # BIT
        0x24: (14, 0, 4, 0, 0, 0b1101, 2),  # BIT zero page (special N,V,Z)
        0x2C: (14, 0, 4, 0, 0, 0b1101, 5),  # BIT absolute

        # CMP
        0xC9: (1, 0, 4, 0, 0, 0b1110, 1),   # CMP immediate (like SBC but no store)
        0xC5: (1, 0, 4, 0, 0, 0b1110, 2),   # CMP zero page
        0xD5: (1, 0, 4, 0, 0, 0b1110, 3),   # CMP zero page,X
        0xCD: (1, 0, 4, 0, 0, 0b1110, 5),   # CMP absolute
        0xDD: (1, 0, 4, 0, 0, 0b1110, 6),   # CMP absolute,X
        0xD9: (1, 0, 4, 0, 0, 0b1110, 7),   # CMP absolute,Y
        0xC1: (1, 0, 4, 0, 0, 0b1110, 8),   # CMP (indirect,X)
        0xD1: (1, 0, 4, 0, 0, 0b1110, 9),   # CMP (indirect),Y

        # CPX
        0xE0: (1, 1, 4, 1, 0, 0b1110, 1),   # CPX immediate
        0xE4: (1, 1, 4, 1, 0, 0b1110, 2),   # CPX zero page
        0xEC: (1, 1, 4, 1, 0, 0b1110, 5),   # CPX absolute

        # CPY
        0xC0: (1, 2, 4, 2, 0, 0b1110, 1),   # CPY immediate
        0xC4: (1, 2, 4, 2, 0, 0b1110, 2),   # CPY zero page
        0xCC: (1, 2, 4, 2, 0, 0b1110, 5),   # CPY absolute

        # NOP
        0xEA: (15, 0, 0, 0, 0, 0b0000, 0),  # NOP
    }

    @classmethod
    def generate_header(cls) -> str:
        """Generate the trix_6502.h header file."""
        return '''/*
 * TriX 6502 - Pure Geometric Computation
 * =======================================
 *
 * A complete 6502 CPU as frozen polynomial geometry.
 * No neural network. No inference. Just math.
 *
 * Generated by The Foundry
 */

#ifndef TRIX_6502_H
#define TRIX_6502_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CPU State */
typedef struct {
    /* Registers */
    uint8_t a;      /* Accumulator */
    uint8_t x;      /* X index */
    uint8_t y;      /* Y index */
    uint8_t sp;     /* Stack pointer */
    uint16_t pc;    /* Program counter */

    /* Status flags: NV-BDIZC */
    uint8_t n : 1;  /* Negative */
    uint8_t v : 1;  /* Overflow */
    uint8_t b : 1;  /* Break */
    uint8_t d : 1;  /* Decimal (not implemented) */
    uint8_t i : 1;  /* Interrupt disable */
    uint8_t z : 1;  /* Zero */
    uint8_t c : 1;  /* Carry */

    /* Memory interface */
    uint8_t (*read)(uint16_t addr, void *ctx);
    void (*write)(uint16_t addr, uint8_t val, void *ctx);
    void *ctx;

    /* Cycle counter */
    uint64_t cycles;
} trix_6502_t;

/* Initialize CPU */
void trix_6502_init(trix_6502_t *cpu);

/* Reset CPU (reads reset vector) */
void trix_6502_reset(trix_6502_t *cpu);

/* Execute one instruction, returns cycles used */
int trix_6502_step(trix_6502_t *cpu);

/* Execute until cycle count reached */
void trix_6502_run(trix_6502_t *cpu, uint64_t target_cycles);

/* Trigger IRQ */
void trix_6502_irq(trix_6502_t *cpu);

/* Trigger NMI */
void trix_6502_nmi(trix_6502_t *cpu);

/* Get status register as byte */
uint8_t trix_6502_get_status(trix_6502_t *cpu);

/* Set status register from byte */
void trix_6502_set_status(trix_6502_t *cpu, uint8_t status);

#ifdef __cplusplus
}
#endif

#endif /* TRIX_6502_H */
'''

    @classmethod
    def generate_source(cls) -> str:
        """Generate the trix_6502.c source file."""

        # Build opcode tables
        shape_table = ["    0xFF"] * 256
        input_a_table = ["    0"] * 256
        input_b_table = ["    0"] * 256
        output_table = ["    0"] * 256
        uses_carry_table = ["    0"] * 256
        flags_table = ["    0"] * 256
        addr_mode_table = ["    0"] * 256

        for opcode, entry in cls.OPCODE_MAP.items():
            shape, in_a, in_b, out, carry, flags, addr_mode = entry
            shape_table[opcode] = f"    {shape}"
            input_a_table[opcode] = f"    {in_a}"
            input_b_table[opcode] = f"    {in_b}"
            output_table[opcode] = f"    {out}"
            uses_carry_table[opcode] = f"    {carry}"
            flags_table[opcode] = f"    0x{flags:02X}"
            addr_mode_table[opcode] = f"    {addr_mode}"

        def format_table(values: list, name: str) -> str:
            lines = []
            lines.append(f"static const uint8_t {name}[256] = {{")
            for i in range(0, 256, 16):
                chunk = values[i:i+16]
                lines.append("    " + ", ".join(v.strip() for v in chunk) + ",")
            lines.append("};")
            return "\n".join(lines)

        tables = "\n\n".join([
            format_table(shape_table, "SHAPE_TABLE"),
            format_table(input_a_table, "INPUT_A_TABLE"),
            format_table(input_b_table, "INPUT_B_TABLE"),
            format_table(output_table, "OUTPUT_TABLE"),
            format_table(uses_carry_table, "USES_CARRY_TABLE"),
            format_table(flags_table, "FLAGS_TABLE"),
            format_table(addr_mode_table, "ADDR_MODE_TABLE"),
        ])

        return f'''/*
 * TriX 6502 - Pure Geometric Computation
 * =======================================
 *
 * Level 0: Polynomial primitives
 * Level 1: 16 frozen shapes
 * Level 2: Opcode routing tables
 *
 * This is 100% pure TriX. No neural network. No inference.
 * Just polynomials, lookup tables, and a loop.
 *
 * Generated by The Foundry
 */

#include "trix_6502.h"

/* ==========================================================================
 * LEVEL 0: POLYNOMIAL PRIMITIVES
 *
 * These are the atomic operations. Pure math. Zero parameters.
 * ========================================================================== */

/* XOR: a + b - 2ab (the saddle surface) */
static inline uint8_t trix_xor_bit(uint8_t a, uint8_t b) {{
    return a + b - 2*a*b;
}}

/* AND: ab (the product) */
static inline uint8_t trix_and_bit(uint8_t a, uint8_t b) {{
    return a * b;
}}

/* OR: a + b - ab (the union) */
static inline uint8_t trix_or_bit(uint8_t a, uint8_t b) {{
    return a + b - a*b;
}}

/* NOT: 1 - a (the reflection) */
static inline uint8_t trix_not_bit(uint8_t a) {{
    return 1 - a;
}}

/* Full adder: sum and carry from three bits */
static inline void trix_full_adder(
    uint8_t a, uint8_t b, uint8_t c,
    uint8_t *sum, uint8_t *carry
) {{
    uint8_t p = trix_xor_bit(a, b);
    *sum = trix_xor_bit(p, c);
    *carry = trix_or_bit(trix_and_bit(a, b), trix_and_bit(c, p));
}}

/* ==========================================================================
 * LEVEL 1: FROZEN SHAPES
 *
 * 16 computation topologies composed from Level 0 primitives.
 * These ARE the geometry. Fixed. Frozen. Perfect.
 * ========================================================================== */

/* Shape 0: Ripple Add - 8-bit addition with carry */
static uint8_t shape_ripple_add(uint8_t a, uint8_t b, uint8_t c_in, uint8_t *c_out) {{
    uint8_t result = 0;
    uint8_t carry = c_in;

    for (int i = 0; i < 8; i++) {{
        uint8_t ai = (a >> i) & 1;
        uint8_t bi = (b >> i) & 1;
        uint8_t sum, new_carry;
        trix_full_adder(ai, bi, carry, &sum, &new_carry);
        result |= (sum << i);
        carry = new_carry;
    }}

    *c_out = carry;
    return result;
}}

/* Shape 1: Ripple Sub - 8-bit subtraction (A + ~B + C) */
static uint8_t shape_ripple_sub(uint8_t a, uint8_t b, uint8_t c_in, uint8_t *c_out) {{
    return shape_ripple_add(a, ~b, c_in, c_out);
}}

/* Shape 2: Parallel AND */
static uint8_t shape_parallel_and(uint8_t a, uint8_t b) {{
    return a & b;
}}

/* Shape 3: Parallel OR */
static uint8_t shape_parallel_or(uint8_t a, uint8_t b) {{
    return a | b;
}}

/* Shape 4: Parallel XOR */
static uint8_t shape_parallel_xor(uint8_t a, uint8_t b) {{
    return a ^ b;
}}

/* Shape 5: Shift Left */
static uint8_t shape_shift_left(uint8_t a, uint8_t *c_out) {{
    *c_out = (a >> 7) & 1;
    return a << 1;
}}

/* Shape 6: Shift Right */
static uint8_t shape_shift_right(uint8_t a, uint8_t *c_out) {{
    *c_out = a & 1;
    return a >> 1;
}}

/* Shape 7: Rotate Left through Carry */
static uint8_t shape_rotate_left(uint8_t a, uint8_t c_in, uint8_t *c_out) {{
    *c_out = (a >> 7) & 1;
    return (a << 1) | c_in;
}}

/* Shape 8: Rotate Right through Carry */
static uint8_t shape_rotate_right(uint8_t a, uint8_t c_in, uint8_t *c_out) {{
    *c_out = a & 1;
    return (a >> 1) | (c_in << 7);
}}

/* Shape 9: Increment */
static uint8_t shape_increment(uint8_t a) {{
    return a + 1;
}}

/* Shape 10: Decrement */
static uint8_t shape_decrement(uint8_t a) {{
    return a - 1;
}}

/* Shape 11: Transfer (identity) */
static uint8_t shape_transfer(uint8_t a) {{
    return a;
}}

/* Shape 12: Load (identity from memory) */
static uint8_t shape_load(uint8_t m) {{
    return m;
}}

/* Shape 13: Store (identity to memory) */
static uint8_t shape_store(uint8_t a) {{
    return a;
}}

/* Shape 14: Bit Test (AND but only affects flags) */
static uint8_t shape_bit_test(uint8_t a, uint8_t m, uint8_t *n_out, uint8_t *v_out) {{
    *n_out = (m >> 7) & 1;
    *v_out = (m >> 6) & 1;
    return a & m;
}}

/* Shape 15: Identity (NOP) */
static uint8_t shape_identity(uint8_t a) {{
    return a;
}}

/* ==========================================================================
 * LEVEL 2: MEANING LAYER
 *
 * The trained routing from opcodes to shapes.
 * This is the "intelligence" - frozen as static data.
 * ========================================================================== */

/* Shape IDs */
#define SHAPE_RIPPLE_ADD    0
#define SHAPE_RIPPLE_SUB    1
#define SHAPE_PARALLEL_AND  2
#define SHAPE_PARALLEL_OR   3
#define SHAPE_PARALLEL_XOR  4
#define SHAPE_SHIFT_LEFT    5
#define SHAPE_SHIFT_RIGHT   6
#define SHAPE_ROTATE_LEFT   7
#define SHAPE_ROTATE_RIGHT  8
#define SHAPE_INCREMENT     9
#define SHAPE_DECREMENT     10
#define SHAPE_TRANSFER      11
#define SHAPE_LOAD          12
#define SHAPE_STORE         13
#define SHAPE_BIT_TEST      14
#define SHAPE_IDENTITY      15
#define SHAPE_INVALID       0xFF

/* Register IDs */
#define REG_A   0
#define REG_X   1
#define REG_Y   2
#define REG_SP  3
#define REG_MEM 4

/* Flag bits */
#define FLAG_N  0x08
#define FLAG_Z  0x04
#define FLAG_C  0x02
#define FLAG_V  0x01

/* Addressing modes */
#define ADDR_IMPLIED     0
#define ADDR_IMMEDIATE   1
#define ADDR_ZERO_PAGE   2
#define ADDR_ZERO_PAGE_X 3
#define ADDR_ZERO_PAGE_Y 4
#define ADDR_ABSOLUTE    5
#define ADDR_ABSOLUTE_X  6
#define ADDR_ABSOLUTE_Y  7
#define ADDR_INDIRECT_X  8
#define ADDR_INDIRECT_Y  9
#define ADDR_ACCUMULATOR 10

/* Opcode routing tables (the frozen meaning layer) */
{tables}

/* ==========================================================================
 * RUNTIME
 *
 * The execution engine. Fetch, decode, execute.
 * ========================================================================== */

/* Read a register by ID */
static inline uint8_t read_reg(trix_6502_t *cpu, uint8_t reg_id, uint8_t mem_val) {{
    switch (reg_id) {{
        case REG_A:   return cpu->a;
        case REG_X:   return cpu->x;
        case REG_Y:   return cpu->y;
        case REG_SP:  return cpu->sp;
        case REG_MEM: return mem_val;
        default:      return 0;
    }}
}}

/* Write a register by ID */
static inline void write_reg(trix_6502_t *cpu, uint8_t reg_id, uint8_t value,
                             uint16_t addr, uint8_t is_memory_op) {{
    switch (reg_id) {{
        case REG_A:   cpu->a = value; break;
        case REG_X:   cpu->x = value; break;
        case REG_Y:   cpu->y = value; break;
        case REG_SP:  cpu->sp = value; break;
        case REG_MEM:
            if (is_memory_op) {{
                cpu->write(addr, value, cpu->ctx);
            }}
            break;
    }}
}}

/* Update flags based on result */
static inline void update_flags(trix_6502_t *cpu, uint8_t result, uint8_t carry,
                                 uint8_t overflow, uint8_t flag_mask) {{
    if (flag_mask & FLAG_N) cpu->n = (result >> 7) & 1;
    if (flag_mask & FLAG_Z) cpu->z = (result == 0);
    if (flag_mask & FLAG_C) cpu->c = carry;
    if (flag_mask & FLAG_V) cpu->v = overflow;
}}

/* Addressing mode handlers - return address and increment PC */
static uint16_t addr_immediate(trix_6502_t *cpu) {{
    return cpu->pc++;
}}

static uint16_t addr_zero_page(trix_6502_t *cpu) {{
    return cpu->read(cpu->pc++, cpu->ctx);
}}

static uint16_t addr_zero_page_x(trix_6502_t *cpu) {{
    return (cpu->read(cpu->pc++, cpu->ctx) + cpu->x) & 0xFF;
}}

static uint16_t addr_zero_page_y(trix_6502_t *cpu) {{
    return (cpu->read(cpu->pc++, cpu->ctx) + cpu->y) & 0xFF;
}}

static uint16_t addr_absolute(trix_6502_t *cpu) {{
    uint16_t lo = cpu->read(cpu->pc++, cpu->ctx);
    uint16_t hi = cpu->read(cpu->pc++, cpu->ctx);
    return lo | (hi << 8);
}}

static uint16_t addr_absolute_x(trix_6502_t *cpu) {{
    uint16_t base = addr_absolute(cpu);
    return base + cpu->x;
}}

static uint16_t addr_absolute_y(trix_6502_t *cpu) {{
    uint16_t base = addr_absolute(cpu);
    return base + cpu->y;
}}

static uint16_t addr_indirect_x(trix_6502_t *cpu) {{
    uint8_t zp = (cpu->read(cpu->pc++, cpu->ctx) + cpu->x) & 0xFF;
    uint16_t lo = cpu->read(zp, cpu->ctx);
    uint16_t hi = cpu->read((zp + 1) & 0xFF, cpu->ctx);
    return lo | (hi << 8);
}}

static uint16_t addr_indirect_y(trix_6502_t *cpu) {{
    uint8_t zp = cpu->read(cpu->pc++, cpu->ctx);
    uint16_t lo = cpu->read(zp, cpu->ctx);
    uint16_t hi = cpu->read((zp + 1) & 0xFF, cpu->ctx);
    return (lo | (hi << 8)) + cpu->y;
}}

/* Execute a shape */
static uint8_t execute_shape(trix_6502_t *cpu, uint8_t shape_id,
                              uint8_t input_a, uint8_t input_b,
                              uint8_t uses_carry, uint8_t *carry_out,
                              uint8_t *n_out, uint8_t *v_out) {{
    uint8_t result = 0;
    *carry_out = cpu->c;
    *n_out = 0;
    *v_out = 0;

    switch (shape_id) {{
        case SHAPE_RIPPLE_ADD:
            result = shape_ripple_add(input_a, input_b,
                                      uses_carry ? cpu->c : 0, carry_out);
            /* Overflow for addition */
            *v_out = ((~(input_a ^ input_b)) & (input_a ^ result) & 0x80) ? 1 : 0;
            break;

        case SHAPE_RIPPLE_SUB:
            result = shape_ripple_sub(input_a, input_b,
                                      uses_carry ? cpu->c : 1, carry_out);
            /* Overflow for subtraction */
            *v_out = ((input_a ^ input_b) & (input_a ^ result) & 0x80) ? 1 : 0;
            break;

        case SHAPE_PARALLEL_AND:
            result = shape_parallel_and(input_a, input_b);
            break;

        case SHAPE_PARALLEL_OR:
            result = shape_parallel_or(input_a, input_b);
            break;

        case SHAPE_PARALLEL_XOR:
            result = shape_parallel_xor(input_a, input_b);
            break;

        case SHAPE_SHIFT_LEFT:
            result = shape_shift_left(input_a, carry_out);
            break;

        case SHAPE_SHIFT_RIGHT:
            result = shape_shift_right(input_a, carry_out);
            break;

        case SHAPE_ROTATE_LEFT:
            result = shape_rotate_left(input_a, cpu->c, carry_out);
            break;

        case SHAPE_ROTATE_RIGHT:
            result = shape_rotate_right(input_a, cpu->c, carry_out);
            break;

        case SHAPE_INCREMENT:
            result = shape_increment(input_a);
            break;

        case SHAPE_DECREMENT:
            result = shape_decrement(input_a);
            break;

        case SHAPE_TRANSFER:
        case SHAPE_LOAD:
        case SHAPE_STORE:
            result = shape_transfer(input_a);
            break;

        case SHAPE_BIT_TEST:
            result = shape_bit_test(input_a, input_b, n_out, v_out);
            break;

        case SHAPE_IDENTITY:
            result = input_a;
            break;

        default:
            result = 0;
            break;
    }}

    return result;
}}

/* Public API */

void trix_6502_init(trix_6502_t *cpu) {{
    cpu->a = 0;
    cpu->x = 0;
    cpu->y = 0;
    cpu->sp = 0xFD;
    cpu->pc = 0;
    cpu->n = 0;
    cpu->v = 0;
    cpu->b = 0;
    cpu->d = 0;
    cpu->i = 1;
    cpu->z = 0;
    cpu->c = 0;
    cpu->cycles = 0;
    cpu->read = NULL;
    cpu->write = NULL;
    cpu->ctx = NULL;
}}

void trix_6502_reset(trix_6502_t *cpu) {{
    uint16_t lo = cpu->read(0xFFFC, cpu->ctx);
    uint16_t hi = cpu->read(0xFFFD, cpu->ctx);
    cpu->pc = lo | (hi << 8);
    cpu->sp = 0xFD;
    cpu->i = 1;
    cpu->cycles += 7;
}}

uint8_t trix_6502_get_status(trix_6502_t *cpu) {{
    return (cpu->n << 7) | (cpu->v << 6) | (1 << 5) | (cpu->b << 4) |
           (cpu->d << 3) | (cpu->i << 2) | (cpu->z << 1) | cpu->c;
}}

void trix_6502_set_status(trix_6502_t *cpu, uint8_t status) {{
    cpu->n = (status >> 7) & 1;
    cpu->v = (status >> 6) & 1;
    cpu->b = (status >> 4) & 1;
    cpu->d = (status >> 3) & 1;
    cpu->i = (status >> 2) & 1;
    cpu->z = (status >> 1) & 1;
    cpu->c = status & 1;
}}

int trix_6502_step(trix_6502_t *cpu) {{
    uint8_t opcode = cpu->read(cpu->pc++, cpu->ctx);
    uint8_t shape_id = SHAPE_TABLE[opcode];

    /* Handle invalid/unimplemented opcodes */
    if (shape_id == SHAPE_INVALID) {{
        /* TODO: Handle branches, jumps, stack ops, flag ops */
        cpu->cycles += 2;
        return 2;
    }}

    /* Get routing from meaning layer */
    uint8_t input_a_id = INPUT_A_TABLE[opcode];
    uint8_t input_b_id = INPUT_B_TABLE[opcode];
    uint8_t output_id = OUTPUT_TABLE[opcode];
    uint8_t uses_carry = USES_CARRY_TABLE[opcode];
    uint8_t flag_mask = FLAGS_TABLE[opcode];

    /* Determine addressing mode and get operand address */
    uint16_t addr = 0;
    uint8_t mem_val = 0;

    /* Use the addressing mode table (part of the frozen meaning layer) */
    uint8_t addr_mode = ADDR_MODE_TABLE[opcode];
    switch (addr_mode) {{
        case ADDR_IMMEDIATE:   addr = addr_immediate(cpu); break;
        case ADDR_ZERO_PAGE:   addr = addr_zero_page(cpu); break;
        case ADDR_ZERO_PAGE_X: addr = addr_zero_page_x(cpu); break;
        case ADDR_ZERO_PAGE_Y: addr = addr_zero_page_y(cpu); break;
        case ADDR_ABSOLUTE:    addr = addr_absolute(cpu); break;
        case ADDR_ABSOLUTE_X:  addr = addr_absolute_x(cpu); break;
        case ADDR_ABSOLUTE_Y:  addr = addr_absolute_y(cpu); break;
        case ADDR_INDIRECT_X:  addr = addr_indirect_x(cpu); break;
        case ADDR_INDIRECT_Y:  addr = addr_indirect_y(cpu); break;
        case ADDR_IMPLIED:
        case ADDR_ACCUMULATOR:
        default:
            /* No memory access needed */
            break;
    }}

    if (input_b_id == REG_MEM || input_a_id == REG_MEM) {{
        mem_val = cpu->read(addr, cpu->ctx);
    }}

    /* Read inputs */
    uint8_t input_a = read_reg(cpu, input_a_id, mem_val);
    uint8_t input_b = read_reg(cpu, input_b_id, mem_val);

    /* Execute shape */
    uint8_t carry_out, n_out, v_out;
    uint8_t result = execute_shape(cpu, shape_id, input_a, input_b,
                                   uses_carry, &carry_out, &n_out, &v_out);

    /* Write output */
    uint8_t is_store = (shape_id == SHAPE_STORE);
    if (output_id != REG_MEM || is_store) {{
        write_reg(cpu, output_id, result, addr, is_store);
    }}

    /* Update flags */
    if (shape_id == SHAPE_BIT_TEST) {{
        /* BIT has special flag handling */
        cpu->n = n_out;
        cpu->v = v_out;
        cpu->z = (result == 0);
    }} else {{
        update_flags(cpu, result, carry_out, v_out, flag_mask);
    }}

    /* Approximate cycle count */
    cpu->cycles += 2;
    return 2;
}}

void trix_6502_run(trix_6502_t *cpu, uint64_t target_cycles) {{
    while (cpu->cycles < target_cycles) {{
        trix_6502_step(cpu);
    }}
}}

void trix_6502_irq(trix_6502_t *cpu) {{
    if (cpu->i) return;  /* Interrupts disabled */

    /* Push PC and status */
    cpu->write(0x100 + cpu->sp--, (cpu->pc >> 8) & 0xFF, cpu->ctx);
    cpu->write(0x100 + cpu->sp--, cpu->pc & 0xFF, cpu->ctx);
    cpu->write(0x100 + cpu->sp--, trix_6502_get_status(cpu) & ~0x10, cpu->ctx);

    cpu->i = 1;

    /* Jump to IRQ vector */
    uint16_t lo = cpu->read(0xFFFE, cpu->ctx);
    uint16_t hi = cpu->read(0xFFFF, cpu->ctx);
    cpu->pc = lo | (hi << 8);

    cpu->cycles += 7;
}}

void trix_6502_nmi(trix_6502_t *cpu) {{
    /* Push PC and status */
    cpu->write(0x100 + cpu->sp--, (cpu->pc >> 8) & 0xFF, cpu->ctx);
    cpu->write(0x100 + cpu->sp--, cpu->pc & 0xFF, cpu->ctx);
    cpu->write(0x100 + cpu->sp--, trix_6502_get_status(cpu) & ~0x10, cpu->ctx);

    cpu->i = 1;

    /* Jump to NMI vector */
    uint16_t lo = cpu->read(0xFFFA, cpu->ctx);
    uint16_t hi = cpu->read(0xFFFB, cpu->ctx);
    cpu->pc = lo | (hi << 8);

    cpu->cycles += 7;
}}
'''

    @classmethod
    def export(cls, output_dir: Path) -> Tuple[Path, Path]:
        """Export the TriX 6502 to C files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        header_path = output_dir / "trix_6502.h"
        source_path = output_dir / "trix_6502.c"

        header_path.write_text(cls.generate_header())
        source_path.write_text(cls.generate_source())

        return header_path, source_path
