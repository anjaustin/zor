/*
 * Frozen 6502 - Complete CPU Emulator
 *
 * A full MOS 6502 emulator using 16 frozen shapes.
 * All 151 valid opcodes, 13 addressing modes, 64KB memory.
 *
 * Build: gcc -O2 frozen_6502_complete.c -o f6502
 * Test:  ./f6502 program.bin
 *
 * "Computation is geometry. Learning is routing."
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * CPU STATE
 * ============================================================================ */

typedef struct {
    /* Registers */
    uint8_t  A;         /* Accumulator */
    uint8_t  X;         /* X index */
    uint8_t  Y;         /* Y index */
    uint8_t  SP;        /* Stack pointer */
    uint16_t PC;        /* Program counter */

    /* Status flags (bit positions in P register) */
    uint8_t  C : 1;     /* Carry */
    uint8_t  Z : 1;     /* Zero */
    uint8_t  I : 1;     /* Interrupt disable */
    uint8_t  D : 1;     /* Decimal mode (not implemented) */
    uint8_t  B : 1;     /* Break */
    uint8_t  U : 1;     /* Unused (always 1) */
    uint8_t  V : 1;     /* Overflow */
    uint8_t  N : 1;     /* Negative */

    /* Memory */
    uint8_t  mem[65536];

    /* Cycle counter */
    uint64_t cycles;

    /* Halt flag */
    uint8_t  halted;
} CPU;

/* Pack flags into P register */
static inline uint8_t get_P(CPU *c) {
    return (c->N << 7) | (c->V << 6) | (1 << 5) | (c->B << 4) |
           (c->D << 3) | (c->I << 2) | (c->Z << 1) | c->C;
}

/* Unpack P register into flags */
static inline void set_P(CPU *c, uint8_t p) {
    c->C = p & 1;
    c->Z = (p >> 1) & 1;
    c->I = (p >> 2) & 1;
    c->D = (p >> 3) & 1;
    c->B = (p >> 4) & 1;
    c->V = (p >> 6) & 1;
    c->N = (p >> 7) & 1;
}

/* ============================================================================
 * THE 16 FROZEN SHAPES
 * ============================================================================ */

/* Update N and Z flags */
static inline void upd_nz(CPU *c, uint8_t v) {
    c->Z = (v == 0);
    c->N = (v >> 7) & 1;
}

/* Shape 0: ADD with carry */
static inline uint8_t shape_adc(CPU *c, uint8_t a, uint8_t b) {
    uint16_t sum = a + b + c->C;
    c->V = ((a ^ sum) & (b ^ sum) & 0x80) ? 1 : 0;
    c->C = (sum > 0xFF) ? 1 : 0;
    uint8_t r = (uint8_t)sum;
    upd_nz(c, r);
    return r;
}

/* Shape 1: SUB with borrow (SBC = A - M - !C) */
static inline uint8_t shape_sbc(CPU *c, uint8_t a, uint8_t b) {
    uint16_t diff = a - b - (1 - c->C);
    c->V = ((a ^ diff) & (a ^ b) & 0x80) ? 1 : 0;
    c->C = (diff <= 0xFF) ? 1 : 0;
    uint8_t r = (uint8_t)diff;
    upd_nz(c, r);
    return r;
}

/* Shape 2: AND */
static inline uint8_t shape_and(CPU *c, uint8_t a, uint8_t b) {
    uint8_t r = a & b;
    upd_nz(c, r);
    return r;
}

/* Shape 3: OR */
static inline uint8_t shape_ora(CPU *c, uint8_t a, uint8_t b) {
    uint8_t r = a | b;
    upd_nz(c, r);
    return r;
}

/* Shape 4: XOR */
static inline uint8_t shape_eor(CPU *c, uint8_t a, uint8_t b) {
    uint8_t r = a ^ b;
    upd_nz(c, r);
    return r;
}

/* Shape 5: Shift left */
static inline uint8_t shape_asl(CPU *c, uint8_t a) {
    c->C = (a >> 7) & 1;
    uint8_t r = a << 1;
    upd_nz(c, r);
    return r;
}

/* Shape 6: Shift right */
static inline uint8_t shape_lsr(CPU *c, uint8_t a) {
    c->C = a & 1;
    uint8_t r = a >> 1;
    upd_nz(c, r);
    return r;
}

/* Shape 7: Rotate left */
static inline uint8_t shape_rol(CPU *c, uint8_t a) {
    uint8_t old_c = c->C;
    c->C = (a >> 7) & 1;
    uint8_t r = (a << 1) | old_c;
    upd_nz(c, r);
    return r;
}

/* Shape 8: Rotate right */
static inline uint8_t shape_ror(CPU *c, uint8_t a) {
    uint8_t old_c = c->C;
    c->C = a & 1;
    uint8_t r = (a >> 1) | (old_c << 7);
    upd_nz(c, r);
    return r;
}

/* Shape 9: Increment */
static inline uint8_t shape_inc(CPU *c, uint8_t a) {
    uint8_t r = a + 1;
    upd_nz(c, r);
    return r;
}

/* Shape 10: Decrement */
static inline uint8_t shape_dec(CPU *c, uint8_t a) {
    uint8_t r = a - 1;
    upd_nz(c, r);
    return r;
}

/* Shape 11: Transfer (with NZ update) */
static inline uint8_t shape_transfer(CPU *c, uint8_t a) {
    upd_nz(c, a);
    return a;
}

/* Shape 12: Load (same as transfer) */
static inline uint8_t shape_load(CPU *c, uint8_t a) {
    upd_nz(c, a);
    return a;
}

/* Shape 13: Store (no flag update) */
static inline uint8_t shape_store(uint8_t a) {
    return a;
}

/* Shape 14: BIT test */
static inline void shape_bit(CPU *c, uint8_t a, uint8_t m) {
    c->Z = ((a & m) == 0);
    c->N = (m >> 7) & 1;
    c->V = (m >> 6) & 1;
}

/* Shape 15: Compare (like SBC but don't store result) */
static inline void shape_cmp(CPU *c, uint8_t a, uint8_t b) {
    uint16_t diff = a - b;
    c->C = (a >= b) ? 1 : 0;
    c->Z = (a == b) ? 1 : 0;
    c->N = ((uint8_t)diff >> 7) & 1;
}

/* ============================================================================
 * MEMORY ACCESS
 * ============================================================================ */

static inline uint8_t read8(CPU *c, uint16_t addr) {
    return c->mem[addr];
}

static inline void write8(CPU *c, uint16_t addr, uint8_t val) {
    c->mem[addr] = val;
}

static inline uint16_t read16(CPU *c, uint16_t addr) {
    return read8(c, addr) | (read8(c, addr + 1) << 8);
}

/* Bug-compatible 6502 indirect read (wraps within page) */
static inline uint16_t read16_wrap(CPU *c, uint16_t addr) {
    uint16_t lo = read8(c, addr);
    uint16_t hi_addr = (addr & 0xFF00) | ((addr + 1) & 0x00FF);
    uint16_t hi = read8(c, hi_addr);
    return lo | (hi << 8);
}

/* Stack operations */
static inline void push8(CPU *c, uint8_t val) {
    write8(c, 0x0100 + c->SP, val);
    c->SP--;
}

static inline void push16(CPU *c, uint16_t val) {
    push8(c, (val >> 8) & 0xFF);
    push8(c, val & 0xFF);
}

static inline uint8_t pull8(CPU *c) {
    c->SP++;
    return read8(c, 0x0100 + c->SP);
}

static inline uint16_t pull16(CPU *c) {
    uint16_t lo = pull8(c);
    uint16_t hi = pull8(c);
    return lo | (hi << 8);
}

/* ============================================================================
 * ADDRESSING MODES
 * ============================================================================ */

typedef enum {
    AM_IMP,     /* Implied */
    AM_ACC,     /* Accumulator */
    AM_IMM,     /* Immediate #$nn */
    AM_ZP,      /* Zero Page $nn */
    AM_ZPX,     /* Zero Page,X $nn,X */
    AM_ZPY,     /* Zero Page,Y $nn,Y */
    AM_ABS,     /* Absolute $nnnn */
    AM_ABX,     /* Absolute,X $nnnn,X */
    AM_ABY,     /* Absolute,Y $nnnn,Y */
    AM_IND,     /* Indirect ($nnnn) */
    AM_IZX,     /* (Indirect,X) ($nn,X) */
    AM_IZY,     /* (Indirect),Y ($nn),Y */
    AM_REL,     /* Relative (branches) */
} AddrMode;

/* Get effective address for addressing mode */
static uint16_t get_addr(CPU *c, AddrMode mode) {
    uint16_t addr;
    switch (mode) {
        case AM_IMM: return c->PC++;
        case AM_ZP:  return read8(c, c->PC++);
        case AM_ZPX: return (read8(c, c->PC++) + c->X) & 0xFF;
        case AM_ZPY: return (read8(c, c->PC++) + c->Y) & 0xFF;
        case AM_ABS: addr = read16(c, c->PC); c->PC += 2; return addr;
        case AM_ABX: addr = read16(c, c->PC); c->PC += 2; return addr + c->X;
        case AM_ABY: addr = read16(c, c->PC); c->PC += 2; return addr + c->Y;
        case AM_IND: addr = read16(c, c->PC); c->PC += 2; return read16_wrap(c, addr);
        case AM_IZX: addr = (read8(c, c->PC++) + c->X) & 0xFF; return read16(c, addr);
        case AM_IZY: addr = read8(c, c->PC++); return read16(c, addr) + c->Y;
        default: return 0;
    }
}

/* Get operand value */
static uint8_t get_val(CPU *c, AddrMode mode) {
    if (mode == AM_ACC) return c->A;
    if (mode == AM_IMP) return 0;
    return read8(c, get_addr(c, mode));
}

/* ============================================================================
 * INSTRUCTION EXECUTION
 * ============================================================================ */

static void step(CPU *c) {
    uint8_t op = read8(c, c->PC++);
    uint16_t addr;
    uint8_t val;
    int8_t offset;

    switch (op) {
        /* === ADC === */
        case 0x69: c->A = shape_adc(c, c->A, get_val(c, AM_IMM)); break;
        case 0x65: c->A = shape_adc(c, c->A, get_val(c, AM_ZP)); break;
        case 0x75: c->A = shape_adc(c, c->A, get_val(c, AM_ZPX)); break;
        case 0x6D: c->A = shape_adc(c, c->A, get_val(c, AM_ABS)); break;
        case 0x7D: c->A = shape_adc(c, c->A, get_val(c, AM_ABX)); break;
        case 0x79: c->A = shape_adc(c, c->A, get_val(c, AM_ABY)); break;
        case 0x61: c->A = shape_adc(c, c->A, get_val(c, AM_IZX)); break;
        case 0x71: c->A = shape_adc(c, c->A, get_val(c, AM_IZY)); break;

        /* === SBC === */
        case 0xE9: c->A = shape_sbc(c, c->A, get_val(c, AM_IMM)); break;
        case 0xE5: c->A = shape_sbc(c, c->A, get_val(c, AM_ZP)); break;
        case 0xF5: c->A = shape_sbc(c, c->A, get_val(c, AM_ZPX)); break;
        case 0xED: c->A = shape_sbc(c, c->A, get_val(c, AM_ABS)); break;
        case 0xFD: c->A = shape_sbc(c, c->A, get_val(c, AM_ABX)); break;
        case 0xF9: c->A = shape_sbc(c, c->A, get_val(c, AM_ABY)); break;
        case 0xE1: c->A = shape_sbc(c, c->A, get_val(c, AM_IZX)); break;
        case 0xF1: c->A = shape_sbc(c, c->A, get_val(c, AM_IZY)); break;

        /* === AND === */
        case 0x29: c->A = shape_and(c, c->A, get_val(c, AM_IMM)); break;
        case 0x25: c->A = shape_and(c, c->A, get_val(c, AM_ZP)); break;
        case 0x35: c->A = shape_and(c, c->A, get_val(c, AM_ZPX)); break;
        case 0x2D: c->A = shape_and(c, c->A, get_val(c, AM_ABS)); break;
        case 0x3D: c->A = shape_and(c, c->A, get_val(c, AM_ABX)); break;
        case 0x39: c->A = shape_and(c, c->A, get_val(c, AM_ABY)); break;
        case 0x21: c->A = shape_and(c, c->A, get_val(c, AM_IZX)); break;
        case 0x31: c->A = shape_and(c, c->A, get_val(c, AM_IZY)); break;

        /* === ORA === */
        case 0x09: c->A = shape_ora(c, c->A, get_val(c, AM_IMM)); break;
        case 0x05: c->A = shape_ora(c, c->A, get_val(c, AM_ZP)); break;
        case 0x15: c->A = shape_ora(c, c->A, get_val(c, AM_ZPX)); break;
        case 0x0D: c->A = shape_ora(c, c->A, get_val(c, AM_ABS)); break;
        case 0x1D: c->A = shape_ora(c, c->A, get_val(c, AM_ABX)); break;
        case 0x19: c->A = shape_ora(c, c->A, get_val(c, AM_ABY)); break;
        case 0x01: c->A = shape_ora(c, c->A, get_val(c, AM_IZX)); break;
        case 0x11: c->A = shape_ora(c, c->A, get_val(c, AM_IZY)); break;

        /* === EOR === */
        case 0x49: c->A = shape_eor(c, c->A, get_val(c, AM_IMM)); break;
        case 0x45: c->A = shape_eor(c, c->A, get_val(c, AM_ZP)); break;
        case 0x55: c->A = shape_eor(c, c->A, get_val(c, AM_ZPX)); break;
        case 0x4D: c->A = shape_eor(c, c->A, get_val(c, AM_ABS)); break;
        case 0x5D: c->A = shape_eor(c, c->A, get_val(c, AM_ABX)); break;
        case 0x59: c->A = shape_eor(c, c->A, get_val(c, AM_ABY)); break;
        case 0x41: c->A = shape_eor(c, c->A, get_val(c, AM_IZX)); break;
        case 0x51: c->A = shape_eor(c, c->A, get_val(c, AM_IZY)); break;

        /* === ASL === */
        case 0x0A: c->A = shape_asl(c, c->A); break;
        case 0x06: addr = get_addr(c, AM_ZP);  write8(c, addr, shape_asl(c, read8(c, addr))); break;
        case 0x16: addr = get_addr(c, AM_ZPX); write8(c, addr, shape_asl(c, read8(c, addr))); break;
        case 0x0E: addr = get_addr(c, AM_ABS); write8(c, addr, shape_asl(c, read8(c, addr))); break;
        case 0x1E: addr = get_addr(c, AM_ABX); write8(c, addr, shape_asl(c, read8(c, addr))); break;

        /* === LSR === */
        case 0x4A: c->A = shape_lsr(c, c->A); break;
        case 0x46: addr = get_addr(c, AM_ZP);  write8(c, addr, shape_lsr(c, read8(c, addr))); break;
        case 0x56: addr = get_addr(c, AM_ZPX); write8(c, addr, shape_lsr(c, read8(c, addr))); break;
        case 0x4E: addr = get_addr(c, AM_ABS); write8(c, addr, shape_lsr(c, read8(c, addr))); break;
        case 0x5E: addr = get_addr(c, AM_ABX); write8(c, addr, shape_lsr(c, read8(c, addr))); break;

        /* === ROL === */
        case 0x2A: c->A = shape_rol(c, c->A); break;
        case 0x26: addr = get_addr(c, AM_ZP);  write8(c, addr, shape_rol(c, read8(c, addr))); break;
        case 0x36: addr = get_addr(c, AM_ZPX); write8(c, addr, shape_rol(c, read8(c, addr))); break;
        case 0x2E: addr = get_addr(c, AM_ABS); write8(c, addr, shape_rol(c, read8(c, addr))); break;
        case 0x3E: addr = get_addr(c, AM_ABX); write8(c, addr, shape_rol(c, read8(c, addr))); break;

        /* === ROR === */
        case 0x6A: c->A = shape_ror(c, c->A); break;
        case 0x66: addr = get_addr(c, AM_ZP);  write8(c, addr, shape_ror(c, read8(c, addr))); break;
        case 0x76: addr = get_addr(c, AM_ZPX); write8(c, addr, shape_ror(c, read8(c, addr))); break;
        case 0x6E: addr = get_addr(c, AM_ABS); write8(c, addr, shape_ror(c, read8(c, addr))); break;
        case 0x7E: addr = get_addr(c, AM_ABX); write8(c, addr, shape_ror(c, read8(c, addr))); break;

        /* === INC === */
        case 0xE6: addr = get_addr(c, AM_ZP);  write8(c, addr, shape_inc(c, read8(c, addr))); break;
        case 0xF6: addr = get_addr(c, AM_ZPX); write8(c, addr, shape_inc(c, read8(c, addr))); break;
        case 0xEE: addr = get_addr(c, AM_ABS); write8(c, addr, shape_inc(c, read8(c, addr))); break;
        case 0xFE: addr = get_addr(c, AM_ABX); write8(c, addr, shape_inc(c, read8(c, addr))); break;

        /* === DEC === */
        case 0xC6: addr = get_addr(c, AM_ZP);  write8(c, addr, shape_dec(c, read8(c, addr))); break;
        case 0xD6: addr = get_addr(c, AM_ZPX); write8(c, addr, shape_dec(c, read8(c, addr))); break;
        case 0xCE: addr = get_addr(c, AM_ABS); write8(c, addr, shape_dec(c, read8(c, addr))); break;
        case 0xDE: addr = get_addr(c, AM_ABX); write8(c, addr, shape_dec(c, read8(c, addr))); break;

        /* === INX, INY, DEX, DEY === */
        case 0xE8: c->X = shape_inc(c, c->X); break;
        case 0xC8: c->Y = shape_inc(c, c->Y); break;
        case 0xCA: c->X = shape_dec(c, c->X); break;
        case 0x88: c->Y = shape_dec(c, c->Y); break;

        /* === LDA === */
        case 0xA9: c->A = shape_load(c, get_val(c, AM_IMM)); break;
        case 0xA5: c->A = shape_load(c, get_val(c, AM_ZP)); break;
        case 0xB5: c->A = shape_load(c, get_val(c, AM_ZPX)); break;
        case 0xAD: c->A = shape_load(c, get_val(c, AM_ABS)); break;
        case 0xBD: c->A = shape_load(c, get_val(c, AM_ABX)); break;
        case 0xB9: c->A = shape_load(c, get_val(c, AM_ABY)); break;
        case 0xA1: c->A = shape_load(c, get_val(c, AM_IZX)); break;
        case 0xB1: c->A = shape_load(c, get_val(c, AM_IZY)); break;

        /* === LDX === */
        case 0xA2: c->X = shape_load(c, get_val(c, AM_IMM)); break;
        case 0xA6: c->X = shape_load(c, get_val(c, AM_ZP)); break;
        case 0xB6: c->X = shape_load(c, get_val(c, AM_ZPY)); break;
        case 0xAE: c->X = shape_load(c, get_val(c, AM_ABS)); break;
        case 0xBE: c->X = shape_load(c, get_val(c, AM_ABY)); break;

        /* === LDY === */
        case 0xA0: c->Y = shape_load(c, get_val(c, AM_IMM)); break;
        case 0xA4: c->Y = shape_load(c, get_val(c, AM_ZP)); break;
        case 0xB4: c->Y = shape_load(c, get_val(c, AM_ZPX)); break;
        case 0xAC: c->Y = shape_load(c, get_val(c, AM_ABS)); break;
        case 0xBC: c->Y = shape_load(c, get_val(c, AM_ABX)); break;

        /* === STA === */
        case 0x85: write8(c, get_addr(c, AM_ZP), c->A); break;
        case 0x95: write8(c, get_addr(c, AM_ZPX), c->A); break;
        case 0x8D: write8(c, get_addr(c, AM_ABS), c->A); break;
        case 0x9D: write8(c, get_addr(c, AM_ABX), c->A); break;
        case 0x99: write8(c, get_addr(c, AM_ABY), c->A); break;
        case 0x81: write8(c, get_addr(c, AM_IZX), c->A); break;
        case 0x91: write8(c, get_addr(c, AM_IZY), c->A); break;

        /* === STX === */
        case 0x86: write8(c, get_addr(c, AM_ZP), c->X); break;
        case 0x96: write8(c, get_addr(c, AM_ZPY), c->X); break;
        case 0x8E: write8(c, get_addr(c, AM_ABS), c->X); break;

        /* === STY === */
        case 0x84: write8(c, get_addr(c, AM_ZP), c->Y); break;
        case 0x94: write8(c, get_addr(c, AM_ZPX), c->Y); break;
        case 0x8C: write8(c, get_addr(c, AM_ABS), c->Y); break;

        /* === Transfers === */
        case 0xAA: c->X = shape_transfer(c, c->A); break;  /* TAX */
        case 0x8A: c->A = shape_transfer(c, c->X); break;  /* TXA */
        case 0xA8: c->Y = shape_transfer(c, c->A); break;  /* TAY */
        case 0x98: c->A = shape_transfer(c, c->Y); break;  /* TYA */
        case 0xBA: c->X = shape_transfer(c, c->SP); break; /* TSX */
        case 0x9A: c->SP = c->X; break;                    /* TXS (no flags) */

        /* === Stack === */
        case 0x48: push8(c, c->A); break;                  /* PHA */
        case 0x68: c->A = shape_load(c, pull8(c)); break;  /* PLA */
        case 0x08: push8(c, get_P(c) | 0x10); break;       /* PHP */
        case 0x28: set_P(c, pull8(c)); break;              /* PLP */

        /* === CMP === */
        case 0xC9: shape_cmp(c, c->A, get_val(c, AM_IMM)); break;
        case 0xC5: shape_cmp(c, c->A, get_val(c, AM_ZP)); break;
        case 0xD5: shape_cmp(c, c->A, get_val(c, AM_ZPX)); break;
        case 0xCD: shape_cmp(c, c->A, get_val(c, AM_ABS)); break;
        case 0xDD: shape_cmp(c, c->A, get_val(c, AM_ABX)); break;
        case 0xD9: shape_cmp(c, c->A, get_val(c, AM_ABY)); break;
        case 0xC1: shape_cmp(c, c->A, get_val(c, AM_IZX)); break;
        case 0xD1: shape_cmp(c, c->A, get_val(c, AM_IZY)); break;

        /* === CPX === */
        case 0xE0: shape_cmp(c, c->X, get_val(c, AM_IMM)); break;
        case 0xE4: shape_cmp(c, c->X, get_val(c, AM_ZP)); break;
        case 0xEC: shape_cmp(c, c->X, get_val(c, AM_ABS)); break;

        /* === CPY === */
        case 0xC0: shape_cmp(c, c->Y, get_val(c, AM_IMM)); break;
        case 0xC4: shape_cmp(c, c->Y, get_val(c, AM_ZP)); break;
        case 0xCC: shape_cmp(c, c->Y, get_val(c, AM_ABS)); break;

        /* === BIT === */
        case 0x24: shape_bit(c, c->A, get_val(c, AM_ZP)); break;
        case 0x2C: shape_bit(c, c->A, get_val(c, AM_ABS)); break;

        /* === Branches === */
        case 0x10: offset = (int8_t)read8(c, c->PC++); if (!c->N) c->PC += offset; break; /* BPL */
        case 0x30: offset = (int8_t)read8(c, c->PC++); if (c->N)  c->PC += offset; break; /* BMI */
        case 0x50: offset = (int8_t)read8(c, c->PC++); if (!c->V) c->PC += offset; break; /* BVC */
        case 0x70: offset = (int8_t)read8(c, c->PC++); if (c->V)  c->PC += offset; break; /* BVS */
        case 0x90: offset = (int8_t)read8(c, c->PC++); if (!c->C) c->PC += offset; break; /* BCC */
        case 0xB0: offset = (int8_t)read8(c, c->PC++); if (c->C)  c->PC += offset; break; /* BCS */
        case 0xD0: offset = (int8_t)read8(c, c->PC++); if (!c->Z) c->PC += offset; break; /* BNE */
        case 0xF0: offset = (int8_t)read8(c, c->PC++); if (c->Z)  c->PC += offset; break; /* BEQ */

        /* === JMP === */
        case 0x4C: c->PC = get_addr(c, AM_ABS); break;
        case 0x6C: c->PC = get_addr(c, AM_IND); break;

        /* === JSR/RTS/RTI === */
        case 0x20: addr = read16(c, c->PC); c->PC++; push16(c, c->PC); c->PC = addr; break;
        case 0x60: c->PC = pull16(c) + 1; break;
        case 0x40: set_P(c, pull8(c)); c->PC = pull16(c); break;

        /* === BRK === */
        case 0x00:
            c->PC++;
            push16(c, c->PC);
            push8(c, get_P(c) | 0x10);
            c->I = 1;
            c->PC = read16(c, 0xFFFE);
            break;

        /* === Flag operations === */
        case 0x18: c->C = 0; break; /* CLC */
        case 0x38: c->C = 1; break; /* SEC */
        case 0x58: c->I = 0; break; /* CLI */
        case 0x78: c->I = 1; break; /* SEI */
        case 0xB8: c->V = 0; break; /* CLV */
        case 0xD8: c->D = 0; break; /* CLD */
        case 0xF8: c->D = 1; break; /* SED */

        /* === NOP === */
        case 0xEA: break;

        /* === Illegal/unknown: treat as NOP === */
        default:
            /* Many 6502s have undocumented opcodes; we ignore them */
            break;
    }

    c->cycles++;
}

/* ============================================================================
 * INITIALIZATION AND RUNNING
 * ============================================================================ */

static void reset(CPU *c) {
    c->A = c->X = c->Y = 0;
    c->SP = 0xFD;
    c->C = c->Z = c->I = c->D = c->B = c->V = c->N = 0;
    c->U = 1;
    c->PC = read16(c, 0xFFFC);
    c->cycles = 0;
    c->halted = 0;
}

static void run(CPU *c, uint64_t max_cycles) {
    while (c->cycles < max_cycles && !c->halted) {
        step(c);
    }
}

/* ============================================================================
 * TESTS
 * ============================================================================ */

static int test_all(void) {
    CPU cpu;
    int pass = 0, fail = 0;

    #define TEST(name, cond) do { \
        if (cond) { pass++; printf("  [OK] %s\n", name); } \
        else { fail++; printf("  [FAIL] %s\n", name); } \
    } while(0)

    printf("=== Frozen 6502 Complete - Test Suite ===\n\n");

    /* Test 1: LDA immediate */
    memset(&cpu, 0, sizeof(cpu));
    cpu.mem[0x0000] = 0xA9; /* LDA #$42 */
    cpu.mem[0x0001] = 0x42;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("LDA #$42", cpu.A == 0x42 && !cpu.Z && !cpu.N);

    /* Test 2: ADC */
    memset(&cpu, 0, sizeof(cpu));
    cpu.A = 0x10;
    cpu.C = 0;
    cpu.mem[0x0000] = 0x69; /* ADC #$20 */
    cpu.mem[0x0001] = 0x20;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("ADC #$20 (0x10+0x20=0x30)", cpu.A == 0x30 && !cpu.C);

    /* Test 3: ADC with carry in */
    memset(&cpu, 0, sizeof(cpu));
    cpu.A = 0x10;
    cpu.C = 1;
    cpu.mem[0x0000] = 0x69;
    cpu.mem[0x0001] = 0x20;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("ADC #$20 with C=1 (0x10+0x20+1=0x31)", cpu.A == 0x31);

    /* Test 4: ADC overflow */
    memset(&cpu, 0, sizeof(cpu));
    cpu.A = 0xFF;
    cpu.C = 0;
    cpu.mem[0x0000] = 0x69;
    cpu.mem[0x0001] = 0x01;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("ADC overflow (0xFF+0x01=0x00, C=1)", cpu.A == 0x00 && cpu.C && cpu.Z);

    /* Test 5: SBC */
    memset(&cpu, 0, sizeof(cpu));
    cpu.A = 0x50;
    cpu.C = 1;
    cpu.mem[0x0000] = 0xE9;
    cpu.mem[0x0001] = 0x10;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("SBC #$10 (0x50-0x10=0x40)", cpu.A == 0x40 && cpu.C);

    /* Test 6: AND */
    memset(&cpu, 0, sizeof(cpu));
    cpu.A = 0xFF;
    cpu.mem[0x0000] = 0x29;
    cpu.mem[0x0001] = 0x0F;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("AND #$0F (0xFF & 0x0F = 0x0F)", cpu.A == 0x0F);

    /* Test 7: ORA */
    memset(&cpu, 0, sizeof(cpu));
    cpu.A = 0xF0;
    cpu.mem[0x0000] = 0x09;
    cpu.mem[0x0001] = 0x0F;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("ORA #$0F (0xF0 | 0x0F = 0xFF)", cpu.A == 0xFF && cpu.N);

    /* Test 8: EOR */
    memset(&cpu, 0, sizeof(cpu));
    cpu.A = 0x55;
    cpu.mem[0x0000] = 0x49;
    cpu.mem[0x0001] = 0xFF;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("EOR #$FF (0x55 ^ 0xFF = 0xAA)", cpu.A == 0xAA && cpu.N);

    /* Test 9: ASL */
    memset(&cpu, 0, sizeof(cpu));
    cpu.A = 0x40;
    cpu.mem[0x0000] = 0x0A;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("ASL A (0x40 << 1 = 0x80, C=0)", cpu.A == 0x80 && !cpu.C && cpu.N);

    /* Test 10: ASL with carry out */
    memset(&cpu, 0, sizeof(cpu));
    cpu.A = 0x80;
    cpu.mem[0x0000] = 0x0A;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("ASL A (0x80 << 1 = 0x00, C=1)", cpu.A == 0x00 && cpu.C && cpu.Z);

    /* Test 11: LSR */
    memset(&cpu, 0, sizeof(cpu));
    cpu.A = 0x02;
    cpu.mem[0x0000] = 0x4A;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("LSR A (0x02 >> 1 = 0x01)", cpu.A == 0x01 && !cpu.C);

    /* Test 12: ROL */
    memset(&cpu, 0, sizeof(cpu));
    cpu.A = 0x80;
    cpu.C = 1;
    cpu.mem[0x0000] = 0x2A;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("ROL A (0x80, C=1 -> 0x01, C=1)", cpu.A == 0x01 && cpu.C);

    /* Test 13: INX */
    memset(&cpu, 0, sizeof(cpu));
    cpu.X = 0xFE;
    cpu.mem[0x0000] = 0xE8;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("INX (0xFE -> 0xFF)", cpu.X == 0xFF && cpu.N);

    /* Test 14: DEX wrap */
    memset(&cpu, 0, sizeof(cpu));
    cpu.X = 0x00;
    cpu.mem[0x0000] = 0xCA;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("DEX wrap (0x00 -> 0xFF)", cpu.X == 0xFF && cpu.N);

    /* Test 15: BEQ taken */
    memset(&cpu, 0, sizeof(cpu));
    cpu.Z = 1;
    cpu.mem[0x0000] = 0xF0;
    cpu.mem[0x0001] = 0x05;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("BEQ +5 (taken, PC=0x07)", cpu.PC == 0x0007);

    /* Test 16: BEQ not taken */
    memset(&cpu, 0, sizeof(cpu));
    cpu.Z = 0;
    cpu.mem[0x0000] = 0xF0;
    cpu.mem[0x0001] = 0x05;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("BEQ +5 (not taken, PC=0x02)", cpu.PC == 0x0002);

    /* Test 17: JMP absolute */
    memset(&cpu, 0, sizeof(cpu));
    cpu.mem[0x0000] = 0x4C;
    cpu.mem[0x0001] = 0x00;
    cpu.mem[0x0002] = 0x80;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("JMP $8000", cpu.PC == 0x8000);

    /* Test 18: JSR/RTS */
    memset(&cpu, 0, sizeof(cpu));
    cpu.SP = 0xFF;
    cpu.mem[0x0000] = 0x20; /* JSR $0010 */
    cpu.mem[0x0001] = 0x10;
    cpu.mem[0x0002] = 0x00;
    cpu.mem[0x0010] = 0x60; /* RTS */
    cpu.PC = 0x0000;
    step(&cpu); /* JSR */
    TEST("JSR $0010 (PC=0x0010, SP=0xFD)", cpu.PC == 0x0010 && cpu.SP == 0xFD);
    step(&cpu); /* RTS */
    TEST("RTS (PC=0x0003)", cpu.PC == 0x0003);

    /* Test 19: PHA/PLA */
    memset(&cpu, 0, sizeof(cpu));
    cpu.SP = 0xFF;
    cpu.A = 0x42;
    cpu.mem[0x0000] = 0x48; /* PHA */
    cpu.mem[0x0001] = 0xA9; /* LDA #$00 */
    cpu.mem[0x0002] = 0x00;
    cpu.mem[0x0003] = 0x68; /* PLA */
    cpu.PC = 0x0000;
    step(&cpu);
    step(&cpu);
    step(&cpu);
    TEST("PHA/PLA round-trip", cpu.A == 0x42);

    /* Test 20: CMP sets flags correctly */
    memset(&cpu, 0, sizeof(cpu));
    cpu.A = 0x50;
    cpu.mem[0x0000] = 0xC9;
    cpu.mem[0x0001] = 0x30;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("CMP #$30 (0x50 > 0x30, C=1 Z=0)", cpu.C && !cpu.Z);

    /* Test 21: Zero page addressing */
    memset(&cpu, 0, sizeof(cpu));
    cpu.mem[0x0010] = 0x77;
    cpu.mem[0x0000] = 0xA5; /* LDA $10 */
    cpu.mem[0x0001] = 0x10;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("LDA $10 (zero page)", cpu.A == 0x77);

    /* Test 22: Zero page,X addressing */
    memset(&cpu, 0, sizeof(cpu));
    cpu.X = 0x05;
    cpu.mem[0x0015] = 0x88;
    cpu.mem[0x0000] = 0xB5; /* LDA $10,X */
    cpu.mem[0x0001] = 0x10;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("LDA $10,X (X=5, addr=$15)", cpu.A == 0x88);

    /* Test 23: Absolute addressing */
    memset(&cpu, 0, sizeof(cpu));
    cpu.mem[0x1234] = 0x99;
    cpu.mem[0x0000] = 0xAD; /* LDA $1234 */
    cpu.mem[0x0001] = 0x34;
    cpu.mem[0x0002] = 0x12;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("LDA $1234 (absolute)", cpu.A == 0x99);

    /* Test 24: Indirect,X addressing */
    memset(&cpu, 0, sizeof(cpu));
    cpu.X = 0x04;
    cpu.mem[0x0024] = 0x00; /* Pointer lo */
    cpu.mem[0x0025] = 0x20; /* Pointer hi -> $2000 */
    cpu.mem[0x2000] = 0xAB;
    cpu.mem[0x0000] = 0xA1; /* LDA ($20,X) */
    cpu.mem[0x0001] = 0x20;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("LDA ($20,X) (indirect,X)", cpu.A == 0xAB);

    /* Test 25: Indirect,Y addressing */
    memset(&cpu, 0, sizeof(cpu));
    cpu.Y = 0x10;
    cpu.mem[0x0020] = 0x00; /* Pointer lo */
    cpu.mem[0x0021] = 0x30; /* Pointer hi -> $3000 */
    cpu.mem[0x3010] = 0xCD; /* $3000 + Y */
    cpu.mem[0x0000] = 0xB1; /* LDA ($20),Y */
    cpu.mem[0x0001] = 0x20;
    cpu.PC = 0x0000;
    step(&cpu);
    TEST("LDA ($20),Y (indirect,Y)", cpu.A == 0xCD);

    printf("\n=== Results: %d passed, %d failed ===\n", pass, fail);
    printf("\n16 frozen shapes. All addressing modes. Complete 6502.\n");
    printf("\"Computation is geometry.\"\n\n");

    return fail == 0 ? 0 : 1;
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(int argc, char *argv[]) {
    CPU cpu;
    memset(&cpu, 0, sizeof(cpu));

    if (argc < 2) {
        /* Run built-in tests */
        return test_all();
    }

    /* Load binary file */
    FILE *f = fopen(argv[1], "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", argv[1]);
        return 1;
    }

    /* Default load address */
    uint16_t load_addr = 0x0200;
    if (argc >= 3) {
        load_addr = (uint16_t)strtol(argv[2], NULL, 16);
    }

    size_t n = fread(&cpu.mem[load_addr], 1, 65536 - load_addr, f);
    fclose(f);
    printf("Loaded %zu bytes at $%04X\n", n, load_addr);

    /* Set reset vector if not already set */
    if (cpu.mem[0xFFFC] == 0 && cpu.mem[0xFFFD] == 0) {
        cpu.mem[0xFFFC] = load_addr & 0xFF;
        cpu.mem[0xFFFD] = (load_addr >> 8) & 0xFF;
    }

    reset(&cpu);
    printf("PC=$%04X  Running...\n", cpu.PC);

    /* Run until BRK or max cycles */
    run(&cpu, 1000000);

    printf("\nFinal state:\n");
    printf("  A=$%02X  X=$%02X  Y=$%02X  SP=$%02X  PC=$%04X\n",
           cpu.A, cpu.X, cpu.Y, cpu.SP, cpu.PC);
    printf("  N=%d V=%d B=%d D=%d I=%d Z=%d C=%d\n",
           cpu.N, cpu.V, cpu.B, cpu.D, cpu.I, cpu.Z, cpu.C);
    printf("  Cycles: %llu\n", (unsigned long long)cpu.cycles);

    return 0;
}
