/*
 * Frozen 6502 - Minimal Standalone (No libc)
 *
 * Build: gcc -Os -s -nostdlib -static frozen_6502_minimal.c -o f6502_tiny
 */

#include <stdint.h>

/* ============================================================================
 * THE 16 FROZEN SHAPES (inlined for size)
 * ============================================================================ */

#define ADD(a, b, cin, res, cout) do { \
    uint16_t s = (uint16_t)(a) + (uint16_t)(b) + (uint16_t)(cin); \
    *(res) = (uint8_t)s; *(cout) = (s >> 8) & 1; } while(0)

#define SUB(a, b, bin, res, bout) do { \
    uint16_t d = (uint16_t)(a) - (uint16_t)(b) - (uint16_t)(1-(bin)); \
    *(res) = (uint8_t)d; *(bout) = (d <= 0xFF) ? 1 : 0; } while(0)

#define AND(a, b) ((a) & (b))
#define OR(a, b)  ((a) | (b))
#define XOR(a, b) ((a) ^ (b))

#define ASL(a, res, cout) do { *(cout) = ((a)>>7)&1; *(res) = (a)<<1; } while(0)
#define LSR(a, res, cout) do { *(cout) = (a)&1; *(res) = (a)>>1; } while(0)
#define ROL(a, cin, res, cout) do { *(cout) = ((a)>>7)&1; *(res) = ((a)<<1)|(cin); } while(0)
#define ROR(a, cin, res, cout) do { *(cout) = (a)&1; *(res) = ((a)>>1)|((cin)<<7); } while(0)

#define INC(a) ((a) + 1)
#define DEC(a) ((a) - 1)
#define TRANSFER(a) (a)

/* ============================================================================
 * ROUTING TABLE - 16 bytes
 * ============================================================================ */

static const uint8_t ROUTE[16] = {0,1,2,3,4,5,6,7,8,9,10,9,10,11,11,12};

/* ============================================================================
 * CPU STATE
 * ============================================================================ */

typedef struct { uint8_t A, X, Y, C, Z, N; } CPU;

#define UPD_NZ(cpu, v) do { (cpu)->Z = ((v)==0); (cpu)->N = ((v)>>7)&1; } while(0)

/* ============================================================================
 * EXECUTE
 * ============================================================================ */

static void exec(CPU *c, uint8_t op, uint8_t m) {
    uint8_t r, co;
    switch (ROUTE[op]) {
        case 0: ADD(c->A, m, c->C, &r, &co); c->A = r; c->C = co; UPD_NZ(c,r); break;
        case 1: SUB(c->A, m, c->C, &r, &co); c->A = r; c->C = co; UPD_NZ(c,r); break;
        case 2: c->A = AND(c->A, m); UPD_NZ(c, c->A); break;
        case 3: c->A = OR(c->A, m);  UPD_NZ(c, c->A); break;
        case 4: c->A = XOR(c->A, m); UPD_NZ(c, c->A); break;
        case 5: ASL(c->A, &r, &co); c->A = r; c->C = co; UPD_NZ(c,r); break;
        case 6: LSR(c->A, &r, &co); c->A = r; c->C = co; UPD_NZ(c,r); break;
        case 7: ROL(c->A, c->C, &r, &co); c->A = r; c->C = co; UPD_NZ(c,r); break;
        case 8: ROR(c->A, c->C, &r, &co); c->A = r; c->C = co; UPD_NZ(c,r); break;
        case 9:  if (op==9||op==11) { if(op==9) { c->X=INC(c->X); UPD_NZ(c,c->X); }
                                       else { c->Y=INC(c->Y); UPD_NZ(c,c->Y); } } break;
        case 10: if (op==10||op==12) { if(op==10) { c->X=DEC(c->X); UPD_NZ(c,c->X); }
                                        else { c->Y=DEC(c->Y); UPD_NZ(c,c->Y); } } break;
        case 11: if (op==13) { c->X=c->A; UPD_NZ(c,c->X); } else { c->A=c->X; UPD_NZ(c,c->A); } break;
        case 12: c->A = m; UPD_NZ(c, c->A); break;
    }
}

/* ============================================================================
 * ENTRY POINT (no libc)
 * ============================================================================ */

void _start(void) {
    CPU cpu = {0,0,0,0,0,0};
    int pass = 1;

    /* Test 1: 42 + 13 = 55 */
    cpu.A = 42; cpu.C = 0;
    exec(&cpu, 0, 13);
    if (cpu.A != 55) pass = 0;

    /* Test 2: 0x55 ^ 0xFF = 0xAA */
    cpu.A = 0x55;
    exec(&cpu, 4, 0xFF);
    if (cpu.A != 0xAA) pass = 0;

    /* Test 3: 0x40 << 1 = 0x80 */
    cpu.A = 0x40;
    exec(&cpu, 5, 0);
    if (cpu.A != 0x80 || cpu.C != 0) pass = 0;

    /* Test 4: 0x80 << 1 = 0x00 with carry */
    cpu.A = 0x80;
    exec(&cpu, 5, 0);
    if (cpu.A != 0x00 || cpu.C != 1) pass = 0;

    /* Test 5: 0xFF & 0x0F = 0x0F */
    cpu.A = 0xFF;
    exec(&cpu, 2, 0x0F);
    if (cpu.A != 0x0F) pass = 0;

    /* Exit with result: 0 = all pass, 1 = fail */
    asm volatile (
        "mov x0, %0\n"
        "mov x8, #93\n"  /* exit syscall on ARM64 */
        "svc #0\n"
        : : "r" (pass ? 0 : 1)
    );
    __builtin_unreachable();
}
