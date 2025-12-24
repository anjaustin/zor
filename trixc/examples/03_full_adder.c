/*
 * TRIXC Example 03: Building a Full Adder
 *
 * Compose frozen shapes into a real arithmetic circuit.
 *
 * Compile: gcc -o full_adder 03_full_adder.c
 * Run:     ./full_adder
 *
 * What you'll learn:
 * - How to compose frozen shapes
 * - What a half adder and full adder do
 * - How CPUs actually add numbers
 */

#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * FROZEN SHAPES (from previous examples)
 * ═══════════════════════════════════════════════════════════════════════════ */

float frozen_and(float a, float b) { return a * b; }
float frozen_or(float a, float b) { return a + b - a * b; }
float frozen_xor(float a, float b) { return a + b - 2.0f * a * b; }

/* ═══════════════════════════════════════════════════════════════════════════
 * HALF ADDER
 *
 * Adds two bits, produces sum and carry.
 *
 *   A ──┬──[XOR]──── Sum
 *       │
 *   B ──┴──[AND]──── Carry
 *
 * Truth table:
 *   A B │ Sum Carry
 *   ────┼───────────
 *   0 0 │  0    0
 *   0 1 │  1    0
 *   1 0 │  1    0
 *   1 1 │  0    1     ← 1+1=2, which is "0 carry 1" in binary
 * ═══════════════════════════════════════════════════════════════════════════ */

void half_adder(float a, float b, float* sum, float* carry) {
    *sum = frozen_xor(a, b);
    *carry = frozen_and(a, b);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * FULL ADDER
 *
 * Adds two bits PLUS a carry-in, produces sum and carry-out.
 * This is what lets you chain adders together for multi-bit addition.
 *
 *   A ────┬──[XOR]──┬──[XOR]──── Sum
 *         │         │
 *   B ────┘    Cin ─┤
 *                   │
 *   A ──[AND]──B    └──[AND]──┬──[OR]──── Cout
 *         │                    │
 *         └────────────────────┘
 *
 * The formula:
 *   Sum  = XOR(XOR(A, B), Cin)
 *   Cout = OR(AND(A, B), AND(XOR(A, B), Cin))
 * ═══════════════════════════════════════════════════════════════════════════ */

void full_adder(float a, float b, float cin, float* sum, float* cout) {
    float xor_ab = frozen_xor(a, b);
    *sum = frozen_xor(xor_ab, cin);

    float and_ab = frozen_and(a, b);
    float and_xor_cin = frozen_and(xor_ab, cin);
    *cout = frozen_or(and_ab, and_xor_cin);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4-BIT RIPPLE CARRY ADDER
 *
 * Chain 4 full adders together to add 4-bit numbers (0-15).
 *
 *   A0 B0     A1 B1     A2 B2     A3 B3
 *    │ │       │ │       │ │       │ │
 *    ▼ ▼       ▼ ▼       ▼ ▼       ▼ ▼
 *   [FA0] ──▶ [FA1] ──▶ [FA2] ──▶ [FA3] ──▶ Cout
 *    │         │         │         │
 *    ▼         ▼         ▼         ▼
 *   S0        S1        S2        S3
 * ═══════════════════════════════════════════════════════════════════════════ */

int add_4bit(int a, int b) {
    /* Extract bits (LSB first, like real hardware) */
    float a_bits[4], b_bits[4], s_bits[4];
    for (int i = 0; i < 4; i++) {
        a_bits[i] = (a >> i) & 1 ? 1.0f : 0.0f;
        b_bits[i] = (b >> i) & 1 ? 1.0f : 0.0f;
    }

    /* Chain the full adders */
    float carry = 0.0f;
    for (int i = 0; i < 4; i++) {
        float cout;
        full_adder(a_bits[i], b_bits[i], carry, &s_bits[i], &cout);
        carry = cout;
    }

    /* Convert back to integer */
    int result = 0;
    for (int i = 0; i < 4; i++) {
        if (s_bits[i] > 0.5f) result |= (1 << i);
    }

    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * DEMO
 * ═══════════════════════════════════════════════════════════════════════════ */

int main() {
    printf("╔════════════════════════════════════════════╗\n");
    printf("║  TRIXC Example 03: Full Adder              ║\n");
    printf("║  Building arithmetic from logic            ║\n");
    printf("╚════════════════════════════════════════════╝\n\n");

    /* Half Adder Demo */
    printf("═══ Half Adder ═══\n\n");
    printf("A half adder adds two bits: Sum = XOR(a,b), Carry = AND(a,b)\n\n");
    printf("┌───┬───┬─────┬───────┐\n");
    printf("│ A │ B │ Sum │ Carry │\n");
    printf("├───┼───┼─────┼───────┤\n");
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            float sum, carry;
            half_adder((float)a, (float)b, &sum, &carry);
            printf("│ %d │ %d │  %.0f  │   %.0f   │\n", a, b, sum, carry);
        }
    }
    printf("└───┴───┴─────┴───────┘\n");

    /* Full Adder Demo */
    printf("\n═══ Full Adder ═══\n\n");
    printf("A full adder adds two bits PLUS a carry-in.\n\n");
    printf("┌───┬───┬─────┬─────┬──────┐\n");
    printf("│ A │ B │ Cin │ Sum │ Cout │\n");
    printf("├───┼───┼─────┼─────┼──────┤\n");
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            for (int cin = 0; cin <= 1; cin++) {
                float sum, cout;
                full_adder((float)a, (float)b, (float)cin, &sum, &cout);
                printf("│ %d │ %d │  %d  │  %.0f  │   %.0f  │\n",
                       a, b, cin, sum, cout);
            }
        }
    }
    printf("└───┴───┴─────┴─────┴──────┘\n");

    /* 4-bit Adder Demo */
    printf("\n═══ 4-Bit Ripple Carry Adder ═══\n\n");
    printf("Chain 4 full adders to add numbers 0-15:\n\n");

    int test_cases[][2] = {
        {0, 0}, {1, 1}, {5, 3}, {7, 8}, {9, 6}, {15, 0}, {15, 1}
    };
    int num_tests = sizeof(test_cases) / sizeof(test_cases[0]);

    printf("┌────────┬────────┬────────┬──────────┐\n");
    printf("│   A    │   B    │ Result │ Expected │\n");
    printf("├────────┼────────┼────────┼──────────┤\n");

    for (int i = 0; i < num_tests; i++) {
        int a = test_cases[i][0];
        int b = test_cases[i][1];
        int result = add_4bit(a, b);
        int expected = (a + b) & 0xF;  /* 4-bit result (mod 16) */
        char* status = (result == expected) ? "✓" : "✗";

        printf("│ %2d     │ %2d     │ %2d     │ %2d  %s    │\n",
               a, b, result, expected, status);
    }
    printf("└────────┴────────┴────────┴──────────┘\n");

    printf("\n═══ What's Happening ═══\n\n");
    printf("We built a working 4-bit adder from:\n");
    printf("  • XOR: a + b - 2ab\n");
    printf("  • AND: a × b\n");
    printf("  • OR:  a + b - ab\n\n");
    printf("This is EXACTLY how the 6502 CPU adds numbers.\n");
    printf("No approximation. No learning. Just math.\n\n");

    printf("CHALLENGE: Extend this to 8 bits!\n");
    printf("Hint: Just use 8 full adders instead of 4.\n");

    return 0;
}
