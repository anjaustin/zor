/*
 * TRIXC Example 02: All Logic Gates
 *
 * All 7 fundamental logic gates as frozen shapes.
 *
 * Compile: gcc -o logic_gates 02_logic_gates.c
 * Run:     ./logic_gates
 *
 * What you'll learn:
 * - Every logic gate has a polynomial form
 * - De Morgan's laws work with these shapes
 * - You can build anything from these primitives
 */

#include <stdio.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * THE SEVEN FROZEN LOGIC SHAPES
 * ═══════════════════════════════════════════════════════════════════════════ */

/* AND: True when both inputs are true */
float frozen_and(float a, float b) {
    return a * b;
}

/* OR: True when at least one input is true */
float frozen_or(float a, float b) {
    return a + b - a * b;
}

/* XOR: True when exactly one input is true */
float frozen_xor(float a, float b) {
    return a + b - 2.0f * a * b;
}

/* NOT: Inverts the input */
float frozen_not(float a) {
    return 1.0f - a;
}

/* NAND: NOT(AND) - Universal gate! */
float frozen_nand(float a, float b) {
    return 1.0f - a * b;
}

/* NOR: NOT(OR) - Also universal! */
float frozen_nor(float a, float b) {
    return 1.0f - (a + b - a * b);
}

/* XNOR: NOT(XOR) - Equality gate */
float frozen_xnor(float a, float b) {
    return 1.0f - (a + b - 2.0f * a * b);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * DEMO
 * ═══════════════════════════════════════════════════════════════════════════ */

void print_truth_table(const char* name, float (*gate)(float, float)) {
    printf("\n%s Truth Table:\n", name);
    printf("┌───┬───┬────────┐\n");
    printf("│ a │ b │ result │\n");
    printf("├───┼───┼────────┤\n");
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            printf("│ %d │ %d │   %.0f    │\n", a, b, gate((float)a, (float)b));
        }
    }
    printf("└───┴───┴────────┘\n");
}

int main() {
    printf("╔════════════════════════════════════════════╗\n");
    printf("║  TRIXC Example 02: All Logic Gates         ║\n");
    printf("║  Seven frozen shapes for Boolean algebra   ║\n");
    printf("╚════════════════════════════════════════════╝\n");

    printf("\nThe Formulas:\n");
    printf("  AND(a, b)  = a × b\n");
    printf("  OR(a, b)   = a + b - ab\n");
    printf("  XOR(a, b)  = a + b - 2ab\n");
    printf("  NOT(a)     = 1 - a\n");
    printf("  NAND(a, b) = 1 - ab\n");
    printf("  NOR(a, b)  = 1 - (a + b - ab)\n");
    printf("  XNOR(a, b) = 1 - (a + b - 2ab)\n");

    print_truth_table("AND", frozen_and);
    print_truth_table("OR", frozen_or);
    print_truth_table("XOR", frozen_xor);
    print_truth_table("NAND", frozen_nand);
    print_truth_table("NOR", frozen_nor);
    print_truth_table("XNOR", frozen_xnor);

    /* Verify De Morgan's Laws */
    printf("\n═══ Verifying De Morgan's Laws ═══\n\n");

    int passed = 1;
    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            float fa = (float)a, fb = (float)b;

            /* NOT(AND(a,b)) = OR(NOT(a), NOT(b)) */
            float lhs1 = frozen_not(frozen_and(fa, fb));
            float rhs1 = frozen_or(frozen_not(fa), frozen_not(fb));

            /* NOT(OR(a,b)) = AND(NOT(a), NOT(b)) */
            float lhs2 = frozen_not(frozen_or(fa, fb));
            float rhs2 = frozen_and(frozen_not(fa), frozen_not(fb));

            if (lhs1 != rhs1 || lhs2 != rhs2) {
                passed = 0;
                printf("FAILED at a=%d, b=%d\n", a, b);
            }
        }
    }

    if (passed) {
        printf("✓ De Morgan's Law 1: NOT(AND(a,b)) = OR(NOT(a), NOT(b))\n");
        printf("✓ De Morgan's Law 2: NOT(OR(a,b)) = AND(NOT(a), NOT(b))\n");
        printf("\nBoth laws verified for all inputs!\n");
    }

    printf("\n═══ Fun Fact ═══\n\n");
    printf("NAND is a 'universal gate' - you can build ANY logic circuit\n");
    printf("using only NAND gates. Same with NOR.\n\n");
    printf("Your CPU is (mostly) built from these primitives.\n");

    return 0;
}
