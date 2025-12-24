/*
 * TRIXC Example 01: Hello XOR
 *
 * Your first frozen shape. The simplest possible example.
 *
 * Compile: gcc -o hello_xor 01_hello_xor.c
 * Run:     ./hello_xor
 *
 * What you'll learn:
 * - What a "frozen shape" is
 * - How XOR becomes a polynomial: a + b - 2ab
 * - That math just... works
 */

#include <stdio.h>

/*
 * THE FROZEN XOR SHAPE
 *
 * XOR(a, b) = a + b - 2ab
 *
 * This is not an approximation. It's exact for binary inputs {0, 1}.
 * Let's prove it:
 *
 *   XOR(0, 0) = 0 + 0 - 2*0*0 = 0 ✓
 *   XOR(0, 1) = 0 + 1 - 2*0*1 = 1 ✓
 *   XOR(1, 0) = 1 + 0 - 2*1*0 = 1 ✓
 *   XOR(1, 1) = 1 + 1 - 2*1*1 = 0 ✓
 */
float frozen_xor(float a, float b) {
    return a + b - 2.0f * a * b;
}

int main() {
    printf("╔════════════════════════════════════════════╗\n");
    printf("║  TRIXC Example 01: Hello XOR               ║\n");
    printf("║  Your first frozen shape                   ║\n");
    printf("╚════════════════════════════════════════════╝\n\n");

    printf("The frozen XOR shape: a + b - 2ab\n\n");

    printf("Truth Table:\n");
    printf("┌───┬───┬──────────┐\n");
    printf("│ a │ b │ XOR(a,b) │\n");
    printf("├───┼───┼──────────┤\n");

    for (int a = 0; a <= 1; a++) {
        for (int b = 0; b <= 1; b++) {
            float result = frozen_xor((float)a, (float)b);
            printf("│ %d │ %d │    %.0f     │\n", a, b, result);
        }
    }

    printf("└───┴───┴──────────┘\n\n");

    printf("That's it. XOR is just a polynomial.\n");
    printf("No training. No runtime. Just math.\n\n");

    /* CHALLENGE: Try these other frozen shapes! */
    printf("CHALLENGE: Implement these shapes:\n");
    printf("  AND(a, b) = a * b\n");
    printf("  OR(a, b)  = a + b - a*b\n");
    printf("  NOT(a)    = 1 - a\n");

    return 0;
}
