/*
 * TRIXC Example 04: Neural Network Activations
 *
 * From logic gates to neural networks. Same principle: frozen shapes.
 *
 * Compile: gcc -o activations 04_activations.c -lm
 * Run:     ./activations
 *
 * What you'll learn:
 * - Neural network activations are just math formulas
 * - ReLU, Sigmoid, Tanh, GELU - all frozen shapes
 * - There's nothing magical about "deep learning"
 */

#include <stdio.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * FROZEN ACTIVATION SHAPES
 *
 * These are the exact same formulas used in PyTorch, TensorFlow, etc.
 * But instead of being computed by a 2GB runtime, they're just... functions.
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * ReLU (Rectified Linear Unit)
 *
 * The simplest activation: "if negative, return 0; otherwise, return input"
 *
 * ReLU(x) = max(0, x)
 *
 * Why it works: Creates non-linearity while being trivially cheap to compute.
 * Most modern neural networks use ReLU or its variants.
 */
float frozen_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

/*
 * Sigmoid
 *
 * Squashes any number into the range (0, 1).
 * Originally the go-to activation; now mainly used for binary classification.
 *
 * Sigmoid(x) = 1 / (1 + e^(-x))
 */
float frozen_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/*
 * Tanh (Hyperbolic Tangent)
 *
 * Squashes any number into the range (-1, 1).
 * Like sigmoid, but centered at 0.
 *
 * Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 *
 * Or equivalently: Tanh(x) = 2 * Sigmoid(2x) - 1
 */
float frozen_tanh(float x) {
    return tanhf(x);
}

/*
 * GELU (Gaussian Error Linear Unit)
 *
 * The darling of modern transformers (GPT, BERT, etc.).
 * Smooth approximation of ReLU with better gradient properties.
 *
 * GELU(x) = x * Φ(x)  where Φ is the standard normal CDF
 *
 * Fast approximation: GELU(x) ≈ x * Sigmoid(1.702 * x)
 */
float frozen_gelu(float x) {
    return x * frozen_sigmoid(1.702f * x);
}

/*
 * SiLU / Swish
 *
 * Self-gated activation. Discovered by neural architecture search!
 *
 * SiLU(x) = x * Sigmoid(x)
 */
float frozen_silu(float x) {
    return x * frozen_sigmoid(x);
}

/*
 * Leaky ReLU
 *
 * ReLU that doesn't completely zero out negatives.
 * Helps with the "dying ReLU" problem.
 *
 * LeakyReLU(x) = x if x > 0, else 0.01 * x
 */
float frozen_leaky_relu(float x, float alpha) {
    return x > 0.0f ? x : alpha * x;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * DEMO
 * ═══════════════════════════════════════════════════════════════════════════ */

void print_header(const char* name, const char* formula) {
    printf("\n═══ %s ═══\n", name);
    printf("Formula: %s\n\n", formula);
}

int main() {
    printf("╔════════════════════════════════════════════╗\n");
    printf("║  TRIXC Example 04: Neural Activations      ║\n");
    printf("║  The building blocks of deep learning      ║\n");
    printf("╚════════════════════════════════════════════╝\n");

    float test_values[] = {-3.0f, -2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};
    int n = sizeof(test_values) / sizeof(test_values[0]);

    /* ReLU */
    print_header("ReLU", "max(0, x)");
    printf("     x    │   ReLU(x)\n");
    printf("──────────┼───────────\n");
    for (int i = 0; i < n; i++) {
        float x = test_values[i];
        printf("  %6.2f  │   %6.3f\n", x, frozen_relu(x));
    }

    /* Sigmoid */
    print_header("Sigmoid", "1 / (1 + e^(-x))");
    printf("     x    │ Sigmoid(x)\n");
    printf("──────────┼───────────\n");
    for (int i = 0; i < n; i++) {
        float x = test_values[i];
        printf("  %6.2f  │   %6.4f\n", x, frozen_sigmoid(x));
    }

    /* Tanh */
    print_header("Tanh", "(e^x - e^(-x)) / (e^x + e^(-x))");
    printf("     x    │   Tanh(x)\n");
    printf("──────────┼───────────\n");
    for (int i = 0; i < n; i++) {
        float x = test_values[i];
        printf("  %6.2f  │   %6.4f\n", x, frozen_tanh(x));
    }

    /* GELU */
    print_header("GELU", "x × Sigmoid(1.702 × x)");
    printf("     x    │   GELU(x)\n");
    printf("──────────┼───────────\n");
    for (int i = 0; i < n; i++) {
        float x = test_values[i];
        printf("  %6.2f  │   %6.4f\n", x, frozen_gelu(x));
    }

    /* SiLU */
    print_header("SiLU (Swish)", "x × Sigmoid(x)");
    printf("     x    │   SiLU(x)\n");
    printf("──────────┼───────────\n");
    for (int i = 0; i < n; i++) {
        float x = test_values[i];
        printf("  %6.2f  │   %6.4f\n", x, frozen_silu(x));
    }

    /* Comparison */
    printf("\n═══ Side-by-Side Comparison ═══\n\n");
    printf("     x    │   ReLU  │ Sigmoid │   Tanh  │   GELU  │   SiLU\n");
    printf("──────────┼─────────┼─────────┼─────────┼─────────┼─────────\n");
    for (int i = 0; i < n; i++) {
        float x = test_values[i];
        printf("  %6.2f  │ %7.3f │ %7.4f │ %7.4f │ %7.4f │ %7.4f\n",
               x,
               frozen_relu(x),
               frozen_sigmoid(x),
               frozen_tanh(x),
               frozen_gelu(x),
               frozen_silu(x));
    }

    /* Key Insight */
    printf("\n═══ The Key Insight ═══\n\n");
    printf("Every activation function in every neural network is one of these\n");
    printf("frozen shapes (or a close variant). There's nothing to learn here.\n\n");

    printf("GPT-4 uses GELU? That's: x × Sigmoid(1.702 × x)\n");
    printf("Your image classifier uses ReLU? That's: max(0, x)\n");
    printf("BERT uses GELU? Same formula. Same math. Forever.\n\n");

    printf("The 'deep' in 'deep learning' just means: apply these simple\n");
    printf("functions many times in a row. That's it.\n\n");

    printf("NEXT: See 05_matmul.c to learn how neural networks combine inputs.\n");

    return 0;
}
