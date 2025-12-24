/*
 * TRIXC Example 05: Matrix Multiplication
 *
 * The heart of every neural network. Surprisingly simple.
 *
 * Compile: gcc -o matmul 05_matmul.c -lm
 * Run:     ./matmul
 *
 * What you'll learn:
 * - How matrix multiplication works
 * - Why neural networks use it
 * - How to build a simple "neuron"
 */

#include <stdio.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * MATRIX MULTIPLICATION
 *
 * This is the operation that makes neural networks work.
 * It's just: multiply rows by columns and sum.
 *
 * C[i][j] = sum over k of: A[i][k] × B[k][j]
 *
 * If A is (M × K) and B is (K × N), then C is (M × N).
 * ═══════════════════════════════════════════════════════════════════════════ */

void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * A SINGLE NEURON
 *
 * A neuron computes:
 *   output = activation(sum(inputs × weights) + bias)
 *
 * That "sum(inputs × weights)" is just a dot product - a tiny matmul!
 * ═══════════════════════════════════════════════════════════════════════════ */

float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

float neuron(const float* inputs, const float* weights, float bias, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += inputs[i] * weights[i];
    }
    return relu(sum + bias);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * A LAYER OF NEURONS
 *
 * A neural network layer is just multiple neurons computed together.
 * This is exactly: output = activation(matmul(input, weights) + bias)
 * ═══════════════════════════════════════════════════════════════════════════ */

void layer(const float* input, const float* weights, const float* bias,
           float* output, int input_size, int output_size) {
    /* Matrix multiply: [1 × input_size] × [input_size × output_size] */
    for (int j = 0; j < output_size; j++) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[i * output_size + j];
        }
        output[j] = relu(sum + bias[j]);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * DEMO
 * ═══════════════════════════════════════════════════════════════════════════ */

void print_matrix(const char* name, const float* M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        printf("  [");
        for (int j = 0; j < cols; j++) {
            printf(" %6.2f", M[i * cols + j]);
        }
        printf(" ]\n");
    }
}

int main() {
    printf("╔════════════════════════════════════════════╗\n");
    printf("║  TRIXC Example 05: Matrix Multiplication   ║\n");
    printf("║  The heart of neural networks              ║\n");
    printf("╚════════════════════════════════════════════╝\n\n");

    /* ═══ Part 1: Basic Matrix Multiply ═══ */
    printf("═══ Part 1: Basic Matrix Multiply ═══\n\n");

    /* A is 2×3, B is 3×2, so C will be 2×2 */
    float A[6] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    float B[6] = {
        7.0f,  8.0f,
        9.0f,  10.0f,
        11.0f, 12.0f
    };
    float C[4];

    print_matrix("A (2×3)", A, 2, 3);
    printf("\n");
    print_matrix("B (3×2)", B, 3, 2);
    printf("\n");

    matmul(A, B, C, 2, 2, 3);

    print_matrix("C = A × B (2×2)", C, 2, 2);

    printf("\nHow C[0][0] = 58 is computed:\n");
    printf("  C[0][0] = A[0][0]×B[0][0] + A[0][1]×B[1][0] + A[0][2]×B[2][0]\n");
    printf("         = 1×7 + 2×9 + 3×11\n");
    printf("         = 7 + 18 + 33 = 58 ✓\n");

    /* ═══ Part 2: A Single Neuron ═══ */
    printf("\n═══ Part 2: A Single Neuron ═══\n\n");

    float inputs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float weights[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float bias = 0.5f;

    float output = neuron(inputs, weights, bias, 4);

    printf("Inputs:  [ 1.0,  2.0,  3.0,  4.0 ]\n");
    printf("Weights: [ 0.1,  0.2,  0.3,  0.4 ]\n");
    printf("Bias:    0.5\n\n");
    printf("Computation:\n");
    printf("  sum = 1×0.1 + 2×0.2 + 3×0.3 + 4×0.4\n");
    printf("      = 0.1 + 0.4 + 0.9 + 1.6 = 3.0\n");
    printf("  sum + bias = 3.0 + 0.5 = 3.5\n");
    printf("  ReLU(3.5) = 3.5\n\n");
    printf("Output: %.2f ✓\n", output);

    /* ═══ Part 3: A Layer of Neurons ═══ */
    printf("\n═══ Part 3: A Layer of Neurons ═══\n\n");

    /* Input: 3 values, Output: 2 neurons */
    float layer_input[3] = {1.0f, 2.0f, 3.0f};

    /* Weights: 3 inputs × 2 outputs */
    float layer_weights[6] = {
        0.1f, 0.4f,   /* input 0 → outputs */
        0.2f, 0.5f,   /* input 1 → outputs */
        0.3f, 0.6f    /* input 2 → outputs */
    };

    float layer_bias[2] = {0.0f, 0.0f};
    float layer_output[2];

    layer(layer_input, layer_weights, layer_bias, layer_output, 3, 2);

    printf("Input (3 values): [ 1.0, 2.0, 3.0 ]\n\n");
    printf("Weights (3→2):\n");
    printf("  Input 0 → [ 0.1, 0.4 ]\n");
    printf("  Input 1 → [ 0.2, 0.5 ]\n");
    printf("  Input 2 → [ 0.3, 0.6 ]\n\n");

    printf("Output neuron 0:\n");
    printf("  = 1×0.1 + 2×0.2 + 3×0.3 = 0.1 + 0.4 + 0.9 = 1.4\n");
    printf("  ReLU(1.4) = 1.4 ✓\n\n");

    printf("Output neuron 1:\n");
    printf("  = 1×0.4 + 2×0.5 + 3×0.6 = 0.4 + 1.0 + 1.8 = 3.2\n");
    printf("  ReLU(3.2) = 3.2 ✓\n\n");

    printf("Layer output: [ %.2f, %.2f ]\n", layer_output[0], layer_output[1]);

    /* ═══ The Big Picture ═══ */
    printf("\n═══ The Big Picture ═══\n\n");

    printf("A neural network is just:\n\n");
    printf("  input → [Layer 1] → [Layer 2] → ... → [Layer N] → output\n\n");
    printf("Each layer does:\n");
    printf("  1. Matrix multiply (combine inputs with weights)\n");
    printf("  2. Add bias\n");
    printf("  3. Apply activation function (ReLU, GELU, etc.)\n\n");
    printf("That's the entire secret of 'deep learning'.\n\n");

    printf("GPT-4? Layers of matmul + activation.\n");
    printf("DALL-E? Layers of matmul + activation.\n");
    printf("Your image classifier? Layers of matmul + activation.\n\n");

    printf("The 'weights' are learned during training.\n");
    printf("The 'operations' are frozen shapes that never change.\n\n");

    printf("NEXT: See 06_tiny_mlp.c to build a complete neural network.\n");

    return 0;
}
