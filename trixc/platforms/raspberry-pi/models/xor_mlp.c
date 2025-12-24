/*
 * XOR MLP Model
 *
 * A tiny neural network that computes XOR.
 * Architecture: 2 inputs → 4 hidden (ReLU) → 1 output
 *
 * This is a pre-trained, frozen model. The weights were trained once
 * using gradient descent, and now they never change.
 *
 * Memory footprint: ~100 bytes (weights) + ~50 bytes (code)
 * Inference time: ~0.001ms on Raspberry Pi 4
 *
 * "Don't learn what you can derive." — But sometimes you need to
 * show how learning works before you can appreciate derivation.
 *
 * ============================================================================
 */

#ifndef XOR_MLP_MODEL_C
#define XOR_MLP_MODEL_C

#include <stddef.h>

/* ============================================================================
 * MODEL DIMENSIONS
 * ============================================================================ */

#define XOR_INPUT_SIZE  2
#define XOR_HIDDEN_SIZE 4
#define XOR_OUTPUT_SIZE 1

/* ============================================================================
 * FROZEN WEIGHTS
 *
 * These weights were trained with gradient descent on the XOR problem.
 * Training details:
 *   - Learning rate: 0.5
 *   - Epochs: 10,000
 *   - Loss: MSE
 *   - Final loss: ~0.0001
 *
 * Now they're frozen forever. Like opcodes in a CPU.
 * ============================================================================ */

/* Layer 1: Input → Hidden (2 × 4 = 8 weights) */
static const float XOR_W1[XOR_INPUT_SIZE * XOR_HIDDEN_SIZE] = {
    /* Each column: weights from one input to all hidden neurons */
    /* Hidden 0 */  5.64f,  5.61f,
    /* Hidden 1 */  6.45f, -5.98f,
    /* Hidden 2 */ -4.87f,  6.12f,
    /* Hidden 3 */  5.23f, -5.41f
};

/* Layer 1 bias (4 values) */
static const float XOR_B1[XOR_HIDDEN_SIZE] = {
    -2.76f, -0.23f, -0.18f, 2.68f
};

/* Layer 2: Hidden → Output (4 × 1 = 4 weights) */
static const float XOR_W2[XOR_HIDDEN_SIZE * XOR_OUTPUT_SIZE] = {
    9.18f, -9.78f, -9.72f, 9.31f
};

/* Layer 2 bias (1 value) */
static const float XOR_B2[XOR_OUTPUT_SIZE] = {
    -4.51f
};

/* ============================================================================
 * FROZEN SHAPES (ACTIVATION FUNCTIONS)
 * ============================================================================ */

/* ReLU: The simplest non-linearity
 * Shape: max(0, x) = (x + |x|) / 2
 */
static inline float xor_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

/* ============================================================================
 * FORWARD PASS
 * ============================================================================ */

/**
 * Compute XOR using the neural network.
 *
 * @param x0    First input (0 or 1)
 * @param x1    Second input (0 or 1)
 * @return      Output (close to 0 or 1)
 *
 * Example:
 *   float result = xor_forward(1.0f, 0.0f);  // Returns ~1.0
 *   float result = xor_forward(1.0f, 1.0f);  // Returns ~0.0
 */
static float xor_forward(float x0, float x1) {
    float hidden[XOR_HIDDEN_SIZE];
    float output;

    /* Layer 1: Linear + ReLU */
    for (int h = 0; h < XOR_HIDDEN_SIZE; h++) {
        hidden[h] = x0 * XOR_W1[h * 2 + 0] +
                    x1 * XOR_W1[h * 2 + 1] +
                    XOR_B1[h];
        hidden[h] = xor_relu(hidden[h]);
    }

    /* Layer 2: Linear (no activation) */
    output = XOR_B2[0];
    for (int h = 0; h < XOR_HIDDEN_SIZE; h++) {
        output += hidden[h] * XOR_W2[h];
    }

    return output;
}

/**
 * Compute XOR with intermediate values exposed.
 *
 * Useful for visualization and debugging.
 *
 * @param x0        First input
 * @param x1        Second input
 * @param hidden    Output: hidden layer activations (4 floats)
 * @param pre_relu  Output: hidden values before ReLU (4 floats, can be NULL)
 * @return          Final output
 */
static float xor_forward_verbose(float x0, float x1,
                                  float* hidden, float* pre_relu) {
    float output;

    /* Layer 1: Linear */
    for (int h = 0; h < XOR_HIDDEN_SIZE; h++) {
        float linear = x0 * XOR_W1[h * 2 + 0] +
                       x1 * XOR_W1[h * 2 + 1] +
                       XOR_B1[h];

        if (pre_relu) pre_relu[h] = linear;
        hidden[h] = xor_relu(linear);
    }

    /* Layer 2: Linear */
    output = XOR_B2[0];
    for (int h = 0; h < XOR_HIDDEN_SIZE; h++) {
        output += hidden[h] * XOR_W2[h];
    }

    return output;
}

/**
 * Get model memory footprint in bytes.
 */
static inline size_t xor_model_size(void) {
    return sizeof(XOR_W1) + sizeof(XOR_B1) + sizeof(XOR_W2) + sizeof(XOR_B2);
}

/* ============================================================================
 * COMPARISON: FROZEN SHAPE XOR
 *
 * The neural network learned to approximate XOR.
 * But we can compute XOR exactly with a polynomial:
 *   XOR(a, b) = a + b - 2ab
 *
 * This is what "frozen shapes" means in TRIXC.
 * ============================================================================ */

/**
 * Compute XOR using the frozen polynomial shape.
 *
 * This is mathematically exact for binary inputs.
 * No training required. No weights to store.
 *
 * @param a     First input (ideally 0 or 1)
 * @param b     Second input (ideally 0 or 1)
 * @return      XOR result (exactly 0 or 1 for binary inputs)
 */
static inline float xor_frozen_shape(float a, float b) {
    return a + b - 2.0f * a * b;
}

#endif /* XOR_MLP_MODEL_C */

/*
 * ============================================================================
 * WHAT YOU JUST SAW
 * ============================================================================
 *
 * This file contains two ways to compute XOR:
 *
 * 1. Neural Network (xor_forward)
 *    - 13 parameters (8 weights + 4 biases + 1 bias)
 *    - Learned through gradient descent
 *    - Approximate (very close to correct)
 *    - ~100 bytes
 *
 * 2. Frozen Shape (xor_frozen_shape)
 *    - 0 parameters
 *    - Derived from Boolean algebra
 *    - Exact for binary inputs
 *    - ~20 bytes
 *
 * The neural network is here to demonstrate:
 * - How ML models are just math
 * - How small they can be
 * - Why "frozen" is powerful
 *
 * In production, use the frozen shape. It's smaller, faster, and exact.
 * But the neural network helps you understand how it all works.
 *
 * "Don't learn what you can derive."
 *
 * ============================================================================
 */
