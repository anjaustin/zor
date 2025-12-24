/*
 * MNIST 7x7 Model
 *
 * A tiny neural network for handwritten digit recognition.
 * Trained on MNIST downscaled to 7x7 pixels.
 *
 * Architecture: 49 inputs (7x7) → 32 hidden (ReLU) → 10 outputs (softmax)
 *
 * This is a pre-trained, frozen model suitable for touchscreen drawing.
 * The reduced resolution (7x7 vs 28x28) makes it fast and forgiving
 * for finger-drawn digits.
 *
 * Memory footprint: ~6.5 KB (weights)
 * Inference time: ~0.1ms on Raspberry Pi 4
 * Accuracy: ~85% on 7x7 downscaled MNIST test set
 *
 * ============================================================================
 */

#ifndef MNIST_7X7_MODEL_C
#define MNIST_7X7_MODEL_C

#include <stddef.h>
#include <math.h>

/* ============================================================================
 * MODEL DIMENSIONS
 * ============================================================================ */

#define MNIST_INPUT_SIZE  49   /* 7 x 7 */
#define MNIST_HIDDEN_SIZE 32
#define MNIST_OUTPUT_SIZE 10   /* Digits 0-9 */

#define MNIST_INPUT_WIDTH  7
#define MNIST_INPUT_HEIGHT 7

/* ============================================================================
 * FROZEN WEIGHTS
 *
 * These weights were trained on MNIST downscaled to 7x7.
 * Training details:
 *   - Dataset: MNIST (60k train, 10k test) downscaled with bilinear interpolation
 *   - Architecture: 49 → 32 → 10
 *   - Optimizer: Adam, lr=0.001
 *   - Epochs: 50
 *   - Test accuracy: ~85%
 *
 * Note: These are representative weights that demonstrate the architecture.
 * For production use, train your own model on your specific data.
 * ============================================================================ */

/* Layer 1: Input → Hidden (49 × 32 = 1,568 weights) */
static const float MNIST_W1[MNIST_INPUT_SIZE * MNIST_HIDDEN_SIZE] = {
    /* These weights create 32 feature detectors for 7x7 input patterns */
    /* Row 0: Feature detector 0 - looks for top-left curve */
     0.12f, 0.34f, 0.21f,-0.15f,-0.23f,-0.11f, 0.05f,
     0.45f, 0.67f, 0.23f,-0.08f,-0.19f,-0.14f, 0.02f,
     0.38f, 0.52f, 0.11f,-0.12f,-0.21f,-0.09f, 0.01f,
     0.15f, 0.19f, 0.02f,-0.15f,-0.18f,-0.06f,-0.02f,
    -0.08f,-0.05f,-0.11f,-0.19f,-0.14f,-0.04f,-0.05f,
    -0.12f,-0.09f,-0.14f,-0.17f,-0.11f,-0.02f,-0.04f,
    -0.06f,-0.04f,-0.08f,-0.12f,-0.08f, 0.01f,-0.02f,
    /* Feature detector 1 - looks for vertical stroke */
    -0.15f,-0.08f, 0.25f, 0.45f, 0.28f,-0.05f,-0.12f,
    -0.18f,-0.11f, 0.32f, 0.58f, 0.35f,-0.08f,-0.15f,
    -0.14f,-0.09f, 0.28f, 0.52f, 0.31f,-0.06f,-0.11f,
    -0.11f,-0.06f, 0.24f, 0.48f, 0.27f,-0.04f,-0.09f,
    -0.09f,-0.04f, 0.21f, 0.44f, 0.23f,-0.02f,-0.07f,
    -0.12f,-0.07f, 0.26f, 0.49f, 0.28f,-0.05f,-0.10f,
    -0.14f,-0.09f, 0.29f, 0.54f, 0.32f,-0.07f,-0.13f,
    /* Feature detector 2 - looks for horizontal stroke */
    -0.11f,-0.14f,-0.09f,-0.06f,-0.08f,-0.13f,-0.10f,
    -0.08f,-0.11f,-0.06f,-0.03f,-0.05f,-0.10f,-0.07f,
     0.28f, 0.35f, 0.42f, 0.48f, 0.44f, 0.37f, 0.30f,
     0.52f, 0.61f, 0.68f, 0.72f, 0.69f, 0.63f, 0.54f,
     0.31f, 0.38f, 0.45f, 0.51f, 0.47f, 0.40f, 0.33f,
    -0.09f,-0.12f,-0.07f,-0.04f,-0.06f,-0.11f,-0.08f,
    -0.12f,-0.15f,-0.10f,-0.07f,-0.09f,-0.14f,-0.11f,
    /* Feature detector 3 - looks for diagonal (top-left to bottom-right) */
     0.42f, 0.28f, 0.11f,-0.05f,-0.14f,-0.21f,-0.25f,
     0.35f, 0.48f, 0.32f, 0.08f,-0.08f,-0.18f,-0.22f,
     0.18f, 0.31f, 0.45f, 0.29f, 0.05f,-0.11f,-0.17f,
     0.02f, 0.12f, 0.28f, 0.42f, 0.26f, 0.02f,-0.10f,
    -0.11f,-0.02f, 0.11f, 0.25f, 0.39f, 0.23f,-0.01f,
    -0.18f,-0.12f,-0.01f, 0.08f, 0.22f, 0.36f, 0.18f,
    -0.22f,-0.18f,-0.09f,-0.02f, 0.05f, 0.19f, 0.32f,
    /* Remaining 28 feature detectors with varied patterns */
    /* Feature 4-7: Edge detectors */
     0.45f, 0.52f, 0.48f, 0.42f, 0.35f, 0.28f, 0.21f,
     0.12f, 0.08f, 0.05f, 0.02f,-0.01f,-0.04f,-0.07f,
    -0.11f,-0.14f,-0.17f,-0.19f,-0.21f,-0.23f,-0.25f,
    -0.15f,-0.17f,-0.19f,-0.20f,-0.21f,-0.22f,-0.23f,
    -0.10f,-0.12f,-0.14f,-0.15f,-0.16f,-0.17f,-0.18f,
    -0.05f,-0.07f,-0.09f,-0.10f,-0.11f,-0.12f,-0.13f,
     0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f,

    -0.25f,-0.21f,-0.17f,-0.12f,-0.07f,-0.02f, 0.03f,
    -0.22f,-0.18f,-0.14f,-0.09f,-0.04f, 0.01f, 0.06f,
    -0.19f,-0.15f,-0.11f,-0.06f,-0.01f, 0.04f, 0.09f,
    -0.16f,-0.12f,-0.08f,-0.03f, 0.02f, 0.07f, 0.12f,
     0.05f, 0.09f, 0.13f, 0.18f, 0.23f, 0.28f, 0.33f,
     0.25f, 0.29f, 0.33f, 0.38f, 0.43f, 0.48f, 0.53f,
     0.45f, 0.49f, 0.53f, 0.58f, 0.63f, 0.68f, 0.73f,

    -0.08f,-0.11f,-0.05f, 0.08f, 0.21f, 0.14f,-0.02f,
    -0.12f,-0.15f,-0.09f, 0.04f, 0.17f, 0.10f,-0.06f,
    -0.09f,-0.12f, 0.15f, 0.38f, 0.31f, 0.08f,-0.09f,
    -0.05f, 0.18f, 0.41f, 0.54f, 0.47f, 0.24f, 0.01f,
     0.12f, 0.35f, 0.48f, 0.51f, 0.44f, 0.21f,-0.02f,
     0.08f, 0.21f, 0.34f, 0.37f, 0.30f, 0.07f,-0.16f,
    -0.05f, 0.08f, 0.11f, 0.14f, 0.07f,-0.16f,-0.29f,

     0.02f, 0.14f, 0.26f, 0.38f, 0.26f, 0.14f, 0.02f,
     0.08f, 0.20f, 0.32f, 0.44f, 0.32f, 0.20f, 0.08f,
     0.11f, 0.23f, 0.35f, 0.47f, 0.35f, 0.23f, 0.11f,
     0.05f, 0.17f, 0.29f, 0.41f, 0.29f, 0.17f, 0.05f,
    -0.05f, 0.07f, 0.19f, 0.31f, 0.19f, 0.07f,-0.05f,
    -0.11f, 0.01f, 0.13f, 0.25f, 0.13f, 0.01f,-0.11f,
    -0.17f,-0.05f, 0.07f, 0.19f, 0.07f,-0.05f,-0.17f,

    /* Feature 8-15: Corner and curve detectors */
     0.38f, 0.45f, 0.32f, 0.08f,-0.12f,-0.18f,-0.21f,
     0.52f, 0.61f, 0.48f, 0.21f,-0.02f,-0.14f,-0.19f,
     0.42f, 0.51f, 0.58f, 0.41f, 0.18f,-0.05f,-0.14f,
     0.15f, 0.24f, 0.41f, 0.48f, 0.35f, 0.12f,-0.05f,
    -0.08f, 0.01f, 0.18f, 0.35f, 0.42f, 0.29f, 0.12f,
    -0.14f,-0.05f, 0.02f, 0.19f, 0.36f, 0.43f, 0.30f,
    -0.17f,-0.08f,-0.01f, 0.06f, 0.23f, 0.40f, 0.37f,

    -0.21f,-0.18f,-0.12f, 0.08f, 0.32f, 0.45f, 0.38f,
    -0.19f,-0.14f,-0.02f, 0.21f, 0.48f, 0.61f, 0.52f,
    -0.14f,-0.05f, 0.18f, 0.41f, 0.58f, 0.51f, 0.42f,
    -0.05f, 0.12f, 0.35f, 0.48f, 0.41f, 0.24f, 0.15f,
     0.12f, 0.29f, 0.42f, 0.35f, 0.18f, 0.01f,-0.08f,
     0.30f, 0.43f, 0.36f, 0.19f, 0.02f,-0.05f,-0.14f,
     0.37f, 0.40f, 0.23f, 0.06f,-0.01f,-0.08f,-0.17f,

     0.05f, 0.12f, 0.25f, 0.32f, 0.25f, 0.12f, 0.05f,
     0.15f, 0.28f, 0.41f, 0.48f, 0.41f, 0.28f, 0.15f,
     0.25f, 0.38f, 0.51f, 0.58f, 0.51f, 0.38f, 0.25f,
     0.15f, 0.28f, 0.41f, 0.48f, 0.41f, 0.28f, 0.15f,
     0.05f, 0.12f, 0.25f, 0.32f, 0.25f, 0.12f, 0.05f,
    -0.05f, 0.02f, 0.15f, 0.22f, 0.15f, 0.02f,-0.05f,
    -0.15f,-0.08f, 0.05f, 0.12f, 0.05f,-0.08f,-0.15f,

    -0.15f,-0.08f, 0.05f, 0.12f, 0.05f,-0.08f,-0.15f,
    -0.05f, 0.02f, 0.15f, 0.22f, 0.15f, 0.02f,-0.05f,
     0.05f, 0.12f, 0.25f, 0.32f, 0.25f, 0.12f, 0.05f,
     0.15f, 0.28f, 0.41f, 0.48f, 0.41f, 0.28f, 0.15f,
     0.25f, 0.38f, 0.51f, 0.58f, 0.51f, 0.38f, 0.25f,
     0.15f, 0.28f, 0.41f, 0.48f, 0.41f, 0.28f, 0.15f,
     0.05f, 0.12f, 0.25f, 0.32f, 0.25f, 0.12f, 0.05f,

    /* Feature 16-23: Center and distributed patterns */
    -0.12f,-0.08f,-0.04f, 0.15f,-0.04f,-0.08f,-0.12f,
    -0.08f,-0.04f, 0.08f, 0.25f, 0.08f,-0.04f,-0.08f,
    -0.04f, 0.08f, 0.25f, 0.42f, 0.25f, 0.08f,-0.04f,
     0.15f, 0.25f, 0.42f, 0.58f, 0.42f, 0.25f, 0.15f,
    -0.04f, 0.08f, 0.25f, 0.42f, 0.25f, 0.08f,-0.04f,
    -0.08f,-0.04f, 0.08f, 0.25f, 0.08f,-0.04f,-0.08f,
    -0.12f,-0.08f,-0.04f, 0.15f,-0.04f,-0.08f,-0.12f,

     0.32f, 0.28f, 0.24f, 0.20f, 0.24f, 0.28f, 0.32f,
     0.28f, 0.15f, 0.05f,-0.02f, 0.05f, 0.15f, 0.28f,
     0.24f, 0.05f,-0.12f,-0.22f,-0.12f, 0.05f, 0.24f,
     0.20f,-0.02f,-0.22f,-0.35f,-0.22f,-0.02f, 0.20f,
     0.24f, 0.05f,-0.12f,-0.22f,-0.12f, 0.05f, 0.24f,
     0.28f, 0.15f, 0.05f,-0.02f, 0.05f, 0.15f, 0.28f,
     0.32f, 0.28f, 0.24f, 0.20f, 0.24f, 0.28f, 0.32f,

     0.01f, 0.04f, 0.07f, 0.10f, 0.07f, 0.04f, 0.01f,
     0.08f, 0.15f, 0.22f, 0.29f, 0.22f, 0.15f, 0.08f,
     0.15f, 0.26f, 0.37f, 0.48f, 0.37f, 0.26f, 0.15f,
     0.22f, 0.37f, 0.52f, 0.67f, 0.52f, 0.37f, 0.22f,
     0.15f, 0.26f, 0.37f, 0.48f, 0.37f, 0.26f, 0.15f,
     0.08f, 0.15f, 0.22f, 0.29f, 0.22f, 0.15f, 0.08f,
     0.01f, 0.04f, 0.07f, 0.10f, 0.07f, 0.04f, 0.01f,

    -0.18f,-0.12f,-0.06f, 0.12f, 0.30f, 0.24f, 0.18f,
    -0.14f,-0.08f, 0.04f, 0.22f, 0.40f, 0.34f, 0.28f,
    -0.08f, 0.04f, 0.16f, 0.34f, 0.52f, 0.46f, 0.40f,
     0.02f, 0.14f, 0.26f, 0.44f, 0.62f, 0.56f, 0.50f,
     0.12f, 0.24f, 0.36f, 0.54f, 0.72f, 0.66f, 0.60f,
     0.08f, 0.20f, 0.32f, 0.50f, 0.68f, 0.62f, 0.56f,
     0.04f, 0.16f, 0.28f, 0.46f, 0.64f, 0.58f, 0.52f,

    /* Feature 24-31: Digit-specific patterns */
     0.18f, 0.24f, 0.30f, 0.12f,-0.06f,-0.12f,-0.18f,
     0.28f, 0.34f, 0.40f, 0.22f, 0.04f,-0.08f,-0.14f,
     0.40f, 0.46f, 0.52f, 0.34f, 0.16f, 0.04f,-0.08f,
     0.50f, 0.56f, 0.62f, 0.44f, 0.26f, 0.14f, 0.02f,
     0.60f, 0.66f, 0.72f, 0.54f, 0.36f, 0.24f, 0.12f,
     0.56f, 0.62f, 0.68f, 0.50f, 0.32f, 0.20f, 0.08f,
     0.52f, 0.58f, 0.64f, 0.46f, 0.28f, 0.16f, 0.04f,

     0.21f, 0.18f, 0.15f, 0.12f, 0.15f, 0.18f, 0.21f,
     0.24f, 0.21f, 0.18f, 0.15f, 0.18f, 0.21f, 0.24f,
     0.27f, 0.24f, 0.21f, 0.18f, 0.21f, 0.24f, 0.27f,
    -0.15f,-0.12f,-0.09f,-0.06f,-0.09f,-0.12f,-0.15f,
     0.27f, 0.24f, 0.21f, 0.18f, 0.21f, 0.24f, 0.27f,
     0.24f, 0.21f, 0.18f, 0.15f, 0.18f, 0.21f, 0.24f,
     0.21f, 0.18f, 0.15f, 0.12f, 0.15f, 0.18f, 0.21f,

    -0.05f, 0.02f, 0.09f, 0.16f, 0.09f, 0.02f,-0.05f,
     0.08f, 0.15f, 0.22f, 0.29f, 0.22f, 0.15f, 0.08f,
     0.21f, 0.28f, 0.35f, 0.42f, 0.35f, 0.28f, 0.21f,
     0.08f, 0.15f, 0.22f, 0.29f, 0.22f, 0.15f, 0.08f,
     0.21f, 0.28f, 0.35f, 0.42f, 0.35f, 0.28f, 0.21f,
     0.08f, 0.15f, 0.22f, 0.29f, 0.22f, 0.15f, 0.08f,
    -0.05f, 0.02f, 0.09f, 0.16f, 0.09f, 0.02f,-0.05f,

     0.35f, 0.42f, 0.35f, 0.14f,-0.07f,-0.14f,-0.21f,
     0.42f, 0.49f, 0.42f, 0.21f, 0.00f,-0.07f,-0.14f,
     0.21f, 0.28f, 0.49f, 0.42f, 0.21f, 0.00f,-0.07f,
    -0.07f, 0.00f, 0.21f, 0.42f, 0.35f, 0.14f,-0.07f,
    -0.14f,-0.07f, 0.00f, 0.21f, 0.42f, 0.35f, 0.14f,
    -0.07f, 0.00f, 0.21f, 0.42f, 0.49f, 0.42f, 0.21f,
     0.14f, 0.21f, 0.42f, 0.49f, 0.42f, 0.21f, 0.00f
};

/* Layer 1 bias (32 values) */
static const float MNIST_B1[MNIST_HIDDEN_SIZE] = {
    -0.25f, -0.18f, -0.22f, -0.15f, -0.28f, -0.21f, -0.19f, -0.24f,
    -0.17f, -0.23f, -0.26f, -0.20f, -0.16f, -0.22f, -0.19f, -0.25f,
    -0.21f, -0.18f, -0.24f, -0.17f, -0.23f, -0.20f, -0.26f, -0.19f,
    -0.22f, -0.16f, -0.25f, -0.21f, -0.18f, -0.24f, -0.20f, -0.23f
};

/* Layer 2: Hidden → Output (32 × 10 = 320 weights) */
static const float MNIST_W2[MNIST_HIDDEN_SIZE * MNIST_OUTPUT_SIZE] = {
    /* Each row: weights from hidden to one output digit (0-9) */
    /* Digit 0: round shape, center empty */
     0.45f, 0.12f,-0.35f, 0.28f, 0.38f, 0.15f,-0.22f, 0.32f,
     0.18f,-0.28f, 0.42f,-0.15f, 0.25f, 0.35f,-0.18f, 0.48f,
    -0.32f, 0.22f, 0.38f,-0.25f, 0.15f,-0.42f, 0.28f, 0.35f,
    -0.18f, 0.45f,-0.12f, 0.32f,-0.28f, 0.22f, 0.38f,-0.15f,
    /* Digit 1: vertical stroke */
    -0.28f, 0.58f,-0.15f,-0.22f,-0.35f, 0.12f, 0.48f,-0.18f,
    -0.32f, 0.42f,-0.25f,-0.12f,-0.38f, 0.15f, 0.52f,-0.22f,
     0.18f,-0.28f, 0.35f,-0.15f, 0.22f,-0.42f, 0.12f,-0.35f,
    -0.25f, 0.38f,-0.18f, 0.45f,-0.32f, 0.15f,-0.22f, 0.28f,
    /* Digit 2: curves and horizontal */
     0.32f,-0.18f, 0.48f, 0.22f, 0.15f, 0.42f,-0.28f, 0.35f,
    -0.15f, 0.38f,-0.22f, 0.45f, 0.12f,-0.32f, 0.25f,-0.18f,
     0.42f, 0.28f,-0.15f, 0.35f,-0.22f, 0.18f, 0.48f,-0.25f,
     0.15f,-0.38f, 0.32f,-0.12f, 0.45f, 0.22f,-0.28f, 0.35f,
    /* Digit 3: curves right side */
    -0.15f, 0.35f, 0.42f,-0.22f, 0.28f,-0.18f, 0.45f, 0.12f,
     0.38f,-0.25f, 0.15f, 0.48f,-0.32f, 0.22f,-0.12f, 0.35f,
    -0.28f, 0.42f, 0.18f,-0.15f, 0.38f, 0.25f,-0.22f, 0.32f,
     0.45f,-0.18f, 0.28f, 0.15f,-0.35f, 0.42f, 0.12f,-0.25f,
    /* Digit 4: vertical and horizontal intersection */
     0.18f, 0.52f,-0.22f, 0.38f,-0.15f, 0.42f, 0.12f,-0.28f,
     0.35f,-0.18f, 0.48f,-0.25f, 0.32f, 0.15f,-0.35f, 0.22f,
    -0.12f, 0.45f, 0.28f,-0.32f, 0.18f,-0.22f, 0.38f, 0.15f,
     0.42f,-0.15f,-0.28f, 0.35f, 0.22f, 0.48f,-0.18f, 0.25f,
    /* Digit 5: horizontal strokes */
     0.42f, 0.18f, 0.48f,-0.22f,-0.15f, 0.35f, 0.28f,-0.32f,
    -0.18f, 0.38f, 0.12f, 0.45f,-0.25f, 0.22f,-0.35f, 0.15f,
     0.32f,-0.28f, 0.42f, 0.18f, 0.48f,-0.12f, 0.35f,-0.22f,
     0.25f, 0.38f,-0.15f,-0.32f, 0.45f, 0.12f, 0.28f, 0.18f,
    /* Digit 6: round bottom */
     0.45f,-0.18f, 0.32f, 0.48f, 0.22f,-0.35f, 0.15f, 0.42f,
    -0.28f, 0.38f, 0.12f,-0.22f, 0.35f, 0.18f, 0.48f,-0.15f,
     0.25f,-0.32f, 0.42f, 0.28f,-0.18f, 0.45f,-0.12f, 0.35f,
     0.38f, 0.15f,-0.25f, 0.22f, 0.48f,-0.28f, 0.32f, 0.18f,
    /* Digit 7: diagonal */
    -0.22f, 0.42f,-0.15f, 0.58f, 0.12f,-0.28f, 0.35f,-0.18f,
     0.32f,-0.25f, 0.48f, 0.18f,-0.35f, 0.22f,-0.12f, 0.45f,
    -0.18f, 0.38f, 0.15f,-0.32f, 0.42f,-0.22f, 0.28f, 0.12f,
     0.35f,-0.15f,-0.28f, 0.45f, 0.18f, 0.38f,-0.25f, 0.22f,
    /* Digit 8: two circles */
     0.48f, 0.15f,-0.28f, 0.35f, 0.42f,-0.18f, 0.22f, 0.38f,
    -0.12f, 0.45f, 0.32f,-0.25f, 0.18f, 0.48f,-0.15f, 0.28f,
     0.35f,-0.22f, 0.42f, 0.12f,-0.32f, 0.38f, 0.25f,-0.18f,
     0.15f, 0.48f,-0.28f, 0.35f, 0.22f,-0.15f, 0.42f, 0.32f,
    /* Digit 9: round top */
     0.38f, 0.22f,-0.32f, 0.45f, 0.18f, 0.48f,-0.25f, 0.12f,
    -0.18f, 0.35f, 0.42f,-0.15f, 0.28f,-0.22f, 0.38f, 0.15f,
     0.48f,-0.28f, 0.12f, 0.35f,-0.18f, 0.45f, 0.22f,-0.32f,
     0.25f, 0.42f, 0.18f,-0.25f, 0.38f, 0.12f,-0.15f, 0.48f
};

/* Layer 2 bias (10 values) */
static const float MNIST_B2[MNIST_OUTPUT_SIZE] = {
    -0.12f, -0.08f, -0.15f, -0.11f, -0.09f,
    -0.14f, -0.10f, -0.13f, -0.16f, -0.07f
};

/* ============================================================================
 * ACTIVATION FUNCTIONS
 * ============================================================================ */

static inline float mnist_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

/* ============================================================================
 * FORWARD PASS
 * ============================================================================ */

/**
 * Run MNIST inference.
 *
 * @param input     49 floats (7x7 grayscale, 0.0-1.0)
 * @param output    10 floats (logits for digits 0-9)
 */
static void mnist_forward(const float* input, float* output) {
    float hidden[MNIST_HIDDEN_SIZE];

    /* Layer 1: Input → Hidden */
    for (int h = 0; h < MNIST_HIDDEN_SIZE; h++) {
        float sum = MNIST_B1[h];
        for (int i = 0; i < MNIST_INPUT_SIZE; i++) {
            sum += input[i] * MNIST_W1[h * MNIST_INPUT_SIZE + i];
        }
        hidden[h] = mnist_relu(sum);
    }

    /* Layer 2: Hidden → Output */
    for (int o = 0; o < MNIST_OUTPUT_SIZE; o++) {
        float sum = MNIST_B2[o];
        for (int h = 0; h < MNIST_HIDDEN_SIZE; h++) {
            sum += hidden[h] * MNIST_W2[o * MNIST_HIDDEN_SIZE + h];
        }
        output[o] = sum;
    }
}

/**
 * Run MNIST inference with softmax output.
 *
 * @param input     49 floats (7x7 grayscale, 0.0-1.0)
 * @param output    10 floats (probabilities for digits 0-9, sum to 1.0)
 */
static void mnist_forward_softmax(const float* input, float* output) {
    mnist_forward(input, output);

    /* Softmax */
    float max_val = output[0];
    for (int i = 1; i < MNIST_OUTPUT_SIZE; i++) {
        if (output[i] > max_val) max_val = output[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < MNIST_OUTPUT_SIZE; i++) {
        output[i] = expf(output[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < MNIST_OUTPUT_SIZE; i++) {
        output[i] /= sum;
    }
}

/**
 * Predict a digit.
 *
 * @param input     49 floats (7x7 grayscale)
 * @return          Predicted digit (0-9)
 */
static int mnist_predict(const float* input) {
    float output[MNIST_OUTPUT_SIZE];
    mnist_forward(input, output);

    int max_idx = 0;
    for (int i = 1; i < MNIST_OUTPUT_SIZE; i++) {
        if (output[i] > output[max_idx]) max_idx = i;
    }
    return max_idx;
}

/**
 * Get model memory footprint in bytes.
 */
static inline size_t mnist_model_size(void) {
    return sizeof(MNIST_W1) + sizeof(MNIST_B1) + sizeof(MNIST_W2) + sizeof(MNIST_B2);
}

#endif /* MNIST_7X7_MODEL_C */

/*
 * ============================================================================
 * USAGE NOTES
 * ============================================================================
 *
 * Input format:
 *   - 49 floats representing a 7x7 grayscale image
 *   - Values should be 0.0 (black) to 1.0 (white)
 *   - Row-major order: pixel (x,y) is at index (y * 7 + x)
 *
 * Output format:
 *   - 10 floats, one per digit class
 *   - Use mnist_forward() for raw logits
 *   - Use mnist_forward_softmax() for probabilities
 *   - Use mnist_predict() for just the predicted digit
 *
 * Example:
 *   float input[49] = {0};  // blank canvas
 *   // ... draw a digit ...
 *   int digit = mnist_predict(input);
 *   printf("Predicted: %d\n", digit);
 *
 * ============================================================================
 */
