/*
 * TRIXC Pi - Hello XOR
 *
 * Your first TRIXC Pi program!
 *
 * This example shows:
 * - A neural network computing XOR
 * - A frozen shape computing XOR (exactly!)
 * - Touch input to cycle through inputs
 * - Real-time performance statistics
 *
 * It's beautiful and it's educational.
 * Welcome to TRIXC Pi.
 *
 * Compile:
 *   gcc -o hello_xor main.c ../../src/trixc_pi.c \
 *       -I../../include $(sdl2-config --cflags --libs) -lm
 *
 * Run:
 *   ./hello_xor
 *
 * Controls:
 *   - Tap anywhere: Cycle to next input
 *   - Press 'v': Toggle verbose mode (show hidden layer)
 *   - Press 'q' or Escape: Quit
 *
 * ============================================================================
 */

#include "trixc_pi.h"
#include "../../models/xor_mlp.c"

#include <stdio.h>
#include <math.h>

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(int argc, char** argv) {
    /* Initialize display */
    if (trixc_init("TRIXC Pi - Hello XOR", 800, 480) != 0) {
        fprintf(stderr, "Failed to initialize TRIXC Pi\n");
        return 1;
    }

    /* Input combinations */
    const float inputs[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    const int expected[4] = {0, 1, 1, 0};

    int current_input = 0;
    bool verbose_mode = false;
    trixc_timer_t timer;
    trixc_timer_init(&timer);

    /* Animation */
    float pulse = 0.0f;

    /* Main loop */
    while (trixc_running()) {
        /* Handle events */
        trixc_event_t event;
        while (trixc_poll(&event)) {
            if (event.type == TRIXC_EVENT_TOUCH_DOWN) {
                current_input = (current_input + 1) % 4;
            }
            else if (event.type == TRIXC_EVENT_KEY_DOWN) {
                if (event.key == 'v' || event.key == 'V') {
                    verbose_mode = !verbose_mode;
                }
                else if (event.key == 'q' || event.key == 'Q') {
                    goto cleanup;
                }
            }
        }

        /* Get current input */
        float x0 = inputs[current_input][0];
        float x1 = inputs[current_input][1];
        int exp = expected[current_input];

        /* Run inference */
        float hidden[4], pre_relu[4];
        trixc_timer_start(&timer);
        float nn_output = xor_forward_verbose(x0, x1, hidden, pre_relu);
        trixc_timer_stop(&timer);

        /* Compute frozen shape */
        float shape_output = xor_frozen_shape(x0, x1);

        /* Round for comparison */
        int nn_rounded = nn_output > 0.5f ? 1 : 0;

        /* Update animation */
        pulse += 0.05f;
        if (pulse > 2.0f * 3.14159f) pulse -= 2.0f * 3.14159f;
        float glow = 0.5f + 0.5f * sinf(pulse);

        /* ================================================================
         * DRAW
         * ================================================================ */

        trixc_clear(TRIXC_BG_DARK);

        /* Title */
        trixc_text_scaled(200, 20, "TRIXC Pi - Hello XOR", TRIXC_WHITE, 3);
        trixc_text(280, 60, "Tap anywhere to cycle inputs", TRIXC_GRAY_60);

        /* ----------------------------------------------------------------
         * LEFT SIDE: Neural Network
         * ---------------------------------------------------------------- */

        int left_x = 50;
        int box_y = 100;

        trixc_text_scaled(left_x, box_y, "Neural Network", TRIXC_CYAN, 2);

        /* Input box */
        trixc_rect_outline(left_x, box_y + 35, 120, 60, TRIXC_GRAY_50);
        char input_str[32];
        snprintf(input_str, sizeof(input_str), "Input");
        trixc_text(left_x + 35, box_y + 40, input_str, TRIXC_GRAY_70);
        snprintf(input_str, sizeof(input_str), "[%.0f, %.0f]", x0, x1);
        trixc_text_scaled(left_x + 20, box_y + 60, input_str, TRIXC_WHITE, 2);

        /* Arrow */
        trixc_text(left_x + 140, box_y + 55, "->", TRIXC_GRAY_50);

        /* Hidden layer visualization */
        if (verbose_mode) {
            trixc_rect_outline(left_x + 170, box_y + 35, 100, 100, TRIXC_GRAY_50);
            trixc_text(left_x + 180, box_y + 40, "Hidden", TRIXC_GRAY_70);

            for (int h = 0; h < 4; h++) {
                int hy = box_y + 60 + h * 18;

                /* Pre-ReLU value */
                char pre_str[16];
                snprintf(pre_str, sizeof(pre_str), "%.1f", pre_relu[h]);
                trixc_text(left_x + 175, hy, pre_str,
                           pre_relu[h] < 0 ? TRIXC_RED : TRIXC_WHITE);

                /* Arrow through ReLU */
                trixc_text(left_x + 210, hy, "->", TRIXC_YELLOW);

                /* Post-ReLU value */
                char post_str[16];
                snprintf(post_str, sizeof(post_str), "%.1f", hidden[h]);
                trixc_text(left_x + 240, hy, post_str, TRIXC_GREEN);
            }

            /* Arrow to output */
            trixc_text(left_x + 280, box_y + 85, "->", TRIXC_GRAY_50);
        } else {
            /* Simplified view */
            trixc_rect(left_x + 170, box_y + 35, 100, 60, TRIXC_GRAY_20);
            trixc_text(left_x + 185, box_y + 55, "4 Hidden", TRIXC_GRAY_60);
            trixc_text(left_x + 190, box_y + 70, "+ ReLU", TRIXC_GRAY_60);
            trixc_text(left_x + 280, box_y + 55, "->", TRIXC_GRAY_50);
        }

        /* Output box */
        int out_x = verbose_mode ? left_x + 300 : left_x + 300;
        trixc_rect_outline(out_x, box_y + 35, 100, 60, TRIXC_GRAY_50);
        trixc_text(out_x + 25, box_y + 40, "Output", TRIXC_GRAY_70);

        char out_str[16];
        snprintf(out_str, sizeof(out_str), "%.4f", nn_output);
        uint32_t out_color = (nn_rounded == exp) ? TRIXC_GREEN : TRIXC_RED;
        trixc_text_scaled(out_x + 10, box_y + 60, out_str, out_color, 2);

        /* Result */
        char result_str[32];
        snprintf(result_str, sizeof(result_str), "= %d %s",
                 nn_rounded, (nn_rounded == exp) ? "(correct)" : "(wrong!)");
        trixc_text(out_x + 10, box_y + 100,  result_str,
                   (nn_rounded == exp) ? TRIXC_GREEN : TRIXC_RED);

        /* ----------------------------------------------------------------
         * RIGHT SIDE: Frozen Shape
         * ---------------------------------------------------------------- */

        int right_x = 450;

        trixc_text_scaled(right_x, box_y, "Frozen Shape", TRIXC_YELLOW, 2);

        /* The formula */
        trixc_rect(right_x, box_y + 35, 300, 80, TRIXC_GRAY_20);
        trixc_text(right_x + 20, box_y + 45, "XOR(a, b) = a + b - 2ab", TRIXC_YELLOW);

        /* Show the calculation */
        char calc_str[64];
        snprintf(calc_str, sizeof(calc_str),
                 "= %.0f + %.0f - 2*%.0f*%.0f", x0, x1, x0, x1);
        trixc_text(right_x + 20, box_y + 65, calc_str, TRIXC_WHITE);

        snprintf(calc_str, sizeof(calc_str), "= %.0f", shape_output);
        trixc_text_scaled(right_x + 20, box_y + 85, calc_str, TRIXC_GREEN, 2);

        /* Comparison */
        trixc_text(right_x, box_y + 130, "The frozen shape is:", TRIXC_GRAY_70);
        trixc_text_scaled(right_x + 20, box_y + 150, "- Exact (not learned)", TRIXC_WHITE, 1);
        trixc_text_scaled(right_x + 20, box_y + 165, "- 0 parameters", TRIXC_WHITE, 1);
        trixc_text_scaled(right_x + 20, box_y + 180, "- ~20 bytes", TRIXC_WHITE, 1);

        /* ----------------------------------------------------------------
         * BOTTOM: Truth Table and Stats
         * ---------------------------------------------------------------- */

        int table_y = 280;

        trixc_text_scaled(50, table_y, "XOR Truth Table", TRIXC_WHITE, 2);

        /* Table header */
        trixc_text(50, table_y + 35, "  A    B   A^B   NN    Shape", TRIXC_GRAY_70);
        trixc_line(50, table_y + 48, 350, table_y + 48, TRIXC_GRAY_50);

        /* Table rows */
        for (int i = 0; i < 4; i++) {
            int row_y = table_y + 55 + i * 25;

            float a = inputs[i][0];
            float b = inputs[i][1];
            float nn = xor_forward(a, b);
            float shape = xor_frozen_shape(a, b);

            char row_str[64];
            snprintf(row_str, sizeof(row_str),
                     "  %.0f    %.0f    %d    %.2f   %.0f",
                     a, b, expected[i], nn, shape);

            /* Highlight current row */
            if (i == current_input) {
                int highlight_alpha = (int)(100 + 50 * glow);
                trixc_rect(45, row_y - 2, 310, 20,
                           TRIXC_RGB(0, highlight_alpha, highlight_alpha));
            }

            trixc_text(50, row_y, row_str, TRIXC_WHITE);
        }

        /* Statistics box */
        trixc_stats(450, table_y, timer.elapsed_ms, xor_model_size(), trixc_fps());

        /* Memory comparison */
        trixc_text(450, table_y + 80, "Neural Network: 52 bytes", TRIXC_CYAN);
        trixc_text(450, table_y + 100, "Frozen Shape:   20 bytes", TRIXC_YELLOW);
        trixc_text(450, table_y + 120, "PyTorch equiv:  2 GB+", TRIXC_RED);

        /* ----------------------------------------------------------------
         * Footer
         * ---------------------------------------------------------------- */

        trixc_line(0, 430, 800, 430, TRIXC_GRAY_30);

        trixc_text(50, 445, "TRIXC Pi v1.0", TRIXC_GRAY_50);
        trixc_text(200, 445, "|", TRIXC_GRAY_30);
        trixc_text(220, 445, "Press 'v' for verbose mode", TRIXC_GRAY_50);
        trixc_text(470, 445, "|", TRIXC_GRAY_30);
        trixc_text(490, 445, "\"It's all in the reflexes.\"", TRIXC_GRAY_50);

        /* Present frame */
        trixc_present();
    }

cleanup:
    trixc_shutdown();

    /* Print final stats */
    printf("\n");
    printf("TRIXC Pi - Hello XOR\n");
    printf("====================\n");
    printf("Inference timing:\n");
    printf("  Min:  %.4f ms\n", timer.min_ms);
    printf("  Max:  %.4f ms\n", timer.max_ms);
    printf("  Avg:  %.4f ms\n", timer.avg_ms);
    printf("  Runs: %d\n", timer.count);
    printf("\n");
    printf("Model size: %zu bytes\n", xor_model_size());
    printf("\n");
    printf("\"It's all in the reflexes.\"\n");

    return 0;
}

/*
 * ============================================================================
 * WHAT YOU JUST SAW
 * ============================================================================
 *
 * This example demonstrates the core TRIXC philosophy:
 *
 * 1. NEURAL NETWORKS ARE JUST MATH
 *    The XOR network has 13 parameters and ~50 bytes of code.
 *    It runs in microseconds on your Pi.
 *
 * 2. FROZEN SHAPES ARE BETTER (WHEN APPLICABLE)
 *    The polynomial XOR(a,b) = a + b - 2ab is exact, has 0 parameters,
 *    and takes ~20 bytes. No training needed.
 *
 * 3. LEGIBILITY MATTERS
 *    You can see exactly what the network does. No black boxes.
 *    The hidden layer values, the ReLU clipping, the final sum.
 *
 * 4. SIZE MATTERS
 *    52 bytes vs 2 GB. Same result. Different approach.
 *
 * Next steps:
 *   - Try example 02: Draw digits on the touchscreen
 *   - Try example 03: GPIO integration with sensors
 *   - Build your own model: See docs/ADDING_MODELS.md
 *
 * ============================================================================
 */
