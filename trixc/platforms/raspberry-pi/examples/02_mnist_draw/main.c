/*
 * TRIXC Pi - MNIST Draw
 *
 * Draw digits on the touchscreen and watch them get classified in real-time!
 *
 * This example shows:
 * - Touch input for drawing
 * - Real-time neural network inference
 * - Probability visualization
 * - The magic of 6.5KB doing digit recognition
 *
 * Compile:
 *   gcc -o mnist_draw main.c ../../src/trixc_pi.c \
 *       -I../../include $(sdl2-config --cflags --libs) -lm
 *
 * Run:
 *   ./mnist_draw
 *
 * Controls:
 *   - Touch/drag: Draw on canvas
 *   - Press 'c': Clear canvas
 *   - Press 'q' or Escape: Quit
 *
 * ============================================================================
 */

#include "trixc_pi.h"
#include "../../models/mnist_7x7.c"

#include <stdio.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * CANVAS SETTINGS
 * ============================================================================ */

#define CANVAS_DISPLAY_SIZE 280  /* Size on screen (pixels) */
#define CANVAS_GRID_SIZE    7    /* Internal grid (7x7) */
#define CELL_SIZE           (CANVAS_DISPLAY_SIZE / CANVAS_GRID_SIZE)

#define CANVAS_X 50   /* Canvas position on screen */
#define CANVAS_Y 80

/* ============================================================================
 * DRAWING STATE
 * ============================================================================ */

static float canvas[CANVAS_GRID_SIZE * CANVAS_GRID_SIZE] = {0};
static int last_draw_x = -1;
static int last_draw_y = -1;

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

/* Clear the canvas */
static void clear_canvas(void) {
    memset(canvas, 0, sizeof(canvas));
    last_draw_x = -1;
    last_draw_y = -1;
}

/* Draw a point on the canvas with some brush size */
static void draw_point(int gx, int gy, float intensity) {
    if (gx < 0 || gx >= CANVAS_GRID_SIZE || gy < 0 || gy >= CANVAS_GRID_SIZE) {
        return;
    }

    /* Set the main pixel */
    int idx = gy * CANVAS_GRID_SIZE + gx;
    canvas[idx] = fmaxf(canvas[idx], intensity);

    /* Light up neighbors slightly for smoother strokes */
    float neighbor_intensity = intensity * 0.3f;
    if (gx > 0) canvas[idx - 1] = fmaxf(canvas[idx - 1], neighbor_intensity);
    if (gx < CANVAS_GRID_SIZE - 1) canvas[idx + 1] = fmaxf(canvas[idx + 1], neighbor_intensity);
    if (gy > 0) canvas[idx - CANVAS_GRID_SIZE] = fmaxf(canvas[idx - CANVAS_GRID_SIZE], neighbor_intensity);
    if (gy < CANVAS_GRID_SIZE - 1) canvas[idx + CANVAS_GRID_SIZE] = fmaxf(canvas[idx + CANVAS_GRID_SIZE], neighbor_intensity);
}

/* Draw a line between two points (for smooth strokes) */
static void draw_line(int x0, int y0, int x1, int y1, float intensity) {
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;

    while (1) {
        draw_point(x0, y0, intensity);

        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y0 += sy;
        }
    }
}

/* Convert screen coordinates to canvas grid coordinates */
static bool screen_to_canvas(int sx, int sy, int* gx, int* gy) {
    int cx = sx - CANVAS_X;
    int cy = sy - CANVAS_Y;

    if (cx < 0 || cx >= CANVAS_DISPLAY_SIZE ||
        cy < 0 || cy >= CANVAS_DISPLAY_SIZE) {
        return false;
    }

    *gx = cx / CELL_SIZE;
    *gy = cy / CELL_SIZE;
    return true;
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(int argc, char** argv) {
    /* Initialize display */
    if (trixc_init("TRIXC Pi - MNIST Draw", 800, 480) != 0) {
        fprintf(stderr, "Failed to initialize TRIXC Pi\n");
        return 1;
    }

    trixc_timer_t timer;
    trixc_timer_init(&timer);

    float output[10] = {0};
    int predicted = -1;
    bool drawing = false;

    /* Digit labels */
    const char* labels[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

    /* Main loop */
    while (trixc_running()) {
        /* Handle events */
        trixc_event_t event;
        while (trixc_poll(&event)) {
            if (event.type == TRIXC_EVENT_TOUCH_DOWN) {
                drawing = true;
                int gx, gy;
                if (screen_to_canvas(event.x, event.y, &gx, &gy)) {
                    draw_point(gx, gy, 1.0f);
                    last_draw_x = gx;
                    last_draw_y = gy;
                }
            }
            else if (event.type == TRIXC_EVENT_TOUCH_UP) {
                drawing = false;
                last_draw_x = -1;
                last_draw_y = -1;
            }
            else if (event.type == TRIXC_EVENT_TOUCH_MOVE && drawing) {
                int gx, gy;
                if (screen_to_canvas(event.x, event.y, &gx, &gy)) {
                    if (last_draw_x >= 0 && last_draw_y >= 0) {
                        draw_line(last_draw_x, last_draw_y, gx, gy, 1.0f);
                    } else {
                        draw_point(gx, gy, 1.0f);
                    }
                    last_draw_x = gx;
                    last_draw_y = gy;
                }
            }
            else if (event.type == TRIXC_EVENT_KEY_DOWN) {
                if (event.key == 'c' || event.key == 'C') {
                    clear_canvas();
                }
                else if (event.key == 'q' || event.key == 'Q') {
                    goto cleanup;
                }
            }
        }

        /* Run inference */
        trixc_timer_start(&timer);
        mnist_forward_softmax(canvas, output);
        trixc_timer_stop(&timer);

        /* Find prediction */
        predicted = 0;
        for (int i = 1; i < 10; i++) {
            if (output[i] > output[predicted]) {
                predicted = i;
            }
        }

        /* ================================================================
         * DRAW
         * ================================================================ */

        trixc_clear(TRIXC_BG_DARK);

        /* Title */
        trixc_text_scaled(220, 15, "TRIXC Pi - MNIST Draw", TRIXC_WHITE, 3);
        trixc_text(270, 55, "Draw a digit! Press 'c' to clear.", TRIXC_GRAY_60);

        /* ----------------------------------------------------------------
         * Canvas
         * ---------------------------------------------------------------- */

        /* Canvas background */
        trixc_rect(CANVAS_X - 2, CANVAS_Y - 2,
                   CANVAS_DISPLAY_SIZE + 4, CANVAS_DISPLAY_SIZE + 4,
                   TRIXC_GRAY_30);
        trixc_rect(CANVAS_X, CANVAS_Y, CANVAS_DISPLAY_SIZE, CANVAS_DISPLAY_SIZE,
                   TRIXC_BLACK);

        /* Draw canvas cells */
        for (int gy = 0; gy < CANVAS_GRID_SIZE; gy++) {
            for (int gx = 0; gx < CANVAS_GRID_SIZE; gx++) {
                float v = canvas[gy * CANVAS_GRID_SIZE + gx];
                if (v > 0.01f) {
                    int brightness = (int)(v * 255);
                    if (brightness > 255) brightness = 255;
                    uint32_t color = TRIXC_RGB(brightness, brightness, brightness);

                    trixc_rect(CANVAS_X + gx * CELL_SIZE,
                               CANVAS_Y + gy * CELL_SIZE,
                               CELL_SIZE - 1, CELL_SIZE - 1, color);
                }
            }
        }

        /* Grid lines (subtle) */
        for (int i = 1; i < CANVAS_GRID_SIZE; i++) {
            trixc_line(CANVAS_X + i * CELL_SIZE, CANVAS_Y,
                       CANVAS_X + i * CELL_SIZE, CANVAS_Y + CANVAS_DISPLAY_SIZE,
                       TRIXC_GRAY_20);
            trixc_line(CANVAS_X, CANVAS_Y + i * CELL_SIZE,
                       CANVAS_X + CANVAS_DISPLAY_SIZE, CANVAS_Y + i * CELL_SIZE,
                       TRIXC_GRAY_20);
        }

        /* Canvas label */
        trixc_text(CANVAS_X, CANVAS_Y + CANVAS_DISPLAY_SIZE + 10,
                   "7x7 input (49 pixels)", TRIXC_GRAY_60);

        /* ----------------------------------------------------------------
         * Prediction Display
         * ---------------------------------------------------------------- */

        int pred_x = 380;
        int pred_y = CANVAS_Y;

        trixc_text_scaled(pred_x, pred_y, "Prediction:", TRIXC_WHITE, 2);

        /* Big prediction digit */
        if (predicted >= 0) {
            char digit_str[2] = {(char)('0' + predicted), '\0'};

            /* Glow effect based on confidence */
            float conf = output[predicted];
            int glow = (int)(conf * 100);
            if (glow > 80) {
                trixc_rect(pred_x + 60, pred_y + 25, 80, 80,
                           TRIXC_RGB(0, glow, glow / 2));
            }

            trixc_text_scaled(pred_x + 80, pred_y + 35, digit_str,
                              conf > 0.5f ? TRIXC_GREEN : TRIXC_YELLOW, 8);

            /* Confidence */
            char conf_str[32];
            snprintf(conf_str, sizeof(conf_str), "%.1f%% confidence", conf * 100);
            trixc_text(pred_x + 50, pred_y + 120, conf_str,
                       conf > 0.5f ? TRIXC_GREEN : TRIXC_YELLOW);
        } else {
            trixc_text_scaled(pred_x + 80, pred_y + 40, "?", TRIXC_GRAY_50, 8);
        }

        /* ----------------------------------------------------------------
         * Probability Bars
         * ---------------------------------------------------------------- */

        int bar_x = 380;
        int bar_y = pred_y + 160;
        int bar_width = 350;
        int bar_height = 20;

        trixc_text(bar_x, bar_y - 20, "Class Probabilities:", TRIXC_WHITE);

        for (int i = 0; i < 10; i++) {
            int by = bar_y + i * (bar_height + 5);

            /* Background */
            trixc_rect(bar_x + 25, by, bar_width - 25, bar_height, TRIXC_GRAY_20);

            /* Fill */
            int fill_width = (int)(output[i] * (bar_width - 25));
            uint32_t bar_color = (i == predicted) ? TRIXC_GREEN : TRIXC_CYAN;
            if (fill_width > 0) {
                trixc_rect(bar_x + 25, by, fill_width, bar_height, bar_color);
            }

            /* Label */
            trixc_text(bar_x + 5, by + 6, labels[i],
                       (i == predicted) ? TRIXC_WHITE : TRIXC_GRAY_70);

            /* Percentage */
            if (output[i] > 0.01f) {
                char pct_str[16];
                snprintf(pct_str, sizeof(pct_str), "%.0f%%", output[i] * 100);
                trixc_text(bar_x + bar_width + 10, by + 6, pct_str, TRIXC_GRAY_70);
            }
        }

        /* ----------------------------------------------------------------
         * Stats and Info
         * ---------------------------------------------------------------- */

        /* Stats box */
        trixc_stats(CANVAS_X, 400, timer.elapsed_ms, mnist_model_size(), trixc_fps());

        /* Model info */
        trixc_text(220, 410, "Model: 49 -> 32 -> 10", TRIXC_GRAY_60);
        trixc_text(220, 425, "Params: 1,930", TRIXC_GRAY_60);
        trixc_text(220, 440, "Size: 6.5 KB", TRIXC_GRAY_60);

        /* Footer */
        trixc_line(0, 460, 800, 460, TRIXC_GRAY_30);
        trixc_text(50, 465, "TRIXC Pi v1.0", TRIXC_GRAY_50);
        trixc_text(200, 465, "|", TRIXC_GRAY_30);
        trixc_text(220, 465, "'c' = clear    'q' = quit", TRIXC_GRAY_50);
        trixc_text(470, 465, "|", TRIXC_GRAY_30);
        trixc_text(490, 465, "\"It's all in the reflexes.\"", TRIXC_GRAY_50);

        /* Present frame */
        trixc_present();
    }

cleanup:
    trixc_shutdown();

    /* Print final stats */
    printf("\n");
    printf("TRIXC Pi - MNIST Draw\n");
    printf("=====================\n");
    printf("Inference timing:\n");
    printf("  Min:  %.4f ms\n", timer.min_ms);
    printf("  Max:  %.4f ms\n", timer.max_ms);
    printf("  Avg:  %.4f ms\n", timer.avg_ms);
    printf("  Runs: %d\n", timer.count);
    printf("\n");
    printf("Model size: %zu bytes\n", mnist_model_size());
    printf("\n");
    printf("\"It's all in the reflexes.\"\n");

    return 0;
}

/*
 * ============================================================================
 * WHAT YOU JUST SAW
 * ============================================================================
 *
 * This example demonstrates real-time neural network inference:
 *
 * 1. TOUCH INPUT
 *    Draw anywhere on the 7x7 canvas. The drawing is interpolated
 *    for smooth strokes even on the low-resolution grid.
 *
 * 2. REAL-TIME INFERENCE
 *    Every frame, the canvas is fed through the neural network.
 *    On a Pi 4, this takes about 0.1ms - completely imperceptible.
 *
 * 3. PROBABILITY VISUALIZATION
 *    See not just the prediction, but the confidence for all 10 digits.
 *    This helps you understand when the network is "unsure."
 *
 * 4. SIZE PERSPECTIVE
 *    - This model: 6.5 KB
 *    - PyTorch's MNIST example: ~1 GB (with dependencies)
 *    - Same task. Very different approach.
 *
 * Tips for better recognition:
 *    - Draw digits that fill most of the canvas
 *    - Try to center them
 *    - Thick strokes work better than thin ones
 *    - The model was trained on centered digits
 *
 * ============================================================================
 */
