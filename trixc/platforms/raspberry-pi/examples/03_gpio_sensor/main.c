/*
 * TRIXC Pi - GPIO Sensor Example
 *
 * Use a neural network to control LEDs based on sensor input!
 *
 * This example shows:
 * - Reading GPIO inputs (buttons, sensors)
 * - Running inference on sensor data
 * - Controlling GPIO outputs (LEDs) based on model output
 * - Real-time hardware integration
 *
 * Hardware setup:
 *   - LED on GPIO 17 (BCM numbering)
 *   - Button on GPIO 27 (with pull-up)
 *   - Optional: Temperature sensor on I2C
 *
 * Compile:
 *   gcc -DTRIXC_PI_GPIO -o gpio_sensor main.c ../../src/trixc_pi.c \
 *       -I../../include $(sdl2-config --cflags --libs) -lpigpio -lm
 *
 * Run (requires root for GPIO):
 *   sudo ./gpio_sensor
 *
 * ============================================================================
 */

#include "trixc_pi.h"
#include "../../models/xor_mlp.c"

#include <stdio.h>
#include <stdlib.h>

/* ============================================================================
 * GPIO PIN DEFINITIONS (BCM numbering)
 * ============================================================================ */

#define LED_GREEN   17
#define LED_RED     18
#define BUTTON_A    27
#define BUTTON_B    22

/* ============================================================================
 * SIMULATED MODE (when GPIO not available)
 * ============================================================================ */

#ifndef TRIXC_PI_GPIO
/* Simulate GPIO when not available */
static int sim_button_a = 0;
static int sim_button_b = 0;

int trixc_gpio_init(void) {
    printf("[SIM] GPIO simulated (no pigpio)\n");
    return 0;
}

void trixc_gpio_shutdown(void) {}
void trixc_gpio_mode(int pin, int mode) { (void)pin; (void)mode; }
void trixc_gpio_write(int pin, int value) {
    printf("[SIM] GPIO %d -> %d\n", pin, value);
}
int trixc_gpio_read(int pin) {
    if (pin == BUTTON_A) return sim_button_a;
    if (pin == BUTTON_B) return sim_button_b;
    return 0;
}
#endif

/* ============================================================================
 * MAIN
 * ============================================================================ */

int main(int argc, char** argv) {
    /* Initialize display */
    if (trixc_init("TRIXC Pi - GPIO Sensor", 800, 480) != 0) {
        fprintf(stderr, "Failed to initialize TRIXC Pi\n");
        return 1;
    }

    /* Initialize GPIO */
    if (trixc_gpio_init() != 0) {
        fprintf(stderr, "Warning: GPIO not available (running in simulation mode)\n");
    }

    /* Configure pins */
    trixc_gpio_mode(LED_GREEN, TRIXC_GPIO_OUTPUT);
    trixc_gpio_mode(LED_RED, TRIXC_GPIO_OUTPUT);
    trixc_gpio_mode(BUTTON_A, TRIXC_GPIO_INPUT);
    trixc_gpio_mode(BUTTON_B, TRIXC_GPIO_INPUT);

    trixc_timer_t timer;
    trixc_timer_init(&timer);

    /* Main loop */
    while (trixc_running()) {
        /* Handle events */
        trixc_event_t event;
        while (trixc_poll(&event)) {
            #ifndef TRIXC_PI_GPIO
            /* Simulate buttons with keyboard in sim mode */
            if (event.type == TRIXC_EVENT_KEY_DOWN) {
                if (event.key == 'a' || event.key == 'A') sim_button_a = 1;
                if (event.key == 'b' || event.key == 'B') sim_button_b = 1;
            }
            if (event.type == TRIXC_EVENT_KEY_UP) {
                if (event.key == 'a' || event.key == 'A') sim_button_a = 0;
                if (event.key == 'b' || event.key == 'B') sim_button_b = 0;
            }
            #endif

            if (event.type == TRIXC_EVENT_KEY_DOWN && event.key == 'q') {
                goto cleanup;
            }
        }

        /* Read inputs */
        float input_a = (float)trixc_gpio_read(BUTTON_A);
        float input_b = (float)trixc_gpio_read(BUTTON_B);

        /* Run inference */
        trixc_timer_start(&timer);
        float xor_result = xor_forward(input_a, input_b);
        trixc_timer_stop(&timer);

        /* Control outputs based on inference */
        int led_green = (xor_result > 0.5f) ? 1 : 0;
        int led_red = (xor_result <= 0.5f) ? 1 : 0;

        trixc_gpio_write(LED_GREEN, led_green);
        trixc_gpio_write(LED_RED, led_red);

        /* ================================================================
         * DRAW
         * ================================================================ */

        trixc_clear(TRIXC_BG_DARK);

        /* Title */
        trixc_text_scaled(180, 20, "TRIXC Pi - GPIO Sensor", TRIXC_WHITE, 3);
        trixc_text(280, 60, "Neural network controlling hardware!", TRIXC_GRAY_60);

        /* ----------------------------------------------------------------
         * Schematic Diagram
         * ---------------------------------------------------------------- */

        int diag_x = 50;
        int diag_y = 100;

        /* Input section */
        trixc_text_scaled(diag_x, diag_y, "INPUTS", TRIXC_CYAN, 2);

        /* Button A */
        trixc_rect_outline(diag_x, diag_y + 40, 100, 50,
                           input_a > 0.5f ? TRIXC_GREEN : TRIXC_GRAY_50);
        trixc_text(diag_x + 10, diag_y + 55, "Button A", TRIXC_WHITE);
        trixc_text(diag_x + 10, diag_y + 70,
                   input_a > 0.5f ? "PRESSED" : "released",
                   input_a > 0.5f ? TRIXC_GREEN : TRIXC_GRAY_60);

        /* Button B */
        trixc_rect_outline(diag_x, diag_y + 100, 100, 50,
                           input_b > 0.5f ? TRIXC_GREEN : TRIXC_GRAY_50);
        trixc_text(diag_x + 10, diag_y + 115, "Button B", TRIXC_WHITE);
        trixc_text(diag_x + 10, diag_y + 130,
                   input_b > 0.5f ? "PRESSED" : "released",
                   input_b > 0.5f ? TRIXC_GREEN : TRIXC_GRAY_60);

        /* Arrows to neural network */
        trixc_line(diag_x + 110, diag_y + 65, diag_x + 170, diag_y + 95, TRIXC_GRAY_50);
        trixc_line(diag_x + 110, diag_y + 125, diag_x + 170, diag_y + 95, TRIXC_GRAY_50);

        /* Neural network box */
        int nn_x = diag_x + 180;
        int nn_y = diag_y + 50;

        trixc_rect(nn_x, nn_y, 180, 100, TRIXC_GRAY_20);
        trixc_rect_outline(nn_x, nn_y, 180, 100, TRIXC_YELLOW);
        trixc_text_scaled(nn_x + 10, nn_y + 10, "Neural Net", TRIXC_YELLOW, 2);
        trixc_text(nn_x + 10, nn_y + 40, "XOR(A, B)", TRIXC_WHITE);

        char result_str[32];
        snprintf(result_str, sizeof(result_str), "Output: %.3f", xor_result);
        trixc_text(nn_x + 10, nn_y + 60, result_str, TRIXC_CYAN);

        char decision_str[32];
        snprintf(decision_str, sizeof(decision_str), "= %d", xor_result > 0.5f ? 1 : 0);
        trixc_text_scaled(nn_x + 10, nn_y + 75, decision_str,
                          xor_result > 0.5f ? TRIXC_GREEN : TRIXC_RED, 2);

        /* Arrow to outputs */
        trixc_line(nn_x + 190, nn_y + 50, nn_x + 250, nn_y + 50, TRIXC_GRAY_50);

        /* Output section */
        int out_x = nn_x + 260;

        trixc_text_scaled(out_x, diag_y, "OUTPUTS", TRIXC_CYAN, 2);

        /* Green LED */
        trixc_circle(out_x + 40, diag_y + 65, 25,
                     led_green ? TRIXC_GREEN : TRIXC_GRAY_30);
        trixc_text(out_x, diag_y + 100, "GREEN LED", TRIXC_WHITE);
        trixc_text(out_x + 10, diag_y + 115,
                   led_green ? "ON" : "off",
                   led_green ? TRIXC_GREEN : TRIXC_GRAY_60);

        /* Red LED */
        trixc_circle(out_x + 40, diag_y + 165, 25,
                     led_red ? TRIXC_RED : TRIXC_GRAY_30);
        trixc_text(out_x + 10, diag_y + 200, "RED LED", TRIXC_WHITE);
        trixc_text(out_x + 15, diag_y + 215,
                   led_red ? "ON" : "off",
                   led_red ? TRIXC_RED : TRIXC_GRAY_60);

        /* ----------------------------------------------------------------
         * Truth Table
         * ---------------------------------------------------------------- */

        int table_x = 50;
        int table_y = 300;

        trixc_text_scaled(table_x, table_y, "XOR Truth Table", TRIXC_WHITE, 2);

        trixc_text(table_x, table_y + 35, "A   B   XOR   LED", TRIXC_GRAY_70);
        trixc_line(table_x, table_y + 48, table_x + 200, table_y + 48, TRIXC_GRAY_50);

        const char* rows[] = {
            "0   0    0    RED",
            "0   1    1    GREEN",
            "1   0    1    GREEN",
            "1   1    0    RED"
        };

        int current_row = (int)(input_a * 2 + input_b);

        for (int i = 0; i < 4; i++) {
            int ry = table_y + 55 + i * 22;
            if (i == current_row) {
                trixc_rect(table_x - 5, ry - 2, 210, 20, TRIXC_GRAY_30);
            }
            trixc_text(table_x, ry, rows[i], TRIXC_WHITE);
        }

        /* ----------------------------------------------------------------
         * Instructions
         * ---------------------------------------------------------------- */

        int inst_x = 400;
        int inst_y = 300;

        trixc_text_scaled(inst_x, inst_y, "How It Works", TRIXC_WHITE, 2);

        trixc_text(inst_x, inst_y + 35, "1. Press buttons A and B", TRIXC_GRAY_70);
        trixc_text(inst_x, inst_y + 55, "2. Neural net computes XOR", TRIXC_GRAY_70);
        trixc_text(inst_x, inst_y + 75, "3. LED shows the result", TRIXC_GRAY_70);
        trixc_text(inst_x, inst_y + 95, "", TRIXC_GRAY_70);
        trixc_text(inst_x, inst_y + 115, "A XOR B = 1 -> GREEN LED", TRIXC_GREEN);
        trixc_text(inst_x, inst_y + 135, "A XOR B = 0 -> RED LED", TRIXC_RED);

        #ifndef TRIXC_PI_GPIO
        trixc_text(inst_x, inst_y + 165, "SIM MODE: Press 'A'/'B' keys", TRIXC_YELLOW);
        #endif

        /* Stats */
        trixc_stats(50, 430, timer.elapsed_ms, xor_model_size(), trixc_fps());

        /* Footer */
        trixc_line(0, 460, 800, 460, TRIXC_GRAY_30);
        trixc_text(50, 465, "TRIXC Pi v1.0", TRIXC_GRAY_50);
        trixc_text(200, 465, "|", TRIXC_GRAY_30);
        trixc_text(220, 465, "GPIO 17/18: LEDs | GPIO 27/22: Buttons", TRIXC_GRAY_50);

        /* Present frame */
        trixc_present();
    }

cleanup:
    /* Turn off LEDs */
    trixc_gpio_write(LED_GREEN, 0);
    trixc_gpio_write(LED_RED, 0);

    trixc_gpio_shutdown();
    trixc_shutdown();

    printf("\n");
    printf("TRIXC Pi - GPIO Sensor\n");
    printf("======================\n");
    printf("Neural network controlled hardware!\n");
    printf("Inference time: %.4f ms avg\n", timer.avg_ms);
    printf("\n");
    printf("\"It's all in the reflexes.\"\n");

    return 0;
}

/*
 * ============================================================================
 * HARDWARE WIRING
 * ============================================================================
 *
 * Components needed:
 *   - 2x LED (green and red)
 *   - 2x 220 ohm resistors (for LEDs)
 *   - 2x Push buttons
 *   - 2x 10K ohm resistors (pull-up for buttons)
 *   - Breadboard and jumper wires
 *
 * Connections (BCM numbering):
 *
 *   Raspberry Pi                  Components
 *   ────────────                  ──────────
 *   GPIO 17  ───── 220Ω ─────── LED Green (+)
 *   GPIO 18  ───── 220Ω ─────── LED Red (+)
 *   GND      ────────────────── LED Green (-), LED Red (-)
 *
 *   GPIO 27  ───────┬────────── Button A
 *   3.3V     ─ 10KΩ ┘           (pull-up)
 *   GND      ────────────────── Button A (other side)
 *
 *   GPIO 22  ───────┬────────── Button B
 *   3.3V     ─ 10KΩ ┘           (pull-up)
 *   GND      ────────────────── Button B (other side)
 *
 * ============================================================================
 * WHAT THIS DEMONSTRATES
 * ============================================================================
 *
 * 1. HARDWARE INTEGRATION
 *    Neural network inference controlling physical outputs.
 *    The model decides which LED to light.
 *
 * 2. REAL-TIME RESPONSE
 *    Press button -> inference -> LED change in < 1ms.
 *    Fast enough to feel instantaneous.
 *
 * 3. EMBEDDED ML
 *    No Python runtime. No TensorFlow. Just C and GPIO.
 *    This is how ML should work on microcontrollers.
 *
 * ============================================================================
 */
