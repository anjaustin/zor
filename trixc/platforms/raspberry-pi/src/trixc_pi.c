/*
 * TRIXC Pi Runtime Implementation
 *
 * A complete SDL2-based runtime for Raspberry Pi.
 * Display, input, timing, visualization - everything you need.
 *
 * Compile with: gcc -c trixc_pi.c -o trixc_pi.o $(sdl2-config --cflags)
 * Link with:    gcc your_app.c trixc_pi.o $(sdl2-config --libs) -lm
 *
 * Or as a unity build:
 *   gcc -DTRIXC_PI_IMPLEMENTATION your_app.c $(sdl2-config --cflags --libs) -lm
 *
 * ============================================================================
 */

#include "trixc_pi.h"

#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>

/* ============================================================================
 * INTERNAL STATE
 * ============================================================================ */

static struct {
    /* SDL state */
    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_Texture* texture;       /* For direct pixel access if needed */

    /* Display info */
    int width;
    int height;
    bool fullscreen;

    /* Runtime state */
    bool initialized;
    bool running;
    int debug_level;

    /* Timing */
    uint64_t start_time_us;
    uint64_t last_frame_time;
    float current_fps;
    int frame_count;
    uint64_t fps_update_time;

    /* Input state */
    bool touch_down;
    int touch_x, touch_y;
    const uint8_t* key_state;

} trixc_state = {0};

/* ============================================================================
 * 8x8 BITMAP FONT
 *
 * Covers ASCII 32 (space) through 126 (~).
 * Each character is 8 bytes, each byte is one row (MSB = left pixel).
 * ============================================================================ */

const uint8_t trixc_font_8x8[95][8] = {
    /* 32: Space */
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
    /* 33: ! */
    {0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x00},
    /* 34: " */
    {0x6C, 0x6C, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00},
    /* 35: # */
    {0x6C, 0x6C, 0xFE, 0x6C, 0xFE, 0x6C, 0x6C, 0x00},
    /* 36: $ */
    {0x18, 0x7E, 0xC0, 0x7C, 0x06, 0xFC, 0x18, 0x00},
    /* 37: % */
    {0x00, 0xC6, 0xCC, 0x18, 0x30, 0x66, 0xC6, 0x00},
    /* 38: & */
    {0x38, 0x6C, 0x38, 0x76, 0xDC, 0xCC, 0x76, 0x00},
    /* 39: ' */
    {0x18, 0x18, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00},
    /* 40: ( */
    {0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00},
    /* 41: ) */
    {0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00},
    /* 42: * */
    {0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00},
    /* 43: + */
    {0x00, 0x18, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00},
    /* 44: , */
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x30},
    /* 45: - */
    {0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00},
    /* 46: . */
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00},
    /* 47: / */
    {0x06, 0x0C, 0x18, 0x30, 0x60, 0xC0, 0x80, 0x00},
    /* 48: 0 */
    {0x7C, 0xCE, 0xDE, 0xF6, 0xE6, 0xC6, 0x7C, 0x00},
    /* 49: 1 */
    {0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00},
    /* 50: 2 */
    {0x7C, 0xC6, 0x06, 0x1C, 0x70, 0xC6, 0xFE, 0x00},
    /* 51: 3 */
    {0x7C, 0xC6, 0x06, 0x3C, 0x06, 0xC6, 0x7C, 0x00},
    /* 52: 4 */
    {0x1C, 0x3C, 0x6C, 0xCC, 0xFE, 0x0C, 0x0C, 0x00},
    /* 53: 5 */
    {0xFE, 0xC0, 0xFC, 0x06, 0x06, 0xC6, 0x7C, 0x00},
    /* 54: 6 */
    {0x3C, 0x60, 0xC0, 0xFC, 0xC6, 0xC6, 0x7C, 0x00},
    /* 55: 7 */
    {0xFE, 0xC6, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00},
    /* 56: 8 */
    {0x7C, 0xC6, 0xC6, 0x7C, 0xC6, 0xC6, 0x7C, 0x00},
    /* 57: 9 */
    {0x7C, 0xC6, 0xC6, 0x7E, 0x06, 0x0C, 0x78, 0x00},
    /* 58: : */
    {0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x00},
    /* 59: ; */
    {0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x30},
    /* 60: < */
    {0x0C, 0x18, 0x30, 0x60, 0x30, 0x18, 0x0C, 0x00},
    /* 61: = */
    {0x00, 0x00, 0x7E, 0x00, 0x7E, 0x00, 0x00, 0x00},
    /* 62: > */
    {0x30, 0x18, 0x0C, 0x06, 0x0C, 0x18, 0x30, 0x00},
    /* 63: ? */
    {0x7C, 0xC6, 0x0C, 0x18, 0x18, 0x00, 0x18, 0x00},
    /* 64: @ */
    {0x7C, 0xC6, 0xDE, 0xDE, 0xDE, 0xC0, 0x78, 0x00},
    /* 65: A */
    {0x38, 0x6C, 0xC6, 0xC6, 0xFE, 0xC6, 0xC6, 0x00},
    /* 66: B */
    {0xFC, 0xC6, 0xC6, 0xFC, 0xC6, 0xC6, 0xFC, 0x00},
    /* 67: C */
    {0x7C, 0xC6, 0xC0, 0xC0, 0xC0, 0xC6, 0x7C, 0x00},
    /* 68: D */
    {0xF8, 0xCC, 0xC6, 0xC6, 0xC6, 0xCC, 0xF8, 0x00},
    /* 69: E */
    {0xFE, 0xC0, 0xC0, 0xFC, 0xC0, 0xC0, 0xFE, 0x00},
    /* 70: F */
    {0xFE, 0xC0, 0xC0, 0xFC, 0xC0, 0xC0, 0xC0, 0x00},
    /* 71: G */
    {0x7C, 0xC6, 0xC0, 0xCE, 0xC6, 0xC6, 0x7C, 0x00},
    /* 72: H */
    {0xC6, 0xC6, 0xC6, 0xFE, 0xC6, 0xC6, 0xC6, 0x00},
    /* 73: I */
    {0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00},
    /* 74: J */
    {0x06, 0x06, 0x06, 0x06, 0xC6, 0xC6, 0x7C, 0x00},
    /* 75: K */
    {0xC6, 0xCC, 0xD8, 0xF0, 0xD8, 0xCC, 0xC6, 0x00},
    /* 76: L */
    {0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xC0, 0xFE, 0x00},
    /* 77: M */
    {0xC6, 0xEE, 0xFE, 0xD6, 0xC6, 0xC6, 0xC6, 0x00},
    /* 78: N */
    {0xC6, 0xE6, 0xF6, 0xDE, 0xCE, 0xC6, 0xC6, 0x00},
    /* 79: O */
    {0x7C, 0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x7C, 0x00},
    /* 80: P */
    {0xFC, 0xC6, 0xC6, 0xFC, 0xC0, 0xC0, 0xC0, 0x00},
    /* 81: Q */
    {0x7C, 0xC6, 0xC6, 0xC6, 0xD6, 0xDE, 0x7C, 0x06},
    /* 82: R */
    {0xFC, 0xC6, 0xC6, 0xFC, 0xD8, 0xCC, 0xC6, 0x00},
    /* 83: S */
    {0x7C, 0xC6, 0xC0, 0x7C, 0x06, 0xC6, 0x7C, 0x00},
    /* 84: T */
    {0xFE, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00},
    /* 85: U */
    {0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x7C, 0x00},
    /* 86: V */
    {0xC6, 0xC6, 0xC6, 0xC6, 0x6C, 0x38, 0x10, 0x00},
    /* 87: W */
    {0xC6, 0xC6, 0xC6, 0xD6, 0xFE, 0xEE, 0xC6, 0x00},
    /* 88: X */
    {0xC6, 0xC6, 0x6C, 0x38, 0x6C, 0xC6, 0xC6, 0x00},
    /* 89: Y */
    {0xC6, 0xC6, 0x6C, 0x38, 0x18, 0x18, 0x18, 0x00},
    /* 90: Z */
    {0xFE, 0x06, 0x0C, 0x18, 0x30, 0x60, 0xFE, 0x00},
    /* 91: [ */
    {0x3C, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3C, 0x00},
    /* 92: \ */
    {0xC0, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x02, 0x00},
    /* 93: ] */
    {0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x3C, 0x00},
    /* 94: ^ */
    {0x10, 0x38, 0x6C, 0xC6, 0x00, 0x00, 0x00, 0x00},
    /* 95: _ */
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF},
    /* 96: ` */
    {0x30, 0x18, 0x0C, 0x00, 0x00, 0x00, 0x00, 0x00},
    /* 97: a */
    {0x00, 0x00, 0x7C, 0x06, 0x7E, 0xC6, 0x7E, 0x00},
    /* 98: b */
    {0xC0, 0xC0, 0xFC, 0xC6, 0xC6, 0xC6, 0xFC, 0x00},
    /* 99: c */
    {0x00, 0x00, 0x7C, 0xC6, 0xC0, 0xC6, 0x7C, 0x00},
    /* 100: d */
    {0x06, 0x06, 0x7E, 0xC6, 0xC6, 0xC6, 0x7E, 0x00},
    /* 101: e */
    {0x00, 0x00, 0x7C, 0xC6, 0xFE, 0xC0, 0x7C, 0x00},
    /* 102: f */
    {0x1C, 0x36, 0x30, 0x7C, 0x30, 0x30, 0x30, 0x00},
    /* 103: g */
    {0x00, 0x00, 0x7E, 0xC6, 0xC6, 0x7E, 0x06, 0x7C},
    /* 104: h */
    {0xC0, 0xC0, 0xFC, 0xC6, 0xC6, 0xC6, 0xC6, 0x00},
    /* 105: i */
    {0x18, 0x00, 0x38, 0x18, 0x18, 0x18, 0x3C, 0x00},
    /* 106: j */
    {0x06, 0x00, 0x06, 0x06, 0x06, 0x06, 0xC6, 0x7C},
    /* 107: k */
    {0xC0, 0xC0, 0xCC, 0xD8, 0xF0, 0xD8, 0xCC, 0x00},
    /* 108: l */
    {0x38, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00},
    /* 109: m */
    {0x00, 0x00, 0xEC, 0xFE, 0xD6, 0xC6, 0xC6, 0x00},
    /* 110: n */
    {0x00, 0x00, 0xFC, 0xC6, 0xC6, 0xC6, 0xC6, 0x00},
    /* 111: o */
    {0x00, 0x00, 0x7C, 0xC6, 0xC6, 0xC6, 0x7C, 0x00},
    /* 112: p */
    {0x00, 0x00, 0xFC, 0xC6, 0xC6, 0xFC, 0xC0, 0xC0},
    /* 113: q */
    {0x00, 0x00, 0x7E, 0xC6, 0xC6, 0x7E, 0x06, 0x06},
    /* 114: r */
    {0x00, 0x00, 0xDC, 0xE6, 0xC0, 0xC0, 0xC0, 0x00},
    /* 115: s */
    {0x00, 0x00, 0x7E, 0xC0, 0x7C, 0x06, 0xFC, 0x00},
    /* 116: t */
    {0x30, 0x30, 0x7C, 0x30, 0x30, 0x36, 0x1C, 0x00},
    /* 117: u */
    {0x00, 0x00, 0xC6, 0xC6, 0xC6, 0xC6, 0x7E, 0x00},
    /* 118: v */
    {0x00, 0x00, 0xC6, 0xC6, 0x6C, 0x38, 0x10, 0x00},
    /* 119: w */
    {0x00, 0x00, 0xC6, 0xC6, 0xD6, 0xFE, 0x6C, 0x00},
    /* 120: x */
    {0x00, 0x00, 0xC6, 0x6C, 0x38, 0x6C, 0xC6, 0x00},
    /* 121: y */
    {0x00, 0x00, 0xC6, 0xC6, 0xC6, 0x7E, 0x06, 0x7C},
    /* 122: z */
    {0x00, 0x00, 0xFE, 0x0C, 0x38, 0x60, 0xFE, 0x00},
    /* 123: { */
    {0x0E, 0x18, 0x18, 0x70, 0x18, 0x18, 0x0E, 0x00},
    /* 124: | */
    {0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00},
    /* 125: } */
    {0x70, 0x18, 0x18, 0x0E, 0x18, 0x18, 0x70, 0x00},
    /* 126: ~ */
    {0x76, 0xDC, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
};

/* ============================================================================
 * HELPER: Get current time in microseconds
 * ============================================================================ */

static uint64_t get_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

/* ============================================================================
 * CORE INITIALIZATION
 * ============================================================================ */

int trixc_init(const char* title, int width, int height) {
    if (trixc_state.initialized) {
        trixc_log(TRIXC_LOG_WARN, "TRIXC Pi already initialized\n");
        return 0;
    }

    /* Initialize SDL */
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) < 0) {
        trixc_log(TRIXC_LOG_ERROR, "SDL_Init failed: %s\n", SDL_GetError());
        return -1;
    }

    /* Auto-detect resolution if not specified */
    if (width == 0 || height == 0) {
        SDL_DisplayMode dm;
        if (SDL_GetCurrentDisplayMode(0, &dm) == 0) {
            width = dm.w;
            height = dm.h;
            trixc_state.fullscreen = true;
        } else {
            width = 800;
            height = 480;
        }
    }

    /* Create window */
    Uint32 flags = SDL_WINDOW_SHOWN;
    if (trixc_state.fullscreen) {
        flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
    }

    trixc_state.window = SDL_CreateWindow(
        title ? title : "TRIXC Pi",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        width, height, flags
    );

    if (!trixc_state.window) {
        trixc_log(TRIXC_LOG_ERROR, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return -1;
    }

    /* Create renderer */
    trixc_state.renderer = SDL_CreateRenderer(
        trixc_state.window, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
    );

    if (!trixc_state.renderer) {
        trixc_log(TRIXC_LOG_ERROR, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(trixc_state.window);
        SDL_Quit();
        return -1;
    }

    /* Get actual window size (may differ from requested in fullscreen) */
    SDL_GetWindowSize(trixc_state.window, &trixc_state.width, &trixc_state.height);

    /* Initialize timing */
    trixc_state.start_time_us = get_time_us();
    trixc_state.last_frame_time = trixc_state.start_time_us;
    trixc_state.fps_update_time = trixc_state.start_time_us;
    trixc_state.frame_count = 0;
    trixc_state.current_fps = 0.0f;

    /* Initialize input */
    trixc_state.key_state = SDL_GetKeyboardState(NULL);
    trixc_state.touch_down = false;
    trixc_state.touch_x = 0;
    trixc_state.touch_y = 0;

    /* Ready! */
    trixc_state.initialized = true;
    trixc_state.running = true;
    trixc_state.debug_level = TRIXC_LOG_INFO;

    trixc_log(TRIXC_LOG_INFO, "TRIXC Pi initialized: %dx%d%s\n",
              trixc_state.width, trixc_state.height,
              trixc_state.fullscreen ? " (fullscreen)" : "");

    return 0;
}

void trixc_shutdown(void) {
    if (!trixc_state.initialized) return;

    if (trixc_state.renderer) {
        SDL_DestroyRenderer(trixc_state.renderer);
        trixc_state.renderer = NULL;
    }

    if (trixc_state.window) {
        SDL_DestroyWindow(trixc_state.window);
        trixc_state.window = NULL;
    }

    SDL_Quit();

    trixc_state.initialized = false;
    trixc_state.running = false;

    trixc_log(TRIXC_LOG_INFO, "TRIXC Pi shutdown complete\n");
}

bool trixc_running(void) {
    return trixc_state.running;
}

int trixc_width(void) {
    return trixc_state.width;
}

int trixc_height(void) {
    return trixc_state.height;
}

/* ============================================================================
 * DISPLAY - BASIC
 * ============================================================================ */

void trixc_clear(uint32_t color) {
    uint8_t r = (color >> 16) & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = color & 0xFF;

    SDL_SetRenderDrawColor(trixc_state.renderer, r, g, b, 255);
    SDL_RenderClear(trixc_state.renderer);
}

void trixc_present(void) {
    SDL_RenderPresent(trixc_state.renderer);

    /* Update FPS counter */
    trixc_state.frame_count++;
    uint64_t now = get_time_us();

    if (now - trixc_state.fps_update_time >= 1000000) {
        trixc_state.current_fps = (float)trixc_state.frame_count * 1000000.0f /
                                  (float)(now - trixc_state.fps_update_time);
        trixc_state.frame_count = 0;
        trixc_state.fps_update_time = now;
    }

    trixc_state.last_frame_time = now;

    /* Process events to keep window responsive */
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_QUIT) {
            trixc_state.running = false;
        } else if (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE) {
            trixc_state.running = false;
        } else if (event.type == SDL_MOUSEBUTTONDOWN || event.type == SDL_FINGERDOWN) {
            trixc_state.touch_down = true;
            if (event.type == SDL_MOUSEBUTTONDOWN) {
                trixc_state.touch_x = event.button.x;
                trixc_state.touch_y = event.button.y;
            } else {
                trixc_state.touch_x = (int)(event.tfinger.x * trixc_state.width);
                trixc_state.touch_y = (int)(event.tfinger.y * trixc_state.height);
            }
        } else if (event.type == SDL_MOUSEBUTTONUP || event.type == SDL_FINGERUP) {
            trixc_state.touch_down = false;
        } else if (event.type == SDL_MOUSEMOTION) {
            trixc_state.touch_x = event.motion.x;
            trixc_state.touch_y = event.motion.y;
        } else if (event.type == SDL_FINGERMOTION) {
            trixc_state.touch_x = (int)(event.tfinger.x * trixc_state.width);
            trixc_state.touch_y = (int)(event.tfinger.y * trixc_state.height);
        }
    }
}

void trixc_pixel(int x, int y, uint32_t color) {
    uint8_t r = (color >> 16) & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = color & 0xFF;

    SDL_SetRenderDrawColor(trixc_state.renderer, r, g, b, 255);
    SDL_RenderDrawPoint(trixc_state.renderer, x, y);
}

void trixc_rect(int x, int y, int w, int h, uint32_t color) {
    uint8_t r = (color >> 16) & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = color & 0xFF;

    SDL_SetRenderDrawColor(trixc_state.renderer, r, g, b, 255);
    SDL_Rect rect = {x, y, w, h};
    SDL_RenderFillRect(trixc_state.renderer, &rect);
}

void trixc_rect_outline(int x, int y, int w, int h, uint32_t color) {
    uint8_t r = (color >> 16) & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = color & 0xFF;

    SDL_SetRenderDrawColor(trixc_state.renderer, r, g, b, 255);
    SDL_Rect rect = {x, y, w, h};
    SDL_RenderDrawRect(trixc_state.renderer, &rect);
}

void trixc_line(int x1, int y1, int x2, int y2, uint32_t color) {
    uint8_t r = (color >> 16) & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = color & 0xFF;

    SDL_SetRenderDrawColor(trixc_state.renderer, r, g, b, 255);
    SDL_RenderDrawLine(trixc_state.renderer, x1, y1, x2, y2);
}

void trixc_circle(int cx, int cy, int radius, uint32_t color) {
    uint8_t r = (color >> 16) & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = color & 0xFF;

    SDL_SetRenderDrawColor(trixc_state.renderer, r, g, b, 255);

    /* Midpoint circle algorithm with filled scanlines */
    int x = radius;
    int y = 0;
    int err = 0;

    while (x >= y) {
        /* Draw horizontal lines to fill the circle */
        SDL_RenderDrawLine(trixc_state.renderer, cx - x, cy + y, cx + x, cy + y);
        SDL_RenderDrawLine(trixc_state.renderer, cx - x, cy - y, cx + x, cy - y);
        SDL_RenderDrawLine(trixc_state.renderer, cx - y, cy + x, cx + y, cy + x);
        SDL_RenderDrawLine(trixc_state.renderer, cx - y, cy - x, cx + y, cy - x);

        y++;
        err += 1 + 2 * y;
        if (2 * (err - x) + 1 > 0) {
            x--;
            err += 1 - 2 * x;
        }
    }
}

void trixc_circle_outline(int cx, int cy, int radius, uint32_t color) {
    uint8_t r = (color >> 16) & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = color & 0xFF;

    SDL_SetRenderDrawColor(trixc_state.renderer, r, g, b, 255);

    int x = radius;
    int y = 0;
    int err = 0;

    while (x >= y) {
        SDL_RenderDrawPoint(trixc_state.renderer, cx + x, cy + y);
        SDL_RenderDrawPoint(trixc_state.renderer, cx + y, cy + x);
        SDL_RenderDrawPoint(trixc_state.renderer, cx - y, cy + x);
        SDL_RenderDrawPoint(trixc_state.renderer, cx - x, cy + y);
        SDL_RenderDrawPoint(trixc_state.renderer, cx - x, cy - y);
        SDL_RenderDrawPoint(trixc_state.renderer, cx - y, cy - x);
        SDL_RenderDrawPoint(trixc_state.renderer, cx + y, cy - x);
        SDL_RenderDrawPoint(trixc_state.renderer, cx + x, cy - y);

        y++;
        err += 1 + 2 * y;
        if (2 * (err - x) + 1 > 0) {
            x--;
            err += 1 - 2 * x;
        }
    }
}

/* ============================================================================
 * DISPLAY - TEXT
 * ============================================================================ */

void trixc_text(int x, int y, const char* text, uint32_t color) {
    trixc_text_scaled(x, y, text, color, 1);
}

void trixc_text_scaled(int x, int y, const char* text, uint32_t color, int scale) {
    if (!text || scale < 1) return;

    uint8_t r = (color >> 16) & 0xFF;
    uint8_t g = (color >> 8) & 0xFF;
    uint8_t b = color & 0xFF;

    SDL_SetRenderDrawColor(trixc_state.renderer, r, g, b, 255);

    int cursor_x = x;
    int cursor_y = y;

    while (*text) {
        char c = *text++;

        if (c == '\n') {
            cursor_x = x;
            cursor_y += 8 * scale + scale;
            continue;
        }

        if (c < 32 || c > 126) {
            c = '?';  /* Replace unprintable with ? */
        }

        const uint8_t* glyph = trixc_font_8x8[c - 32];

        for (int row = 0; row < 8; row++) {
            uint8_t row_data = glyph[row];
            for (int col = 0; col < 8; col++) {
                if (row_data & (0x80 >> col)) {
                    if (scale == 1) {
                        SDL_RenderDrawPoint(trixc_state.renderer,
                                            cursor_x + col, cursor_y + row);
                    } else {
                        SDL_Rect pixel = {
                            cursor_x + col * scale,
                            cursor_y + row * scale,
                            scale, scale
                        };
                        SDL_RenderFillRect(trixc_state.renderer, &pixel);
                    }
                }
            }
        }

        cursor_x += 8 * scale;
    }
}

void trixc_text_size(const char* text, int* w, int* h) {
    if (!text) {
        if (w) *w = 0;
        if (h) *h = 0;
        return;
    }

    int max_width = 0;
    int current_width = 0;
    int height = 8;

    while (*text) {
        if (*text == '\n') {
            if (current_width > max_width) max_width = current_width;
            current_width = 0;
            height += 9;
        } else {
            current_width += 8;
        }
        text++;
    }

    if (current_width > max_width) max_width = current_width;

    if (w) *w = max_width;
    if (h) *h = height;
}

void trixc_text_centered(int cx, int cy, const char* text, uint32_t color) {
    int w, h;
    trixc_text_size(text, &w, &h);
    trixc_text(cx - w / 2, cy - h / 2, text, color);
}

/* ============================================================================
 * DISPLAY - VISUALIZATION
 * ============================================================================ */

/* Colormap lookup tables */
static uint32_t colormap_grayscale(float v) {
    int c = (int)(v * 255);
    if (c < 0) c = 0;
    if (c > 255) c = 255;
    return TRIXC_RGB(c, c, c);
}

static uint32_t colormap_viridis(float v) {
    /* Simplified viridis approximation */
    if (v < 0) v = 0;
    if (v > 1) v = 1;

    int r, g, b;
    if (v < 0.5f) {
        float t = v * 2.0f;
        r = (int)(68 + t * (49 - 68));
        g = (int)(1 + t * (104 - 1));
        b = (int)(84 + t * (142 - 84));
    } else {
        float t = (v - 0.5f) * 2.0f;
        r = (int)(49 + t * (253 - 49));
        g = (int)(104 + t * (231 - 104));
        b = (int)(142 + t * (37 - 142));
    }

    return TRIXC_RGB(r, g, b);
}

static uint32_t colormap_plasma(float v) {
    if (v < 0) v = 0;
    if (v > 1) v = 1;

    int r, g, b;
    r = (int)(13 + v * (240 - 13));
    g = (int)(8 + v * v * (249 - 8));
    b = (int)(135 - v * 135);

    return TRIXC_RGB(r, g, b);
}

static uint32_t colormap_hot(float v) {
    if (v < 0) v = 0;
    if (v > 1) v = 1;

    int r, g, b;
    r = (int)(v * 3 * 255);
    if (r > 255) r = 255;
    g = (int)((v - 0.33f) * 3 * 255);
    if (g < 0) g = 0;
    if (g > 255) g = 255;
    b = (int)((v - 0.67f) * 3 * 255);
    if (b < 0) b = 0;
    if (b > 255) b = 255;

    return TRIXC_RGB(r, g, b);
}

void trixc_heatmap(int x, int y, int cell_w, int cell_h,
                   const float* data, int rows, int cols, int colormap) {
    if (!data) return;

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            float v = data[row * cols + col];

            uint32_t color;
            switch (colormap) {
                case 1:  color = colormap_viridis(v); break;
                case 2:  color = colormap_plasma(v); break;
                case 3:  color = colormap_hot(v); break;
                default: color = colormap_grayscale(v); break;
            }

            trixc_rect(x + col * cell_w, y + row * cell_h, cell_w - 1, cell_h - 1, color);
        }
    }
}

void trixc_bars(int x, int y, int w, int h,
                const float* values, const char** labels, int count,
                int highlight) {
    if (!values || count <= 0) return;

    /* Find max value for scaling */
    float max_val = 0.0f;
    for (int i = 0; i < count; i++) {
        if (values[i] > max_val) max_val = values[i];
    }
    if (max_val < 0.001f) max_val = 1.0f;

    int bar_height = h / count - 4;
    int label_width = labels ? 30 : 0;
    int bar_width = w - label_width - 60;

    for (int i = 0; i < count; i++) {
        int bar_y = y + i * (bar_height + 4);
        float normalized = values[i] / max_val;
        int filled_width = (int)(normalized * bar_width);

        /* Background */
        trixc_rect(x + label_width, bar_y, bar_width, bar_height, TRIXC_GRAY_20);

        /* Bar */
        uint32_t bar_color = (i == highlight) ? TRIXC_HIGHLIGHT : TRIXC_CYAN;
        trixc_rect(x + label_width, bar_y, filled_width, bar_height, bar_color);

        /* Label */
        if (labels && labels[i]) {
            trixc_text(x, bar_y + bar_height / 2 - 4, labels[i], TRIXC_WHITE);
        }

        /* Value */
        char val_str[16];
        snprintf(val_str, sizeof(val_str), "%.2f", values[i]);
        trixc_text(x + label_width + bar_width + 5, bar_y + bar_height / 2 - 4,
                   val_str, TRIXC_WHITE);
    }
}

void trixc_graph(int x, int y, int w, int h,
                 const float* values, int count, uint32_t color) {
    if (!values || count < 2) return;

    /* Find min/max for scaling */
    float min_val = values[0], max_val = values[0];
    for (int i = 1; i < count; i++) {
        if (values[i] < min_val) min_val = values[i];
        if (values[i] > max_val) max_val = values[i];
    }

    float range = max_val - min_val;
    if (range < 0.001f) range = 1.0f;

    /* Draw background */
    trixc_rect(x, y, w, h, TRIXC_GRAY_20);

    /* Draw line */
    for (int i = 0; i < count - 1; i++) {
        int x1 = x + i * w / (count - 1);
        int x2 = x + (i + 1) * w / (count - 1);
        int y1 = y + h - (int)((values[i] - min_val) / range * h);
        int y2 = y + h - (int)((values[i + 1] - min_val) / range * h);

        trixc_line(x1, y1, x2, y2, color);
    }
}

void trixc_stats(int x, int y, double inference_ms, size_t memory_bytes, float fps) {
    char buf[128];

    /* Box background */
    trixc_rect(x, y, 200, 60, TRIXC_GRAY_20);
    trixc_rect_outline(x, y, 200, 60, TRIXC_GRAY_50);

    /* Inference time */
    snprintf(buf, sizeof(buf), "Inference: %.3f ms", inference_ms);
    trixc_text(x + 10, y + 8, buf, TRIXC_GREEN);

    /* Memory */
    if (memory_bytes > 0) {
        if (memory_bytes < 1024) {
            snprintf(buf, sizeof(buf), "Memory: %zu bytes", memory_bytes);
        } else if (memory_bytes < 1024 * 1024) {
            snprintf(buf, sizeof(buf), "Memory: %.1f KB", memory_bytes / 1024.0);
        } else {
            snprintf(buf, sizeof(buf), "Memory: %.1f MB", memory_bytes / (1024.0 * 1024.0));
        }
        trixc_text(x + 10, y + 24, buf, TRIXC_CYAN);
    }

    /* FPS */
    if (fps > 0) {
        snprintf(buf, sizeof(buf), "FPS: %.1f", fps);
        trixc_text(x + 10, y + 40, buf, TRIXC_YELLOW);
    }
}

/* ============================================================================
 * INPUT
 * ============================================================================ */

bool trixc_poll(trixc_event_t* event) {
    if (!event) return false;

    SDL_Event sdl_event;
    while (SDL_PollEvent(&sdl_event)) {
        switch (sdl_event.type) {
            case SDL_QUIT:
                event->type = TRIXC_EVENT_QUIT;
                trixc_state.running = false;
                return true;

            case SDL_KEYDOWN:
                if (sdl_event.key.keysym.sym == SDLK_ESCAPE) {
                    event->type = TRIXC_EVENT_QUIT;
                    trixc_state.running = false;
                } else {
                    event->type = TRIXC_EVENT_KEY_DOWN;
                    event->key = sdl_event.key.keysym.sym;
                    event->key_char = (sdl_event.key.keysym.sym < 128) ?
                                      (char)sdl_event.key.keysym.sym : 0;
                }
                return true;

            case SDL_KEYUP:
                event->type = TRIXC_EVENT_KEY_UP;
                event->key = sdl_event.key.keysym.sym;
                return true;

            case SDL_MOUSEBUTTONDOWN:
                event->type = TRIXC_EVENT_TOUCH_DOWN;
                event->x = sdl_event.button.x;
                event->y = sdl_event.button.y;
                event->finger_id = 0;
                trixc_state.touch_down = true;
                trixc_state.touch_x = event->x;
                trixc_state.touch_y = event->y;
                return true;

            case SDL_MOUSEBUTTONUP:
                event->type = TRIXC_EVENT_TOUCH_UP;
                event->x = sdl_event.button.x;
                event->y = sdl_event.button.y;
                event->finger_id = 0;
                trixc_state.touch_down = false;
                return true;

            case SDL_MOUSEMOTION:
                if (trixc_state.touch_down) {
                    event->type = TRIXC_EVENT_TOUCH_MOVE;
                    event->x = sdl_event.motion.x;
                    event->y = sdl_event.motion.y;
                    event->finger_id = 0;
                    trixc_state.touch_x = event->x;
                    trixc_state.touch_y = event->y;
                    return true;
                }
                break;

            case SDL_FINGERDOWN:
                event->type = TRIXC_EVENT_TOUCH_DOWN;
                event->x = (int)(sdl_event.tfinger.x * trixc_state.width);
                event->y = (int)(sdl_event.tfinger.y * trixc_state.height);
                event->finger_id = (int)sdl_event.tfinger.fingerId;
                trixc_state.touch_down = true;
                trixc_state.touch_x = event->x;
                trixc_state.touch_y = event->y;
                return true;

            case SDL_FINGERUP:
                event->type = TRIXC_EVENT_TOUCH_UP;
                event->x = (int)(sdl_event.tfinger.x * trixc_state.width);
                event->y = (int)(sdl_event.tfinger.y * trixc_state.height);
                event->finger_id = (int)sdl_event.tfinger.fingerId;
                trixc_state.touch_down = false;
                return true;

            case SDL_FINGERMOTION:
                event->type = TRIXC_EVENT_TOUCH_MOVE;
                event->x = (int)(sdl_event.tfinger.x * trixc_state.width);
                event->y = (int)(sdl_event.tfinger.y * trixc_state.height);
                event->finger_id = (int)sdl_event.tfinger.fingerId;
                trixc_state.touch_x = event->x;
                trixc_state.touch_y = event->y;
                return true;
        }
    }

    event->type = TRIXC_EVENT_NONE;
    return false;
}

bool trixc_key_down(int key) {
    if (!trixc_state.key_state) return false;
    SDL_Scancode scancode = SDL_GetScancodeFromKey(key);
    return trixc_state.key_state[scancode] != 0;
}

bool trixc_touch_pos(int* x, int* y) {
    if (x) *x = trixc_state.touch_x;
    if (y) *y = trixc_state.touch_y;
    return trixc_state.touch_down;
}

/* ============================================================================
 * TIMING
 * ============================================================================ */

uint64_t trixc_time_us(void) {
    return get_time_us() - trixc_state.start_time_us;
}

double trixc_time_ms(void) {
    return (double)trixc_time_us() / 1000.0;
}

void trixc_sleep_ms(int ms) {
    SDL_Delay(ms);
}

void trixc_timer_init(trixc_timer_t* timer) {
    if (!timer) return;
    memset(timer, 0, sizeof(trixc_timer_t));
    timer->min_ms = 1e9;
}

void trixc_timer_start(trixc_timer_t* timer) {
    if (!timer) return;
    timer->start_us = get_time_us();
}

void trixc_timer_stop(trixc_timer_t* timer) {
    if (!timer) return;

    timer->end_us = get_time_us();
    timer->elapsed_ms = (double)(timer->end_us - timer->start_us) / 1000.0;

    /* Update statistics */
    if (timer->elapsed_ms < timer->min_ms) timer->min_ms = timer->elapsed_ms;
    if (timer->elapsed_ms > timer->max_ms) timer->max_ms = timer->elapsed_ms;

    timer->count++;
    double delta = timer->elapsed_ms - timer->avg_ms;
    timer->avg_ms += delta / timer->count;
}

void trixc_timer_reset(trixc_timer_t* timer) {
    if (!timer) return;
    timer->min_ms = 1e9;
    timer->max_ms = 0;
    timer->avg_ms = 0;
    timer->count = 0;
}

float trixc_fps(void) {
    return trixc_state.current_fps;
}

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

int trixc_argmax(const float* values, int count) {
    if (!values || count <= 0) return -1;

    int max_idx = 0;
    float max_val = values[0];

    for (int i = 1; i < count; i++) {
        if (values[i] > max_val) {
            max_val = values[i];
            max_idx = i;
        }
    }

    return max_idx;
}

void trixc_softmax(float* values, int count) {
    if (!values || count <= 0) return;

    /* Find max for numerical stability */
    float max_val = values[0];
    for (int i = 1; i < count; i++) {
        if (values[i] > max_val) max_val = values[i];
    }

    /* Compute exp and sum */
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        values[i] = expf(values[i] - max_val);
        sum += values[i];
    }

    /* Normalize */
    for (int i = 0; i < count; i++) {
        values[i] /= sum;
    }
}

/* ============================================================================
 * DEBUG / LOGGING
 * ============================================================================ */

void trixc_debug_level(int level) {
    trixc_state.debug_level = level;
}

void trixc_log(int level, const char* fmt, ...) {
    if (level > trixc_state.debug_level) return;

    const char* prefix;
    switch (level) {
        case TRIXC_LOG_ERROR: prefix = "[ERROR] "; break;
        case TRIXC_LOG_WARN:  prefix = "[WARN]  "; break;
        case TRIXC_LOG_INFO:  prefix = "[INFO]  "; break;
        case TRIXC_LOG_DEBUG: prefix = "[DEBUG] "; break;
        default:             prefix = "";         break;
    }

    fprintf(stderr, "%s", prefix);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

/* ============================================================================
 * END OF IMPLEMENTATION
 * ============================================================================ */
