/*
 * TRIXC Pi Runtime Header
 *
 * The bridge between frozen shapes and the physical world.
 * Display, input, timing, GPIO, camera - all in one clean API.
 *
 * "When you're ready, you won't have to dodge bullets."
 *   - Actually that's The Matrix, but the point stands.
 *
 * "It's all in the reflexes."
 *   - Jack Burton, and that's more like it.
 *
 * ============================================================================
 * DESIGN PHILOSOPHY
 * ============================================================================
 *
 * 1. ZERO FRICTION
 *    Everything should "just work" on a Raspberry Pi with minimal setup.
 *    No configuration files. No environment variables. Sensible defaults.
 *
 * 2. PROGRESSIVE DISCLOSURE
 *    Simple things should be simple. Complex things should be possible.
 *    The basic display API is 5 functions. The advanced API is there when needed.
 *
 * 3. LEGIBILITY
 *    The code you write should read like what it does.
 *    trixc_display_text(100, 100, "Hello", WHITE);  // Obviously correct
 *
 * 4. PERFORMANCE VISIBILITY
 *    Always know how fast your inference is. Timing is built-in, not bolted on.
 *
 * ============================================================================
 * DEPENDENCIES
 * ============================================================================
 *
 * Required: SDL2 (apt install libsdl2-dev)
 * Optional: SDL2_ttf for text (apt install libsdl2-ttf-dev)
 * Optional: pigpio for GPIO (apt install pigpio)
 * Optional: libcamera for Pi camera
 *
 * ============================================================================
 * QUICK START
 * ============================================================================
 *
 *   #include "trixc_pi.h"
 *
 *   int main() {
 *       trixc_init("My App", 800, 480);
 *
 *       while (trixc_running()) {
 *           trixc_clear(TRIXC_BLACK);
 *           trixc_text(100, 100, "Hello TRIXC Pi!", TRIXC_WHITE);
 *           trixc_present();
 *       }
 *
 *       trixc_shutdown();
 *       return 0;
 *   }
 *
 * ============================================================================
 */

#ifndef TRIXC_PI_H
#define TRIXC_PI_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * VERSION
 * ============================================================================ */

#define TRIXC_PI_VERSION_MAJOR 1
#define TRIXC_PI_VERSION_MINOR 0
#define TRIXC_PI_VERSION_PATCH 0
#define TRIXC_PI_VERSION_STRING "1.0.0"

/* ============================================================================
 * COLORS
 *
 * Pre-defined colors for convenience. Format: 0xRRGGBB
 * ============================================================================ */

#define TRIXC_BLACK       0x000000
#define TRIXC_WHITE       0xFFFFFF
#define TRIXC_RED         0xFF0000
#define TRIXC_GREEN       0x00FF00
#define TRIXC_BLUE        0x0000FF
#define TRIXC_YELLOW      0xFFFF00
#define TRIXC_CYAN        0x00FFFF
#define TRIXC_MAGENTA     0xFF00FF
#define TRIXC_ORANGE      0xFF8800
#define TRIXC_PURPLE      0x8800FF

/* TRIXC brand colors */
#define TRIXC_BG_DARK     0x1A1A2E   /* Dark blue-purple background */
#define TRIXC_BG_MEDIUM   0x16213E   /* Medium background */
#define TRIXC_ACCENT      0x0F3460   /* Accent color */
#define TRIXC_HIGHLIGHT   0xE94560   /* Highlight/attention color */

/* Grayscale */
#define TRIXC_GRAY_10     0x1A1A1A
#define TRIXC_GRAY_20     0x333333
#define TRIXC_GRAY_30     0x4D4D4D
#define TRIXC_GRAY_40     0x666666
#define TRIXC_GRAY_50     0x808080
#define TRIXC_GRAY_60     0x999999
#define TRIXC_GRAY_70     0xB3B3B3
#define TRIXC_GRAY_80     0xCCCCCC
#define TRIXC_GRAY_90     0xE6E6E6

/* Helper to create RGB color */
#define TRIXC_RGB(r, g, b) (((r) << 16) | ((g) << 8) | (b))

/* ============================================================================
 * CORE INITIALIZATION
 *
 * The simple API - one call to set up everything.
 * ============================================================================ */

/**
 * Initialize TRIXC Pi runtime.
 *
 * Creates a window/fullscreen display, initializes input handling,
 * and prepares the timing system.
 *
 * @param title     Window title (shown in windowed mode)
 * @param width     Display width in pixels (0 = auto-detect)
 * @param height    Display height in pixels (0 = auto-detect)
 * @return          0 on success, -1 on failure
 *
 * Example:
 *   trixc_init("My TRIXC App", 800, 480);
 *   trixc_init("Fullscreen", 0, 0);  // Auto-detect resolution
 */
int trixc_init(const char* title, int width, int height);

/**
 * Shut down TRIXC Pi runtime.
 *
 * Releases all resources. Call before exit.
 */
void trixc_shutdown(void);

/**
 * Check if the application should continue running.
 *
 * Returns false when the user closes the window or presses Escape.
 * Use this as your main loop condition.
 *
 * Example:
 *   while (trixc_running()) {
 *       // Your code here
 *   }
 */
bool trixc_running(void);

/**
 * Get display dimensions.
 */
int trixc_width(void);
int trixc_height(void);

/* ============================================================================
 * DISPLAY - BASIC
 *
 * The essential drawing functions. Simple and fast.
 * ============================================================================ */

/**
 * Clear the screen to a solid color.
 *
 * @param color     Color in 0xRRGGBB format
 *
 * Example:
 *   trixc_clear(TRIXC_BLACK);
 *   trixc_clear(TRIXC_RGB(30, 30, 60));
 */
void trixc_clear(uint32_t color);

/**
 * Present the current frame to the display.
 *
 * Call this once per frame after all drawing is done.
 * This also handles VSync and frame timing.
 */
void trixc_present(void);

/**
 * Draw a single pixel.
 *
 * @param x, y      Position
 * @param color     Color in 0xRRGGBB format
 */
void trixc_pixel(int x, int y, uint32_t color);

/**
 * Draw a filled rectangle.
 *
 * @param x, y      Top-left corner
 * @param w, h      Width and height
 * @param color     Fill color
 */
void trixc_rect(int x, int y, int w, int h, uint32_t color);

/**
 * Draw a rectangle outline.
 *
 * @param x, y      Top-left corner
 * @param w, h      Width and height
 * @param color     Outline color
 */
void trixc_rect_outline(int x, int y, int w, int h, uint32_t color);

/**
 * Draw a line.
 *
 * @param x1, y1    Start point
 * @param x2, y2    End point
 * @param color     Line color
 */
void trixc_line(int x1, int y1, int x2, int y2, uint32_t color);

/**
 * Draw a filled circle.
 *
 * @param cx, cy    Center position
 * @param r         Radius
 * @param color     Fill color
 */
void trixc_circle(int cx, int cy, int r, uint32_t color);

/**
 * Draw a circle outline.
 */
void trixc_circle_outline(int cx, int cy, int r, uint32_t color);

/* ============================================================================
 * DISPLAY - TEXT
 *
 * Text rendering with built-in font or SDL2_ttf.
 * ============================================================================ */

/**
 * Draw text at position.
 *
 * Uses a built-in bitmap font if SDL2_ttf is not available.
 *
 * @param x, y      Position (top-left of text)
 * @param text      String to draw
 * @param color     Text color
 */
void trixc_text(int x, int y, const char* text, uint32_t color);

/**
 * Draw text with custom scale.
 *
 * @param scale     Size multiplier (1 = normal, 2 = double, etc.)
 */
void trixc_text_scaled(int x, int y, const char* text, uint32_t color, int scale);

/**
 * Get text dimensions.
 *
 * @param text      String to measure
 * @param w, h      Output: width and height in pixels (can be NULL)
 */
void trixc_text_size(const char* text, int* w, int* h);

/**
 * Draw text centered at position.
 *
 * @param cx, cy    Center position
 */
void trixc_text_centered(int cx, int cy, const char* text, uint32_t color);

/* ============================================================================
 * DISPLAY - VISUALIZATION
 *
 * Higher-level visualization for ML outputs.
 * ============================================================================ */

/**
 * Draw a heatmap visualization.
 *
 * Perfect for visualizing activations, attention maps, etc.
 *
 * @param x, y          Top-left position
 * @param cell_w, cell_h   Size of each cell
 * @param data          Float array of values (0.0 to 1.0 expected)
 * @param rows, cols    Grid dimensions
 * @param colormap      0=grayscale, 1=viridis, 2=plasma, 3=hot
 */
void trixc_heatmap(int x, int y, int cell_w, int cell_h,
                   const float* data, int rows, int cols, int colormap);

/**
 * Draw a bar chart.
 *
 * Great for classification probabilities.
 *
 * @param x, y          Top-left position
 * @param w, h          Total width and height
 * @param values        Array of values
 * @param labels        Array of label strings (can be NULL)
 * @param count         Number of bars
 * @param highlight     Index to highlight (-1 for none)
 */
void trixc_bars(int x, int y, int w, int h,
                const float* values, const char** labels, int count,
                int highlight);

/**
 * Draw a line graph.
 *
 * @param x, y          Top-left position
 * @param w, h          Total width and height
 * @param values        Array of values
 * @param count         Number of points
 * @param color         Line color
 */
void trixc_graph(int x, int y, int w, int h,
                 const float* values, int count, uint32_t color);

/**
 * Draw inference statistics.
 *
 * Shows timing and memory usage in a formatted box.
 *
 * @param x, y              Position
 * @param inference_ms      Inference time in milliseconds
 * @param memory_bytes      Model memory usage (0 to hide)
 * @param fps               Frames per second (0 to hide)
 */
void trixc_stats(int x, int y, double inference_ms, size_t memory_bytes, float fps);

/* ============================================================================
 * INPUT - EVENTS
 *
 * Unified event handling for touch, mouse, and keyboard.
 * ============================================================================ */

/**
 * Event types.
 */
typedef enum {
    TRIXC_EVENT_NONE = 0,
    TRIXC_EVENT_QUIT,           /* Window closed or Escape pressed */
    TRIXC_EVENT_TOUCH_DOWN,     /* Finger/mouse pressed */
    TRIXC_EVENT_TOUCH_UP,       /* Finger/mouse released */
    TRIXC_EVENT_TOUCH_MOVE,     /* Finger/mouse moved while pressed */
    TRIXC_EVENT_KEY_DOWN,       /* Key pressed */
    TRIXC_EVENT_KEY_UP          /* Key released */
} trixc_event_type_t;

/**
 * Event structure.
 */
typedef struct {
    trixc_event_type_t type;
    int x, y;           /* Touch/mouse position */
    int finger_id;      /* For multi-touch (0 = primary) */
    int key;            /* SDL key code for key events */
    char key_char;      /* Character for printable keys */
} trixc_event_t;

/**
 * Poll for events.
 *
 * Call this in your main loop to handle input.
 * Returns true if an event was retrieved, false if queue is empty.
 *
 * Example:
 *   trixc_event_t event;
 *   while (trixc_poll(&event)) {
 *       if (event.type == TRIXC_EVENT_TOUCH_DOWN) {
 *           printf("Touched at %d, %d\n", event.x, event.y);
 *       }
 *   }
 */
bool trixc_poll(trixc_event_t* event);

/**
 * Check if a key is currently pressed.
 *
 * @param key       SDL key code (e.g., 'a', SDLK_SPACE, etc.)
 */
bool trixc_key_down(int key);

/**
 * Get current touch/mouse position.
 *
 * @param x, y      Output: current position (can be NULL)
 * @return          true if currently touched/pressed
 */
bool trixc_touch_pos(int* x, int* y);

/* ============================================================================
 * TIMING
 *
 * High-resolution timing for performance measurement.
 * ============================================================================ */

/**
 * Get time in microseconds since initialization.
 */
uint64_t trixc_time_us(void);

/**
 * Get time in milliseconds since initialization.
 */
double trixc_time_ms(void);

/**
 * Sleep for specified milliseconds.
 *
 * Note: For frame timing, prefer using trixc_present() which handles VSync.
 */
void trixc_sleep_ms(int ms);

/**
 * Timer structure for measuring code sections.
 */
typedef struct {
    uint64_t start_us;
    uint64_t end_us;
    double elapsed_ms;
    double min_ms;
    double max_ms;
    double avg_ms;
    int count;
} trixc_timer_t;

/**
 * Initialize a timer.
 */
void trixc_timer_init(trixc_timer_t* timer);

/**
 * Start timing.
 */
void trixc_timer_start(trixc_timer_t* timer);

/**
 * Stop timing and update statistics.
 */
void trixc_timer_stop(trixc_timer_t* timer);

/**
 * Reset timer statistics.
 */
void trixc_timer_reset(trixc_timer_t* timer);

/**
 * Get current FPS (updated by trixc_present).
 */
float trixc_fps(void);

/* ============================================================================
 * GPIO (Raspberry Pi specific)
 *
 * Available when compiled with -DTRIXC_PI_GPIO and linked with -lpigpio.
 * ============================================================================ */

#ifdef TRIXC_PI_GPIO

/* Pin modes */
#define TRIXC_GPIO_INPUT  0
#define TRIXC_GPIO_OUTPUT 1
#define TRIXC_GPIO_PWM    2

/* Pull-up/down */
#define TRIXC_GPIO_PULL_OFF  0
#define TRIXC_GPIO_PULL_DOWN 1
#define TRIXC_GPIO_PULL_UP   2

/**
 * Initialize GPIO subsystem.
 *
 * @return  0 on success, -1 on failure
 */
int trixc_gpio_init(void);

/**
 * Shut down GPIO subsystem.
 */
void trixc_gpio_shutdown(void);

/**
 * Configure a GPIO pin.
 *
 * @param pin       BCM GPIO number (not physical pin number!)
 * @param mode      TRIXC_GPIO_INPUT, TRIXC_GPIO_OUTPUT, or TRIXC_GPIO_PWM
 */
void trixc_gpio_mode(int pin, int mode);

/**
 * Set pull-up/pull-down resistor.
 *
 * @param pin       BCM GPIO number
 * @param pull      TRIXC_GPIO_PULL_OFF, TRIXC_GPIO_PULL_DOWN, or TRIXC_GPIO_PULL_UP
 */
void trixc_gpio_pull(int pin, int pull);

/**
 * Write to an output pin.
 *
 * @param pin       BCM GPIO number
 * @param value     0 (low) or 1 (high)
 */
void trixc_gpio_write(int pin, int value);

/**
 * Read from an input pin.
 *
 * @param pin       BCM GPIO number
 * @return          0 (low) or 1 (high)
 */
int trixc_gpio_read(int pin);

/**
 * Set PWM duty cycle.
 *
 * @param pin       BCM GPIO number (must be configured as PWM)
 * @param duty      Duty cycle 0-255 (0 = off, 255 = full on)
 */
void trixc_gpio_pwm(int pin, int duty);

/**
 * Convenience: Set an LED.
 */
static inline void trixc_led(int pin, bool on) {
    trixc_gpio_write(pin, on ? 1 : 0);
}

/**
 * Convenience: Read a button (with debounce).
 *
 * @param pin       BCM GPIO number
 * @return          true if pressed
 */
bool trixc_button(int pin);

#endif /* TRIXC_PI_GPIO */

/* ============================================================================
 * CAMERA (Raspberry Pi specific)
 *
 * Available when compiled with -DTRIXC_PI_CAMERA.
 * ============================================================================ */

#ifdef TRIXC_PI_CAMERA

/**
 * Camera frame structure.
 */
typedef struct {
    int width;
    int height;
    uint8_t* rgb;       /* RGB24 data (3 bytes per pixel) */
    size_t size;        /* Total size in bytes */
} trixc_frame_t;

/**
 * Initialize camera.
 *
 * @param width     Desired width (0 = default 640)
 * @param height    Desired height (0 = default 480)
 * @return          0 on success, -1 on failure
 */
int trixc_camera_init(int width, int height);

/**
 * Shut down camera.
 */
void trixc_camera_shutdown(void);

/**
 * Capture a frame.
 *
 * @param frame     Output frame (must be initialized)
 * @return          0 on success, -1 on failure
 */
int trixc_camera_capture(trixc_frame_t* frame);

/**
 * Allocate frame buffer.
 */
int trixc_frame_alloc(trixc_frame_t* frame, int width, int height);

/**
 * Free frame buffer.
 */
void trixc_frame_free(trixc_frame_t* frame);

/**
 * Display a frame on screen.
 *
 * @param x, y      Position
 * @param frame     Frame to display
 * @param scale     Scale factor (1 = original size)
 */
void trixc_frame_display(int x, int y, const trixc_frame_t* frame, int scale);

/* Frame preprocessing for ML models */

/**
 * Convert frame to grayscale float array.
 *
 * @param frame     Input frame
 * @param output    Output array (must be frame->width * frame->height floats)
 * @param normalize If true, output is 0.0-1.0; otherwise 0-255
 */
void trixc_frame_grayscale(const trixc_frame_t* frame, float* output, bool normalize);

/**
 * Resize frame to target dimensions.
 *
 * @param src       Source frame
 * @param dst       Destination frame (must be pre-allocated)
 */
void trixc_frame_resize(const trixc_frame_t* src, trixc_frame_t* dst);

/**
 * Crop frame to region.
 *
 * @param src       Source frame
 * @param dst       Destination frame (must be pre-allocated)
 * @param x, y      Top-left of crop region
 */
void trixc_frame_crop(const trixc_frame_t* src, trixc_frame_t* dst, int x, int y);

#endif /* TRIXC_PI_CAMERA */

/* ============================================================================
 * AUDIO (Optional)
 *
 * Available when compiled with -DTRIXC_PI_AUDIO.
 * ============================================================================ */

#ifdef TRIXC_PI_AUDIO

/**
 * Initialize audio capture.
 *
 * @param sample_rate   Sample rate (e.g., 16000 for voice)
 * @param buffer_size   Buffer size in samples
 * @return              0 on success, -1 on failure
 */
int trixc_audio_init(int sample_rate, int buffer_size);

/**
 * Shut down audio.
 */
void trixc_audio_shutdown(void);

/**
 * Capture audio samples.
 *
 * @param buffer    Output buffer (float, -1.0 to 1.0)
 * @param count     Number of samples to capture
 * @return          Number of samples actually captured
 */
int trixc_audio_capture(float* buffer, int count);

#endif /* TRIXC_PI_AUDIO */

/* ============================================================================
 * UTILITY FUNCTIONS
 * ============================================================================ */

/**
 * Find the index of the maximum value in an array.
 *
 * @param values    Array of floats
 * @param count     Array size
 * @return          Index of maximum value
 */
int trixc_argmax(const float* values, int count);

/**
 * Apply softmax to an array in-place.
 *
 * @param values    Array of floats
 * @param count     Array size
 */
void trixc_softmax(float* values, int count);

/**
 * Clamp a value to a range.
 */
static inline float trixc_clamp(float x, float min, float max) {
    return x < min ? min : (x > max ? max : x);
}

/**
 * Linear interpolation.
 */
static inline float trixc_lerp(float a, float b, float t) {
    return a + t * (b - a);
}

/**
 * Map a value from one range to another.
 */
static inline float trixc_map(float x, float in_min, float in_max, float out_min, float out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

/* ============================================================================
 * DEBUG / DEVELOPMENT
 * ============================================================================ */

/**
 * Set debug verbosity level.
 *
 * 0 = silent, 1 = errors only, 2 = warnings, 3 = info, 4 = verbose
 */
void trixc_debug_level(int level);

/**
 * Log a message (respects debug level).
 */
void trixc_log(int level, const char* fmt, ...);

#define TRIXC_LOG_ERROR 1
#define TRIXC_LOG_WARN  2
#define TRIXC_LOG_INFO  3
#define TRIXC_LOG_DEBUG 4

/* ============================================================================
 * BUILT-IN FONT DATA
 *
 * 8x8 bitmap font for text rendering without SDL2_ttf.
 * Covers ASCII 32-126.
 * ============================================================================ */

extern const uint8_t trixc_font_8x8[95][8];

#ifdef __cplusplus
}
#endif

#endif /* TRIXC_PI_H */

/*
 * ============================================================================
 * IMPLEMENTATION NOTES
 * ============================================================================
 *
 * This header defines the complete API for TRIXC Pi.
 *
 * Implementation files:
 *   - trixc_pi_core.c     : Core init, shutdown, event loop
 *   - trixc_pi_display.c  : All rendering functions
 *   - trixc_pi_input.c    : Event handling
 *   - trixc_pi_timing.c   : High-resolution timing
 *   - trixc_pi_gpio.c     : GPIO (optional)
 *   - trixc_pi_camera.c   : Camera (optional)
 *   - trixc_pi_audio.c    : Audio (optional)
 *   - trixc_pi_font.c     : Built-in 8x8 font data
 *
 * To use as a single-file library, compile with:
 *   gcc -DTRIXC_PI_IMPLEMENTATION your_app.c -lSDL2 -lm
 *
 * ============================================================================
 */
