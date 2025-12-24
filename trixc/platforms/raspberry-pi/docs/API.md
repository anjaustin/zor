# TRIXC Pi API Reference

Complete reference for all functions in `trixc_pi.h`.

---

## Table of Contents

1. [Core Functions](#core-functions)
2. [Display - Basic](#display---basic)
3. [Display - Text](#display---text)
4. [Display - Visualization](#display---visualization)
5. [Input](#input)
6. [Timing](#timing)
7. [GPIO](#gpio-optional)
8. [Camera](#camera-optional)
9. [Utility Functions](#utility-functions)
10. [Constants](#constants)

---

## Core Functions

### trixc_init

```c
int trixc_init(const char* title, int width, int height);
```

Initialize the TRIXC Pi runtime.

**Parameters:**
- `title` - Window title (shown in windowed mode)
- `width` - Display width in pixels (0 = auto-detect)
- `height` - Display height in pixels (0 = auto-detect)

**Returns:** 0 on success, -1 on failure

**Example:**
```c
// Windowed mode
trixc_init("My App", 800, 480);

// Fullscreen (auto-detect resolution)
trixc_init("Fullscreen App", 0, 0);
```

---

### trixc_shutdown

```c
void trixc_shutdown(void);
```

Shut down the TRIXC Pi runtime. Call before exit.

---

### trixc_running

```c
bool trixc_running(void);
```

Check if the application should continue running.

**Returns:** `false` when user closes window or presses Escape

**Example:**
```c
while (trixc_running()) {
    // Main loop
}
```

---

### trixc_width / trixc_height

```c
int trixc_width(void);
int trixc_height(void);
```

Get display dimensions in pixels.

---

## Display - Basic

### trixc_clear

```c
void trixc_clear(uint32_t color);
```

Clear the screen to a solid color.

**Parameters:**
- `color` - Color in 0xRRGGBB format

**Example:**
```c
trixc_clear(TRIXC_BLACK);
trixc_clear(TRIXC_RGB(30, 30, 60));
```

---

### trixc_present

```c
void trixc_present(void);
```

Present the current frame to the display. Call once per frame after all drawing.

Also updates FPS counter and processes input events.

---

### trixc_pixel

```c
void trixc_pixel(int x, int y, uint32_t color);
```

Draw a single pixel.

---

### trixc_rect

```c
void trixc_rect(int x, int y, int w, int h, uint32_t color);
```

Draw a filled rectangle.

**Parameters:**
- `x, y` - Top-left corner
- `w, h` - Width and height
- `color` - Fill color

---

### trixc_rect_outline

```c
void trixc_rect_outline(int x, int y, int w, int h, uint32_t color);
```

Draw a rectangle outline (not filled).

---

### trixc_line

```c
void trixc_line(int x1, int y1, int x2, int y2, uint32_t color);
```

Draw a line between two points.

---

### trixc_circle

```c
void trixc_circle(int cx, int cy, int r, uint32_t color);
```

Draw a filled circle.

**Parameters:**
- `cx, cy` - Center position
- `r` - Radius
- `color` - Fill color

---

### trixc_circle_outline

```c
void trixc_circle_outline(int cx, int cy, int r, uint32_t color);
```

Draw a circle outline (not filled).

---

## Display - Text

### trixc_text

```c
void trixc_text(int x, int y, const char* text, uint32_t color);
```

Draw text at position using the built-in 8x8 bitmap font.

**Parameters:**
- `x, y` - Position (top-left of text)
- `text` - String to draw
- `color` - Text color

**Example:**
```c
trixc_text(100, 100, "Hello World!", TRIXC_WHITE);
```

---

### trixc_text_scaled

```c
void trixc_text_scaled(int x, int y, const char* text, uint32_t color, int scale);
```

Draw text with custom scale.

**Parameters:**
- `scale` - Size multiplier (1 = normal, 2 = double, etc.)

**Example:**
```c
trixc_text_scaled(100, 50, "BIG TEXT", TRIXC_WHITE, 3);
```

---

### trixc_text_size

```c
void trixc_text_size(const char* text, int* w, int* h);
```

Get text dimensions in pixels.

---

### trixc_text_centered

```c
void trixc_text_centered(int cx, int cy, const char* text, uint32_t color);
```

Draw text centered at position.

---

## Display - Visualization

### trixc_heatmap

```c
void trixc_heatmap(int x, int y, int cell_w, int cell_h,
                   const float* data, int rows, int cols, int colormap);
```

Draw a heatmap visualization.

**Parameters:**
- `x, y` - Top-left position
- `cell_w, cell_h` - Size of each cell
- `data` - Float array of values (0.0 to 1.0 expected)
- `rows, cols` - Grid dimensions
- `colormap` - 0=grayscale, 1=viridis, 2=plasma, 3=hot

**Example:**
```c
float activations[16];  // 4x4 grid
trixc_heatmap(100, 100, 20, 20, activations, 4, 4, 1);  // Viridis
```

---

### trixc_bars

```c
void trixc_bars(int x, int y, int w, int h,
                const float* values, const char** labels, int count,
                int highlight);
```

Draw a horizontal bar chart.

**Parameters:**
- `x, y` - Top-left position
- `w, h` - Total width and height
- `values` - Array of values
- `labels` - Array of label strings (can be NULL)
- `count` - Number of bars
- `highlight` - Index to highlight (-1 for none)

**Example:**
```c
float probs[10] = {...};
const char* digits[] = {"0","1","2","3","4","5","6","7","8","9"};
trixc_bars(100, 100, 300, 200, probs, digits, 10, predicted);
```

---

### trixc_graph

```c
void trixc_graph(int x, int y, int w, int h,
                 const float* values, int count, uint32_t color);
```

Draw a line graph.

---

### trixc_stats

```c
void trixc_stats(int x, int y, double inference_ms, size_t memory_bytes, float fps);
```

Draw inference statistics in a formatted box.

**Parameters:**
- `x, y` - Position
- `inference_ms` - Inference time in milliseconds
- `memory_bytes` - Model memory usage (0 to hide)
- `fps` - Frames per second (0 to hide)

**Example:**
```c
trixc_stats(50, 400, timer.elapsed_ms, model_size(), trixc_fps());
```

---

## Input

### trixc_event_t

```c
typedef struct {
    trixc_event_type_t type;
    int x, y;           // Touch/mouse position
    int finger_id;      // For multi-touch (0 = primary)
    int key;            // SDL key code for key events
    char key_char;      // Character for printable keys
} trixc_event_t;
```

### Event Types

```c
typedef enum {
    TRIXC_EVENT_NONE,
    TRIXC_EVENT_QUIT,        // Window closed or Escape pressed
    TRIXC_EVENT_TOUCH_DOWN,  // Finger/mouse pressed
    TRIXC_EVENT_TOUCH_UP,    // Finger/mouse released
    TRIXC_EVENT_TOUCH_MOVE,  // Finger/mouse moved while pressed
    TRIXC_EVENT_KEY_DOWN,    // Key pressed
    TRIXC_EVENT_KEY_UP       // Key released
} trixc_event_type_t;
```

---

### trixc_poll

```c
bool trixc_poll(trixc_event_t* event);
```

Poll for input events.

**Returns:** `true` if an event was retrieved, `false` if queue is empty

**Example:**
```c
trixc_event_t event;
while (trixc_poll(&event)) {
    if (event.type == TRIXC_EVENT_TOUCH_DOWN) {
        printf("Touched at %d, %d\n", event.x, event.y);
    }
}
```

---

### trixc_key_down

```c
bool trixc_key_down(int key);
```

Check if a key is currently pressed.

**Example:**
```c
if (trixc_key_down('w')) {
    // W key is held
}
```

---

### trixc_touch_pos

```c
bool trixc_touch_pos(int* x, int* y);
```

Get current touch/mouse position.

**Returns:** `true` if currently touched/pressed

---

## Timing

### trixc_time_us

```c
uint64_t trixc_time_us(void);
```

Get time in microseconds since initialization.

---

### trixc_time_ms

```c
double trixc_time_ms(void);
```

Get time in milliseconds since initialization.

---

### trixc_sleep_ms

```c
void trixc_sleep_ms(int ms);
```

Sleep for specified milliseconds.

---

### trixc_timer_t

```c
typedef struct {
    uint64_t start_us;
    uint64_t end_us;
    double elapsed_ms;
    double min_ms;
    double max_ms;
    double avg_ms;
    int count;
} trixc_timer_t;
```

Timer structure with statistics.

---

### Timer Functions

```c
void trixc_timer_init(trixc_timer_t* timer);
void trixc_timer_start(trixc_timer_t* timer);
void trixc_timer_stop(trixc_timer_t* timer);
void trixc_timer_reset(trixc_timer_t* timer);
```

**Example:**
```c
trixc_timer_t timer;
trixc_timer_init(&timer);

trixc_timer_start(&timer);
model_forward(input, output);
trixc_timer_stop(&timer);

printf("Inference: %.3f ms (avg: %.3f)\n", timer.elapsed_ms, timer.avg_ms);
```

---

### trixc_fps

```c
float trixc_fps(void);
```

Get current frames per second (updated by `trixc_present`).

---

## GPIO (Optional)

Available when compiled with `-DTRIXC_PI_GPIO` and linked with `-lpigpio`.

### Constants

```c
#define TRIXC_GPIO_INPUT  0
#define TRIXC_GPIO_OUTPUT 1
#define TRIXC_GPIO_PWM    2

#define TRIXC_GPIO_PULL_OFF  0
#define TRIXC_GPIO_PULL_DOWN 1
#define TRIXC_GPIO_PULL_UP   2
```

---

### trixc_gpio_init / trixc_gpio_shutdown

```c
int trixc_gpio_init(void);
void trixc_gpio_shutdown(void);
```

Initialize/shutdown GPIO subsystem.

---

### trixc_gpio_mode

```c
void trixc_gpio_mode(int pin, int mode);
```

Configure a GPIO pin.

**Parameters:**
- `pin` - BCM GPIO number (not physical pin!)
- `mode` - `TRIXC_GPIO_INPUT`, `TRIXC_GPIO_OUTPUT`, or `TRIXC_GPIO_PWM`

---

### trixc_gpio_write / trixc_gpio_read

```c
void trixc_gpio_write(int pin, int value);
int trixc_gpio_read(int pin);
```

Write to output pin / read from input pin.

---

### trixc_gpio_pwm

```c
void trixc_gpio_pwm(int pin, int duty);
```

Set PWM duty cycle (0-255).

---

### trixc_led

```c
static inline void trixc_led(int pin, bool on);
```

Convenience function for LED control.

---

## Camera (Optional)

Available when compiled with `-DTRIXC_PI_CAMERA`.

### trixc_frame_t

```c
typedef struct {
    int width;
    int height;
    uint8_t* rgb;       // RGB24 data
    size_t size;        // Total size in bytes
} trixc_frame_t;
```

---

### Camera Functions

```c
int trixc_camera_init(int width, int height);
void trixc_camera_shutdown(void);
int trixc_camera_capture(trixc_frame_t* frame);

int trixc_frame_alloc(trixc_frame_t* frame, int width, int height);
void trixc_frame_free(trixc_frame_t* frame);
void trixc_frame_display(int x, int y, const trixc_frame_t* frame, int scale);

void trixc_frame_grayscale(const trixc_frame_t* frame, float* output, bool normalize);
void trixc_frame_resize(const trixc_frame_t* src, trixc_frame_t* dst);
void trixc_frame_crop(const trixc_frame_t* src, trixc_frame_t* dst, int x, int y);
```

---

## Utility Functions

### trixc_argmax

```c
int trixc_argmax(const float* values, int count);
```

Find the index of the maximum value in an array.

**Example:**
```c
int predicted = trixc_argmax(output, 10);
```

---

### trixc_softmax

```c
void trixc_softmax(float* values, int count);
```

Apply softmax to an array in-place.

---

### Inline Helpers

```c
static inline float trixc_clamp(float x, float min, float max);
static inline float trixc_lerp(float a, float b, float t);
static inline float trixc_map(float x, float in_min, float in_max, float out_min, float out_max);
```

---

## Constants

### Colors

```c
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

// TRIXC brand colors
#define TRIXC_BG_DARK     0x1A1A2E
#define TRIXC_BG_MEDIUM   0x16213E
#define TRIXC_ACCENT      0x0F3460
#define TRIXC_HIGHLIGHT   0xE94560

// Grayscale (10% increments)
#define TRIXC_GRAY_10 through TRIXC_GRAY_90

// Color helper
#define TRIXC_RGB(r, g, b) (((r) << 16) | ((g) << 8) | (b))
```

### Log Levels

```c
#define TRIXC_LOG_ERROR 1
#define TRIXC_LOG_WARN  2
#define TRIXC_LOG_INFO  3
#define TRIXC_LOG_DEBUG 4
```

---

## Complete Example

```c
#include "trixc_pi.h"
#include "your_model.c"

int main() {
    // Initialize
    if (trixc_init("Classifier", 800, 480) != 0) {
        return 1;
    }

    float input[49] = {0};
    float output[10];
    trixc_timer_t timer;
    trixc_timer_init(&timer);

    // Main loop
    while (trixc_running()) {
        // Handle input
        trixc_event_t event;
        while (trixc_poll(&event)) {
            if (event.type == TRIXC_EVENT_TOUCH_DOWN) {
                // Handle touch
            }
        }

        // Run inference
        trixc_timer_start(&timer);
        model_forward(input, output);
        trixc_timer_stop(&timer);

        // Draw
        trixc_clear(TRIXC_BG_DARK);
        trixc_text(100, 50, "My Classifier", TRIXC_WHITE);

        int predicted = trixc_argmax(output, 10);
        trixc_bars(100, 100, 300, 250, output, NULL, 10, predicted);
        trixc_stats(100, 380, timer.elapsed_ms, 0, trixc_fps());

        trixc_present();
    }

    trixc_shutdown();
    return 0;
}
```

---

*"It's all in the reflexes."*
