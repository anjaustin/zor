# TRIXC Pi Architecture

Internal design and implementation details of the TRIXC Pi platform.

> *"It's not magic. It's just really good engineering."*

---

## Overview

TRIXC Pi is a minimal runtime that connects frozen shape models to physical hardware. It provides just enough abstraction to make ML deployment easy while staying small enough to understand completely.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           TRIXC Pi Architecture                          │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                         User Application                           │ │
│  │                                                                    │ │
│  │   main.c → model.c (frozen shapes) → trixc_pi.h (runtime API)     │ │
│  │                                                                    │ │
│  └─────────────────────────────┬──────────────────────────────────────┘ │
│                                │                                         │
│  ┌─────────────────────────────┴──────────────────────────────────────┐ │
│  │                        TRIXC Pi Runtime                            │ │
│  │                                                                    │ │
│  │   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │ │
│  │   │  Display   │  │   Input    │  │   Timing   │  │    GPIO    │  │ │
│  │   │            │  │            │  │            │  │            │  │ │
│  │   │ - Clear    │  │ - Touch    │  │ - Timer    │  │ - Read     │  │ │
│  │   │ - Rect     │  │ - Keyboard │  │ - FPS      │  │ - Write    │  │ │
│  │   │ - Circle   │  │ - Mouse    │  │ - Sleep    │  │ - PWM      │  │ │
│  │   │ - Text     │  │            │  │            │  │            │  │ │
│  │   │ - Heatmap  │  │            │  │            │  │            │  │ │
│  │   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  │ │
│  │         │               │               │               │          │ │
│  └─────────┼───────────────┼───────────────┼───────────────┼──────────┘ │
│            │               │               │               │            │
│  ┌─────────┴───────────────┴───────────────┴───────────────┴──────────┐ │
│  │                      System Libraries                              │ │
│  │                                                                    │ │
│  │   SDL2 (display, input)   │   pigpio (GPIO)   │   libc (timing)   │ │
│  │                                                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                         Hardware                                   │ │
│  │                                                                    │ │
│  │   Framebuffer   │   Touch/Keyboard   │   CPU Timer   │   GPIO     │ │
│  │                                                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Design Principles

### 1. Single Header API

All public API is defined in `trixc_pi.h`. Users include one header and get everything.

```c
#include "trixc_pi.h"  // That's it.
```

### 2. Zero Global State (Almost)

State is contained in a single static context structure. Functions access this internally.

```c
// Internal: Single global context
static struct {
    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_Texture* texture;
    uint32_t* pixels;
    int width, height;
    bool running;
    // ... timing, input state
} ctx;

// External: No visible state
trixc_init("App", 800, 480);  // Configures internal context
trixc_clear(0x000000);        // Uses internal context
```

### 3. Models Are Just C Code

Models are compiled directly into the binary. No runtime loading.

```c
// models/xor_mlp.c - This is the entire model
static const float weights_h[8] = {...};
static const float bias_h[4] = {...};
static const float weights_o[4] = {...};
static const float bias_o[1] = {...};

static float xor_forward(float x0, float x1) {
    // 2 → 4 → 1 MLP with ReLU
    float input[2] = {x0, x1};
    float hidden[4], output[1];

    // Layer 1: hidden = relu(input @ weights_h + bias_h)
    for (int i = 0; i < 4; i++) {
        float sum = bias_h[i];
        for (int j = 0; j < 2; j++) {
            sum += input[j] * weights_h[j * 4 + i];
        }
        hidden[i] = sum > 0 ? sum : 0;  // ReLU
    }

    // Layer 2: output = hidden @ weights_o + bias_o
    output[0] = bias_o[0];
    for (int i = 0; i < 4; i++) {
        output[0] += hidden[i] * weights_o[i];
    }

    return output[0];
}
```

### 4. Conditional Compilation for Features

Optional features (GPIO, Camera) are gated by preprocessor defines:

```c
#ifdef TRIXC_PI_GPIO
int trixc_gpio_init(void) {
    return gpioInitialise();
}
#endif

#ifdef TRIXC_PI_CAMERA
int trixc_camera_init(int w, int h) {
    // libcamera setup
}
#endif
```

---

## Module Breakdown

### Display System

The display system uses SDL2 with a software rendering pipeline for maximum compatibility.

```
┌─────────────────────────────────────────────────────────────────┐
│                       Display Pipeline                           │
│                                                                  │
│   trixc_clear() ──┐                                              │
│   trixc_rect()  ──┼──▶ pixels[] ──▶ SDL_Texture ──▶ Renderer    │
│   trixc_text()  ──┤        ↑              ↑              │       │
│   trixc_line()  ──┘        │              │              ▼       │
│                     trixc_pixel()   trixc_present()   Screen     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key implementation details:**

```c
// Pixel buffer is uint32_t ARGB format
uint32_t* pixels;  // [width * height]

// All drawing operates on pixel buffer
void trixc_pixel(int x, int y, uint32_t color) {
    if (x >= 0 && x < ctx.width && y >= 0 && y < ctx.height) {
        ctx.pixels[y * ctx.width + x] = color | 0xFF000000;
    }
}

// Present uploads buffer to GPU texture
void trixc_present(void) {
    SDL_UpdateTexture(ctx.texture, NULL, ctx.pixels, ctx.width * 4);
    SDL_RenderCopy(ctx.renderer, ctx.texture, NULL, NULL);
    SDL_RenderPresent(ctx.renderer);
    // Update FPS, process events...
}
```

### Text Rendering

Built-in 8×8 bitmap font covering ASCII 32-126 (95 characters).

```
Font Storage (bit-packed):
┌────────────────────────────────────────────────────────────┐
│  Each character = 8 bytes (64 bits = 8×8 pixels)           │
│                                                            │
│  'A' (ASCII 65):                                           │
│      0x18 = ....##..   Row 0                               │
│      0x3C = ..####..   Row 1                               │
│      0x66 = .##..##.   Row 2                               │
│      0x7E = .######.   Row 3                               │
│      0x66 = .##..##.   Row 4                               │
│      0x66 = .##..##.   Row 5                               │
│      0x66 = .##..##.   Row 6                               │
│      0x00 = ........   Row 7                               │
│                                                            │
│  Total font data: 95 × 8 = 760 bytes                       │
└────────────────────────────────────────────────────────────┘
```

**Rendering algorithm:**

```c
void trixc_text(int x, int y, const char* text, uint32_t color) {
    while (*text) {
        int idx = *text - 32;  // ASCII offset
        if (idx >= 0 && idx < 95) {
            for (int row = 0; row < 8; row++) {
                uint8_t bits = font_data[idx * 8 + row];
                for (int col = 0; col < 8; col++) {
                    if (bits & (0x80 >> col)) {
                        trixc_pixel(x + col, y + row, color);
                    }
                }
            }
        }
        x += 8;  // Monospace
        text++;
    }
}
```

### Input System

Unified input abstraction over touch, mouse, and keyboard.

```
┌─────────────────────────────────────────────────────────────────┐
│                       Input System                               │
│                                                                  │
│   SDL Events                                                     │
│       │                                                          │
│       ├── SDL_FINGERDOWN ──┐                                     │
│       ├── SDL_FINGERUP    ──┼──▶ TRIXC_EVENT_TOUCH_*            │
│       ├── SDL_FINGERMOTION ┘                                     │
│       │                                                          │
│       ├── SDL_MOUSEBUTTONDOWN ──┐                                │
│       ├── SDL_MOUSEBUTTONUP   ──┼──▶ TRIXC_EVENT_TOUCH_*        │
│       ├── SDL_MOUSEMOTION     ──┘    (mapped to touch)          │
│       │                                                          │
│       ├── SDL_KEYDOWN ─────────────▶ TRIXC_EVENT_KEY_DOWN       │
│       └── SDL_KEYUP   ─────────────▶ TRIXC_EVENT_KEY_UP         │
│                                                                  │
│   Output: Unified trixc_event_t                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Event queue implementation:**

```c
#define MAX_EVENTS 64

static struct {
    trixc_event_t queue[MAX_EVENTS];
    int head, tail;
} events;

bool trixc_poll(trixc_event_t* event) {
    if (events.head == events.tail) return false;
    *event = events.queue[events.head];
    events.head = (events.head + 1) % MAX_EVENTS;
    return true;
}
```

### Timing System

High-resolution timing using platform-specific APIs.

```c
// Primary timing source
uint64_t trixc_time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

// Timer with statistics
typedef struct {
    uint64_t start_us, end_us;
    double elapsed_ms;
    double min_ms, max_ms, avg_ms;
    int count;
} trixc_timer_t;

void trixc_timer_stop(trixc_timer_t* t) {
    t->end_us = trixc_time_us();
    t->elapsed_ms = (t->end_us - t->start_us) / 1000.0;

    // Update statistics
    if (t->count == 0 || t->elapsed_ms < t->min_ms) t->min_ms = t->elapsed_ms;
    if (t->elapsed_ms > t->max_ms) t->max_ms = t->elapsed_ms;

    // Running average
    t->avg_ms = (t->avg_ms * t->count + t->elapsed_ms) / (t->count + 1);
    t->count++;
}
```

### GPIO System (Optional)

GPIO control via the pigpio library.

```
┌─────────────────────────────────────────────────────────────────┐
│                       GPIO System                                │
│                                                                  │
│   trixc_gpio_* functions                                         │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────────┐                                           │
│   │     pigpio      │  (when TRIXC_PI_GPIO defined)             │
│   │   Library       │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │   /dev/mem      │  (requires root)                          │
│   │   BCM2835       │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   Physical GPIO Pins (BCM numbering)                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**GPIO API:**

```c
// Mode configuration
void trixc_gpio_mode(int pin, int mode) {
    switch (mode) {
        case TRIXC_GPIO_INPUT:
            gpioSetMode(pin, PI_INPUT);
            gpioSetPullUpDown(pin, PI_PUD_UP);  // Enable pull-up
            break;
        case TRIXC_GPIO_OUTPUT:
            gpioSetMode(pin, PI_OUTPUT);
            break;
        case TRIXC_GPIO_PWM:
            gpioSetMode(pin, PI_OUTPUT);
            gpioSetPWMfrequency(pin, 1000);  // 1 kHz default
            break;
    }
}

// Digital I/O
void trixc_gpio_write(int pin, int value) {
    gpioWrite(pin, value);
}

int trixc_gpio_read(int pin) {
    return gpioRead(pin);
}

// PWM output
void trixc_gpio_pwm(int pin, int duty) {
    gpioPWM(pin, duty);  // 0-255
}
```

---

## Memory Layout

### Runtime Memory Usage

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRIXC Pi Memory Layout                        │
│                                                                  │
│   Code (.text)                                                   │
│   ├── trixc_pi.c functions        ~15 KB                         │
│   ├── Font bitmap data            ~760 B                         │
│   └── Colormap tables             ~3 KB                          │
│                                                                  │
│   Read-only Data (.rodata)                                       │
│   ├── Model weights               varies (e.g., 6.5 KB for MNIST)│
│   └── String literals             ~2 KB                          │
│                                                                  │
│   Heap (malloc)                                                  │
│   └── Pixel buffer                width × height × 4 bytes       │
│       (800×480 = 1.5 MB)                                         │
│                                                                  │
│   Stack                                                          │
│   └── Local variables             ~8 KB typical                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Model Memory

Models store weights in `.rodata` section (read-only, flash-friendly):

```c
// Weights declared as static const → .rodata
static const float weights[1568] = {
    // ... 6.5 KB for MNIST model
};

// Inference uses stack for activations
static void mnist_forward(const float* input, float* output) {
    float hidden[32];  // Stack allocation (128 bytes)
    // ... compute
}
```

---

## Build System

### Makefile Structure

```makefile
# Main targets
hello:     Build 01_hello_xor
mnist:     Build 02_mnist_draw
gpio:      Build 03_gpio_sensor (requires pigpio)

# Feature flags
-DTRIXC_PI_GPIO      Enable GPIO support
-DTRIXC_PI_CAMERA    Enable camera support

# Dependencies
SDL2:      Always required
pigpio:    Required for GPIO
libcamera: Required for camera
```

### Compilation Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    Build Process                              │
│                                                               │
│   Source Files:                                               │
│   ├── examples/01_hello_xor/main.c                           │
│   ├── src/trixc_pi.c                                         │
│   └── models/xor_mlp.c (included in main.c)                  │
│           │                                                   │
│           ▼                                                   │
│   ┌───────────────┐                                          │
│   │      GCC      │  -O2 -Wall                               │
│   │               │  -I../../include                         │
│   │               │  $(sdl2-config --cflags --libs)          │
│   └───────┬───────┘                                          │
│           │                                                   │
│           ▼                                                   │
│   hello_xor (single executable, ~100 KB)                     │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Performance Characteristics

### Timing Benchmarks (Raspberry Pi 4)

| Operation | Time | Notes |
|-----------|------|-------|
| `trixc_clear()` | ~0.3 ms | Full 800×480 buffer clear |
| `trixc_rect(100×100)` | ~0.01 ms | Small rectangle |
| `trixc_text(32 chars)` | ~0.02 ms | Text rendering |
| `trixc_heatmap(8×8)` | ~0.05 ms | With colormap lookup |
| `trixc_present()` | ~1.5 ms | Texture upload + render |
| XOR inference | ~0.003 ms | 2→4→1 MLP |
| MNIST inference | ~0.08 ms | 49→32→10 MLP |

### Optimization Opportunities

1. **Rectangle filling** - Could use memset for horizontal lines
2. **Text rendering** - Could cache scaled fonts
3. **Heatmap** - Could use lookup tables for colormap
4. **Present** - Could use dirty rectangles

These optimizations are deliberately NOT implemented. The current performance is sufficient, and simplicity is prioritized.

---

## Extension Points

### Adding a New Visualization

```c
// In trixc_pi.c

void trixc_sparkline(int x, int y, int w, int h,
                     const float* values, int count, uint32_t color) {
    // Find min/max
    float min = values[0], max = values[0];
    for (int i = 1; i < count; i++) {
        if (values[i] < min) min = values[i];
        if (values[i] > max) max = values[i];
    }

    // Draw sparkline
    float range = max - min;
    if (range < 0.0001f) range = 1.0f;

    for (int i = 0; i < count - 1 && i < w; i++) {
        int x1 = x + i * w / count;
        int x2 = x + (i + 1) * w / count;
        int y1 = y + h - (int)((values[i] - min) / range * h);
        int y2 = y + h - (int)((values[i+1] - min) / range * h);
        trixc_line(x1, y1, x2, y2, color);
    }
}
```

### Adding a New Model

1. Train in PyTorch/TensorFlow
2. Export to ONNX
3. Convert with `onnx2trix.py --emit-c`
4. Include in your application

```c
// Your application
#include "trixc_pi.h"
#include "../../models/your_model.c"

int main() {
    trixc_init("Your App", 800, 480);

    float input[YOUR_INPUT_SIZE];
    float output[YOUR_OUTPUT_SIZE];

    while (trixc_running()) {
        // Get input...
        your_model_forward(input, output);
        // Display output...
        trixc_present();
    }

    trixc_shutdown();
}
```

---

## Thread Safety

TRIXC Pi is **NOT thread-safe**. All functions must be called from the main thread.

For multi-threaded applications:
- Run inference in worker threads
- Queue results for main thread
- Main thread handles all TRIXC Pi calls

---

## Error Handling

TRIXC Pi uses a simple error model:

- Functions that can fail return `int` (0 = success, -1 = failure)
- Errors are logged to `stderr`
- No exceptions, no error codes, no errno

```c
if (trixc_init("App", 800, 480) != 0) {
    fprintf(stderr, "Failed to initialize\n");
    return 1;
}
```

---

## Future Directions

### Potential Extensions

1. **Vulkan backend** - For GPU compute on Pi 4
2. **Audio output** - For accessibility and feedback
3. **Network** - For distributed inference
4. **Video recording** - Capture demos

### What We Won't Add

1. **Widget library** - Not a GUI framework
2. **3D graphics** - Not a game engine
3. **Text layout** - Not a document processor

TRIXC Pi is for ML deployment. It does that well. Everything else is out of scope.

---

*"Shapes are opcodes. SDL is the framebuffer. GPIO is the I/O bus. It's a 6502 for ML."*
