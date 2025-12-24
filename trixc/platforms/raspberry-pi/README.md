# TRIXC Pi

> *"When some wild-eyed, eight-foot-tall maniac grabs your neck, taps the back of your favorite head up against the barroom wall, and he looks you crooked in the eye and he asks you if ya paid your dues, you just stare that big sucker right back in the eye, and you remember what ol' Jack Burton always says at a time like this:*
>
> *Have ya paid your dues, Jack?*
>
> *Yes sir, the check is in the mail."*

**Machine learning on Raspberry Pi. No PyTorch. No TensorFlow. Just C.**

---

## What's in the Six Demon Bag?

| Feature | TRIXC Pi | PyTorch |
|---------|----------|---------|
| Binary size | ~70 KB | 2 GB+ |
| Startup time | Instant | Seconds |
| Dependencies | SDL2, libc | CUDA, cuDNN, Python, ... |
| Inference time | Microseconds | Milliseconds |
| Power usage | Minimal | Significant |
| Understanding | Read the code | Good luck |

---

## Quick Start

```bash
# Clone TRIXC (if you haven't)
git clone https://github.com/.../trixc
cd trixc/platforms/raspberry-pi

# Setup (one time)
./scripts/setup_pi.sh

# Build and run!
make hello
```

Tap the screen. Watch XOR compute. Feel the power.

---

## Examples

| # | Example | What It Does | Build |
|---|---------|--------------|-------|
| 01 | `hello_xor` | XOR neural network visualization | `make hello` |
| 02 | `mnist_draw` | Draw digits, see them recognized | `make mnist` |
| 03 | `gpio_sensor` | Model controls LEDs via GPIO | `make gpio` |
| 04 | `camera_classify` | Point camera, see classification | `make camera` |

### Hello XOR (Example 01)

```
┌────────────────────────────────────────────────────────────────┐
│                    TRIXC Pi - Hello XOR                        │
│                                                                │
│  Neural Network              Frozen Shape                      │
│  [0, 1] -> Hidden -> 0.9987  XOR(a, b) = a + b - 2ab           │
│                              = 0 + 1 - 0 = 1                   │
│                                                                │
│  Tap to cycle inputs         Both compute XOR!                 │
│                              One learned. One derived.         │
│                                                                │
│  Inference: 0.003ms | Memory: 52 bytes                        │
└────────────────────────────────────────────────────────────────┘
```

### MNIST Draw (Example 02)

```
┌────────────────────────────────────────────────────────────────┐
│                    TRIXC Pi - MNIST Draw                       │
│                                                                │
│  ┌───────────────┐           Prediction: 7                    │
│  │   ▓▓          │           87.3% confidence                 │
│  │   ▓▓          │                                             │
│  │   ▓▓▓▓▓▓▓     │           Class Probabilities:             │
│  │       ▓▓      │           0 ████ 12%                       │
│  │       ▓▓      │           7 ████████████████████ 87%       │
│  └───────────────┘                                             │
│   Draw a digit!              Inference: 0.08ms                │
└────────────────────────────────────────────────────────────────┘
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Your Application                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Models    │  │   Runtime   │  │       Hardware          │ │
│  │  (Pure C)   │  │   (SDL2)    │  │   (Pi-specific)         │ │
│  │             │  │             │  │                         │ │
│  │ xor_mlp.c   │  │ Display     │  │ GPIO (pigpio)          │ │
│  │ mnist_7x7.c │  │ Touch Input │  │ Camera (libcamera)     │ │
│  │ your_model.c│  │ Timing      │  │ Audio (ALSA)           │ │
│  │             │  │ Visualization│ │                         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## The API

```c
#include "trixc_pi.h"

int main() {
    // Initialize
    trixc_init("My App", 800, 480);

    // Main loop
    while (trixc_running()) {
        // Handle input
        trixc_event_t event;
        while (trixc_poll(&event)) {
            if (event.type == TRIXC_EVENT_TOUCH_DOWN) {
                // User touched at (event.x, event.y)
            }
        }

        // Draw
        trixc_clear(TRIXC_BLACK);
        trixc_text(100, 100, "Hello TRIXC Pi!", TRIXC_WHITE);

        // Run your model
        float output = your_model_forward(input);
        trixc_stats(100, 200, timer.elapsed_ms, model_size, trixc_fps());

        // Present
        trixc_present();
    }

    trixc_shutdown();
    return 0;
}
```

See [include/trixc_pi.h](include/trixc_pi.h) for the full API.

---

## Adding Your Own Models

1. **Train in PyTorch/TensorFlow**
2. **Export to ONNX**
3. **Convert with TRIXC**

```bash
# Convert your model
python ../../tools/onnx2trix.py my_model.onnx my_model.c --emit-c

# Copy to models directory
cp my_model.c models/

# Use in your example
#include "../../models/my_model.c"
```

See [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) for the full guide.

---

## Hardware Integration

### GPIO (Example 03)

```c
#define LED_PIN 17
#define BUTTON_PIN 27

trixc_gpio_init();
trixc_gpio_mode(LED_PIN, TRIXC_GPIO_OUTPUT);
trixc_gpio_mode(BUTTON_PIN, TRIXC_GPIO_INPUT);

// In your loop:
if (trixc_gpio_read(BUTTON_PIN)) {
    float output = model_forward(input);
    trixc_led(LED_PIN, output > 0.5f);
}
```

### Camera (Example 04)

```c
trixc_frame_t frame;
trixc_camera_init(640, 480);
trixc_frame_alloc(&frame, 7, 7);  // Downscale to model input size

// In your loop:
trixc_camera_capture(&frame);
trixc_frame_grayscale(&frame, model_input, true);
int class = model_predict(model_input);
```

---

## Directory Structure

```
platforms/raspberry-pi/
├── include/
│   └── trixc_pi.h          # Main header (the API)
├── src/
│   └── trixc_pi.c          # Runtime implementation
├── models/
│   ├── xor_mlp.c           # XOR neural network
│   ├── mnist_7x7.c         # Digit classifier
│   └── README.md           # How to add models
├── examples/
│   ├── 01_hello_xor/       # Your first program
│   ├── 02_mnist_draw/      # Touch digit classifier
│   ├── 03_gpio_sensor/     # Hardware integration
│   └── 04_camera_classify/ # Camera inference
├── scripts/
│   └── setup_pi.sh         # One-time setup
├── docs/
│   ├── QUICKSTART.md       # 5-minute guide
│   ├── API.md              # Full API reference
│   └── ADDING_MODELS.md    # Model conversion guide
├── Makefile                # Build everything
└── README.md               # You are here
```

---

## Requirements

### Hardware
- Raspberry Pi 4 (8GB recommended, 4GB works)
- Touchscreen (official 7" or compatible)
- Optional: GPIO components (LEDs, buttons, sensors)
- Optional: Pi Camera or USB webcam

### Software
- Ubuntu (or Raspberry Pi OS)
- SDL2 (`apt install libsdl2-dev`)
- GCC (`apt install build-essential`)
- Optional: pigpio for GPIO (`apt install pigpio`)
- Optional: Python + ONNX for model conversion

---

## Performance

On Raspberry Pi 4 (8GB, Ubuntu 22.04):

| Operation | Time |
|-----------|------|
| XOR inference | 0.003 ms |
| MNIST 7x7 inference | 0.08 ms |
| Full frame render | 16.6 ms (60 FPS) |
| Touch response | <1 ms |

Compare to PyTorch on the same hardware:
- Import time: 3-5 seconds
- First inference: 100+ ms
- Memory usage: 500+ MB

---

## Philosophy

### 1. Visible Computing

Like the 6502 that powered the Apple II and Commodore 64, TRIXC Pi makes computation visible. You can see what the neural network is doing. No black boxes.

### 2. Minimal Dependencies

SDL2 and libc. That's it. No Python runtime, no CUDA, no heavyweight frameworks. Your binary runs anywhere with a C compiler.

### 3. Size Matters

Every kilobyte counts on embedded systems. A 70KB binary that does digit recognition is more useful than a 2GB framework that does the same thing.

### 4. Education First

TRIXC Pi is for learning. The code is readable. The examples are documented. The math is exposed. Understanding beats magic.

---

## Troubleshooting

### "SDL2 not found"
```bash
./scripts/setup_pi.sh
# Or manually:
sudo apt install libsdl2-dev
```

### "Display not working"
Make sure you're running on the console, not over SSH. SDL2 needs a display.
```bash
# If using SSH with X forwarding:
export DISPLAY=:0
```

### "GPIO permission denied"
GPIO requires root access:
```bash
sudo ./build/gpio_sensor
```

### "Model accuracy is low"
- Draw digits that fill most of the canvas
- Center your digits
- Use thick strokes
- The 7x7 model is intentionally simple

---

## Contributing

- Add support for more ONNX operations
- Optimize for ARM NEON
- Improve the GPIO/Camera integration
- Write more examples
- Fix bugs and improve documentation

---

## Credits

**Created by:**
- **Tripp** - Vision, guidance, hardware testing
- **Claude** (Anthropic) - Implementation, documentation

**Built with:**
- TRIXC - The frozen shape compiler
- SDL2 - Cross-platform media library
- Raspberry Pi - The hobbyist's computer

---

## License

MIT License. Use freely. Attribution appreciated.

---

## The Last Word

> *"It's all in the reflexes."*

Machine learning doesn't have to be complicated. It doesn't have to require a GPU. It doesn't have to be a black box.

TRIXC Pi shows that a neural network is just math. Frozen shapes. Polynomials. Code you can read and understand.

Now go build something.

---

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   "Everybody relax. I'm here."                                 │
│                                  — Jack Burton                  │
│                                                                 │
│   Welcome to TRIXC Pi.                                         │
│   Your Raspberry Pi just got a lot smarter.                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
