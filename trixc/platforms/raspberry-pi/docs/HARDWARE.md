# TRIXC Pi Hardware Guide

Complete hardware reference for building physical ML projects with TRIXC Pi.

> *"When some wild-eyed, eight-foot-tall maniac grabs your neck, tightens the grip, and tells you to write clean wiring diagrams... that's the right time to listen."*

---

## Table of Contents

1. [Supported Hardware](#supported-hardware)
2. [Raspberry Pi GPIO Reference](#raspberry-pi-gpio-reference)
3. [Display Options](#display-options)
4. [Input Devices](#input-devices)
5. [LED Circuits](#led-circuits)
6. [Button Circuits](#button-circuits)
7. [Sensor Integration](#sensor-integration)
8. [Motor Control](#motor-control)
9. [Power Considerations](#power-considerations)
10. [Complete Project Examples](#complete-project-examples)

---

## Supported Hardware

### Raspberry Pi Models

| Model | Status | Notes |
|-------|--------|-------|
| **Raspberry Pi 4** (2GB/4GB/8GB) | Fully Supported | Primary development platform |
| Raspberry Pi 3B+ | Supported | Slightly slower inference |
| Raspberry Pi 3B | Supported | Adequate for most models |
| Raspberry Pi Zero 2 W | Supported | Good for embedded projects |
| Raspberry Pi 5 | Expected Compatible | Faster inference expected |

### Minimum Requirements

- **RAM:** 512MB (1GB+ recommended)
- **Storage:** 4GB SD card (8GB+ recommended)
- **OS:** Raspberry Pi OS (Bookworm) or Ubuntu 22.04+
- **Display:** Any HDMI or DSI display (optional for headless)

---

## Raspberry Pi GPIO Reference

### GPIO Header Pinout (40-pin)

```
                    Raspberry Pi GPIO Header
    ┌─────────────────────────────────────────────────────┐
    │                         USB                          │
    │  ┌─────────────────────────────────────────────────┐ │
    │  │   3.3V (1) ●  ● (2) 5V                          │ │
    │  │  GPIO2 (3) ●  ● (4) 5V                          │ │
    │  │  GPIO3 (5) ●  ● (6) GND                         │ │
    │  │  GPIO4 (7) ●  ● (8) GPIO14 (TX)                 │ │
    │  │    GND (9) ●  ● (10) GPIO15 (RX)                │ │
    │  │ GPIO17 (11) ●  ● (12) GPIO18 (PWM0)             │ │
    │  │ GPIO27 (13) ●  ● (14) GND                       │ │
    │  │ GPIO22 (15) ●  ● (16) GPIO23                    │ │
    │  │   3.3V (17) ●  ● (18) GPIO24                    │ │
    │  │ GPIO10 (19) ●  ● (20) GND                       │ │
    │  │  GPIO9 (21) ●  ● (22) GPIO25                    │ │
    │  │ GPIO11 (23) ●  ● (24) GPIO8                     │ │
    │  │    GND (25) ●  ● (26) GPIO7                     │ │
    │  │ GPIO0* (27) ●  ● (28) GPIO1*                    │ │
    │  │  GPIO5 (29) ●  ● (30) GND                       │ │
    │  │  GPIO6 (31) ●  ● (32) GPIO12 (PWM0)             │ │
    │  │ GPIO13 (33) ●  ● (34) GND                       │ │
    │  │ GPIO19 (35) ●  ● (36) GPIO16                    │ │
    │  │ GPIO26 (37) ●  ● (38) GPIO20                    │ │
    │  │    GND (39) ●  ● (40) GPIO21                    │ │
    │  └─────────────────────────────────────────────────┘ │
    │                       ETHERNET                       │
    └─────────────────────────────────────────────────────┘

    * GPIO0/GPIO1 reserved for HAT EEPROM (avoid if possible)
```

### BCM vs Physical Pin Numbering

**TRIXC Pi uses BCM (Broadcom) numbering.** This is the GPIO number, not the physical pin position.

| Function | BCM GPIO | Physical Pin | Notes |
|----------|----------|--------------|-------|
| LED Green | GPIO 17 | Pin 11 | Standard output |
| LED Red | GPIO 18 | Pin 12 | Also PWM capable |
| Button A | GPIO 27 | Pin 13 | Input with pull-up |
| Button B | GPIO 22 | Pin 15 | Input with pull-up |
| PWM Output | GPIO 12 | Pin 32 | Hardware PWM |
| PWM Output | GPIO 13 | Pin 33 | Hardware PWM |

### GPIO Modes

```c
// Available modes in TRIXC Pi
#define TRIXC_GPIO_INPUT   0   // Digital input
#define TRIXC_GPIO_OUTPUT  1   // Digital output
#define TRIXC_GPIO_PWM     2   // PWM output (duty 0-255)

// Pull resistor options
#define TRIXC_GPIO_PULL_OFF   0
#define TRIXC_GPIO_PULL_DOWN  1
#define TRIXC_GPIO_PULL_UP    2
```

---

## Display Options

### Official Raspberry Pi 7" Touchscreen

The primary development display for TRIXC Pi.

**Specifications:**
- 800 × 480 resolution
- Capacitive 10-point touch
- DSI connection (no HDMI needed)
- 60 fps refresh rate

**Installation:**
```
Display Ribbon Cable Connection:

   Raspberry Pi 4                    Display Board
   ┌─────────────┐                   ┌─────────────┐
   │             │    DSI Ribbon     │             │
   │   DSI  ●────┼──────────────────┼────●  DSI   │
   │        ●────┼──────────────────┼────●        │
   │             │                   │             │
   │   5V   ●────┼───────┬──────────┼────●  5V    │
   │   GND  ●────┼───────┴──────────┼────●  GND   │
   └─────────────┘    Power Jumpers  └─────────────┘
```

### HDMI Displays

Any HDMI display works with TRIXC Pi.

**Configuration for touchscreen over HDMI:**
```bash
# /boot/config.txt (for HDMI touchscreens)
hdmi_group=2
hdmi_mode=87
hdmi_cvt=800 480 60 6 0 0 0
```

### Headless Mode

TRIXC Pi can run without a display for inference-only applications:
```c
// Skip display initialization
// Just use GPIO directly
trixc_gpio_init();
// ... run inference, control outputs ...
trixc_gpio_shutdown();
```

---

## Input Devices

### Touch Input

The built-in touch support works with capacitive touchscreens.

```c
// Check for touch
int x, y;
if (trixc_touch_pos(&x, &y)) {
    // Currently being touched at (x, y)
}

// Or use event system
trixc_event_t event;
while (trixc_poll(&event)) {
    if (event.type == TRIXC_EVENT_TOUCH_DOWN) {
        // Touch started at event.x, event.y
    }
}
```

### Keyboard Input

USB keyboards work automatically:
```c
// Check if key is held
if (trixc_key_down('w')) {
    // W key is pressed
}

// Or use events for press/release
if (event.type == TRIXC_EVENT_KEY_DOWN) {
    char pressed = event.key_char;  // Printable character
    int scancode = event.key;       // SDL scancode
}
```

### Mouse Input

Mouse works as single-touch input:
```c
// Mouse position reported as touch
if (trixc_touch_pos(&x, &y)) {
    // Mouse button held, position is (x, y)
}
```

---

## LED Circuits

### Basic LED Circuit

```
        GPIO Pin (e.g., GPIO 17)
             │
             │
           ┌─┴─┐
           │   │  220Ω Resistor
           │   │  (current limiting)
           └─┬─┘
             │
             ▼
           ┌───┐
           │ ┼ │  LED
           │   │  (long leg = anode = positive)
           └─┬─┘
             │
             ▼
            GND
```

**Component Values:**
| LED Color | Forward Voltage | Resistor (3.3V) | Current |
|-----------|-----------------|-----------------|---------|
| Red | 1.8V | 220Ω | ~7mA |
| Yellow | 2.0V | 180Ω | ~7mA |
| Green | 2.2V | 150Ω | ~7mA |
| Blue | 3.0V | 47Ω | ~6mA |
| White | 3.2V | 22Ω | ~5mA |

**Code:**
```c
#define LED_PIN 17

trixc_gpio_mode(LED_PIN, TRIXC_GPIO_OUTPUT);
trixc_gpio_write(LED_PIN, 1);  // LED ON
trixc_gpio_write(LED_PIN, 0);  // LED OFF
```

### PWM LED (Brightness Control)

```c
#define LED_PWM 18

trixc_gpio_mode(LED_PWM, TRIXC_GPIO_PWM);

// Set brightness (0-255)
trixc_gpio_pwm(LED_PWM, 128);  // 50% brightness
trixc_gpio_pwm(LED_PWM, 255);  // Full brightness
trixc_gpio_pwm(LED_PWM, 0);    // Off
```

### RGB LED (Common Cathode)

```
        GPIO 17          GPIO 18          GPIO 27
        (Red)            (Green)          (Blue)
           │                │                │
         ┌─┴─┐            ┌─┴─┐            ┌─┴─┐
         │   │ 220Ω       │   │ 220Ω       │   │ 220Ω
         └─┬─┘            └─┬─┘            └─┬─┘
           │                │                │
           └────────────────┼────────────────┘
                            │
                          ┌─┴─┐
                          │RGB│
                          │LED│
                          └─┬─┘
                            │
                           GND
                    (Common Cathode)
```

**Code:**
```c
// Set color using PWM
void set_rgb(int r, int g, int b) {
    trixc_gpio_pwm(17, r);  // Red   (0-255)
    trixc_gpio_pwm(18, g);  // Green (0-255)
    trixc_gpio_pwm(27, b);  // Blue  (0-255)
}

set_rgb(255, 0, 0);    // Red
set_rgb(0, 255, 0);    // Green
set_rgb(0, 0, 255);    // Blue
set_rgb(255, 255, 0);  // Yellow
set_rgb(255, 0, 255);  // Magenta
set_rgb(0, 255, 255);  // Cyan
set_rgb(255, 255, 255); // White
```

---

## Button Circuits

### Basic Button with Pull-Up Resistor

```
          3.3V
           │
         ┌─┴─┐
         │   │  10KΩ Pull-up
         │   │  (keeps input HIGH when not pressed)
         └─┬─┘
           │
           ├────────── GPIO Pin (e.g., GPIO 27)
           │
         ┌─┴─┐
         │   │  Tactile Button
         │   │  (momentary, normally open)
         └─┬─┘
           │
          GND
```

**How it works:**
- Button not pressed: GPIO reads HIGH (1) through pull-up
- Button pressed: GPIO reads LOW (0), connected to GND

**Code:**
```c
#define BUTTON_PIN 27

trixc_gpio_mode(BUTTON_PIN, TRIXC_GPIO_INPUT);

// Read button (inverted logic with pull-up)
int raw = trixc_gpio_read(BUTTON_PIN);
int pressed = !raw;  // Invert: LOW = pressed
```

### Using Internal Pull-Up

Raspberry Pi has internal pull-up resistors. With pigpio, these are configured automatically for inputs.

```c
// Using internal pull-up (no external resistor needed)
trixc_gpio_mode(BUTTON_PIN, TRIXC_GPIO_INPUT);
// pigpio enables internal pull-up by default for input mode
```

### Debouncing

Mechanical buttons "bounce" when pressed, causing multiple triggers. Software debouncing:

```c
#define DEBOUNCE_MS 50

static int last_state = 0;
static uint64_t last_change = 0;

int read_button_debounced(int pin) {
    int current = trixc_gpio_read(pin);
    uint64_t now = trixc_time_us();

    if (current != last_state) {
        if ((now - last_change) > DEBOUNCE_MS * 1000) {
            last_state = current;
            last_change = now;
            return current;
        }
    }
    return last_state;
}
```

---

## Sensor Integration

### Analog Sensors via ADC

Raspberry Pi doesn't have built-in ADC. Use an external ADC like MCP3008:

```
                      MCP3008 ADC
                   ┌──────────────┐
     Sensor ──────▶│ CH0      VDD │──── 3.3V
                   │ CH1      VREF│──── 3.3V
                   │ CH2      AGND│──── GND
                   │ CH3      CLK │──── GPIO 11 (SPI CLK)
                   │ CH4      DOUT│──── GPIO 9  (SPI MISO)
                   │ CH5      DIN │──── GPIO 10 (SPI MOSI)
                   │ CH6      CS  │──── GPIO 8  (SPI CE0)
                   │ CH7      DGND│──── GND
                   └──────────────┘
```

### I2C Sensors

Many sensors use I2C. Common address examples:
- **BMP280** (pressure/temp): 0x76 or 0x77
- **MPU6050** (accelerometer): 0x68
- **OLED displays**: 0x3C

**I2C Connections:**
```
   Raspberry Pi          Sensor
   ┌──────────┐         ┌──────────┐
   │  GPIO 2  │─────────│   SDA    │
   │  GPIO 3  │─────────│   SCL    │
   │   3.3V   │─────────│   VCC    │
   │   GND    │─────────│   GND    │
   └──────────┘         └──────────┘
```

### Temperature Sensor (DS18B20)

Digital 1-Wire temperature sensor:
```
        3.3V
         │
       ┌─┴─┐
       │   │  4.7KΩ
       └─┬─┘
         │
         ├────── GPIO 4 (1-Wire default)
         │
      ┌──┴──┐
      │     │  DS18B20
      │ VCC │──── 3.3V
      │ DQ  │──── GPIO 4
      │ GND │──── GND
      └─────┘
```

---

## Motor Control

### Servo Motor (PWM)

Standard hobby servos (SG90, MG996R):

```
                    Servo Motor
        GPIO 12 ───────────────┬───── Signal (Orange/White)
        (PWM)                  │
                               │
          5V ──────────────────┼───── Power (Red)
                               │
         GND ──────────────────┴───── Ground (Brown/Black)
```

**PWM Frequency:** 50 Hz (20ms period)
**Pulse Width:**
- 1ms = 0° (duty ~5%)
- 1.5ms = 90° (duty ~7.5%)
- 2ms = 180° (duty ~10%)

```c
// Servo control (approximate, requires timing calibration)
void set_servo_angle(int pin, int angle) {
    // Map 0-180° to duty cycle 25-125 (out of 255)
    int duty = 25 + (angle * 100 / 180);
    trixc_gpio_pwm(pin, duty);
}
```

### DC Motor (H-Bridge)

Use an H-bridge driver (L298N, TB6612FNG) for DC motors:

```
   Raspberry Pi              L298N H-Bridge              Motor
   ┌──────────┐              ┌─────────────┐          ┌────────┐
   │  GPIO 17 │──────────────│ IN1         │          │        │
   │  GPIO 18 │──────────────│ IN2     OUT1│──────────│  Motor │
   │  GPIO 12 │──────────────│ ENA     OUT2│──────────│        │
   │   (PWM)  │              │             │          └────────┘
   │          │              │         +12V│──── Motor Power
   │    GND   │──────────────│   GND   GND │──── GND
   └──────────┘              └─────────────┘
```

**Code:**
```c
#define MOTOR_IN1  17
#define MOTOR_IN2  18
#define MOTOR_PWM  12

void motor_init() {
    trixc_gpio_mode(MOTOR_IN1, TRIXC_GPIO_OUTPUT);
    trixc_gpio_mode(MOTOR_IN2, TRIXC_GPIO_OUTPUT);
    trixc_gpio_mode(MOTOR_PWM, TRIXC_GPIO_PWM);
}

void motor_forward(int speed) {  // speed 0-255
    trixc_gpio_write(MOTOR_IN1, 1);
    trixc_gpio_write(MOTOR_IN2, 0);
    trixc_gpio_pwm(MOTOR_PWM, speed);
}

void motor_reverse(int speed) {
    trixc_gpio_write(MOTOR_IN1, 0);
    trixc_gpio_write(MOTOR_IN2, 1);
    trixc_gpio_pwm(MOTOR_PWM, speed);
}

void motor_stop() {
    trixc_gpio_write(MOTOR_IN1, 0);
    trixc_gpio_write(MOTOR_IN2, 0);
    trixc_gpio_pwm(MOTOR_PWM, 0);
}
```

---

## Power Considerations

### GPIO Current Limits

| Limit | Value | Notes |
|-------|-------|-------|
| Per GPIO pin | 16mA max | Safe limit, brief 50mA absolute max |
| Total GPIO | 50mA max | Sum of all GPIO outputs |
| 3.3V rail | 50mA max | For external circuits |
| 5V rail | 1.5A max | Shared with USB ports |

### Power Supply Requirements

| Configuration | Minimum PSU | Recommended |
|---------------|-------------|-------------|
| Pi 4 alone | 3.0A | 3.0A official PSU |
| Pi 4 + touchscreen | 3.0A | 3.0A official PSU |
| Pi 4 + touchscreen + sensors | 3.0A | 3.5A+ |
| Pi 4 + motors | External motor power | Separate 5V/12V supply |

### Separate Power for Motors

**Never power motors from Pi GPIO!** Use external power:

```
   Pi GPIO ────────── H-Bridge Logic

   External 5V ────── H-Bridge Motor Power
   or 12V

   IMPORTANT: Connect all GNDs together!

   Pi GND ──┬── H-Bridge GND
            └── External Power GND
```

---

## Complete Project Examples

### XOR LED Controller

From `examples/03_gpio_sensor/`:

```
Complete Wiring:

                    Raspberry Pi 4
                ┌─────────────────────┐
                │  3.3V    5V         │
                │   ●───┐   ●         │
                │       │             │
                │  GND  │  GND        │
                │   ●   │   ●         │
                │   │   │   │         │
                │   │   │   │         │
                │  17  18  27  22     │  (BCM GPIO numbers)
                │   ●   ●   ●   ●     │
                └───┼───┼───┼───┼─────┘
                    │   │   │   │
                    │   │   │   │
   ┌────────────────┘   │   │   └────────────────┐
   │                    │   │                    │
   ▼                    ▼   │                    ▼
┌───────┐           ┌───────┐                ┌───────┐
│ 220Ω  │           │ 220Ω  │                │ 10KΩ  │
└───┬───┘           └───┬───┘                └───┬───┘
    │                   │                        │
    ▼                   ▼                        ├──── 3.3V
┌───────┐           ┌───────┐                    │
│ GREEN │           │  RED  │                ┌───┴───┐
│  LED  │           │  LED  │                │BUTTON │──── GPIO 27
└───┬───┘           └───┬───┘                │   A   │
    │                   │                    └───┬───┘
    │                   │                        │
    └────────┬──────────┘                        │
             │                                   ▼
             ▼                                  GND
            GND

                                  (Same for Button B on GPIO 22)
```

### Parts List:
- 2× LED (green and red)
- 2× 220Ω resistor
- 2× Tactile button
- 2× 10KΩ resistor
- Breadboard
- Jumper wires

### MNIST Digit Classifier

Minimal hardware for the MNIST example:
- Raspberry Pi 4
- 7" Official Touchscreen
- (No GPIO needed - touch input only)

### Camera Classifier (Optional)

```
   Raspberry Pi 4
   ┌─────────────────┐
   │                 │
   │   CSI Camera    │
   │     Port        │
   │       ●         │
   └───────┼─────────┘
           │
           │  CSI Ribbon Cable
           │  (15-pin)
           │
   ┌───────┼─────────┐
   │       ●         │
   │   Pi Camera V2  │
   │   or HQ Camera  │
   └─────────────────┘
```

---

## Safety Guidelines

### Electrical Safety

1. **Never exceed 3.3V on GPIO pins**
   - 5V will damage the GPIO permanently

2. **Always use current-limiting resistors with LEDs**
   - Without resistor, LED + GPIO can burn out

3. **Don't connect motors directly to GPIO**
   - Use motor drivers with external power

4. **Measure before connecting**
   - Use multimeter to verify voltages

### Static Electricity

1. Ground yourself before handling the Pi
2. Handle boards by edges
3. Store in anti-static bags

### Heat Management

1. Use heatsinks for intensive inference
2. Consider active cooling for sustained load
3. Monitor with `vcgencmd measure_temp`

---

## Troubleshooting

### GPIO Not Responding

```bash
# Check pigpio is installed
dpkg -l | grep pigpio

# Make sure pigpiod is NOT running (TRIXC uses direct access)
sudo killall pigpiod

# Run with sudo (required for GPIO)
sudo ./your_program
```

### Wrong Pin Responding

```bash
# Verify you're using BCM numbering, not physical pin numbers!
# GPIO 17 = Physical Pin 11 (not Pin 17!)

# Use pinout command to visualize
pinout
```

### LED Not Lighting

1. Check polarity (long leg = positive = anode)
2. Verify resistor connection
3. Test LED with 3.3V and resistor directly
4. Measure GPIO output with multimeter

### Button Always Reading Same Value

1. Check pull-up resistor connection
2. Verify button is normally-open type
3. Test button with multimeter (continuity when pressed)
4. Check for cold solder joints

---

## Resources

### Documentation
- [Raspberry Pi GPIO Documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html)
- [pigpio Library](http://abyz.me.uk/rpi/pigpio/)
- [Raspberry Pi Pinout](https://pinout.xyz/)

### Tools
- `pinout` - Terminal GPIO reference
- `gpio readall` - Show all GPIO states
- `vcgencmd` - System monitoring

### Where to Buy
- Official Raspberry Pi resellers for reliable components
- Adafruit, SparkFun for quality sensors and modules
- Local electronics stores for basic components

---

*"It's all in the reflexes."*

From frozen shapes to physical hardware. That's TRIXC Pi.
