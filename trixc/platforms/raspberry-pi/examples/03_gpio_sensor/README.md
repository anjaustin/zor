# GPIO Sensor

**Neural network controlling real hardware.**

This is where computation meets the physical world. Press buttons, watch LEDs respond based on what the neural network decides.

---

## What You'll See

```
┌────────────────────────────────────────────────────────────────────┐
│                   TRIXC Pi - GPIO Sensor                           │
│                                                                    │
│  INPUTS              Neural Net              OUTPUTS               │
│  ┌──────────┐       ┌──────────────┐        ┌──────────┐          │
│  │ Button A │──┐    │              │    ┌──▶│  GREEN   │          │
│  │ released │  ├───▶│  XOR(A, B)   │────┤   │   LED    │          │
│  └──────────┘  │    │              │    │   └──────────┘          │
│  ┌──────────┐  │    │ Output: 1.0  │    │   ┌──────────┐          │
│  │ Button B │──┘    │ = 1 (GREEN)  │    └──▶│   RED    │          │
│  │ PRESSED  │       └──────────────┘        │   LED    │          │
│  └──────────┘                               └──────────┘          │
│                                                                    │
│  XOR Truth Table          How It Works                            │
│  A   B   XOR   LED        1. Press buttons A and B                │
│  0   0    0    RED        2. Neural net computes XOR              │
│  0   1    1    GREEN  <-  3. LED shows the result                 │
│  1   0    1    GREEN                                              │
│  1   1    0    RED        Inference: 0.003ms                      │
└────────────────────────────────────────────────────────────────────┘
```

---

## Hardware Setup

### Components Needed

| Component | Quantity | Notes |
|-----------|----------|-------|
| LED (green) | 1 | Any standard 5mm LED |
| LED (red) | 1 | Any standard 5mm LED |
| 220Ω resistor | 2 | For LED current limiting |
| Push button | 2 | Momentary tactile switch |
| 10KΩ resistor | 2 | Pull-up for buttons |
| Breadboard | 1 | Any size |
| Jumper wires | ~10 | Male-to-female for Pi |

### Wiring Diagram

```
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
   ┌───────┐            ┌───────┐                ┌───────┐
   │ 220Ω  │            │ 220Ω  │                │ 10KΩ  │
   └───┬───┘            └───┬───┘                └───┬───┘
       │                    │                        │
       ▼                    ▼                        ├──── 3.3V
   ┌───────┐            ┌───────┐                    │
   │ GREEN │            │  RED  │                ┌───┴───┐
   │  LED  │            │  LED  │                │BUTTON │──── GPIO 27
   └───┬───┘            └───┬───┘                │   A   │
       │                    │                    └───┬───┘
       │                    │                        │
       └────────┬───────────┘                        │
                │                                    │
                ▼                                    ▼
               GND                                  GND

                                        (Same for Button B on GPIO 22)
```

### Pin Reference (BCM Numbering)

| Function | BCM Pin | Physical Pin | Color Suggestion |
|----------|---------|--------------|------------------|
| Green LED | GPIO 17 | Pin 11 | Green wire |
| Red LED | GPIO 18 | Pin 12 | Red wire |
| Button A | GPIO 27 | Pin 13 | Blue wire |
| Button B | GPIO 22 | Pin 15 | Yellow wire |
| 3.3V | - | Pin 1 or 17 | Red wire |
| Ground | - | Pin 6, 9, 14, etc. | Black wire |

---

## Build & Run

### With GPIO Hardware

```bash
make
sudo ./gpio_sensor
```

Note: `sudo` is required for GPIO access via pigpio.

### Simulation Mode (No Hardware)

```bash
make sim
./gpio_sensor_sim
```

Press 'A' and 'B' keys on your keyboard to simulate button presses.

---

## Controls

| Input | Action |
|-------|--------|
| Button A (GPIO 27) | First XOR input |
| Button B (GPIO 22) | Second XOR input |
| Press 'q' | Quit |
| Press 'A' key | Simulate Button A (sim mode) |
| Press 'B' key | Simulate Button B (sim mode) |

---

## How It Works

### 1. Input Reading
```c
float input_a = (float)trixc_gpio_read(BUTTON_A);
float input_b = (float)trixc_gpio_read(BUTTON_B);
```

### 2. Neural Network Inference
```c
float xor_result = xor_forward(input_a, input_b);
```

### 3. Output Control
```c
int led_green = (xor_result > 0.5f) ? 1 : 0;
int led_red = (xor_result <= 0.5f) ? 1 : 0;

trixc_gpio_write(LED_GREEN, led_green);
trixc_gpio_write(LED_RED, led_red);
```

**That's it.** Three steps: read inputs, run inference, control outputs.

---

## The XOR Logic

| Button A | Button B | XOR Result | LED |
|----------|----------|------------|-----|
| Released (0) | Released (0) | 0 | RED |
| Released (0) | Pressed (1) | 1 | GREEN |
| Pressed (1) | Released (0) | 1 | GREEN |
| Pressed (1) | Pressed (1) | 0 | RED |

The neural network learned this truth table. Now it runs in 0.003ms on your Pi, controlling real LEDs.

---

## Extending This Example

### Different Model

Replace XOR with any classification model:

```c
#include "../../models/your_classifier.c"

// Read sensor data
float sensor_data[6] = read_accelerometer();

// Run inference
float output[4];
your_classifier_forward(sensor_data, output);

// Control hardware based on result
int action = trixc_argmax(output, 4);
if (action == ALERT) {
    trixc_gpio_write(BUZZER_PIN, 1);
}
```

### More Sensors

Add analog sensors via I2C or SPI:
- Temperature sensor → fan control
- Light sensor → LED brightness
- Accelerometer → motion detection

### PWM Control

Use PWM for proportional control:
```c
// Output proportional to model confidence
int duty = (int)(output[predicted] * 255);
trixc_gpio_pwm(LED_PIN, duty);
```

---

## Troubleshooting

### "Failed to initialize GPIO"
- Make sure pigpio daemon is not running: `sudo killall pigpiod`
- Run with sudo: `sudo ./gpio_sensor`
- Check pigpio is installed: `apt install pigpio`

### "LED always on/off"
- Check wiring polarity (LED long leg = positive)
- Verify resistor values (220Ω for LEDs)
- Test with simple blink script first

### "Button not responding"
- Check pull-up resistor connection
- Verify button wiring (normally open)
- Test button with multimeter

### "Wrong GPIO pin"
- This example uses BCM numbering, NOT physical pins
- GPIO 17 = Physical pin 11 (not pin 17!)
- See pin reference table above

---

## What This Demonstrates

### 1. Hardware Integration
Neural network inference controlling physical outputs. The model's decision turns into light.

### 2. Real-Time Response
Button press → inference → LED change in under 1 millisecond. Feels instantaneous.

### 3. Embedded ML
No Python. No TensorFlow. Just C, GPIO, and frozen shapes. This is how ML should work on microcontrollers.

### 4. The Future
Today it's XOR and LEDs. Tomorrow:
- Robot control
- Smart home automation
- Industrial monitoring
- Wearable devices

---

## Safety Notes

- Never exceed 3.3V on GPIO pins
- Always use current-limiting resistors with LEDs
- Don't connect motors/relays directly to GPIO (use drivers)
- When in doubt, measure with a multimeter first

---

## Next Steps

- **04_camera_classify**: Add computer vision
- **Custom sensors**: I2C temperature, pressure, IMU
- **Motor control**: Add H-bridge for DC motors
- **Wireless**: Add WiFi/Bluetooth for remote control

---

*"It's all in the reflexes."*

The neural network doesn't care that it's controlling LEDs. It just computes. That's the power of frozen shapes - they work anywhere.
