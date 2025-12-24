# MNIST Draw

**Draw digits on the touchscreen. Watch them get recognized.**

This is real machine learning running on your Raspberry Pi - in 6.5 KB.

## What You'll See

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRIXC Pi - MNIST Draw                            │
│                                                                     │
│  ┌─────────────────────┐      Prediction:                          │
│  │                     │                                            │
│  │   ▓▓                │         ████████                          │
│  │   ▓▓                │         ██    ██                          │
│  │   ▓▓                │              ██                           │
│  │   ▓▓                │            ██                              │
│  │   ▓▓▓▓▓▓▓           │          ██                                │
│  │                     │         ████████                          │
│  └─────────────────────┘                                            │
│   7x7 input (49 pixels)         7  (87.3% confidence)              │
│                                                                     │
│  Class Probabilities:                                               │
│  0 ████                    12%                                      │
│  1 ██                       4%                                      │
│  2 ███                      8%                                      │
│  ... (more bars)                                                    │
│  7 ████████████████████████ 87%  <- highlighted                    │
│                                                                     │
│  Inference: 0.082ms | Memory: 6,564 bytes | FPS: 60                │
└─────────────────────────────────────────────────────────────────────┘
```

## Build & Run

```bash
make
./mnist_draw
```

## Controls

| Input | Action |
|-------|--------|
| Touch & drag | Draw on canvas |
| Press 'c' | Clear canvas |
| Press 'q' or Escape | Quit |

## How It Works

1. **7x7 Canvas**: The touchscreen maps to a 7x7 grid (49 pixels)
2. **Real-time Inference**: Every frame, the canvas runs through the neural network
3. **Softmax Output**: See probabilities for all 10 digits
4. **Sub-millisecond**: ~0.1ms per inference on Pi 4

## The Model

```
Architecture: 49 inputs → 32 hidden (ReLU) → 10 outputs

Layer 1: 49 × 32 = 1,568 weights + 32 biases = 1,600 params
Layer 2: 32 × 10 = 320 weights + 10 biases = 330 params

Total: 1,930 parameters = 6,564 bytes
```

## Tips for Better Recognition

1. **Fill the canvas** - Draw digits that use most of the space
2. **Center your digits** - The model was trained on centered digits
3. **Use thick strokes** - The 7x7 resolution is very low
4. **Try multiple times** - Some digits (5 vs S, 1 vs 7) are tricky

## What This Demonstrates

### 1. Touch Input
Smooth drawing on a low-resolution canvas with interpolation.

### 2. Real-time ML
Neural network inference fast enough for interactive use.

### 3. Probability Visualization
Not just "what digit?" but "how confident?"

### 4. Size Perspective
| System | Size |
|--------|------|
| This example | 6.5 KB model |
| PyTorch + MNIST | ~1 GB |
| TensorFlow Lite | ~100 MB |

## The Code

The interesting parts:

```c
/* Run inference every frame */
mnist_forward_softmax(canvas, output);

/* Find prediction */
int predicted = 0;
for (int i = 1; i < 10; i++) {
    if (output[i] > output[predicted]) {
        predicted = i;
    }
}
```

That's it. Two function calls and a loop. No framework overhead.

## Next Steps

- **03_gpio_sensor**: Use model output to control hardware
- **04_camera_classify**: Point camera at objects
- **Train your own**: See docs/ADDING_MODELS.md

## Why 7x7?

The original MNIST uses 28x28 images. We downsample to 7x7 because:

1. **Touch-friendly**: Finger drawing is imprecise
2. **Fast inference**: 49 inputs instead of 784
3. **Forgiving**: Low resolution hides sloppy strokes
4. **Educational**: You can see every pixel

For production, you'd use the full 28x28 or higher. But for learning? 7x7 is perfect.

---

*"It's all in the reflexes."*
