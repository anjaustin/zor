# Hello XOR

**Your first TRIXC Pi program!**

This example shows a tiny neural network computing XOR, alongside the frozen polynomial shape that computes it exactly.

## What You'll See

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRIXC Pi - Hello XOR                             │
│                                                                     │
│  Neural Network                    Frozen Shape                     │
│  ┌─────────┐    ┌────────┐        ┌────────────────────────────┐   │
│  │ [0, 1]  │ -> │ Hidden │ ->     │ XOR(a, b) = a + b - 2ab   │   │
│  └─────────┘    │ +ReLU  │        │ = 0 + 1 - 2*0*1           │   │
│                 └────────┘        │ = 1                        │   │
│  Output: 0.9987 = 1 (correct)     └────────────────────────────┘   │
│                                                                     │
│  XOR Truth Table                                                    │
│  ────────────────                                                   │
│  A   B   A^B   NN      Shape                                        │
│  0   0   0     0.00    0        <- tap to highlight                │
│  0   1   1     1.00    1                                            │
│  1   0   1     0.99    1                                            │
│  1   1   0     0.01    0                                            │
│                                                                     │
│  Inference: 0.003ms | Memory: 52 bytes | FPS: 60                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Build & Run

```bash
make
./hello_xor
```

## Controls

| Input | Action |
|-------|--------|
| Tap anywhere | Cycle to next input |
| Press 'v' | Toggle verbose mode (see hidden layer) |
| Press 'q' or Escape | Quit |

## What You'll Learn

1. **Neural networks are just math**
   - 13 parameters, 52 bytes
   - Matrix multiply + ReLU + matrix multiply
   - Runs in microseconds

2. **Frozen shapes are exact**
   - XOR(a, b) = a + b - 2ab
   - 0 parameters, ~20 bytes
   - Mathematically perfect

3. **Size matters**
   - This example: ~70 KB compiled
   - PyTorch equivalent: 2 GB+

## The Code

The interesting parts:

```c
/* Neural network inference */
float nn_output = xor_forward(x0, x1);

/* Frozen shape - exact! */
float shape_output = xor_frozen_shape(x0, x1);
// = a + b - 2*a*b
```

## Next Steps

- **02_mnist_draw**: Draw digits on the touchscreen
- **03_gpio_sensor**: Control LEDs based on model output
- **04_camera_classify**: Point camera at objects

## Philosophy

> "Don't learn what you can derive."

The neural network *learned* to compute XOR through thousands of gradient descent steps. The frozen shape *derives* XOR from Boolean algebra.

Both produce the same result. One took training. One took thinking.

TRIXC is about finding the frozen shapes hiding inside neural networks.

---

*"It's all in the reflexes."*
