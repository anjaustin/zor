# TRIXC Pi Quick Start

**5 minutes from zero to running neural network.**

---

## Prerequisites

- Raspberry Pi 4 (any RAM size)
- Touchscreen connected
- Ubuntu or Raspberry Pi OS
- Terminal access

---

## Step 1: Install Dependencies (1 minute)

```bash
cd trixc/platforms/raspberry-pi
./scripts/setup_pi.sh --minimal
```

Or manually:
```bash
sudo apt update
sudo apt install -y libsdl2-dev build-essential
```

---

## Step 2: Build (30 seconds)

```bash
make hello
```

You should see:
```
Built: build/hello_xor
```

---

## Step 3: Run (instant)

```bash
./build/hello_xor
```

**What you'll see:**
- A visualization of XOR computation
- Neural network output vs frozen shape
- Tap anywhere to cycle inputs
- Inference timing in microseconds

---

## Step 4: Try MNIST Draw

```bash
make mnist
./build/mnist_draw
```

**What you'll see:**
- A drawing canvas
- Real-time digit recognition
- Probability bars for all 10 digits
- Draw with your finger, see it classified

---

## What Just Happened?

You ran two neural networks:

1. **XOR Network** (52 bytes)
   - 2 inputs → 4 hidden → 1 output
   - Computes XOR function
   - Shows both learned and derived versions

2. **MNIST Network** (6.5 KB)
   - 49 inputs (7x7 image) → 32 hidden → 10 outputs
   - Classifies handwritten digits
   - Real-time inference while you draw

**No Python. No PyTorch. No TensorFlow. Just C.**

---

## Next Steps

### Explore the Code
```bash
# Look at the source
cat examples/01_hello_xor/main.c
cat models/xor_mlp.c
```

### Build All Examples
```bash
make          # Build everything
make help     # See all options
```

### Add Your Own Model
```bash
# Convert an ONNX model
python ../../tools/onnx2trix.py my_model.onnx my_model.c --emit-c
```

---

## Common Issues

### "Command not found: sdl2-config"
```bash
sudo apt install libsdl2-dev
```

### "Cannot open display"
Run on the Pi's console, not over SSH:
```bash
export DISPLAY=:0
```

### "Permission denied"
For GPIO examples:
```bash
sudo ./build/gpio_sensor
```

---

## Resources

- [README.md](../README.md) - Full platform documentation
- [API.md](API.md) - Complete API reference
- [ADDING_MODELS.md](ADDING_MODELS.md) - Model conversion guide
- [Examples](../examples/) - Working code to study

---

*"It's all in the reflexes."*

**Total time: ~5 minutes**

Now you're running neural networks on a Raspberry Pi. Go build something!
