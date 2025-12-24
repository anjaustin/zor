# TRIXC Pi Models

Pre-trained models for TRIXC Pi examples.

## Included Models

| Model | Input | Output | Size | Purpose |
|-------|-------|--------|------|---------|
| `xor_mlp.c` | 2 floats | 1 float | 52 bytes | XOR computation demo |
| `mnist_7x7.c` | 49 floats (7x7) | 10 floats | 6.5 KB | Digit classification |

## Using a Model

```c
#include "../../models/xor_mlp.c"

// Call the forward function
float output = xor_forward(0.0f, 1.0f);  // Returns ~1.0
```

```c
#include "../../models/mnist_7x7.c"

float input[49] = { /* 7x7 grayscale image */ };
float output[10];

mnist_forward_softmax(input, output);
int digit = trixc_argmax(output, 10);
```

## Adding Your Own Model

See [docs/ADDING_MODELS.md](../docs/ADDING_MODELS.md) for the full guide.

Quick version:

1. Train in PyTorch/TensorFlow
2. Export to ONNX
3. Convert: `python onnx2trix.py model.onnx model.c --emit-c`
4. Copy to this directory
5. Include in your code: `#include "../../models/model.c"`

## Model Guidelines

- Keep models small (< 100 KB for responsive inference)
- Use simple architectures (MLPs work great)
- Test on desktop before deploying to Pi
- Validate outputs match your training framework

---

*"Don't learn what you can derive."*
