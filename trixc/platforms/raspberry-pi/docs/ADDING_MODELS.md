# Adding Your Own Models

**From PyTorch to Raspberry Pi in three steps.**

---

## Overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   PyTorch    │     │    ONNX      │     │   TRIXC Pi   │
│   Model      │ --> │   Export     │ --> │   C Code     │
│              │     │              │     │              │
│  model.pt    │     │  model.onnx  │     │  model.c     │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## Step 1: Train Your Model

Use your favorite framework. Here's a PyTorch example:

```python
import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create and train model
model = SimpleClassifier(input_size=49, hidden_size=32, num_classes=10)
# ... training code ...
```

---

## Step 2: Export to ONNX

```python
import torch.onnx

# Create dummy input matching your model's expected input
dummy_input = torch.randn(1, 49)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "my_model.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=13
)

print("Exported to my_model.onnx")
```

---

## Step 3: Convert to C

```bash
# Navigate to TRIXC tools
cd trixc/tools

# Convert ONNX to C
python onnx2trix.py my_model.onnx my_model.c --emit-c

# Move to TRIXC Pi models
mv my_model.c ../platforms/raspberry-pi/models/
```

---

## Using Your Model

```c
#include "trixc_pi.h"
#include "../../models/my_model.c"  // Your converted model

int main() {
    trixc_init("My Classifier", 800, 480);

    float input[49] = {0};  // Your input data
    float output[10];       // Model output

    while (trixc_running()) {
        // Get input from somewhere (touch, camera, sensors, etc.)

        // Run inference
        my_model_forward(input, output);

        // Use output
        int predicted = trixc_argmax(output, 10);

        // Display results
        trixc_clear(TRIXC_BLACK);
        // ... draw UI ...
        trixc_present();
    }

    trixc_shutdown();
    return 0;
}
```

---

## Supported Operations

TRIXC supports most common neural network operations:

### Activations
- ReLU, LeakyReLU
- Sigmoid, Tanh
- GELU (exact and fast approximation)
- SiLU/Swish
- Softmax, LogSoftmax

### Arithmetic
- Add, Sub, Mul, Div
- Neg, Abs, Sqrt, Exp, Log, Pow

### Matrix Operations
- MatMul, Gemm
- Transpose

### Normalization
- LayerNorm, BatchNorm, RMSNorm

### Shape Operations
- Reshape, Flatten
- Concat, Gather

### Reductions
- ReduceSum, ReduceMean, ReduceMax, ReduceMin

---

## Model Guidelines

### Keep It Small

| Model Size | Inference Time | Suitability |
|------------|---------------|-------------|
| < 10 KB | < 0.1 ms | Excellent for Pi |
| 10-100 KB | 0.1-1 ms | Good for Pi |
| 100 KB - 1 MB | 1-10 ms | Acceptable |
| > 1 MB | > 10 ms | Consider optimization |

### Prefer Simple Architectures

For Raspberry Pi, simpler is better:

**Good:**
- MLPs (fully connected)
- Small CNNs (3-5 layers)
- Simple attention (single head)

**Challenging:**
- Deep CNNs (ResNet-50+)
- Large transformers
- GANs

### Quantization

Consider training with reduced precision:
- FP16 weights: 2x smaller
- INT8 weights: 4x smaller

TRIXC's APU supports FP4, FP8, FP16, FP32.

---

## Example: Custom Sensor Classifier

```python
# train_sensor_model.py

import torch
import torch.nn as nn
import torch.onnx

# Simple model for sensor data (e.g., accelerometer)
class SensorClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 16)   # 6 sensor values
        self.fc2 = nn.Linear(16, 4)   # 4 classes (idle, walking, running, falling)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Create model
model = SensorClassifier()

# ... train on your data ...

# Export
dummy_input = torch.randn(1, 6)
torch.onnx.export(model, dummy_input, "sensor_classifier.onnx",
                  input_names=['sensor_data'],
                  output_names=['activity'],
                  opset_version=13)
```

Convert and use:
```bash
python onnx2trix.py sensor_classifier.onnx sensor_classifier.c --emit-c
```

```c
// In your TRIXC Pi application
float sensor_data[6] = read_accelerometer();
float output[4];

sensor_classifier_forward(sensor_data, output);
int activity = trixc_argmax(output, 4);

const char* activities[] = {"idle", "walking", "running", "falling"};
printf("Activity: %s\n", activities[activity]);
```

---

## Troubleshooting

### "Unsupported operation: XXX"

Check the supported operations list. You may need to:
- Replace the operation with supported alternatives
- Implement a custom shape (advanced)

### "Model too large"

Try:
- Reducing hidden layer sizes
- Using fewer layers
- Quantization
- Knowledge distillation to a smaller model

### "Accuracy dropped after conversion"

This shouldn't happen for supported operations. Check:
- ONNX export worked correctly (`netron` tool helps)
- Input/output shapes match
- Data preprocessing is identical

---

## Best Practices

1. **Test on desktop first**: Compile with GCC on your laptop before moving to Pi

2. **Use the same preprocessing**: The model expects the same input format as training

3. **Validate outputs**: Compare TRIXC output with PyTorch output on test cases

4. **Start simple**: Get a tiny model working before scaling up

5. **Profile on hardware**: Actual Pi timing may differ from estimates

---

## Resources

- [ONNX Operators](https://onnx.ai/onnx/operators/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [TRIXC ONNX Shapes](../../docs/ONNX_SHAPES.md)
- [TRIXC APU Precision](../../docs/APU.md)

---

*"Shapes are opcodes. Polynomials are microcode. C is machine code."*
