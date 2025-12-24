# TRIX Model Format Specification

**Version:** 1.0
**Status:** Stable

---

## Overview

The `.trix` format is a human-readable model definition format for TRIX Forge. It uses YAML syntax and describes:

- Model metadata (name, version, description)
- Input/output specifications
- Layer/shape definitions
- Tunable levers
- Custom metadata

---

## File Structure

```yaml
# Header
version: "1.0"
name: model_name
description: "Optional description"

# I/O Specification
input:
  shape: [dim1, dim2, ...]
  dtype: float32

output:
  shape: [dim1, dim2, ...]
  dtype: float32

# Shapes (layers)
shapes:
  - name: layer_name
    type: layer_type
    ...

# Levers (tunable parameters)
levers:
  lever_name:
    type: enum|int|float|bool
    ...

# Custom metadata
metadata:
  key: value
```

---

## Sections

### Header

Required fields:
- `version`: Format version (currently "1.0")
- `name`: Model identifier (alphanumeric + underscore)

Optional fields:
- `description`: Human-readable description

```yaml
version: "1.0"
name: my_model
description: "A classifier for XYZ task"
```

### Input/Output

Specify the model's interface:

```yaml
input:
  shape: [4]           # 4-element vector
  dtype: float32       # Data type

output:
  shape: [2]           # 2-class output
  dtype: float32
  activation: softmax  # Optional final activation
```

Supported dtypes:
- `float32` (default)
- `float16`
- `int8`
- `uint8`

### Shapes

Shapes are the computational building blocks. Each shape has:
- `name`: Unique identifier
- `type`: Shape type
- Type-specific parameters

```yaml
shapes:
  - name: linear_1
    type: linear
    input: [4]
    output: [8]
    init: xavier

  - name: relu_1
    type: relu

  - name: linear_2
    type: linear
    input: [8]
    output: [2]
```

#### Shape Types

**Linear Layers:**
```yaml
- name: fc1
  type: linear
  input: [in_features]
  output: [out_features]
  bias: true           # default: true
  init: xavier         # xavier, kaiming, zeros, ones
```

**Activations:**
```yaml
- name: act
  type: relu           # relu, gelu, sigmoid, tanh, softmax
```

**Frozen Logic (no learnable parameters):**
```yaml
- name: xor_gate
  type: xor            # xor, and, or, not
  frozen: true
```

**Normalization:**
```yaml
- name: norm
  type: layernorm
  shape: [features]
  eps: 1e-5
```

### Levers

Levers are tunable parameters exposed for control:

```yaml
levers:
  precision:
    type: enum
    options: [fp32, fp16, fp8]
    default: fp16
    description: "Numeric precision"

  learning_rate:
    type: float
    min: 0.000001
    max: 1.0
    default: 0.001
    description: "Learning rate"

  batch_size:
    type: int
    min: 1
    max: 1024
    default: 32

  debug:
    type: bool
    default: false
```

#### Lever Types

| Type | Fields | Example |
|------|--------|---------|
| `enum` | `options`, `default` | `[fp32, fp16]` |
| `int` | `min`, `max`, `default` | `1-1024` |
| `float` | `min`, `max`, `default` | `0.0-1.0` |
| `bool` | `default` | `true/false` |

### Metadata

Custom key-value pairs:

```yaml
metadata:
  author: "Your Name"
  license: "MIT"
  created: "2025-12-22"
  tags: ["classifier", "production"]
  accuracy: 0.95
```

---

## Complete Example

```yaml
# tiny_mlp.trix
version: "1.0"
name: tiny_mlp
description: "Binary classifier with one hidden layer"

input:
  shape: [4]
  dtype: float32

output:
  shape: [2]
  dtype: float32
  activation: softmax

shapes:
  - name: linear_1
    type: linear
    input: [4]
    output: [8]
    init: xavier

  - name: relu_1
    type: relu

  - name: linear_2
    type: linear
    input: [8]
    output: [2]
    init: xavier

  - name: softmax
    type: softmax

levers:
  precision:
    type: enum
    options: [fp32, fp16, fp8]
    default: fp16

  optimize:
    type: enum
    options: [none, aggressive]
    default: aggressive

  fuse_ops:
    type: bool
    default: true

metadata:
  author: "TRIX Forge"
  task: "binary_classification"
```

---

## Validation Rules

1. `name` must be unique within the file
2. `version` must be a valid semver string
3. Shape `input`/`output` dimensions must be compatible
4. Lever `default` must be within bounds
5. All referenced shapes must be defined

---

## Parsing

TRIX files are parsed as YAML. The reference parser is in `trix_forge.model.ModelIR.load()`.

```python
from trix_forge import ModelIR

model = ModelIR.load("model.trix")
print(model.name)
print(model.shapes)
```

---

*"Geometry is computation."*
