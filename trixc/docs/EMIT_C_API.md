# C Code Generation API Reference

**Python API for generating C code from ONNX models.**

---

## Module: `onnx2trix`

Location: `trixc/tools/onnx2trix.py`

---

## Classes

### `ONNX2TrixConverter`

Converts ONNX models to Octave IR (intermediate representation).

```python
from onnx2trix import ONNX2TrixConverter

converter = ONNX2TrixConverter(
    emit_weights=True,   # Include weight data
    precision="fp32"     # Weight precision: fp16, fp32, fp64
)

trix = converter.convert("model.onnx")
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `emit_weights` | `bool` | `True` | Include weight data in output |
| `precision` | `str` | `"fp32"` | Weight precision |

#### Methods

##### `convert(onnx_path: str) -> Dict[str, Any]`

Convert an ONNX model to Octave IR.

**Returns:** Dictionary containing:
- `name`: Model name
- `shapes`: List of operations
- `weights`: Weight tensors
- `inputs`: Input specifications
- `outputs`: Output specifications
- `routing`: Execution order
- `metadata`: ONNX version info

---

### `TrixcError`

Exception raised during C code generation.

```python
from onnx2trix import TrixcError

try:
    code = generate_c_code(trix)
except TrixcError as e:
    print(f"Generation failed: {e}")
```

---

## Functions

### `generate_c_code`

```python
def generate_c_code(trix: Dict[str, Any], standalone: bool = True) -> str
```

Generate complete C source from Octave IR.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trix` | `Dict` | required | Octave IR dictionary |
| `standalone` | `bool` | `True` | Include `main()` for standalone binary |

**Returns:** Complete C source code as string.

**Example:**

```python
from onnx2trix import ONNX2TrixConverter, generate_c_code

converter = ONNX2TrixConverter()
trix = converter.convert("model.onnx")
c_code = generate_c_code(trix, standalone=True)

with open("model.c", "w") as f:
    f.write(c_code)
```

---

### `generate_c_header`

```python
def generate_c_header(trix: Dict[str, Any]) -> str
```

Generate C header file with forward declarations.

**Returns:** Header file content.

**Example:**

```python
header = generate_c_header(trix)
with open("model.h", "w") as f:
    f.write(header)
```

---

### `emit_weights`

```python
def emit_weights(weights: Dict[str, Any], max_per_line: int = 8) -> str
```

Emit weight arrays as static const declarations.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weights` | `Dict` | required | Weight dictionary from Octave IR |
| `max_per_line` | `int` | `8` | Floats per line in output |

**Returns:** C code declaring weight arrays.

**Example:**

```python
weights_c = emit_weights(trix["weights"])
# Output:
# static const float W_fc1_weight[768] = {
#     0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
#     ...
# };
```

---

### `emit_dimensions`

```python
def emit_dimensions(trix: Dict[str, Any]) -> str
```

Emit dimension #defines from shape info.

**Returns:** C preprocessor defines.

**Example:**

```python
dims_c = emit_dimensions(trix)
# Output:
# #define DIM_INPUT_0 1
# #define DIM_INPUT_1 768
# #define DIM_OUTPUT_0 1
# #define DIM_OUTPUT_1 768
```

---

### `emit_forward`

```python
def emit_forward(trix: Dict[str, Any], tensor_map: Dict) -> str
```

Emit the forward pass function body.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `trix` | `Dict` | Octave IR dictionary |
| `tensor_map` | `Dict` | Tensor name to C identifier mapping |

**Returns:** C code for forward pass body.

---

### `emit_shape_call`

```python
def emit_shape_call(shape: Dict, tensor_map: Dict) -> str
```

Emit C code for a single shape invocation.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `Dict` | Shape specification |
| `tensor_map` | `Dict` | Tensor name to C identifier mapping |

**Returns:** Single line of C code.

**Example:**

```python
shape = {"kind": "RELU", "inputs": ["t0"], "outputs": ["t1"]}
code = emit_shape_call(shape, tensor_map)
# Output: "    for (int i = 0; i < 768; i++) t1[i] = trix_onnx_relu(t0[i]);"
```

---

### `emit_main`

```python
def emit_main(trix: Dict[str, Any], tensor_map: Dict, model_name: str) -> str
```

Emit main() for standalone binary.

**Returns:** C code for main function with I/O handling.

---

### `build_tensor_map`

```python
def build_tensor_map(trix: Dict[str, Any]) -> Dict[str, Dict]
```

Build mapping from ONNX tensor names to C identifiers.

**Returns:** Dictionary mapping tensor names to:
- `kind`: "input", "output", "weight", or "intermediate"
- `c_name`: Valid C identifier
- `shape`: Tensor dimensions
- `size`: Total element count

**Example:**

```python
tensor_map = build_tensor_map(trix)
# {
#     "input": {"kind": "input", "c_name": "input", "shape": [1, 768], "size": 768},
#     "fc1.weight": {"kind": "weight", "c_name": "W_fc1_weight", "shape": [768, 3072], "size": 2359296},
#     "relu_out": {"kind": "intermediate", "c_name": "t0", "shape": [1, 3072], "size": 3072},
# }
```

---

### `infer_tensor_shapes`

```python
def infer_tensor_shapes(trix: Dict[str, Any], tensor_map: Dict) -> None
```

Infer shapes for intermediate tensors based on operations.

Modifies `tensor_map` in place.

---

## Helper Functions

### `_sanitize_name`

```python
def _sanitize_name(name: str) -> str
```

Convert ONNX tensor name to valid C identifier.

**Examples:**

```python
_sanitize_name("fc1.weight")      # "fc1_weight"
_sanitize_name("model/layer/w")   # "model_layer_w"
_sanitize_name("0_layer")         # "_0_layer"
```

---

### `_get_tensor_size`

```python
def _get_tensor_size(shape: List) -> int
```

Calculate total size from shape.

**Examples:**

```python
_get_tensor_size([1, 768])        # 768
_get_tensor_size([2, 3, 4])       # 24
_get_tensor_size(["batch", 768])  # 768 (dynamic dims → 1)
```

---

### `_format_float`

```python
def _format_float(f: float) -> str
```

Format float for C code.

**Examples:**

```python
_format_float(0.0)    # "0.0f"
_format_float(1.0)    # "1.0f"
_format_float(0.123)  # "0.123f"
_format_float(-1.5)   # "-1.5f"
```

---

## Shape Templates

The `emit_shape_call` function uses templates for each operation:

| Kind | Template |
|------|----------|
| `RELU` | `for (int i = 0; i < {n}; i++) {out}[i] = trix_onnx_relu({in}[i]);` |
| `GELU` | `for (int i = 0; i < {n}; i++) {out}[i] = trix_onnx_gelu({in}[i]);` |
| `SIGMOID` | `for (int i = 0; i < {n}; i++) {out}[i] = trix_onnx_sigmoid({in}[i]);` |
| `ADD` | `trix_onnx_add({a}, {b}, {out}, {n});` |
| `SUB` | `trix_onnx_sub({a}, {b}, {out}, {n});` |
| `MUL` | `trix_onnx_mul({a}, {b}, {out}, {n});` |
| `DIV` | `trix_onnx_div({a}, {b}, {out}, {n});` |
| `MATMUL` | `trix_onnx_matmul({A}, {B}, {C}, {M}, {N}, {K});` |
| `GEMM` | `trix_onnx_gemm({A}, {B}, {bias}, {C}, {M}, {N}, {K}, {alpha}, {beta});` |
| `SOFTMAX` | `trix_onnx_softmax({in}, {out}, {n});` |
| `LAYER_NORM` | `trix_onnx_layer_norm({in}, {gamma}, {beta}, {out}, {n}, {eps});` |
| `IDENTITY` | `memcpy({out}, {in}, {n} * sizeof(float));` |

---

## Octave IR Format

The intermediate representation is a JSON-serializable dictionary:

```json
{
  "version": "1.0",
  "format": "octave-ir",
  "name": "model_name",
  "producer": "onnx2trix (from pytorch)",
  "precision": "fp32",

  "inputs": [
    {"name": "input", "shape": [1, 768], "dtype": "float32"}
  ],

  "outputs": [
    {"name": "output", "shape": [1, 768], "dtype": "float32"}
  ],

  "weights": {
    "fc1.weight": {
      "shape": [768, 3072],
      "dtype": "float32",
      "size": 2359296,
      "data": [0.1, 0.2, ...]
    }
  },

  "shapes": [
    {
      "id": 0,
      "kind": "MATMUL",
      "onnx_op": "MatMul",
      "inputs": ["input", "fc1.weight"],
      "outputs": ["matmul_out"],
      "composed": false
    },
    {
      "id": 1,
      "kind": "GELU",
      "onnx_op": "Gelu",
      "inputs": ["matmul_out"],
      "outputs": ["output"],
      "composed": false
    }
  ],

  "routing": {
    "mode": "static",
    "table": [0, 1]
  },

  "metadata": {
    "onnx_opset": 13,
    "ir_version": 8
  }
}
```

---

## Usage Patterns

### Basic Conversion

```python
from onnx2trix import ONNX2TrixConverter, generate_c_code

# Convert
converter = ONNX2TrixConverter()
trix = converter.convert("model.onnx")

# Generate C
c_code = generate_c_code(trix)

# Save
with open("model.c", "w") as f:
    f.write(c_code)
```

### Library Mode

```python
c_code = generate_c_code(trix, standalone=False)
```

### Custom Weight Handling

```python
# Convert without weights
converter = ONNX2TrixConverter(emit_weights=False)
trix = converter.convert("model.onnx")

# Generate C (weights will be extern)
c_code = generate_c_code(trix)
```

### Inspecting the IR

```python
import json

converter = ONNX2TrixConverter()
trix = converter.convert("model.onnx")

# Save IR for inspection
with open("model.trix", "w") as f:
    json.dump(trix, f, indent=2)

# Print summary
print(f"Shapes: {len(trix['shapes'])}")
print(f"Weights: {len(trix['weights'])}")
print(f"Input: {trix['inputs'][0]}")
print(f"Output: {trix['outputs'][0]}")
```

---

*"The API is frozen. The shapes are frozen. The code just writes itself."*
