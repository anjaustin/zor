#!/usr/bin/env python3
"""
ONNX to Native C Compiler

Converts ONNX models to:
  - Octave IR (.trix) - JSON intermediate representation
  - Native C code (.c) - Self-contained, compilable source

"ONNX is notation. Shapes are truth. C is execution."

Usage:
    # Convert to Octave IR (JSON)
    python onnx2trix.py model.onnx model.trix

    # Convert directly to C code
    python onnx2trix.py model.onnx model.c --emit-c

    # Compile and run
    gcc -O3 -DTRIXC_STANDALONE -I./include model.c -o model -lm
    ./model input.bin output.bin

Options:
    --emit-c         Generate C code instead of Octave IR
    --standalone     Include main() for standalone binary (default)
    --no-standalone  Omit main() for library mode
    --emit-header    Also generate .h header file
    --precision      Weight precision: fp16, fp32, fp64
    --no-weights     Exclude weight data
    --verbose        Print conversion details

API Functions:
    ONNX2TrixConverter  - Convert ONNX to Octave IR
    generate_c_code     - Generate C source from Octave IR
    generate_c_header   - Generate C header file
    emit_weights        - Emit weight arrays
    emit_forward        - Emit forward pass function
    emit_shape_call     - Emit single operation

See docs/ONNX2C.md for complete documentation.
See docs/EMIT_C_API.md for API reference.
See docs/SUPPORTED_OPS.md for operation compatibility.

Created by: Tripp + Claude
Date: December 2025
Version: 0.3.0
"""

import json
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path

# Try to import onnx - if not available, provide helpful message
try:
    import onnx
    from onnx import numpy_helper
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ═══════════════════════════════════════════════════════════════════════════
# Shape Mapping: ONNX Ops → TRIXC Shapes
# ═══════════════════════════════════════════════════════════════════════════

SHAPE_MAP = {
    # Arithmetic (element-wise)
    "Add": {"kind": "ADD", "composed": False},
    "Sub": {"kind": "SUB", "composed": False},
    "Mul": {"kind": "MUL", "composed": False},
    "Div": {"kind": "DIV", "composed": False},
    "Neg": {"kind": "NEG", "composed": False},
    "Abs": {"kind": "ABS", "composed": False},
    "Sqrt": {"kind": "SQRT", "composed": False},
    "Exp": {"kind": "EXP", "composed": False},
    "Log": {"kind": "LOG", "composed": False},
    "Pow": {"kind": "POW", "composed": False},

    # Matrix operations
    "MatMul": {"kind": "MATMUL", "composed": False},
    "Gemm": {"kind": "GEMM", "composed": False},

    # Activations
    "Relu": {"kind": "RELU", "composed": False},
    "Sigmoid": {"kind": "SIGMOID", "composed": False},
    "Tanh": {"kind": "TANH", "composed": False},
    "Softmax": {"kind": "SOFTMAX", "composed": False},
    "Gelu": {"kind": "GELU", "composed": False},
    "Silu": {"kind": "SILU", "composed": False},
    "LeakyRelu": {"kind": "LEAKY_RELU", "composed": False},

    # Reductions
    "ReduceSum": {"kind": "REDUCE_SUM", "composed": False},
    "ReduceMean": {"kind": "REDUCE_MEAN", "composed": False},
    "ReduceMax": {"kind": "REDUCE_MAX", "composed": False},
    "ReduceMin": {"kind": "REDUCE_MIN", "composed": False},
    "ReduceProd": {"kind": "REDUCE_PROD", "composed": False},

    # Shape operations
    "Reshape": {"kind": "RESHAPE", "composed": False},
    "Transpose": {"kind": "TRANSPOSE", "composed": False},
    "Concat": {"kind": "CONCAT", "composed": False},
    "Split": {"kind": "SPLIT", "composed": False},
    "Squeeze": {"kind": "SQUEEZE", "composed": False},
    "Unsqueeze": {"kind": "UNSQUEEZE", "composed": False},
    "Flatten": {"kind": "FLATTEN", "composed": False},
    "Gather": {"kind": "GATHER", "composed": False},

    # Normalization (composed from primitives)
    "BatchNormalization": {
        "kind": "BATCH_NORM",
        "composed": True,
        "decomposition": ["SUB", "DIV", "MUL", "ADD"]
    },
    "LayerNormalization": {
        "kind": "LAYER_NORM",
        "composed": True,
        "decomposition": ["REDUCE_MEAN", "SUB", "MUL", "REDUCE_MEAN", "SQRT", "DIV", "MUL", "ADD"]
    },

    # Comparison
    "Equal": {"kind": "EQUAL", "composed": False},
    "Greater": {"kind": "GREATER", "composed": False},
    "Less": {"kind": "LESS", "composed": False},
    "Where": {"kind": "WHERE", "composed": False},
    "Clip": {"kind": "CLIP", "composed": False},

    # Pooling (can be composed)
    "GlobalAveragePool": {"kind": "GLOBAL_AVG_POOL", "composed": False},
    "GlobalMaxPool": {"kind": "GLOBAL_MAX_POOL", "composed": False},

    # Attention (high-level shape)
    "Attention": {"kind": "ATTENTION", "composed": True},

    # Conv (lowered to im2col + matmul)
    "Conv": {
        "kind": "CONV",
        "composed": True,
        "decomposition": ["IM2COL", "MATMUL", "ADD"]
    },

    # Identity (passthrough)
    "Identity": {"kind": "IDENTITY", "composed": False},
    "Dropout": {"kind": "IDENTITY", "composed": False},  # No-op at inference
}


# ═══════════════════════════════════════════════════════════════════════════
# Converter
# ═══════════════════════════════════════════════════════════════════════════

class ONNX2TrixConverter:
    """Converts ONNX models to Octave IR."""

    def __init__(self, emit_weights: bool = True, precision: str = "fp32"):
        self.emit_weights = emit_weights
        self.precision = precision
        self.warnings = []

    def convert(self, onnx_path: str) -> Dict[str, Any]:
        """Convert an ONNX model to Octave IR."""
        if not HAS_ONNX:
            raise ImportError("onnx package required. Install with: pip install onnx")

        model = onnx.load(onnx_path)

        # Basic validation
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            self.warnings.append(f"ONNX validation warning: {e}")

        graph = model.graph

        trix = {
            "version": "1.0",
            "format": "octave-ir",
            "name": graph.name or Path(onnx_path).stem,
            "producer": f"onnx2trix (from {model.producer_name or 'unknown'})",
            "precision": self.precision,
            "shapes": [],
            "routing": {"mode": "static", "table": []},
            "weights": {},
            "inputs": [],
            "outputs": [],
            "metadata": {
                "onnx_opset": model.opset_import[0].version if model.opset_import else None,
                "ir_version": model.ir_version,
            }
        }

        # Extract weights (initializers)
        weight_names = set()
        for init in graph.initializer:
            weight_names.add(init.name)
            if self.emit_weights:
                trix["weights"][init.name] = self._extract_weight(init)
            else:
                # Just record shape info
                arr = numpy_helper.to_array(init)
                trix["weights"][init.name] = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "size": arr.size,
                    "data": None  # Not emitted
                }

        # Inputs (excluding weights)
        for inp in graph.input:
            if inp.name not in weight_names:
                trix["inputs"].append(self._tensor_info(inp))

        # Outputs
        for out in graph.output:
            trix["outputs"].append(self._tensor_info(out))

        # Convert operations
        for node in graph.node:
            shape_info = self._convert_node(node)
            if shape_info:
                shape_info["id"] = len(trix["shapes"])
                trix["shapes"].append(shape_info)
                trix["routing"]["table"].append(shape_info["id"])

        # Add warnings if any
        if self.warnings:
            trix["metadata"]["warnings"] = self.warnings

        return trix

    def _convert_node(self, node) -> Optional[Dict[str, Any]]:
        """Convert a single ONNX node to a shape."""
        op_type = node.op_type

        if op_type not in SHAPE_MAP:
            self.warnings.append(f"Unsupported op: {op_type} (skipped)")
            return None

        mapping = SHAPE_MAP[op_type]

        shape = {
            "kind": mapping["kind"],
            "onnx_op": op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "composed": mapping.get("composed", False),
        }

        # Extract attributes
        attrs = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.FLOAT:
                attrs[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.INT:
                attrs[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.INTS:
                attrs[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOATS:
                attrs[attr.name] = list(attr.floats)
            elif attr.type == onnx.AttributeProto.STRING:
                attrs[attr.name] = attr.s.decode('utf-8')

        if attrs:
            shape["attributes"] = attrs

        # Add decomposition info for composed shapes
        if mapping.get("composed") and "decomposition" in mapping:
            shape["decomposition"] = mapping["decomposition"]

        return shape

    def _extract_weight(self, init) -> Dict[str, Any]:
        """Extract weight tensor as JSON-serializable dict."""
        if not HAS_NUMPY:
            return {"error": "numpy not available"}

        arr = numpy_helper.to_array(init)

        # Convert to target precision
        if self.precision == "fp16":
            arr = arr.astype(np.float16)
        elif self.precision == "fp32":
            arr = arr.astype(np.float32)

        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "size": arr.size,
            "data": arr.flatten().tolist()
        }

    def _tensor_info(self, tensor) -> Dict[str, Any]:
        """Extract tensor info (name, shape, type)."""
        info = {"name": tensor.name}

        if tensor.type.tensor_type.HasField('shape'):
            dims = []
            for dim in tensor.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    dims.append(dim.dim_value)
                elif dim.HasField('dim_param'):
                    dims.append(dim.dim_param)  # Dynamic dimension
                else:
                    dims.append(-1)  # Unknown
            info["shape"] = dims

        elem_type = tensor.type.tensor_type.elem_type
        type_map = {
            1: "float32", 2: "uint8", 3: "int8", 4: "uint16",
            5: "int16", 6: "int32", 7: "int64", 8: "string",
            9: "bool", 10: "float16", 11: "float64"
        }
        info["dtype"] = type_map.get(elem_type, f"type_{elem_type}")

        return info


# ═══════════════════════════════════════════════════════════════════════════
# C Code Generator
# ═══════════════════════════════════════════════════════════════════════════

class TrixcError(Exception):
    """Error during C code generation."""
    pass


def _sanitize_name(name: str) -> str:
    """Convert ONNX tensor name to valid C identifier."""
    # Replace invalid chars with underscore
    result = ""
    for c in name:
        if c.isalnum() or c == '_':
            result += c
        else:
            result += '_'
    # Ensure doesn't start with digit
    if result and result[0].isdigit():
        result = '_' + result
    return result or '_unnamed'


def _get_tensor_size(shape: List) -> int:
    """Calculate total size from shape, treating dynamic dims as 1."""
    size = 1
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            size *= dim
        else:
            size *= 1  # Dynamic or unknown dim treated as 1
    return size


def _format_float(f: float) -> str:
    """Format float for C code."""
    if f == 0.0:
        return "0.0f"
    elif f == int(f):
        return f"{int(f)}.0f"
    else:
        return f"{f:.8g}f"


def emit_weights(weights: Dict[str, Any], max_per_line: int = 8) -> str:
    """Emit weight arrays as static const declarations."""
    lines = []

    for name, weight_info in weights.items():
        c_name = "W_" + _sanitize_name(name)
        shape = weight_info.get("shape", [])
        data = weight_info.get("data", [])

        if not data:
            lines.append(f"/* {c_name}: no data (shape {shape}) */")
            continue

        size = len(data)
        shape_str = " x ".join(str(d) for d in shape)

        lines.append(f"/* {name}: [{shape_str}] = {size} elements */")
        lines.append(f"static const float {c_name}[{size}] = {{")

        # Format data in rows
        for i in range(0, size, max_per_line):
            chunk = data[i:i + max_per_line]
            formatted = ", ".join(_format_float(v) for v in chunk)
            if i + max_per_line < size:
                formatted += ","
            lines.append(f"    {formatted}")

        lines.append("};")
        lines.append("")

    return "\n".join(lines)


def emit_dimensions(trix: Dict[str, Any]) -> str:
    """Emit dimension #defines from shape info."""
    lines = []
    defined = set()

    # Input dimensions
    for inp in trix.get("inputs", []):
        shape = inp.get("shape", [])
        name = _sanitize_name(inp["name"]).upper()
        for i, dim in enumerate(shape):
            if isinstance(dim, int) and dim > 0:
                dim_name = f"DIM_{name}_{i}"
                if dim_name not in defined:
                    lines.append(f"#define {dim_name} {dim}")
                    defined.add(dim_name)

    # Output dimensions
    for out in trix.get("outputs", []):
        shape = out.get("shape", [])
        name = _sanitize_name(out["name"]).upper()
        for i, dim in enumerate(shape):
            if isinstance(dim, int) and dim > 0:
                dim_name = f"DIM_{name}_{i}"
                if dim_name not in defined:
                    lines.append(f"#define {dim_name} {dim}")
                    defined.add(dim_name)

    # Weight dimensions
    for wname, winfo in trix.get("weights", {}).items():
        shape = winfo.get("shape", [])
        name = _sanitize_name(wname).upper()
        for i, dim in enumerate(shape):
            if isinstance(dim, int) and dim > 0:
                dim_name = f"DIM_{name}_{i}"
                if dim_name not in defined:
                    lines.append(f"#define {dim_name} {dim}")
                    defined.add(dim_name)

    return "\n".join(lines)


def build_tensor_map(trix: Dict[str, Any]) -> Dict[str, Dict]:
    """Build mapping from tensor names to C identifiers and metadata."""
    tensor_map = {}
    buffer_idx = 0

    # Inputs
    for i, inp in enumerate(trix.get("inputs", [])):
        name = inp["name"]
        tensor_map[name] = {
            "kind": "input",
            "c_name": f"input_{i}" if i > 0 else "input",
            "shape": inp.get("shape", []),
            "size": _get_tensor_size(inp.get("shape", [])),
        }

    # Weights
    for wname, winfo in trix.get("weights", {}).items():
        tensor_map[wname] = {
            "kind": "weight",
            "c_name": "W_" + _sanitize_name(wname),
            "shape": winfo.get("shape", []),
            "size": winfo.get("size", 0),
        }

    # Outputs
    for i, out in enumerate(trix.get("outputs", [])):
        name = out["name"]
        tensor_map[name] = {
            "kind": "output",
            "c_name": f"output_{i}" if i > 0 else "output",
            "shape": out.get("shape", []),
            "size": _get_tensor_size(out.get("shape", [])),
        }

    # Intermediates (outputs of shapes that aren't final outputs)
    output_names = {out["name"] for out in trix.get("outputs", [])}
    for shape in trix.get("shapes", []):
        for out_name in shape.get("outputs", []):
            if out_name not in tensor_map:
                # This is an intermediate
                tensor_map[out_name] = {
                    "kind": "intermediate",
                    "c_name": f"t{buffer_idx}",
                    "shape": [],  # Will be inferred
                    "size": 0,    # Will be computed
                }
                buffer_idx += 1

    return tensor_map


def infer_tensor_shapes(trix: Dict[str, Any], tensor_map: Dict) -> None:
    """Infer shapes for intermediate tensors based on operations."""
    for shape in trix.get("shapes", []):
        kind = shape.get("kind", "")
        inputs = shape.get("inputs", [])
        outputs = shape.get("outputs", [])

        if not outputs:
            continue

        out_name = outputs[0]
        if out_name not in tensor_map:
            continue

        out_info = tensor_map[out_name]
        if out_info["shape"]:  # Already has shape
            continue

        # Infer based on operation type
        if kind in ("ADD", "SUB", "MUL", "DIV", "RELU", "GELU", "SIGMOID", "TANH", "SILU"):
            # Element-wise: output shape = first input shape
            if inputs and inputs[0] in tensor_map:
                out_info["shape"] = tensor_map[inputs[0]]["shape"].copy()
                out_info["size"] = tensor_map[inputs[0]]["size"]

        elif kind == "MATMUL":
            # MatMul: [M, K] x [K, N] -> [M, N]
            if len(inputs) >= 2 and inputs[0] in tensor_map and inputs[1] in tensor_map:
                shape_a = tensor_map[inputs[0]]["shape"]
                shape_b = tensor_map[inputs[1]]["shape"]
                if len(shape_a) >= 2 and len(shape_b) >= 2:
                    out_info["shape"] = [shape_a[-2], shape_b[-1]]
                    out_info["size"] = shape_a[-2] * shape_b[-1]

        elif kind == "SOFTMAX":
            # Softmax: same shape as input
            if inputs and inputs[0] in tensor_map:
                out_info["shape"] = tensor_map[inputs[0]]["shape"].copy()
                out_info["size"] = tensor_map[inputs[0]]["size"]

        elif kind == "LAYER_NORM":
            # LayerNorm: same shape as input
            if inputs and inputs[0] in tensor_map:
                out_info["shape"] = tensor_map[inputs[0]]["shape"].copy()
                out_info["size"] = tensor_map[inputs[0]]["size"]

        elif kind == "IDENTITY":
            # Identity: passthrough
            if inputs and inputs[0] in tensor_map:
                out_info["shape"] = tensor_map[inputs[0]]["shape"].copy()
                out_info["size"] = tensor_map[inputs[0]]["size"]


def emit_shape_call(shape: Dict, tensor_map: Dict) -> str:
    """Emit C code for a single shape invocation."""
    kind = shape.get("kind", "")
    inputs = shape.get("inputs", [])
    outputs = shape.get("outputs", [])
    attrs = shape.get("attributes", {})

    if not outputs:
        return f"    /* No output for {kind} */"

    # Get C names
    def get_c(name):
        if name in tensor_map:
            return tensor_map[name]["c_name"]
        return "_unknown_"

    def get_size(name):
        if name in tensor_map:
            return tensor_map[name].get("size", 0)
        return 0

    def get_shape(name):
        if name in tensor_map:
            return tensor_map[name].get("shape", [])
        return []

    out_c = get_c(outputs[0])
    out_size = get_size(outputs[0])

    # Element-wise activations (scalar loop)
    if kind in ("RELU", "GELU", "SIGMOID", "TANH", "SILU"):
        in_c = get_c(inputs[0]) if inputs else "input"
        func_name = f"trix_onnx_{kind.lower()}"
        return f"    for (int i = 0; i < {out_size}; i++) {out_c}[i] = {func_name}({in_c}[i]);"

    # Element-wise binary ops
    if kind in ("ADD", "SUB", "MUL", "DIV"):
        if len(inputs) < 2:
            return f"    /* {kind} needs 2 inputs */"
        a_c = get_c(inputs[0])
        b_c = get_c(inputs[1])
        func_name = f"trix_onnx_{kind.lower()}"
        return f"    {func_name}({a_c}, {b_c}, {out_c}, {out_size});"

    # MatMul
    if kind == "MATMUL":
        if len(inputs) < 2:
            return f"    /* MATMUL needs 2 inputs */"
        a_c = get_c(inputs[0])
        b_c = get_c(inputs[1])
        shape_a = get_shape(inputs[0])
        shape_b = get_shape(inputs[1])

        # [M, K] x [K, N] -> [M, N]
        M = shape_a[-2] if len(shape_a) >= 2 else 1
        K = shape_a[-1] if len(shape_a) >= 1 else 1
        N = shape_b[-1] if len(shape_b) >= 1 else 1

        return f"    trix_onnx_matmul({a_c}, {b_c}, {out_c}, {M}, {N}, {K});"

    # Gemm: Y = alpha * A @ B + beta * C
    if kind == "GEMM":
        if len(inputs) < 2:
            return f"    /* GEMM needs at least 2 inputs */"
        a_c = get_c(inputs[0])
        b_c = get_c(inputs[1])
        c_c = get_c(inputs[2]) if len(inputs) > 2 else "NULL"
        shape_a = get_shape(inputs[0])
        shape_b = get_shape(inputs[1])

        M = shape_a[-2] if len(shape_a) >= 2 else 1
        K = shape_a[-1] if len(shape_a) >= 1 else 1
        N = shape_b[-1] if len(shape_b) >= 1 else 1

        alpha = attrs.get("alpha", 1.0)
        beta = attrs.get("beta", 1.0)

        return f"    trix_onnx_gemm({a_c}, {b_c}, {c_c}, {out_c}, {M}, {N}, {K}, {_format_float(alpha)}, {_format_float(beta)});"

    # Softmax
    if kind == "SOFTMAX":
        in_c = get_c(inputs[0]) if inputs else "input"
        return f"    trix_onnx_softmax({in_c}, {out_c}, {out_size});"

    # LayerNorm
    if kind == "LAYER_NORM":
        if len(inputs) < 3:
            return f"    /* LAYER_NORM needs input, gamma, beta */"
        in_c = get_c(inputs[0])
        gamma_c = get_c(inputs[1])
        beta_c = get_c(inputs[2])
        eps = attrs.get("epsilon", 1e-5)
        return f"    trix_onnx_layer_norm({in_c}, {gamma_c}, {beta_c}, {out_c}, {out_size}, {_format_float(eps)});"

    # Neg
    if kind == "NEG":
        in_c = get_c(inputs[0]) if inputs else "input"
        return f"    trix_onnx_neg({in_c}, {out_c}, {out_size});"

    # Abs
    if kind == "ABS":
        in_c = get_c(inputs[0]) if inputs else "input"
        return f"    for (int i = 0; i < {out_size}; i++) {out_c}[i] = fabsf({in_c}[i]);"

    # Sqrt
    if kind == "SQRT":
        in_c = get_c(inputs[0]) if inputs else "input"
        return f"    for (int i = 0; i < {out_size}; i++) {out_c}[i] = sqrtf({in_c}[i]);"

    # Exp
    if kind == "EXP":
        in_c = get_c(inputs[0]) if inputs else "input"
        return f"    for (int i = 0; i < {out_size}; i++) {out_c}[i] = expf({in_c}[i]);"

    # Log
    if kind == "LOG":
        in_c = get_c(inputs[0]) if inputs else "input"
        return f"    for (int i = 0; i < {out_size}; i++) {out_c}[i] = logf({in_c}[i]);"

    # Identity / Dropout (passthrough)
    if kind == "IDENTITY":
        in_c = get_c(inputs[0]) if inputs else "input"
        return f"    memcpy({out_c}, {in_c}, {out_size} * sizeof(float));"

    # ReduceMean
    if kind == "REDUCE_MEAN":
        in_c = get_c(inputs[0]) if inputs else "input"
        in_size = get_size(inputs[0]) if inputs else out_size
        return f"    trix_onnx_reduce_mean({in_c}, {out_c}, {in_size}, {out_size});"

    # ReduceSum
    if kind == "REDUCE_SUM":
        in_c = get_c(inputs[0]) if inputs else "input"
        in_size = get_size(inputs[0]) if inputs else out_size
        return f"    trix_onnx_reduce_sum({in_c}, {out_c}, {in_size}, {out_size});"

    # Reshape (no-op in C, just pointer assignment if possible)
    if kind == "RESHAPE":
        in_c = get_c(inputs[0]) if inputs else "input"
        return f"    /* Reshape: {in_c} -> {out_c} (logical, no copy needed) */\n    memcpy({out_c}, {in_c}, {out_size} * sizeof(float));"

    # Transpose (2D only)
    if kind == "TRANSPOSE":
        in_c = get_c(inputs[0]) if inputs else "input"
        in_shape = get_shape(inputs[0]) if inputs else []
        if len(in_shape) == 2:
            rows, cols = in_shape
            return f"    trix_onnx_transpose_2d({in_c}, {out_c}, {rows}, {cols});"
        return f"    /* TRANSPOSE: unsupported shape {in_shape} */"

    # Unsupported
    return f"    /* TODO: {kind} not yet implemented */"


def emit_forward(trix: Dict[str, Any], tensor_map: Dict) -> str:
    """Emit the forward pass function."""
    lines = []

    # Collect intermediate buffer sizes
    intermediates = []
    for name, info in tensor_map.items():
        if info["kind"] == "intermediate":
            size = max(info.get("size", 0), 1)  # At least 1
            intermediates.append((info["c_name"], size))

    # Buffer declarations
    if intermediates:
        lines.append("    /* Intermediate buffers */")
        for c_name, size in intermediates:
            lines.append(f"    float* {c_name} = (float*)alloca({size} * sizeof(float));")
        lines.append("")

    # Shape calls
    lines.append("    /* Computation graph */")
    for i, shape in enumerate(trix.get("shapes", [])):
        onnx_op = shape.get("onnx_op", shape.get("kind", "unknown"))
        lines.append(f"    /* [{i}] {onnx_op} */")
        lines.append(emit_shape_call(shape, tensor_map))
        lines.append("")

    return "\n".join(lines)


def emit_main(trix: Dict[str, Any], tensor_map: Dict, model_name: str) -> str:
    """Emit main() for standalone binary."""
    inputs = trix.get("inputs", [])
    outputs = trix.get("outputs", [])

    if not inputs or not outputs:
        return "/* No main: missing inputs or outputs */"

    inp = inputs[0]
    out = outputs[0]
    inp_size = _get_tensor_size(inp.get("shape", []))
    out_size = _get_tensor_size(out.get("shape", []))

    return f'''#ifdef TRIXC_STANDALONE

int main(int argc, char** argv) {{
    /* Allocate I/O buffers */
    float* input = (float*)malloc({inp_size} * sizeof(float));
    float* output = (float*)malloc({out_size} * sizeof(float));

    if (!input || !output) {{
        fprintf(stderr, "Memory allocation failed\\n");
        return 1;
    }}

    /* Read input */
    FILE* fin = (argc > 1) ? fopen(argv[1], "rb") : stdin;
    if (!fin) {{
        perror("Failed to open input");
        return 1;
    }}
    size_t read = fread(input, sizeof(float), {inp_size}, fin);
    if (read != {inp_size}) {{
        fprintf(stderr, "Expected {inp_size} floats, got %zu\\n", read);
        if (fin != stdin) fclose(fin);
        return 1;
    }}
    if (fin != stdin) fclose(fin);

    /* Run model */
    {model_name}_forward(input, output);

    /* Write output */
    FILE* fout = (argc > 2) ? fopen(argv[2], "wb") : stdout;
    if (!fout) {{
        perror("Failed to open output");
        return 1;
    }}
    fwrite(output, sizeof(float), {out_size}, fout);
    if (fout != stdout) fclose(fout);

    /* Cleanup */
    free(input);
    free(output);

    return 0;
}}

#endif /* TRIXC_STANDALONE */'''


def generate_c_code(trix: Dict[str, Any], standalone: bool = True) -> str:
    """Generate complete C source from Octave IR."""
    model_name = _sanitize_name(trix.get("name", "model"))

    # Build tensor map
    tensor_map = build_tensor_map(trix)
    infer_tensor_shapes(trix, tensor_map)

    # Get input/output info
    inputs = trix.get("inputs", [])
    outputs = trix.get("outputs", [])

    inp_size = _get_tensor_size(inputs[0].get("shape", [])) if inputs else 0
    out_size = _get_tensor_size(outputs[0].get("shape", [])) if outputs else 0

    # Build sections
    weights_section = emit_weights(trix.get("weights", {}))
    dims_section = emit_dimensions(trix)
    forward_body = emit_forward(trix, tensor_map)
    main_section = emit_main(trix, tensor_map, model_name) if standalone else ""

    # Assemble
    code = f'''/* {model_name}.c
 * Generated by trixc from {trix.get("name", "unknown")}
 *
 * Compile: gcc -O3 -DTRIXC_STANDALONE {model_name}.c -o {model_name} -lm
 *
 * "ONNX is notation. Shapes are truth. C is execution."
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <alloca.h>

/* Include TRIXC shapes - adjust path as needed */
#ifndef TRIXC_SHAPES_INCLUDED
#include "trixc/onnx_shapes.h"
#define TRIXC_SHAPES_INCLUDED
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Configuration
 * ═══════════════════════════════════════════════════════════════════════════ */

#define MODEL_INPUT_SIZE {inp_size}
#define MODEL_OUTPUT_SIZE {out_size}

{dims_section}

/* ═══════════════════════════════════════════════════════════════════════════
 * Weights (static const - goes in .rodata section)
 * ═══════════════════════════════════════════════════════════════════════════ */

{weights_section}

/* ═══════════════════════════════════════════════════════════════════════════
 * Forward Pass
 * ═══════════════════════════════════════════════════════════════════════════ */

void {model_name}_forward(const float* input, float* output) {{
{forward_body}
}}

/* ═══════════════════════════════════════════════════════════════════════════
 * Standalone Main
 * ═══════════════════════════════════════════════════════════════════════════ */

{main_section}
'''

    return code


def generate_c_header(trix: Dict[str, Any]) -> str:
    """Generate a simple C header for the model."""
    model_name = _sanitize_name(trix.get("name", "model"))

    lines = [
        f"/* Generated by onnx2trix from {trix['name']} */",
        "",
        f"#ifndef {model_name.upper()}_H",
        f"#define {model_name.upper()}_H",
        "",
        '#include "trixc/onnx_shapes.h"',
        "",
    ]

    # Forward declaration
    if trix["inputs"] and trix["outputs"]:
        inp = trix["inputs"][0]
        out = trix["outputs"][0]
        lines.append(f"/* Input: {inp['name']} {inp.get('shape', '?')} */")
        lines.append(f"/* Output: {out['name']} {out.get('shape', '?')} */")
        lines.append("")
        lines.append(f"void {model_name}_forward(const float* input, float* output);")

    lines.append("")
    lines.append(f"#endif /* {model_name.upper()}_H */")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert ONNX models to TRIXC native code",
        epilog='"ONNX is notation. Shapes are truth. C is execution."'
    )
    parser.add_argument("input", help="Input ONNX model (.onnx)")
    parser.add_argument("output", help="Output file (.trix for IR, .c for C code)")
    parser.add_argument("--emit-c", action="store_true",
                        help="Generate C code instead of Octave IR")
    parser.add_argument("--standalone", action="store_true", default=True,
                        help="Include main() in generated C (default: True)")
    parser.add_argument("--no-standalone", action="store_true",
                        help="Omit main() from generated C (library mode)")
    parser.add_argument("--emit-weights", action="store_true", default=True,
                        help="Include weight data in output (default: True)")
    parser.add_argument("--no-weights", action="store_true",
                        help="Exclude weight data (only shapes)")
    parser.add_argument("--precision", choices=["fp16", "fp32", "fp64"],
                        default="fp32", help="Weight precision (default: fp32)")
    parser.add_argument("--emit-header", action="store_true",
                        help="Also generate C header file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print conversion details")

    args = parser.parse_args()

    if not HAS_ONNX:
        print("Error: onnx package required. Install with: pip install onnx")
        sys.exit(1)

    # Convert ONNX to Octave IR
    emit_weights = not args.no_weights
    converter = ONNX2TrixConverter(emit_weights=emit_weights, precision=args.precision)

    if args.verbose:
        print(f"Converting: {args.input}")

    try:
        trix = converter.convert(args.input)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if args.verbose:
        print(f"  Shapes: {len(trix['shapes'])}")
        print(f"  Weights: {len(trix['weights'])}")
        print(f"  Inputs: {[i['name'] for i in trix['inputs']]}")
        print(f"  Outputs: {[o['name'] for o in trix['outputs']]}")

        if converter.warnings:
            print(f"  Warnings: {len(converter.warnings)}")
            for w in converter.warnings:
                print(f"    - {w}")

    # Determine output mode
    output_path = Path(args.output)
    emit_c = args.emit_c or output_path.suffix == '.c'

    if emit_c:
        # Generate C code
        standalone = not args.no_standalone
        c_code = generate_c_code(trix, standalone=standalone)

        with open(args.output, 'w') as f:
            f.write(c_code)

        print(f"Generated: {args.output}")
        print(f"  Compile: gcc -O3 -DTRIXC_STANDALONE -I/path/to/trixc/include {args.output} -o {output_path.stem} -lm")

        # Also generate header if requested
        if args.emit_header:
            header_path = output_path.with_suffix('.h')
            header = generate_c_header(trix)
            with open(header_path, 'w') as f:
                f.write(header)
            print(f"Generated: {header_path}")

    else:
        # Write Octave IR (JSON)
        with open(args.output, 'w') as f:
            json.dump(trix, f, indent=2)

        print(f"Converted: {args.input} → {args.output}")

        # Optionally generate header
        if args.emit_header:
            header_path = output_path.with_suffix('.h')
            header = generate_c_header(trix)
            with open(header_path, 'w') as f:
                f.write(header)
            print(f"Generated: {header_path}")


if __name__ == "__main__":
    main()
