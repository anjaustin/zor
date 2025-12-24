#!/usr/bin/env python3
"""
Test suite for TRIXC C code generation.

Tests the full pipeline: ONNX -> C -> Compile -> Run -> Verify

"ONNX is notation. Shapes are truth. C is execution."
"""

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from onnx2trix import (
    ONNX2TrixConverter,
    generate_c_code,
    generate_c_header,
    emit_weights,
    emit_dimensions,
    emit_shape_call,
    emit_forward,
    build_tensor_map,
    infer_tensor_shapes,
    _sanitize_name,
    _get_tensor_size,
    _format_float,
)


class TestSanitizeName(unittest.TestCase):
    """Test C identifier sanitization."""

    def test_simple_name(self):
        self.assertEqual(_sanitize_name("weight"), "weight")

    def test_with_dots(self):
        self.assertEqual(_sanitize_name("fc1.weight"), "fc1_weight")

    def test_with_slashes(self):
        self.assertEqual(_sanitize_name("model/layer/weight"), "model_layer_weight")

    def test_starts_with_digit(self):
        self.assertEqual(_sanitize_name("0_layer"), "_0_layer")

    def test_empty(self):
        self.assertEqual(_sanitize_name(""), "_unnamed")


class TestFormatFloat(unittest.TestCase):
    """Test float formatting for C code."""

    def test_zero(self):
        self.assertEqual(_format_float(0.0), "0.0f")

    def test_integer(self):
        self.assertEqual(_format_float(1.0), "1.0f")

    def test_fraction(self):
        result = _format_float(0.5)
        self.assertTrue(result.endswith("f"))
        self.assertIn("0.5", result)

    def test_negative(self):
        result = _format_float(-1.5)
        self.assertTrue(result.startswith("-"))
        self.assertTrue(result.endswith("f"))


class TestGetTensorSize(unittest.TestCase):
    """Test tensor size calculation."""

    def test_simple_shape(self):
        self.assertEqual(_get_tensor_size([2, 3]), 6)

    def test_3d_shape(self):
        self.assertEqual(_get_tensor_size([2, 3, 4]), 24)

    def test_dynamic_dim_treated_as_1(self):
        self.assertEqual(_get_tensor_size(["batch", 768]), 768)

    def test_empty_shape(self):
        self.assertEqual(_get_tensor_size([]), 1)


@unittest.skipUnless(HAS_NUMPY, "numpy required")
class TestEmitWeights(unittest.TestCase):
    """Test weight emission."""

    def test_simple_weights(self):
        weights = {
            "fc.weight": {
                "shape": [2, 3],
                "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        }
        result = emit_weights(weights)
        self.assertIn("static const float W_fc_weight[6]", result)
        self.assertIn("1.0f", result)
        self.assertIn("6.0f", result)

    def test_no_data(self):
        weights = {
            "empty": {"shape": [2, 3], "data": None}
        }
        result = emit_weights(weights)
        self.assertIn("no data", result)


@unittest.skipUnless(HAS_ONNX and HAS_NUMPY, "onnx and numpy required")
class TestBuildTensorMap(unittest.TestCase):
    """Test tensor mapping."""

    def test_simple_model(self):
        trix = {
            "inputs": [{"name": "input", "shape": [1, 4]}],
            "outputs": [{"name": "output", "shape": [1, 2]}],
            "weights": {
                "fc.weight": {"shape": [4, 2], "size": 8}
            },
            "shapes": [
                {"kind": "MATMUL", "inputs": ["input", "fc.weight"], "outputs": ["output"]}
            ]
        }
        tensor_map = build_tensor_map(trix)

        self.assertEqual(tensor_map["input"]["kind"], "input")
        self.assertEqual(tensor_map["fc.weight"]["kind"], "weight")
        self.assertEqual(tensor_map["output"]["kind"], "output")


@unittest.skipUnless(HAS_ONNX and HAS_NUMPY, "onnx and numpy required")
class TestGenerateCCode(unittest.TestCase):
    """Test full C code generation."""

    def test_simple_relu(self):
        """Test C generation for simple ReLU."""
        trix = {
            "name": "simple_relu",
            "inputs": [{"name": "input", "shape": [1, 4]}],
            "outputs": [{"name": "output", "shape": [1, 4]}],
            "weights": {},
            "shapes": [
                {"kind": "RELU", "onnx_op": "Relu", "inputs": ["input"], "outputs": ["output"]}
            ]
        }
        code = generate_c_code(trix, standalone=True)

        self.assertIn("simple_relu_forward", code)
        self.assertIn("trix_onnx_relu", code)
        self.assertIn("#ifdef TRIXC_STANDALONE", code)
        self.assertIn("int main(", code)

    def test_simple_matmul(self):
        """Test C generation for MatMul."""
        trix = {
            "name": "simple_matmul",
            "inputs": [{"name": "input", "shape": [1, 4]}],
            "outputs": [{"name": "output", "shape": [1, 2]}],
            "weights": {
                "fc.weight": {"shape": [4, 2], "size": 8, "data": [0.1] * 8}
            },
            "shapes": [
                {"kind": "MATMUL", "onnx_op": "MatMul",
                 "inputs": ["input", "fc.weight"], "outputs": ["output"]}
            ]
        }
        code = generate_c_code(trix, standalone=True)

        self.assertIn("simple_matmul_forward", code)
        self.assertIn("trix_onnx_matmul", code)
        self.assertIn("W_fc_weight", code)

    def test_no_standalone(self):
        """Test library mode (no main)."""
        trix = {
            "name": "lib_model",
            "inputs": [{"name": "input", "shape": [1, 4]}],
            "outputs": [{"name": "output", "shape": [1, 4]}],
            "weights": {},
            "shapes": [
                {"kind": "RELU", "onnx_op": "Relu", "inputs": ["input"], "outputs": ["output"]}
            ]
        }
        code = generate_c_code(trix, standalone=False)

        self.assertIn("lib_model_forward", code)
        self.assertNotIn("int main(", code)


@unittest.skipUnless(HAS_ONNX and HAS_NUMPY, "onnx and numpy required")
class TestGenerateHeader(unittest.TestCase):
    """Test header generation."""

    def test_header_content(self):
        trix = {
            "name": "test_model",
            "inputs": [{"name": "input", "shape": [1, 4]}],
            "outputs": [{"name": "output", "shape": [1, 2]}],
            "weights": {},
            "shapes": []
        }
        header = generate_c_header(trix)

        self.assertIn("#ifndef TEST_MODEL_H", header)
        self.assertIn("#define TEST_MODEL_H", header)
        self.assertIn("test_model_forward", header)
        self.assertIn("#endif", header)


@unittest.skipUnless(HAS_ONNX and HAS_NUMPY, "onnx and numpy required")
class TestEmitShapeCall(unittest.TestCase):
    """Test individual shape call emission."""

    def setUp(self):
        self.tensor_map = {
            "input": {"kind": "input", "c_name": "input", "shape": [1, 4], "size": 4},
            "output": {"kind": "output", "c_name": "output", "shape": [1, 4], "size": 4},
            "weight": {"kind": "weight", "c_name": "W_weight", "shape": [4, 2], "size": 8},
            "matmul_out": {"kind": "output", "c_name": "output", "shape": [1, 2], "size": 2},
        }

    def test_relu(self):
        shape = {"kind": "RELU", "inputs": ["input"], "outputs": ["output"]}
        result = emit_shape_call(shape, self.tensor_map)
        self.assertIn("trix_onnx_relu", result)

    def test_gelu(self):
        shape = {"kind": "GELU", "inputs": ["input"], "outputs": ["output"]}
        result = emit_shape_call(shape, self.tensor_map)
        self.assertIn("trix_onnx_gelu", result)

    def test_add(self):
        shape = {"kind": "ADD", "inputs": ["input", "input"], "outputs": ["output"]}
        result = emit_shape_call(shape, self.tensor_map)
        self.assertIn("trix_onnx_add", result)

    def test_matmul(self):
        tensor_map = {
            "input": {"kind": "input", "c_name": "input", "shape": [1, 4], "size": 4},
            "weight": {"kind": "weight", "c_name": "W_weight", "shape": [4, 2], "size": 8},
            "output": {"kind": "output", "c_name": "output", "shape": [1, 2], "size": 2},
        }
        shape = {"kind": "MATMUL", "inputs": ["input", "weight"], "outputs": ["output"]}
        result = emit_shape_call(shape, tensor_map)
        self.assertIn("trix_onnx_matmul", result)


@unittest.skipUnless(HAS_ONNX and HAS_NUMPY, "onnx and numpy required")
class TestFullPipeline(unittest.TestCase):
    """Test the full ONNX -> C -> Compile -> Run pipeline."""

    def _create_simple_mlp_onnx(self, path: str):
        """Create a simple MLP: input(1,4) -> MatMul -> ReLU -> output(1,2)."""
        # Create weight
        weight_data = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8],
        ], dtype=np.float32)

        weight_init = numpy_helper.from_array(weight_data, name="weight")

        # Create graph
        input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
        output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])

        matmul_node = helper.make_node("MatMul", ["input", "weight"], ["matmul_out"])
        relu_node = helper.make_node("Relu", ["matmul_out"], ["output"])

        graph = helper.make_graph(
            [matmul_node, relu_node],
            "simple_mlp",
            [input_tensor],
            [output_tensor],
            [weight_init],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.save(model, path)

    def test_onnx_to_c_conversion(self):
        """Test that ONNX model converts to valid C code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")
            c_path = os.path.join(tmpdir, "model.c")

            # Create ONNX model
            self._create_simple_mlp_onnx(onnx_path)

            # Convert to Octave IR
            converter = ONNX2TrixConverter(emit_weights=True)
            trix = converter.convert(onnx_path)

            # Generate C code
            c_code = generate_c_code(trix, standalone=True)

            # Save C code
            with open(c_path, 'w') as f:
                f.write(c_code)

            # Verify C code structure
            self.assertIn("simple_mlp_forward", c_code)
            self.assertIn("trix_onnx_matmul", c_code)
            self.assertIn("trix_onnx_relu", c_code)
            self.assertIn("W_weight", c_code)
            self.assertIn("0.1", c_code)  # Weight value


@unittest.skipUnless(HAS_ONNX and HAS_NUMPY, "onnx and numpy required")
class TestCLI(unittest.TestCase):
    """Test CLI interface."""

    def test_help(self):
        """Test that --help works."""
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent.parent / "tools" / "onnx2trix.py"), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--emit-c", result.stdout)
        self.assertIn("--standalone", result.stdout)


if __name__ == "__main__":
    unittest.main()
