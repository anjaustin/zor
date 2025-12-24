#!/usr/bin/env python3
"""
End-to-end demo: ONNX -> C -> Compile -> Run

This script demonstrates the complete TRIXC pipeline:
1. Create a simple MLP model in ONNX format
2. Convert it to C code using onnx2trix.py
3. Compile the C code with gcc
4. Run the binary and verify output

"ONNX is notation. Shapes are truth. C is execution."
"""

import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# Add tools to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from onnx2trix import ONNX2TrixConverter, generate_c_code


def create_simple_mlp(path: str):
    """
    Create a simple MLP: input(1,4) -> MatMul -> ReLU -> output(1,2)

    Weight matrix (4x2):
    [[0.1, 0.2],
     [0.3, 0.4],
     [0.5, 0.6],
     [0.7, 0.8]]
    """
    weight_data = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
    ], dtype=np.float32)

    weight_init = numpy_helper.from_array(weight_data, name="weight")

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
    print(f"Created ONNX model: {path}")
    return weight_data


def compute_expected_output(input_data: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Compute expected output using NumPy."""
    matmul_out = input_data @ weight
    relu_out = np.maximum(0, matmul_out)
    return relu_out


def main():
    print("=" * 60)
    print("TRIXC End-to-End Demo: ONNX -> C -> Compile -> Run")
    print("=" * 60)
    print()

    trixc_root = Path(__file__).parent.parent
    include_path = trixc_root / "include"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Step 1: Create ONNX model
        print("Step 1: Creating ONNX model...")
        onnx_path = tmpdir / "model.onnx"
        weight = create_simple_mlp(str(onnx_path))

        # Step 2: Convert to C
        print("\nStep 2: Converting ONNX to C...")
        converter = ONNX2TrixConverter(emit_weights=True)
        trix = converter.convert(str(onnx_path))

        c_code = generate_c_code(trix, standalone=True)
        c_path = tmpdir / "model.c"
        with open(c_path, 'w') as f:
            f.write(c_code)
        print(f"Generated C code: {c_path}")
        print(f"  - Lines: {len(c_code.splitlines())}")
        print(f"  - Size: {len(c_code)} bytes")

        # Step 3: Compile
        print("\nStep 3: Compiling C code...")
        exe_path = tmpdir / "model"
        compile_cmd = [
            "gcc", "-O3", "-DTRIXC_STANDALONE",
            f"-I{include_path}",
            str(c_path),
            "-o", str(exe_path),
            "-lm"
        ]
        print(f"  Command: {' '.join(compile_cmd)}")

        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Compilation failed!")
            print(f"stderr: {result.stderr}")
            return 1

        exe_size = exe_path.stat().st_size
        print(f"Compiled successfully: {exe_path}")
        print(f"  - Binary size: {exe_size} bytes ({exe_size / 1024:.1f} KB)")

        # Step 4: Prepare input
        print("\nStep 4: Preparing test input...")
        input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        print(f"  Input: {input_data.flatten().tolist()}")

        input_path = tmpdir / "input.bin"
        with open(input_path, 'wb') as f:
            f.write(input_data.tobytes())

        # Step 5: Run
        print("\nStep 5: Running compiled model...")
        output_path = tmpdir / "output.bin"

        run_cmd = [str(exe_path), str(input_path), str(output_path)]
        result = subprocess.run(run_cmd, capture_output=True)

        if result.returncode != 0:
            print(f"Execution failed!")
            print(f"stderr: {result.stderr.decode()}")
            return 1

        # Step 6: Verify
        print("\nStep 6: Verifying output...")
        with open(output_path, 'rb') as f:
            output_bytes = f.read()

        output_data = np.frombuffer(output_bytes, dtype=np.float32).reshape(1, 2)
        expected = compute_expected_output(input_data, weight)

        print(f"  TRIXC output:    {output_data.flatten().tolist()}")
        print(f"  Expected output: {expected.flatten().tolist()}")

        # Check if they match
        if np.allclose(output_data, expected, rtol=1e-5, atol=1e-6):
            print("\n" + "=" * 60)
            print("SUCCESS! Output matches expected values.")
            print("=" * 60)
            return 0
        else:
            diff = np.abs(output_data - expected).max()
            print(f"\nFAILED! Max difference: {diff}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
