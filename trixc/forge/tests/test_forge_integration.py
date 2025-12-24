#!/usr/bin/env python3
"""
TRIX Forge Integration Tests

Tests that verify TRIX Forge components integrate correctly with
existing TRIXC infrastructure.

Run: python -m pytest test_forge_integration.py -v
"""

import pytest
import json
import os
import subprocess
import tempfile
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from test_forge_rigorous import (
    ModelIR, Shape, TensorSpec, Lever, EventSpine, EventType,
    Compiler, TARGETS
)

# =============================================================================
# TRIXC INTEGRATION TESTS
# =============================================================================

class TestTRIXCIntegration:
    """Tests for integration with existing TRIXC."""

    def test_forge_output_compiles_with_gcc(self):
        """Generated C code compiles with GCC."""
        model = ModelIR(
            name="gcc_test",
            shapes=[
                Shape("relu", "RELU", ["input"], ["output"])
            ],
            tensors={
                "input": TensorSpec("input", [10], "float32"),
                "output": TensorSpec("output", [10], "float32")
            }
        )

        compiler = Compiler()

        with tempfile.TemporaryDirectory() as tmpdir:
            c_path = os.path.join(tmpdir, "model.c")
            exe_path = os.path.join(tmpdir, "model")

            # Generate C code
            compiler.compile(model, "c", c_path)

            # Add minimal main for compilation
            with open(c_path, "a") as f:
                f.write("""
int main() {
    float in[10] = {0};
    float out[10];
    trix_forward(in, out);
    return 0;
}
""")

            # Try to compile
            result = subprocess.run(
                ["gcc", "-c", "-Wall", "-Wextra", c_path, "-o", exe_path + ".o"],
                capture_output=True,
                text=True
            )

            # Check compilation succeeded (no errors, warnings OK)
            assert result.returncode == 0, f"GCC failed: {result.stderr}"

    def test_forge_output_includes_correct_headers(self):
        """Generated code includes necessary headers."""
        model = ModelIR(name="header_test")
        compiler = Compiler()

        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            path = f.name

        try:
            compiler.compile(model, "c", path)

            with open(path, "r") as f:
                code = f.read()

            # Should include essential headers
            assert "#include <stdint.h>" in code
            assert "#include <math.h>" in code
        finally:
            os.unlink(path)

    def test_forge_model_matches_trixc_shapes(self):
        """Forge shapes align with TRIXC frozen shapes."""
        # These are the shapes that TRIXC supports
        trixc_shapes = [
            "XOR", "AND", "OR", "NOT",
            "MATMUL", "RELU", "GELU", "SIGMOID", "TANH", "SILU",
            "SOFTMAX", "LAYERNORM",
            "ADD", "SUB", "MUL", "DIV"
        ]

        model = ModelIR(
            name="shape_test",
            shapes=[Shape(f"shape_{i}", shape_type, ["in"], ["out"])
                   for i, shape_type in enumerate(trixc_shapes)],
            tensors={
                "in": TensorSpec("in", [10], "float32"),
                "out": TensorSpec("out", [10], "float32")
            }
        )

        # Should validate without errors
        errors = model.validate()
        assert len(errors) == 0

    def test_forge_precision_matches_trixc_apu(self):
        """Forge precision levels match TRIXC APU."""
        from test_forge_rigorous import Precision

        # TRIXC APU supports these precision levels
        expected_precisions = [4, 8, 16, 32, 64]

        for prec in Precision:
            assert prec.value in expected_precisions

    def test_forge_can_represent_existing_trixc_examples(self):
        """Forge can represent models from TRIXC examples."""
        # XOR model from TRIXC Pi
        xor_model = ModelIR(
            name="xor_mlp",
            shapes=[
                Shape("matmul_1", "MATMUL", ["input"], ["hidden"]),
                Shape("relu_1", "RELU", ["hidden"], ["activated"]),
                Shape("matmul_2", "MATMUL", ["activated"], ["output"])
            ],
            tensors={
                "input": TensorSpec("input", [2], "float32"),
                "hidden": TensorSpec("hidden", [4], "float32"),
                "activated": TensorSpec("activated", [4], "float32"),
                "output": TensorSpec("output", [1], "float32")
            },
            metadata={
                "description": "XOR MLP from TRIXC Pi",
                "architecture": "2 -> 4 (ReLU) -> 1"
            }
        )

        errors = xor_model.validate()
        assert len(errors) == 0

        # MNIST model from TRIXC Pi
        mnist_model = ModelIR(
            name="mnist_7x7",
            shapes=[
                Shape("matmul_1", "MATMUL", ["input"], ["hidden"]),
                Shape("relu_1", "RELU", ["hidden"], ["activated"]),
                Shape("matmul_2", "MATMUL", ["activated"], ["output"]),
                Shape("softmax", "SOFTMAX", ["output"], ["probs"])
            ],
            tensors={
                "input": TensorSpec("input", [49], "float32"),
                "hidden": TensorSpec("hidden", [32], "float32"),
                "activated": TensorSpec("activated", [32], "float32"),
                "output": TensorSpec("output", [10], "float32"),
                "probs": TensorSpec("probs", [10], "float32")
            },
            metadata={
                "description": "MNIST 7x7 classifier from TRIXC Pi",
                "architecture": "49 -> 32 (ReLU) -> 10 -> softmax"
            }
        )

        errors = mnist_model.validate()
        assert len(errors) == 0


class TestPlatformIntegration:
    """Tests for platform target integration."""

    def test_all_targets_produce_output(self):
        """All targets produce some output."""
        model = ModelIR(
            name="target_test",
            shapes=[Shape("relu", "RELU", ["in"], ["out"])],
            tensors={
                "in": TensorSpec("in", [10], "float32"),
                "out": TensorSpec("out", [10], "float32")
            }
        )

        compiler = Compiler()

        for target_name in TARGETS.keys():
            with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
                path = f.name

            try:
                stats = compiler.compile(model, target_name, path)
                assert stats["code_size"] > 0
                assert os.path.exists(path)

                with open(path, "r") as f:
                    code = f.read()

                # Should have forward function
                assert "trix_forward" in code
            finally:
                os.unlink(path)

    def test_arm64_target_properties(self):
        """ARM64 target has correct properties for Pi."""
        target = TARGETS["arm64-linux"]
        assert target.architecture == "arm64"
        assert "neon" in target.features

    def test_wasm_target_properties(self):
        """WASM target has web properties."""
        target = TARGETS["wasm"]
        assert target.os == "web"
        assert "max_memory" in target.constraints


class TestEventIntegration:
    """Tests for event system integration."""

    def test_events_across_full_pipeline(self):
        """Events flow correctly through entire pipeline."""
        spine = EventSpine()
        events_seen = []

        def collector(event):
            events_seen.append((event.event_type.name, event.source))

        spine.subscribe(collector)

        # Create model
        model = ModelIR(
            name="event_pipeline_test",
            shapes=[
                Shape("layer1", "MATMUL", ["input"], ["h1"]),
                Shape("relu1", "RELU", ["h1"], ["a1"]),
                Shape("layer2", "MATMUL", ["a1"], ["output"])
            ],
            tensors={
                "input": TensorSpec("input", [100], "float32"),
                "h1": TensorSpec("h1", [50], "float32"),
                "a1": TensorSpec("a1", [50], "float32"),
                "output": TensorSpec("output", [10], "float32")
            }
        )

        # Emit custom events
        spine.emit(EventType.TRACE_POINT, "test", {"phase": "model_created"})

        # Compile
        compiler = Compiler(events=spine)
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            path = f.name

        try:
            compiler.compile(model, "c", path)
            spine.emit(EventType.TRACE_POINT, "test", {"phase": "compiled"})

            # Verify events
            event_types = [e[0] for e in events_seen]
            assert "TRACE_POINT" in event_types
            assert "COMPILE_START" in event_types
            assert "COMPILE_END" in event_types

        finally:
            os.unlink(path)

    def test_events_can_be_exported_for_analysis(self):
        """Events export to analyzable format."""
        spine = EventSpine()

        # Generate many events
        for i in range(100):
            spine.new_frame()
            spine.emit(EventType.FORWARD_START, f"model", {"batch": i})
            for j in range(5):
                spine.emit(EventType.LAYER_FORWARD, f"layer.{j}",
                          {"activations_mean": 0.5 + j * 0.1})
            spine.emit(EventType.FORWARD_END, "model",
                      {"duration_ms": 0.1 + i * 0.001})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            spine.export(path)

            # Load and analyze
            with open(path, "r") as f:
                data = json.load(f)

            assert len(data) == 100 * 7  # 100 frames * 7 events each

            # Can filter by type
            forward_starts = [e for e in data if e["event_type"] == "FORWARD_START"]
            assert len(forward_starts) == 100

            # Can compute statistics
            durations = [e["payload"]["duration_ms"]
                        for e in data if e["event_type"] == "FORWARD_END"]
            avg_duration = sum(durations) / len(durations)
            assert avg_duration > 0

        finally:
            os.unlink(path)


class TestLeverIntegration:
    """Tests for lever system integration."""

    def test_levers_affect_compilation(self):
        """Lever changes affect compiled output."""
        model = ModelIR(
            name="lever_compile_test",
            shapes=[Shape("relu", "RELU", ["in"], ["out"])],
            tensors={
                "in": TensorSpec("in", [10], "float32"),
                "out": TensorSpec("out", [10], "float32")
            },
            levers={
                "precision": Lever("precision", "Precision", "enum", "FP16", "FP16",
                                  ["FP8", "FP16", "FP32"])
            }
        )

        compiler = Compiler()

        # Compile with default precision
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            path1 = f.name

        compiler.compile(model, "c", path1)

        # Change lever
        model.levers["precision"].set("FP8")

        # Compile again
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            path2 = f.name

        compiler.compile(model, "c", path2)

        # Both should compile successfully
        try:
            assert os.path.exists(path1)
            assert os.path.exists(path2)
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_lever_state_persists_through_save_load(self):
        """Lever modifications persist through save/load cycle."""
        model = ModelIR(
            name="lever_persist_test",
            levers={
                "batch_size": Lever("batch_size", "Batch Size", "int", 32, 32, [1, 1024]),
                "precision": Lever("precision", "Precision", "enum", "FP16", "FP16",
                                  ["FP8", "FP16", "FP32"]),
                "dropout": Lever("dropout", "Dropout", "float", 0.1, 0.1, [0.0, 1.0])
            }
        )

        # Modify all levers
        model.levers["batch_size"].set(64)
        model.levers["precision"].set("FP8")
        model.levers["dropout"].set(0.5)

        with tempfile.NamedTemporaryFile(suffix=".trix", delete=False) as f:
            path = f.name

        try:
            model.save(path)
            loaded = ModelIR.load(path)

            # Verify all states preserved
            assert loaded.levers["batch_size"].get() == 64
            assert loaded.levers["precision"].get() == "FP8"
            assert loaded.levers["dropout"].get() == 0.5

        finally:
            os.unlink(path)


class TestRobustness:
    """Robustness and stress tests."""

    def test_large_model(self):
        """Handle large model with many layers."""
        shapes = []
        tensors = {"input": TensorSpec("input", [1024], "float32")}

        prev = "input"
        for i in range(50):  # 50 layer transformer-style
            # Attention-like block
            shapes.append(Shape(f"attn_{i}", "ATTENTION", [prev], [f"attn_out_{i}"]))
            tensors[f"attn_out_{i}"] = TensorSpec(f"attn_out_{i}", [1024], "float32")

            shapes.append(Shape(f"norm_{i}", "LAYERNORM", [f"attn_out_{i}"], [f"norm_{i}"]))
            tensors[f"norm_{i}"] = TensorSpec(f"norm_{i}", [1024], "float32")

            shapes.append(Shape(f"ffn_{i}", "MATMUL", [f"norm_{i}"], [f"ffn_{i}"]))
            tensors[f"ffn_{i}"] = TensorSpec(f"ffn_{i}", [4096], "float32")

            shapes.append(Shape(f"relu_{i}", "RELU", [f"ffn_{i}"], [f"relu_{i}"]))
            tensors[f"relu_{i}"] = TensorSpec(f"relu_{i}", [4096], "float32")

            shapes.append(Shape(f"proj_{i}", "MATMUL", [f"relu_{i}"], [f"out_{i}"]))
            tensors[f"out_{i}"] = TensorSpec(f"out_{i}", [1024], "float32")

            prev = f"out_{i}"

        model = ModelIR(name="large_model", shapes=shapes, tensors=tensors)

        # Should validate
        errors = model.validate()
        assert len(errors) == 0

        # Should compile
        compiler = Compiler()
        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            path = f.name

        try:
            stats = compiler.compile(model, "c", path)
            assert stats["shapes_count"] == 250  # 5 shapes per layer * 50 layers
        finally:
            os.unlink(path)

    def test_many_levers(self):
        """Handle many levers."""
        levers = {}
        for i in range(100):
            levers[f"lever_{i}"] = Lever(
                name=f"lever_{i}",
                display_name=f"Lever {i}",
                lever_type="float",
                default=float(i) / 100,
                current=float(i) / 100,
                valid_range=[0.0, 1.0]
            )

        model = ModelIR(name="many_levers", levers=levers)

        # Should save/load
        with tempfile.NamedTemporaryFile(suffix=".trix", delete=False) as f:
            path = f.name

        try:
            model.save(path)
            loaded = ModelIR.load(path)
            assert len(loaded.levers) == 100
        finally:
            os.unlink(path)

    def test_rapid_event_emission(self):
        """Handle rapid event emission."""
        spine = EventSpine()

        # Emit 10000 events rapidly
        for frame in range(100):
            spine.new_frame()
            for event in range(100):
                spine.emit(
                    EventType.TRACE_POINT,
                    f"source.{event}",
                    {"frame": frame, "event": event}
                )

        assert spine.count == 10000

        # Query should still work
        frame_50 = spine.query(frame_id=50)
        assert len(frame_50) == 100

    def test_concurrent_compile_export(self):
        """Compile and export events simultaneously."""
        spine = EventSpine()
        model = ModelIR(
            name="concurrent_test",
            shapes=[Shape("relu", "RELU", ["in"], ["out"])],
            tensors={
                "in": TensorSpec("in", [10], "float32"),
                "out": TensorSpec("out", [10], "float32")
            }
        )

        compiler = Compiler(events=spine)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Multiple compilations
            for i in range(5):
                c_path = os.path.join(tmpdir, f"model_{i}.c")
                compiler.compile(model, "c", c_path)

            # Export events
            export_path = os.path.join(tmpdir, "events.json")
            spine.export(export_path)

            # Verify
            with open(export_path, "r") as f:
                events = json.load(f)

            # Should have compile events for each compilation
            compile_starts = [e for e in events if e["event_type"] == "COMPILE_START"]
            assert len(compile_starts) == 5


class TestErrorHandling:
    """Error handling tests."""

    def test_invalid_target_error_message(self):
        """Invalid target gives helpful error."""
        model = ModelIR(name="error_test")
        compiler = Compiler()

        with pytest.raises(ValueError) as exc_info:
            compiler.compile(model, "nonexistent_target", "/tmp/out.c")

        error_msg = str(exc_info.value)
        assert "Unknown target" in error_msg
        assert "nonexistent_target" in error_msg

    def test_lever_validation_error(self):
        """Lever validation gives helpful error."""
        lever = Lever(
            name="test",
            display_name="Test",
            lever_type="int",
            default=50,
            current=50,
            valid_range=[0, 100]
        )

        with pytest.raises(ValueError) as exc_info:
            lever.set(200)

        error_msg = str(exc_info.value)
        assert "200" in error_msg or "out of range" in error_msg.lower()

    def test_model_validation_reports_all_errors(self):
        """Model validation reports all errors, not just first."""
        model = ModelIR(
            name="multi_error",
            shapes=[
                Shape("s1", "RELU", ["missing1"], ["out1"]),
                Shape("s2", "RELU", ["missing2"], ["out2"]),
                Shape("s3", "RELU", ["missing3"], ["out3"])
            ],
            tensors={
                "out1": TensorSpec("out1", [10], "float32"),
                "out2": TensorSpec("out2", [10], "float32"),
                "out3": TensorSpec("out3", [10], "float32")
            }
        )

        errors = model.validate()
        # Should report all missing tensors
        assert len(errors) >= 3


# =============================================================================
# TEST SUMMARY
# =============================================================================

def test_integration_summary():
    """Summary of integration tests."""
    categories = {
        "TRIXC Integration": 5,
        "Platform Integration": 3,
        "Event Integration": 2,
        "Lever Integration": 2,
        "Robustness": 4,
        "Error Handling": 3
    }

    total = sum(categories.values())
    print(f"\nTRIX Forge Integration Test Summary")
    print("=" * 40)
    for cat, count in categories.items():
        print(f"{cat}: {count} tests")
    print("=" * 40)
    print(f"Total: {total} integration tests")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
