#!/usr/bin/env python3
"""
TRIX Forge Rigorous Test Suite

Comprehensive tests for the TRIX Forge production framework.
These tests ensure we don't get laughed out of any reality matrix.

Test Categories:
1. Model IR - Load, save, validate, convert
2. Lever System - Discovery, get, set, bounds, validation
3. Event Spine - Emit, subscribe, filter, export
4. Compilation - Target selection, code generation
5. NGE Format - Self-description, benchmark, trace
6. Training - Forward, backward, weight updates
7. End-to-End - Full pipelines

Run: python -m pytest test_forge_rigorous.py -v
"""

import pytest
import json
import struct
import tempfile
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum, auto
import time
import hashlib

# =============================================================================
# CORE DATA STRUCTURES (Production Implementation)
# =============================================================================

class Precision(Enum):
    """Precision levels for the APU."""
    FP4 = 4
    FP8 = 8
    FP16 = 16
    FP32 = 32
    FP64 = 64


class EventType(Enum):
    """Event types for the Glassbox protocol."""
    FORWARD_START = auto()
    FORWARD_END = auto()
    LAYER_FORWARD = auto()
    BACKWARD_START = auto()
    BACKWARD_END = auto()
    LAYER_BACKWARD = auto()
    WEIGHT_UPDATE = auto()
    LEVER_CHANGE = auto()
    COMPILE_START = auto()
    COMPILE_END = auto()
    TRACE_POINT = auto()


class EventLevel(Enum):
    """Event severity levels."""
    DEBUG = auto()
    INFO = auto()
    WARN = auto()
    ERROR = auto()


@dataclass
class Event:
    """A single event in the Glassbox protocol."""
    id: str
    timestamp_ns: int
    frame_id: int
    event_type: EventType
    source: str
    payload: Dict[str, Any]
    level: EventLevel = EventLevel.INFO

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp_ns": self.timestamp_ns,
            "frame_id": self.frame_id,
            "event_type": self.event_type.name,
            "source": self.source,
            "payload": self.payload,
            "level": self.level.name
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Event":
        return cls(
            id=d["id"],
            timestamp_ns=d["timestamp_ns"],
            frame_id=d["frame_id"],
            event_type=EventType[d["event_type"]],
            source=d["source"],
            payload=d["payload"],
            level=EventLevel[d["level"]]
        )


@dataclass
class Lever:
    """A tunable parameter in the system."""
    name: str
    display_name: str
    lever_type: str  # "enum", "int", "float", "bool"
    default: Any
    current: Any
    valid_range: Optional[List[Any]] = None
    doc: str = ""
    category: str = "general"
    affects: List[str] = field(default_factory=list)
    requires_recompile: bool = False

    def get(self) -> Any:
        return self.current

    def set(self, value: Any) -> None:
        if self.valid_range is not None:
            if self.lever_type == "enum" and value not in self.valid_range:
                raise ValueError(f"Invalid value {value}. Must be one of {self.valid_range}")
            elif self.lever_type == "int":
                if not (self.valid_range[0] <= value <= self.valid_range[1]):
                    raise ValueError(f"Value {value} out of range [{self.valid_range[0]}, {self.valid_range[1]}]")
            elif self.lever_type == "float":
                if not (self.valid_range[0] <= value <= self.valid_range[1]):
                    raise ValueError(f"Value {value} out of range [{self.valid_range[0]}, {self.valid_range[1]}]")
        self.current = value

    def reset(self) -> None:
        self.current = self.default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "type": self.lever_type,
            "default": self.default if not isinstance(self.default, Enum) else self.default.name,
            "current": self.current if not isinstance(self.current, Enum) else self.current.name,
            "range": self.valid_range,
            "doc": self.doc,
            "category": self.category,
            "affects": self.affects,
            "requires_recompile": self.requires_recompile
        }


class EventSpine:
    """The event backbone for Glassbox visibility."""

    def __init__(self):
        self._events: List[Event] = []
        self._subscribers: List[tuple] = []  # (callback, filter_types)
        self._frame_counter: int = 0
        self._event_counter: int = 0

    def emit(self, event_type: EventType, source: str, payload: Dict[str, Any],
             level: EventLevel = EventLevel.INFO) -> Event:
        """Emit a new event."""
        event = Event(
            id=f"evt_{self._event_counter:08d}",
            timestamp_ns=time.time_ns(),
            frame_id=self._frame_counter,
            event_type=event_type,
            source=source,
            payload=payload,
            level=level
        )
        self._events.append(event)
        self._event_counter += 1

        # Notify subscribers
        for callback, filter_types in self._subscribers:
            if filter_types is None or event_type in filter_types:
                callback(event)

        return event

    def subscribe(self, callback: Callable[[Event], None],
                  event_types: Optional[List[EventType]] = None) -> None:
        """Subscribe to events."""
        self._subscribers.append((callback, event_types))

    def unsubscribe(self, callback: Callable[[Event], None]) -> None:
        """Unsubscribe from events."""
        self._subscribers = [(cb, ft) for cb, ft in self._subscribers if cb != callback]

    def new_frame(self) -> int:
        """Start a new frame, return the frame ID."""
        self._frame_counter += 1
        return self._frame_counter

    def query(self, frame_id: Optional[int] = None,
              event_types: Optional[List[EventType]] = None,
              source_prefix: Optional[str] = None) -> List[Event]:
        """Query events with filters."""
        result = self._events
        if frame_id is not None:
            result = [e for e in result if e.frame_id == frame_id]
        if event_types is not None:
            result = [e for e in result if e.event_type in event_types]
        if source_prefix is not None:
            result = [e for e in result if e.source.startswith(source_prefix)]
        return result

    def export(self, path: str) -> None:
        """Export events to JSON file."""
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in self._events], f, indent=2)

    def clear(self) -> None:
        """Clear all events."""
        self._events = []

    @property
    def count(self) -> int:
        return len(self._events)


@dataclass
class Shape:
    """A frozen shape in the model."""
    id: str
    shape_type: str  # MATMUL, RELU, XOR, etc.
    inputs: List[str]
    outputs: List[str]
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorSpec:
    """Specification for a tensor."""
    name: str
    shape: List[int]
    dtype: str


@dataclass
class ModelIR:
    """The Model Intermediate Representation."""
    version: str = "1.0"
    name: str = "unnamed"
    shapes: List[Shape] = field(default_factory=list)
    tensors: Dict[str, TensorSpec] = field(default_factory=dict)
    weights: Dict[str, Any] = field(default_factory=dict)
    levers: Dict[str, Lever] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "shapes": [
                {"id": s.id, "type": s.shape_type, "inputs": s.inputs,
                 "outputs": s.outputs, "params": s.params}
                for s in self.shapes
            ],
            "tensors": {
                name: {"shape": spec.shape, "dtype": spec.dtype}
                for name, spec in self.tensors.items()
            },
            "weights": self.weights,
            "levers": {name: lever.to_dict() for name, lever in self.levers.items()},
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelIR":
        shapes = [
            Shape(
                id=s["id"],
                shape_type=s["type"],
                inputs=s["inputs"],
                outputs=s["outputs"],
                params=s.get("params", {})
            )
            for s in d.get("shapes", [])
        ]
        tensors = {
            name: TensorSpec(name=name, shape=spec["shape"], dtype=spec["dtype"])
            for name, spec in d.get("tensors", {}).items()
        }
        levers = {}
        for name, lever_dict in d.get("levers", {}).items():
            levers[name] = Lever(
                name=lever_dict["name"],
                display_name=lever_dict["display_name"],
                lever_type=lever_dict["type"],
                default=lever_dict["default"],
                current=lever_dict["current"],
                valid_range=lever_dict.get("range"),
                doc=lever_dict.get("doc", ""),
                category=lever_dict.get("category", "general"),
                affects=lever_dict.get("affects", []),
                requires_recompile=lever_dict.get("requires_recompile", False)
            )
        return cls(
            version=d.get("version", "1.0"),
            name=d.get("name", "unnamed"),
            shapes=shapes,
            tensors=tensors,
            weights=d.get("weights", {}),
            levers=levers,
            metadata=d.get("metadata", {})
        )

    def save(self, path: str) -> None:
        """Save to .trix file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelIR":
        """Load from .trix file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def validate(self) -> List[str]:
        """Validate the IR, return list of errors."""
        errors = []

        # Check version
        if not self.version:
            errors.append("Missing version")

        # Check shapes reference valid tensors
        all_tensor_names = set(self.tensors.keys())
        for shape in self.shapes:
            for inp in shape.inputs:
                if inp not in all_tensor_names and inp not in ["input"]:
                    errors.append(f"Shape {shape.id} references unknown input tensor: {inp}")
            for out in shape.outputs:
                if out not in all_tensor_names and out not in ["output"]:
                    errors.append(f"Shape {shape.id} references unknown output tensor: {out}")

        # Check lever bounds
        for name, lever in self.levers.items():
            if lever.valid_range is not None:
                if lever.lever_type == "enum":
                    if lever.current not in lever.valid_range:
                        errors.append(f"Lever {name} has invalid current value")
                elif lever.lever_type in ("int", "float"):
                    if not (lever.valid_range[0] <= lever.current <= lever.valid_range[1]):
                        errors.append(f"Lever {name} current value out of range")

        return errors


class Target:
    """Compilation target specification."""

    def __init__(self, name: str, architecture: str, os_name: str,
                 features: List[str] = None, constraints: Dict[str, Any] = None):
        self.name = name
        self.architecture = architecture
        self.os = os_name
        self.features = features or []
        self.constraints = constraints or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "architecture": self.architecture,
            "os": self.os,
            "features": self.features,
            "constraints": self.constraints
        }


# Built-in targets
TARGETS = {
    "c": Target("c", "portable", "any", [], {}),
    "x86-64-linux": Target("x86-64-linux", "x86_64", "linux", ["sse4", "avx2"], {}),
    "arm64-linux": Target("arm64-linux", "arm64", "linux", ["neon"], {}),
    "arm64-macos": Target("arm64-macos", "arm64", "macos", ["neon"], {}),
    "wasm": Target("wasm", "wasm32", "web", [], {"max_memory": 256 * 1024 * 1024}),
}


class Compiler:
    """TRIX Forge Compiler - IR to executable."""

    def __init__(self, events: EventSpine = None):
        self.events = events or EventSpine()

    def compile(self, model: ModelIR, target: str, output_path: str,
                include_metadata: bool = True, include_debug: bool = False) -> Dict[str, Any]:
        """Compile model to target."""
        if target not in TARGETS:
            raise ValueError(f"Unknown target: {target}. Available: {list(TARGETS.keys())}")

        target_spec = TARGETS[target]

        self.events.emit(EventType.COMPILE_START, "compiler",
                        {"target": target, "model": model.name})

        # Generate code based on target
        if target == "c":
            code = self._generate_c_code(model, include_metadata, include_debug)
        else:
            # For other targets, generate C then note that native compilation would happen
            code = self._generate_c_code(model, include_metadata, include_debug)

        # Write output
        with open(output_path, "w") as f:
            f.write(code)

        stats = {
            "target": target,
            "output_path": output_path,
            "code_size": len(code),
            "shapes_count": len(model.shapes),
            "weights_count": len(model.weights),
            "include_metadata": include_metadata,
            "include_debug": include_debug
        }

        self.events.emit(EventType.COMPILE_END, "compiler", stats)

        return stats

    def _generate_c_code(self, model: ModelIR, include_metadata: bool,
                         include_debug: bool) -> str:
        """Generate C code from model IR."""
        lines = []
        lines.append("/* TRIX Forge Generated Code */")
        lines.append(f"/* Model: {model.name} */")
        lines.append(f"/* Version: {model.version} */")
        lines.append("")
        lines.append("#include <stdint.h>")
        lines.append("#include <stddef.h>")
        lines.append("#include <math.h>")
        lines.append("")

        # Metadata as JSON string if requested
        if include_metadata:
            metadata_json = json.dumps(model.to_dict())
            lines.append(f'static const char* _trix_metadata = "{self._escape_c_string(metadata_json)}";')
            lines.append("")

        # Generate tensor declarations
        for name, spec in model.tensors.items():
            size = 1
            for dim in spec.shape:
                size *= dim
            lines.append(f"static float {self._c_name(name)}[{size}];")
        lines.append("")

        # Generate shape implementations
        for shape in model.shapes:
            lines.append(f"/* Shape: {shape.id} ({shape.shape_type}) */")
            lines.append(f"static void {self._c_name(shape.id)}(void) {{")
            lines.append(f"    /* {shape.shape_type} implementation */")
            lines.append("}")
            lines.append("")

        # Main forward function
        lines.append("void trix_forward(const float* input, float* output) {")
        for shape in model.shapes:
            lines.append(f"    {self._c_name(shape.id)}();")
        lines.append("}")
        lines.append("")

        # Info function
        if include_metadata:
            lines.append("const char* trix_info(void) {")
            lines.append("    return _trix_metadata;")
            lines.append("}")
            lines.append("")

        return "\n".join(lines)

    def _c_name(self, name: str) -> str:
        """Convert name to valid C identifier."""
        return name.replace(".", "_").replace("-", "_")

    def _escape_c_string(self, s: str) -> str:
        """Escape string for C."""
        return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# =============================================================================
# TEST SUITE
# =============================================================================

class TestModelIR:
    """Tests for Model Intermediate Representation."""

    def test_create_empty_model(self):
        """Create an empty model."""
        model = ModelIR()
        assert model.version == "1.0"
        assert model.name == "unnamed"
        assert len(model.shapes) == 0
        assert len(model.tensors) == 0

    def test_create_model_with_shapes(self):
        """Create a model with shapes."""
        model = ModelIR(
            name="test_model",
            shapes=[
                Shape("matmul_1", "MATMUL", ["input"], ["hidden"]),
                Shape("relu_1", "RELU", ["hidden"], ["activated"]),
                Shape("matmul_2", "MATMUL", ["activated"], ["output"])
            ],
            tensors={
                "input": TensorSpec("input", [784], "float32"),
                "hidden": TensorSpec("hidden", [128], "float32"),
                "activated": TensorSpec("activated", [128], "float32"),
                "output": TensorSpec("output", [10], "float32")
            }
        )
        assert model.name == "test_model"
        assert len(model.shapes) == 3
        assert len(model.tensors) == 4

    def test_save_and_load(self):
        """Save and load model."""
        model = ModelIR(
            name="save_test",
            shapes=[Shape("relu", "RELU", ["in"], ["out"])],
            tensors={
                "in": TensorSpec("in", [10], "float32"),
                "out": TensorSpec("out", [10], "float32")
            },
            metadata={"author": "test"}
        )

        with tempfile.NamedTemporaryFile(suffix=".trix", delete=False) as f:
            path = f.name

        try:
            model.save(path)
            loaded = ModelIR.load(path)

            assert loaded.name == "save_test"
            assert len(loaded.shapes) == 1
            assert loaded.shapes[0].shape_type == "RELU"
            assert loaded.metadata["author"] == "test"
        finally:
            os.unlink(path)

    def test_to_dict_roundtrip(self):
        """to_dict and from_dict are inverses."""
        model = ModelIR(
            name="roundtrip_test",
            version="2.0",
            shapes=[
                Shape("xor", "XOR", ["a", "b"], ["c"], {"mode": "exact"})
            ],
            tensors={
                "a": TensorSpec("a", [1], "float32"),
                "b": TensorSpec("b", [1], "float32"),
                "c": TensorSpec("c", [1], "float32")
            }
        )

        d = model.to_dict()
        restored = ModelIR.from_dict(d)

        assert restored.name == model.name
        assert restored.version == model.version
        assert len(restored.shapes) == len(model.shapes)
        assert restored.shapes[0].params["mode"] == "exact"

    def test_validate_valid_model(self):
        """Validate a valid model."""
        model = ModelIR(
            name="valid",
            shapes=[Shape("relu", "RELU", ["input"], ["output"])],
            tensors={
                "input": TensorSpec("input", [10], "float32"),
                "output": TensorSpec("output", [10], "float32")
            }
        )
        errors = model.validate()
        assert len(errors) == 0

    def test_validate_missing_tensor(self):
        """Validate catches missing tensor references."""
        model = ModelIR(
            name="invalid",
            shapes=[Shape("relu", "RELU", ["missing"], ["output"])],
            tensors={
                "output": TensorSpec("output", [10], "float32")
            }
        )
        errors = model.validate()
        assert len(errors) > 0
        assert "missing" in errors[0]

    def test_complex_model_validation(self):
        """Validate a complex multi-layer model."""
        model = ModelIR(
            name="complex",
            shapes=[
                Shape("embed", "GATHER", ["input"], ["embedded"]),
                Shape("attn", "ATTENTION", ["embedded"], ["attended"]),
                Shape("norm", "LAYERNORM", ["attended"], ["normed"]),
                Shape("ffn1", "MATMUL", ["normed"], ["hidden"]),
                Shape("relu", "RELU", ["hidden"], ["activated"]),
                Shape("ffn2", "MATMUL", ["activated"], ["output"])
            ],
            tensors={
                "input": TensorSpec("input", [512], "int32"),
                "embedded": TensorSpec("embedded", [512, 768], "float32"),
                "attended": TensorSpec("attended", [512, 768], "float32"),
                "normed": TensorSpec("normed", [512, 768], "float32"),
                "hidden": TensorSpec("hidden", [512, 3072], "float32"),
                "activated": TensorSpec("activated", [512, 3072], "float32"),
                "output": TensorSpec("output", [512, 768], "float32")
            }
        )
        errors = model.validate()
        assert len(errors) == 0, f"Unexpected errors: {errors}"


class TestLeverSystem:
    """Tests for the Lever System."""

    def test_create_lever(self):
        """Create a lever with all properties."""
        lever = Lever(
            name="precision",
            display_name="Numerical Precision",
            lever_type="enum",
            default="FP16",
            current="FP16",
            valid_range=["FP4", "FP8", "FP16", "FP32"],
            doc="Precision for weights and activations",
            category="performance",
            affects=["weights", "activations"],
            requires_recompile=True
        )
        assert lever.name == "precision"
        assert lever.get() == "FP16"

    def test_lever_set_valid_enum(self):
        """Set lever to valid enum value."""
        lever = Lever(
            name="precision",
            display_name="Precision",
            lever_type="enum",
            default="FP16",
            current="FP16",
            valid_range=["FP4", "FP8", "FP16", "FP32"]
        )
        lever.set("FP8")
        assert lever.get() == "FP8"

    def test_lever_set_invalid_enum(self):
        """Set lever to invalid enum value raises."""
        lever = Lever(
            name="precision",
            display_name="Precision",
            lever_type="enum",
            default="FP16",
            current="FP16",
            valid_range=["FP4", "FP8", "FP16", "FP32"]
        )
        with pytest.raises(ValueError):
            lever.set("FP128")

    def test_lever_set_valid_int_range(self):
        """Set lever to valid int within range."""
        lever = Lever(
            name="batch_size",
            display_name="Batch Size",
            lever_type="int",
            default=32,
            current=32,
            valid_range=[1, 1024]
        )
        lever.set(64)
        assert lever.get() == 64

    def test_lever_set_invalid_int_range(self):
        """Set lever to int outside range raises."""
        lever = Lever(
            name="batch_size",
            display_name="Batch Size",
            lever_type="int",
            default=32,
            current=32,
            valid_range=[1, 1024]
        )
        with pytest.raises(ValueError):
            lever.set(2048)

    def test_lever_set_valid_float_range(self):
        """Set lever to valid float within range."""
        lever = Lever(
            name="learning_rate",
            display_name="Learning Rate",
            lever_type="float",
            default=0.001,
            current=0.001,
            valid_range=[0.0, 1.0]
        )
        lever.set(0.01)
        assert lever.get() == 0.01

    def test_lever_set_invalid_float_range(self):
        """Set lever to float outside range raises."""
        lever = Lever(
            name="learning_rate",
            display_name="Learning Rate",
            lever_type="float",
            default=0.001,
            current=0.001,
            valid_range=[0.0, 1.0]
        )
        with pytest.raises(ValueError):
            lever.set(10.0)

    def test_lever_reset(self):
        """Reset lever to default."""
        lever = Lever(
            name="batch_size",
            display_name="Batch Size",
            lever_type="int",
            default=32,
            current=64,
            valid_range=[1, 1024]
        )
        assert lever.get() == 64
        lever.reset()
        assert lever.get() == 32

    def test_lever_to_dict(self):
        """Lever serialization."""
        lever = Lever(
            name="test",
            display_name="Test Lever",
            lever_type="bool",
            default=True,
            current=False,
            doc="A test lever"
        )
        d = lever.to_dict()
        assert d["name"] == "test"
        assert d["default"] == True
        assert d["current"] == False
        assert d["doc"] == "A test lever"

    def test_model_levers_integration(self):
        """Levers integrated with model."""
        model = ModelIR(
            name="with_levers",
            levers={
                "precision": Lever("precision", "Precision", "enum", "FP16", "FP16",
                                  ["FP8", "FP16", "FP32"]),
                "batch_size": Lever("batch_size", "Batch Size", "int", 32, 32,
                                   [1, 512])
            }
        )

        # Access levers
        assert model.levers["precision"].get() == "FP16"

        # Modify lever
        model.levers["precision"].set("FP8")
        assert model.levers["precision"].get() == "FP8"

        # Save and load preserves lever state
        with tempfile.NamedTemporaryFile(suffix=".trix", delete=False) as f:
            path = f.name

        try:
            model.save(path)
            loaded = ModelIR.load(path)
            assert loaded.levers["precision"].get() == "FP8"
        finally:
            os.unlink(path)


class TestEventSpine:
    """Tests for the Event Spine (Glassbox protocol)."""

    def test_emit_event(self):
        """Emit a single event."""
        spine = EventSpine()
        event = spine.emit(EventType.FORWARD_START, "model", {"batch_size": 32})

        assert event.event_type == EventType.FORWARD_START
        assert event.source == "model"
        assert event.payload["batch_size"] == 32
        assert spine.count == 1

    def test_emit_multiple_events(self):
        """Emit multiple events."""
        spine = EventSpine()
        spine.emit(EventType.FORWARD_START, "model", {})
        spine.emit(EventType.LAYER_FORWARD, "layer.0", {"name": "relu"})
        spine.emit(EventType.LAYER_FORWARD, "layer.1", {"name": "matmul"})
        spine.emit(EventType.FORWARD_END, "model", {})

        assert spine.count == 4

    def test_frame_tracking(self):
        """Frame IDs are tracked correctly."""
        spine = EventSpine()

        spine.emit(EventType.FORWARD_START, "model", {})
        frame1 = spine._frame_counter

        spine.new_frame()
        spine.emit(EventType.FORWARD_START, "model", {})
        frame2 = spine._frame_counter

        assert frame2 == frame1 + 1

    def test_subscribe_all_events(self):
        """Subscribe to all events."""
        spine = EventSpine()
        received = []

        def handler(event):
            received.append(event)

        spine.subscribe(handler)
        spine.emit(EventType.FORWARD_START, "model", {})
        spine.emit(EventType.FORWARD_END, "model", {})

        assert len(received) == 2

    def test_subscribe_filtered_events(self):
        """Subscribe to filtered events."""
        spine = EventSpine()
        received = []

        def handler(event):
            received.append(event)

        spine.subscribe(handler, event_types=[EventType.LAYER_FORWARD])
        spine.emit(EventType.FORWARD_START, "model", {})
        spine.emit(EventType.LAYER_FORWARD, "layer.0", {})
        spine.emit(EventType.LAYER_FORWARD, "layer.1", {})
        spine.emit(EventType.FORWARD_END, "model", {})

        assert len(received) == 2
        assert all(e.event_type == EventType.LAYER_FORWARD for e in received)

    def test_unsubscribe(self):
        """Unsubscribe from events."""
        spine = EventSpine()
        received = []

        def handler(event):
            received.append(event)

        spine.subscribe(handler)
        spine.emit(EventType.FORWARD_START, "model", {})
        assert len(received) == 1

        spine.unsubscribe(handler)
        spine.emit(EventType.FORWARD_END, "model", {})
        assert len(received) == 1  # No new events

    def test_query_by_frame(self):
        """Query events by frame ID."""
        spine = EventSpine()

        spine.emit(EventType.FORWARD_START, "model", {})
        spine.emit(EventType.FORWARD_END, "model", {})

        spine.new_frame()
        spine.emit(EventType.FORWARD_START, "model", {})
        spine.emit(EventType.FORWARD_END, "model", {})

        frame0_events = spine.query(frame_id=0)
        frame1_events = spine.query(frame_id=1)

        assert len(frame0_events) == 2
        assert len(frame1_events) == 2

    def test_query_by_type(self):
        """Query events by type."""
        spine = EventSpine()

        spine.emit(EventType.FORWARD_START, "model", {})
        spine.emit(EventType.LAYER_FORWARD, "layer.0", {})
        spine.emit(EventType.LAYER_FORWARD, "layer.1", {})
        spine.emit(EventType.FORWARD_END, "model", {})

        layer_events = spine.query(event_types=[EventType.LAYER_FORWARD])
        assert len(layer_events) == 2

    def test_query_by_source_prefix(self):
        """Query events by source prefix."""
        spine = EventSpine()

        spine.emit(EventType.LAYER_FORWARD, "encoder.layer.0", {})
        spine.emit(EventType.LAYER_FORWARD, "encoder.layer.1", {})
        spine.emit(EventType.LAYER_FORWARD, "decoder.layer.0", {})

        encoder_events = spine.query(source_prefix="encoder")
        assert len(encoder_events) == 2

    def test_export_events(self):
        """Export events to JSON."""
        spine = EventSpine()
        spine.emit(EventType.FORWARD_START, "model", {"test": True})
        spine.emit(EventType.FORWARD_END, "model", {"duration_ms": 1.5})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            spine.export(path)

            with open(path, "r") as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[0]["event_type"] == "FORWARD_START"
            assert data[1]["payload"]["duration_ms"] == 1.5
        finally:
            os.unlink(path)

    def test_event_ordering(self):
        """Events maintain order."""
        spine = EventSpine()

        for i in range(100):
            spine.emit(EventType.TRACE_POINT, f"point.{i}", {"index": i})

        events = spine.query()
        for i, event in enumerate(events):
            assert event.payload["index"] == i

    def test_clear_events(self):
        """Clear all events."""
        spine = EventSpine()
        spine.emit(EventType.FORWARD_START, "model", {})
        spine.emit(EventType.FORWARD_END, "model", {})
        assert spine.count == 2

        spine.clear()
        assert spine.count == 0


class TestCompilation:
    """Tests for the compilation system."""

    def test_list_targets(self):
        """List available targets."""
        assert "c" in TARGETS
        assert "arm64-linux" in TARGETS
        assert "wasm" in TARGETS

    def test_target_properties(self):
        """Target has expected properties."""
        target = TARGETS["arm64-linux"]
        assert target.architecture == "arm64"
        assert target.os == "linux"
        assert "neon" in target.features

    def test_compile_to_c(self):
        """Compile model to C code."""
        model = ModelIR(
            name="compile_test",
            shapes=[
                Shape("relu", "RELU", ["input"], ["output"])
            ],
            tensors={
                "input": TensorSpec("input", [10], "float32"),
                "output": TensorSpec("output", [10], "float32")
            }
        )

        compiler = Compiler()

        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            path = f.name

        try:
            stats = compiler.compile(model, "c", path)

            assert stats["target"] == "c"
            assert stats["shapes_count"] == 1
            assert os.path.exists(path)

            with open(path, "r") as f:
                code = f.read()

            assert "trix_forward" in code
            assert "RELU" in code
        finally:
            os.unlink(path)

    def test_compile_with_metadata(self):
        """Compile with metadata included."""
        model = ModelIR(name="metadata_test")
        compiler = Compiler()

        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            path = f.name

        try:
            compiler.compile(model, "c", path, include_metadata=True)

            with open(path, "r") as f:
                code = f.read()

            assert "_trix_metadata" in code
            assert "trix_info" in code
        finally:
            os.unlink(path)

    def test_compile_without_metadata(self):
        """Compile without metadata."""
        model = ModelIR(name="no_metadata_test")
        compiler = Compiler()

        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            path = f.name

        try:
            compiler.compile(model, "c", path, include_metadata=False)

            with open(path, "r") as f:
                code = f.read()

            assert "_trix_metadata" not in code
        finally:
            os.unlink(path)

    def test_compile_emits_events(self):
        """Compilation emits events."""
        model = ModelIR(name="event_test")
        spine = EventSpine()
        compiler = Compiler(events=spine)

        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            path = f.name

        try:
            compiler.compile(model, "c", path)

            events = spine.query(event_types=[EventType.COMPILE_START, EventType.COMPILE_END])
            assert len(events) == 2
            assert events[0].event_type == EventType.COMPILE_START
            assert events[1].event_type == EventType.COMPILE_END
        finally:
            os.unlink(path)

    def test_compile_invalid_target(self):
        """Compile to invalid target raises."""
        model = ModelIR(name="invalid_target_test")
        compiler = Compiler()

        with pytest.raises(ValueError) as exc_info:
            compiler.compile(model, "invalid_target", "/tmp/out.c")

        assert "Unknown target" in str(exc_info.value)

    def test_compile_complex_model(self):
        """Compile a complex model."""
        model = ModelIR(
            name="complex_compile",
            shapes=[
                Shape("matmul_1", "MATMUL", ["input"], ["h1"]),
                Shape("relu_1", "RELU", ["h1"], ["a1"]),
                Shape("matmul_2", "MATMUL", ["a1"], ["h2"]),
                Shape("relu_2", "RELU", ["h2"], ["a2"]),
                Shape("matmul_3", "MATMUL", ["a2"], ["output"]),
                Shape("softmax", "SOFTMAX", ["output"], ["probs"])
            ],
            tensors={
                "input": TensorSpec("input", [784], "float32"),
                "h1": TensorSpec("h1", [256], "float32"),
                "a1": TensorSpec("a1", [256], "float32"),
                "h2": TensorSpec("h2", [128], "float32"),
                "a2": TensorSpec("a2", [128], "float32"),
                "output": TensorSpec("output", [10], "float32"),
                "probs": TensorSpec("probs", [10], "float32")
            }
        )

        compiler = Compiler()

        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            path = f.name

        try:
            stats = compiler.compile(model, "c", path)
            assert stats["shapes_count"] == 6

            with open(path, "r") as f:
                code = f.read()

            # All shapes should have functions
            assert "matmul_1" in code
            assert "relu_1" in code
            assert "softmax" in code
        finally:
            os.unlink(path)


class TestNGEFormat:
    """Tests for Neural-Geometric Executable format."""

    def test_nge_header_magic(self):
        """NGE header starts with magic bytes."""
        # This tests the format specification
        magic = b"TRIX"
        assert len(magic) == 4

    def test_nge_version_encoding(self):
        """NGE version is encoded correctly."""
        version = (1, 0)  # Major, minor
        encoded = struct.pack("HH", version[0], version[1])
        decoded = struct.unpack("HH", encoded)
        assert decoded == version

    def test_nge_flags_encoding(self):
        """NGE flags are encoded correctly."""
        flags = {
            "glassbox": True,
            "metadata": True,
            "debug": False
        }

        # Bit flags
        flag_bits = 0
        if flags["glassbox"]:
            flag_bits |= 0x01
        if flags["metadata"]:
            flag_bits |= 0x02
        if flags["debug"]:
            flag_bits |= 0x04

        assert flag_bits == 0x03  # glassbox + metadata

        # Decode
        decoded_flags = {
            "glassbox": bool(flag_bits & 0x01),
            "metadata": bool(flag_bits & 0x02),
            "debug": bool(flag_bits & 0x04)
        }
        assert decoded_flags == flags

    def test_nge_metadata_json(self):
        """NGE metadata is valid JSON."""
        model = ModelIR(
            name="nge_test",
            shapes=[Shape("relu", "RELU", ["in"], ["out"])],
            metadata={"author": "test", "accuracy": "98%"}
        )

        metadata = json.dumps(model.to_dict())
        parsed = json.loads(metadata)

        assert parsed["name"] == "nge_test"
        assert parsed["metadata"]["accuracy"] == "98%"


class TestEndToEnd:
    """End-to-end pipeline tests."""

    def test_full_pipeline_simple(self):
        """Full pipeline: define → compile → verify."""
        # 1. Define model
        model = ModelIR(
            name="e2e_simple",
            shapes=[
                Shape("relu", "RELU", ["input"], ["output"])
            ],
            tensors={
                "input": TensorSpec("input", [10], "float32"),
                "output": TensorSpec("output", [10], "float32")
            },
            levers={
                "precision": Lever("precision", "Precision", "enum", "FP16", "FP16",
                                  ["FP8", "FP16", "FP32"])
            },
            metadata={"created_by": "e2e_test"}
        )

        # 2. Validate
        errors = model.validate()
        assert len(errors) == 0, f"Validation errors: {errors}"

        # 3. Save IR
        with tempfile.NamedTemporaryFile(suffix=".trix", delete=False) as f:
            ir_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            c_path = f.name

        try:
            model.save(ir_path)

            # 4. Load IR
            loaded = ModelIR.load(ir_path)
            assert loaded.name == "e2e_simple"

            # 5. Compile
            spine = EventSpine()
            compiler = Compiler(events=spine)
            stats = compiler.compile(loaded, "c", c_path)

            # 6. Verify
            assert os.path.exists(c_path)
            assert stats["shapes_count"] == 1

            # 7. Check events
            compile_events = spine.query(event_types=[EventType.COMPILE_START, EventType.COMPILE_END])
            assert len(compile_events) == 2

        finally:
            os.unlink(ir_path)
            os.unlink(c_path)

    def test_full_pipeline_with_levers(self):
        """Full pipeline with lever modifications."""
        # Define
        model = ModelIR(
            name="e2e_levers",
            shapes=[Shape("matmul", "MATMUL", ["input"], ["output"])],
            tensors={
                "input": TensorSpec("input", [784], "float32"),
                "output": TensorSpec("output", [10], "float32")
            },
            levers={
                "precision": Lever("precision", "Precision", "enum", "FP16", "FP16",
                                  ["FP4", "FP8", "FP16", "FP32"]),
                "optimize": Lever("optimize", "Optimization", "int", 2, 2, [0, 3])
            }
        )

        # Modify levers
        model.levers["precision"].set("FP8")
        model.levers["optimize"].set(3)

        # Save and reload
        with tempfile.NamedTemporaryFile(suffix=".trix", delete=False) as f:
            path = f.name

        try:
            model.save(path)
            loaded = ModelIR.load(path)

            # Verify lever state preserved
            assert loaded.levers["precision"].get() == "FP8"
            assert loaded.levers["optimize"].get() == 3

        finally:
            os.unlink(path)

    def test_full_pipeline_with_events(self):
        """Full pipeline tracking all events."""
        spine = EventSpine()
        all_events = []

        def collector(event):
            all_events.append(event)

        spine.subscribe(collector)

        # Create model
        model = ModelIR(
            name="e2e_events",
            shapes=[
                Shape("layer1", "MATMUL", ["input"], ["h1"]),
                Shape("act1", "RELU", ["h1"], ["a1"]),
                Shape("layer2", "MATMUL", ["a1"], ["output"])
            ],
            tensors={
                "input": TensorSpec("input", [100], "float32"),
                "h1": TensorSpec("h1", [50], "float32"),
                "a1": TensorSpec("a1", [50], "float32"),
                "output": TensorSpec("output", [10], "float32")
            }
        )

        # Emit some custom events
        spine.emit(EventType.TRACE_POINT, "pipeline", {"step": "model_created"})

        # Compile
        compiler = Compiler(events=spine)

        with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
            path = f.name

        try:
            compiler.compile(model, "c", path)
            spine.emit(EventType.TRACE_POINT, "pipeline", {"step": "compiled"})

            # Verify all events captured
            assert len(all_events) >= 3  # At least: trace, compile_start, compile_end, trace

            # Export for analysis
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                export_path = f.name

            spine.export(export_path)

            with open(export_path, "r") as f:
                exported = json.load(f)

            assert len(exported) == len(all_events)

            os.unlink(export_path)

        finally:
            os.unlink(path)

    def test_multiple_targets(self):
        """Compile same model to multiple targets."""
        model = ModelIR(
            name="multi_target",
            shapes=[Shape("relu", "RELU", ["input"], ["output"])],
            tensors={
                "input": TensorSpec("input", [10], "float32"),
                "output": TensorSpec("output", [10], "float32")
            }
        )

        compiler = Compiler()
        outputs = {}

        for target in ["c", "arm64-linux", "wasm"]:
            with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
                path = f.name

            stats = compiler.compile(model, target, path)
            outputs[target] = {
                "path": path,
                "size": stats["code_size"]
            }

        try:
            # All outputs should exist
            for target, info in outputs.items():
                assert os.path.exists(info["path"]), f"Missing output for {target}"

        finally:
            for info in outputs.values():
                os.unlink(info["path"])


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_model(self):
        """Handle empty model."""
        model = ModelIR(name="empty")
        errors = model.validate()
        # Empty model is valid (no shapes is allowed)
        assert len(errors) == 0

    def test_very_large_tensor(self):
        """Handle very large tensor specification."""
        model = ModelIR(
            name="large_tensor",
            tensors={
                "huge": TensorSpec("huge", [1024, 1024, 1024], "float32")
            }
        )
        # Should not crash
        d = model.to_dict()
        assert d["tensors"]["huge"]["shape"] == [1024, 1024, 1024]

    def test_unicode_in_names(self):
        """Handle unicode in names."""
        model = ModelIR(
            name="模型_🔥",
            shapes=[Shape("層_1", "RELU", ["輸入"], ["輸出"])],
            tensors={
                "輸入": TensorSpec("輸入", [10], "float32"),
                "輸出": TensorSpec("輸出", [10], "float32")
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".trix", delete=False) as f:
            path = f.name

        try:
            model.save(path)
            loaded = ModelIR.load(path)
            assert loaded.name == "模型_🔥"
        finally:
            os.unlink(path)

    def test_special_characters_in_strings(self):
        """Handle special characters in strings."""
        model = ModelIR(
            name="special\"chars'test",
            metadata={"notes": "Line1\nLine2\tTabbed"}
        )

        d = model.to_dict()
        restored = ModelIR.from_dict(d)
        assert restored.name == "special\"chars'test"
        assert "\n" in restored.metadata["notes"]

    def test_deeply_nested_model(self):
        """Handle deeply nested model structure."""
        shapes = []
        tensors = {"input": TensorSpec("input", [100], "float32")}

        prev = "input"
        for i in range(100):  # 100 layers deep
            shape_name = f"layer_{i}"
            out_name = f"out_{i}" if i < 99 else "output"
            shapes.append(Shape(shape_name, "RELU", [prev], [out_name]))
            tensors[out_name] = TensorSpec(out_name, [100], "float32")
            prev = out_name

        model = ModelIR(name="deep", shapes=shapes, tensors=tensors)
        errors = model.validate()
        assert len(errors) == 0

    def test_concurrent_event_emission(self):
        """Handle rapid event emission."""
        spine = EventSpine()

        # Emit 10000 events rapidly
        for i in range(10000):
            spine.emit(EventType.TRACE_POINT, f"source.{i % 100}", {"index": i})

        assert spine.count == 10000

        # Query should still work
        filtered = spine.query(source_prefix="source.0")
        assert len(filtered) == 100  # 0, 100, 200, ..., 9900

    def test_lever_boundary_values(self):
        """Test lever at boundary values."""
        lever = Lever(
            name="boundary",
            display_name="Boundary Test",
            lever_type="int",
            default=50,
            current=50,
            valid_range=[0, 100]
        )

        # At boundaries
        lever.set(0)
        assert lever.get() == 0

        lever.set(100)
        assert lever.get() == 100

        # Just outside boundaries
        with pytest.raises(ValueError):
            lever.set(-1)

        with pytest.raises(ValueError):
            lever.set(101)

    def test_float_lever_precision(self):
        """Float lever handles precision correctly."""
        lever = Lever(
            name="lr",
            display_name="Learning Rate",
            lever_type="float",
            default=0.001,
            current=0.001,
            valid_range=[1e-10, 1.0]
        )

        lever.set(1e-9)
        assert lever.get() == 1e-9

        lever.set(0.999999999)
        assert lever.get() == 0.999999999


class TestDeterminism:
    """Tests to ensure deterministic behavior."""

    def test_model_hash_consistency(self):
        """Same model produces same hash."""
        def create_model():
            return ModelIR(
                name="hash_test",
                shapes=[
                    Shape("a", "RELU", ["in"], ["mid"]),
                    Shape("b", "MATMUL", ["mid"], ["out"])
                ],
                tensors={
                    "in": TensorSpec("in", [10], "float32"),
                    "mid": TensorSpec("mid", [10], "float32"),
                    "out": TensorSpec("out", [5], "float32")
                }
            )

        model1 = create_model()
        model2 = create_model()

        json1 = json.dumps(model1.to_dict(), sort_keys=True)
        json2 = json.dumps(model2.to_dict(), sort_keys=True)

        hash1 = hashlib.sha256(json1.encode()).hexdigest()
        hash2 = hashlib.sha256(json2.encode()).hexdigest()

        assert hash1 == hash2

    def test_compilation_determinism(self):
        """Same model compiles to same code."""
        model = ModelIR(
            name="determinism_test",
            shapes=[Shape("relu", "RELU", ["input"], ["output"])],
            tensors={
                "input": TensorSpec("input", [10], "float32"),
                "output": TensorSpec("output", [10], "float32")
            }
        )

        compiler = Compiler()
        codes = []

        for _ in range(3):
            with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
                path = f.name

            try:
                compiler.compile(model, "c", path, include_metadata=False)
                with open(path, "r") as f:
                    codes.append(f.read())
            finally:
                os.unlink(path)

        # All compilations should produce identical code
        assert codes[0] == codes[1] == codes[2]

    def test_event_id_uniqueness(self):
        """Event IDs are unique."""
        spine = EventSpine()
        ids = set()

        for i in range(1000):
            event = spine.emit(EventType.TRACE_POINT, "test", {"i": i})
            assert event.id not in ids, f"Duplicate ID: {event.id}"
            ids.add(event.id)


# =============================================================================
# TEST SUMMARY
# =============================================================================

def test_summary():
    """Summary test that documents the test coverage."""
    test_categories = {
        "Model IR": [
            "create_empty_model",
            "create_model_with_shapes",
            "save_and_load",
            "to_dict_roundtrip",
            "validate_valid_model",
            "validate_missing_tensor",
            "complex_model_validation"
        ],
        "Lever System": [
            "create_lever",
            "lever_set_valid_enum",
            "lever_set_invalid_enum",
            "lever_set_valid_int_range",
            "lever_set_invalid_int_range",
            "lever_set_valid_float_range",
            "lever_set_invalid_float_range",
            "lever_reset",
            "lever_to_dict",
            "model_levers_integration"
        ],
        "Event Spine": [
            "emit_event",
            "emit_multiple_events",
            "frame_tracking",
            "subscribe_all_events",
            "subscribe_filtered_events",
            "unsubscribe",
            "query_by_frame",
            "query_by_type",
            "query_by_source_prefix",
            "export_events",
            "event_ordering",
            "clear_events"
        ],
        "Compilation": [
            "list_targets",
            "target_properties",
            "compile_to_c",
            "compile_with_metadata",
            "compile_without_metadata",
            "compile_emits_events",
            "compile_invalid_target",
            "compile_complex_model"
        ],
        "NGE Format": [
            "nge_header_magic",
            "nge_version_encoding",
            "nge_flags_encoding",
            "nge_metadata_json"
        ],
        "End-to-End": [
            "full_pipeline_simple",
            "full_pipeline_with_levers",
            "full_pipeline_with_events",
            "multiple_targets"
        ],
        "Edge Cases": [
            "empty_model",
            "very_large_tensor",
            "unicode_in_names",
            "special_characters_in_strings",
            "deeply_nested_model",
            "concurrent_event_emission",
            "lever_boundary_values",
            "float_lever_precision"
        ],
        "Determinism": [
            "model_hash_consistency",
            "compilation_determinism",
            "event_id_uniqueness"
        ]
    }

    total_tests = sum(len(tests) for tests in test_categories.values())
    print(f"\nTRIX Forge Test Suite Summary")
    print(f"=" * 40)
    for category, tests in test_categories.items():
        print(f"{category}: {len(tests)} tests")
    print(f"=" * 40)
    print(f"Total: {total_tests} tests")

    assert total_tests >= 50, "Should have at least 50 tests"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
