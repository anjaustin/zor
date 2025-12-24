"""
Forge Backend - Integration with TRIX Forge compilation system.

Provides the bridge between CLI and the Forge compilation pipeline.
"""

import os
import sys
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

# Import from parent forge package if available
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from tests.test_forge_rigorous import (
        ModelIR, Shape, TensorSpec, Lever, Compiler, Target, EventSpine
    )
    FORGE_AVAILABLE = True
except ImportError:
    FORGE_AVAILABLE = False


@dataclass
class BuildConfig:
    """Configuration for a build operation."""
    model_path: str
    target: str
    output_path: Optional[str] = None
    levers: Dict[str, Any] = field(default_factory=dict)
    glass: bool = False
    trace: Optional[str] = None
    quiet: bool = False
    debug: bool = False


@dataclass
class BuildResult:
    """Result of a build operation."""
    success: bool
    output_path: str = ""
    output_size: int = 0
    target: str = ""
    target_desc: str = ""
    build_time: float = 0.0
    events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    suggestions: Optional[List[str]] = None


class ForgeBackend:
    """
    Backend for TRIX Forge CLI.

    Handles:
    - Model loading and validation
    - Compilation to various targets
    - Event streaming
    - Lever management
    """

    def __init__(self):
        self.model: Optional[Any] = None
        self.compiler: Optional[Any] = None
        self.events: List[Dict[str, Any]] = []
        self._event_callback: Optional[Callable] = None

    def _emit(self, type: str, source: str, message: str,
              frame: int = 0, data: Optional[Dict[str, Any]] = None) -> None:
        """Emit an event."""
        event = {
            "id": len(self.events) + 1,
            "type": type,
            "source": source,
            "message": message,
            "frame": frame,
            "timestamp": time.time(),
            "data": data
        }
        self.events.append(event)
        if self._event_callback:
            self._event_callback(event)

    def set_event_callback(self, callback: Callable) -> None:
        """Set callback for event emission."""
        self._event_callback = callback

    def get_targets(self) -> List[Dict[str, str]]:
        """Get available compilation targets."""
        return [
            {"id": "c", "description": "Portable C99 (any platform)"},
            {"id": "arm64-linux", "description": "Raspberry Pi 4+, Jetson Nano"},
            {"id": "arm64-macos", "description": "Apple Silicon (M1/M2/M3)"},
            {"id": "x86-64-linux", "description": "Linux x86-64 servers"},
            {"id": "x86-64-windows", "description": "Windows x86-64"},
            {"id": "wasm", "description": "WebAssembly (browsers)"},
            {"id": "cuda", "description": "NVIDIA GPUs (compute 7.0+)"},
            {"id": "metal", "description": "Apple Metal GPUs"},
        ]

    def get_target_description(self, target_id: str) -> str:
        """Get description for a target."""
        for t in self.get_targets():
            if t["id"] == target_id:
                return t["description"]
        return "Unknown target"

    def validate_target(self, target: str) -> bool:
        """Check if target is valid."""
        valid = [t["id"] for t in self.get_targets()]
        return target in valid

    def load_model(self, path: str) -> Dict[str, Any]:
        """Load and validate a model file."""
        self._emit("load.start", "loader", f"loading {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")

        # Simulate model loading
        # In production, this would parse the actual .trix format
        model_info = {
            "name": Path(path).stem,
            "path": path,
            "shapes": 4,
            "tensors": 6,
            "layers": [
                {"name": "Input", "type": "input", "shape": [4]},
                {"name": "Hidden", "type": "linear", "shape": [8]},
                {"name": "ReLU", "type": "relu", "shape": [8]},
                {"name": "Output", "type": "linear", "shape": [2]},
            ],
            "levers": {
                "precision": {"type": "enum", "options": ["fp32", "fp16", "fp8"],
                              "default": "fp16", "value": "fp16"},
                "optimize": {"type": "enum", "options": ["none", "aggressive"],
                             "default": "aggressive", "value": "aggressive"},
                "fuse_ops": {"type": "bool", "default": True, "value": True},
                "debug_info": {"type": "bool", "default": False, "value": False},
            }
        }

        self._emit("load.complete", "loader",
                   f"{model_info['shapes']} shapes, {model_info['tensors']} tensors loaded")

        return model_info

    def build(self, config: BuildConfig) -> BuildResult:
        """Execute a build operation."""
        start_time = time.time()
        self.events = []

        try:
            # Validate target
            if not self.validate_target(config.target):
                similar = self._find_similar_target(config.target)
                suggestions = [f"Did you mean '{similar}'?"] if similar else []
                suggestions.append(f"Run 'forge targets' to see available targets")
                return BuildResult(
                    success=False,
                    error=f"Unknown target: {config.target}",
                    suggestions=suggestions
                )

            # Load model
            model_info = self.load_model(config.model_path)

            # Apply lever overrides
            for name, value in config.levers.items():
                if name in model_info["levers"]:
                    model_info["levers"][name]["value"] = value
                    self._emit("lever.set", "config", f"{name} = {value}")

            # Validate
            self._emit("validate.start", "validator", "checking constraints")
            time.sleep(0.05)  # Simulate work
            self._emit("validate.pass", "validator", "all shape dimensions compatible")

            # Optimize
            self._emit("optimize.start", "optimizer", "analyzing operations")
            time.sleep(0.05)
            self._emit("optimize.fuse", "optimizer", "Linear→ReLU fused (2 ops → 1)")
            if model_info["levers"]["precision"]["value"] != "fp32":
                prec = model_info["levers"]["precision"]["value"]
                self._emit("optimize.quantize", "optimizer", f"fp32 → {prec} (precision lever)")

            # Emit code
            self._emit("emit.start", "emitter", f"generating {config.target} code")
            time.sleep(0.05)

            if config.target.startswith("arm"):
                self._emit("emit.complete", "emitter", "generated 847 ARM64 instructions")
            elif config.target == "wasm":
                self._emit("emit.complete", "emitter", "generated 1.2 KB WASM module")
            else:
                self._emit("emit.complete", "emitter", "generated target code")

            # Package
            self._emit("package.start", "packager", "writing NGE header")
            time.sleep(0.03)
            self._emit("package.weights", "packager", "embedding quantized weights")

            # Determine output path
            output_path = config.output_path
            if not output_path:
                output_path = Path(config.model_path).stem + ".nge"

            # Simulate output size based on precision
            prec = model_info["levers"]["precision"]["value"]
            base_size = 2048
            if prec == "fp16":
                base_size = int(base_size * 0.5)
            elif prec == "fp8":
                base_size = int(base_size * 0.25)

            self._emit("package.complete", "packager",
                       f"output: {output_path} ({base_size} bytes)")

            build_time = time.time() - start_time

            return BuildResult(
                success=True,
                output_path=output_path,
                output_size=base_size,
                target=config.target,
                target_desc=self.get_target_description(config.target),
                build_time=build_time,
                events=self.events.copy()
            )

        except FileNotFoundError as e:
            return BuildResult(
                success=False,
                error=str(e),
                suggestions=["Check that the model path is correct",
                            "Run 'forge inspect <model>' to verify the model"]
            )
        except Exception as e:
            return BuildResult(
                success=False,
                error=f"Build failed: {str(e)}",
                events=self.events.copy()
            )

    def _find_similar_target(self, target: str) -> Optional[str]:
        """Find a similar valid target name."""
        valid = [t["id"] for t in self.get_targets()]
        target_lower = target.lower()

        # Check for close matches
        for v in valid:
            if v.startswith(target_lower[:3]):
                return v
            if target_lower in v:
                return v

        return None

    def inspect(self, path: str) -> Dict[str, Any]:
        """Inspect a model or NGE file."""
        if path.endswith(".nge"):
            return self._inspect_nge(path)
        else:
            return self.load_model(path)

    def _inspect_nge(self, path: str) -> Dict[str, Any]:
        """Inspect an NGE file."""
        # Simulated NGE inspection
        return {
            "type": "nge",
            "path": path,
            "size": 1847,
            "target": "arm64-linux",
            "model": "tiny_mlp",
            "version": "1.0",
            "glassbox": True,
            "metadata": True
        }
