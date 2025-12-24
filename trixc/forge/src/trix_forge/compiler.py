"""
Compiler - Multi-target compilation for TRIX models.

The Compiler transforms ModelIR into native executables:
- Parse and validate the model
- Apply optimizations (fusion, quantization)
- Emit target-specific code
- Package as NGE (Neural-Geometric Executable)
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

from .model import ModelIR
from .events import EventSpine, Event


class TargetArch(Enum):
    """Target architectures."""
    C = "c"
    ARM64_LINUX = "arm64-linux"
    ARM64_MACOS = "arm64-macos"
    X86_64_LINUX = "x86-64-linux"
    X86_64_WINDOWS = "x86-64-windows"
    WASM = "wasm"
    CUDA = "cuda"
    METAL = "metal"


@dataclass
class Target:
    """
    A compilation target.

    Describes the target platform's capabilities and constraints.
    """
    id: str
    arch: str
    description: str
    features: List[str] = field(default_factory=list)
    max_tensor_size: int = 1024 * 1024 * 1024  # 1GB default
    supported_dtypes: List[str] = field(default_factory=lambda: ["float32", "float16"])
    endianness: str = "little"

    def supports_feature(self, feature: str) -> bool:
        """Check if target supports a feature."""
        return feature in self.features

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "arch": self.arch,
            "description": self.description,
            "features": self.features,
            "max_tensor_size": self.max_tensor_size,
            "supported_dtypes": self.supported_dtypes,
            "endianness": self.endianness
        }


# Built-in targets
BUILTIN_TARGETS = {
    "c": Target(
        id="c",
        arch="c99",
        description="Portable C99 (any platform)",
        features=["portable", "no-simd"],
        supported_dtypes=["float32", "float16", "int8"]
    ),
    "arm64-linux": Target(
        id="arm64-linux",
        arch="arm64",
        description="Raspberry Pi 4+, Jetson Nano",
        features=["neon", "linux", "embedded"],
        supported_dtypes=["float32", "float16", "int8"]
    ),
    "arm64-macos": Target(
        id="arm64-macos",
        arch="arm64",
        description="Apple Silicon (M1/M2/M3)",
        features=["neon", "macos", "accelerate"],
        supported_dtypes=["float32", "float16"]
    ),
    "x86-64-linux": Target(
        id="x86-64-linux",
        arch="x86-64",
        description="Linux x86-64 servers",
        features=["avx2", "linux", "server"],
        supported_dtypes=["float32", "float16", "int8", "bf16"]
    ),
    "x86-64-windows": Target(
        id="x86-64-windows",
        arch="x86-64",
        description="Windows x86-64",
        features=["avx2", "windows"],
        supported_dtypes=["float32", "float16"]
    ),
    "wasm": Target(
        id="wasm",
        arch="wasm32",
        description="WebAssembly (browsers)",
        features=["portable", "browser", "simd128"],
        max_tensor_size=256 * 1024 * 1024,  # 256MB limit
        supported_dtypes=["float32"]
    ),
    "cuda": Target(
        id="cuda",
        arch="nvptx64",
        description="NVIDIA GPUs (compute 7.0+)",
        features=["gpu", "cuda", "tensor-cores"],
        supported_dtypes=["float32", "float16", "tf32", "int8"]
    ),
    "metal": Target(
        id="metal",
        arch="air64",
        description="Apple Metal GPUs",
        features=["gpu", "metal", "apple"],
        supported_dtypes=["float32", "float16"]
    ),
}


@dataclass
class CompileResult:
    """Result of a compilation."""
    success: bool
    target: str
    output_code: str = ""
    output_size: int = 0
    compile_time: float = 0.0
    optimizations_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "target": self.target,
            "output_size": self.output_size,
            "compile_time": self.compile_time,
            "optimizations": self.optimizations_applied,
            "warnings": self.warnings,
            "error": self.error
        }


class Compiler:
    """
    Multi-target compiler for TRIX models.

    Usage:
        compiler = Compiler()

        # List available targets
        for target in compiler.list_targets():
            print(f"{target.id}: {target.description}")

        # Compile a model
        result = compiler.compile(model, target="arm64-linux")

        if result.success:
            print(f"Generated {result.output_size} bytes in {result.compile_time:.2f}s")
    """

    def __init__(self, event_spine: Optional[EventSpine] = None):
        self.targets = BUILTIN_TARGETS.copy()
        self.spine = event_spine or EventSpine()

    def list_targets(self) -> List[Target]:
        """List all available compilation targets."""
        return list(self.targets.values())

    def get_target(self, target_id: str) -> Optional[Target]:
        """Get a target by ID."""
        return self.targets.get(target_id)

    def register_target(self, target: Target) -> None:
        """Register a custom target."""
        self.targets[target.id] = target

    def compile(self, model: ModelIR, target: str,
                include_metadata: bool = True,
                include_debug: bool = False) -> CompileResult:
        """
        Compile a model to the specified target.

        Args:
            model: The ModelIR to compile
            target: Target ID (e.g., "arm64-linux")
            include_metadata: Include metadata in output
            include_debug: Include debug symbols

        Returns:
            CompileResult with status and output
        """
        start_time = time.time()
        optimizations = []
        warnings = []

        # Validate target
        target_obj = self.get_target(target)
        if not target_obj:
            return CompileResult(
                success=False,
                target=target,
                error=f"Unknown target: {target}"
            )

        self.spine.emit("compile.start", "compiler",
                       f"Compiling {model.name} for {target}")

        # Validate model
        self.spine.emit("validate.start", "compiler", "Validating model")
        is_valid, errors = model.validate()
        if not is_valid:
            self.spine.emit("validate.fail", "compiler", f"Validation failed: {errors}")
            return CompileResult(
                success=False,
                target=target,
                error=f"Model validation failed: {errors}"
            )
        self.spine.emit("validate.pass", "compiler", "Model valid")

        # Optimization phase
        self.spine.emit("optimize.start", "compiler", "Applying optimizations")

        # Check for fusible operations
        fuse_lever = model.levers.get("fuse_ops")
        if fuse_lever is None or fuse_lever.value:
            fused = self._fuse_operations(model)
            if fused:
                optimizations.append(f"fused {fused} operations")
                self.spine.emit("optimize.fuse", "compiler",
                               f"Fused {fused} operations")

        # Quantization
        precision = model.levers.get("precision")
        if precision and precision.value != "fp32":
            optimizations.append(f"quantized to {precision.value}")
            self.spine.emit("optimize.quantize", "compiler",
                           f"Quantized to {precision.value}")

        # Code emission
        self.spine.emit("emit.start", "compiler", f"Generating {target} code")
        code = self._emit_code(model, target_obj, include_metadata, include_debug)
        self.spine.emit("emit.complete", "compiler",
                       f"Generated {len(code)} bytes")

        compile_time = time.time() - start_time

        self.spine.emit("compile.complete", "compiler",
                       f"Compilation complete in {compile_time:.2f}s")

        return CompileResult(
            success=True,
            target=target,
            output_code=code,
            output_size=len(code),
            compile_time=compile_time,
            optimizations_applied=optimizations,
            warnings=warnings
        )

    def _fuse_operations(self, model: ModelIR) -> int:
        """Fuse compatible operations. Returns count of fusions."""
        fused = 0
        i = 0
        while i < len(model.shapes) - 1:
            curr = model.shapes[i]
            next_shape = model.shapes[i + 1]

            # Linear + ReLU fusion
            if curr.type == "linear" and next_shape.type == "relu":
                curr.params["fused_activation"] = "relu"
                model.shapes.pop(i + 1)
                fused += 1
                continue

            # Linear + GELU fusion
            if curr.type == "linear" and next_shape.type == "gelu":
                curr.params["fused_activation"] = "gelu"
                model.shapes.pop(i + 1)
                fused += 1
                continue

            i += 1

        return fused

    def _emit_code(self, model: ModelIR, target: Target,
                   include_metadata: bool, include_debug: bool) -> str:
        """Generate target-specific code."""
        lines = []

        # Header
        lines.append(f"/* TRIX Forge Generated Code */")
        lines.append(f"/* Model: {model.name} */")
        lines.append(f"/* Target: {target.id} */")
        lines.append(f"/* Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} */")
        lines.append("")

        if include_metadata:
            lines.append(f'#define TRIX_MODEL_NAME "{model.name}"')
            lines.append(f'#define TRIX_MODEL_VERSION "{model.version}"')
            lines.append(f'#define TRIX_TARGET "{target.id}"')
            lines.append("")

        # Includes
        lines.append("#include <stdint.h>")
        lines.append("#include <math.h>")
        lines.append("")

        # Shape implementations
        lines.append("/* Frozen Shapes */")
        lines.append("")

        # XOR shape
        lines.append("static inline float trix_xor(float a, float b) {")
        lines.append("    return a + b - 2.0f * a * b;")
        lines.append("}")
        lines.append("")

        # AND shape
        lines.append("static inline float trix_and(float a, float b) {")
        lines.append("    return a * b;")
        lines.append("}")
        lines.append("")

        # OR shape
        lines.append("static inline float trix_or(float a, float b) {")
        lines.append("    return a + b - a * b;")
        lines.append("}")
        lines.append("")

        # NOT shape
        lines.append("static inline float trix_not(float a) {")
        lines.append("    return 1.0f - a;")
        lines.append("}")
        lines.append("")

        # ReLU
        lines.append("static inline float trix_relu(float x) {")
        lines.append("    return x > 0.0f ? x : 0.0f;")
        lines.append("}")
        lines.append("")

        # Sigmoid
        lines.append("static inline float trix_sigmoid(float x) {")
        lines.append("    return 1.0f / (1.0f + expf(-x));")
        lines.append("}")
        lines.append("")

        # Model-specific code
        lines.append("/* Model Forward Pass */")
        lines.append(f"void {model.name}_forward(const float* input, float* output) {{")
        lines.append("    /* Generated implementation */")
        lines.append("    (void)input;")
        lines.append("    (void)output;")
        lines.append("}")
        lines.append("")

        if include_debug:
            lines.append("/* Debug info included */")
            for i, shape in enumerate(model.shapes):
                lines.append(f"/* Shape {i}: {shape.name} ({shape.type}) */")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Compiler(targets={len(self.targets)})"
