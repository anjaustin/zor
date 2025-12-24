"""
Model IR - Intermediate Representation for TRIX models.

The ModelIR is the central data structure that holds:
- Shapes (frozen computational patterns)
- Tensors (named tensor specifications)
- Weights (actual weight data)
- Levers (tunable parameters)
- Metadata (model information)
"""

import json
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


@dataclass
class Shape:
    """A frozen computational shape."""
    name: str
    type: str  # linear, relu, softmax, xor, and, or, not, etc.
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "input": self.input_shape,
            "output": self.output_shape,
            "params": self.params
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Shape":
        return cls(
            name=d["name"],
            type=d["type"],
            input_shape=d.get("input"),
            output_shape=d.get("output"),
            params=d.get("params", {})
        )


@dataclass
class TensorSpec:
    """Specification for a named tensor."""
    name: str
    shape: Tuple[int, ...]
    dtype: str = "float32"
    requires_grad: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "requires_grad": self.requires_grad
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TensorSpec":
        return cls(
            name=d["name"],
            shape=tuple(d["shape"]),
            dtype=d.get("dtype", "float32"),
            requires_grad=d.get("requires_grad", False)
        )


@dataclass
class ModelIR:
    """
    Intermediate Representation for a TRIX model.

    This is the canonical in-memory representation of a model,
    used for validation, optimization, and compilation.
    """
    version: str = "1.0"
    name: str = "unnamed"
    shapes: List[Shape] = field(default_factory=list)
    tensors: Dict[str, TensorSpec] = field(default_factory=dict)
    weights: Dict[str, Any] = field(default_factory=dict)
    levers: Dict[str, "Lever"] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_shape(self, shape: Shape) -> None:
        """Add a shape to the model."""
        self.shapes.append(shape)

    def add_tensor(self, spec: TensorSpec) -> None:
        """Add a tensor specification."""
        self.tensors[spec.name] = spec

    def set_weight(self, name: str, data: Any) -> None:
        """Set weight data for a tensor."""
        self.weights[name] = data

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the model.

        Returns (is_valid, list_of_errors).
        """
        errors = []

        # Check version
        if not self.version:
            errors.append("Missing version")

        # Check name
        if not self.name:
            errors.append("Missing name")

        # Check shapes
        if not self.shapes:
            errors.append("No shapes defined")

        # Check tensor references in shapes
        for shape in self.shapes:
            for tensor_ref in shape.params.get("tensors", []):
                if tensor_ref not in self.tensors:
                    errors.append(f"Shape '{shape.name}' references undefined tensor '{tensor_ref}'")

        # Check weight data for tensors that need it
        for name, spec in self.tensors.items():
            if spec.requires_grad or "weight" in name.lower():
                if name not in self.weights:
                    errors.append(f"Tensor '{name}' missing weight data")

        return len(errors) == 0, errors

    def hash(self) -> str:
        """Compute a deterministic hash of the model."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        from .levers import Lever
        return {
            "version": self.version,
            "name": self.name,
            "shapes": [s.to_dict() for s in self.shapes],
            "tensors": {k: v.to_dict() for k, v in self.tensors.items()},
            "levers": {k: v.to_dict() for k, v in self.levers.items()},
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelIR":
        """Create from dictionary."""
        from .levers import Lever
        model = cls(
            version=d.get("version", "1.0"),
            name=d.get("name", "unnamed"),
            metadata=d.get("metadata", {})
        )
        for s in d.get("shapes", []):
            model.add_shape(Shape.from_dict(s))
        for name, t in d.get("tensors", {}).items():
            model.tensors[name] = TensorSpec.from_dict(t)
        for name, l in d.get("levers", {}).items():
            model.levers[name] = Lever.from_dict(l)
        return model

    def save(self, path: str) -> None:
        """Save model to file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModelIR":
        """Load model from file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def __repr__(self) -> str:
        return f"ModelIR(name='{self.name}', shapes={len(self.shapes)}, tensors={len(self.tensors)})"
