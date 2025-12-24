"""
Lever System - Externalized tunable parameters.

Every tunable parameter in TRIX Forge is a Lever:
- Discoverable (list all levers)
- Typed (enum, int, float, bool)
- Bounded (min/max for numeric, options for enum)
- Documented (description, units)
- Observable (changes emit events)
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, Union
from enum import Enum


class LeverType(Enum):
    """Types of levers."""
    ENUM = "enum"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"


@dataclass
class Lever:
    """
    A tunable parameter with type, bounds, and documentation.

    Examples:
        precision = Lever(
            name="precision",
            type=LeverType.ENUM,
            options=["fp32", "fp16", "fp8"],
            default="fp16",
            description="Numeric precision for weights and activations"
        )

        learning_rate = Lever(
            name="learning_rate",
            type=LeverType.FLOAT,
            min_value=1e-6,
            max_value=1.0,
            default=0.001,
            description="Learning rate for optimization"
        )
    """
    name: str
    type: LeverType
    default: Any
    value: Any = None
    description: str = ""

    # For ENUM type
    options: Optional[List[str]] = None

    # For INT/FLOAT types
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None

    # For documentation
    units: Optional[str] = None
    category: str = "general"

    def __post_init__(self):
        if self.value is None:
            self.value = self.default

    def set(self, value: Any) -> bool:
        """
        Set the lever value with validation.

        Returns True if successful, raises ValueError if invalid.
        """
        if self.type == LeverType.ENUM:
            if self.options and value not in self.options:
                raise ValueError(
                    f"Invalid value '{value}' for lever '{self.name}'. "
                    f"Valid options: {self.options}"
                )
        elif self.type == LeverType.INT:
            if not isinstance(value, int):
                raise ValueError(f"Lever '{self.name}' requires int, got {type(value)}")
            if self.min_value is not None and value < self.min_value:
                raise ValueError(f"Value {value} below minimum {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                raise ValueError(f"Value {value} above maximum {self.max_value}")
        elif self.type == LeverType.FLOAT:
            if not isinstance(value, (int, float)):
                raise ValueError(f"Lever '{self.name}' requires float, got {type(value)}")
            if self.min_value is not None and value < self.min_value:
                raise ValueError(f"Value {value} below minimum {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                raise ValueError(f"Value {value} above maximum {self.max_value}")
        elif self.type == LeverType.BOOL:
            if not isinstance(value, bool):
                raise ValueError(f"Lever '{self.name}' requires bool, got {type(value)}")

        self.value = value
        return True

    def reset(self) -> None:
        """Reset to default value."""
        self.value = self.default

    def bounds(self) -> str:
        """Get human-readable bounds string."""
        if self.type == LeverType.ENUM:
            return f"one of {self.options}"
        elif self.type in (LeverType.INT, LeverType.FLOAT):
            lo = self.min_value if self.min_value is not None else "-∞"
            hi = self.max_value if self.max_value is not None else "∞"
            return f"[{lo}, {hi}]"
        elif self.type == LeverType.BOOL:
            return "true/false"
        return "any"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "name": self.name,
            "type": self.type.value,
            "default": self.default,
            "value": self.value,
            "description": self.description,
            "category": self.category
        }
        if self.options:
            d["options"] = self.options
        if self.min_value is not None:
            d["min"] = self.min_value
        if self.max_value is not None:
            d["max"] = self.max_value
        if self.units:
            d["units"] = self.units
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Lever":
        """Create from dictionary."""
        return cls(
            name=d["name"],
            type=LeverType(d["type"]),
            default=d["default"],
            value=d.get("value", d["default"]),
            description=d.get("description", ""),
            options=d.get("options"),
            min_value=d.get("min"),
            max_value=d.get("max"),
            units=d.get("units"),
            category=d.get("category", "general")
        )

    def __repr__(self) -> str:
        return f"Lever({self.name}={self.value}, type={self.type.value})"


# Standard levers used across TRIX Forge
STANDARD_LEVERS = {
    "precision": Lever(
        name="precision",
        type=LeverType.ENUM,
        options=["fp32", "fp16", "fp8", "int8"],
        default="fp16",
        description="Numeric precision for weights and activations",
        category="quantization"
    ),
    "optimize": Lever(
        name="optimize",
        type=LeverType.ENUM,
        options=["none", "basic", "aggressive"],
        default="aggressive",
        description="Optimization level for compilation",
        category="compilation"
    ),
    "fuse_ops": Lever(
        name="fuse_ops",
        type=LeverType.BOOL,
        default=True,
        description="Fuse compatible operations (e.g., Linear+ReLU)",
        category="compilation"
    ),
    "debug_info": Lever(
        name="debug_info",
        type=LeverType.BOOL,
        default=False,
        description="Include debug symbols in output",
        category="compilation"
    ),
    "glassbox": Lever(
        name="glassbox",
        type=LeverType.BOOL,
        default=True,
        description="Enable Glassbox event emission",
        category="runtime"
    ),
}
