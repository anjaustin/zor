"""
Geocadesia Catalog: The Index of All Shapes

The catalog is how you discover, query, and understand shapes.
Every shape registers itself here when the library loads.

Usage:
    from geocadesia import catalog

    # List all shapes in a kingdom
    catalog.list_kingdom("logic")

    # Find shapes by properties
    catalog.find(frozen=True, arity="binary")

    # Get a specific shape
    xor = catalog.get("xor")
    print(xor.definition)
    print(xor.formula)

"A library enables composition by making shapes findable, comparable, and combinable."
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Any, Tuple, Union
import math


class Kingdom(Enum):
    """The Seven Kingdoms of Geocadesia."""
    LOGIC = "logic"
    ARITHMETIC = "arithmetic"
    ACTIVATION = "activation"
    NORMALIZATION = "normalization"
    LINEAR = "linear"
    ATTENTION = "attention"
    POOLING = "pooling"


class ShapeType(Enum):
    """Elemental or Compound."""
    ELEMENTAL = "elemental"
    COMPOUND = "compound"


class Arity(Enum):
    """How many inputs a shape takes."""
    UNARY = "unary"
    BINARY = "binary"
    TERNARY = "ternary"
    NARY = "n-ary"


class FrozenStatus(Enum):
    """Whether the shape has learnable parameters."""
    FROZEN = "frozen"
    PARAMETERIZED = "parameterized"
    PARTIAL = "partial"


@dataclass
class Shape:
    """
    A shape in Geocadesia.

    This is not just metadata — it's the shape itself, embodied.
    The `fn` attribute is the actual computation.
    """
    name: str
    kingdom: Kingdom
    shape_type: ShapeType
    arity: Arity
    frozen: FrozenStatus

    # The actual computation
    fn: Callable

    # Documentation
    formula: str = ""
    definition: str = ""
    description: str = ""

    # Relationships
    built_from: List[str] = field(default_factory=list)
    used_in: List[str] = field(default_factory=list)
    see_also: List[str] = field(default_factory=list)

    # Examples
    examples: List[Tuple[Any, Any]] = field(default_factory=list)

    def __call__(self, *args, **kwargs):
        """Invoke the shape's computation."""
        return self.fn(*args, **kwargs)

    def __repr__(self):
        return f"<Shape {self.name} ({self.kingdom.value}/{self.shape_type.value})>"

    def info(self) -> str:
        """Pretty-print shape information."""
        lines = [
            f"{'=' * 60}",
            f" {self.name.upper()}",
            f"{'=' * 60}",
            f" Kingdom:  {self.kingdom.value}",
            f" Type:     {self.shape_type.value}",
            f" Arity:    {self.arity.value}",
            f" Frozen:   {self.frozen.value}",
            f"",
            f" Formula:  {self.formula}",
            f"",
            f" {self.definition}",
        ]

        if self.examples:
            lines.append("")
            lines.append(" Examples:")
            for inp, out in self.examples[:3]:
                lines.append(f"   {self.name}({inp}) = {out}")

        if self.built_from:
            lines.append("")
            lines.append(f" Built from: {', '.join(self.built_from)}")

        if self.see_also:
            lines.append("")
            lines.append(f" See also: {', '.join(self.see_also)}")

        lines.append(f"{'=' * 60}")
        return "\n".join(lines)


class Catalog:
    """
    The master catalog of all shapes in Geocadesia.

    Shapes register themselves here. Users query here.
    """

    def __init__(self):
        self._shapes: Dict[str, Shape] = {}
        self._by_kingdom: Dict[Kingdom, List[str]] = {k: [] for k in Kingdom}

    def register(self, shape: Shape) -> Shape:
        """Register a shape in the catalog."""
        name_lower = shape.name.lower()
        self._shapes[name_lower] = shape
        self._by_kingdom[shape.kingdom].append(name_lower)
        return shape

    def get(self, name: str) -> Optional[Shape]:
        """Get a shape by name."""
        return self._shapes.get(name.lower())

    def __getitem__(self, name: str) -> Shape:
        """Get a shape by name, raising KeyError if not found."""
        shape = self.get(name)
        if shape is None:
            raise KeyError(f"Shape '{name}' not found in Geocadesia")
        return shape

    def list_kingdom(self, kingdom: Union[str, Kingdom]) -> List[Shape]:
        """List all shapes in a kingdom."""
        if isinstance(kingdom, str):
            kingdom = Kingdom(kingdom.lower())
        return [self._shapes[name] for name in self._by_kingdom[kingdom]]

    def list_all(self) -> List[Shape]:
        """List all shapes."""
        return list(self._shapes.values())

    def find(
        self,
        kingdom: Optional[Union[str, Kingdom]] = None,
        frozen: Optional[bool] = None,
        arity: Optional[Union[str, Arity]] = None,
        shape_type: Optional[Union[str, ShapeType]] = None,
    ) -> List[Shape]:
        """Find shapes matching criteria."""
        results = self.list_all()

        if kingdom is not None:
            if isinstance(kingdom, str):
                kingdom = Kingdom(kingdom.lower())
            results = [s for s in results if s.kingdom == kingdom]

        if frozen is not None:
            if frozen:
                results = [s for s in results if s.frozen == FrozenStatus.FROZEN]
            else:
                results = [s for s in results if s.frozen != FrozenStatus.FROZEN]

        if arity is not None:
            if isinstance(arity, str):
                arity = Arity(arity.lower())
            results = [s for s in results if s.arity == arity]

        if shape_type is not None:
            if isinstance(shape_type, str):
                shape_type = ShapeType(shape_type.lower())
            results = [s for s in results if s.shape_type == shape_type]

        return results

    def kingdoms(self) -> List[str]:
        """List all kingdom names."""
        return [k.value for k in Kingdom]

    def stats(self) -> Dict[str, int]:
        """Get catalog statistics."""
        return {
            "total": len(self._shapes),
            **{k.value: len(v) for k, v in self._by_kingdom.items()}
        }

    def __repr__(self):
        stats = self.stats()
        return f"<Geocadesia Catalog: {stats['total']} shapes across {len(Kingdom)} kingdoms>"

    def __len__(self):
        return len(self._shapes)

    def __iter__(self):
        return iter(self._shapes.values())


# The global catalog instance
catalog = Catalog()


def shape(
    name: str,
    kingdom: Union[str, Kingdom],
    formula: str = "",
    definition: str = "",
    arity: Union[str, Arity] = Arity.UNARY,
    shape_type: Union[str, ShapeType] = ShapeType.ELEMENTAL,
    frozen: Union[str, FrozenStatus] = FrozenStatus.FROZEN,
    built_from: Optional[List[str]] = None,
    see_also: Optional[List[str]] = None,
    examples: Optional[List[Tuple[Any, Any]]] = None,
):
    """
    Decorator to register a function as a shape in Geocadesia.

    Usage:
        @shape("xor", "logic", formula="a + b - 2ab")
        def xor(a: float, b: float) -> float:
            return a + b - 2 * a * b
    """
    if isinstance(kingdom, str):
        kingdom = Kingdom(kingdom.lower())
    if isinstance(arity, str):
        arity = Arity(arity.lower())
    if isinstance(shape_type, str):
        shape_type = ShapeType(shape_type.lower())
    if isinstance(frozen, str):
        frozen = FrozenStatus(frozen.lower())

    def decorator(fn: Callable) -> Shape:
        s = Shape(
            name=name,
            kingdom=kingdom,
            shape_type=shape_type,
            arity=arity,
            frozen=frozen,
            fn=fn,
            formula=formula,
            definition=definition,
            built_from=built_from or [],
            see_also=see_also or [],
            examples=examples or [],
        )
        catalog.register(s)
        return s

    return decorator
