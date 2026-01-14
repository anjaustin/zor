"""
Shape Composer - Build hierarchical shape trees from discovered patterns.

The composer takes patterns from the matcher and constructs a hierarchical
representation of the circuit, identifying reusable composed shapes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Union
from collections import defaultdict

from .matcher import find_patterns, Pattern
from .hasher import build_wire_maps
from .library import match_known_pattern


@dataclass
class AtomicShape:
    """An atomic (primitive) shape - a single gate."""
    gate_id: str
    shape_type: str         # "$_AND_", "$_XOR_", etc.
    inputs: List[int]       # Input wire bits
    output: int             # Output wire bit

    @property
    def display_type(self) -> str:
        """Human-readable type name."""
        return self.shape_type.strip("$_")


@dataclass
class ComposedShape:
    """A composed shape - a pattern of gates."""
    shape_id: str
    name: str                   # "FullAdder", "Pattern_001", etc.
    known_type: Optional[str]   # If matched to known pattern
    gate_types: Tuple[str, ...] # Sorted tuple of gate types
    size: int                   # Number of gates
    instances: int              # How many times this appears
    children: List[str]         # Gate IDs in canonical instance
    input_wires: Set[int]       # External inputs
    output_wires: Set[int]      # External outputs

    @property
    def display_name(self) -> str:
        if self.known_type:
            return f"{self.known_type} ({self.size} gates)"
        return f"{self.name} ({self.size} gates)"

    @property
    def compression(self) -> int:
        """Gates saved by using this composed shape."""
        return self.size * (self.instances - 1)


@dataclass
class ShapeTree:
    """A hierarchical tree of shapes representing a circuit."""
    name: str
    root_gates: List[str]               # Execution order of all gates
    composed_shapes: Dict[str, ComposedShape]  # Discovered composed shapes
    gate_to_shape: Dict[str, str]       # Map gate_id to composed shape containing it
    atomic_gates: Set[str]              # Gates not in any composed shape

    # Statistics
    total_gates: int
    composed_gates: int
    unique_shapes: int
    total_instances: int

    @property
    def compression_ratio(self) -> float:
        if self.total_gates == 0:
            return 0.0
        composed = sum(s.compression for s in self.composed_shapes.values())
        return composed / self.total_gates

    def summary(self) -> str:
        lines = [
            f"ShapeTree: {self.name}",
            f"=" * 50,
            f"Total gates: {self.total_gates}",
            f"Composed into {self.unique_shapes} shapes ({self.total_instances} instances)",
            f"Atomic gates: {len(self.atomic_gates)}",
            f"Gates in compositions: {self.composed_gates}",
            f"Compression ratio: {self.compression_ratio:.1%}",
            "",
            "Discovered Shapes:",
        ]

        for shape in sorted(self.composed_shapes.values(), key=lambda s: -s.compression):
            lines.append(f"  {shape.display_name}: {shape.instances}x (saves {shape.compression} gates)")

        return "\n".join(lines)


def compose(
    system,
    min_instances: int = 2,
    min_size: int = 2,
    max_depth: int = 3
) -> ShapeTree:
    """
    Compose a flat gate graph into a hierarchical shape tree.

    Args:
        system: A System from ingest
        min_instances: Minimum pattern occurrences
        min_size: Minimum pattern size
        max_depth: Maximum pattern expansion depth

    Returns:
        ShapeTree representing the hierarchical structure
    """
    # Find patterns
    patterns = find_patterns(system, min_instances, min_size, max_depth)

    # Build composed shapes
    composed_shapes: Dict[str, ComposedShape] = {}
    gate_to_shape: Dict[str, str] = {}
    used_gates: Set[str] = set()

    for pattern in patterns:
        # Skip if gates already claimed by a higher-value pattern
        canonical_gates = list(pattern.instances[0])
        if any(g in used_gates for g in canonical_gates):
            continue

        # Try to match known pattern
        known_type = match_known_pattern(pattern.gate_types)

        # Create composed shape
        shape = ComposedShape(
            shape_id=pattern.pattern_id,
            name=known_type or pattern.pattern_id,
            known_type=known_type,
            gate_types=pattern.gate_types,
            size=pattern.size,
            instances=pattern.instance_count,
            children=canonical_gates,
            input_wires=pattern.input_wires,
            output_wires=pattern.output_wires,
        )
        composed_shapes[shape.shape_id] = shape

        # Mark gates as used
        for instance in pattern.instances:
            for gate_id in instance:
                if gate_id not in used_gates:
                    gate_to_shape[gate_id] = shape.shape_id
                    used_gates.add(gate_id)

    # Find atomic gates (not in any pattern)
    atomic_gates = set(system.cells.keys()) - used_gates

    # Build statistics
    total_gates = len(system.cells)
    composed_gates = len(used_gates)
    unique_shapes = len(composed_shapes)
    total_instances = sum(s.instances for s in composed_shapes.values())

    return ShapeTree(
        name=system.name,
        root_gates=system.execution_order,
        composed_shapes=composed_shapes,
        gate_to_shape=gate_to_shape,
        atomic_gates=atomic_gates,
        total_gates=total_gates,
        composed_gates=composed_gates,
        unique_shapes=unique_shapes,
        total_instances=total_instances,
    )


def decompose(tree: ShapeTree, system) -> None:
    """
    Expand a shape tree back to flat gates.

    This verifies that composition is lossless.
    """
    # The original system already has all the gates
    # This function would verify equivalence
    pass


def shape_hierarchy(tree: ShapeTree) -> str:
    """
    Generate a hierarchical view of the shape tree.

    Returns:
        ASCII tree representation
    """
    lines = [f"Circuit: {tree.name}"]

    # Group shapes by type
    known = [s for s in tree.composed_shapes.values() if s.known_type]
    unknown = [s for s in tree.composed_shapes.values() if not s.known_type]

    if known:
        lines.append("├── Known Patterns:")
        for shape in sorted(known, key=lambda s: -s.instances):
            lines.append(f"│   ├── {shape.known_type}: {shape.instances}x")

    if unknown:
        lines.append("├── Discovered Patterns:")
        for shape in sorted(unknown, key=lambda s: -s.instances):
            lines.append(f"│   ├── {shape.name}: {shape.instances}x ({shape.size} gates)")

    if tree.atomic_gates:
        lines.append(f"└── Atomic: {len(tree.atomic_gates)} ungrouped gates")

    return "\n".join(lines)
