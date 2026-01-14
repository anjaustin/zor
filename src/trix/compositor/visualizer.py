"""
Visualizer - Generate visual representations of shape trees.

Provides ASCII tree output, statistics, and optional Graphviz DOT format.
"""

from typing import Dict, List, Optional
from .composer import ShapeTree, ComposedShape
from .library import pattern_description


def visualize(tree: ShapeTree, show_gates: bool = False) -> str:
    """
    Generate an ASCII visualization of a shape tree.

    Args:
        tree: The ShapeTree to visualize
        show_gates: Whether to show individual gate IDs

    Returns:
        ASCII tree representation
    """
    lines = []

    # Header
    lines.append(f"{'=' * 60}")
    lines.append(f"  SHAPE TREE: {tree.name}")
    lines.append(f"{'=' * 60}")
    lines.append("")

    # Statistics
    lines.append("STATISTICS:")
    lines.append(f"  Total gates:         {tree.total_gates}")
    lines.append(f"  Gates in patterns:   {tree.composed_gates}")
    lines.append(f"  Atomic gates:        {len(tree.atomic_gates)}")
    lines.append(f"  Unique shapes:       {tree.unique_shapes}")
    lines.append(f"  Total instances:     {tree.total_instances}")
    lines.append(f"  Compression:         {tree.compression_ratio:.1%}")
    lines.append("")

    # Composed shapes
    if tree.composed_shapes:
        lines.append("DISCOVERED PATTERNS:")
        lines.append("-" * 40)

        # Group by known vs unknown
        known = [(s.known_type, s) for s in tree.composed_shapes.values() if s.known_type]
        unknown = [(s.name, s) for s in tree.composed_shapes.values() if not s.known_type]

        if known:
            lines.append("")
            lines.append("  Known Patterns:")
            for name, shape in sorted(known, key=lambda x: -x[1].instances):
                desc = pattern_description(name)
                lines.append(f"    [{name}] {shape.instances}x")
                lines.append(f"      {desc}")
                lines.append(f"      {shape.size} gates, saves {shape.compression} gates")
                if show_gates:
                    lines.append(f"      Gates: {', '.join(shape.children[:5])}{'...' if len(shape.children) > 5 else ''}")

        if unknown:
            lines.append("")
            lines.append("  Discovered Patterns:")
            for name, shape in sorted(unknown, key=lambda x: -x[1].instances):
                lines.append(f"    [{name}] {shape.instances}x")
                lines.append(f"      {shape.size} gates: {', '.join(t.strip('$_') for t in shape.gate_types[:5])}{'...' if len(shape.gate_types) > 5 else ''}")
                lines.append(f"      Saves {shape.compression} gates")
                if show_gates:
                    lines.append(f"      Gates: {', '.join(shape.children[:5])}{'...' if len(shape.children) > 5 else ''}")

    # Atomic gates summary
    if tree.atomic_gates:
        lines.append("")
        lines.append("ATOMIC GATES:")
        lines.append(f"  {len(tree.atomic_gates)} gates not matching any pattern")
        if show_gates and len(tree.atomic_gates) <= 20:
            lines.append(f"  IDs: {', '.join(sorted(tree.atomic_gates))}")

    lines.append("")
    lines.append(f"{'=' * 60}")

    return "\n".join(lines)


def visualize_stats(tree: ShapeTree) -> str:
    """
    Generate a compact statistics summary.

    Args:
        tree: The ShapeTree to summarize

    Returns:
        Compact stats string
    """
    known_count = sum(1 for s in tree.composed_shapes.values() if s.known_type)
    unknown_count = tree.unique_shapes - known_count

    return (
        f"[{tree.name}] "
        f"{tree.total_gates} gates → "
        f"{tree.unique_shapes} patterns "
        f"({known_count} known, {unknown_count} discovered) | "
        f"Compression: {tree.compression_ratio:.1%}"
    )


def visualize_hierarchy(tree: ShapeTree, system) -> str:
    """
    Generate a hierarchical block diagram view.

    Args:
        tree: The ShapeTree
        system: Original System for additional info

    Returns:
        Hierarchical ASCII diagram
    """
    lines = [f"CIRCUIT: {tree.name}"]
    lines.append("│")

    # Group shapes by size (larger = higher level)
    shapes_by_size = sorted(
        tree.composed_shapes.values(),
        key=lambda s: (-s.size, -s.instances)
    )

    for i, shape in enumerate(shapes_by_size):
        is_last = (i == len(shapes_by_size) - 1) and not tree.atomic_gates
        prefix = "└──" if is_last else "├──"

        name = shape.known_type or shape.name
        lines.append(f"{prefix} {name} ({shape.instances}x)")

        # Show gate composition
        child_prefix = "    " if is_last else "│   "
        gate_summary = ", ".join(t.strip("$_") for t in shape.gate_types[:4])
        if len(shape.gate_types) > 4:
            gate_summary += f", +{len(shape.gate_types) - 4} more"
        lines.append(f"{child_prefix}└── [{gate_summary}]")

    if tree.atomic_gates:
        lines.append(f"└── Atomic ({len(tree.atomic_gates)} gates)")

    return "\n".join(lines)


def to_dot(tree: ShapeTree) -> str:
    """
    Generate Graphviz DOT format for visualization.

    Args:
        tree: The ShapeTree

    Returns:
        DOT format string (can be rendered with Graphviz)
    """
    lines = ["digraph ShapeTree {"]
    lines.append("  rankdir=TB;")
    lines.append("  node [shape=box];")
    lines.append("")

    # Root node
    lines.append(f'  root [label="{tree.name}\\n{tree.total_gates} gates"];')

    # Shape nodes
    for shape_id, shape in tree.composed_shapes.items():
        name = shape.known_type or shape.name
        label = f"{name}\\n{shape.instances}x, {shape.size} gates"
        color = "lightblue" if shape.known_type else "lightyellow"
        lines.append(f'  {shape_id} [label="{label}", style=filled, fillcolor={color}];')
        lines.append(f'  root -> {shape_id};')

    # Atomic node
    if tree.atomic_gates:
        lines.append(f'  atomic [label="Atomic\\n{len(tree.atomic_gates)} gates", style=filled, fillcolor=lightgray];')
        lines.append('  root -> atomic;')

    lines.append("}")
    return "\n".join(lines)
