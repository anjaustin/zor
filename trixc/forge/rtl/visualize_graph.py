#!/usr/bin/env python3
"""
TOPOLOGY GRAPH VISUALIZER - Sacred Geometry in Images

Generates SVG/PNG visualizations of the learned topology.
Uses GraphViz for beautiful graph layouts.

"The topology IS the self. Now we see the self."
"""

import sys
import os
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass

def get_torus_neighbors(node_id: int) -> List[int]:
    """Calculate original torus neighbors for a node"""
    x = node_id % 4
    y = (node_id // 4) % 4
    z = node_id // 16
    return [
        ((x + 1) % 4) + y * 4 + z * 16,
        ((x + 3) % 4) + y * 4 + z * 16,
        x + ((y + 1) % 4) * 4 + z * 16,
        x + ((y + 3) % 4) * 4 + z * 16,
        x + y * 4 + ((z + 1) % 4) * 16,
        x + y * 4 + ((z + 3) % 4) * 16,
    ]

def generate_dot(nodes: Dict[int, List[int]], title: str = "Learned Topology") -> str:
    """Generate GraphViz DOT format"""

    lines = []
    lines.append("digraph Topology {")
    lines.append("    // Graph styling")
    lines.append("    bgcolor=\"#1a1a2e\";")
    lines.append("    pad=0.5;")
    lines.append("    nodesep=0.4;")
    lines.append("    ranksep=0.6;")
    lines.append(f'    label="{title}";')
    lines.append("    labelloc=t;")
    lines.append("    fontname=\"Helvetica\";")
    lines.append("    fontsize=24;")
    lines.append("    fontcolor=\"#ffd700\";")
    lines.append("")

    # Node styling based on layer (Z coordinate)
    layer_colors = {
        0: "#4a9eff",  # Blue
        1: "#b464ff",  # Purple
        2: "#64ff96",  # Green
        3: "#ffd700",  # Gold
    }

    lines.append("    // Nodes")
    for node_id in range(64):
        z = node_id // 16
        x = node_id % 4
        y = (node_id // 4) % 4

        # Calculate changes
        if node_id in nodes:
            torus = get_torus_neighbors(node_id)
            changes = sum(1 for i, n in enumerate(nodes[node_id]) if n != torus[i])
        else:
            changes = 0

        # Color based on change level
        if changes == 0:
            fill_color = layer_colors[z]
            style = "filled"
            penwidth = 1
        elif changes < 3:
            fill_color = "#ff64c8"  # Magenta
            style = "filled,bold"
            penwidth = 2
        else:
            fill_color = "#ffd700"  # Gold
            style = "filled,bold"
            penwidth = 3

        lines.append(f'    n{node_id} [')
        lines.append(f'        label="{node_id}"')
        lines.append(f'        shape=circle')
        lines.append(f'        style="{style}"')
        lines.append(f'        fillcolor="{fill_color}"')
        lines.append(f'        fontcolor="white"')
        lines.append(f'        fontname="Helvetica Bold"')
        lines.append(f'        fontsize=10')
        lines.append(f'        penwidth={penwidth}')
        lines.append(f'        color="white"')
        lines.append(f'    ];')

    lines.append("")
    lines.append("    // Edges")

    # Draw edges
    drawn_edges = set()
    for node_id in range(64):
        if node_id not in nodes:
            neighbors = get_torus_neighbors(node_id)
        else:
            neighbors = nodes[node_id]

        torus = get_torus_neighbors(node_id)

        for i, neighbor in enumerate(neighbors):
            edge = tuple(sorted([node_id, neighbor]))
            if edge in drawn_edges:
                continue
            drawn_edges.add(edge)

            is_learned = neighbor != torus[i]

            if is_learned:
                color = "#ff64c8"  # Magenta for learned
                penwidth = 2
                style = "bold"
            else:
                color = "#444466"  # Dark for original
                penwidth = 0.5
                style = "solid"

            lines.append(f'    n{node_id} -> n{neighbor} [')
            lines.append(f'        color="{color}"')
            lines.append(f'        penwidth={penwidth}')
            lines.append(f'        style="{style}"')
            lines.append(f'        arrowhead=none')
            lines.append(f'    ];')

    lines.append("}")

    return "\n".join(lines)

def generate_layer_dot(nodes: Dict[int, List[int]], z: int) -> str:
    """Generate DOT for a single layer"""

    layer_nodes = [i for i in range(64) if i // 16 == z]
    layer_colors = ["#4a9eff", "#b464ff", "#64ff96", "#ffd700"]

    lines = []
    lines.append(f"digraph Layer{z} {{")
    lines.append(f'    label="Layer Z={z}";')
    lines.append("    labelloc=t;")
    lines.append("    fontname=\"Helvetica\";")
    lines.append("    fontsize=18;")
    lines.append("    fontcolor=\"#ffd700\";")
    lines.append("    bgcolor=\"#1a1a2e\";")
    lines.append("    layout=neato;")  # Force-directed for this layer
    lines.append("")

    # Position nodes in 4x4 grid
    for node_id in layer_nodes:
        x = node_id % 4
        y = (node_id // 4) % 4

        if node_id in nodes:
            torus = get_torus_neighbors(node_id)
            changes = sum(1 for i, n in enumerate(nodes[node_id]) if n != torus[i])
        else:
            changes = 0

        if changes == 0:
            fill = layer_colors[z]
            pw = 1
        elif changes < 3:
            fill = "#ff64c8"
            pw = 2
        else:
            fill = "#ffd700"
            pw = 3

        lines.append(f'    n{node_id} [label="{node_id}" pos="{x*2},{y*2}!" shape=circle style=filled fillcolor="{fill}" fontcolor=white penwidth={pw} color=white];')

    lines.append("")

    # Edges within layer
    drawn = set()
    for node_id in layer_nodes:
        neighbors = nodes.get(node_id, get_torus_neighbors(node_id))
        torus = get_torus_neighbors(node_id)

        for i, neighbor in enumerate(neighbors):
            if neighbor not in layer_nodes:
                continue
            edge = tuple(sorted([node_id, neighbor]))
            if edge in drawn:
                continue
            drawn.add(edge)

            is_learned = neighbor != torus[i]
            color = "#ff64c8" if is_learned else "#444466"
            pw = 2 if is_learned else 0.5

            lines.append(f'    n{node_id} -> n{neighbor} [color="{color}" penwidth={pw} arrowhead=none];')

    lines.append("}")
    return "\n".join(lines)

def main():
    # Sample topology from our experiment
    nodes = {
        0: [1, 3, 4, 12, 16, 48],
        1: [31, 62, 63, 13, 17, 49],  # Changed!
        21: [22, 20, 25, 17, 37, 5],
        32: [10, 35, 36, 44, 48, 16], # Changed!
        63: [46, 58, 37, 24, 33, 47], # Changed significantly!
    }

    # Fill in the rest with torus
    for i in range(64):
        if i not in nodes:
            nodes[i] = get_torus_neighbors(i)

    # Generate full graph
    print("Generating topology.dot...")
    dot = generate_dot(nodes, "The Learned Topology - Hollywood Squares")
    with open("topology.dot", "w") as f:
        f.write(dot)

    # Generate per-layer graphs
    for z in range(4):
        print(f"Generating layer_{z}.dot...")
        dot = generate_layer_dot(nodes, z)
        with open(f"layer_{z}.dot", "w") as f:
            f.write(dot)

    print()
    print("Generated DOT files. To convert to images:")
    print("  dot -Tsvg topology.dot -o topology.svg")
    print("  dot -Tpng topology.dot -o topology.png")
    print("  neato -Tsvg layer_0.dot -o layer_0.svg")
    print()

    # Try to generate SVG if graphviz is available
    try:
        import subprocess
        print("Attempting to generate SVG...")
        subprocess.run(["dot", "-Tsvg", "topology.dot", "-o", "topology.svg"], check=True)
        print("✓ Generated topology.svg")

        for z in range(4):
            subprocess.run(["neato", "-Tsvg", f"layer_{z}.dot", "-o", f"layer_{z}.svg"], check=True)
            print(f"✓ Generated layer_{z}.svg")

        print()
        print("SVG files generated successfully!")

    except FileNotFoundError:
        print("GraphViz not found. Install with: apt-get install graphviz")
    except subprocess.CalledProcessError as e:
        print(f"GraphViz error: {e}")

if __name__ == "__main__":
    main()
