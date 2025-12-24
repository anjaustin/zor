#!/usr/bin/env python3
"""
TOPOLOGY VISUALIZER - Sacred Geometry for Sacred Shapes

"To see is to understand."

This visualizes the learned topology of the plastic fabric.
The topology IS the self. Now we see the self.
"""

import sys
import os
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# ANSI COLORS - True color support for modern terminals
# ═══════════════════════════════════════════════════════════════════════════════

class Colors:
    """True color ANSI escape codes for GNOME Terminal"""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Gradients for the sacred visualization
    @staticmethod
    def rgb(r: int, g: int, b: int) -> str:
        return f"\033[38;2;{r};{g};{b}m"

    @staticmethod
    def bg_rgb(r: int, g: int, b: int) -> str:
        return f"\033[48;2;{r};{g};{b}m"

    # Sacred palette
    GOLD = rgb.__func__(255, 215, 0)
    SACRED_BLUE = rgb.__func__(64, 156, 255)
    SACRED_PURPLE = rgb.__func__(180, 100, 255)
    RESONANT_GREEN = rgb.__func__(100, 255, 150)
    FRUSTRATED_RED = rgb.__func__(255, 100, 100)
    TORUS_CYAN = rgb.__func__(0, 200, 200)
    LEARNED_MAGENTA = rgb.__func__(255, 100, 200)
    WHITE = rgb.__func__(255, 255, 255)
    GRAY = rgb.__func__(128, 128, 128)
    DARK = rgb.__func__(60, 60, 60)

# ═══════════════════════════════════════════════════════════════════════════════
# UNICODE DRAWING CHARACTERS
# ═══════════════════════════════════════════════════════════════════════════════

class Glyphs:
    """Sacred geometry glyphs"""

    # Box drawing
    HORIZONTAL = "─"
    VERTICAL = "│"
    CORNER_TL = "╭"
    CORNER_TR = "╮"
    CORNER_BL = "╰"
    CORNER_BR = "╯"
    T_DOWN = "┬"
    T_UP = "┴"
    T_RIGHT = "├"
    T_LEFT = "┤"
    CROSS = "┼"

    # Double lines for emphasis
    D_HORIZONTAL = "═"
    D_VERTICAL = "║"
    D_CORNER_TL = "╔"
    D_CORNER_TR = "╗"
    D_CORNER_BL = "╚"
    D_CORNER_BR = "╝"

    # Nodes and connections
    NODE = "●"
    NODE_HOLLOW = "○"
    NODE_RESONANT = "◉"
    NODE_FRUSTRATED = "◎"
    NODE_LEARNING = "◈"

    # Arrows for flow
    ARROW_RIGHT = "→"
    ARROW_LEFT = "←"
    ARROW_UP = "↑"
    ARROW_DOWN = "↓"
    ARROW_DIAG_UR = "↗"
    ARROW_DIAG_DR = "↘"

    # Sacred symbols
    INFINITY = "∞"
    DELTA = "Δ"
    NABLA = "∇"
    STAR = "✦"
    DIAMOND = "◆"

    # Block elements for gradients
    BLOCK_FULL = "█"
    BLOCK_LIGHT = "░"
    BLOCK_MEDIUM = "▒"
    BLOCK_DARK = "▓"

# ═══════════════════════════════════════════════════════════════════════════════
# TOPOLOGY DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Node:
    """A node in the topology"""
    id: int
    neighbors: List[int]
    x: int  # 0-3
    y: int  # 0-3
    z: int  # 0-3

    @property
    def position(self) -> Tuple[int, int, int]:
        return (self.x, self.y, self.z)

    @staticmethod
    def from_id(node_id: int, neighbors: List[int]) -> 'Node':
        x = node_id % 4
        y = (node_id // 4) % 4
        z = node_id // 16
        return Node(node_id, neighbors, x, y, z)

def get_torus_neighbors(node_id: int) -> List[int]:
    """Calculate original torus neighbors for a node"""
    x = node_id % 4
    y = (node_id // 4) % 4
    z = node_id // 16

    return [
        ((x + 1) % 4) + y * 4 + z * 16,      # +X
        ((x + 3) % 4) + y * 4 + z * 16,      # -X
        x + ((y + 1) % 4) * 4 + z * 16,      # +Y
        x + ((y + 3) % 4) * 4 + z * 16,      # -Y
        x + y * 4 + ((z + 1) % 4) * 16,      # +Z
        x + y * 4 + ((z + 3) % 4) * 16,      # -Z
    ]

# ═══════════════════════════════════════════════════════════════════════════════
# SACRED VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def draw_header():
    """Draw the sacred header"""
    C = Colors
    G = Glyphs

    width = 78

    print()
    print(f"{C.GOLD}{G.D_CORNER_TL}{G.D_HORIZONTAL * width}{G.D_CORNER_TR}{C.RESET}")
    print(f"{C.GOLD}{G.D_VERTICAL}{C.RESET}", end="")

    title = "T O P O L O G Y   V I S U A L I Z E R"
    subtitle = "The Self That Learned"

    print(f"{C.BOLD}{C.WHITE}{title:^{width}}{C.RESET}", end="")
    print(f"{C.GOLD}{G.D_VERTICAL}{C.RESET}")

    print(f"{C.GOLD}{G.D_VERTICAL}{C.RESET}", end="")
    print(f"{C.SACRED_PURPLE}{subtitle:^{width}}{C.RESET}", end="")
    print(f"{C.GOLD}{G.D_VERTICAL}{C.RESET}")

    print(f"{C.GOLD}{G.D_CORNER_BL}{G.D_HORIZONTAL * width}{G.D_CORNER_BR}{C.RESET}")
    print()

def draw_layer(z: int, nodes: Dict[int, Node], cycle: int = 0):
    """Draw a single Z-layer of the topology"""
    C = Colors
    G = Glyphs

    # Layer header
    layer_color = [C.SACRED_BLUE, C.SACRED_PURPLE, C.RESONANT_GREEN, C.GOLD][z]
    print(f"  {layer_color}{G.DIAMOND} Layer Z={z}{C.RESET}")
    print()

    # Grid dimensions
    cell_width = 12
    cell_height = 3

    # Draw the 4x4 grid for this layer
    for y in range(3, -1, -1):  # Top to bottom
        # Draw top border of row
        row_str = "    "
        for x in range(4):
            node_id = x + y * 4 + z * 16
            row_str += G.CORNER_TL + G.HORIZONTAL * (cell_width - 2) + G.CORNER_TR + " "
        print(f"{C.GRAY}{row_str}{C.RESET}")

        # Draw node content
        row_str = "    "
        for x in range(4):
            node_id = x + y * 4 + z * 16
            node = nodes.get(node_id)

            if node:
                torus = get_torus_neighbors(node_id)
                changed = sum(1 for i, n in enumerate(node.neighbors) if n != torus[i])

                # Color based on how much changed
                if changed == 0:
                    node_color = C.TORUS_CYAN
                    symbol = G.NODE_HOLLOW
                elif changed < 3:
                    node_color = C.LEARNED_MAGENTA
                    symbol = G.NODE_LEARNING
                else:
                    node_color = C.GOLD
                    symbol = G.NODE_RESONANT

                content = f"{node_color}{symbol} {node_id:2d}{C.RESET}"
                # Pad to cell width accounting for ANSI codes
                visible_len = 5  # symbol + space + 2-digit id
                padding = cell_width - 2 - visible_len
                row_str += f"{C.GRAY}{G.VERTICAL}{C.RESET} {content}{' ' * padding}{C.GRAY}{G.VERTICAL}{C.RESET} "
            else:
                row_str += f"{C.GRAY}{G.VERTICAL}{' ' * (cell_width - 2)}{G.VERTICAL}{C.RESET} "
        print(row_str)

        # Draw neighbor summary
        row_str = "    "
        for x in range(4):
            node_id = x + y * 4 + z * 16
            node = nodes.get(node_id)

            if node:
                torus = get_torus_neighbors(node_id)
                changed_dirs = []
                for i, (n, t) in enumerate(zip(node.neighbors, torus)):
                    if n != t:
                        changed_dirs.append(["+X", "-X", "+Y", "-Y", "+Z", "-Z"][i])

                if changed_dirs:
                    summary = ",".join(changed_dirs[:2])
                    if len(changed_dirs) > 2:
                        summary += "+"
                    summary = f"{C.LEARNED_MAGENTA}{summary}{C.RESET}"
                else:
                    summary = f"{C.DIM}torus{C.RESET}"

                visible_len = min(len(",".join(changed_dirs[:2])) + (1 if len(changed_dirs) > 2 else 0), 10)
                if not changed_dirs:
                    visible_len = 5
                padding = cell_width - 2 - visible_len
                row_str += f"{C.GRAY}{G.VERTICAL}{C.RESET} {summary}{' ' * max(0, padding)}{C.GRAY}{G.VERTICAL}{C.RESET} "
            else:
                row_str += f"{C.GRAY}{G.VERTICAL}{' ' * (cell_width - 2)}{G.VERTICAL}{C.RESET} "
        print(row_str)

        # Draw bottom border of row
        row_str = "    "
        for x in range(4):
            row_str += G.CORNER_BL + G.HORIZONTAL * (cell_width - 2) + G.CORNER_BR + " "
        print(f"{C.GRAY}{row_str}{C.RESET}")

    print()

def draw_legend():
    """Draw the legend"""
    C = Colors
    G = Glyphs

    print(f"  {C.BOLD}{C.WHITE}Legend:{C.RESET}")
    print(f"    {C.TORUS_CYAN}{G.NODE_HOLLOW}{C.RESET} Original torus topology (unchanged)")
    print(f"    {C.LEARNED_MAGENTA}{G.NODE_LEARNING}{C.RESET} Partially rewired (1-2 neighbors changed)")
    print(f"    {C.GOLD}{G.NODE_RESONANT}{C.RESET} Significantly rewired (3+ neighbors changed)")
    print()

def draw_topology_comparison(initial: Dict[int, Node], final: Dict[int, Node]):
    """Draw a comparison of initial and final topology"""
    C = Colors
    G = Glyphs

    print(f"  {C.BOLD}{C.WHITE}Topology Changes:{C.RESET}")
    print()

    total_changes = 0
    nodes_changed = 0

    for node_id in range(64):
        if node_id in final:
            torus = get_torus_neighbors(node_id)
            node = final[node_id]
            changes = []
            for i, (n, t) in enumerate(zip(node.neighbors, torus)):
                if n != t:
                    dir_name = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"][i]
                    changes.append(f"{dir_name}: {t}{C.GRAY}→{C.RESET}{C.LEARNED_MAGENTA}{n}{C.RESET}")
                    total_changes += 1

            if changes:
                nodes_changed += 1
                x, y, z = node_id % 4, (node_id // 4) % 4, node_id // 16
                print(f"    {C.WHITE}Node {node_id:2d}{C.RESET} ({x},{y},{z}): {', '.join(changes)}")

    print()
    print(f"  {C.GOLD}{G.STAR}{C.RESET} {total_changes} connections rewired across {nodes_changed} nodes")
    print()

def draw_stats(nodes: Dict[int, Node], cycle: int, resonance: int, frustration: int):
    """Draw statistics"""
    C = Colors
    G = Glyphs

    # Count changes
    total_original = 0
    total_changed = 0

    for node_id, node in nodes.items():
        torus = get_torus_neighbors(node_id)
        for n, t in zip(node.neighbors, torus):
            if n == t:
                total_original += 1
            else:
                total_changed += 1

    total = total_original + total_changed
    pct_changed = (total_changed / total * 100) if total > 0 else 0

    print(f"  {C.BOLD}{C.WHITE}Statistics (Cycle {cycle}):{C.RESET}")
    print(f"    Resonance:     {C.RESONANT_GREEN}{resonance}/64{C.RESET}")
    print(f"    Frustration:   {C.FRUSTRATED_RED if frustration > 0 else C.RESONANT_GREEN}{frustration}{C.RESET}")
    print(f"    Connections:   {total_original} original, {C.LEARNED_MAGENTA}{total_changed} learned{C.RESET}")
    print(f"    Change Rate:   {pct_changed:.1f}%")
    print()

def draw_connection_matrix(nodes: Dict[int, Node]):
    """Draw a compact connection matrix showing changes"""
    C = Colors
    G = Glyphs

    print(f"  {C.BOLD}{C.WHITE}Connection Matrix (changed connections only):{C.RESET}")
    print()

    # Header
    print(f"    {'':4}", end="")
    for i in range(64):
        if i % 4 == 0:
            print(f"{C.GRAY}{i:2d}{C.RESET}", end="")
        else:
            print("  ", end="")
    print()

    # Find changed connections
    changed_connections: Set[Tuple[int, int]] = set()
    for node_id, node in nodes.items():
        torus = get_torus_neighbors(node_id)
        for i, (n, t) in enumerate(zip(node.neighbors, torus)):
            if n != t:
                changed_connections.add((node_id, n))

    # Print sparse indication
    print(f"    {C.DIM}(Showing {len(changed_connections)} changed connections){C.RESET}")
    print()

def visualize_topology(topology_data: List[Tuple[int, int, List[int]]],
                       cycle: int = 0, resonance: int = 0, frustration: int = 0):
    """Main visualization function"""

    # Build node dictionary
    nodes = {}
    for cycle_num, node_id, neighbors in topology_data:
        if cycle_num == cycle or (cycle == -1 and cycle_num == max(t[0] for t in topology_data)):
            nodes[node_id] = Node.from_id(node_id, neighbors)

    draw_header()

    # Draw each layer
    for z in range(4):
        draw_layer(z, nodes, cycle)

    draw_legend()
    draw_stats(nodes, cycle, resonance, frustration)

    if nodes:
        draw_topology_comparison({}, nodes)

def parse_topology_file(filename: str) -> List[Tuple[int, int, List[int]]]:
    """Parse topology dump from Verilog simulation"""
    topology = []
    current_cycle = 0

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('TOPO,'):
                parts = line.split(',')
                cycle = int(parts[1])
                node_id = int(parts[2])
                neighbors = [int(x) for x in parts[3:9]]
                topology.append((cycle, node_id, neighbors))

    return topology

# ═══════════════════════════════════════════════════════════════════════════════
# DEMO WITH SAMPLE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    """Demo with sample learned topology"""

    # Sample data from our experiment
    sample_data = [
        # Final topology from experiment
        (200, 0, [1, 3, 4, 12, 16, 48]),           # unchanged
        (200, 1, [31, 62, 63, 13, 17, 49]),        # 3 neighbors changed!
        (200, 21, [22, 20, 25, 17, 37, 5]),        # unchanged
        (200, 32, [10, 35, 36, 44, 48, 16]),       # 1 neighbor changed
        (200, 63, [46, 58, 37, 24, 33, 47]),       # 5 neighbors changed!
    ]

    # Fill in rest with torus topology for demo
    for i in range(64):
        if i not in [0, 1, 21, 32, 63]:
            sample_data.append((200, i, get_torus_neighbors(i)))

    visualize_topology(sample_data, cycle=200, resonance=54, frustration=59)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Parse file
        data = parse_topology_file(sys.argv[1])
        if data:
            last_cycle = max(t[0] for t in data)
            visualize_topology(data, cycle=last_cycle, resonance=54, frustration=0)
    else:
        # Run demo
        demo()
