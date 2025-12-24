"""
MDS Lattice - Morphogenic Deterministic System

The lattice contains all computation implicitly.
Navigation through the lattice IS computation.
No arithmetic. Just routing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Node:
    """A node in the MDS lattice."""
    id: str
    op: str              # 'INPUT', 'XOR', 'AND', 'OR', 'NOT'
    src_a: Optional[str] = None
    src_b: Optional[str] = None
    value: int = 0
    is_output: bool = False
    output_name: Optional[str] = None


class MDS_Lattice:
    """
    Morphogenic Deterministic System Lattice.

    The lattice is a static structure that contains all possible
    computations for a given function. Navigation through the
    lattice produces results without arithmetic computation.
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.levels: List[List[str]] = []
        self.outputs: List[str] = []
        self.input_names: List[str] = []

    def add_input(self, name: str) -> str:
        """Add an input node to level 0."""
        node = Node(id=name, op='INPUT')
        self.nodes[name] = node
        if len(self.levels) == 0:
            self.levels.append([])
        self.levels[0].append(name)
        self.input_names.append(name)
        return name

    def add_node(self, node_id: str, op: str, src_a: str, src_b: Optional[str] = None,
                 level: int = None, is_output: bool = False, output_name: str = None) -> str:
        """Add a computation node."""
        node = Node(
            id=node_id,
            op=op,
            src_a=src_a,
            src_b=src_b,
            is_output=is_output,
            output_name=output_name
        )
        self.nodes[node_id] = node

        # Ensure level exists
        while len(self.levels) <= level:
            self.levels.append([])
        self.levels[level].append(node_id)

        if is_output:
            self.outputs.append(node_id)

        return node_id

    def navigate(self, inputs: Dict[str, int], verbose: bool = False) -> Dict[str, int]:
        """
        Navigate the lattice with given inputs.

        This is NOT computation. This is ROUTING.
        Each node's value is determined by following paths,
        not by calculating polynomials.
        """
        if verbose:
            print(f"\n  Navigating lattice...")
            print(f"  Inputs: {inputs}")

        # Set input values
        for name, value in inputs.items():
            if name in self.nodes:
                self.nodes[name].value = value

        # Navigate through levels
        for level_idx, level in enumerate(self.levels):
            if level_idx == 0:
                continue  # Skip input level

            if verbose:
                print(f"\n  Level {level_idx}:")

            for node_id in level:
                node = self.nodes[node_id]
                a = self.nodes[node.src_a].value
                b = self.nodes[node.src_b].value if node.src_b else 0

                # ROUTING DECISIONS (not arithmetic!)
                # These are path selections based on input signals
                if node.op == 'XOR':
                    # Route: if signals differ, take path 1; else path 0
                    node.value = 1 if a != b else 0
                    path_desc = "different" if a != b else "same"
                elif node.op == 'AND':
                    # Route: if both signals high, take path 1; else path 0
                    node.value = 1 if (a == 1 and b == 1) else 0
                    path_desc = "both high" if node.value else "not both"
                elif node.op == 'OR':
                    # Route: if either signal high, take path 1; else path 0
                    node.value = 1 if (a == 1 or b == 1) else 0
                    path_desc = "at least one" if node.value else "neither"
                elif node.op == 'NOT':
                    # Route: invert the path
                    node.value = 1 - a
                    path_desc = "inverted"
                else:
                    node.value = 0
                    path_desc = "unknown"

                if verbose:
                    out_marker = f" ({node.output_name})" if node.is_output else ""
                    print(f"    {node_id} = {node.op}({a},{b}) → path: {path_desc} → {node.value}{out_marker}")

        # Collect outputs
        result = {}
        for node_id in self.outputs:
            node = self.nodes[node_id]
            name = node.output_name or node_id
            result[name] = node.value

        return result

    def node_count(self) -> int:
        """Return total number of nodes."""
        return len(self.nodes)

    def visualize(self) -> str:
        """Return ASCII visualization of the lattice."""
        lines = []
        lines.append("MDS LATTICE STRUCTURE")
        lines.append("=" * 50)

        for level_idx, level in enumerate(self.levels):
            if level_idx == 0:
                lines.append(f"\nLevel {level_idx} (INPUTS):")
            else:
                lines.append(f"\nLevel {level_idx}:")

            for node_id in level:
                node = self.nodes[node_id]
                if node.op == 'INPUT':
                    lines.append(f"  [{node_id}] INPUT")
                else:
                    src_b_str = f", {node.src_b}" if node.src_b else ""
                    out_str = f" → {node.output_name}" if node.is_output else ""
                    lines.append(f"  [{node_id}] {node.op}({node.src_a}{src_b_str}){out_str}")

        lines.append("\n" + "=" * 50)
        lines.append(f"Total nodes: {self.node_count()}")
        lines.append(f"Outputs: {[self.nodes[o].output_name or o for o in self.outputs]}")

        return "\n".join(lines)


def build_adder_lattice(bits: int) -> MDS_Lattice:
    """
    Build an MDS lattice for an n-bit adder.

    The lattice structure IS the adder. Navigation produces sums.
    """
    lattice = MDS_Lattice()

    # Level 0: Inputs
    for i in range(bits):
        lattice.add_input(f"a{i}")
        lattice.add_input(f"b{i}")

    current_level = 1
    prev_carry = None

    for i in range(bits):
        a_node = f"a{i}"
        b_node = f"b{i}"

        # XOR(a, b) - partial sum
        xor_ab = f"xor_{i}"
        lattice.add_node(xor_ab, 'XOR', a_node, b_node, level=current_level)

        # AND(a, b) - generate carry
        and_ab = f"and_{i}"
        lattice.add_node(and_ab, 'AND', a_node, b_node, level=current_level)

        if prev_carry is None:
            # First bit: sum = XOR(a,b), carry = AND(a,b)
            # Mark XOR as output s0
            lattice.nodes[xor_ab].is_output = True
            lattice.nodes[xor_ab].output_name = f"s{i}"
            lattice.outputs.append(xor_ab)
            prev_carry = and_ab
        else:
            current_level += 1

            # Sum bit: XOR(XOR(a,b), carry_in)
            sum_node = f"sum_{i}"
            lattice.add_node(sum_node, 'XOR', xor_ab, prev_carry, level=current_level,
                           is_output=True, output_name=f"s{i}")

            # Propagate carry: AND(carry_in, XOR(a,b))
            prop_carry = f"prop_{i}"
            lattice.add_node(prop_carry, 'AND', prev_carry, xor_ab, level=current_level)

            # New carry: OR(AND(a,b), propagate)
            new_carry = f"carry_{i}"
            lattice.add_node(new_carry, 'OR', and_ab, prop_carry, level=current_level)

            prev_carry = new_carry

        current_level += 1

    # Final carry is the MSB of the sum
    if prev_carry:
        lattice.nodes[prev_carry].is_output = True
        lattice.nodes[prev_carry].output_name = f"s{bits}"
        if prev_carry not in lattice.outputs:
            lattice.outputs.append(prev_carry)

    # Sort outputs by name
    lattice.outputs = sorted(lattice.outputs,
                             key=lambda x: int(lattice.nodes[x].output_name[1:]))

    return lattice
