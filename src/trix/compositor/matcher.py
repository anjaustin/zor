"""
Pattern Matcher - Find and verify repeated patterns in gate graphs.

Takes candidate patterns from the hasher and expands them to find
maximal matching subgraphs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, FrozenSet
from collections import defaultdict

from .hasher import build_wire_maps, hash_all_gates


@dataclass
class Pattern:
    """A discovered pattern in the gate graph."""
    pattern_id: str
    gates: FrozenSet[str]           # Set of gate IDs in the pattern
    gate_types: Tuple[str, ...]     # Sorted tuple of gate types
    internal_wires: Set[int]        # Wires internal to the pattern
    input_wires: Set[int]           # Wires coming from outside
    output_wires: Set[int]          # Wires going to outside
    instances: List[FrozenSet[str]] # All instances of this pattern

    @property
    def size(self) -> int:
        return len(self.gates)

    @property
    def instance_count(self) -> int:
        return len(self.instances)

    @property
    def gates_saved(self) -> int:
        """How many gates are saved by this pattern."""
        return self.size * (self.instance_count - 1)

    def __hash__(self):
        return hash(self.pattern_id)


@dataclass
class PatternInstance:
    """A specific instance of a pattern."""
    pattern_id: str
    gates: FrozenSet[str]
    wire_mapping: Dict[int, int]  # Maps pattern wires to instance wires


def extract_subgraph(
    seed_gate: str,
    cells: Dict,
    wire_to_driver: Dict[int, str],
    wire_to_consumers: Dict[int, List[str]],
    depth: int = 1
) -> Set[str]:
    """
    Extract the subgraph around a seed gate.

    Args:
        seed_gate: Starting gate
        cells: All cells in the system
        wire_to_driver: Map from wire to driving cell
        wire_to_consumers: Map from wire to consuming cells
        depth: How many levels to expand

    Returns:
        Set of gate IDs in the subgraph
    """
    subgraph = {seed_gate}
    frontier = {seed_gate}

    for _ in range(depth):
        new_frontier = set()

        for gate_id in frontier:
            if gate_id not in cells:
                continue
            cell = cells[gate_id]

            # Add fanin gates
            for port in cell.input_ports:
                for bit in cell.connections.get(port, []):
                    if isinstance(bit, int) and bit in wire_to_driver:
                        driver = wire_to_driver[bit]
                        if driver in cells and driver not in subgraph:
                            new_frontier.add(driver)
                            subgraph.add(driver)

            # Add fanout gates
            out_bits = cell.connections.get(cell.output_port, [])
            for bit in out_bits:
                if isinstance(bit, int) and bit in wire_to_consumers:
                    for consumer in wire_to_consumers[bit]:
                        if consumer in cells and consumer not in subgraph:
                            new_frontier.add(consumer)
                            subgraph.add(consumer)

        frontier = new_frontier

    return subgraph


def get_subgraph_signature(gates: Set[str], cells: Dict) -> Tuple[str, ...]:
    """Get a canonical signature for a subgraph (sorted gate types)."""
    types = [cells[g].cell_type for g in gates if g in cells]
    return tuple(sorted(types))


def find_boundary_wires(
    gates: Set[str],
    cells: Dict,
    wire_to_driver: Dict[int, str]
) -> Tuple[Set[int], Set[int], Set[int]]:
    """
    Find the boundary wires of a subgraph.

    Returns:
        (internal_wires, input_wires, output_wires)
    """
    internal = set()
    inputs = set()
    outputs = set()

    # Collect all wires used by this subgraph
    for gate_id in gates:
        if gate_id not in cells:
            continue
        cell = cells[gate_id]

        # Input wires
        for port in cell.input_ports:
            for bit in cell.connections.get(port, []):
                if isinstance(bit, int):
                    driver = wire_to_driver.get(bit)
                    if driver in gates:
                        internal.add(bit)
                    else:
                        inputs.add(bit)

        # Output wires - check if they go outside
        out_bits = cell.connections.get(cell.output_port, [])
        for bit in out_bits:
            if isinstance(bit, int):
                internal.add(bit)  # Assume internal, may be output too

    # A wire is an output if it's driven by a gate in the subgraph
    # and consumed by something outside (handled later in composition)
    for gate_id in gates:
        if gate_id not in cells:
            continue
        cell = cells[gate_id]
        out_bits = cell.connections.get(cell.output_port, [])
        for bit in out_bits:
            if isinstance(bit, int):
                outputs.add(bit)

    return internal, inputs, outputs


def subgraphs_isomorphic(
    gates_a: Set[str],
    gates_b: Set[str],
    cells: Dict,
    wire_to_driver: Dict[int, str]
) -> bool:
    """
    Check if two subgraphs are isomorphic.

    Simple check: same gate type signature.
    More sophisticated: check connectivity pattern.
    """
    sig_a = get_subgraph_signature(gates_a, cells)
    sig_b = get_subgraph_signature(gates_b, cells)

    if sig_a != sig_b:
        return False

    # For now, signature match is enough
    # Could add connectivity check for higher precision
    return True


def expand_pattern(
    seed_gates: List[str],
    cells: Dict,
    wire_to_driver: Dict[int, str],
    wire_to_consumers: Dict[int, List[str]],
    max_depth: int = 3
) -> Tuple[Set[str], List[Set[str]]]:
    """
    Expand from seed gates to find maximal common subgraph pattern.

    Args:
        seed_gates: List of gates with matching neighborhoods
        cells: All cells
        wire_to_driver: Wire to driver map
        wire_to_consumers: Wire to consumers map
        max_depth: Maximum expansion depth

    Returns:
        (pattern_gates, list of instance gate sets)
    """
    # Start with depth 1
    pattern = extract_subgraph(seed_gates[0], cells, wire_to_driver, wire_to_consumers, 1)
    instances = [pattern]

    # Check other seeds match at depth 1
    for seed in seed_gates[1:]:
        instance = extract_subgraph(seed, cells, wire_to_driver, wire_to_consumers, 1)
        if subgraphs_isomorphic(pattern, instance, cells, wire_to_driver):
            instances.append(instance)

    # Try to expand
    for depth in range(2, max_depth + 1):
        expanded = extract_subgraph(seed_gates[0], cells, wire_to_driver, wire_to_consumers, depth)

        # Check if all instances still match at this depth
        new_instances = [expanded]
        all_match = True

        for seed in seed_gates[1:]:
            instance = extract_subgraph(seed, cells, wire_to_driver, wire_to_consumers, depth)
            if subgraphs_isomorphic(expanded, instance, cells, wire_to_driver):
                new_instances.append(instance)
            else:
                all_match = False
                break

        if all_match and len(new_instances) >= 2:
            pattern = expanded
            instances = new_instances
        else:
            break

    return pattern, instances


def find_patterns(
    system,
    min_instances: int = 2,
    min_size: int = 2,
    max_depth: int = 3
) -> List[Pattern]:
    """
    Find all repeated patterns in a system.

    Args:
        system: A System from ingest
        min_instances: Minimum occurrences to be a pattern
        min_size: Minimum gates in a pattern
        max_depth: Maximum pattern depth

    Returns:
        List of discovered Patterns, sorted by gates_saved
    """
    wire_to_driver, wire_to_consumers = build_wire_maps(system)
    hashes = hash_all_gates(system)

    # Group by hash
    buckets: Dict[int, List[str]] = defaultdict(list)
    for gate_id, nh in hashes.items():
        buckets[nh.hash_value].append(gate_id)

    # Process each bucket with enough candidates
    patterns = []
    seen_patterns: Set[Tuple[str, ...]] = set()
    pattern_counter = 0

    for hash_val, seed_gates in buckets.items():
        if len(seed_gates) < min_instances:
            continue

        # Expand to find pattern
        pattern_gates, instances = expand_pattern(
            seed_gates,
            system.cells,
            wire_to_driver,
            wire_to_consumers,
            max_depth
        )

        if len(pattern_gates) < min_size:
            continue

        if len(instances) < min_instances:
            continue

        # Check if we've seen this pattern
        sig = get_subgraph_signature(pattern_gates, system.cells)
        if sig in seen_patterns:
            continue
        seen_patterns.add(sig)

        # Analyze boundary
        internal, inputs, outputs = find_boundary_wires(
            pattern_gates, system.cells, wire_to_driver
        )

        pattern = Pattern(
            pattern_id=f"P{pattern_counter:04d}",
            gates=frozenset(pattern_gates),
            gate_types=sig,
            internal_wires=internal,
            input_wires=inputs,
            output_wires=outputs,
            instances=[frozenset(inst) for inst in instances]
        )
        patterns.append(pattern)
        pattern_counter += 1

    # Sort by gates saved (most valuable first)
    patterns.sort(key=lambda p: p.gates_saved, reverse=True)

    return patterns
