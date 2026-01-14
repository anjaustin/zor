"""
Neighborhood Hasher - Hash local gate neighborhoods for pattern discovery.

The hasher computes a canonical hash for each gate's local neighborhood,
enabling fast detection of repeated patterns through hash collision.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict


@dataclass
class NeighborhoodHash:
    """A hash of a gate's local neighborhood."""
    gate_id: str
    gate_type: str
    hash_value: int
    depth: int
    fanin_signature: Tuple[str, ...]
    fanout_signature: Tuple[str, ...]


def hash_neighborhood(
    gate_id: str,
    cells: Dict,
    wire_to_driver: Dict[int, str],
    wire_to_consumers: Dict[int, List[str]],
    depth: int = 1
) -> NeighborhoodHash:
    """
    Compute a hash for a gate's local neighborhood.

    Args:
        gate_id: The gate to hash
        cells: Dict of all cells (from System.cells)
        wire_to_driver: Map from wire bit to driving cell
        wire_to_consumers: Map from wire bit to consuming cells
        depth: How many levels of fanin/fanout to include

    Returns:
        NeighborhoodHash with computed hash value
    """
    cell = cells[gate_id]

    # Get immediate fanin cell types
    fanin_types = []
    for port in cell.input_ports:
        for bit in cell.connections.get(port, []):
            if isinstance(bit, int) and bit in wire_to_driver:
                driver_id = wire_to_driver[bit]
                if driver_id in cells:
                    fanin_types.append(cells[driver_id].cell_type)

    # Get immediate fanout cell types
    fanout_types = []
    out_bits = cell.connections.get(cell.output_port, [])
    for bit in out_bits:
        if isinstance(bit, int) and bit in wire_to_consumers:
            for consumer_id in wire_to_consumers[bit]:
                if consumer_id in cells and consumer_id != gate_id:
                    fanout_types.append(cells[consumer_id].cell_type)

    # Sort for canonical form
    fanin_sig = tuple(sorted(fanin_types))
    fanout_sig = tuple(sorted(fanout_types))

    # Compute hash
    hash_val = hash((cell.cell_type, fanin_sig, fanout_sig))

    return NeighborhoodHash(
        gate_id=gate_id,
        gate_type=cell.cell_type,
        hash_value=hash_val,
        depth=depth,
        fanin_signature=fanin_sig,
        fanout_signature=fanout_sig,
    )


def build_wire_maps(system) -> Tuple[Dict[int, str], Dict[int, List[str]]]:
    """
    Build maps from wires to their drivers and consumers.

    Args:
        system: A System from ingest

    Returns:
        (wire_to_driver, wire_to_consumers)
    """
    wire_to_driver: Dict[int, str] = {}
    wire_to_consumers: Dict[int, List[str]] = defaultdict(list)

    # Mark primary inputs as having no driver (external)
    for wire in system.inputs.values():
        for bit in wire.bits:
            if isinstance(bit, int):
                wire_to_driver[bit] = "__INPUT__"

    # Process all cells
    for cell_name, cell in system.cells.items():
        # This cell drives its output wires
        out_bits = cell.connections.get(cell.output_port, [])
        for bit in out_bits:
            if isinstance(bit, int):
                wire_to_driver[bit] = cell_name

        # This cell consumes its input wires
        for port in cell.input_ports:
            for bit in cell.connections.get(port, []):
                if isinstance(bit, int):
                    wire_to_consumers[bit].append(cell_name)

    return wire_to_driver, wire_to_consumers


def hash_all_gates(system, depth: int = 1) -> Dict[str, NeighborhoodHash]:
    """
    Hash all gates in a system.

    Args:
        system: A System from ingest
        depth: Neighborhood depth for hashing

    Returns:
        Dict mapping gate_id to its NeighborhoodHash
    """
    wire_to_driver, wire_to_consumers = build_wire_maps(system)

    hashes = {}
    for gate_id in system.cells:
        hashes[gate_id] = hash_neighborhood(
            gate_id,
            system.cells,
            wire_to_driver,
            wire_to_consumers,
            depth
        )

    return hashes


def find_hash_buckets(hashes: Dict[str, NeighborhoodHash]) -> Dict[int, List[str]]:
    """
    Group gates by their hash values.

    Args:
        hashes: Dict of gate hashes from hash_all_gates

    Returns:
        Dict mapping hash_value to list of gate_ids with that hash
    """
    buckets: Dict[int, List[str]] = defaultdict(list)

    for gate_id, nh in hashes.items():
        buckets[nh.hash_value].append(gate_id)

    return buckets


def find_candidate_patterns(
    system,
    min_instances: int = 2,
    depth: int = 1
) -> Dict[int, List[str]]:
    """
    Find candidate patterns by hash collision.

    Args:
        system: A System from ingest
        min_instances: Minimum occurrences to be a candidate
        depth: Neighborhood depth for hashing

    Returns:
        Dict mapping hash_value to gate_ids (only buckets with >= min_instances)
    """
    hashes = hash_all_gates(system, depth)
    buckets = find_hash_buckets(hashes)

    # Filter to buckets with enough instances
    candidates = {
        h: gates for h, gates in buckets.items()
        if len(gates) >= min_instances
    }

    return candidates
