"""
Known Pattern Library - Matches discovered patterns to known circuit structures.

This library contains signatures of common circuit patterns like adders,
multiplexers, registers, etc. When a discovered pattern matches a known
signature, it can be labeled with a meaningful name.
"""

from typing import Dict, List, Tuple, Optional


# Known pattern signatures: gate_types tuple -> pattern name
# Gate types are sorted, so order doesn't matter
KNOWN_PATTERNS: Dict[Tuple[str, ...], str] = {
    # Half Adder: XOR + AND
    ("$_AND_", "$_XOR_"): "HalfAdder",
    ("$_ANDNOT_", "$_XOR_"): "HalfAdder",  # Yosys variant

    # Full Adder: Many variants depending on synthesis optimization
    ("$_AND_", "$_AND_", "$_OR_", "$_XOR_", "$_XOR_"): "FullAdder",
    ("$_AND_", "$_AND_", "$_XOR_", "$_XOR_", "$_XOR_"): "FullAdder",
    ("$_ANDNOT_", "$_XNOR_", "$_XNOR_"): "FullAdder",  # Yosys optimized 3-gate
    ("$_ANDNOT_", "$_XOR_", "$_XNOR_"): "FullAdder",   # Yosys variant
    ("$_ANDNOT_", "$_XNOR_", "$_XOR_"): "FullAdder",   # Yosys variant
    ("$_NOR_", "$_NOR_", "$_XNOR_", "$_XOR_"): "FullAdder",  # NOR-based
    ("$_NOR_", "$_NOR_", "$_XOR_", "$_XNOR_"): "FullAdder",  # NOR-based variant
    ("$_ANDNOT_", "$_NAND_"): "CarryLogic",  # Part of adder carry chain
    ("$_AND_", "$_OR_"): "CarryLogic",  # Carry generate/propagate

    # Mux2: NOT + 2 AND + OR (standard mux)
    ("$_AND_", "$_AND_", "$_NOT_", "$_OR_"): "Mux2",
    ("$_AND_", "$_AND_", "$_OR_"): "Mux2",  # If select already has complement

    # NAND-based gates (after optimization)
    ("$_NAND_", "$_NAND_"): "ANDviaNAND",
    ("$_NAND_", "$_NAND_", "$_NAND_"): "ORviaNAND",

    # NOR-based gates
    ("$_NOR_", "$_NOR_"): "ORviaNOR",
    ("$_NOR_", "$_NOR_", "$_NOR_"): "ANDviaNOR",

    # XOR tree (parity)
    ("$_XOR_", "$_XOR_"): "XOR3",
    ("$_XOR_", "$_XOR_", "$_XOR_"): "XOR4",
    ("$_XOR_", "$_XOR_", "$_XOR_", "$_XOR_"): "Parity5",

    # AND tree
    ("$_AND_", "$_AND_"): "AND3",
    ("$_AND_", "$_AND_", "$_AND_"): "AND4",

    # OR tree
    ("$_OR_", "$_OR_"): "OR3",
    ("$_OR_", "$_OR_", "$_OR_"): "OR4",

    # Comparator bits (XNOR for equality)
    ("$_XNOR_",): "EqualBit",
    ("$_AND_", "$_XNOR_", "$_XNOR_"): "Equal2Bit",

    # Inverter chain
    ("$_NOT_", "$_NOT_"): "Buffer2",

    # SR Latch (if we see it)
    ("$_NOR_", "$_NOR_"): "SRLatch",

    # ANDNOT / ORNOT patterns
    ("$_ANDNOT_", "$_ANDNOT_"): "MaskedAND",
    ("$_ORNOT_", "$_ORNOT_"): "MaskedOR",

    # Decoder patterns
    ("$_AND_", "$_AND_", "$_NOT_"): "Decode2to1",
    ("$_AND_", "$_AND_", "$_AND_", "$_AND_", "$_NOT_", "$_NOT_"): "Decode2to4",
}


# Larger composed patterns (for hierarchical recognition)
COMPOSED_PATTERNS: Dict[str, List[str]] = {
    "RippleCarryAdder4": ["FullAdder", "FullAdder", "FullAdder", "FullAdder"],
    "RippleCarryAdder8": ["FullAdder"] * 8,
    "RippleCarryAdder16": ["FullAdder"] * 16,
}


def match_known_pattern(gate_types: Tuple[str, ...]) -> Optional[str]:
    """
    Try to match a pattern signature to a known pattern.

    Args:
        gate_types: Sorted tuple of gate types in the pattern

    Returns:
        Pattern name if matched, None otherwise
    """
    return KNOWN_PATTERNS.get(gate_types)


def match_composed_pattern(child_patterns: List[str]) -> Optional[str]:
    """
    Try to match a sequence of patterns to a larger composed pattern.

    Args:
        child_patterns: List of pattern names

    Returns:
        Composed pattern name if matched, None otherwise
    """
    sorted_children = sorted(child_patterns)
    for name, expected in COMPOSED_PATTERNS.items():
        if sorted_children == sorted(expected):
            return name
    return None


def pattern_description(name: str) -> str:
    """Get a human-readable description of a pattern."""
    descriptions = {
        "HalfAdder": "Half adder (1-bit addition without carry-in)",
        "FullAdder": "Full adder (1-bit addition with carry)",
        "Mux2": "2-to-1 multiplexer",
        "XOR3": "3-input XOR (parity of 3 bits)",
        "XOR4": "4-input XOR (parity of 4 bits)",
        "AND3": "3-input AND gate",
        "AND4": "4-input AND gate",
        "OR3": "3-input OR gate",
        "OR4": "4-input OR gate",
        "EqualBit": "Single-bit equality comparator",
        "Equal2Bit": "2-bit equality comparator",
        "Buffer2": "Double-inverter buffer",
        "SRLatch": "SR latch (cross-coupled NORs)",
        "Decode2to1": "2-to-1 decoder",
        "Decode2to4": "2-to-4 decoder",
        "RippleCarryAdder4": "4-bit ripple-carry adder",
        "RippleCarryAdder8": "8-bit ripple-carry adder",
        "RippleCarryAdder16": "16-bit ripple-carry adder",
    }
    return descriptions.get(name, f"Pattern: {name}")


def get_all_known_patterns() -> List[str]:
    """Get list of all known pattern names."""
    return list(set(KNOWN_PATTERNS.values())) + list(COMPOSED_PATTERNS.keys())
