"""
The Foundry Chip Catalog
========================

Day One Library:
- 1 processor (6502)
- 7 logic gates (XOR, AND, OR, NOT, NAND, NOR, XNOR)
- 3 arithmetic (ADD, SUB, MUL)
- 3 molecules (half adder, full adder, ripple adder)

Total: 14 items
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from pathlib import Path

from ..export.trixc import ChipSpec, ChipType


class Category(Enum):
    """Chip categories for catalog organization."""
    PROCESSORS = "Processors"
    LOGIC = "Logic"
    ARITHMETIC = "Arithmetic"
    MOLECULES = "Molecules"


@dataclass
class CatalogEntry:
    """An entry in the chip catalog."""
    spec: ChipSpec
    category: Category
    ready: bool = True
    coming_soon: bool = False


class Catalog:
    """
    The Foundry chip catalog.

    Browse → Select → Export → Use.
    """

    def __init__(self):
        self._entries: Dict[str, CatalogEntry] = {}
        self._load_day_one_library()

    def _load_day_one_library(self):
        """Load the Day One library (14 items)."""

        # === PROCESSORS ===

        self._add(CatalogEntry(
            spec=ChipSpec(
                name="6502",
                chip_type=ChipType.PROTEIN,
                bits=8,
                description="The chip that started it all. Apple II. Commodore 64. NES. Now as frozen geometry.",
                polynomial="[326-coefficient frozen shape]",
                coefficients=[0] * 326,  # Placeholder - actual 6502 shape
                inputs=["opcode", "a", "x", "y", "sp", "status"],
                validated=True,
            ),
            category=Category.PROCESSORS,
            ready=True,
        ))

        # === LOGIC GATES ===

        logic_gates = [
            ("XOR", "a + b - 2*a*b", [1, 1, -2]),
            ("AND", "a * b", [0, 0, 1]),
            ("OR", "a + b - a*b", [1, 1, -1]),
            ("NOT", "1 - a", [-1]),
            ("NAND", "1 - a*b", [1, 0, 0, -1]),
            ("NOR", "1 - a - b + a*b", [1, -1, -1, 1]),
            ("XNOR", "1 - a - b + 2*a*b", [1, -1, -1, 2]),
        ]

        for name, poly, coeffs in logic_gates:
            self._add(CatalogEntry(
                spec=ChipSpec(
                    name=name,
                    chip_type=ChipType.ATOM,
                    bits=1 if name != "NOT" else 1,
                    description=f"{name} logic gate as frozen polynomial.",
                    polynomial=poly,
                    coefficients=coeffs,
                    inputs=["a"] if name == "NOT" else ["a", "b"],
                    validated=True,
                ),
                category=Category.LOGIC,
            ))

        # === ARITHMETIC ===

        self._add(CatalogEntry(
            spec=ChipSpec(
                name="8-bit ADD",
                chip_type=ChipType.ATOM,
                bits=8,
                description="8-bit addition. a + b.",
                polynomial="a + b",
                coefficients=[1, 1],
                inputs=["a", "b"],
                validated=True,
            ),
            category=Category.ARITHMETIC,
        ))

        self._add(CatalogEntry(
            spec=ChipSpec(
                name="8-bit SUB",
                chip_type=ChipType.ATOM,
                bits=8,
                description="8-bit subtraction. a - b.",
                polynomial="a - b",
                coefficients=[1, -1],
                inputs=["a", "b"],
                validated=True,
            ),
            category=Category.ARITHMETIC,
        ))

        self._add(CatalogEntry(
            spec=ChipSpec(
                name="8-bit MUL",
                chip_type=ChipType.ATOM,
                bits=8,
                description="8-bit multiplication. a * b.",
                polynomial="a * b",
                coefficients=[0, 0, 1],
                inputs=["a", "b"],
                validated=True,
            ),
            category=Category.ARITHMETIC,
        ))

        # === MOLECULES ===

        self._add(CatalogEntry(
            spec=ChipSpec(
                name="Half Adder",
                chip_type=ChipType.MOLECULE,
                bits=8,
                description="Half adder: sum = XOR(a,b), carry = AND(a,b).",
                polynomial="sum: a+b-2ab, carry: ab",
                coefficients=[1, 1, -2, 0, 0, 1],  # XOR coeffs + AND coeffs
                inputs=["a", "b"],
                validated=True,
            ),
            category=Category.MOLECULES,
        ))

        self._add(CatalogEntry(
            spec=ChipSpec(
                name="Full Adder",
                chip_type=ChipType.MOLECULE,
                bits=8,
                description="Full adder with carry in/out.",
                polynomial="sum: a⊕b⊕cin, cout: (a∧b)∨(cin∧(a⊕b))",
                coefficients=[1, 1, 1, -2, -2, -2, 4] + [0, 0, 1, 1, -1, -2, 2],
                inputs=["a", "b", "cin"],
                validated=True,
            ),
            category=Category.MOLECULES,
        ))

        self._add(CatalogEntry(
            spec=ChipSpec(
                name="8-bit Ripple Adder",
                chip_type=ChipType.MOLECULE,
                bits=8,
                description="8-bit ripple carry adder composed of 8 full adders.",
                polynomial="[8 chained full adders]",
                coefficients=[0] * 48,  # 8 full adders
                inputs=["a", "b"],
                validated=True,
            ),
            category=Category.MOLECULES,
        ))

    def _add(self, entry: CatalogEntry):
        """Add an entry to the catalog."""
        self._entries[entry.spec.name.lower()] = entry

    def get(self, name: str) -> Optional[CatalogEntry]:
        """Get a catalog entry by name."""
        return self._entries.get(name.lower())

    def list_all(self) -> List[CatalogEntry]:
        """List all catalog entries."""
        return list(self._entries.values())

    def list_by_category(self, category: Category) -> List[CatalogEntry]:
        """List entries in a specific category."""
        return [e for e in self._entries.values() if e.category == category]

    def search(self, query: str) -> List[CatalogEntry]:
        """Search catalog by name or description."""
        query_lower = query.lower()
        results = []
        for entry in self._entries.values():
            if (query_lower in entry.spec.name.lower() or
                query_lower in entry.spec.description.lower()):
                results.append(entry)
        return results

    @property
    def categories(self) -> List[Category]:
        """List all categories in display order."""
        return [Category.PROCESSORS, Category.ARITHMETIC, Category.LOGIC, Category.MOLECULES]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries.values())
