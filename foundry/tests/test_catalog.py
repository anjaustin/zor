"""
Tests for The Foundry Chip Catalog
===================================

The catalog contains the Day One library of pre-trained chips.
"""

import pytest

from foundry.library.catalog import Catalog, Category, CatalogEntry
from foundry.export.trixc import ChipType


class TestCatalog:
    """Tests for the chip catalog."""

    @pytest.fixture
    def catalog(self):
        """Create a fresh catalog instance."""
        return Catalog()

    def test_catalog_loads(self, catalog):
        """Catalog should load without errors."""
        assert catalog is not None

    def test_day_one_library_size(self, catalog):
        """Day One library should have 14 chips."""
        assert len(catalog) == 14

    def test_catalog_is_iterable(self, catalog):
        """Catalog should be iterable."""
        entries = list(catalog)
        assert len(entries) == 14
        assert all(isinstance(e, CatalogEntry) for e in entries)

    def test_get_by_name(self, catalog):
        """Should be able to get chip by name."""
        xor = catalog.get("xor")
        assert xor is not None
        assert xor.spec.name == "XOR"

    def test_get_by_name_case_insensitive(self, catalog):
        """Name lookup should be case insensitive."""
        assert catalog.get("XOR") is not None
        assert catalog.get("xor") is not None
        assert catalog.get("Xor") is not None

    def test_get_nonexistent_returns_none(self, catalog):
        """Getting nonexistent chip should return None."""
        assert catalog.get("nonexistent") is None

    def test_list_all(self, catalog):
        """list_all should return all entries."""
        all_entries = catalog.list_all()
        assert len(all_entries) == 14


class TestCatalogCategories:
    """Tests for catalog categories."""

    @pytest.fixture
    def catalog(self):
        return Catalog()

    def test_categories_property(self, catalog):
        """Categories should be in display order."""
        categories = catalog.categories
        assert Category.PROCESSORS in categories
        assert Category.ARITHMETIC in categories
        assert Category.LOGIC in categories
        assert Category.MOLECULES in categories

    def test_list_by_category_processors(self, catalog):
        """Should have 1 processor (6502)."""
        processors = catalog.list_by_category(Category.PROCESSORS)
        assert len(processors) == 1
        assert processors[0].spec.name == "6502"

    def test_list_by_category_logic(self, catalog):
        """Should have 7 logic gates."""
        logic = catalog.list_by_category(Category.LOGIC)
        assert len(logic) == 7

        names = {e.spec.name for e in logic}
        assert "XOR" in names
        assert "AND" in names
        assert "OR" in names
        assert "NOT" in names
        assert "NAND" in names
        assert "NOR" in names
        assert "XNOR" in names

    def test_list_by_category_arithmetic(self, catalog):
        """Should have 3 arithmetic chips."""
        arithmetic = catalog.list_by_category(Category.ARITHMETIC)
        assert len(arithmetic) == 3

        names = {e.spec.name for e in arithmetic}
        assert "8-bit ADD" in names
        assert "8-bit SUB" in names
        assert "8-bit MUL" in names

    def test_list_by_category_molecules(self, catalog):
        """Should have 3 molecule chips."""
        molecules = catalog.list_by_category(Category.MOLECULES)
        assert len(molecules) == 3

        names = {e.spec.name for e in molecules}
        assert "Half Adder" in names
        assert "Full Adder" in names
        assert "8-bit Ripple Adder" in names


class TestCatalogSearch:
    """Tests for catalog search functionality."""

    @pytest.fixture
    def catalog(self):
        return Catalog()

    def test_search_by_name(self, catalog):
        """Search should find chips by name."""
        results = catalog.search("xor")
        assert len(results) >= 1
        assert any(e.spec.name == "XOR" for e in results)

    def test_search_by_partial_name(self, catalog):
        """Search should find chips by partial name."""
        results = catalog.search("add")
        assert len(results) >= 2  # 8-bit ADD, Half Adder, Full Adder, Ripple Adder

    def test_search_by_description(self, catalog):
        """Search should find chips by description."""
        results = catalog.search("Apple")
        assert len(results) >= 1
        assert any(e.spec.name == "6502" for e in results)

    def test_search_case_insensitive(self, catalog):
        """Search should be case insensitive."""
        results_upper = catalog.search("XOR")
        results_lower = catalog.search("xor")
        assert len(results_upper) == len(results_lower)

    def test_search_no_results(self, catalog):
        """Search with no matches should return empty list."""
        results = catalog.search("nonexistent_chip_xyz")
        assert results == []


class TestCatalogChipSpecs:
    """Tests for chip specifications in catalog."""

    @pytest.fixture
    def catalog(self):
        return Catalog()

    def test_6502_spec(self, catalog):
        """6502 should have correct spec."""
        entry = catalog.get("6502")
        assert entry is not None
        assert entry.spec.chip_type == ChipType.PROTEIN
        assert entry.spec.bits == 8
        assert entry.ready is True

    def test_xor_spec(self, catalog):
        """XOR should have correct spec."""
        entry = catalog.get("xor")
        assert entry is not None
        assert entry.spec.chip_type == ChipType.ATOM
        assert entry.spec.bits == 1
        assert entry.spec.polynomial == "a + b - 2*a*b"
        assert entry.spec.coefficients == [1, 1, -2]

    def test_and_spec(self, catalog):
        """AND should have correct spec."""
        entry = catalog.get("and")
        assert entry is not None
        assert entry.spec.polynomial == "a * b"
        assert entry.spec.coefficients == [0, 0, 1]

    def test_or_spec(self, catalog):
        """OR should have correct spec."""
        entry = catalog.get("or")
        assert entry is not None
        assert entry.spec.polynomial == "a + b - a*b"

    def test_not_spec(self, catalog):
        """NOT should have correct spec."""
        entry = catalog.get("not")
        assert entry is not None
        assert entry.spec.polynomial == "1 - a"
        assert entry.spec.inputs == ["a"]

    def test_all_chips_validated(self, catalog):
        """All Day One chips should be validated."""
        for entry in catalog:
            assert entry.spec.validated is True

    def test_all_chips_ready(self, catalog):
        """All Day One chips should be ready."""
        for entry in catalog:
            assert entry.ready is True

    def test_chip_types_correct(self, catalog):
        """Chip types should be appropriate for each category."""
        for entry in catalog.list_by_category(Category.LOGIC):
            assert entry.spec.chip_type == ChipType.ATOM

        for entry in catalog.list_by_category(Category.ARITHMETIC):
            assert entry.spec.chip_type == ChipType.ATOM

        for entry in catalog.list_by_category(Category.MOLECULES):
            assert entry.spec.chip_type == ChipType.MOLECULE

        for entry in catalog.list_by_category(Category.PROCESSORS):
            assert entry.spec.chip_type == ChipType.PROTEIN
