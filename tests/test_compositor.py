"""
Tests for the TriX Compositor.

Tests pattern discovery, composition, and known pattern matching.
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestHasher:
    """Tests for neighborhood hashing."""

    def test_hash_all_gates(self):
        """Test that all gates get hashed."""
        from trix.forge.ingest import ingest_yosys_json
        from trix.compositor.hasher import hash_all_gates

        system = ingest_yosys_json("examples/ingest/full_adder.json")
        hashes = hash_all_gates(system)

        assert len(hashes) == len(system.cells)
        for gate_id, nh in hashes.items():
            assert nh.gate_id == gate_id
            assert nh.gate_type in system.cells[gate_id].cell_type

    def test_find_candidate_patterns(self):
        """Test finding candidate patterns by hash collision."""
        from trix.forge.ingest import ingest_yosys_json
        from trix.compositor.hasher import find_candidate_patterns

        system = ingest_yosys_json("examples/ingest/adder8.json")
        candidates = find_candidate_patterns(system, min_instances=2)

        # 8-bit adder should have repeated patterns
        assert len(candidates) > 0

        # All candidates should have at least 2 instances
        for h, gates in candidates.items():
            assert len(gates) >= 2


class TestMatcher:
    """Tests for pattern matching."""

    def test_find_patterns_full_adder(self):
        """Test pattern finding on full adder."""
        from trix.forge.ingest import ingest_yosys_json
        from trix.compositor.matcher import find_patterns

        system = ingest_yosys_json("examples/ingest/full_adder.json")
        patterns = find_patterns(system, min_instances=1, min_size=1)

        # Full adder is small, may not have repeated patterns
        # But we should be able to find structure
        assert isinstance(patterns, list)

    def test_find_patterns_8bit_adder(self):
        """Test pattern finding on 8-bit adder."""
        from trix.forge.ingest import ingest_yosys_json
        from trix.compositor.matcher import find_patterns

        system = ingest_yosys_json("examples/ingest/adder8.json")
        patterns = find_patterns(system, min_instances=2, min_size=2)

        # 8-bit adder should have repeated full adder patterns
        assert len(patterns) > 0

        # Check pattern properties
        for p in patterns:
            assert p.instance_count >= 2
            assert p.size >= 2
            assert p.gates_saved >= 0


class TestComposer:
    """Tests for shape composition."""

    def test_compose_full_adder(self):
        """Test composing full adder."""
        from trix.forge.ingest import ingest_yosys_json
        from trix.compositor.composer import compose

        system = ingest_yosys_json("examples/ingest/full_adder.json")
        tree = compose(system)

        assert tree.name == system.name
        assert tree.total_gates == len(system.cells)
        assert tree.total_gates > 0

    def test_compose_8bit_adder(self):
        """Test composing 8-bit adder."""
        from trix.forge.ingest import ingest_yosys_json
        from trix.compositor.composer import compose

        system = ingest_yosys_json("examples/ingest/adder8.json")
        tree = compose(system)

        assert tree.name == system.name
        assert tree.total_gates == len(system.cells)

        # Should find some patterns
        assert tree.unique_shapes > 0 or len(tree.atomic_gates) == tree.total_gates

        # Compression should be non-negative
        assert tree.compression_ratio >= 0

    def test_compose_summary(self):
        """Test that summary generation works."""
        from trix.forge.ingest import ingest_yosys_json
        from trix.compositor.composer import compose

        system = ingest_yosys_json("examples/ingest/adder8.json")
        tree = compose(system)

        summary = tree.summary()
        assert "ShapeTree" in summary
        assert str(tree.total_gates) in summary


class TestVisualizer:
    """Tests for visualization."""

    def test_visualize(self):
        """Test ASCII visualization."""
        from trix.forge.ingest import ingest_yosys_json
        from trix.compositor.composer import compose
        from trix.compositor.visualizer import visualize

        system = ingest_yosys_json("examples/ingest/adder8.json")
        tree = compose(system)

        output = visualize(tree)
        assert "SHAPE TREE" in output
        assert "STATISTICS" in output

    def test_visualize_stats(self):
        """Test compact stats visualization."""
        from trix.forge.ingest import ingest_yosys_json
        from trix.compositor.composer import compose
        from trix.compositor.visualizer import visualize_stats

        system = ingest_yosys_json("examples/ingest/adder8.json")
        tree = compose(system)

        stats = visualize_stats(tree)
        assert tree.name in stats
        assert "gates" in stats

    def test_to_dot(self):
        """Test DOT format generation."""
        from trix.forge.ingest import ingest_yosys_json
        from trix.compositor.composer import compose
        from trix.compositor.visualizer import to_dot

        system = ingest_yosys_json("examples/ingest/adder8.json")
        tree = compose(system)

        dot = to_dot(tree)
        assert "digraph" in dot
        assert tree.name in dot


class TestLibrary:
    """Tests for known pattern library."""

    def test_match_half_adder(self):
        """Test matching half adder pattern."""
        from trix.compositor.library import match_known_pattern

        # Half adder: XOR + AND
        sig = ("$_AND_", "$_XOR_")
        result = match_known_pattern(sig)
        assert result == "HalfAdder"

    def test_match_full_adder(self):
        """Test matching full adder pattern."""
        from trix.compositor.library import match_known_pattern

        # Full adder: 2 XOR + 2 AND + OR
        sig = ("$_AND_", "$_AND_", "$_OR_", "$_XOR_", "$_XOR_")
        result = match_known_pattern(sig)
        assert result == "FullAdder"

    def test_no_match(self):
        """Test that unknown patterns return None."""
        from trix.compositor.library import match_known_pattern

        sig = ("$_WEIRD_", "$_UNKNOWN_")
        result = match_known_pattern(sig)
        assert result is None


class TestIntegration:
    """Integration tests for the full compositor pipeline."""

    def test_full_pipeline(self):
        """Test the complete pipeline from ingest to visualization."""
        from trix.forge.ingest import ingest_yosys_json
        from trix.compositor import compose, visualize

        # Load system
        system = ingest_yosys_json("examples/ingest/adder8.json")

        # Compose
        tree = compose(system)

        # Visualize
        output = visualize(tree)

        # Verify structure
        assert tree.total_gates == len(system.cells)
        assert "SHAPE TREE" in output

        print("\n" + output)  # Show output for manual inspection


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
