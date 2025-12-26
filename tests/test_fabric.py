"""
Tests for TriX Fabric.

The fabric IS the program.
The fabric IS the database.
The fabric IS the knowledge.
"""

import pytest
import tempfile
import os

from trix.fabric import (
    Fabric,
    Cluster,
    FabricProcessor,
    Router,
    Message,
    parse_fdl,
    emit_fdl,
    query_fabric,
)
from trix.fabric.cluster import hamming_distance
from trix.fabric.query import fabric_stats, find_by_shape, trace_path
from trix.fabric.fdl import save_fdl, load_fdl, save_json, load_json


class TestMessage:
    """Tests for Message class."""

    def test_message_creation(self):
        msg = Message(target="/0/0b1010", payload={"a": 0.5, "b": 0.3})
        assert msg.target == "/0/0b1010"
        assert msg.payload == {"a": 0.5, "b": 0.3}
        assert msg.context["depth"] == 0
        assert msg.context["trace"] == []

    def test_message_derive(self):
        msg = Message(target="/0/0b0000", payload={"a": 1.0})
        derived = msg.derive(
            new_target="/0/0b0001",
            new_payload={"a": 0.5},
            source="/0/0b0000"
        )
        assert derived.target == "/0/0b0001"
        assert derived.payload == {"a": 0.5}
        assert derived.context["depth"] == 1
        assert derived.context["trace"] == ["/0/0b0000"]
        assert derived.source == "/0/0b0000"

    def test_message_serialization(self):
        msg = Message(target="/0/0b1010", payload={"x": 1.0})
        d = msg.to_dict()
        restored = Message.from_dict(d)
        assert restored.target == msg.target
        assert restored.payload == msg.payload


class TestHammingDistance:
    """Tests for Hamming distance calculation."""

    def test_same_signature(self):
        assert hamming_distance("0b1010", "0b1010") == 0

    def test_one_bit_diff(self):
        assert hamming_distance("0b1010", "0b1011") == 1

    def test_all_bits_diff(self):
        assert hamming_distance("0b0000", "0b1111") == 4

    def test_without_prefix(self):
        assert hamming_distance("1010", "1011") == 1

    def test_different_lengths(self):
        assert hamming_distance("0b10", "0b0010") == 0  # Padded


class TestCluster:
    """Tests for Cluster class."""

    def test_cluster_creation(self):
        cluster = Cluster(path="/0")
        assert cluster.path == "/0"
        assert cluster.processor_count == 0

    def test_add_processor(self):
        cluster = Cluster(path="/0")
        proc = FabricProcessor(signature="/0/0b00", shape="xor")
        cluster.add_processor(proc)
        assert cluster.processor_count == 1
        assert cluster.get_processor("0b00") == proc

    def test_add_child(self):
        root = Cluster(path="/")
        child = root.add_child("0")
        assert child.path == "/0"
        assert root.get_child("0") == child

    def test_find_nearest_processor(self):
        cluster = Cluster(path="/0")
        cluster.add_processor(FabricProcessor(signature="/0/0b0000", shape="xor"))
        cluster.add_processor(FabricProcessor(signature="/0/0b1111", shape="xor"))

        # Nearest to 0b0001 should be 0b0000 (distance 1)
        nearest = cluster.find_nearest_processor("0b0001")
        assert nearest.local_signature == "0b0000"

        # Nearest to 0b1110 should be 0b1111 (distance 1)
        nearest = cluster.find_nearest_processor("0b1110")
        assert nearest.local_signature == "0b1111"

    def test_iter_all_processors(self):
        root = Cluster(path="/")
        child = root.add_child("0")

        root.add_processor(FabricProcessor(signature="/0b00", shape="xor"))
        child.add_processor(FabricProcessor(signature="/0/0b00", shape="and"))

        all_procs = list(root.iter_all_processors())
        assert len(all_procs) == 2

    def test_serialization(self):
        cluster = Cluster(path="/0", name="test_cluster")
        cluster.add_processor(FabricProcessor(signature="/0/0b00", shape="xor"))

        d = cluster.to_dict()
        restored = Cluster.from_dict(d)
        assert restored.path == cluster.path
        assert restored.name == cluster.name
        assert restored.processor_count == 1


class TestRouter:
    """Tests for Router class."""

    def test_parse_signature(self):
        path, local = Router.parse_signature("/0/1/0b1010")
        assert path == ["/", "0", "1"]
        assert local == "0b1010"

    def test_parse_signature_no_local(self):
        # When path ends in something that looks like binary, it's treated as local sig
        # So /0/1 -> path=["/", "0"], local="1"
        path, local = Router.parse_signature("/0/1")
        assert path == ["/", "0"]
        assert local == "1"

    def test_parse_signature_cluster_only(self):
        # Use letters to ensure it's treated as path, not binary
        path, local = Router.parse_signature("/cluster/sub")
        assert path == ["/", "cluster", "sub"]
        assert local == ""

    def test_route_exact(self):
        root = Cluster(path="/")
        child = root.add_child("0")
        proc = FabricProcessor(signature="/0/0b1010", shape="xor")
        child.add_processor(proc)

        router = Router(root)
        result = router.route("/0/0b1010")

        assert result.exact_match
        assert result.processor == proc

    def test_route_hamming(self):
        root = Cluster(path="/")
        child = root.add_child("0")
        child.add_processor(FabricProcessor(signature="/0/0b0000", shape="xor"))

        router = Router(root)
        result = router.route("/0/0b0001")  # Not exact, but close

        assert not result.exact_match
        assert result.processor.local_signature == "0b0000"


class TestFabricProcessor:
    """Tests for FabricProcessor class."""

    def test_processor_creation(self):
        proc = FabricProcessor(signature="/0/0b1010", shape="xor", name="test_xor")
        assert proc.signature == "/0/0b1010"
        assert proc.shape == "xor"
        assert proc.name == "test_xor"

    def test_evaluate_xor(self):
        proc = FabricProcessor(signature="/0/0b00", shape="xor")
        result = proc.evaluate({"a": 1.0, "b": 0.0})
        assert abs(result["result"] - 1.0) < 0.001

        result = proc.evaluate({"a": 1.0, "b": 1.0})
        assert abs(result["result"] - 0.0) < 0.001

    def test_evaluate_and(self):
        proc = FabricProcessor(signature="/0/0b00", shape="and")
        result = proc.evaluate({"a": 1.0, "b": 1.0})
        assert abs(result["result"] - 1.0) < 0.001

        result = proc.evaluate({"a": 1.0, "b": 0.0})
        assert abs(result["result"] - 0.0) < 0.001

    def test_cluster_path(self):
        proc = FabricProcessor(signature="/0/1/0b1010", shape="xor")
        assert proc.cluster_path == "/0/1"
        assert proc.local_signature == "0b1010"

    def test_connections(self):
        proc = FabricProcessor(signature="/0/0b00", shape="xor")
        proc.connect_to("/0/0b01", "a", "result")
        assert len(proc.connections) == 1
        assert proc.connections[0].target_signature == "/0/0b01"


class TestFabric:
    """Tests for Fabric class."""

    def test_fabric_creation(self):
        fabric = Fabric(name="test_fabric")
        assert fabric.name == "test_fabric"
        assert fabric.processor_count == 0

    def test_add_processor(self):
        fabric = Fabric(name="test")
        proc = FabricProcessor(signature="/0/0b1010", shape="xor")
        fabric.add_processor(proc)
        assert fabric.processor_count == 1
        assert fabric.get_processor("/0/0b1010") == proc

    def test_add_processor_creates_clusters(self):
        fabric = Fabric(name="test")
        fabric.add_processor(FabricProcessor(signature="/a/b/c/0b00", shape="xor"))

        # Should have created intermediate clusters
        cluster = fabric.router.resolve_path("/a/b/c")
        assert cluster is not None
        assert cluster.processor_count == 1

    def test_connect_processors(self):
        fabric = Fabric(name="test")
        fabric.add_processor(FabricProcessor(signature="/0/0b00", shape="xor", name="a"))
        fabric.add_processor(FabricProcessor(signature="/0/0b01", shape="xor", name="b"))

        fabric.connect("/0/0b00", "/0/0b01", "result", "a")

        proc_a = fabric.get_processor("/0/0b00")
        assert len(proc_a.connections) == 1

    def test_execute_single(self):
        fabric = Fabric(name="test")
        fabric.add_processor(FabricProcessor(signature="/0/0b00", shape="xor", name="gate"))
        fabric.set_export("/0/0b00")

        msg = Message(target="/0/0b00", payload={"a": 1.0, "b": 0.0})
        result = fabric.inject(msg)

        assert result.messages_processed == 1
        assert "/0/0b00" in result.outputs
        assert abs(result.outputs["/0/0b00"]["result"] - 1.0) < 0.001

    def test_execute_chain(self):
        """Test a chain of connected processors."""
        fabric = Fabric(name="chain")

        # Create chain: a -> b (XOR -> NOT)
        fabric.add_processor(FabricProcessor(signature="/0/0b00", shape="xor", name="a"))
        fabric.add_processor(FabricProcessor(signature="/0/0b01", shape="not", name="b"))

        fabric.connect("/0/0b00", "/0/0b01", "result", "a")
        fabric.set_export("/0/0b01")

        # XOR(1, 0) = 1, NOT(1) = 0
        msg = Message(target="/0/0b00", payload={"a": 1.0, "b": 0.0})
        result = fabric.inject(msg)

        assert result.messages_processed == 2
        assert abs(result.outputs["/0/0b01"]["result"] - 0.0) < 0.001

    def test_serialization(self):
        fabric = Fabric(name="test")
        fabric.add_processor(FabricProcessor(signature="/0/0b00", shape="xor", name="a"))
        fabric.set_export("/0/0b00")

        d = fabric.to_dict()
        restored = Fabric.from_dict(d)

        assert restored.name == fabric.name
        assert restored.processor_count == fabric.processor_count
        assert "/0/0b00" in restored.exports


class TestFDL:
    """Tests for Fabric Definition Language."""

    def test_parse_simple(self):
        fdl = '''
        fabric "test" {
            processor gate {
                shape: xor
                signature: /0/0b00
            }
            export: gate
        }
        '''
        fabric = parse_fdl(fdl)
        assert fabric.name == "test"
        assert fabric.processor_count == 1
        assert len(fabric.exports) == 1

    def test_parse_with_connections(self):
        fdl = '''
        fabric "chain" {
            processor a {
                shape: xor
                signature: /0/0b00
            }
            processor b {
                shape: not
                signature: /0/0b01
                input.a: a.result
            }
            export: b
        }
        '''
        fabric = parse_fdl(fdl)
        assert fabric.processor_count == 2

        proc_a = fabric.get_processor("/0/0b00")
        assert len(proc_a.connections) == 1

    def test_emit_fdl(self):
        fabric = Fabric(name="test")
        fabric.add_processor(FabricProcessor(signature="/0/0b00", shape="xor", name="gate"))
        fabric.set_export("/0/0b00")

        fdl = emit_fdl(fabric)
        assert 'fabric "test"' in fdl
        assert 'processor gate' in fdl
        assert 'shape: xor' in fdl

    def test_roundtrip(self):
        """Parse -> Emit -> Parse should preserve structure."""
        fdl_original = '''
        fabric "roundtrip" {
            processor a {
                shape: xor
                signature: /0/0b00
            }
            processor b {
                shape: and
                signature: /0/0b01
                input.a: a.result
            }
            export: b
        }
        '''
        fabric1 = parse_fdl(fdl_original)
        fdl_emitted = emit_fdl(fabric1)
        fabric2 = parse_fdl(fdl_emitted)

        assert fabric1.name == fabric2.name
        assert fabric1.processor_count == fabric2.processor_count

    def test_file_io(self):
        """Test saving and loading FDL files."""
        fabric = Fabric(name="file_test")
        fabric.add_processor(FabricProcessor(signature="/0/0b00", shape="xor", name="gate"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.fdl")
            save_fdl(fabric, path)

            loaded = load_fdl(path)
            assert loaded.name == fabric.name
            assert loaded.processor_count == 1

    def test_json_io(self):
        """Test JSON serialization."""
        fabric = Fabric(name="json_test")
        fabric.add_processor(FabricProcessor(signature="/0/0b00", shape="xor", name="gate"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            save_json(fabric, path)

            loaded = load_json(path)
            assert loaded.name == fabric.name


class TestQuery:
    """Tests for fabric queries."""

    @pytest.fixture
    def sample_fabric(self):
        fabric = Fabric(name="query_test")
        fabric.add_processor(FabricProcessor(signature="/0/0b00", shape="xor", name="xor1"))
        fabric.add_processor(FabricProcessor(signature="/0/0b01", shape="xor", name="xor2"))
        fabric.add_processor(FabricProcessor(signature="/0/0b10", shape="and", name="and1"))
        fabric.add_processor(FabricProcessor(signature="/1/0b00", shape="or", name="or1"))
        fabric.connect("/0/0b00", "/0/0b10", "result", "a")
        return fabric

    def test_query_by_shape(self, sample_fabric):
        result = query_fabric(sample_fabric, shape="xor")
        assert result.count == 2

    def test_query_by_cluster(self, sample_fabric):
        result = query_fabric(sample_fabric, cluster="/0")
        assert result.count == 3

    def test_query_by_name(self, sample_fabric):
        result = query_fabric(sample_fabric, name="xor*")
        assert result.count == 2

    def test_find_by_shape(self, sample_fabric):
        procs = find_by_shape(sample_fabric, "and")
        assert len(procs) == 1
        assert procs[0].name == "and1"

    def test_trace_path(self, sample_fabric):
        path = trace_path(sample_fabric, "/0/0b00", "/0/0b10")
        assert path == ["/0/0b00", "/0/0b10"]

    def test_fabric_stats(self, sample_fabric):
        stats = fabric_stats(sample_fabric)
        assert stats["total_processors"] == 4
        assert stats["shapes"]["xor"] == 2
        assert stats["shapes"]["and"] == 1
        assert stats["shapes"]["or"] == 1


class TestIntegration:
    """Integration tests for complete fabric workflows."""

    def test_half_adder(self):
        """Build a half adder from XOR and AND."""
        fdl = '''
        fabric "half_adder" {
            processor xor_gate {
                shape: xor
                signature: /0/0b00
            }
            processor and_gate {
                shape: and
                signature: /0/0b01
            }
            export: xor_gate
            export: and_gate
        }
        '''
        fabric = parse_fdl(fdl)

        # Half adder: sum = a XOR b, carry = a AND b
        # Test 1 + 1 = 10 (binary): sum=0, carry=1
        inputs = {
            "/0/0b00": {"a": 1.0, "b": 1.0},
            "/0/0b01": {"a": 1.0, "b": 1.0}
        }
        result = fabric.execute(inputs)

        xor_out = result.outputs["/0/0b00"]["result"]
        and_out = result.outputs["/0/0b01"]["result"]

        assert abs(xor_out - 0.0) < 0.001  # 1 XOR 1 = 0
        assert abs(and_out - 1.0) < 0.001  # 1 AND 1 = 1

    def test_fuzzy_logic(self):
        """Test with continuous values (fuzzy logic)."""
        fabric = Fabric(name="fuzzy")
        fabric.add_processor(FabricProcessor(signature="/0/0b00", shape="xor", name="gate"))
        fabric.set_export("/0/0b00")

        # XOR with continuous inputs: 0.7 + 0.3 - 2*0.7*0.3 = 0.58
        msg = Message(target="/0/0b00", payload={"a": 0.7, "b": 0.3})
        result = fabric.inject(msg)

        expected = 0.7 + 0.3 - 2 * 0.7 * 0.3  # 0.58
        actual = result.outputs["/0/0b00"]["result"]
        assert abs(actual - expected) < 0.001

    def test_deep_hierarchy(self):
        """Test deeply nested cluster hierarchy."""
        fabric = Fabric(name="deep")

        # Create processors at various depths
        fabric.add_processor(FabricProcessor(signature="/a/b/c/d/0b00", shape="xor"))
        fabric.add_processor(FabricProcessor(signature="/a/b/c/d/0b01", shape="and"))
        fabric.add_processor(FabricProcessor(signature="/a/b/0b00", shape="or"))

        assert fabric.processor_count == 3

        # Routing should work at any depth
        proc = fabric.get_processor("/a/b/c/d/0b00")
        assert proc is not None
        assert proc.shape == "xor"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
