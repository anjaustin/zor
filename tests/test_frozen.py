"""
Tests for Frozen Shapes Module

Unit tests for the frozen shapes infrastructure:
- FrozenShape and signature derivation
- FrozenTile interface compatibility
- FrozenShapeRegistry
- FrozenTriXFFN routing

All tests should pass with 100% accuracy on frozen computations.
"""

import pytest
import torch
import torch.nn as nn

from trix.nn.frozen import (
    FrozenShape,
    FrozenTile,
    FrozenShapeRegistry,
    FrozenTriXFFN,
    derive_signature,
    derive_signature_from_fn,
    get_default_registry,
    create_frozen_ffn_from_names,
    verify_frozen_tile,
)

from trix.compiler.atoms_fp4 import (
    make_AND,
    make_OR,
    make_XOR,
    make_NOT,
    make_SUM,
    make_CARRY,
)


class TestDeriveSignature:
    """Test signature derivation from circuits."""

    def test_signature_dimension(self):
        """Signatures have correct dimension."""
        circuit = make_XOR()
        sig = derive_signature(circuit, sig_dim=64)
        assert sig.shape == (64,)

    def test_signature_is_ternary(self):
        """Signatures are ternary (-1 or +1)."""
        circuit = make_AND()
        sig = derive_signature(circuit, sig_dim=64)
        assert torch.all((sig == -1) | (sig == 1))

    def test_signature_deterministic(self):
        """Same circuit produces same signature."""
        circuit = make_OR()
        sig1 = derive_signature(circuit, sig_dim=64)
        sig2 = derive_signature(circuit, sig_dim=64)
        assert torch.allclose(sig1, sig2)

    def test_different_circuits_different_signatures(self):
        """Different circuits produce different signatures."""
        xor = make_XOR()
        and_c = make_AND()

        sig_xor = derive_signature(xor, sig_dim=64)
        sig_and = derive_signature(and_c, sig_dim=64)

        # Signatures should be different (not identical)
        assert not torch.allclose(sig_xor, sig_and)

    def test_signature_from_fn(self):
        """Signature derivation from Python function."""
        def my_xor(a, b):
            return a ^ b

        sig = derive_signature_from_fn(my_xor, num_inputs=2, sig_dim=64)
        assert sig.shape == (64,)
        assert torch.all((sig == -1) | (sig == 1))


class TestFrozenShape:
    """Test FrozenShape dataclass."""

    def test_forward_pass(self):
        """FrozenShape forward computes correctly."""
        circuit = make_XOR()
        sig = derive_signature(circuit)

        shape = FrozenShape(
            name="XOR",
            circuit=circuit,
            signature=sig,
            num_inputs=2,
            num_outputs=1,
        )

        # Test XOR truth table
        x = torch.tensor([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], dtype=torch.float32)

        y = shape(x)

        expected = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
        assert torch.allclose(y, expected)

    def test_verify(self):
        """Shapes verify as 100% accurate."""
        circuit = make_AND()
        sig = derive_signature(circuit)

        shape = FrozenShape(
            name="AND",
            circuit=circuit,
            signature=sig,
            num_inputs=2,
            num_outputs=1,
        )

        passed, accuracy = shape.verify()
        assert passed
        assert accuracy == 1.0


class TestFrozenShapeRegistry:
    """Test FrozenShapeRegistry."""

    def test_builtin_shapes_registered(self):
        """Registry has builtin FP4 atoms."""
        registry = FrozenShapeRegistry()

        expected = ["AND", "OR", "XOR", "NOT", "NAND", "NOR", "XNOR", "SUM", "CARRY", "MUX"]
        for name in expected:
            assert name in registry

    def test_get_shape(self):
        """Can retrieve shapes by name."""
        registry = FrozenShapeRegistry()
        xor = registry.get("XOR")

        assert isinstance(xor, FrozenShape)
        assert xor.name == "XOR"
        assert xor.num_inputs == 2
        assert xor.num_outputs == 1

    def test_register_custom_shape(self):
        """Can register custom shapes."""
        registry = FrozenShapeRegistry(auto_register_builtins=False)

        circuit = make_XOR()
        shape = registry.register("MY_XOR", circuit, "My custom XOR")

        assert "MY_XOR" in registry
        assert shape.description == "My custom XOR"

    def test_list_shapes(self):
        """list_shapes returns all registered names."""
        registry = FrozenShapeRegistry()
        shapes = registry.list_shapes()

        assert len(shapes) >= 10
        assert "XOR" in shapes
        assert "AND" in shapes

    def test_default_registry(self):
        """Global default registry works."""
        reg1 = get_default_registry()
        reg2 = get_default_registry()

        assert reg1 is reg2  # Same instance
        assert "XOR" in reg1


class TestFrozenTile:
    """Test FrozenTile nn.Module."""

    def test_tile_is_module(self):
        """FrozenTile is a proper nn.Module."""
        registry = get_default_registry()
        shape = registry.get("XOR")
        tile = FrozenTile(shape, tile_id=0)

        assert isinstance(tile, nn.Module)

    def test_no_learned_params(self):
        """FrozenTile has zero learned parameters."""
        registry = get_default_registry()
        shape = registry.get("AND")
        tile = FrozenTile(shape)

        # Count learnable parameters
        param_count = sum(p.numel() for p in tile.parameters())
        assert param_count == 0

    def test_get_signature(self):
        """get_signature returns correct signature."""
        registry = get_default_registry()
        shape = registry.get("OR")
        tile = FrozenTile(shape)

        sig = tile.get_signature()
        assert sig.shape == (64,)
        assert torch.allclose(sig, shape.signature)

    def test_forward(self):
        """Forward computes correctly."""
        registry = get_default_registry()
        shape = registry.get("XOR")
        tile = FrozenTile(shape)

        x = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
        y = tile(x)

        assert y.shape == (2, 1)
        assert y[0, 0] == 1.0
        assert y[1, 0] == 1.0

    def test_usage_tracking(self):
        """Usage tracking works."""
        registry = get_default_registry()
        shape = registry.get("NOT")
        tile = FrozenTile(shape)

        assert tile.usage_rate == 0.0

        tile.update_usage(5, 10)
        assert tile.usage_rate == 0.5

        tile.update_usage(5, 10)
        assert tile.usage_rate == 0.5

    def test_device_compatibility(self):
        """Tile works on CPU (and GPU if available)."""
        registry = get_default_registry()
        shape = registry.get("AND")
        tile = FrozenTile(shape)

        x = torch.tensor([[1, 1]], dtype=torch.float32)
        y = tile(x)
        assert y.device.type == "cpu"

        if torch.cuda.is_available():
            tile_gpu = tile.cuda()
            x_gpu = x.cuda()
            y_gpu = tile_gpu(x_gpu)
            assert y_gpu.device.type == "cuda"

    def test_verify_frozen_tile(self):
        """verify_frozen_tile utility works."""
        registry = get_default_registry()
        shape = registry.get("SUM")
        tile = FrozenTile(shape)

        passed, accuracy = verify_frozen_tile(tile)
        assert passed
        assert accuracy == 1.0


class TestFrozenTriXFFN:
    """Test FrozenTriXFFN routing layer."""

    def test_creation_from_tiles(self):
        """Can create FFN from tiles."""
        registry = get_default_registry()
        tiles = [
            FrozenTile(registry.get("AND"), tile_id=0),
            FrozenTile(registry.get("OR"), tile_id=1),
            FrozenTile(registry.get("XOR"), tile_id=2),
        ]

        ffn = FrozenTriXFFN(tiles)
        assert ffn.num_tiles == 3

    def test_creation_from_names(self):
        """Can create FFN from shape names."""
        ffn = create_frozen_ffn_from_names(["AND", "OR", "XOR"])
        assert ffn.num_tiles == 3

    def test_no_learned_params_without_meaning(self):
        """Without meaning layer, no learned params."""
        ffn = create_frozen_ffn_from_names(["AND", "OR"])
        assert ffn.param_count() == 0

    def test_meaning_layer_params(self):
        """With meaning layer, has learnable routing."""
        ffn = create_frozen_ffn_from_names(
            ["AND", "OR", "XOR"],
            use_meaning_layer=True,
            num_opcodes=10,
        )

        # 10 opcodes * 3 tiles = 30 params
        assert ffn.param_count() == 30

    def test_content_routing(self):
        """Content-based routing works."""
        ffn = create_frozen_ffn_from_names(["AND", "OR", "XOR"])

        x = torch.tensor([[0, 1]], dtype=torch.float32)
        out = ffn(x)

        assert 'result' in out
        assert 'tile_ids' in out
        assert out['result'].shape == (1, 1)

    def test_opcode_routing(self):
        """Opcode-based routing works."""
        ffn = create_frozen_ffn_from_names(
            ["AND", "OR", "XOR"],
            use_meaning_layer=True,
            num_opcodes=3,
        )

        x = torch.tensor([[1, 1]], dtype=torch.float32)
        opcode = torch.tensor([0])

        ffn.eval()  # Use hard routing
        out = ffn(x, opcode_id=opcode)

        assert 'result' in out
        assert 'tile_ids' in out

    def test_init_meaning_from_spec(self):
        """Can initialize meaning layer from specification."""
        ffn = create_frozen_ffn_from_names(
            ["AND", "OR", "XOR"],
            use_meaning_layer=True,
            num_opcodes=3,
        )

        # Map opcodes to specific tiles
        spec = {0: 0, 1: 1, 2: 2}  # opcode 0->AND, 1->OR, 2->XOR
        ffn.init_meaning_from_spec(spec)

        # Verify routing
        ffn.eval()
        for opcode_id in range(3):
            x = torch.tensor([[1, 1]], dtype=torch.float32)
            opcode = torch.tensor([opcode_id])
            out = ffn(x, opcode_id=opcode)
            assert out['tile_ids'][0].item() == opcode_id

    def test_routing_stats(self):
        """Routing statistics work."""
        ffn = create_frozen_ffn_from_names(["AND", "OR", "XOR"])
        ffn.eval()

        # Run some inputs
        for _ in range(10):
            x = torch.randint(0, 2, (1, 2), dtype=torch.float32)
            ffn(x)

        stats = ffn.get_routing_stats()
        assert "tile_0_usage" in stats
        assert "tile_1_usage" in stats
        assert "tile_2_usage" in stats


class TestFrozenTileCompatibility:
    """Test FrozenTile compatibility with TriXTile interface."""

    def test_has_get_signature(self):
        """FrozenTile has get_signature method."""
        registry = get_default_registry()
        tile = FrozenTile(registry.get("XOR"))

        assert hasattr(tile, 'get_signature')
        assert callable(tile.get_signature)

    def test_has_forward(self):
        """FrozenTile has forward method."""
        registry = get_default_registry()
        tile = FrozenTile(registry.get("XOR"))

        assert hasattr(tile, 'forward')
        assert callable(tile.forward)

    def test_has_usage_tracking(self):
        """FrozenTile has usage tracking attributes."""
        registry = get_default_registry()
        tile = FrozenTile(registry.get("XOR"))

        assert hasattr(tile, 'activation_count')
        assert hasattr(tile, 'total_count')
        assert hasattr(tile, 'update_usage')
        assert hasattr(tile, 'usage_rate')


class TestFrozenShapeAccuracy:
    """Test that frozen shapes achieve 100% accuracy."""

    @pytest.mark.parametrize("shape_name", ["AND", "OR", "XOR", "NOT", "NAND", "NOR"])
    def test_builtin_shape_100_percent(self, shape_name):
        """All builtin shapes achieve 100% accuracy."""
        registry = get_default_registry()
        shape = registry.get(shape_name)
        tile = FrozenTile(shape)

        passed, accuracy = verify_frozen_tile(tile)
        assert passed, f"{shape_name} did not pass verification"
        assert accuracy == 1.0, f"{shape_name} accuracy was {accuracy}"

    def test_sum_100_percent(self):
        """SUM (3-input XOR) achieves 100%."""
        registry = get_default_registry()
        tile = FrozenTile(registry.get("SUM"))

        # Test all 8 inputs
        x = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ], dtype=torch.float32)

        y = tile(x)

        # SUM = a XOR b XOR c
        expected = torch.tensor([
            [0], [1], [1], [0],
            [1], [0], [0], [1],
        ], dtype=torch.float32)

        assert torch.allclose(y, expected)

    def test_carry_100_percent(self):
        """CARRY (majority) achieves 100%."""
        registry = get_default_registry()
        tile = FrozenTile(registry.get("CARRY"))

        # Test all 8 inputs
        x = torch.tensor([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
            [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
        ], dtype=torch.float32)

        y = tile(x)

        # CARRY = majority(a, b, c)
        expected = torch.tensor([
            [0], [0], [0], [1],
            [0], [1], [1], [1],
        ], dtype=torch.float32)

        assert torch.allclose(y, expected)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_tiles_raises(self):
        """Creating FFN with no tiles raises error."""
        with pytest.raises(ValueError):
            FrozenTriXFFN([])

    def test_unknown_shape_raises(self):
        """Getting unknown shape raises KeyError."""
        registry = get_default_registry()
        with pytest.raises(KeyError):
            registry.get("NONEXISTENT_SHAPE")

    def test_meaning_layer_without_flag(self):
        """init_meaning_from_spec without meaning layer raises."""
        ffn = create_frozen_ffn_from_names(["AND", "OR"])
        with pytest.raises(ValueError):
            ffn.init_meaning_from_spec({0: 0})

    def test_batched_input(self):
        """Handles batched inputs correctly."""
        ffn = create_frozen_ffn_from_names(["XOR"])
        ffn.eval()

        x = torch.randint(0, 2, (32, 2), dtype=torch.float32)
        out = ffn(x)

        assert out['result'].shape == (32, 1)
        assert out['tile_ids'].shape == (32,)
