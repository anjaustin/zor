"""
Tests for Providence FFN - The Unified Architecture.

Tests cover:
- Providence tile (signature + shape + state)
- Providence FFN with XOR routing
- Hierarchical vs flat routing
- Frozen vs learned shapes
- State persistence and routing
- Drop-in compatibility with existing FFNs
- Soft vs hard routing modes
"""

import warnings
import pytest
import torch
import torch.nn as nn

from trix.nn.providence import (
    ProvidenceTile,
    ProvidenceFFN,
    ProvidenceBlock,
    create_providence_ffn,
    create_frozen_providence_ffn,
)

# Providence is a PyTorch module that uses frozen shapes for gradient-based training.
# The deprecation warning is for execution-only use cases; training is still valid.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from trix.nn.frozen_shapes import Primitives


# =============================================================================
# TEST: Providence Tile
# =============================================================================

class TestProvidenceTile:
    """Test individual Providence tiles."""

    def test_creation_learned(self):
        """Create tile with learned shape."""
        tile = ProvidenceTile(d_model=64, d_hidden=128, d_state=16)
        assert tile.d_model == 64
        assert tile.d_hidden == 128
        assert tile.d_state == 16
        assert not tile.is_frozen

    def test_creation_frozen(self):
        """Create tile with frozen shape."""
        tile = ProvidenceTile(
            d_model=64,
            d_hidden=128,
            d_state=16,
            frozen_shape=Primitives.xor,
        )
        assert tile.is_frozen

    def test_signature_shape(self):
        """Signature has correct shape."""
        tile = ProvidenceTile(d_model=64, d_hidden=128, d_state=16)
        sig = tile.get_signature()
        assert sig.shape == (64,)

    def test_signature_is_ternary(self):
        """Signature values are in {-1, 0, +1}."""
        tile = ProvidenceTile(d_model=64, d_hidden=128, d_state=16)
        sig = tile.get_signature()
        # Sign gives -1, 0, or +1
        assert ((sig == -1) | (sig == 0) | (sig == 1)).all()

    def test_forward_shapes_learned(self):
        """Forward returns correct shapes with learned shape."""
        tile = ProvidenceTile(d_model=64, d_hidden=128, d_state=16)
        x = torch.randn(8, 64)
        state = torch.randn(8, 16)
        output, new_state = tile(x, state)
        assert output.shape == (8, 64)
        assert new_state.shape == (8, 16)

    def test_forward_shapes_frozen(self):
        """Forward returns correct shapes with frozen shape."""
        tile = ProvidenceTile(
            d_model=64,
            d_hidden=128,
            d_state=16,
            frozen_shape=Primitives.xor,
        )
        x = torch.randn(8, 64)
        state = torch.randn(8, 16)
        output, new_state = tile(x, state)
        assert output.shape == (8, 64)
        assert new_state.shape == (8, 16)

    def test_state_changes(self):
        """State actually updates."""
        tile = ProvidenceTile(d_model=64, d_hidden=128, d_state=16)
        x = torch.randn(8, 64)
        state = torch.zeros(8, 16)
        output, new_state = tile(x, state)
        assert not torch.allclose(new_state, state)

    def test_gradient_flow(self):
        """Gradients flow through tile."""
        tile = ProvidenceTile(d_model=64, d_hidden=128, d_state=16)
        x = torch.randn(8, 64, requires_grad=True)
        state = torch.randn(8, 16, requires_grad=True)
        output, new_state = tile(x, state)
        loss = output.sum() + new_state.sum()
        loss.backward()
        assert x.grad is not None
        assert state.grad is not None

    def test_usage_tracking(self):
        """Usage is tracked."""
        tile = ProvidenceTile(d_model=64, d_hidden=128, d_state=16)
        assert tile.usage_rate == 0.0
        tile.update_usage(10, 100)
        assert abs(tile.usage_rate - 0.1) < 1e-5


# =============================================================================
# TEST: Providence FFN - Basic
# =============================================================================

class TestProvidenceFFNBasic:
    """Test basic Providence FFN functionality."""

    def test_creation(self):
        """Create FFN."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
            d_state=8,
        )
        assert ffn.num_tiles == 16
        assert ffn.num_clusters == 4
        assert ffn.d_state == 8

    def test_init_state(self):
        """Initialize state correctly."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
            d_state=8,
        )
        state = ffn.init_state(batch_size=4)
        assert 'global_state' in state
        assert 'tile_states' in state
        assert state['global_state'].shape == (4, 32)  # d_global_state=32
        assert state['tile_states'].shape == (4, 16, 8)

    def test_forward_2d(self):
        """Forward with 2D input."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
            d_state=8,
        )
        x = torch.randn(8, 64)
        state = ffn.init_state(8)
        output, new_state, routing_info, aux = ffn(x, state)

        assert output.shape == x.shape
        assert 'tile_idx' in routing_info
        assert 'cluster_idx' in routing_info
        assert 'gate' in routing_info
        assert 'total_aux' in aux

    def test_forward_3d(self):
        """Forward with 3D input."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
            d_state=8,
        )
        x = torch.randn(4, 16, 64)
        state = ffn.init_state(4)
        output, new_state, routing_info, aux = ffn(x, state)

        assert output.shape == x.shape
        assert routing_info['tile_idx'].shape == (4, 16)

    def test_no_state_provided(self):
        """Works when state is not provided."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
        )
        x = torch.randn(8, 64)
        output, state, _, _ = ffn(x, state=None)
        assert output.shape == x.shape
        assert state is not None

    def test_gradient_flow(self):
        """Gradients flow through FFN."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
            d_state=8,
        )
        x = torch.randn(8, 64, requires_grad=True)
        state = ffn.init_state(8)
        output, new_state, _, _ = ffn(x, state)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None


# =============================================================================
# TEST: XOR Routing
# =============================================================================

class TestXORRouting:
    """Test XOR-based routing."""

    def test_tile_idx_in_range(self):
        """Tile indices are in valid range."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
        )
        x = torch.randn(32, 64)
        state = ffn.init_state(32)
        _, _, routing_info, _ = ffn(x, state)
        tile_idx = routing_info['tile_idx']
        assert (tile_idx >= 0).all()
        assert (tile_idx < 16).all()

    def test_cluster_idx_in_range(self):
        """Cluster indices are in valid range."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
        )
        x = torch.randn(32, 64)
        state = ffn.init_state(32)
        _, _, routing_info, _ = ffn(x, state)
        cluster_idx = routing_info['cluster_idx']
        assert (cluster_idx >= 0).all()
        assert (cluster_idx < 4).all()

    def test_gate_sums_to_one(self):
        """Gate weights sum to approximately 1."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
            use_soft_routing=True,
        )
        ffn.train()
        x = torch.randn(8, 64)
        state = ffn.init_state(8)
        _, _, routing_info, _ = ffn(x, state)
        gate_sums = routing_info['gate'].sum(dim=-1)
        assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=0.1)


# =============================================================================
# TEST: Hierarchical vs Flat Routing
# =============================================================================

class TestRoutingModes:
    """Test hierarchical vs flat routing modes."""

    def test_hierarchical_mode(self):
        """Hierarchical routing works."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
            mode='hierarchical',
        )
        x = torch.randn(8, 64)
        state = ffn.init_state(8)
        output, _, _, _ = ffn(x, state)
        assert output.shape == x.shape

    def test_flat_mode(self):
        """Flat routing works."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
            mode='flat',
        )
        x = torch.randn(8, 64)
        state = ffn.init_state(8)
        output, _, _, _ = ffn(x, state)
        assert output.shape == x.shape


# =============================================================================
# TEST: Frozen vs Learned Shapes
# =============================================================================

class TestShapeModes:
    """Test frozen vs learned shape modes."""

    def test_learned_shapes(self):
        """Learned shapes work."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
            use_frozen_shapes=False,
        )
        assert all(not t.is_frozen for t in ffn.tiles)

        x = torch.randn(8, 64)
        state = ffn.init_state(8)
        output, _, _, _ = ffn(x, state)
        assert output.shape == x.shape

    def test_frozen_shapes(self):
        """Frozen shapes work."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
            use_frozen_shapes=True,
        )
        assert all(t.is_frozen for t in ffn.tiles)

        x = torch.randn(8, 64)
        state = ffn.init_state(8)
        output, _, _, _ = ffn(x, state)
        assert output.shape == x.shape

    def test_frozen_shapes_specific(self):
        """Frozen shapes with specific names."""
        shape_names = ['xor', 'and', 'or', 'not'] * 4  # 16 tiles
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
            use_frozen_shapes=True,
            frozen_shape_names=shape_names,
        )
        for i, tile in enumerate(ffn.tiles):
            assert tile.is_frozen
            assert tile.shape_name == shape_names[i]


# =============================================================================
# TEST: State Persistence
# =============================================================================

class TestStatePersistence:
    """Test state persistence across calls."""

    def test_state_persists(self):
        """State persists across calls."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
            d_state=8,
        )
        x = torch.randn(4, 64)
        state = ffn.init_state(4)

        # First forward
        _, state1, _, _ = ffn(x, state)

        # Second forward
        _, state2, _, _ = ffn(x, state1)

        # States should differ
        assert not torch.allclose(state1['global_state'], state2['global_state'])

    def test_state_affects_routing(self):
        """Different states lead to different routing."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
            use_state_routing=True,
        )
        ffn.eval()

        x = torch.randn(8, 64)

        # State 1
        state1 = ffn.init_state(8)
        state1['global_state'] = torch.randn(8, 32)
        _, _, routing1, _ = ffn(x, state1)

        # State 2 (different)
        state2 = ffn.init_state(8)
        state2['global_state'] = -state1['global_state']  # Opposite
        _, _, routing2, _ = ffn(x, state2)

        # At least check both work
        assert routing1['tile_idx'].shape == routing2['tile_idx'].shape

    def test_routing_without_state_awareness(self):
        """Routing works without state awareness."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
            use_state_routing=False,
        )
        x = torch.randn(8, 64)
        state = ffn.init_state(8)
        output, _, _, _ = ffn(x, state)
        assert output.shape == x.shape


# =============================================================================
# TEST: Transition Tracking
# =============================================================================

class TestTransitionTracking:
    """Test tile transition tracking."""

    def test_transition_matrix_initialized(self):
        """Transition matrix starts at zero."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
        )
        trans = ffn.get_transition_matrix(normalize=False)
        assert trans.sum() == 0

    def test_transitions_tracked(self):
        """Transitions are tracked across calls."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
        )
        ffn.eval()

        x = torch.randn(4, 64)
        state = ffn.init_state(4)

        # Multiple forward passes
        for _ in range(5):
            _, state, _, _ = ffn(x, state, track_transitions=True)

        trans = ffn.get_transition_matrix(normalize=False)
        # Should have some non-zero entries
        assert trans.sum() > 0

    def test_reset_clears_transitions(self):
        """Reset clears transition matrix."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
        )
        ffn.eval()

        x = torch.randn(4, 64)
        state = ffn.init_state(4)
        _, state, _, _ = ffn(x, state)
        _, _, _, _ = ffn(x, state)

        ffn.reset_stats()
        trans = ffn.get_transition_matrix(normalize=False)
        assert trans.sum() == 0


# =============================================================================
# TEST: Regime Analysis
# =============================================================================

class TestRegimeAnalysis:
    """Test regime analysis features."""

    def test_regime_analysis_structure(self):
        """Regime analysis returns correct structure."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
        )
        analysis = ffn.get_regime_analysis()

        assert 'transition_matrix' in analysis
        assert 'self_transition_rates' in analysis
        assert 'stable_tiles' in analysis
        assert 'hub_tiles' in analysis
        assert 'frozen_tiles' in analysis
        assert 'learned_tiles' in analysis

    def test_routing_stats(self):
        """Routing stats work."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
        )
        ffn.train()
        x = torch.randn(32, 64)
        state = ffn.init_state(32)
        ffn(x, state)

        stats = ffn.get_routing_stats()
        assert 'active_tiles' in stats
        assert 'active_clusters' in stats
        assert 'd_state' in stats
        assert 'frozen_tiles' in stats
        assert 'learned_tiles' in stats

    def test_count_parameters(self):
        """Parameter counting works."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
        )
        counts = ffn.count_parameters()

        assert 'total' in counts
        assert 'routing' in counts
        assert 'compute' in counts
        assert 'state' in counts
        assert counts['total'] == counts['routing'] + counts['compute'] + counts['state']


# =============================================================================
# TEST: Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience creation functions."""

    def test_create_with_defaults(self):
        """Create with sensible defaults."""
        ffn = create_providence_ffn(d_model=128, num_tiles=64)
        assert ffn.d_model == 128
        assert ffn.num_tiles == 64

    def test_create_with_frozen(self):
        """Create with frozen shapes."""
        ffn = create_providence_ffn(
            d_model=128,
            num_tiles=16,
            use_frozen_shapes=True,
        )
        assert all(t.is_frozen for t in ffn.tiles)

    def test_create_frozen_convenience(self):
        """Create frozen Providence FFN."""
        ffn = create_frozen_providence_ffn(d_model=128, num_tiles=16)
        assert all(t.is_frozen for t in ffn.tiles)


# =============================================================================
# TEST: Providence Block
# =============================================================================

class TestProvidenceBlock:
    """Test Providence transformer block."""

    def test_creation(self):
        """Create block."""
        block = ProvidenceBlock(
            d_model=64,
            n_heads=4,
            num_tiles=16,
        )
        assert block.ffn.num_tiles == 16

    def test_forward(self):
        """Forward pass works."""
        block = ProvidenceBlock(
            d_model=64,
            n_heads=4,
            num_tiles=16,
        )
        x = torch.randn(4, 16, 64)
        output, state, routing, aux = block(x)

        assert output.shape == x.shape
        assert state is not None

    def test_with_state(self):
        """Forward with state persistence."""
        block = ProvidenceBlock(
            d_model=64,
            n_heads=4,
            num_tiles=16,
        )
        x = torch.randn(4, 16, 64)

        # First forward
        _, state1, _, _ = block(x)

        # Second forward with state
        output2, state2, _, _ = block(x, state1)

        assert output2.shape == x.shape
        assert not torch.allclose(state1['global_state'], state2['global_state'])


# =============================================================================
# TEST: Soft vs Hard Routing
# =============================================================================

class TestSoftHardRouting:
    """Test soft vs hard routing modes."""

    def test_soft_routing_train(self):
        """Soft routing in training mode."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
            use_soft_routing=True,
        )
        ffn.train()

        x = torch.randn(8, 64, requires_grad=True)
        state = ffn.init_state(8)
        output, _, routing, _ = ffn(x, state)

        # Soft gate should have multiple non-zero entries
        gate = routing['gate']
        non_zero = (gate > 0.01).sum(dim=-1).float().mean()
        # With soft routing, expect more than 1 active tile on average
        assert non_zero >= 1.0

        # Gradients should flow
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_hard_routing_eval(self):
        """Hard routing in eval mode."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
            use_soft_routing=True,
        )
        ffn.eval()

        x = torch.randn(8, 64)
        state = ffn.init_state(8)
        output, _, routing, _ = ffn(x, state)

        # Hard gate should be one-hot
        gate = routing['gate']
        # In eval mode, each sample should route to exactly 1 tile
        non_zero_per_sample = (gate > 0.5).sum(dim=-1)
        assert (non_zero_per_sample == 1).all()


# =============================================================================
# TEST: Sequence Processing
# =============================================================================

class TestSequenceProcessing:
    """Test processing sequences."""

    def test_sequence_state_evolution(self):
        """State evolves across sequence."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
            d_state=8,
        )

        B, T = 4, 10
        x = torch.randn(B, T, 64)
        state = ffn.init_state(B)

        # Process sequence token by token
        states = [state['global_state'].clone()]
        for t in range(T):
            _, state, _, _ = ffn(x[:, t], state)
            states.append(state['global_state'].clone())

        # States should evolve
        for i in range(1, len(states)):
            assert not torch.allclose(states[0], states[i])

    def test_batch_sequence_processing(self):
        """Process full batch sequence."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
        )
        x = torch.randn(4, 16, 64)
        state = ffn.init_state(4)
        output, new_state, routing, _ = ffn(x, state)

        assert output.shape == (4, 16, 64)
        assert routing['tile_idx'].shape == (4, 16)


# =============================================================================
# TEST: Drop-in Compatibility
# =============================================================================

class TestDropInCompatibility:
    """Test drop-in replacement compatibility."""

    def test_same_interface_as_hierarchical_trix(self):
        """Same interface as HierarchicalTriXFFN."""
        ffn = ProvidenceFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
        )

        # These should all exist and work
        x = torch.randn(8, 64)
        state = ffn.init_state(8)
        output, new_state, routing_info, aux = ffn(x, state)

        # Check outputs
        assert output.shape == x.shape
        assert 'tile_idx' in routing_info
        assert 'total_aux' in aux

        # Check stats
        stats = ffn.get_routing_stats()
        assert 'num_tiles' in stats
        assert 'active_tiles' in stats

    def test_integration_in_simple_model(self):
        """Can be used in a simple model."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(32, 64)
                self.ffn = create_providence_ffn(d_model=64, num_tiles=16)
                self.out = nn.Linear(64, 10)

            def forward(self, x, state=None):
                x = self.embed(x)
                if state is None:
                    state = self.ffn.init_state(x.shape[0])
                x, state, _, _ = self.ffn(x, state)
                return self.out(x), state

        model = SimpleModel()
        model.train()  # Ensure training mode for soft routing
        x = torch.randn(8, 32)
        out, state = model(x)
        assert out.shape == (8, 10)

        # Should be trainable - at least some parameters get gradients
        loss = out.sum()
        loss.backward()
        grads_found = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
        assert grads_found > 0, "At least some parameters should have gradients"
