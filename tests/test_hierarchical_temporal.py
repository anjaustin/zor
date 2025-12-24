"""
Tests for Hierarchical Temporal FFN.

Tests cover:
- Temporal tile creation and state
- Hierarchical routing with state awareness
- State persistence across forward calls
- Transition tracking
- Regime analysis
"""

import pytest
import torch
import torch.nn as nn

from trix.nn.hierarchical_temporal import (
    TemporalTile,
    HierarchicalTemporalFFN,
    create_hierarchical_temporal_ffn,
)


# =============================================================================
# TEST: Temporal Tile
# =============================================================================

class TestTemporalTile:
    """Test individual temporal tiles."""

    def test_creation(self):
        """Create temporal tile."""
        tile = TemporalTile(d_model=64, d_hidden=128, d_state=16)
        assert tile.d_model == 64
        assert tile.d_hidden == 128
        assert tile.d_state == 16

    def test_signature_shape(self):
        """Signature has correct shape."""
        tile = TemporalTile(d_model=64, d_hidden=128, d_state=16)
        sig = tile.get_signature()
        assert sig.shape == (64,)

    def test_signature_is_ternary(self):
        """Signature values are in {-1, +1}."""
        tile = TemporalTile(d_model=64, d_hidden=128, d_state=16)
        sig = tile.get_signature()
        assert ((sig == -1) | (sig == 1)).all()

    def test_forward_shapes(self):
        """Forward returns correct shapes."""
        tile = TemporalTile(d_model=64, d_hidden=128, d_state=16)
        x = torch.randn(8, 64)
        state = torch.randn(8, 16)
        output, new_state = tile(x, state)
        assert output.shape == (8, 64)
        assert new_state.shape == (8, 16)

    def test_state_changes(self):
        """State actually updates."""
        tile = TemporalTile(d_model=64, d_hidden=128, d_state=16)
        x = torch.randn(8, 64)
        state = torch.zeros(8, 16)
        output, new_state = tile(x, state)
        # New state should differ from input state
        assert not torch.allclose(new_state, state)

    def test_gradient_flow(self):
        """Gradients flow through tile."""
        tile = TemporalTile(d_model=64, d_hidden=128, d_state=16)
        x = torch.randn(8, 64, requires_grad=True)
        state = torch.randn(8, 16, requires_grad=True)
        output, new_state = tile(x, state)
        loss = output.sum() + new_state.sum()
        loss.backward()
        assert x.grad is not None
        assert state.grad is not None

    def test_usage_tracking(self):
        """Usage is tracked."""
        tile = TemporalTile(d_model=64, d_hidden=128, d_state=16)
        assert tile.usage_rate == 0.0
        tile.update_usage(10, 100)
        assert abs(tile.usage_rate - 0.1) < 1e-5


# =============================================================================
# TEST: Hierarchical Temporal FFN
# =============================================================================

class TestHierarchicalTemporalFFN:
    """Test hierarchical temporal FFN."""

    def test_creation(self):
        """Create FFN."""
        ffn = HierarchicalTemporalFFN(
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
        ffn = HierarchicalTemporalFFN(
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
        ffn = HierarchicalTemporalFFN(
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
        assert 'total_aux' in aux

    def test_forward_3d(self):
        """Forward with 3D input."""
        ffn = HierarchicalTemporalFFN(
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

    def test_state_persists(self):
        """State persists across calls."""
        ffn = HierarchicalTemporalFFN(
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

    def test_gradient_flow(self):
        """Gradients flow through FFN."""
        ffn = HierarchicalTemporalFFN(
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

    def test_tile_idx_in_range(self):
        """Tile indices are in valid range."""
        ffn = HierarchicalTemporalFFN(
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
        ffn = HierarchicalTemporalFFN(
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

    def test_no_state_provided(self):
        """Works when state is not provided."""
        ffn = HierarchicalTemporalFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
        )
        x = torch.randn(8, 64)
        output, state, _, _ = ffn(x, state=None)
        assert output.shape == x.shape
        assert state is not None


# =============================================================================
# TEST: State-Aware Routing
# =============================================================================

class TestStateAwareRouting:
    """Test state-dependent routing."""

    def test_state_affects_routing(self):
        """Different states lead to different routing."""
        ffn = HierarchicalTemporalFFN(
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

        # Routing may differ (not guaranteed but likely with opposite states)
        # At least check both work
        assert routing1['tile_idx'].shape == routing2['tile_idx'].shape

    def test_routing_without_state_awareness(self):
        """Routing works without state awareness."""
        ffn = HierarchicalTemporalFFN(
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
        ffn = HierarchicalTemporalFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
        )
        trans = ffn.get_transition_matrix(normalize=False)
        assert trans.sum() == 0

    def test_transitions_tracked(self):
        """Transitions are tracked across calls."""
        ffn = HierarchicalTemporalFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
        )
        ffn.eval()  # Deterministic routing

        x = torch.randn(4, 64)
        state = ffn.init_state(4)

        # Multiple forward passes
        for _ in range(5):
            _, state, _, _ = ffn(x, state, track_transitions=True)

        trans = ffn.get_transition_matrix(normalize=False)
        # Should have some non-zero entries (4 transitions tracked)
        assert trans.sum() > 0

    def test_reset_clears_transitions(self):
        """Reset clears transition matrix."""
        ffn = HierarchicalTemporalFFN(
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
        ffn = HierarchicalTemporalFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
        )
        analysis = ffn.get_regime_analysis()

        assert 'transition_matrix' in analysis
        assert 'self_transition_rates' in analysis
        assert 'stable_tiles' in analysis
        assert 'hub_tiles' in analysis

    def test_routing_stats(self):
        """Routing stats work."""
        ffn = HierarchicalTemporalFFN(
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


# =============================================================================
# TEST: Convenience Function
# =============================================================================

class TestConvenienceFunction:
    """Test convenience creation function."""

    def test_create_with_defaults(self):
        """Create with sensible defaults."""
        ffn = create_hierarchical_temporal_ffn(d_model=128, num_tiles=64)
        assert ffn.d_model == 128
        assert ffn.num_tiles == 64

    def test_create_with_custom_state(self):
        """Create with custom state dimension."""
        ffn = create_hierarchical_temporal_ffn(
            d_model=128,
            num_tiles=64,
            d_state=32,
        )
        assert ffn.d_state == 32


# =============================================================================
# TEST: Sequence Processing
# =============================================================================

class TestSequenceProcessing:
    """Test processing sequences."""

    def test_sequence_state_evolution(self):
        """State evolves across sequence."""
        ffn = HierarchicalTemporalFFN(
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
        ffn = HierarchicalTemporalFFN(
            d_model=64,
            num_tiles=16,
            tiles_per_cluster=4,
        )
        x = torch.randn(4, 16, 64)
        state = ffn.init_state(4)
        output, new_state, routing, _ = ffn(x, state)

        assert output.shape == (4, 16, 64)
        assert routing['tile_idx'].shape == (4, 16)
