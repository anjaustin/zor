"""
Tests for XOR Routing FFN - Content-addressable routing via Hamming distance.

Tests cover:
- Core XOR operations (binarize, ternarize, hamming)
- XOR router
- XORRoutingFFN
- HierarchicalXORRoutingFFN
- Gradient flow
- Providence connection
"""

import pytest
import torch
import torch.nn as nn

from trix.nn.xor_ffn import (
    binarize_ste,
    ternarize_ste,
    hamming_distance,
    soft_hamming_distance,
    XORRouter,
    XORRoutingFFN,
    HierarchicalXORRoutingFFN,
)


# =============================================================================
# TEST: Core Operations
# =============================================================================

class TestBinarize:
    """Test binarization with STE."""

    def test_positive_values(self):
        """Positive values become +1."""
        x = torch.tensor([0.5, 1.0, 2.0])
        result = binarize_ste(x)
        assert (result == 1.0).all()

    def test_negative_values(self):
        """Negative values become -1."""
        x = torch.tensor([-0.5, -1.0, -2.0])
        result = binarize_ste(x)
        assert (result == -1.0).all()

    def test_zero_becomes_positive(self):
        """Zero becomes +1."""
        x = torch.tensor([0.0])
        result = binarize_ste(x)
        assert result.item() == 1.0

    def test_gradient_flows(self):
        """Gradient flows through STE."""
        x = torch.tensor([0.5, -0.5], requires_grad=True)
        y = binarize_ste(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert (x.grad == 1.0).all()  # STE: gradient = 1


class TestTernarize:
    """Test ternarization with STE."""

    def test_high_values(self):
        """Values > threshold become +1."""
        x = torch.tensor([0.5, 1.0, 2.0])
        result = ternarize_ste(x, threshold=0.3)
        assert (result == 1.0).all()

    def test_low_values(self):
        """Values < -threshold become -1."""
        x = torch.tensor([-0.5, -1.0, -2.0])
        result = ternarize_ste(x, threshold=0.3)
        assert (result == -1.0).all()

    def test_middle_values(self):
        """Values in [-threshold, threshold] become 0."""
        x = torch.tensor([-0.2, 0.0, 0.2])
        result = ternarize_ste(x, threshold=0.3)
        assert (result == 0.0).all()

    def test_gradient_flows(self):
        """Gradient flows through STE."""
        x = torch.tensor([0.5, -0.5, 0.1], requires_grad=True)
        y = ternarize_ste(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None


class TestHammingDistance:
    """Test Hamming distance computation."""

    def test_identical(self):
        """Distance to self is 0."""
        a = torch.tensor([[1., 0., -1., 1.]])
        b = torch.tensor([[1., 0., -1., 1.]])
        dist = hamming_distance(a, b)
        assert dist.item() == 0

    def test_all_different(self):
        """All positions different gives d."""
        a = torch.tensor([[1., 1., 1., 1.]])
        b = torch.tensor([[-1., -1., -1., -1.]])
        dist = hamming_distance(a, b)
        assert dist.item() == 4

    def test_half_different(self):
        """Half different gives d/2."""
        a = torch.tensor([[1., 1., -1., -1.]])
        b = torch.tensor([[1., 1., 1., 1.]])
        dist = hamming_distance(a, b)
        assert dist.item() == 2

    def test_batch_to_multiple(self):
        """Batch of queries against multiple signatures."""
        a = torch.tensor([
            [1., 1., 1., 1.],
            [-1., -1., -1., -1.],
        ])
        b = torch.tensor([
            [1., 1., 1., 1.],
            [-1., -1., -1., -1.],
            [0., 0., 0., 0.],
        ])
        dist = hamming_distance(a, b)
        assert dist.shape == (2, 3)
        assert dist[0, 0] == 0  # First query matches first sig
        assert dist[1, 1] == 0  # Second query matches second sig
        assert dist[0, 1] == 4  # First query vs second sig

    def test_symmetric(self):
        """Distance is symmetric."""
        a = torch.tensor([[1., 0., -1., 1.]])
        b = torch.tensor([[0., 1., 1., -1.]])
        assert hamming_distance(a, b) == hamming_distance(b, a)


class TestSoftHammingDistance:
    """Test soft (differentiable) Hamming distance."""

    def test_identical_is_zero(self):
        """Identical vectors have ~0 distance."""
        a = torch.tensor([[1., 0., -1., 1.]])
        b = torch.tensor([[1., 0., -1., 1.]])
        dist = soft_hamming_distance(a, b)
        assert dist.item() < 0.01

    def test_opposite_is_large(self):
        """Opposite vectors have large distance."""
        a = torch.tensor([[1., 1., 1., 1.]])
        b = torch.tensor([[-1., -1., -1., -1.]])
        dist = soft_hamming_distance(a, b)
        assert dist.item() > 1.5  # Close to 2

    def test_gradient_flows(self):
        """Soft distance allows gradient flow."""
        a = torch.randn(4, 8, requires_grad=True)
        b = torch.randn(3, 8, requires_grad=True)
        dist = soft_hamming_distance(a, b)
        loss = dist.sum()
        loss.backward()
        assert a.grad is not None
        assert b.grad is not None


# =============================================================================
# TEST: XOR Router
# =============================================================================

class TestXORRouter:
    """Test XOR-based routing."""

    def test_creation(self):
        """Create router."""
        router = XORRouter(d_model=64, num_tiles=8)
        assert router.d_model == 64
        assert router.num_tiles == 8

    def test_forward_shape(self):
        """Forward returns correct shapes."""
        router = XORRouter(d_model=64, num_tiles=8)
        x = torch.randn(16, 64)
        tile_idx, gate = router(x)
        assert tile_idx.shape == (16,)
        assert gate.shape == (16, 8)

    def test_hard_routing(self):
        """Hard routing gives one-hot gates."""
        router = XORRouter(d_model=64, num_tiles=8, use_soft_training=False)
        router.eval()
        x = torch.randn(16, 64)
        tile_idx, gate = router(x)
        # Each row should sum to 1 (one-hot)
        assert torch.allclose(gate.sum(dim=1), torch.ones(16))

    def test_soft_routing_sums_to_one(self):
        """Soft routing gives weights that sum to 1."""
        router = XORRouter(d_model=64, num_tiles=8, use_soft_training=True)
        router.train()
        x = torch.randn(16, 64)
        tile_idx, gate = router(x)
        assert torch.allclose(gate.sum(dim=1), torch.ones(16), atol=1e-5)

    def test_get_ternary_signatures(self):
        """Signatures are ternary."""
        router = XORRouter(d_model=64, num_tiles=8)
        sigs = router.get_ternary_signatures()
        assert sigs.shape == (8, 64)
        # All values should be in {-1, 0, 1}
        valid = (sigs == -1) | (sigs == 0) | (sigs == 1)
        assert valid.all()


# =============================================================================
# TEST: XOR Routing FFN
# =============================================================================

class TestXORRoutingFFN:
    """Test XOR Routing FFN."""

    def test_creation(self):
        """Create FFN."""
        ffn = XORRoutingFFN(d_model=64, num_tiles=8)
        assert ffn.d_model == 64
        assert ffn.num_tiles == 8

    def test_forward_2d(self):
        """Forward with 2D input."""
        ffn = XORRoutingFFN(d_model=64, num_tiles=8)
        x = torch.randn(16, 64)
        output, routing_info, aux = ffn(x)
        assert output.shape == x.shape
        assert 'tile_idx' in routing_info
        assert 'gate' in routing_info
        assert 'total_aux' in aux

    def test_forward_3d(self):
        """Forward with 3D input."""
        ffn = XORRoutingFFN(d_model=64, num_tiles=8)
        x = torch.randn(4, 16, 64)
        output, routing_info, aux = ffn(x)
        assert output.shape == x.shape
        assert routing_info['tile_idx'].shape == (4, 16)

    def test_gradient_flow(self):
        """Gradients flow through XOR routing."""
        ffn = XORRoutingFFN(d_model=64, num_tiles=8)
        x = torch.randn(16, 64, requires_grad=True)
        output, _, _ = ffn(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_residual_connection(self):
        """Residual connection works."""
        ffn = XORRoutingFFN(d_model=64, num_tiles=8, use_residual=True)
        x = torch.randn(16, 64)
        output, _, _ = ffn(x)
        # Output should not be zero (residual adds input)
        assert output.abs().mean() > 0.1

    def test_no_residual(self):
        """Without residual, output is just tile output."""
        ffn = XORRoutingFFN(d_model=64, num_tiles=8, use_residual=False)
        x = torch.randn(16, 64)
        output, _, _ = ffn(x)
        assert output.shape == x.shape

    def test_routing_stats(self):
        """Routing stats are tracked in hard mode."""
        ffn = XORRoutingFFN(d_model=64, num_tiles=8, use_soft_training=False)
        ffn.train()
        x = torch.randn(32, 64)
        ffn(x)
        stats = ffn.get_routing_stats()
        assert stats['active_tiles'] > 0
        assert 'usage_mean' in stats

    def test_stats_reset(self):
        """Stats can be reset."""
        ffn = XORRoutingFFN(d_model=64, num_tiles=8)
        ffn.train()
        x = torch.randn(32, 64)
        ffn(x)
        ffn.reset_stats()
        stats = ffn.get_routing_stats()
        assert stats['active_tiles'] == 0


# =============================================================================
# TEST: Hierarchical XOR Routing FFN
# =============================================================================

class TestHierarchicalXORRoutingFFN:
    """Test Hierarchical XOR Routing FFN."""

    def test_creation(self):
        """Create hierarchical FFN."""
        ffn = HierarchicalXORRoutingFFN(
            d_model=64,
            num_tiles=32,
            tiles_per_cluster=8,
        )
        assert ffn.num_tiles == 32
        assert ffn.num_clusters == 4
        assert ffn.tiles_per_cluster == 8

    def test_forward_2d(self):
        """Forward with 2D input."""
        ffn = HierarchicalXORRoutingFFN(
            d_model=64,
            num_tiles=32,
            tiles_per_cluster=8,
        )
        x = torch.randn(16, 64)
        output, routing_info, aux = ffn(x)
        assert output.shape == x.shape
        assert 'tile_idx' in routing_info
        assert 'cluster_idx' in routing_info

    def test_forward_3d(self):
        """Forward with 3D input."""
        ffn = HierarchicalXORRoutingFFN(
            d_model=64,
            num_tiles=32,
            tiles_per_cluster=8,
        )
        x = torch.randn(4, 16, 64)
        output, routing_info, aux = ffn(x)
        assert output.shape == x.shape

    def test_gradient_flow(self):
        """Gradients flow through hierarchical routing."""
        ffn = HierarchicalXORRoutingFFN(
            d_model=64,
            num_tiles=32,
            tiles_per_cluster=8,
        )
        x = torch.randn(16, 64, requires_grad=True)
        output, _, _ = ffn(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_tile_idx_in_range(self):
        """Tile indices are in valid range."""
        ffn = HierarchicalXORRoutingFFN(
            d_model=64,
            num_tiles=32,
            tiles_per_cluster=8,
        )
        x = torch.randn(100, 64)
        output, routing_info, aux = ffn(x)
        tile_idx = routing_info['tile_idx']
        assert (tile_idx >= 0).all()
        assert (tile_idx < 32).all()

    def test_cluster_idx_in_range(self):
        """Cluster indices are in valid range."""
        ffn = HierarchicalXORRoutingFFN(
            d_model=64,
            num_tiles=32,
            tiles_per_cluster=8,
        )
        x = torch.randn(100, 64)
        output, routing_info, aux = ffn(x)
        cluster_idx = routing_info['cluster_idx']
        assert (cluster_idx >= 0).all()
        assert (cluster_idx < 4).all()

    def test_routing_stats(self):
        """Hierarchical routing stats work."""
        ffn = HierarchicalXORRoutingFFN(
            d_model=64,
            num_tiles=32,
            tiles_per_cluster=8,
        )
        ffn.train()
        x = torch.randn(100, 64)
        ffn(x)
        stats = ffn.get_routing_stats()
        assert 'active_tiles' in stats
        assert 'active_clusters' in stats


# =============================================================================
# TEST: Providence Connection
# =============================================================================

class TestProvidenceConnection:
    """Test that XOR routing mirrors Providence semantics."""

    def test_routing_is_minimum_distance(self):
        """Routing goes to minimum Hamming distance tile."""
        router = XORRouter(d_model=32, num_tiles=4, use_soft_training=False)
        router.eval()

        # Set known signatures
        with torch.no_grad():
            router.signatures.zero_()
            router.signatures[0, :8] = 1.0    # First 8 bits positive
            router.signatures[1, 8:16] = 1.0  # Second 8 bits positive
            router.signatures[2, 16:24] = 1.0  # Third 8 bits positive
            router.signatures[3, 24:] = 1.0   # Last 8 bits positive

        # Query matching first pattern
        x = torch.zeros(1, 32)
        x[0, :8] = 1.0

        tile_idx, _ = router(x)
        assert tile_idx.item() == 0

    def test_soft_routing_is_attention(self):
        """Soft routing = softmax(-distance) = attention."""
        router = XORRouter(d_model=32, num_tiles=4, use_soft_training=True, temperature=1.0)
        router.train()

        x = torch.randn(8, 32)
        tile_idx, gate = router(x)

        # Gate should sum to 1 (like attention)
        assert torch.allclose(gate.sum(dim=-1), torch.ones(8), atol=1e-5)

        # Gate values should be non-negative
        assert (gate >= 0).all()


# =============================================================================
# TEST: Training
# =============================================================================

class TestTraining:
    """Test training with XOR routing."""

    def test_parameter_update(self):
        """Parameters update during training."""
        ffn = XORRoutingFFN(d_model=64, num_tiles=8)
        optimizer = torch.optim.Adam(ffn.parameters(), lr=0.01)

        # Get initial signature
        initial_sigs = ffn.get_signatures().clone()

        # Training step
        x = torch.randn(32, 64)
        target = torch.randn(32, 64)

        for _ in range(10):
            optimizer.zero_grad()
            output, _, aux = ffn(x)
            loss = F.mse_loss(output, target) + aux['total_aux']
            loss.backward()
            optimizer.step()

        # Signatures should have changed
        final_sigs = ffn.get_signatures()
        assert not torch.allclose(initial_sigs, final_sigs)

    def test_balance_loss_effect(self):
        """Balance loss encourages even distribution."""
        ffn = XORRoutingFFN(d_model=64, num_tiles=8)
        ffn.train()

        # Run some data through
        for _ in range(10):
            x = torch.randn(64, 64)
            _, _, aux = ffn(x)

        # Balance loss should be small-ish (not huge)
        assert aux['balance'] < 1.0


# Make F available for the test
import torch.nn.functional as F
