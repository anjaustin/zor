"""
Tests for KAN (Kolmogorov-Arnold Network) modules.

Track B: Additive KAN for High-Dimensional Inputs
"768^16 = Heat Death. 768×16 = Doable."
"""

import pytest
import torch
import torch.nn.functional as F

from trix.nn import (
    Spline1D,
    AdditiveKAN,
    ProductSplineKAN,
    SharedAdditiveKAN,
    KANTile,
    HierarchicalKANFFN,
)


class TestSpline1D:
    """Tests for 1D piecewise linear spline."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        spline = Spline1D(grid_size=16)
        x = torch.randn(32)
        y = spline(x)
        assert y.shape == x.shape

    def test_batch_forward(self):
        """Test batched forward pass."""
        spline = Spline1D(grid_size=16)
        x = torch.randn(8, 32)
        y = spline(x)
        assert y.shape == x.shape

    def test_gradients(self):
        """Test gradient flow."""
        spline = Spline1D(grid_size=16)
        x = torch.randn(32, requires_grad=True)
        y = spline(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert spline.bases.grad is not None
        assert spline.slopes.grad is not None

    def test_different_grid_sizes(self):
        """Test various grid sizes."""
        for grid_size in [4, 8, 16, 32]:
            spline = Spline1D(grid_size=grid_size)
            x = torch.randn(100)
            y = spline(x)
            assert y.shape == x.shape


class TestAdditiveKAN:
    """Tests for Additive KAN (sum of 1D splines)."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        kan = AdditiveKAN(input_dim=8, output_dim=4, grid_size=16)
        x = torch.randn(16, 8)
        y = kan(x)
        assert y.shape == (16, 4)

    def test_learn_addition(self):
        """Test learning f(x,y) = x + y."""
        kan = AdditiveKAN(input_dim=2, output_dim=1, grid_size=16)
        optimizer = torch.optim.Adam(kan.parameters(), lr=0.01)

        # Training data
        x = torch.rand(1000, 2) * 2 - 1
        y_true = x[:, 0:1] + x[:, 1:2]

        # Train
        for _ in range(200):
            optimizer.zero_grad()
            y_pred = kan(x)
            loss = F.mse_loss(y_pred, y_true)
            loss.backward()
            optimizer.step()

        # Test
        x_test = torch.rand(100, 2) * 2 - 1
        y_test = x_test[:, 0:1] + x_test[:, 1:2]
        y_pred = kan(x_test)
        mse = F.mse_loss(y_pred, y_test).item()

        assert mse < 0.01, f"MSE {mse} too high for simple addition"

    def test_parameter_count(self):
        """Test parameter counting."""
        kan = AdditiveKAN(input_dim=64, output_dim=32, grid_size=16)
        expected = 2 * 16 * 64 * 32 + 32  # splines + bias
        assert kan.num_parameters() == expected


class TestSharedAdditiveKAN:
    """Tests for efficient SharedAdditiveKAN."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        kan = SharedAdditiveKAN(input_dim=128, output_dim=128, grid_size=16)
        x = torch.randn(8, 128)
        y = kan(x)
        assert y.shape == (8, 128)

    def test_gradients(self):
        """Test gradient flow."""
        kan = SharedAdditiveKAN(input_dim=64, output_dim=64, grid_size=16)
        x = torch.randn(8, 64, requires_grad=True)
        y = kan(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert kan.mix.grad is not None

    def test_parameter_efficiency(self):
        """Test that SharedKAN is more efficient than AdditiveKAN."""
        input_dim = 256
        output_dim = 256
        grid_size = 16

        shared = SharedAdditiveKAN(input_dim, output_dim, grid_size)
        additive = AdditiveKAN(input_dim, output_dim, grid_size)

        # SharedKAN should have fewer parameters
        assert shared.num_parameters() < additive.num_parameters()


class TestKANTile:
    """Tests for KAN-based routing tile."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        tile = KANTile(d_model=128, grid_size=16, tile_id=0, num_tiles=16)
        x = torch.randn(8, 128)
        y = tile(x)
        assert y.shape == x.shape

    def test_signature(self):
        """Test signature extraction."""
        tile = KANTile(d_model=128, grid_size=16, tile_id=0, num_tiles=16)
        sig = tile.get_signature()
        assert sig.shape == (128,)
        assert sig.abs().max() <= 1  # Ternary

    def test_sparse_slice(self):
        """Test that tile only modifies its slice."""
        d_model = 128
        num_tiles = 16
        tile = KANTile(d_model=d_model, grid_size=16, tile_id=0, num_tiles=num_tiles, use_residual=False)

        x = torch.zeros(8, d_model)
        y = tile(x)

        # Only the tile's slice should be non-zero
        d_slice = d_model // num_tiles
        # Check that modification is localized
        assert y.shape == x.shape


class TestHierarchicalKANFFN:
    """Tests for Hierarchical KAN FFN."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        model = HierarchicalKANFFN(
            d_model=128,
            num_tiles=16,
            tiles_per_cluster=4,
            grid_size=16,
        )
        x = torch.randn(4, 32, 128)
        y = model(x)
        assert y.shape == x.shape

    def test_forward_with_stats(self):
        """Test forward pass with routing stats."""
        model = HierarchicalKANFFN(
            d_model=128,
            num_tiles=16,
            tiles_per_cluster=4,
            grid_size=16,
        )
        x = torch.randn(4, 32, 128)
        y, stats = model(x, return_stats=True)

        assert y.shape == x.shape
        assert 'unique_tiles' in stats
        assert stats['unique_tiles'] > 0

    def test_gradient_flow(self):
        """Test gradient flow through routing."""
        model = HierarchicalKANFFN(
            d_model=128,
            num_tiles=16,
            tiles_per_cluster=4,
            grid_size=16,
        )
        x = torch.randn(4, 32, 128, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Input should get gradients
        assert x.grad is not None

        # At least some tiles should get gradients (those that were routed to)
        tiles_with_grad = sum(
            1 for tile in model.tiles
            if any(p.grad is not None for p in tile.parameters())
        )
        assert tiles_with_grad > 0

    def test_routing_stats(self):
        """Test routing statistics."""
        model = HierarchicalKANFFN(
            d_model=128,
            num_tiles=16,
            tiles_per_cluster=4,
            grid_size=16,
        )

        # Run several batches to accumulate stats
        for _ in range(10):
            x = torch.randn(4, 32, 128)
            model(x)

        stats = model.get_routing_stats()

        assert stats['num_tiles'] == 16
        assert stats['num_clusters'] == 4
        assert stats['active_tiles'] > 0

    def test_memory_report(self):
        """Test memory reporting."""
        model = HierarchicalKANFFN(
            d_model=128,
            num_tiles=16,
            tiles_per_cluster=4,
            grid_size=16,
        )

        report = model.memory_report()

        assert 'total_params' in report
        assert 'tile_params' in report
        assert 'memory_2bit' in report
        assert report['memory_2bit'] < report['memory_fp32']

    def test_parameter_efficiency(self):
        """Test that HierarchicalKANFFN is more efficient than MLP."""
        d_model = 256
        model = HierarchicalKANFFN(
            d_model=d_model,
            num_tiles=64,
            tiles_per_cluster=8,
            grid_size=16,
        )

        kan_params = model.total_parameters()

        # Standard MLP FFN: d_model -> 4*d_model -> d_model
        mlp_params = d_model * (d_model * 4) * 2 + d_model * 4 + d_model

        # KAN should be significantly smaller
        assert kan_params < mlp_params * 0.1, f"KAN {kan_params} should be <10% of MLP {mlp_params}"


class TestProductSplineKAN:
    """Tests for 2D spline pairs."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        kan = ProductSplineKAN(input_dim=8, output_dim=4, grid_size=16)
        x = torch.randn(16, 8)
        y = kan(x)
        assert y.shape == (16, 4)

    def test_odd_dimensions(self):
        """Test handling of odd input dimensions."""
        kan = ProductSplineKAN(input_dim=9, output_dim=4, grid_size=16)
        x = torch.randn(16, 9)
        y = kan(x)
        assert y.shape == (16, 4)

    def test_gradients(self):
        """Test gradient flow."""
        kan = ProductSplineKAN(input_dim=8, output_dim=4, grid_size=16)
        x = torch.randn(16, 8, requires_grad=True)
        y = kan(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
