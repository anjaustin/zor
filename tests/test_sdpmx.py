"""
Tests for SDPMX Pipeline (Vi's Synthesis).

S → D → P → M → X
Smooth → Differentiate → Project → Mask → XOR

"The geometry dictates the operator order. AB ≠ BA."
"""

import pytest
import torch
import torch.nn.functional as F

from trix.nn import (
    SmoothingOperator,
    DifferentiationOperator,
    ProjectionOperator,
    MaskOperator,
    XOROperator,
    SDPMXPipeline,
)


class TestSmoothingOperator:
    """Tests for S: Discrete → Continuous."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        S = SmoothingOperator(dim=64, grid_size=16)
        x = torch.randn(8, 64)
        y = S(x)
        assert y.shape == x.shape

    def test_derivative(self):
        """Test derivative computation."""
        S = SmoothingOperator(dim=64, grid_size=16)
        x = torch.randn(8, 64)
        dy = S.derivative(x)
        assert dy.shape == x.shape

    def test_gradients(self):
        """Test gradient flow."""
        S = SmoothingOperator(dim=32, grid_size=16)
        x = torch.randn(4, 32, requires_grad=True)
        y = S(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None


class TestDifferentiationOperator:
    """Tests for D: Surface → Gradients."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        S = SmoothingOperator(dim=64, grid_size=16)
        D = DifferentiationOperator(S)
        x = torch.randn(8, 64)
        gradients = D(x)
        assert gradients.shape == x.shape

    def test_derivative_consistency(self):
        """Test that D matches S.derivative."""
        S = SmoothingOperator(dim=32, grid_size=16)
        D = DifferentiationOperator(S)
        x = torch.randn(8, 32)

        d1 = D(x)
        d2 = S.derivative(x)

        assert torch.allclose(d1, d2)


class TestProjectionOperator:
    """Tests for P: Gradients → Subspace."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        P = ProjectionOperator(dim=64, num_subspaces=8)
        gradients = torch.randn(8, 64)
        projected, indices = P(gradients)

        assert projected.shape == gradients.shape
        assert indices.shape == (8,)

    def test_ternary_signatures(self):
        """Test ternary signature extraction."""
        P = ProjectionOperator(dim=64, num_subspaces=8)
        sigs = P.get_ternary_signatures()

        assert sigs.shape == (8, 64)
        assert sigs.abs().max() <= 1  # Ternary

    def test_subspace_routing(self):
        """Test that different inputs route to different subspaces."""
        P = ProjectionOperator(dim=64, num_subspaces=16)

        # Random inputs should route to various subspaces
        x = torch.randn(100, 64)
        _, indices = P(x)

        unique_subspaces = len(indices.unique())
        assert unique_subspaces > 1  # Should use multiple subspaces


class TestMaskOperator:
    """Tests for M: Projection → XOR Mask."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        M = MaskOperator(dim=64, threshold=0.0)
        projected = torch.randn(8, 64)
        mask = M(projected)

        assert mask.shape == projected.shape
        assert mask.min() >= 0
        assert mask.max() <= 1

    def test_threshold_behavior(self):
        """Test that threshold affects mask density."""
        M_low = MaskOperator(dim=64, threshold=-1.0)
        M_high = MaskOperator(dim=64, threshold=1.0)

        projected = torch.randn(8, 64)

        mask_low = M_low(projected)
        mask_high = M_high(projected)

        # Lower threshold should produce denser masks
        assert mask_low.mean() >= mask_high.mean()


class TestXOROperator:
    """Tests for X: Mask + Memory → New Memory."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        X = XOROperator(dim=64)
        x = torch.randn(8, 64)
        mask = torch.ones(8, 64)
        projected = torch.randn(8, 64)

        x_new = X(x, mask, projected)
        assert x_new.shape == x.shape

    def test_zero_mask_identity(self):
        """Test that zero mask preserves input."""
        X = XOROperator(dim=64)
        x = torch.randn(8, 64)
        mask = torch.zeros(8, 64)
        projected = torch.randn(8, 64)

        x_new = X(x, mask, projected)
        assert torch.allclose(x_new, x)


class TestSDPMXPipeline:
    """Tests for complete S → D → P → M → X pipeline."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        pipeline = SDPMXPipeline(dim=64, grid_size=16, num_subspaces=8)
        x = torch.randn(8, 64)
        x_new, info = pipeline(x)

        assert x_new.shape == x.shape
        assert 'subspace_idx' in info
        assert 'mask_density' in info

    def test_forward_with_intermediates(self):
        """Test forward pass with intermediate values."""
        pipeline = SDPMXPipeline(dim=64, grid_size=16, num_subspaces=8)
        x = torch.randn(8, 64)
        x_new, info = pipeline(x, return_intermediates=True)

        assert 'smoothed' in info
        assert 'gradients' in info
        assert 'projected' in info
        assert 'mask' in info

    def test_gradient_flow(self):
        """Test gradient flow through entire pipeline."""
        pipeline = SDPMXPipeline(dim=64, grid_size=16, num_subspaces=8)
        x = torch.randn(8, 64, requires_grad=True)
        x_new, info = pipeline(x)
        loss = x_new.sum()
        loss.backward()

        assert x.grad is not None
        assert not x.grad.isnan().any()

    def test_forward_with_loss(self):
        """Test forward pass with reconstruction loss."""
        pipeline = SDPMXPipeline(dim=64, grid_size=16, num_subspaces=8)
        x = torch.randn(8, 64)
        target = torch.randn(8, 64)

        x_new, loss, info = pipeline.forward_with_loss(x, target)

        assert x_new.shape == x.shape
        assert loss.ndim == 0  # Scalar loss

    def test_learning(self):
        """Test that pipeline can learn a simple transformation."""
        pipeline = SDPMXPipeline(dim=32, grid_size=8, num_subspaces=4)

        # Simple target: scale input by 2
        x = torch.randn(100, 32)
        target = x * 2

        optimizer = torch.optim.Adam(pipeline.parameters(), lr=0.01)

        initial_loss = None
        final_loss = None

        for epoch in range(50):
            optimizer.zero_grad()
            _, loss, _ = pipeline.forward_with_loss(x, target)
            loss.backward()
            optimizer.step()

            if epoch == 0:
                initial_loss = loss.item()
            if epoch == 49:
                final_loss = loss.item()

        # Loss should decrease
        assert final_loss < initial_loss

    def test_batch_independence(self):
        """Test that batches are processed independently."""
        pipeline = SDPMXPipeline(dim=32, grid_size=8, num_subspaces=4)

        x1 = torch.randn(1, 32)
        x2 = torch.randn(1, 32)
        x_batch = torch.cat([x1, x2], dim=0)

        # Process separately
        y1, _ = pipeline(x1)
        y2, _ = pipeline(x2)

        # Process together
        y_batch, _ = pipeline(x_batch)

        assert torch.allclose(y_batch[0], y1[0], atol=1e-5)
        assert torch.allclose(y_batch[1], y2[0], atol=1e-5)

    def test_parameter_count(self):
        """Test parameter counting."""
        pipeline = SDPMXPipeline(dim=64, grid_size=16, num_subspaces=8)

        total_params = sum(p.numel() for p in pipeline.parameters())
        assert total_params > 0

        # S should have params
        s_params = sum(p.numel() for p in pipeline.S.parameters())
        assert s_params > 0

        # P should have params
        p_params = sum(p.numel() for p in pipeline.P.parameters())
        assert p_params > 0


class TestOperatorOrder:
    """Tests verifying that operator order matters (AB ≠ BA)."""

    def test_sd_vs_ds(self):
        """Test that S→D ≠ D→S conceptually."""
        # D depends on S, so D→S doesn't make sense
        # This test verifies the architectural constraint
        S = SmoothingOperator(dim=32, grid_size=16)
        D = DifferentiationOperator(S)

        # D needs S's splines to compute derivatives
        assert hasattr(D, 'smoother')
        assert D.smoother is S

    def test_pipeline_composition(self):
        """Test that pipeline composes operators correctly."""
        pipeline = SDPMXPipeline(dim=32, grid_size=8, num_subspaces=4)

        # Verify operator chain exists
        assert hasattr(pipeline, 'S')
        assert hasattr(pipeline, 'D')
        assert hasattr(pipeline, 'P')
        assert hasattr(pipeline, 'M')
        assert hasattr(pipeline, 'X')

        # D should reference S
        assert pipeline.D.smoother is pipeline.S


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
