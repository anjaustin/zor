"""
Tests for Gradient Truth architecture.

Verifies:
1. No STE is used (gradients don't flow through shapes)
2. Gradients DO flow through routing and scales
3. Shapes are truly frozen
4. Forward pass produces valid outputs
5. Training converges without STE
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from trix.nn.gradient_truth import (
    Shape,
    ShapeBank,
    PolynomialShapeBank,
    DistilledShapeBank,
    GradientTruthFFN,
    GradientTruthBlock,
    RoutingInfo,
    ShapeGenesis,
    create_gradient_truth_ffn,
    create_gradient_truth_block,
)


class TestShapeBank:
    """Tests for ShapeBank and its variants."""
    
    def test_shape_creation(self):
        """Test creating a simple shape."""
        shape = Shape(
            name="test",
            fn=lambda x: x * 2,
            signature=torch.ones(64),
        )
        
        x = torch.randn(8, 64)
        out = shape(x)
        
        assert out.shape == x.shape
        assert torch.allclose(out, x * 2)
    
    def test_polynomial_bank_creation(self):
        """Test creating polynomial shape bank."""
        d_model = 128
        num_shapes = 16
        
        bank = PolynomialShapeBank.from_primitives(
            d_model=d_model,
            num_shapes=num_shapes,
        )
        
        assert len(bank) == num_shapes
        assert bank.signatures.shape == (num_shapes, d_model)
    
    def test_shape_bank_forward(self):
        """Test shape bank forward pass."""
        d_model = 64
        batch = 8
        num_shapes = 8
        
        bank = PolynomialShapeBank.from_primitives(
            d_model=d_model,
            num_shapes=num_shapes,
        )
        
        x = torch.randn(batch, d_model)
        out = bank(x)
        
        assert out.shape == (batch, num_shapes, d_model)
    
    def test_shape_bank_frozen(self):
        """Test that shape bank is frozen (no gradients)."""
        bank = PolynomialShapeBank.from_primitives(d_model=64, num_shapes=8)
        
        for param in bank.parameters():
            assert not param.requires_grad, "Shape bank should have no trainable params"
    
    def test_signatures_are_ternary(self):
        """Test that signatures are ternary values."""
        bank = PolynomialShapeBank.from_primitives(d_model=64, num_shapes=8)
        
        sigs = bank.get_signatures()
        unique_values = sigs.unique()
        
        # Should only contain -1, 0, +1
        for v in unique_values:
            assert v.item() in [-1.0, 0.0, 1.0], f"Unexpected signature value: {v}"


class TestGradientTruthFFN:
    """Tests for GradientTruthFFN."""
    
    @pytest.fixture
    def ffn(self):
        """Create a test FFN."""
        shape_bank = PolynomialShapeBank.from_primitives(d_model=64, num_shapes=8)
        return GradientTruthFFN(d_model=64, shape_bank=shape_bank)
    
    def test_forward_2d(self, ffn):
        """Test forward pass with 2D input."""
        x = torch.randn(16, 64)
        output, routing = ffn(x)
        
        assert output.shape == x.shape
        assert routing.weights.shape == (16, 8)
        assert routing.selected.shape == (16,)
    
    def test_forward_3d(self, ffn):
        """Test forward pass with 3D input."""
        x = torch.randn(4, 16, 64)
        output, routing = ffn(x)
        
        assert output.shape == x.shape
        assert routing.weights.shape == (4, 16, 8)
        assert routing.selected.shape == (4, 16)
    
    def test_gradients_flow_through_routing(self, ffn):
        """Test that gradients flow through routing layer."""
        x = torch.randn(8, 64)
        output, _ = ffn(x)
        loss = output.sum()
        loss.backward()
        
        assert ffn.router.weight.grad is not None
        assert ffn.router.weight.grad.abs().sum() > 0
    
    def test_gradients_flow_through_scales(self, ffn):
        """Test that gradients flow through magnitude scales."""
        x = torch.randn(8, 64)
        output, _ = ffn(x)
        loss = output.sum()
        loss.backward()
        
        assert ffn.scales.grad is not None
        assert ffn.scales.grad.abs().sum() > 0
    
    def test_no_gradients_through_shapes(self, ffn):
        """Test that NO gradients flow through shape bank."""
        x = torch.randn(8, 64)
        output, _ = ffn(x)
        loss = output.sum()
        loss.backward()
        
        # Shape bank should have no gradient-enabled parameters
        for param in ffn.shape_bank.parameters():
            assert param.grad is None or param.grad.abs().sum() == 0
    
    def test_routing_weights_sum_to_one(self, ffn):
        """Test that routing weights are valid probabilities."""
        x = torch.randn(8, 64)
        _, routing = ffn(x)
        
        sums = routing.weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    
    def test_routing_entropy_valid(self, ffn):
        """Test that routing entropy is non-negative."""
        x = torch.randn(8, 64)
        _, routing = ffn(x)
        
        assert (routing.entropy >= 0).all()
    
    def test_deterministic_eval(self, ffn):
        """Test that eval mode is deterministic."""
        ffn.eval()
        x = torch.randn(8, 64)
        
        out1, _ = ffn(x)
        out2, _ = ffn(x)
        
        assert torch.allclose(out1, out2)
    
    def test_hard_routing(self):
        """Test Gumbel-softmax hard routing."""
        shape_bank = PolynomialShapeBank.from_primitives(d_model=64, num_shapes=8)
        ffn = GradientTruthFFN(
            d_model=64,
            shape_bank=shape_bank,
            routing_type='hard',
        )
        
        x = torch.randn(8, 64)
        output, routing = ffn(x)
        
        # Hard routing should still work
        assert output.shape == x.shape
        
        # Gradients should still flow
        loss = output.sum()
        loss.backward()
        assert ffn.router.weight.grad is not None


class TestGradientTruthBlock:
    """Tests for GradientTruthBlock transformer block."""
    
    @pytest.fixture
    def block(self):
        """Create a test block."""
        shape_bank = PolynomialShapeBank.from_primitives(d_model=64, num_shapes=8)
        return GradientTruthBlock(
            d_model=64,
            n_heads=4,
            shape_bank=shape_bank,
        )
    
    def test_forward(self, block):
        """Test forward pass."""
        x = torch.randn(2, 16, 64)
        output, routing = block(x)
        
        assert output.shape == x.shape
    
    def test_causal_attention(self, block):
        """Test causal attention masking."""
        x = torch.randn(2, 16, 64)
        
        output_causal, _ = block(x, is_causal=True)
        output_bidir, _ = block(x, is_causal=False)
        
        # Outputs should differ due to masking
        assert not torch.allclose(output_causal, output_bidir)
    
    def test_gradients_flow(self, block):
        """Test that gradients flow through entire block."""
        x = torch.randn(2, 16, 64, requires_grad=True)
        output, _ = block(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestShapeGenesis:
    """Tests for shape discovery utilities."""
    
    def test_derive_boolean_shapes(self):
        """Test deriving Boolean shapes."""
        bank = ShapeGenesis.derive_boolean_shapes(d_model=64)
        
        assert len(bank) > 0
        assert isinstance(bank, ShapeBank)
    
    def test_evolve_shapes(self):
        """Test evolving shapes."""
        # Simple fitness: prefer shapes that produce non-zero output
        def fitness(fn):
            x = torch.randn(8, 32)
            out = fn(x)
            return out.abs().mean().item()
        
        bank = ShapeGenesis.evolve_shapes(
            d_model=32,
            num_shapes=4,
            fitness_fn=fitness,
            generations=5,
            population=10,
        )
        
        assert len(bank) == 4
        assert isinstance(bank, ShapeBank)


class TestTrainingConvergence:
    """Tests that verify training works without STE."""
    
    def test_simple_training_loop(self):
        """Test that a simple training loop converges."""
        d_model = 32
        batch = 16
        
        # Create model
        ffn = create_gradient_truth_ffn(d_model=d_model, num_shapes=8)
        optimizer = torch.optim.Adam(ffn.parameters(), lr=0.01)
        
        # Simple target: learn to approximately double the input
        losses = []
        for step in range(50):
            x = torch.randn(batch, d_model)
            target = x * 2
            
            output, _ = ffn(x)
            loss = F.mse_loss(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Loss should decrease
        assert losses[-1] < losses[0], "Training should reduce loss"
    
    def test_routing_learns_specialization(self):
        """Test that routing learns to specialize."""
        d_model = 32
        batch = 32
        
        # Create model with few shapes
        ffn = create_gradient_truth_ffn(d_model=d_model, num_shapes=4)
        optimizer = torch.optim.Adam(ffn.parameters(), lr=0.01)
        
        # Train on diverse inputs
        for step in range(100):
            x = torch.randn(batch, d_model)
            target = torch.tanh(x)  # Non-linear target
            
            output, routing = ffn(x)
            loss = F.mse_loss(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Check routing uses multiple shapes
        x = torch.randn(batch, d_model)
        _, routing = ffn(x)
        
        unique_shapes = routing.selected.unique()
        assert len(unique_shapes) > 1, "Routing should use multiple shapes"
    
    def test_no_nan_or_inf(self):
        """Test numerical stability - no NaN or Inf."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=16)
        
        # Test with various input scales
        for scale in [0.01, 1.0, 100.0]:
            x = torch.randn(16, 64) * scale
            output, _ = ffn(x)
            
            assert not torch.isnan(output).any(), f"NaN detected at scale {scale}"
            assert not torch.isinf(output).any(), f"Inf detected at scale {scale}"


class TestConvenienceFunctions:
    """Tests for convenience factory functions."""
    
    def test_create_gradient_truth_ffn(self):
        """Test FFN factory function."""
        ffn = create_gradient_truth_ffn(d_model=128, num_shapes=16)
        
        assert isinstance(ffn, GradientTruthFFN)
        assert ffn.num_shapes == 16
    
    def test_create_gradient_truth_block(self):
        """Test block factory function."""
        block = create_gradient_truth_block(
            d_model=128,
            n_heads=8,
            num_shapes=16,
        )
        
        assert isinstance(block, GradientTruthBlock)
        assert isinstance(block.ffn, GradientTruthFFN)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_shape(self):
        """Test with single shape."""
        bank = PolynomialShapeBank.from_primitives(d_model=64, num_shapes=1)
        ffn = GradientTruthFFN(d_model=64, shape_bank=bank)
        
        x = torch.randn(8, 64)
        output, routing = ffn(x)
        
        # With single shape, routing should be trivial
        assert (routing.selected == 0).all()
    
    def test_batch_size_one(self):
        """Test with batch size 1."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        
        x = torch.randn(1, 64)
        output, _ = ffn(x)
        
        assert output.shape == (1, 64)
    
    def test_large_batch(self):
        """Test with large batch."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        
        x = torch.randn(1024, 64)
        output, _ = ffn(x)
        
        assert output.shape == (1024, 64)
    
    def test_varying_sequence_lengths(self):
        """Test 3D input with varying sequence lengths."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        
        for seq_len in [1, 16, 128]:
            x = torch.randn(4, seq_len, 64)
            output, _ = ffn(x)
            assert output.shape == x.shape


class TestGradientCorrectness:
    """
    Tests that verify gradients are mathematically correct.
    
    This is the key test: we should have REAL gradients, not STE approximations.
    """
    
    def test_gradients_are_real_not_ste(self):
        """
        The core test for Gradient Truth: verify no STE is used.
        
        We test this by checking:
        1. Gradients flow through routing (already tested)
        2. Gradients flow through scales (already tested)
        3. NO gradients flow through shape bank (already tested)
        4. Changing routing weights changes the output smoothly
        """
        ffn = create_gradient_truth_ffn(d_model=32, num_shapes=8, dropout=0.0)
        ffn.eval()
        
        x = torch.randn(4, 32)
        
        # Get base output
        with torch.no_grad():
            base_output, base_routing = ffn(x)
        
        # Perturb routing weights slightly
        with torch.no_grad():
            original_weight = ffn.router.weight.clone()
            ffn.router.weight.add_(torch.randn_like(ffn.router.weight) * 0.01)
            perturbed_output, perturbed_routing = ffn(x)
            ffn.router.weight.copy_(original_weight)
        
        # Outputs should be different (routing matters)
        assert not torch.allclose(base_output, perturbed_output, atol=1e-6), \
            "Routing should affect output"
        
        # The difference should be small (smooth, not discrete jump)
        diff = (base_output - perturbed_output).abs().mean()
        assert diff < 1.0, f"Output change should be smooth, got {diff}"
    
    def test_gradient_chain_rule(self):
        """
        Test that gradients obey chain rule through routing.
        
        Key insight: gradients through soft attention should satisfy:
        d(loss)/d(input) = sum_i(weight_i * d(shape_i)/d(input)) + d(weights)/d(input) * ...
        """
        ffn = create_gradient_truth_ffn(d_model=32, num_shapes=4, dropout=0.0)
        
        x = torch.randn(4, 32, requires_grad=True)
        output, routing = ffn(x)
        
        # Gradient should exist and be non-zero
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        
        # Gradient magnitude should be reasonable (not exploding)
        assert x.grad.abs().max() < 1000
    
    def test_router_gradients_vary(self):
        """
        Test that router gradients are not uniform.
        
        Real gradients through softmax should vary across weights.
        STE-like behavior would show more uniform gradients.
        """
        ffn = create_gradient_truth_ffn(d_model=32, num_shapes=8)
        
        # Use diverse inputs to get varied gradients
        x = torch.randn(16, 32, requires_grad=True)
        output, _ = ffn(x)
        loss = output.sum()
        loss.backward()
        
        # Router weight gradients should vary (not uniform)
        router_grad = ffn.router.weight.grad
        assert router_grad is not None
        
        # Check variance of gradients - should be non-trivial
        grad_std = router_grad.std()
        grad_mean = router_grad.abs().mean()
        
        # Coefficient of variation should be reasonable (not near zero)
        cv = grad_std / (grad_mean + 1e-8)
        assert cv > 0.1, f"Router gradients are too uniform (CV={cv})"
    
    def test_scales_receive_gradients(self):
        """
        Test that magnitude scales receive meaningful gradients.
        """
        ffn = create_gradient_truth_ffn(d_model=32, num_shapes=8)
        
        x = torch.randn(16, 32)
        output, _ = ffn(x)
        loss = output.sum()
        loss.backward()
        
        # Scales should have non-zero gradients
        scales_grad = ffn.scales.grad
        assert scales_grad is not None
        assert scales_grad.abs().sum() > 0
        
        # Different scales should have different gradients (based on usage)
        assert scales_grad.std() > 0, "Scale gradients should vary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
