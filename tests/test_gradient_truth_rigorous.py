"""
Rigorous Tests for Gradient Truth Architecture.

Categories:
1. Edge Cases - Boundary conditions, extreme values
2. Numerical Stability - NaN, Inf, precision
3. Mathematical Correctness - Shapes exact, routing correct
4. Gradient Integrity - Correct flow, no STE artifacts
5. Training Dynamics - Convergence, specialization
6. Stress Tests - Scale behavior
7. Comparison to STE - Verify equivalence or improvement

"Gradients should flow only where there is genuine uncertainty."
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

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


# =============================================================================
# 1. EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge case and boundary condition tests."""

    # --- Batch Size ---

    def test_single_sample(self):
        """Batch size 1."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.randn(1, 64)
        output, routing = ffn(x)
        assert output.shape == (1, 64)
        assert not output.isnan().any()

    def test_large_batch(self):
        """Batch size 1024."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.randn(1024, 64)
        output, routing = ffn(x)
        assert output.shape == (1024, 64)
        assert not output.isnan().any()

    def test_very_large_batch(self):
        """Batch size 4096."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.randn(4096, 64)
        output, routing = ffn(x)
        assert output.shape == (4096, 64)
        assert not output.isnan().any()

    # --- Input Values ---

    def test_zero_input(self):
        """All-zero input."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.zeros(8, 64)
        output, _ = ffn(x)
        assert not output.isnan().any()
        assert not output.isinf().any()

    def test_constant_input(self):
        """Constant input (no variance)."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.ones(8, 64) * 0.5
        output, _ = ffn(x)
        assert not output.isnan().any()

    def test_extreme_positive(self):
        """Large positive values."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.ones(8, 64) * 100
        output, _ = ffn(x)
        assert not output.isnan().any()
        assert not output.isinf().any()

    def test_extreme_negative(self):
        """Large negative values."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.ones(8, 64) * -100
        output, _ = ffn(x)
        assert not output.isnan().any()
        assert not output.isinf().any()

    def test_mixed_extreme(self):
        """Mix of extreme positive and negative."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.randn(8, 64) * 50
        x[::2] *= -1
        output, _ = ffn(x)
        assert not output.isnan().any()
        assert not output.isinf().any()

    # --- Dimensions ---

    def test_minimal_dimension(self):
        """d_model=4 (minimal)."""
        ffn = create_gradient_truth_ffn(d_model=4, num_shapes=2)
        x = torch.randn(8, 4)
        output, _ = ffn(x)
        assert output.shape == (8, 4)

    def test_large_dimension(self):
        """d_model=512."""
        ffn = create_gradient_truth_ffn(d_model=512, num_shapes=16)
        x = torch.randn(4, 512)
        output, _ = ffn(x)
        assert output.shape == (4, 512)

    def test_very_large_dimension(self):
        """d_model=1024."""
        ffn = create_gradient_truth_ffn(d_model=1024, num_shapes=32)
        x = torch.randn(2, 1024)
        output, _ = ffn(x)
        assert output.shape == (2, 1024)

    # --- Shape Counts ---

    def test_single_shape(self):
        """num_shapes=1."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=1)
        x = torch.randn(8, 64)
        output, routing = ffn(x)
        assert output.shape == (8, 64)
        # All should route to shape 0
        assert (routing.selected == 0).all()

    def test_many_shapes(self):
        """num_shapes=64."""
        ffn = create_gradient_truth_ffn(d_model=128, num_shapes=64)
        x = torch.randn(32, 128)
        output, routing = ffn(x)
        assert output.shape == (32, 128)

    def test_shapes_more_than_batch(self):
        """More shapes than batch samples."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=32)
        x = torch.randn(4, 64)  # batch < num_shapes
        output, routing = ffn(x)
        assert output.shape == (4, 64)

    # --- 3D Input ---

    def test_3d_single_token(self):
        """3D input with single token."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.randn(4, 1, 64)
        output, routing = ffn(x)
        assert output.shape == (4, 1, 64)

    def test_3d_long_sequence(self):
        """3D input with long sequence."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.randn(2, 512, 64)
        output, routing = ffn(x)
        assert output.shape == (2, 512, 64)


# =============================================================================
# 2. NUMERICAL STABILITY
# =============================================================================

class TestNumericalStability:
    """Numerical stability tests - NaN, Inf, precision."""

    def test_no_nan_random_inputs(self):
        """No NaN with random inputs over 100 iterations."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=16)
        for _ in range(100):
            x = torch.randn(16, 64)
            output, _ = ffn(x)
            assert not output.isnan().any(), "NaN detected"

    def test_no_inf_random_inputs(self):
        """No Inf with random inputs over 100 iterations."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=16)
        for _ in range(100):
            x = torch.randn(16, 64)
            output, _ = ffn(x)
            assert not output.isinf().any(), "Inf detected"

    def test_gradient_no_nan(self):
        """Gradients should never be NaN."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        for _ in range(50):
            x = torch.randn(16, 64, requires_grad=True)
            output, _ = ffn(x)
            loss = output.sum()
            loss.backward()
            
            assert not x.grad.isnan().any(), "Input gradient NaN"
            for name, p in ffn.named_parameters():
                if p.grad is not None:
                    assert not p.grad.isnan().any(), f"Param {name} gradient NaN"
            
            ffn.zero_grad()

    def test_gradient_no_inf(self):
        """Gradients should never be Inf."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        for _ in range(50):
            x = torch.randn(16, 64, requires_grad=True)
            output, _ = ffn(x)
            loss = output.sum()
            loss.backward()
            
            assert not x.grad.isinf().any(), "Input gradient Inf"
            for name, p in ffn.named_parameters():
                if p.grad is not None:
                    assert not p.grad.isinf().any(), f"Param {name} gradient Inf"
            
            ffn.zero_grad()

    def test_routing_weights_valid_probabilities(self):
        """Routing weights must be valid probabilities."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        for _ in range(50):
            x = torch.randn(16, 64)
            _, routing = ffn(x)
            
            # Non-negative
            assert (routing.weights >= 0).all(), "Negative routing weight"
            # Sum to 1
            sums = routing.weights.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), "Weights don't sum to 1"

    def test_output_bounded(self):
        """Output should be bounded for bounded input."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.randn(32, 64).clamp(-1, 1)
        output, _ = ffn(x)
        
        # Output shouldn't explode
        assert output.abs().max() < 1000, f"Output exploded: {output.abs().max()}"


# =============================================================================
# 3. MATHEMATICAL CORRECTNESS
# =============================================================================

class TestMathematicalCorrectness:
    """Verify mathematical properties hold."""

    def test_shapes_are_frozen(self):
        """Shape bank parameters should have requires_grad=False."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        
        for param in ffn.shape_bank.parameters():
            assert not param.requires_grad, "Shape bank param has requires_grad=True"

    def test_routing_sums_to_one(self):
        """Soft routing weights must sum to 1."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.randn(32, 64)
        _, routing = ffn(x)
        
        sums = routing.weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_selected_matches_argmax(self):
        """Selected shape should match argmax of weights."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.randn(32, 64)
        _, routing = ffn(x)
        
        expected_selected = routing.weights.argmax(dim=-1)
        assert (routing.selected == expected_selected).all()

    def test_entropy_formula(self):
        """Entropy should follow correct formula."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.randn(32, 64)
        _, routing = ffn(x)
        
        # Manual entropy calculation
        weights = routing.weights
        expected_entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1)
        
        assert torch.allclose(routing.entropy, expected_entropy, atol=1e-5)

    def test_residual_connection(self):
        """Verify residual connection is applied."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8, dropout=0.0)
        ffn.eval()
        
        x = torch.randn(8, 64)
        output, _ = ffn(x)
        
        # Output should contain the input (residual)
        # Correlation should be high
        correlation = F.cosine_similarity(output.flatten(), x.flatten(), dim=0)
        assert correlation > 0.1, "Residual connection seems broken"


# =============================================================================
# 4. GRADIENT INTEGRITY
# =============================================================================

class TestGradientIntegrity:
    """Verify gradient flow is correct - the core of Gradient Truth."""

    def test_router_receives_gradients(self):
        """Router weights must receive gradients."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.randn(16, 64)
        output, _ = ffn(x)
        loss = output.sum()
        loss.backward()
        
        assert ffn.router.weight.grad is not None
        assert ffn.router.weight.grad.abs().sum() > 0

    def test_scales_receive_gradients(self):
        """Magnitude scales must receive gradients."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.randn(16, 64)
        output, _ = ffn(x)
        loss = output.sum()
        loss.backward()
        
        assert ffn.scales.grad is not None
        assert ffn.scales.grad.abs().sum() > 0

    def test_output_scale_receives_gradients(self):
        """Output scale must receive gradients."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.randn(16, 64)
        output, _ = ffn(x)
        loss = output.sum()
        loss.backward()
        
        assert ffn.output_scale.grad is not None
        assert ffn.output_scale.grad.abs().sum() > 0

    def test_output_proj_receives_gradients(self):
        """Output projection must receive gradients."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.randn(16, 64)
        output, _ = ffn(x)
        loss = output.sum()
        loss.backward()
        
        assert ffn.output_proj.weight.grad is not None
        assert ffn.output_proj.weight.grad.abs().sum() > 0

    def test_shapes_no_gradients(self):
        """Shape bank must NOT receive gradients."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.randn(16, 64)
        output, _ = ffn(x)
        loss = output.sum()
        loss.backward()
        
        for name, param in ffn.shape_bank.named_parameters():
            assert param.grad is None, f"Shape bank param {name} has gradient"

    def test_input_receives_gradients(self):
        """Input tensor must receive gradients."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.randn(16, 64, requires_grad=True)
        output, _ = ffn(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gradient_magnitude_reasonable(self):
        """Gradients should not explode or vanish."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        x = torch.randn(16, 64, requires_grad=True)
        output, _ = ffn(x)
        loss = output.mean()  # Use mean for stable scale
        loss.backward()
        
        # Check gradient magnitude
        for name, p in ffn.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.norm()
                assert grad_norm < 1000, f"Gradient exploding for {name}: {grad_norm}"
                # Very small gradients are OK for some params

    def test_gradients_vary_with_input(self):
        """Different inputs should produce different gradients."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        
        x1 = torch.randn(8, 64, requires_grad=True)
        output1, _ = ffn(x1)
        loss1 = output1.sum()
        loss1.backward()
        grad1 = ffn.router.weight.grad.clone()
        ffn.zero_grad()
        
        x2 = torch.randn(8, 64, requires_grad=True) * 2  # Different input
        output2, _ = ffn(x2)
        loss2 = output2.sum()
        loss2.backward()
        grad2 = ffn.router.weight.grad.clone()
        
        # Gradients should differ
        assert not torch.allclose(grad1, grad2, atol=1e-6), "Gradients identical for different inputs"


# =============================================================================
# 5. TRAINING DYNAMICS
# =============================================================================

class TestTrainingDynamics:
    """Verify training actually works."""

    def test_loss_decreases(self):
        """Loss should decrease during training."""
        ffn = create_gradient_truth_ffn(d_model=32, num_shapes=8)
        optimizer = torch.optim.Adam(ffn.parameters(), lr=0.01)
        
        # Simple regression target
        x = torch.randn(64, 32)
        target = torch.tanh(x)
        
        losses = []
        for _ in range(100):
            output, _ = ffn(x)
            loss = F.mse_loss(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Loss should decrease
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"
        # Significant decrease
        assert losses[-1] < losses[0] * 0.5, f"Loss decrease insufficient: {losses[0]} -> {losses[-1]}"

    def test_routing_specializes(self):
        """Routing should use multiple shapes after training."""
        ffn = create_gradient_truth_ffn(d_model=32, num_shapes=8)
        optimizer = torch.optim.Adam(ffn.parameters(), lr=0.01)
        
        # Train on diverse data
        for _ in range(200):
            x = torch.randn(32, 32)
            target = torch.sin(x) + torch.cos(x * 2)
            
            output, _ = ffn(x)
            loss = F.mse_loss(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Check routing uses multiple shapes
        x = torch.randn(64, 32)
        _, routing = ffn(x)
        unique_shapes = routing.selected.unique()
        
        assert len(unique_shapes) > 1, "Routing should specialize to multiple shapes"

    def test_scales_adapt(self):
        """Scale parameters should change during training."""
        ffn = create_gradient_truth_ffn(d_model=32, num_shapes=8)
        initial_scales = ffn.scales.clone()
        
        optimizer = torch.optim.Adam(ffn.parameters(), lr=0.01)
        
        for _ in range(50):
            x = torch.randn(16, 32)
            target = x * 2
            output, _ = ffn(x)
            loss = F.mse_loss(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Scales should have changed
        assert not torch.allclose(ffn.scales, initial_scales, atol=1e-4), "Scales did not adapt"

    def test_convergence_speed(self):
        """Should converge reasonably fast on simple task."""
        ffn = create_gradient_truth_ffn(d_model=32, num_shapes=4)
        optimizer = torch.optim.Adam(ffn.parameters(), lr=0.01)
        
        # Simple task: learn identity
        x = torch.randn(32, 32)
        target = x
        
        for epoch in range(50):
            output, _ = ffn(x)
            loss = F.mse_loss(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if loss.item() < 0.01:
                break
        
        final_loss = F.mse_loss(ffn(x)[0], target).item()
        assert final_loss < 0.5, f"Did not converge: loss={final_loss}"


# =============================================================================
# 6. STRESS TESTS
# =============================================================================

class TestStress:
    """Stress tests for scale and performance."""

    def test_many_shapes_stable(self):
        """64 shapes should work stably."""
        ffn = create_gradient_truth_ffn(d_model=128, num_shapes=64)
        
        for _ in range(10):
            x = torch.randn(32, 128)
            output, routing = ffn(x)
            
            assert not output.isnan().any()
            assert not output.isinf().any()
            assert (routing.weights >= 0).all()

    def test_large_model_gradient_flow(self):
        """Large model should have gradient flow."""
        ffn = create_gradient_truth_ffn(d_model=256, num_shapes=32)
        
        x = torch.randn(8, 256, requires_grad=True)
        output, _ = ffn(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert ffn.router.weight.grad is not None
        assert ffn.scales.grad is not None

    def test_repeated_forward_stable(self):
        """1000 forward passes should be stable."""
        ffn = create_gradient_truth_ffn(d_model=64, num_shapes=8)
        ffn.eval()
        
        for i in range(1000):
            x = torch.randn(16, 64)
            output, _ = ffn(x)
            
            if i % 100 == 0:
                assert not output.isnan().any(), f"NaN at iteration {i}"
                assert not output.isinf().any(), f"Inf at iteration {i}"

    def test_long_training_stable(self):
        """500 training steps should be stable."""
        ffn = create_gradient_truth_ffn(d_model=32, num_shapes=8)
        optimizer = torch.optim.Adam(ffn.parameters(), lr=0.001)
        
        for step in range(500):
            x = torch.randn(16, 32)
            target = torch.randn(16, 32)
            
            output, _ = ffn(x)
            loss = F.mse_loss(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                assert not loss.isnan(), f"NaN loss at step {step}"
                assert not loss.isinf(), f"Inf loss at step {step}"


# =============================================================================
# 7. BLOCK TESTS
# =============================================================================

class TestGradientTruthBlock:
    """Tests for the transformer block variant."""

    def test_block_forward(self):
        """Basic forward pass."""
        shapes = PolynomialShapeBank.from_primitives(d_model=64, num_shapes=8)
        block = GradientTruthBlock(d_model=64, n_heads=4, shape_bank=shapes)
        
        x = torch.randn(2, 16, 64)
        output, routing = block(x)
        
        assert output.shape == x.shape

    def test_block_gradient_flow(self):
        """Gradients flow through block."""
        shapes = PolynomialShapeBank.from_primitives(d_model=64, num_shapes=8)
        block = GradientTruthBlock(d_model=64, n_heads=4, shape_bank=shapes)
        
        x = torch.randn(2, 16, 64, requires_grad=True)
        output, _ = block(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_block_causal_vs_bidirectional(self):
        """Causal and bidirectional should differ."""
        shapes = PolynomialShapeBank.from_primitives(d_model=64, num_shapes=8)
        block = GradientTruthBlock(d_model=64, n_heads=4, shape_bank=shapes)
        block.eval()
        
        x = torch.randn(2, 16, 64)
        output_causal, _ = block(x, is_causal=True)
        output_bidir, _ = block(x, is_causal=False)
        
        assert not torch.allclose(output_causal, output_bidir)

    def test_block_deterministic_eval(self):
        """Eval mode should be deterministic."""
        shapes = PolynomialShapeBank.from_primitives(d_model=64, num_shapes=8)
        block = GradientTruthBlock(d_model=64, n_heads=4, shape_bank=shapes)
        block.eval()
        
        x = torch.randn(2, 16, 64)
        out1, _ = block(x)
        out2, _ = block(x)
        
        assert torch.allclose(out1, out2)


# =============================================================================
# 8. SHAPE GENESIS TESTS
# =============================================================================

class TestShapeGenesisRigorous:
    """Rigorous tests for shape discovery."""

    def test_derived_shapes_work(self):
        """Derived shapes should produce valid outputs."""
        bank = ShapeGenesis.derive_boolean_shapes(d_model=32)
        ffn = GradientTruthFFN(d_model=32, shape_bank=bank)
        
        x = torch.randn(16, 32)
        output, _ = ffn(x)
        
        assert not output.isnan().any()
        assert not output.isinf().any()

    def test_evolved_shapes_work(self):
        """Evolved shapes should produce valid outputs."""
        def fitness(fn):
            x = torch.randn(8, 16)
            try:
                out = fn(x)
                return out.abs().mean().item()
            except:
                return 0.0
        
        bank = ShapeGenesis.evolve_shapes(
            d_model=16,
            num_shapes=4,
            fitness_fn=fitness,
            generations=10,
            population=20,
        )
        
        ffn = GradientTruthFFN(d_model=16, shape_bank=bank)
        x = torch.randn(8, 16)
        output, _ = ffn(x)
        
        assert not output.isnan().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
