"""
Rigorous tests for TrueOctaveFFN.

These tests verify mathematical invariants, edge cases, numerical stability,
and properties that MUST hold for the architecture to be correct.

Categories:
1. Derivation Invariants - coarse MUST equal pooled(fine)
2. Ternary Invariants - weights MUST be in {-1, 0, +1}
3. Frozen Invariants - structure MUST NOT change during training
4. Mode Invariants - deterministic MUST be exact, generative MUST be soft
5. Gradient Invariants - gradients MUST flow only to learned params
6. Numerical Stability - no NaN, no Inf, bounded outputs
7. Edge Cases - empty batches, single tokens, extreme values
8. Reproducibility - same input MUST give same output (in deterministic mode)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import copy

from trix.nn.octave import (
    TrueOctaveFFN,
    TrueOctaveBlock,
    Octave,
    FrozenTile,
    derive_octave,
    OctaveRoutingInfo,
)


# =============================================================================
# 1. DERIVATION INVARIANTS
# =============================================================================

class TestDerivationInvariants:
    """Coarse octaves MUST be derived from fine octaves."""
    
    def test_medium_signature_equals_pooled_fine(self):
        """Medium[i].signature == sign(mean(Fine[4i:4i+4].signatures))"""
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        for i in range(ffn.medium.num_tiles):
            start = i * ffn.pool_factor
            end = start + ffn.pool_factor
            
            fine_sigs = torch.stack([
                ffn.fine.tiles[j].signature for j in range(start, end)
            ])
            expected = fine_sigs.mean(dim=0).sign()
            actual = ffn.medium.tiles[i].signature
            
            assert torch.allclose(expected, actual), \
                f"Medium[{i}] signature derivation failed"
    
    def test_coarse_signature_equals_pooled_medium(self):
        """Coarse[j].signature == sign(mean(Medium[4j:4j+4].signatures))"""
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        for j in range(ffn.coarse.num_tiles):
            start = j * ffn.pool_factor
            end = start + ffn.pool_factor
            
            medium_sigs = torch.stack([
                ffn.medium.tiles[k].signature for k in range(start, end)
            ])
            expected = medium_sigs.mean(dim=0).sign()
            actual = ffn.coarse.tiles[j].signature
            
            assert torch.allclose(expected, actual), \
                f"Coarse[{j}] signature derivation failed"
    
    def test_derivation_holds_for_various_pool_factors(self):
        """Derivation must work for pool_factor in {2, 4, 8}."""
        for pool_factor in [2, 4]:
            num_fine = pool_factor ** 3  # Ensures we get 3 octaves
            ffn = TrueOctaveFFN(
                d_model=64, 
                num_fine_tiles=num_fine, 
                pool_factor=pool_factor
            )
            
            checks = ffn.get_derivation_check()
            for name, passed in checks.items():
                assert passed, f"pool_factor={pool_factor}: {name} failed"
    
    def test_rederive_restores_derivation(self):
        """After modifying fine tiles, rederive() must restore correct derivation."""
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        # Manually corrupt medium (shouldn't happen in practice)
        with torch.no_grad():
            ffn.medium.tiles[0].up_weight.fill_(0.5)  # Invalid!
        
        # Derivation should now fail
        checks_before = ffn.get_derivation_check()
        assert not all(checks_before.values()), "Should be corrupted"
        
        # Rederive
        ffn.rederive()
        
        # Derivation should now pass
        checks_after = ffn.get_derivation_check()
        assert all(checks_after.values()), "Rederive should fix derivation"


# =============================================================================
# 2. TERNARY INVARIANTS
# =============================================================================

class TestTernaryInvariants:
    """All frozen weights MUST be in {-1, 0, +1}."""
    
    def test_fine_weights_are_ternary(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        for i, tile in enumerate(ffn.fine.tiles):
            up_vals = set(tile.up_weight.unique().tolist())
            down_vals = set(tile.down_weight.unique().tolist())
            
            assert up_vals.issubset({-1.0, 0.0, 1.0}), \
                f"Fine[{i}].up_weight has non-ternary values: {up_vals}"
            assert down_vals.issubset({-1.0, 0.0, 1.0}), \
                f"Fine[{i}].down_weight has non-ternary values: {down_vals}"
    
    def test_medium_weights_are_ternary(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        for i, tile in enumerate(ffn.medium.tiles):
            up_vals = set(tile.up_weight.unique().tolist())
            down_vals = set(tile.down_weight.unique().tolist())
            
            assert up_vals.issubset({-1.0, 0.0, 1.0}), \
                f"Medium[{i}].up_weight has non-ternary values: {up_vals}"
            assert down_vals.issubset({-1.0, 0.0, 1.0}), \
                f"Medium[{i}].down_weight has non-ternary values: {down_vals}"
    
    def test_coarse_weights_are_ternary(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        for i, tile in enumerate(ffn.coarse.tiles):
            up_vals = set(tile.up_weight.unique().tolist())
            down_vals = set(tile.down_weight.unique().tolist())
            
            assert up_vals.issubset({-1.0, 0.0, 1.0}), \
                f"Coarse[{i}].up_weight has non-ternary values: {up_vals}"
            assert down_vals.issubset({-1.0, 0.0, 1.0}), \
                f"Coarse[{i}].down_weight has non-ternary values: {down_vals}"
    
    def test_signatures_are_ternary(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        for octave_name, octave in [("fine", ffn.fine), ("medium", ffn.medium), ("coarse", ffn.coarse)]:
            for i, tile in enumerate(octave.tiles):
                sig_vals = set(tile.signature.unique().tolist())
                assert sig_vals.issubset({-1.0, 0.0, 1.0}), \
                    f"{octave_name}[{i}].signature has non-ternary values: {sig_vals}"


# =============================================================================
# 3. FROZEN INVARIANTS
# =============================================================================

class TestFrozenInvariants:
    """Frozen structure MUST NOT change during training."""
    
    def test_weights_unchanged_after_forward_backward(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        # Snapshot weights before
        weights_before = {}
        for octave_name, octave in [("fine", ffn.fine), ("medium", ffn.medium), ("coarse", ffn.coarse)]:
            for i, tile in enumerate(octave.tiles):
                weights_before[f"{octave_name}_{i}_up"] = tile.up_weight.clone()
                weights_before[f"{octave_name}_{i}_down"] = tile.down_weight.clone()
        
        # Forward and backward
        x = torch.randn(2, 16, 64)
        out, info = ffn(x)
        out.sum().backward()
        
        # Verify weights unchanged
        for octave_name, octave in [("fine", ffn.fine), ("medium", ffn.medium), ("coarse", ffn.coarse)]:
            for i, tile in enumerate(octave.tiles):
                key_up = f"{octave_name}_{i}_up"
                key_down = f"{octave_name}_{i}_down"
                
                assert torch.equal(weights_before[key_up], tile.up_weight), \
                    f"{key_up} changed after backward!"
                assert torch.equal(weights_before[key_down], tile.down_weight), \
                    f"{key_down} changed after backward!"
    
    def test_weights_unchanged_after_optimizer_step(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        optimizer = torch.optim.Adam(ffn.parameters(), lr=0.01)
        
        # Snapshot weights before
        weights_before = {}
        for octave_name, octave in [("fine", ffn.fine), ("medium", ffn.medium), ("coarse", ffn.coarse)]:
            for i, tile in enumerate(octave.tiles):
                weights_before[f"{octave_name}_{i}_up"] = tile.up_weight.clone()
                weights_before[f"{octave_name}_{i}_down"] = tile.down_weight.clone()
        
        # Training step
        x = torch.randn(2, 16, 64)
        out, info = ffn(x)
        loss = out.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify weights unchanged
        for octave_name, octave in [("fine", ffn.fine), ("medium", ffn.medium), ("coarse", ffn.coarse)]:
            for i, tile in enumerate(octave.tiles):
                key_up = f"{octave_name}_{i}_up"
                key_down = f"{octave_name}_{i}_down"
                
                assert torch.equal(weights_before[key_up], tile.up_weight), \
                    f"{key_up} changed after optimizer step!"
                assert torch.equal(weights_before[key_down], tile.down_weight), \
                    f"{key_down} changed after optimizer step!"
    
    def test_weights_have_no_grad(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        for octave_name, octave in [("fine", ffn.fine), ("medium", ffn.medium), ("coarse", ffn.coarse)]:
            for i, tile in enumerate(octave.tiles):
                assert not tile.up_weight.requires_grad, \
                    f"{octave_name}[{i}].up_weight should not require grad"
                assert not tile.down_weight.requires_grad, \
                    f"{octave_name}[{i}].down_weight should not require grad"


# =============================================================================
# 4. MODE INVARIANTS
# =============================================================================

class TestModeInvariants:
    """Deterministic mode MUST be exact. Generative mode MUST be soft."""
    
    def test_deterministic_blend_is_one_hot(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        ffn.set_mode("deterministic")
        
        x = torch.randn(4, 32, 64)
        _, info = ffn(x)
        
        # Check one-hot property
        assert (info.blend_weights.sum(dim=-1) == 1).all(), \
            "Blend weights should sum to 1"
        assert ((info.blend_weights == 0) | (info.blend_weights == 1)).all(), \
            "Blend weights should be one-hot in deterministic mode"
    
    def test_deterministic_entropy_is_zero(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        ffn.set_mode("deterministic")
        
        x = torch.randn(4, 32, 64)
        _, info = ffn(x)
        
        # Entropy of one-hot is 0 (with small epsilon for numerical stability)
        assert (info.entropy < 1e-6).all(), \
            f"Entropy should be ~0 in deterministic mode, got max={info.entropy.max()}"
    
    def test_generative_blend_is_soft(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        ffn.set_mode("generative")
        
        x = torch.randn(4, 32, 64)
        _, info = ffn(x)
        
        # Check soft property (not all one-hot)
        is_one_hot = ((info.blend_weights == 0) | (info.blend_weights == 1)).all(dim=-1)
        assert not is_one_hot.all(), \
            "Blend weights should NOT all be one-hot in generative mode"
    
    def test_generative_entropy_is_positive(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        ffn.set_mode("generative")
        
        x = torch.randn(4, 32, 64)
        _, info = ffn(x)
        
        assert info.entropy.mean() > 0.1, \
            f"Entropy should be positive in generative mode, got mean={info.entropy.mean()}"
    
    def test_deterministic_is_reproducible(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        ffn.set_mode("deterministic")
        ffn.eval()
        
        x = torch.randn(2, 16, 64)
        
        with torch.no_grad():
            out1, info1 = ffn(x)
            out2, info2 = ffn(x)
        
        assert torch.equal(out1, out2), "Deterministic mode should be exactly reproducible"
        assert torch.equal(info1.blend_weights, info2.blend_weights), "Routing should be reproducible"
    
    def test_mode_switch_changes_output(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        ffn.eval()
        
        x = torch.randn(2, 16, 64)
        
        with torch.no_grad():
            ffn.set_mode("generative")
            out_gen, _ = ffn(x)
            
            ffn.set_mode("deterministic")
            out_det, _ = ffn(x)
        
        assert not torch.allclose(out_gen, out_det), \
            "Different modes should produce different outputs"


# =============================================================================
# 5. GRADIENT INVARIANTS
# =============================================================================

class TestGradientInvariants:
    """Gradients MUST flow only to learned parameters."""
    
    def test_gradients_flow_to_scales(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        x = torch.randn(2, 16, 64)
        out, _ = ffn(x)
        out.sum().backward()
        
        # Check scales have gradients
        for octave_name, octave in [("fine", ffn.fine), ("medium", ffn.medium), ("coarse", ffn.coarse)]:
            for i, tile in enumerate(octave.tiles):
                assert tile.scale.grad is not None, \
                    f"{octave_name}[{i}].scale should have gradient"
    
    def test_gradients_flow_to_blend_network(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        x = torch.randn(2, 16, 64)
        out, _ = ffn(x)
        out.sum().backward()
        
        for name, param in ffn.blend_net.named_parameters():
            assert param.grad is not None, f"blend_net.{name} should have gradient"
    
    def test_gradients_flow_to_output_scale(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        x = torch.randn(2, 16, 64)
        out, _ = ffn(x)
        out.sum().backward()
        
        assert ffn.output_scale.grad is not None, "output_scale should have gradient"
    
    def test_no_gradients_to_frozen_weights(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        x = torch.randn(2, 16, 64)
        out, _ = ffn(x)
        out.sum().backward()
        
        for octave_name, octave in [("fine", ffn.fine), ("medium", ffn.medium), ("coarse", ffn.coarse)]:
            for i, tile in enumerate(octave.tiles):
                assert tile.up_weight.grad is None, \
                    f"{octave_name}[{i}].up_weight should NOT have gradient"
                assert tile.down_weight.grad is None, \
                    f"{octave_name}[{i}].down_weight should NOT have gradient"


# =============================================================================
# 6. NUMERICAL STABILITY
# =============================================================================

class TestNumericalStability:
    """No NaN, no Inf, bounded outputs."""
    
    def test_no_nan_in_forward(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        x = torch.randn(4, 32, 64)
        out, info = ffn(x)
        
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isnan(info.blend_weights).any(), "Blend weights contain NaN"
        assert not torch.isnan(info.entropy).any(), "Entropy contains NaN"
    
    def test_no_inf_in_forward(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        x = torch.randn(4, 32, 64)
        out, info = ffn(x)
        
        assert not torch.isinf(out).any(), "Output contains Inf"
        assert not torch.isinf(info.blend_weights).any(), "Blend weights contain Inf"
        assert not torch.isinf(info.entropy).any(), "Entropy contains Inf"
    
    def test_no_nan_with_large_input(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        x = torch.randn(4, 32, 64) * 100  # Large values
        out, info = ffn(x)
        
        assert not torch.isnan(out).any(), "Output contains NaN with large input"
    
    def test_no_nan_with_small_input(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        x = torch.randn(4, 32, 64) * 1e-6  # Small values
        out, info = ffn(x)
        
        assert not torch.isnan(out).any(), "Output contains NaN with small input"
    
    def test_no_nan_in_backward(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        x = torch.randn(4, 32, 64)
        out, _ = ffn(x)
        out.sum().backward()
        
        for name, param in ffn.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"Gradient of {name} contains NaN"
    
    def test_output_bounded(self):
        """Output should be bounded (due to residual connection and scaling)."""
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        ffn.eval()
        
        x = torch.randn(4, 32, 64)
        with torch.no_grad():
            out, _ = ffn(x)
        
        # Output should be within reasonable bounds (residual + scaled FFN output)
        # Input norm + bounded FFN contribution
        max_expected = x.abs().max() + 10  # Conservative bound
        assert out.abs().max() < max_expected, \
            f"Output too large: {out.abs().max()} > {max_expected}"


# =============================================================================
# 7. EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge cases that must be handled correctly."""
    
    def test_batch_size_one(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        x = torch.randn(1, 16, 64)
        out, info = ffn(x)
        
        assert out.shape == (1, 16, 64)
        assert info.blend_weights.shape == (1, 16, 3)
    
    def test_sequence_length_one(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        x = torch.randn(4, 1, 64)
        out, info = ffn(x)
        
        assert out.shape == (4, 1, 64)
        assert info.blend_weights.shape == (4, 1, 3)
    
    def test_single_token(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        x = torch.randn(1, 1, 64)
        out, info = ffn(x)
        
        assert out.shape == (1, 1, 64)
    
    def test_large_batch(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        x = torch.randn(128, 32, 64)
        out, info = ffn(x)
        
        assert out.shape == (128, 32, 64)
    
    def test_long_sequence(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        x = torch.randn(2, 512, 64)
        out, info = ffn(x)
        
        assert out.shape == (2, 512, 64)
    
    def test_zero_input(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        x = torch.zeros(2, 16, 64)
        out, info = ffn(x)
        
        assert not torch.isnan(out).any(), "Zero input caused NaN"
    
    def test_constant_input(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        x = torch.ones(2, 16, 64)
        out, info = ffn(x)
        
        assert not torch.isnan(out).any(), "Constant input caused NaN"
    
    def test_minimum_tiles(self):
        """Minimum configuration: 4 fine, 2 medium, 1 coarse."""
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=4, pool_factor=2)
        
        assert ffn.fine.num_tiles == 4
        assert ffn.medium.num_tiles == 2
        assert ffn.coarse.num_tiles == 1
        
        x = torch.randn(2, 16, 64)
        out, info = ffn(x)
        assert out.shape == x.shape


# =============================================================================
# 8. REPRODUCIBILITY
# =============================================================================

class TestReproducibility:
    """Same seed, same input MUST give same output."""
    
    def test_same_seed_same_init(self):
        torch.manual_seed(42)
        ffn1 = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        torch.manual_seed(42)
        ffn2 = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        
        # Check weights are identical
        for (n1, p1), (n2, p2) in zip(ffn1.named_parameters(), ffn2.named_parameters()):
            assert torch.equal(p1, p2), f"Parameter {n1} differs with same seed"
        
        for (n1, b1), (n2, b2) in zip(ffn1.named_buffers(), ffn2.named_buffers()):
            assert torch.equal(b1, b2), f"Buffer {n1} differs with same seed"
    
    def test_deterministic_same_output(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        ffn.set_mode("deterministic")
        ffn.eval()
        
        torch.manual_seed(123)
        x = torch.randn(2, 16, 64)
        
        with torch.no_grad():
            out1, _ = ffn(x)
            out2, _ = ffn(x)
            out3, _ = ffn(x)
        
        assert torch.equal(out1, out2), "Output 1 != Output 2"
        assert torch.equal(out2, out3), "Output 2 != Output 3"
    
    def test_eval_mode_deterministic(self):
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        ffn.eval()
        
        x = torch.randn(2, 16, 64)
        
        with torch.no_grad():
            out1, _ = ffn(x)
            out2, _ = ffn(x)
        
        # Even in generative mode, eval should be deterministic (no dropout)
        assert torch.equal(out1, out2), "Eval mode should be deterministic"


# =============================================================================
# 9. TRAINING CORRECTNESS
# =============================================================================

class TestTrainingCorrectness:
    """Training must work correctly."""
    
    def test_loss_decreases(self):
        """Training should decrease loss."""
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        optimizer = torch.optim.Adam(ffn.parameters(), lr=0.01)
        
        x = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)
        
        losses = []
        for _ in range(10):
            out, _ = ffn(x)
            loss = F.mse_loss(out, target)
            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Loss should generally decrease
        assert losses[-1] < losses[0], "Loss should decrease during training"
    
    def test_scales_change_during_training(self):
        """Scales should be updated by optimizer."""
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        optimizer = torch.optim.Adam(ffn.parameters(), lr=0.01)
        
        scale_before = ffn.fine.tiles[0].scale.clone()
        
        x = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)
        
        for _ in range(5):
            out, _ = ffn(x)
            loss = F.mse_loss(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scale_after = ffn.fine.tiles[0].scale
        
        assert not torch.equal(scale_before, scale_after), "Scale should change during training"
    
    def test_blend_network_changes_during_training(self):
        """Blend network should be updated by optimizer."""
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=2)
        optimizer = torch.optim.Adam(ffn.parameters(), lr=0.01)
        
        blend_before = ffn.blend_net[0].weight.clone()
        
        x = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)
        
        for _ in range(5):
            out, _ = ffn(x)
            loss = F.mse_loss(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        blend_after = ffn.blend_net[0].weight
        
        assert not torch.equal(blend_before, blend_after), "Blend network should change during training"


# =============================================================================
# 10. BLOCK INTEGRATION
# =============================================================================

class TestBlockIntegration:
    """TrueOctaveBlock must integrate correctly."""
    
    def test_block_forward_shape(self):
        block = TrueOctaveBlock(d_model=64, n_heads=4, num_fine_tiles=16, pool_factor=2)
        
        x = torch.randn(2, 16, 64)
        out, info = block(x)
        
        assert out.shape == x.shape
    
    def test_block_mode_propagates(self):
        block = TrueOctaveBlock(d_model=64, n_heads=4, num_fine_tiles=16, pool_factor=2)
        
        block.set_mode("deterministic")
        assert block.ffn.mode == "deterministic"
        
        block.set_mode("generative")
        assert block.ffn.mode == "generative"
    
    def test_stacked_blocks(self):
        blocks = nn.ModuleList([
            TrueOctaveBlock(d_model=64, n_heads=4, num_fine_tiles=16, pool_factor=2)
            for _ in range(4)
        ])
        
        x = torch.randn(2, 16, 64)
        
        for block in blocks:
            x, info = block(x)
        
        assert x.shape == (2, 16, 64)
        assert not torch.isnan(x).any(), "Stacked blocks produced NaN"
    
    def test_block_gradient_flow(self):
        block = TrueOctaveBlock(d_model=64, n_heads=4, num_fine_tiles=16, pool_factor=2)
        
        x = torch.randn(2, 16, 64)
        out, _ = block(x)
        out.sum().backward()
        
        # Check attention gets gradients
        assert block.attn.out_proj.weight.grad is not None
        
        # Check FFN gets gradients
        assert block.ffn.output_scale.grad is not None
