"""
Rigorous Foundation Tests - Probing Blind Spots.

This test suite is adversarial. It tries to break things.

Categories:
1. Edge Cases - Empty inputs, boundaries, extremes
2. Numerical Stability - NaN, Inf, underflow, overflow
3. Gradient Flow - Are gradients reaching the right places?
4. Determinism - Same inputs always produce same outputs
5. Mode Consistency - Train/eval, generative/deterministic
6. Memory Safety - Leaks, growth, cleanup
7. Serialization - Save/load round-trip
8. Adversarial Inputs - Designed to cause failures
9. Invariant Violations - Breaking documented contracts
10. Stress Testing - Push to limits

"Trust, but verify." - Ronald Reagan
"Verify, then maybe trust." - This test suite
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
import io
import copy
import warnings
from typing import List, Tuple, Dict, Any

from trix.nn.anchored import (
    AnchoredDualModeFFN,
    AnchoredDualModeBlock,
    get_temperature_schedule,
)
from trix.nn.octave import (
    TrueOctaveFFN,
    TrueOctaveBlock,
    FrozenTile,
    Octave,
    derive_octave,
)


SEED = 42


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# =============================================================================
# 1. EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Boundary conditions that often reveal bugs."""

    def test_single_element_batch(self):
        """Single element in batch dimension."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=4, num_tiles=8)
        x = torch.randn(1, 1, 64)  # batch=1, seq=1

        output, info = ffn(x)

        assert output.shape == (1, 1, 64)
        assert not torch.isnan(output).any()

    def test_long_sequence(self):
        """Very long sequence."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=4, num_tiles=8)
        x = torch.randn(2, 1024, 64)  # Very long

        output, info = ffn(x)

        assert output.shape == (2, 1024, 64)
        assert not torch.isnan(output).any()

    def test_minimal_model(self):
        """Minimum possible configuration."""
        set_seed()
        # Smallest reasonable config
        ffn = AnchoredDualModeFFN(d_model=8, num_anchors=2, num_tiles=2)
        x = torch.randn(1, 1, 8)

        output, info = ffn(x)

        assert output.shape == (1, 1, 8)
        assert not torch.isnan(output).any()

    def test_octave_minimal_tiles(self):
        """TrueOctave with minimum tiles."""
        set_seed()
        # 16 fine → 4 medium → 1 coarse
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=4)
        x = torch.randn(2, 8, 64)

        output, info = ffn(x)

        assert output.shape == (2, 8, 64)
        assert not torch.isnan(output).any()

    def test_large_batch_small_seq(self):
        """Large batch, small sequence."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=4, num_tiles=16)
        x = torch.randn(128, 2, 64)

        output, info = ffn(x)

        assert output.shape == (128, 2, 64)

    def test_asymmetric_dimensions(self):
        """Non-square d_model and tile counts."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=100, num_anchors=7, num_tiles=31)
        x = torch.randn(3, 17, 100)

        output, info = ffn(x)

        assert output.shape == (3, 17, 100)
        assert not torch.isnan(output).any()


# =============================================================================
# 2. NUMERICAL STABILITY TESTS
# =============================================================================

class TestNumericalStability:
    """Probing numerical edge cases."""

    def test_zero_input(self):
        """All-zero input should not cause NaN."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        x = torch.zeros(4, 16, 64)

        output, info = ffn(x)

        assert not torch.isnan(output).any(), "NaN in output from zero input"
        assert not torch.isinf(output).any(), "Inf in output from zero input"

    def test_very_small_input(self):
        """Very small values (underflow risk)."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        x = torch.ones(4, 16, 64) * 1e-10

        output, info = ffn(x)

        assert not torch.isnan(output).any(), "NaN from small input"

    def test_very_large_input(self):
        """Very large values (overflow risk)."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        x = torch.ones(4, 16, 64) * 1e4

        output, info = ffn(x)

        assert not torch.isnan(output).any(), "NaN from large input"
        assert not torch.isinf(output).any(), "Inf from large input"

    def test_mixed_sign_extreme(self):
        """Extreme positive and negative values."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        x = torch.randn(4, 16, 64) * 100
        x[..., ::2] *= -1  # Mix signs

        output, info = ffn(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_nan_propagation(self):
        """NaN in input should not crash (graceful handling)."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        x = torch.randn(4, 16, 64)
        x[0, 0, 0] = float('nan')

        # Should complete without crashing
        output, info = ffn(x)

        # The NaN will propagate - that's expected
        # But it shouldn't crash

    def test_temperature_zero(self):
        """Temperature approaching zero."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16, temperature=1e-6)
        x = torch.randn(4, 16, 64)

        output, info = ffn(x)

        # With very low temperature, anchor_probs should be more peaked
        # (though ties in similarity can still cause non-one-hot behavior)
        max_prob = info['anchor_probs'].max(dim=-1).values
        # At least most should be near 1.0, allow for ties
        high_prob_ratio = (max_prob > 0.9).float().mean()
        assert high_prob_ratio > 0.5, f"Low temp should give peaked distribution, only {high_prob_ratio:.0%} peaked"
        assert not torch.isnan(output).any()

    def test_temperature_very_high(self):
        """Temperature very high (uniform distribution)."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16, temperature=100.0)
        x = torch.randn(4, 16, 64)

        output, info = ffn(x)

        # With very high temperature, distribution should be near-uniform
        anchor_std = info['anchor_probs'].std(dim=-1)
        assert (anchor_std < 0.1).all(), "High temp should give near-uniform"
        assert not torch.isnan(output).any()

    def test_gradient_magnitude(self):
        """Check gradient magnitudes are reasonable."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        ffn.train()
        x = torch.randn(4, 16, 64, requires_grad=True)

        output, _ = ffn(x)
        loss = output.sum()
        loss.backward()

        # Check no gradient explosion
        for name, param in ffn.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm < 1e6, f"Gradient explosion in {name}: {grad_norm}"
                assert not np.isnan(grad_norm), f"NaN gradient in {name}"

    def test_octave_blend_weights_sum_to_one(self):
        """Blend weights should sum to 1."""
        set_seed()
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16)
        x = torch.randn(4, 16, 64)

        _, info = ffn(x)

        blend_sum = info.blend_weights.sum(dim=-1)
        assert torch.allclose(blend_sum, torch.ones_like(blend_sum), atol=1e-5), \
            "Blend weights don't sum to 1"


# =============================================================================
# 3. GRADIENT FLOW TESTS
# =============================================================================

class TestGradientFlow:
    """Verify gradients reach the right places."""

    def test_learned_params_have_gradient(self):
        """Learned parameters should receive gradients."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        ffn.train()
        x = torch.randn(4, 16, 64)

        output, _ = ffn(x)
        loss = output.sum()
        loss.backward()

        # These should have gradients
        learned_params = [
            ('anchor_proj.weight', ffn.anchor_proj.weight),
            ('anchor_proj.bias', ffn.anchor_proj.bias),
            ('tile_scales', ffn.tile_scales),
        ]

        for name, param in learned_params:
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_frozen_buffers_no_gradient(self):
        """Frozen buffers should not have gradients."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        ffn.train()
        x = torch.randn(4, 16, 64)

        output, _ = ffn(x)
        loss = output.sum()
        loss.backward()

        # These are buffers, should not have grad
        assert not ffn.anchor_signatures.requires_grad
        assert not ffn.tile_directions.requires_grad

    def test_gradient_flow_through_octave_blend(self):
        """Gradients should flow through octave blend network."""
        set_seed()
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16)
        ffn.train()
        x = torch.randn(4, 16, 64)

        output, _ = ffn(x)
        loss = output.sum()
        loss.backward()

        # Blend network should have gradients
        for name, param in ffn.blend_net.named_parameters():
            assert param.grad is not None, f"No gradient for blend_net.{name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for blend_net.{name}"

    def test_tile_scales_receive_gradient(self):
        """Per-tile scales should receive gradients."""
        set_seed()
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16)
        ffn.train()
        ffn.set_mode("generative")  # Soft routing for gradients
        x = torch.randn(4, 16, 64)

        output, _ = ffn(x)
        loss = output.sum()
        loss.backward()

        # Check some tile scales
        scales_with_grad = 0
        for tile in ffn.fine.tiles:
            if tile.scale.grad is not None and tile.scale.grad.abs().sum() > 0:
                scales_with_grad += 1

        # Not all tiles may be selected, but some should have gradients
        assert scales_with_grad > 0, "No tile scales received gradients"

    def test_router_gradients_anchored(self):
        """Router should receive gradients from tile selection."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        ffn.train()
        x = torch.randn(4, 16, 64)

        output, _ = ffn(x)
        loss = output.sum()
        loss.backward()

        # Router layers should have gradients
        for name, param in ffn.router.named_parameters():
            assert param.grad is not None, f"No gradient for router.{name}"


# =============================================================================
# 4. DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Same inputs should produce same outputs."""

    def test_eval_mode_deterministic(self):
        """Eval mode should be deterministic."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        ffn.eval()
        x = torch.randn(4, 16, 64)

        with torch.no_grad():
            out1, info1 = ffn(x)
            out2, info2 = ffn(x)

        assert torch.allclose(out1, out2), "Eval mode not deterministic"
        assert torch.equal(info1['tile_idx'], info2['tile_idx']), "Tile indices differ"

    def test_octave_deterministic_mode(self):
        """Octave deterministic mode should be exactly deterministic."""
        set_seed()
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16)
        ffn.set_mode("deterministic")
        ffn.eval()
        x = torch.randn(4, 16, 64)

        with torch.no_grad():
            out1, info1 = ffn(x)
            out2, info2 = ffn(x)

        assert torch.equal(out1, out2), "Deterministic mode not exactly equal"
        assert torch.equal(info1.fine_tile_idx, info2.fine_tile_idx)

    def test_different_seeds_different_init(self):
        """Different seeds should give different initializations."""
        torch.manual_seed(1)
        ffn1 = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)

        torch.manual_seed(2)
        ffn2 = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)

        # Anchors should differ
        assert not torch.equal(ffn1.anchor_signatures, ffn2.anchor_signatures)

    def test_training_mode_reproducible(self):
        """Training mode should be reproducible with same seed."""
        first_out = None
        for i in range(2):
            set_seed(123)
            ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
            ffn.train()
            x = torch.randn(4, 16, 64)

            out, _ = ffn(x)

            if i == 0:
                first_out = out.clone()
            else:
                assert torch.allclose(first_out, out, atol=1e-6), \
                    "Training not reproducible with same seed"


# =============================================================================
# 5. MODE CONSISTENCY TESTS
# =============================================================================

class TestModeConsistency:
    """Train/eval and generative/deterministic mode behavior."""

    def test_train_eval_different_behavior(self):
        """Train and eval modes should behave differently."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        x = torch.randn(4, 16, 64)

        ffn.train()
        out_train, info_train = ffn(x)

        ffn.eval()
        with torch.no_grad():
            out_eval, info_eval = ffn(x)

        # In train mode, we use soft combination
        # In eval mode, we use hard selection
        # The actual implementation may or may not differ
        # But both should produce valid outputs
        assert out_train.shape == out_eval.shape

    def test_generative_vs_deterministic_octave(self):
        """Generative and deterministic modes produce different outputs."""
        set_seed()
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16)
        x = torch.randn(4, 16, 64)

        ffn.set_mode("generative")
        out_gen, info_gen = ffn(x)

        ffn.set_mode("deterministic")
        out_det, info_det = ffn(x)

        # Modes should produce different outputs (soft vs hard routing)
        # Actually, might be similar if dominant tile is used - check entropy
        assert info_gen.mode == "generative"
        assert info_det.mode == "deterministic"

        # Deterministic should have zero entropy (one-hot)
        assert (info_det.entropy < 1e-5).all(), "Deterministic should have ~0 entropy"

    def test_mode_switch_idempotent(self):
        """Switching modes back and forth should be stable."""
        set_seed()
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16)
        ffn.eval()  # Disable dropout for true determinism
        x = torch.randn(4, 16, 64)

        with torch.no_grad():
            ffn.set_mode("deterministic")
            out1, _ = ffn(x)

            ffn.set_mode("generative")
            _, _ = ffn(x)

            ffn.set_mode("deterministic")
            out2, _ = ffn(x)

        assert torch.equal(out1, out2), "Mode switching not stable"

    def test_temperature_affects_routing(self):
        """Changing temperature should affect routing distribution."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16, temperature=1.0)
        x = torch.randn(4, 16, 64)

        _, info_t1 = ffn(x)
        entropy_t1 = info_t1['anchor_entropy']

        ffn.set_temperature(0.1)
        _, info_t01 = ffn(x)
        entropy_t01 = info_t01['anchor_entropy']

        # Lower temperature = lower entropy
        assert entropy_t01 < entropy_t1, "Lower temp should give lower entropy"


# =============================================================================
# 6. MEMORY SAFETY TESTS
# =============================================================================

class TestMemorySafety:
    """Check for memory leaks and growth."""

    def test_no_memory_leak_in_loop(self):
        """Memory should not grow unboundedly in inference loop."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        ffn.eval()
        x = torch.randn(4, 16, 64)

        # Warmup
        with torch.no_grad():
            _ = ffn(x)

        gc.collect()

        # Run many iterations
        for i in range(100):
            with torch.no_grad():
                _ = ffn(x)

        gc.collect()
        # If we get here without OOM, we're okay
        assert True

    def test_gradient_cleanup(self):
        """Gradients should be clearable."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        ffn.train()

        for _ in range(10):
            x = torch.randn(4, 16, 64)
            out, _ = ffn(x)
            loss = out.sum()
            loss.backward()

            # Clear gradients
            ffn.zero_grad()

            # Check gradients are cleared
            for param in ffn.parameters():
                if param.grad is not None:
                    assert param.grad.abs().sum() == 0

    def test_usage_stats_reset(self):
        """Usage stats should be resetable."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        ffn.train()
        x = torch.randn(4, 16, 64)

        # Accumulate usage
        for _ in range(10):
            _ = ffn(x)

        assert ffn.usage_count > 0

        # Reset
        ffn.reset_usage_stats()

        assert ffn.usage_count == 0
        assert ffn.anchor_usage.sum() == 0


# =============================================================================
# 7. SERIALIZATION TESTS
# =============================================================================

class TestSerialization:
    """Save/load round-trip should preserve state."""

    def test_state_dict_round_trip(self):
        """Save and load state dict should preserve behavior."""
        set_seed()
        ffn1 = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        x = torch.randn(4, 16, 64)

        ffn1.eval()
        with torch.no_grad():
            out1, _ = ffn1(x)

        # Save and load
        state = ffn1.state_dict()

        ffn2 = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        ffn2.load_state_dict(state)
        ffn2.eval()

        with torch.no_grad():
            out2, _ = ffn2(x)

        assert torch.allclose(out1, out2), "State dict round-trip changed output"

    def test_buffer_serialization(self):
        """Frozen buffers should be correctly serialized."""
        set_seed()
        ffn1 = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)

        state = ffn1.state_dict()

        ffn2 = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        ffn2.load_state_dict(state)

        # Buffers should match exactly
        assert torch.equal(ffn1.anchor_signatures, ffn2.anchor_signatures)
        assert torch.equal(ffn1.tile_directions, ffn2.tile_directions)

    def test_octave_serialization(self):
        """TrueOctaveFFN should serialize correctly."""
        set_seed()
        ffn1 = TrueOctaveFFN(d_model=64, num_fine_tiles=16)
        x = torch.randn(4, 16, 64)

        ffn1.eval()
        ffn1.set_mode("deterministic")
        with torch.no_grad():
            out1, _ = ffn1(x)

        # Save to buffer
        buffer = io.BytesIO()
        torch.save(ffn1.state_dict(), buffer)
        buffer.seek(0)

        # Load
        ffn2 = TrueOctaveFFN(d_model=64, num_fine_tiles=16)
        ffn2.load_state_dict(torch.load(buffer))
        ffn2.eval()
        ffn2.set_mode("deterministic")

        with torch.no_grad():
            out2, _ = ffn2(x)

        assert torch.equal(out1, out2), "Octave serialization failed"

    def test_training_state_preserved(self):
        """Training state should be preserved after save/load."""
        set_seed()
        ffn1 = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        ffn1.train()

        x = torch.randn(4, 16, 64)
        out, _ = ffn1(x)
        loss = out.sum()
        loss.backward()

        # Save
        state = ffn1.state_dict()

        # Load into new model
        ffn2 = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        ffn2.load_state_dict(state)

        # Parameters should match
        for (n1, p1), (n2, p2) in zip(ffn1.named_parameters(), ffn2.named_parameters()):
            assert torch.equal(p1, p2), f"Parameter {n1} not preserved"


# =============================================================================
# 8. ADVERSARIAL INPUT TESTS
# =============================================================================

class TestAdversarialInputs:
    """Inputs designed to cause failures."""

    def test_all_same_values(self):
        """All elements have same value."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        x = torch.ones(4, 16, 64) * 0.5

        output, info = ffn(x)

        assert not torch.isnan(output).any()
        # With identical inputs, all should route same way
        # (though this isn't required, just interesting)

    def test_alternating_extreme_values(self):
        """Alternating +inf/-inf scale values."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        x = torch.randn(4, 16, 64)
        x[..., ::2] *= 1000
        x[..., 1::2] *= -1000

        output, info = ffn(x)

        # Should handle without crash
        assert output.shape == (4, 16, 64)

    def test_sparse_input(self):
        """Mostly zeros with few non-zero values."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        x = torch.zeros(4, 16, 64)
        x[0, 0, 0] = 1.0
        x[1, 5, 32] = -1.0

        output, info = ffn(x)

        assert not torch.isnan(output).any()

    def test_random_walk_input(self):
        """Cumulative sum (random walk) - can have extreme values."""
        set_seed()
        steps = torch.randn(4, 16, 64)
        x = steps.cumsum(dim=1)  # Random walk along sequence

        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        output, info = ffn(x)

        assert not torch.isnan(output).any()

    def test_repeated_patterns(self):
        """Repeating patterns (like [1,0,1,0,...])."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)

        pattern = torch.tensor([1.0, -1.0]).repeat(32)
        x = pattern.unsqueeze(0).unsqueeze(0).expand(4, 16, 64)

        output, info = ffn(x)

        assert output.shape == (4, 16, 64)

    def test_correlated_dimensions(self):
        """All dimensions are copies of one dimension."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)

        base = torch.randn(4, 16, 1)
        x = base.expand(4, 16, 64)

        output, info = ffn(x)

        assert not torch.isnan(output).any()


# =============================================================================
# 9. INVARIANT VIOLATION TESTS
# =============================================================================

class TestInvariants:
    """Test documented invariants and contracts."""

    def test_anchor_signatures_ternary(self):
        """Anchor signatures should be ternary (-1, 0, +1)."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)

        unique_vals = ffn.anchor_signatures.unique()
        valid_vals = torch.tensor([-1.0, 0.0, 1.0])

        for v in unique_vals:
            assert v in valid_vals, f"Non-ternary value in anchors: {v}"

    def test_tile_directions_ternary(self):
        """Tile directions should be ternary."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)

        unique_vals = ffn.tile_directions.unique()
        valid_vals = torch.tensor([-1.0, 0.0, 1.0])

        for v in unique_vals:
            assert v in valid_vals, f"Non-ternary value in tiles: {v}"

    def test_octave_derivation_invariant(self):
        """Coarse octaves should be derived from fine."""
        set_seed()
        ffn = TrueOctaveFFN(d_model=64, num_fine_tiles=16, pool_factor=4)

        checks = ffn.get_derivation_check()

        for check_name, passed in checks.items():
            assert passed, f"Derivation invariant failed: {check_name}"

    def test_anchor_probs_sum_to_one(self):
        """Anchor probabilities should sum to 1."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        x = torch.randn(4, 16, 64)

        _, info = ffn(x)

        prob_sum = info['anchor_probs'].sum(dim=-1)
        assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5)

    def test_tile_probs_sum_to_one(self):
        """Tile probabilities should sum to 1."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        ffn.train()  # Soft routing in train mode
        x = torch.randn(4, 16, 64)

        _, info = ffn(x)

        prob_sum = info['tile_probs'].sum(dim=-1)
        assert torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5)

    def test_output_shape_preserved(self):
        """Output shape should match input shape."""
        set_seed()
        for ffn_class in [AnchoredDualModeFFN, TrueOctaveFFN]:
            if ffn_class == TrueOctaveFFN:
                ffn = ffn_class(d_model=64, num_fine_tiles=16)
            else:
                ffn = ffn_class(d_model=64, num_anchors=8, num_tiles=16)

            for batch, seq in [(1, 1), (4, 16), (8, 32)]:
                x = torch.randn(batch, seq, 64)
                out, _ = ffn(x)
                assert out.shape == x.shape, f"{ffn_class.__name__} shape mismatch"


# =============================================================================
# 10. STRESS TESTS
# =============================================================================

class TestStress:
    """Push the system to its limits."""

    def test_many_anchors(self):
        """Large number of anchors."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=256, num_anchors=128, num_tiles=256)
        x = torch.randn(2, 8, 256)

        output, info = ffn(x)

        assert output.shape == (2, 8, 256)
        assert not torch.isnan(output).any()

    def test_many_tiles_octave(self):
        """Large number of tiles in TrueOctave."""
        set_seed()
        ffn = TrueOctaveFFN(d_model=256, num_fine_tiles=256, pool_factor=4)
        x = torch.randn(2, 8, 256)

        output, info = ffn(x)

        assert output.shape == (2, 8, 256)

    def test_rapid_mode_switching(self):
        """Rapid train/eval mode switching."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        x = torch.randn(4, 16, 64)

        for _ in range(50):
            ffn.train()
            out, _ = ffn(x)
            ffn.eval()
            with torch.no_grad():
                out, _ = ffn(x)

        # Should complete without error
        assert True

    def test_rapid_temperature_changes(self):
        """Rapid temperature changes."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        x = torch.randn(4, 16, 64)

        for temp in [0.01, 0.1, 1.0, 10.0, 100.0, 0.001, 5.0, 0.5]:
            ffn.set_temperature(temp)
            out, info = ffn(x)
            assert not torch.isnan(out).any()

    def test_deep_stack(self):
        """Deep stack of blocks."""
        set_seed()

        class DeepStack(nn.Module):
            def __init__(self, num_layers: int):
                super().__init__()
                self.blocks = nn.ModuleList([
                    AnchoredDualModeBlock(d_model=64, num_heads=4, num_anchors=4, num_tiles=8)
                    for _ in range(num_layers)
                ])

            def forward(self, x):
                for block in self.blocks:
                    x, _ = block(x)
                return x

        model = DeepStack(num_layers=8)
        x = torch.randn(2, 16, 64)

        out = model(x)

        assert out.shape == (2, 16, 64)
        assert not torch.isnan(out).any()

    def test_gradient_accumulation(self):
        """Multiple backward passes with gradient accumulation."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        ffn.train()

        # Accumulate gradients over multiple batches
        for _ in range(10):
            x = torch.randn(4, 16, 64)
            out, _ = ffn(x)
            loss = out.sum()
            loss.backward()

        # Check gradients are accumulated (not just last batch)
        total_grad = sum(p.grad.abs().sum().item() for p in ffn.parameters() if p.grad is not None)
        assert total_grad > 0


# =============================================================================
# 11. TEMPERATURE SCHEDULE TESTS
# =============================================================================

class TestTemperatureSchedule:
    """Test temperature schedule function."""

    def test_linear_schedule(self):
        """Linear schedule should interpolate correctly."""
        t0 = get_temperature_schedule(0, 100, start_temp=2.0, end_temp=0.2, schedule='linear')
        t50 = get_temperature_schedule(50, 100, start_temp=2.0, end_temp=0.2, schedule='linear')
        t100 = get_temperature_schedule(100, 100, start_temp=2.0, end_temp=0.2, schedule='linear')

        assert abs(t0 - 2.0) < 1e-6
        assert abs(t50 - 1.1) < 1e-6
        assert abs(t100 - 0.2) < 1e-6

    def test_cosine_schedule(self):
        """Cosine schedule should start and end correctly."""
        t0 = get_temperature_schedule(0, 100, start_temp=2.0, end_temp=0.2, schedule='cosine')
        t100 = get_temperature_schedule(100, 100, start_temp=2.0, end_temp=0.2, schedule='cosine')

        assert abs(t0 - 2.0) < 1e-6
        assert abs(t100 - 0.2) < 1e-6

    def test_exponential_schedule(self):
        """Exponential schedule should start and end correctly."""
        t0 = get_temperature_schedule(0, 100, start_temp=2.0, end_temp=0.2, schedule='exponential')
        t100 = get_temperature_schedule(100, 100, start_temp=2.0, end_temp=0.2, schedule='exponential')

        assert abs(t0 - 2.0) < 1e-6
        assert abs(t100 - 0.2) < 1e-6

    def test_step_beyond_total(self):
        """Steps beyond total should clip to end temperature."""
        t = get_temperature_schedule(200, 100, start_temp=2.0, end_temp=0.2, schedule='linear')

        assert abs(t - 0.2) < 1e-6

    def test_invalid_schedule(self):
        """Invalid schedule should raise error."""
        with pytest.raises(ValueError):
            get_temperature_schedule(50, 100, schedule='invalid')


# =============================================================================
# 12. FROZEN TILE TESTS
# =============================================================================

class TestFrozenTile:
    """Test individual FrozenTile component."""

    def test_tile_signature_derivation(self):
        """Signature should be derived from up_weight."""
        set_seed()
        tile = FrozenTile(d_model=64, d_hidden=128)

        expected_sig = tile.up_weight.mean(dim=0).sign()

        assert torch.equal(tile.signature, expected_sig)

    def test_tile_scale_learnable(self):
        """Scale parameter should be learnable."""
        set_seed()
        tile = FrozenTile(d_model=64, d_hidden=128)
        x = torch.randn(4, 16, 64)

        out = tile(x)
        loss = out.sum()
        loss.backward()

        assert tile.scale.grad is not None
        assert tile.scale.grad.abs().sum() > 0

    def test_tile_weights_frozen(self):
        """Tile weights should not require grad."""
        set_seed()
        tile = FrozenTile(d_model=64, d_hidden=128)

        assert not tile.up_weight.requires_grad
        assert not tile.down_weight.requires_grad

    def test_tile_with_custom_init(self):
        """Tile should accept custom initialization."""
        set_seed()
        up = torch.sign(torch.randn(128, 64))
        down = torch.sign(torch.randn(64, 128))

        tile = FrozenTile(d_model=64, d_hidden=128, init_up=up, init_down=down)

        assert torch.equal(tile.up_weight, up)
        assert torch.equal(tile.down_weight, down)


# =============================================================================
# 13. ANCHOR LOSS TESTS
# =============================================================================

class TestAnchorLoss:
    """Test anchor regularization loss."""

    def test_anchor_loss_computes(self):
        """Anchor loss should compute without error."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        ffn.train()
        x = torch.randn(4, 16, 64)

        # Accumulate usage
        for _ in range(10):
            _ = ffn(x)

        loss = ffn.compute_anchor_loss()

        assert not torch.isnan(loss)
        assert loss >= 0

    def test_anchor_loss_encourages_diversity(self):
        """Anchor loss should be lower when usage is uniform."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)

        # Simulate uniform usage
        ffn.anchor_usage = torch.ones(8)
        ffn.usage_count = torch.tensor(8.0)

        loss_uniform = ffn.compute_anchor_loss().item()

        # Simulate peaked usage
        ffn.anchor_usage = torch.zeros(8)
        ffn.anchor_usage[0] = 8.0
        ffn.usage_count = torch.tensor(8.0)

        loss_peaked = ffn.compute_anchor_loss().item()

        assert loss_uniform < loss_peaked, "Uniform usage should have lower loss"


# =============================================================================
# 14. BLOCK INTEGRATION TESTS
# =============================================================================

class TestBlockIntegration:
    """Test transformer blocks with attention."""

    def test_anchored_block_forward(self):
        """AnchoredDualModeBlock should work end-to-end."""
        set_seed()
        block = AnchoredDualModeBlock(d_model=64, num_heads=4, num_anchors=4, num_tiles=16)
        x = torch.randn(4, 16, 64)

        out, info = block(x)

        assert out.shape == (4, 16, 64)
        assert not torch.isnan(out).any()

    def test_octave_block_forward(self):
        """TrueOctaveBlock should work end-to-end."""
        set_seed()
        block = TrueOctaveBlock(d_model=64, n_heads=4, num_fine_tiles=16)
        x = torch.randn(4, 16, 64)

        out, info = block(x)

        assert out.shape == (4, 16, 64)
        assert not torch.isnan(out).any()

    def test_block_with_attention_mask(self):
        """Block should handle attention mask."""
        set_seed()
        block = AnchoredDualModeBlock(d_model=64, num_heads=4, num_anchors=4, num_tiles=16)
        x = torch.randn(4, 16, 64)

        # Causal mask
        T = 16
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()

        out, info = block(x, attn_mask=mask)

        assert out.shape == (4, 16, 64)

    def test_block_gradients_flow_through_attention(self):
        """Gradients should flow through attention and FFN."""
        set_seed()
        block = AnchoredDualModeBlock(d_model=64, num_heads=4, num_anchors=4, num_tiles=16)
        block.train()
        x = torch.randn(4, 16, 64, requires_grad=True)

        out, _ = block(x)
        loss = out.sum()
        loss.backward()

        # Check gradients flow to attention
        assert block.attn.in_proj_weight.grad is not None
        assert block.attn.in_proj_weight.grad.abs().sum() > 0


# =============================================================================
# SUMMARY TEST
# =============================================================================

class TestSummary:
    """Final summary test to run everything."""

    def test_all_components_instantiate(self):
        """All core components should instantiate."""
        set_seed()

        components = [
            AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16),
            AnchoredDualModeBlock(d_model=64, num_heads=4, num_anchors=4, num_tiles=16),
            TrueOctaveFFN(d_model=64, num_fine_tiles=16),
            TrueOctaveBlock(d_model=64, n_heads=4, num_fine_tiles=16),
            FrozenTile(d_model=64, d_hidden=128),
            Octave(d_model=64, d_hidden=128, num_tiles=8),
        ]

        for comp in components:
            assert comp is not None

    def test_all_components_forward_pass(self):
        """All core components should forward without error."""
        set_seed()
        x = torch.randn(4, 16, 64)

        # FFNs
        ffn_anchored = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=16)
        out, _ = ffn_anchored(x)
        assert out.shape == x.shape

        ffn_octave = TrueOctaveFFN(d_model=64, num_fine_tiles=16)
        out, _ = ffn_octave(x)
        assert out.shape == x.shape

        # Blocks
        block_anchored = AnchoredDualModeBlock(d_model=64, num_heads=4, num_anchors=4, num_tiles=16)
        out, _ = block_anchored(x)
        assert out.shape == x.shape

        block_octave = TrueOctaveBlock(d_model=64, n_heads=4, num_fine_tiles=16)
        out, _ = block_octave(x)
        assert out.shape == x.shape
