"""
Rigorous Tests for Anchored Dual-Mode FFN.

Categories:
1. Frozen Component Invariants - Anchors and tiles never change
2. Gradient Flow Invariants - Correct flow paths
3. Partition Invariants - Anchor behavior is consistent
4. Training Invariants - Convergence, specialization
5. Numerical Stability - NaN, Inf, precision
6. Edge Cases - Boundary conditions, extreme values
7. Temperature Invariants - Annealing behaves correctly
8. Determinism Invariants - Reproducibility
9. Scale Invariants - Behavior at different sizes
10. Comparison Tests - vs SparseLookupFFN baseline

"The anchor partitions the question. The routing searches the partition."
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from typing import List, Tuple

from trix.nn.anchored import (
    AnchoredDualModeFFN,
    AnchoredDualModeBlock,
    get_temperature_schedule,
)


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 1. FROZEN COMPONENT INVARIANTS
# =============================================================================

class TestFrozenInvariants:
    """Verify frozen components never change."""

    def test_anchor_signatures_unchanged_after_forward(self):
        """Anchor signatures should not change during forward pass."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)

        anchors_before = ffn.anchor_signatures.clone()

        x = torch.randn(4, 16, 64)
        for _ in range(10):
            ffn(x)

        anchors_after = ffn.anchor_signatures

        assert torch.equal(anchors_before, anchors_after), \
            "Anchor signatures should never change"

    def test_anchor_signatures_unchanged_after_backward(self):
        """Anchor signatures should not change during backward pass."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn.train()

        anchors_before = ffn.anchor_signatures.clone()

        optimizer = torch.optim.Adam(ffn.parameters(), lr=1e-3)
        x = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)

        for _ in range(10):
            optimizer.zero_grad()
            output, _ = ffn(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()

        anchors_after = ffn.anchor_signatures

        assert torch.equal(anchors_before, anchors_after), \
            "Anchor signatures should never change during training"

    def test_tile_directions_unchanged_after_forward(self):
        """Tile directions should not change during forward pass."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)

        tiles_before = ffn.tile_directions.clone()

        x = torch.randn(4, 16, 64)
        for _ in range(10):
            ffn(x)

        tiles_after = ffn.tile_directions

        assert torch.equal(tiles_before, tiles_after), \
            "Tile directions should never change"

    def test_tile_directions_unchanged_after_backward(self):
        """Tile directions should not change during backward pass."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn.train()

        tiles_before = ffn.tile_directions.clone()

        optimizer = torch.optim.Adam(ffn.parameters(), lr=1e-3)
        x = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)

        for _ in range(10):
            optimizer.zero_grad()
            output, _ = ffn(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()

        tiles_after = ffn.tile_directions

        assert torch.equal(tiles_before, tiles_after), \
            "Tile directions should never change during training"

    def test_frozen_components_are_ternary(self):
        """All frozen components should be ternary {-1, 0, +1}."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=128, num_anchors=16, num_tiles=64)

        valid_values = {-1.0, 0.0, 1.0}

        anchor_values = set(ffn.anchor_signatures.unique().tolist())
        assert anchor_values.issubset(valid_values), \
            f"Anchor values {anchor_values} not subset of {valid_values}"

        tile_values = set(ffn.tile_directions.unique().tolist())
        assert tile_values.issubset(valid_values), \
            f"Tile values {tile_values} not subset of {valid_values}"

    def test_frozen_components_not_in_parameters(self):
        """Frozen components should not be in model.parameters()."""
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)

        param_names = {name for name, _ in ffn.named_parameters()}

        assert 'anchor_signatures' not in param_names, \
            "anchor_signatures should not be a parameter"
        assert 'tile_directions' not in param_names, \
            "tile_directions should not be a parameter"

    def test_frozen_components_in_buffers(self):
        """Frozen components should be registered as buffers."""
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)

        buffer_names = {name for name, _ in ffn.named_buffers()}

        assert 'anchor_signatures' in buffer_names, \
            "anchor_signatures should be a buffer"
        assert 'tile_directions' in buffer_names, \
            "tile_directions should be a buffer"


# =============================================================================
# 2. GRADIENT FLOW INVARIANTS
# =============================================================================

class TestGradientFlowInvariants:
    """Verify gradients flow through correct paths only."""

    def test_gradients_flow_to_tile_scales(self):
        """Tile scales must receive non-zero gradients."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn.train()

        x = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)

        output, _ = ffn(x)
        loss = F.mse_loss(output, target)
        loss.backward()

        assert ffn.tile_scales.grad is not None, "tile_scales should have gradient"
        assert ffn.tile_scales.grad.abs().sum() > 0, "tile_scales gradient should be non-zero"

    def test_gradients_flow_to_router(self):
        """Router must receive non-zero gradients."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn.train()

        x = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)

        output, _ = ffn(x)
        loss = F.mse_loss(output, target)
        loss.backward()

        router_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in ffn.router.parameters()
        )
        assert router_has_grad, "Router should receive gradients"

    def test_gradients_flow_to_anchor_proj(self):
        """Anchor projection must receive non-zero gradients."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn.train()

        x = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)

        output, _ = ffn(x)
        loss = F.mse_loss(output, target)
        loss.backward()

        assert ffn.anchor_proj.weight.grad is not None, \
            "anchor_proj should have gradient"
        assert ffn.anchor_proj.weight.grad.abs().sum() > 0, \
            "anchor_proj gradient should be non-zero"

    def test_no_gradients_to_frozen_components(self):
        """Frozen components must not receive gradients."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn.train()

        x = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)

        output, _ = ffn(x)
        loss = F.mse_loss(output, target)
        loss.backward()

        # Buffers don't have .grad attribute, but let's verify they're not Parameters
        assert not ffn.anchor_signatures.requires_grad, \
            "anchor_signatures should not require grad"
        assert not ffn.tile_directions.requires_grad, \
            "tile_directions should not require grad"

    def test_gradient_magnitude_reasonable(self):
        """Gradients should have reasonable magnitude (no explosion)."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn.train()

        x = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)

        output, _ = ffn(x)
        loss = F.mse_loss(output, target)
        loss.backward()

        for name, param in ffn.named_parameters():
            if param.grad is not None:
                grad_max = param.grad.abs().max().item()
                assert grad_max < 1000, f"Gradient explosion in {name}: {grad_max}"


# =============================================================================
# 3. PARTITION INVARIANTS
# =============================================================================

class TestPartitionInvariants:
    """Verify anchor partitioning behavior."""

    def test_anchor_probs_sum_to_one(self):
        """Anchor probabilities must sum to 1."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)

        x = torch.randn(4, 16, 64)
        _, info = ffn(x)

        sums = info['anchor_probs'].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            "Anchor probs must sum to 1"

    def test_anchor_probs_non_negative(self):
        """Anchor probabilities must be non-negative."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)

        x = torch.randn(4, 16, 64)
        _, info = ffn(x)

        assert (info['anchor_probs'] >= 0).all(), \
            "Anchor probs must be non-negative"

    def test_same_input_same_anchor(self):
        """Same input must produce same anchor assignment."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn.eval()

        x = torch.randn(4, 16, 64)

        with torch.no_grad():
            _, info1 = ffn(x)
            _, info2 = ffn(x)

        assert torch.equal(info1['dominant_anchor'], info2['dominant_anchor']), \
            "Same input must give same anchor"

    def test_similar_inputs_similar_anchors(self):
        """Similar inputs should tend to get same anchor (locality)."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn.eval()

        x = torch.randn(1, 1, 64)
        x_perturbed = x + torch.randn_like(x) * 0.01  # Small perturbation

        with torch.no_grad():
            _, info1 = ffn(x)
            _, info2 = ffn(x_perturbed)

        # With small perturbation, anchors should often match
        # (not always, but we check the probs are similar)
        prob_diff = (info1['anchor_probs'] - info2['anchor_probs']).abs().max()
        assert prob_diff < 0.5, \
            f"Similar inputs should have similar anchor probs, diff={prob_diff}"

    def test_all_anchors_reachable(self):
        """All anchors should be reachable with diverse enough input."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn.eval()

        # Use enough samples to cover anchor space
        x = torch.randn(1000, 1, 64)

        with torch.no_grad():
            _, info = ffn(x)

        unique_anchors = info['dominant_anchor'].unique()
        coverage = len(unique_anchors) / ffn.num_anchors

        assert coverage >= 0.5, \
            f"At least 50% of anchors should be used, got {coverage:.1%}"


# =============================================================================
# 4. TRAINING INVARIANTS
# =============================================================================

class TestTrainingInvariants:
    """Verify training dynamics."""

    def test_loss_decreases(self):
        """Loss must decrease over training."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn.train()

        optimizer = torch.optim.Adam(ffn.parameters(), lr=1e-3)
        x = torch.randn(16, 32, 64)
        target = torch.randn(16, 32, 64)

        # Initial loss
        output, _ = ffn(x)
        initial_loss = F.mse_loss(output, target).item()

        # Train
        for _ in range(100):
            optimizer.zero_grad()
            output, _ = ffn(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()

        # Final loss
        output, _ = ffn(x)
        final_loss = F.mse_loss(output, target).item()

        assert final_loss < initial_loss * 0.9, \
            f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_tile_scales_change_during_training(self):
        """Tile scales should change during training."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn.train()

        scales_before = ffn.tile_scales.clone()

        optimizer = torch.optim.Adam(ffn.parameters(), lr=1e-2)
        x = torch.randn(16, 32, 64)
        target = torch.randn(16, 32, 64)

        for _ in range(50):
            optimizer.zero_grad()
            output, _ = ffn(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()

        scales_after = ffn.tile_scales

        assert not torch.equal(scales_before, scales_after), \
            "Tile scales should change during training"

    def test_router_weights_change_during_training(self):
        """Router weights should change during training."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn.train()

        # Get first router layer weights
        router_weights_before = list(ffn.router.parameters())[0].clone()

        optimizer = torch.optim.Adam(ffn.parameters(), lr=1e-2)
        x = torch.randn(16, 32, 64)
        target = torch.randn(16, 32, 64)

        for _ in range(50):
            optimizer.zero_grad()
            output, _ = ffn(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()

        router_weights_after = list(ffn.router.parameters())[0]

        assert not torch.equal(router_weights_before, router_weights_after), \
            "Router weights should change during training"

    def test_tile_utilization_improves(self):
        """Tile utilization should be reasonable after training."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn.train()

        optimizer = torch.optim.Adam(ffn.parameters(), lr=1e-3)
        x = torch.randn(64, 32, 64)
        target = torch.randn(64, 32, 64)

        for _ in range(100):
            optimizer.zero_grad()
            output, info = ffn(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()

        # Check utilization
        ffn.eval()
        with torch.no_grad():
            _, info = ffn(x)

        assert info['tile_utilization'] > 0.1, \
            f"Tile utilization too low: {info['tile_utilization']:.1%}"


# =============================================================================
# 5. NUMERICAL STABILITY
# =============================================================================

class TestNumericalStability:
    """Verify numerical stability."""

    def test_no_nan_forward(self):
        """Forward pass should not produce NaN."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)

        x = torch.randn(4, 16, 64)
        output, info = ffn(x)

        assert not output.isnan().any(), "Output contains NaN"
        assert not info['anchor_probs'].isnan().any(), "Anchor probs contain NaN"

    def test_no_inf_forward(self):
        """Forward pass should not produce Inf."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)

        x = torch.randn(4, 16, 64)
        output, info = ffn(x)

        assert not output.isinf().any(), "Output contains Inf"
        assert not info['anchor_probs'].isinf().any(), "Anchor probs contain Inf"

    def test_no_nan_backward(self):
        """Backward pass should not produce NaN gradients."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn.train()

        x = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)

        output, _ = ffn(x)
        loss = F.mse_loss(output, target)
        loss.backward()

        for name, param in ffn.named_parameters():
            if param.grad is not None:
                assert not param.grad.isnan().any(), f"NaN gradient in {name}"

    def test_large_input_stability(self):
        """Large input values should not cause instability."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)

        x = torch.randn(4, 16, 64) * 100
        output, info = ffn(x)

        assert not output.isnan().any(), "NaN with large input"
        assert not output.isinf().any(), "Inf with large input"

    def test_small_input_stability(self):
        """Small input values should not cause instability."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)

        x = torch.randn(4, 16, 64) * 1e-6
        output, info = ffn(x)

        assert not output.isnan().any(), "NaN with small input"
        assert not output.isinf().any(), "Inf with small input"

    def test_zero_input_stability(self):
        """Zero input should not cause instability."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)

        x = torch.zeros(4, 16, 64)
        output, info = ffn(x)

        assert not output.isnan().any(), "NaN with zero input"
        assert not output.isinf().any(), "Inf with zero input"

    def test_output_bounded(self):
        """Output should be bounded relative to input."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn.eval()

        x = torch.randn(4, 16, 64)
        with torch.no_grad():
            output, _ = ffn(x)

        # Output should not explode beyond input + reasonable FFN contribution
        max_expected = x.abs().max() + 20  # Conservative bound
        assert output.abs().max() < max_expected, \
            f"Output too large: {output.abs().max()} > {max_expected}"


# =============================================================================
# 6. EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_batch_size_1(self):
        """Single sample batch."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)

        x = torch.randn(1, 16, 64)
        output, info = ffn(x)

        assert output.shape == x.shape
        assert not output.isnan().any()

    def test_seq_len_1(self):
        """Single position sequence."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)

        x = torch.randn(4, 1, 64)
        output, info = ffn(x)

        assert output.shape == x.shape
        assert not output.isnan().any()

    def test_batch_1_seq_1(self):
        """Single sample, single position."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)

        x = torch.randn(1, 1, 64)
        output, info = ffn(x)

        assert output.shape == x.shape
        assert not output.isnan().any()

    def test_large_batch(self):
        """Large batch size."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)

        x = torch.randn(256, 16, 64)
        output, info = ffn(x)

        assert output.shape == x.shape
        assert not output.isnan().any()

    def test_large_seq_len(self):
        """Large sequence length."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)

        x = torch.randn(4, 512, 64)
        output, info = ffn(x)

        assert output.shape == x.shape
        assert not output.isnan().any()

    def test_num_anchors_equals_1(self):
        """Single anchor (degenerate case)."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=1, num_tiles=32)

        x = torch.randn(4, 16, 64)
        output, info = ffn(x)

        assert output.shape == x.shape
        # With 1 anchor, all probs should be 1
        assert torch.allclose(info['anchor_probs'], torch.ones_like(info['anchor_probs']))

    def test_num_tiles_equals_1(self):
        """Single tile (degenerate case)."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=1)

        x = torch.randn(4, 16, 64)
        output, info = ffn(x)

        assert output.shape == x.shape
        # With 1 tile, all tile_idx should be 0
        assert (info['tile_idx'] == 0).all()


# =============================================================================
# 7. TEMPERATURE INVARIANTS
# =============================================================================

class TestTemperatureInvariants:
    """Verify temperature annealing behavior."""

    def test_low_temperature_sharpens_distribution(self):
        """Lower temperature should produce sharper anchor distribution."""
        set_seed()

        # Same architecture, different temperatures
        ffn_warm = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32, temperature=2.0)
        ffn_cold = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32, temperature=0.1)

        # Copy anchors to ensure same partitioning basis
        ffn_cold.anchor_signatures.copy_(ffn_warm.anchor_signatures)

        x = torch.randn(4, 16, 64)

        _, info_warm = ffn_warm(x)
        _, info_cold = ffn_cold(x)

        assert info_cold['anchor_entropy'] < info_warm['anchor_entropy'], \
            "Lower temperature should give lower entropy"

    def test_very_low_temperature_near_one_hot(self):
        """Very low temperature should produce near-one-hot distribution."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32, temperature=0.01)

        x = torch.randn(4, 16, 64)
        _, info = ffn(x)

        # Max prob should be very close to 1 for MOST inputs
        # (Edge case: equal Hamming distance to multiple anchors gives 0.5)
        max_probs = info['anchor_probs'].max(dim=-1).values
        high_confidence_ratio = (max_probs > 0.9).float().mean()
        assert high_confidence_ratio > 0.8, \
            f"Very low temp should give mostly near-one-hot, got {high_confidence_ratio:.1%} high confidence"

    def test_high_temperature_near_uniform(self):
        """Very high temperature should produce near-uniform distribution."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32, temperature=100.0)

        x = torch.randn(4, 16, 64)
        _, info = ffn(x)

        # Distribution should be close to uniform
        uniform = 1.0 / 8  # num_anchors
        max_deviation = (info['anchor_probs'] - uniform).abs().max()
        assert max_deviation < 0.1, \
            f"High temp should give near-uniform, got max_dev={max_deviation}"

    def test_temperature_schedule_monotonic(self):
        """Temperature schedule should be monotonically decreasing."""
        total_steps = 1000
        temps = [get_temperature_schedule(s, total_steps, 2.0, 0.1, 'linear')
                 for s in range(0, total_steps + 1, 100)]

        for i in range(len(temps) - 1):
            assert temps[i] >= temps[i + 1], \
                f"Temperature should decrease: {temps[i]} -> {temps[i+1]}"

    def test_temperature_schedule_bounds(self):
        """Temperature schedule should stay within bounds."""
        total_steps = 1000
        start, end = 2.0, 0.1

        for schedule in ['linear', 'cosine', 'exponential']:
            for step in range(total_steps + 1):
                temp = get_temperature_schedule(step, total_steps, start, end, schedule)
                assert end <= temp <= start, \
                    f"Temp {temp} out of bounds [{end}, {start}] at step {step}"


# =============================================================================
# 8. DETERMINISM INVARIANTS
# =============================================================================

class TestDeterminismInvariants:
    """Verify reproducibility."""

    def test_same_seed_same_init(self):
        """Same seed should produce same initialization."""
        set_seed(123)
        ffn1 = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)

        set_seed(123)
        ffn2 = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)

        assert torch.equal(ffn1.anchor_signatures, ffn2.anchor_signatures)
        assert torch.equal(ffn1.tile_directions, ffn2.tile_directions)
        assert torch.equal(ffn1.tile_scales, ffn2.tile_scales)

    def test_eval_mode_deterministic(self):
        """Eval mode should be deterministic."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn.eval()

        x = torch.randn(4, 16, 64)

        with torch.no_grad():
            output1, info1 = ffn(x)
            output2, info2 = ffn(x)

        assert torch.equal(output1, output2), "Eval mode should be deterministic"
        assert torch.equal(info1['tile_idx'], info2['tile_idx'])

    def test_train_mode_with_seed_deterministic(self):
        """Train mode with same seed should be deterministic."""
        x = torch.randn(4, 16, 64)
        target = torch.randn(4, 16, 64)

        # Run 1
        set_seed(456)
        ffn1 = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn1.train()
        optimizer1 = torch.optim.Adam(ffn1.parameters(), lr=1e-3)

        for _ in range(10):
            optimizer1.zero_grad()
            out, _ = ffn1(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer1.step()

        # Run 2
        set_seed(456)
        ffn2 = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=32)
        ffn2.train()
        optimizer2 = torch.optim.Adam(ffn2.parameters(), lr=1e-3)

        for _ in range(10):
            optimizer2.zero_grad()
            out, _ = ffn2(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer2.step()

        assert torch.allclose(ffn1.tile_scales, ffn2.tile_scales, atol=1e-6), \
            "Same seed should give same training result"


# =============================================================================
# 9. SCALE INVARIANTS
# =============================================================================

class TestScaleInvariants:
    """Verify behavior at different scales."""

    @pytest.mark.parametrize("d_model", [32, 64, 128, 256])
    def test_different_d_model(self, d_model):
        """Should work with different model dimensions."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=d_model, num_anchors=8, num_tiles=32)

        x = torch.randn(4, 16, d_model)
        output, info = ffn(x)

        assert output.shape == x.shape
        assert not output.isnan().any()

    @pytest.mark.parametrize("num_anchors", [4, 8, 16, 32])
    def test_different_num_anchors(self, num_anchors):
        """Should work with different anchor counts."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=num_anchors, num_tiles=32)

        x = torch.randn(4, 16, 64)
        output, info = ffn(x)

        assert output.shape == x.shape
        assert info['anchor_probs'].shape[-1] == num_anchors

    @pytest.mark.parametrize("num_tiles", [8, 16, 32, 64])
    def test_different_num_tiles(self, num_tiles):
        """Should work with different tile counts."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=64, num_anchors=8, num_tiles=num_tiles)

        x = torch.randn(4, 16, 64)
        output, info = ffn(x)

        assert output.shape == x.shape
        assert info['tile_logits'].shape[-1] == num_tiles


# =============================================================================
# 10. COMPARISON TESTS
# =============================================================================

class TestComparison:
    """Compare with baseline implementations."""

    def test_parameter_count_reasonable(self):
        """Parameter count should be reasonable."""
        ffn = AnchoredDualModeFFN(d_model=512, num_anchors=16, num_tiles=64)

        param_count = sum(p.numel() for p in ffn.parameters())

        # Should be much less than dense FFN (512 * 2048 * 2 = 2M for standard)
        assert param_count < 500_000, \
            f"Too many parameters: {param_count}"

    def test_inference_faster_than_training(self):
        """Inference (hard routing) should be faster than training (soft routing)."""
        set_seed()
        ffn = AnchoredDualModeFFN(d_model=256, num_anchors=16, num_tiles=64)

        x = torch.randn(32, 64, 256)

        # Warmup
        for _ in range(5):
            ffn.train()
            ffn(x)
            ffn.eval()
            with torch.no_grad():
                ffn(x)

        # Time training mode
        ffn.train()
        start = time.perf_counter()
        for _ in range(20):
            ffn(x)
        train_time = time.perf_counter() - start

        # Time inference mode
        ffn.eval()
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(20):
                ffn(x)
            eval_time = time.perf_counter() - start

        # Inference should be faster (hard selection vs soft combination)
        assert eval_time < train_time, \
            f"Inference ({eval_time:.4f}s) should be faster than training ({train_time:.4f}s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
