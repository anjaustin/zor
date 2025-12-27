"""
Tests for AnchoredDualModeFFN - Partition First, Search Within

Tests the dual-mode architecture where:
- Frozen anchors partition the input space (shoreline)
- Learned routing searches within partitions (navigation)
- Frozen tiles execute with learned scales (port operations)
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from trix.nn.anchored import (
    AnchoredDualModeFFN,
    AnchoredDualModeBlock,
    get_temperature_schedule,
)


# Test constants
BATCH_SIZE = 4
SEQ_LEN = 16
D_MODEL = 64
NUM_ANCHORS = 8
NUM_TILES = 32
SEED = 42


class TestAnchoredDualModeFFN:
    """Core tests for AnchoredDualModeFFN."""

    def test_forward_shape(self):
        """Output shape matches input shape."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        output, info = ffn(x)

        assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"

    def test_info_contains_expected_keys(self):
        """Info dict contains all expected keys."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        _, info = ffn(x)

        expected_keys = [
            'anchor_probs',
            'anchor_logits',
            'anchor_entropy',
            'dominant_anchor',
            'tile_idx',
            'tile_logits',
            'tile_utilization',
            'temperature',
        ]

        for key in expected_keys:
            assert key in info, f"Missing key: {key}"

    def test_anchor_probs_sum_to_one(self):
        """Anchor probabilities sum to 1 (valid distribution)."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        _, info = ffn(x)

        sums = info['anchor_probs'].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            "Anchor probs should sum to 1"

    def test_anchor_probs_shape(self):
        """Anchor probs have correct shape."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        _, info = ffn(x)

        expected_shape = (BATCH_SIZE, SEQ_LEN, NUM_ANCHORS)
        assert info['anchor_probs'].shape == expected_shape, \
            f"Expected {expected_shape}, got {info['anchor_probs'].shape}"

    def test_tile_idx_in_valid_range(self):
        """Tile indices are in valid range."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        _, info = ffn(x)

        assert info['tile_idx'].min() >= 0, "Tile idx should be >= 0"
        assert info['tile_idx'].max() < NUM_TILES, f"Tile idx should be < {NUM_TILES}"


class TestFrozenComponents:
    """Tests verifying frozen components don't receive gradients."""

    def test_anchor_signatures_are_frozen(self):
        """Anchor signatures should not receive gradients."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        # Anchor signatures should be a buffer, not a parameter
        assert 'anchor_signatures' not in dict(ffn.named_parameters()), \
            "anchor_signatures should be a buffer, not a parameter"
        assert 'anchor_signatures' in dict(ffn.named_buffers()), \
            "anchor_signatures should be registered as a buffer"

    def test_tile_directions_are_frozen(self):
        """Tile directions should not receive gradients."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        assert 'tile_directions' not in dict(ffn.named_parameters()), \
            "tile_directions should be a buffer, not a parameter"
        assert 'tile_directions' in dict(ffn.named_buffers()), \
            "tile_directions should be registered as a buffer"

    def test_anchor_signatures_are_ternary(self):
        """Anchor signatures should be ternary {-1, 0, +1}."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        unique_values = ffn.anchor_signatures.unique()
        valid_values = torch.tensor([-1., 0., 1.])

        for v in unique_values:
            assert v in valid_values, f"Invalid anchor value: {v}"

    def test_tile_directions_are_ternary(self):
        """Tile directions should be ternary {-1, 0, +1}."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        unique_values = ffn.tile_directions.unique()
        valid_values = torch.tensor([-1., 0., 1.])

        for v in unique_values:
            assert v in valid_values, f"Invalid tile direction value: {v}"


class TestLearnedComponents:
    """Tests verifying learned components receive gradients."""

    def test_tile_scales_receive_gradients(self):
        """Tile scales should receive gradients."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        target = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

        output, _ = ffn(x)
        loss = F.mse_loss(output, target)
        loss.backward()

        assert ffn.tile_scales.grad is not None, \
            "tile_scales should receive gradients"
        assert ffn.tile_scales.grad.abs().sum() > 0, \
            "tile_scales gradient should be non-zero"

    def test_router_receives_gradients(self):
        """Router should receive gradients."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        target = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

        output, _ = ffn(x)
        loss = F.mse_loss(output, target)
        loss.backward()

        # Check router has gradients
        router_has_grad = False
        for name, param in ffn.router.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                router_has_grad = True
                break

        assert router_has_grad, "Router should receive gradients"

    def test_anchor_proj_receives_gradients(self):
        """Anchor projection should receive gradients."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        target = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

        output, _ = ffn(x)
        loss = F.mse_loss(output, target)
        loss.backward()

        assert ffn.anchor_proj.weight.grad is not None, \
            "anchor_proj should receive gradients"


class TestTemperature:
    """Tests for temperature control."""

    def test_set_temperature(self):
        """Temperature can be set."""
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
            temperature=1.0,
        )

        assert ffn.temperature == 1.0
        ffn.set_temperature(0.5)
        assert ffn.temperature == 0.5

    def test_low_temperature_sharpens_anchors(self):
        """Lower temperature produces sharper anchor distributions."""
        torch.manual_seed(SEED)

        ffn_warm = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
            temperature=2.0,
        )

        ffn_cold = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
            temperature=0.1,
        )

        # Use same frozen weights
        ffn_cold.anchor_signatures.copy_(ffn_warm.anchor_signatures)

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

        _, info_warm = ffn_warm(x)
        _, info_cold = ffn_cold(x)

        # Cold should have lower entropy (sharper)
        assert info_cold['anchor_entropy'] < info_warm['anchor_entropy'], \
            "Lower temperature should produce lower entropy"

    def test_temperature_schedule_linear(self):
        """Linear temperature schedule works correctly."""
        total_steps = 1000
        start_temp = 2.0
        end_temp = 0.1

        # Start
        temp_0 = get_temperature_schedule(0, total_steps, start_temp, end_temp, 'linear')
        assert abs(temp_0 - start_temp) < 1e-5, f"Expected {start_temp}, got {temp_0}"

        # End
        temp_end = get_temperature_schedule(total_steps, total_steps, start_temp, end_temp, 'linear')
        assert abs(temp_end - end_temp) < 1e-5, f"Expected {end_temp}, got {temp_end}"

        # Middle
        temp_mid = get_temperature_schedule(500, total_steps, start_temp, end_temp, 'linear')
        expected_mid = (start_temp + end_temp) / 2
        assert abs(temp_mid - expected_mid) < 1e-5, f"Expected {expected_mid}, got {temp_mid}"

    def test_temperature_schedule_cosine(self):
        """Cosine temperature schedule works correctly."""
        total_steps = 1000
        start_temp = 2.0
        end_temp = 0.1

        temp_0 = get_temperature_schedule(0, total_steps, start_temp, end_temp, 'cosine')
        temp_end = get_temperature_schedule(total_steps, total_steps, start_temp, end_temp, 'cosine')

        assert abs(temp_0 - start_temp) < 1e-5
        assert abs(temp_end - end_temp) < 1e-5


class TestAnchorUtilization:
    """Tests for anchor utilization tracking."""

    def test_anchor_usage_tracking(self):
        """Anchor usage is tracked during training."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )
        ffn.train()

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        ffn(x)

        assert ffn.usage_count > 0, "Usage count should be tracked"
        assert ffn.anchor_usage.sum() > 0, "Anchor usage should be tracked"

    def test_anchor_usage_reset(self):
        """Anchor usage can be reset."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )
        ffn.train()

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        ffn(x)

        ffn.reset_usage_stats()

        assert ffn.usage_count == 0, "Usage count should be reset"
        assert ffn.anchor_usage.sum() == 0, "Anchor usage should be reset"

    def test_get_anchor_utilization(self):
        """Anchor utilization is a valid distribution."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )
        ffn.train()

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        ffn(x)

        utilization = ffn.get_anchor_utilization()

        assert utilization.shape == (NUM_ANCHORS,)
        assert torch.allclose(utilization.sum(), torch.tensor(1.0), atol=1e-5), \
            "Utilization should sum to 1"


class TestTraining:
    """Tests for training dynamics."""

    def test_training_reduces_loss(self):
        """Training for a few steps reduces loss."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )
        ffn.train()

        optimizer = torch.optim.Adam(ffn.parameters(), lr=1e-3)

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        target = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

        # Initial loss
        output, _ = ffn(x)
        initial_loss = F.mse_loss(output, target).item()

        # Train for a few steps
        for _ in range(50):
            optimizer.zero_grad()
            output, _ = ffn(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()

        # Final loss
        output, _ = ffn(x)
        final_loss = F.mse_loss(output, target).item()

        assert final_loss < initial_loss, \
            f"Training should reduce loss: {initial_loss} -> {final_loss}"

    def test_no_nan_in_forward(self):
        """Forward pass produces no NaN."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        output, info = ffn(x)

        assert not torch.isnan(output).any(), "Output should not contain NaN"
        assert not torch.isnan(info['anchor_probs']).any(), "Anchor probs should not contain NaN"

    def test_no_nan_in_backward(self):
        """Backward pass produces no NaN gradients."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        target = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

        output, _ = ffn(x)
        loss = F.mse_loss(output, target)
        loss.backward()

        for name, param in ffn.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), \
                    f"Gradient for {name} should not contain NaN"


class TestAnchoredDualModeBlock:
    """Tests for the full transformer block."""

    def test_block_forward_shape(self):
        """Block output has correct shape."""
        torch.manual_seed(SEED)
        block = AnchoredDualModeBlock(
            d_model=D_MODEL,
            num_heads=4,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
        output, info = block(x)

        assert output.shape == x.shape

    def test_block_with_attention_mask(self):
        """Block works with attention mask."""
        torch.manual_seed(SEED)
        block = AnchoredDualModeBlock(
            d_model=D_MODEL,
            num_heads=4,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

        # Causal mask
        mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool()

        output, info = block(x, attn_mask=mask)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_block_set_temperature(self):
        """Block temperature can be set."""
        block = AnchoredDualModeBlock(
            d_model=D_MODEL,
            num_heads=4,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
            temperature=1.0,
        )

        assert block.ffn.temperature == 1.0
        block.set_temperature(0.5)
        assert block.ffn.temperature == 0.5


class TestDeterministicBehavior:
    """Tests verifying deterministic behavior of frozen components."""

    def test_anchor_computation_is_deterministic(self):
        """Same input produces same anchor probs."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )
        ffn.eval()

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

        with torch.no_grad():
            _, info1 = ffn(x)
            _, info2 = ffn(x)

        assert torch.allclose(info1['anchor_probs'], info2['anchor_probs']), \
            "Anchor computation should be deterministic"

    def test_inference_is_deterministic(self):
        """Same input produces same output in eval mode."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )
        ffn.eval()

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

        with torch.no_grad():
            output1, _ = ffn(x)
            output2, _ = ffn(x)

        assert torch.allclose(output1, output2), \
            "Inference should be deterministic"


class TestEdgeCases:
    """Edge case tests."""

    def test_batch_size_one(self):
        """Works with batch size 1."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        x = torch.randn(1, SEQ_LEN, D_MODEL)
        output, info = ffn(x)

        assert output.shape == x.shape

    def test_seq_len_one(self):
        """Works with sequence length 1."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        x = torch.randn(BATCH_SIZE, 1, D_MODEL)
        output, info = ffn(x)

        assert output.shape == x.shape

    def test_large_input(self):
        """Works with larger input values."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL) * 100
        output, info = ffn(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_zero_input(self):
        """Works with zero input."""
        torch.manual_seed(SEED)
        ffn = AnchoredDualModeFFN(
            d_model=D_MODEL,
            num_anchors=NUM_ANCHORS,
            num_tiles=NUM_TILES,
        )

        x = torch.zeros(BATCH_SIZE, SEQ_LEN, D_MODEL)
        output, info = ffn(x)

        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
