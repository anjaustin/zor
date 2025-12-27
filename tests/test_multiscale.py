"""
Tests for MultiScaleTriXFFN.

Tests both generative (soft) and deterministic (hard) modes.

NOTE: This tests archived modules. MultiScaleTriXFFN is superseded by TrueOctaveFFN.
See trix.nn.archive.ARCHIVE_INDEX.md for details.
"""

import torch
import pytest
from trix.nn.archive.multiscale import (
    MultiScaleTriXFFN,
    MultiScaleTriXBlock,
    Octave,
    OctaveTile,
)


class TestOctaveTile:
    """Tests for single tile."""
    
    def test_forward_shape(self):
        tile = OctaveTile(d_model=64, d_hidden=128)
        x = torch.randn(2, 16, 64)
        out = tile(x)
        assert out.shape == x.shape
    
    def test_weights_are_ternary(self):
        tile = OctaveTile(d_model=64, d_hidden=128)
        assert set(tile.up_weight.unique().tolist()).issubset({-1.0, 0.0, 1.0})
        assert set(tile.down_weight.unique().tolist()).issubset({-1.0, 0.0, 1.0})
    
    def test_weights_are_frozen(self):
        tile = OctaveTile(d_model=64, d_hidden=128)
        assert not tile.up_weight.requires_grad
        assert not tile.down_weight.requires_grad
    
    def test_scales_are_learned(self):
        tile = OctaveTile(d_model=64, d_hidden=128)
        assert tile.up_scale.requires_grad
        assert tile.down_scale.requires_grad
    
    def test_gradient_flow(self):
        tile = OctaveTile(d_model=64, d_hidden=128)
        x = torch.randn(2, 16, 64)
        out = tile(x)
        out.sum().backward()
        assert tile.up_scale.grad is not None
        assert tile.down_scale.grad is not None


class TestOctave:
    """Tests for single octave (scale level)."""
    
    def test_forward_shape(self):
        octave = Octave(d_model=64, num_tiles=8, d_hidden=128, octave_level=0)
        x = torch.randn(2, 16, 64)
        out, weights, indices = octave(x)
        assert out.shape == x.shape
        assert weights.shape == (2, 16, 8)
        assert indices.shape == (2, 16)
    
    def test_soft_routing(self):
        octave = Octave(d_model=64, num_tiles=8, d_hidden=128, octave_level=0)
        x = torch.randn(2, 16, 64)
        out, weights, indices = octave(x, hard=False)
        # Weights should sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 16))
    
    def test_hard_routing(self):
        octave = Octave(d_model=64, num_tiles=8, d_hidden=128, octave_level=0)
        x = torch.randn(2, 16, 64)
        out, weights, indices = octave(x, hard=True)
        # Weights should be one-hot
        assert (weights.sum(dim=-1) == 1).all()
        assert ((weights == 0) | (weights == 1)).all()


class TestMultiScaleTriXFFN:
    """Tests for multi-scale FFN."""
    
    def test_forward_shape(self):
        ffn = MultiScaleTriXFFN(d_model=64, num_tiles_fine=16, num_tiles_medium=8, num_tiles_coarse=4)
        x = torch.randn(2, 16, 64)
        out, info = ffn(x)
        assert out.shape == x.shape
    
    def test_routing_info(self):
        ffn = MultiScaleTriXFFN(d_model=64, num_tiles_fine=16, num_tiles_medium=8, num_tiles_coarse=4)
        x = torch.randn(2, 16, 64)
        out, info = ffn(x)
        
        assert info.scale_weights.shape == (2, 16, 3)  # 3 octaves
        assert info.tile_indices.shape == (2, 16, 3)   # tile per octave
        assert info.entropy.shape == (2, 16)
        assert info.mode == "generative"
    
    def test_generative_mode_soft_blend(self):
        ffn = MultiScaleTriXFFN(d_model=64, num_tiles_fine=16, num_tiles_medium=8, num_tiles_coarse=4)
        ffn.set_mode("generative")
        x = torch.randn(2, 16, 64)
        out, info = ffn(x)
        
        # Scale weights should sum to 1 and be soft
        assert torch.allclose(info.scale_weights.sum(dim=-1), torch.ones(2, 16))
        # Not all one-hot (some blending)
        assert not ((info.scale_weights == 0) | (info.scale_weights == 1)).all()
    
    def test_deterministic_mode_fine_only(self):
        ffn = MultiScaleTriXFFN(d_model=64, num_tiles_fine=16, num_tiles_medium=8, num_tiles_coarse=4)
        ffn.set_mode("deterministic")
        x = torch.randn(2, 16, 64)
        out, info = ffn(x)
        
        # Should use only fine scale (octave 0)
        assert (info.scale_weights[..., 0] == 1.0).all()
        assert (info.scale_weights[..., 1] == 0.0).all()
        assert (info.scale_weights[..., 2] == 0.0).all()
        assert info.mode == "deterministic"
    
    def test_deterministic_is_exact(self):
        """Same input should give same output in deterministic mode."""
        ffn = MultiScaleTriXFFN(d_model=64, num_tiles_fine=16, num_tiles_medium=8, num_tiles_coarse=4)
        ffn.set_mode("deterministic")
        ffn.eval()
        
        x = torch.randn(2, 16, 64)
        out1, _ = ffn(x)
        out2, _ = ffn(x)
        
        assert torch.allclose(out1, out2)
    
    def test_gradient_flow_generative(self):
        ffn = MultiScaleTriXFFN(d_model=64, num_tiles_fine=16, num_tiles_medium=8, num_tiles_coarse=4)
        ffn.set_mode("generative")
        x = torch.randn(2, 16, 64)
        out, info = ffn(x)
        out.sum().backward()
        
        # Blend network should get gradients
        assert ffn.blend_net[0].weight.grad is not None
        # Tile scales should get gradients
        assert ffn.octaves[0].tiles[0].up_scale.grad is not None
    
    def test_mode_switch(self):
        ffn = MultiScaleTriXFFN(d_model=64)
        
        assert ffn.mode == "generative"
        ffn.set_mode("deterministic")
        assert ffn.mode == "deterministic"
        ffn.set_mode("generative")
        assert ffn.mode == "generative"
    
    def test_entropy_varies(self):
        """Entropy should vary with input uncertainty."""
        ffn = MultiScaleTriXFFN(d_model=64, num_tiles_fine=16, num_tiles_medium=8, num_tiles_coarse=4)
        x = torch.randn(4, 32, 64)
        out, info = ffn(x)
        
        # Entropy should not be constant
        assert info.entropy.std() > 0


class TestMultiScaleTriXBlock:
    """Tests for full transformer block."""
    
    def test_forward_shape(self):
        block = MultiScaleTriXBlock(d_model=64, n_heads=4)
        x = torch.randn(2, 16, 64)
        out, info = block(x)
        assert out.shape == x.shape
    
    def test_mode_propagates(self):
        block = MultiScaleTriXBlock(d_model=64, n_heads=4)
        block.set_mode("deterministic")
        assert block.ffn.mode == "deterministic"


class TestIntegration:
    """Integration tests."""
    
    def test_stack_of_blocks(self):
        blocks = nn.ModuleList([
            MultiScaleTriXBlock(d_model=64, n_heads=4)
            for _ in range(3)
        ])
        
        x = torch.randn(2, 16, 64)
        for block in blocks:
            x, info = block(x)
        
        assert x.shape == (2, 16, 64)
    
    def test_training_loop(self):
        ffn = MultiScaleTriXFFN(d_model=64)
        optimizer = torch.optim.Adam(ffn.parameters(), lr=1e-3)
        
        for _ in range(3):
            x = torch.randn(2, 16, 64)
            target = torch.randn(2, 16, 64)
            
            out, info = ffn(x)
            loss = F.mse_loss(out, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Should complete without error
        assert True
    
    def test_mode_switch_during_inference(self):
        """Test switching modes for same model."""
        ffn = MultiScaleTriXFFN(d_model=64)
        ffn.eval()
        x = torch.randn(2, 16, 64)
        
        # Generative
        ffn.set_mode("generative")
        out_gen, info_gen = ffn(x)
        
        # Deterministic
        ffn.set_mode("deterministic")
        out_det, info_det = ffn(x)
        
        # Outputs should differ (different routing)
        assert not torch.allclose(out_gen, out_det)
        
        # Deterministic should have zero entropy (no uncertainty)
        assert (info_det.entropy == 0).all()


# Need this import for nn.ModuleList in integration tests
import torch.nn as nn
import torch.nn.functional as F
