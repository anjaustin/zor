"""
Tests for Native Programmable Tiles.

Verifies the CuPy-native Guardian interface works correctly.
"""

import pytest
import numpy as np

# Use numpy as fallback when cupy not available
try:
    import cupy as cp
except ImportError:
    import numpy as cp


class TestNativeProgrammableTile:
    """Tests for single tile."""
    
    def test_creation(self):
        from trix.native import NativeProgrammableTile
        tile = NativeProgrammableTile(d_model=64, d_hidden=128, tile_id=0)
        
        assert tile.signature.shape == (64,)
        assert tile.weights_up.shape == (64, 128)
        assert tile.weights_down.shape == (128, 64)
        assert tile.bias.shape == (128,)
    
    def test_read_interface(self):
        from trix.native import NativeProgrammableTile
        tile = NativeProgrammableTile(d_model=64, d_hidden=128)
        
        sig = tile.read_signature()
        assert sig.shape == (64,)
        
        up, down = tile.read_weights()
        assert up.shape == (64, 128)
        assert down.shape == (128, 64)
        
        grads = tile.read_gradients()
        assert 'signature' in grads
        assert 'weights_up' in grads
    
    def test_write_interface(self):
        from trix.native import NativeProgrammableTile
        tile = NativeProgrammableTile(d_model=64, d_hidden=128)
        
        old_sig = tile.read_signature().copy()
        new_sig = cp.random.randn(64).astype(cp.float32)
        
        result = tile.write_signature(new_sig, blend=0.5, reason='test')
        assert result is True
        assert tile.version == 1
        assert len(tile.history) == 1
        
        # Check blending worked
        current = tile.read_signature()
        expected = 0.5 * old_sig + 0.5 * new_sig
        assert cp.allclose(current, expected, atol=1e-5)
    
    def test_freeze_unfreeze(self):
        from trix.native import NativeProgrammableTile
        tile = NativeProgrammableTile(d_model=64, d_hidden=128)
        
        assert not tile.is_frozen
        
        tile.freeze()
        assert tile.is_frozen
        
        new_sig = cp.random.randn(64).astype(cp.float32)
        result = tile.write_signature(new_sig)
        assert result is False  # Should fail when frozen
        
        tile.unfreeze()
        result = tile.write_signature(new_sig)
        assert result is True
    
    def test_forward_backward(self):
        from trix.native import NativeProgrammableTile
        tile = NativeProgrammableTile(d_model=64, d_hidden=128)
        
        x = cp.random.randn(8, 64).astype(cp.float32)
        out = tile.forward(x)
        assert out.shape == (8, 64)
        
        d_out = cp.random.randn(8, 64).astype(cp.float32)
        d_in = tile.backward(d_out)
        assert d_in.shape == (8, 64)
        
        # Check gradients were accumulated
        grads = tile.read_gradients()
        assert not cp.allclose(grads['weights_up'], 0)
        assert not cp.allclose(grads['weights_down'], 0)
    
    def test_signature_movement(self):
        from trix.native import NativeProgrammableTile
        tile = NativeProgrammableTile(d_model=64, d_hidden=128)
        tile.save_initial_state()
        
        assert tile.signature_movement == 0.0
        
        new_sig = tile.signature + 1.0
        tile.write_signature(new_sig, blend=1.0)
        
        assert tile.signature_movement > 0


class TestNativeProgrammableTileBank:
    """Tests for tile bank."""
    
    def test_creation(self):
        from trix.native import NativeProgrammableTileBank
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        
        assert bank.num_tiles == 8
        assert len(bank.tiles) == 8
    
    def test_get_signatures(self):
        from trix.native import NativeProgrammableTileBank
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        
        sigs = bank.get_signatures()
        assert sigs.shape == (8, 64)
    
    def test_routing(self):
        from trix.native import NativeProgrammableTileBank
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        
        x = cp.random.randn(16, 64).astype(cp.float32)
        tile_idx, scores = bank.route(x)
        
        assert tile_idx.shape == (16,)
        assert scores.shape == (16,)
        assert tile_idx.min() >= 0
        assert tile_idx.max() < 8
    
    def test_forward_backward(self):
        from trix.native import NativeProgrammableTileBank
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        
        x = cp.random.randn(16, 64).astype(cp.float32)
        out = bank.forward(x)
        assert out.shape == (16, 64)
        
        d_out = cp.random.randn(16, 64).astype(cp.float32)
        d_in = bank.backward(d_out)
        assert d_in.shape == (16, 64)
    
    def test_load_balance(self):
        from trix.native import NativeProgrammableTileBank
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        
        # Create skewed routing
        tile_idx = cp.array([0, 0, 0, 0, 1, 1, 2, 3])
        load = bank.get_load_balance(tile_idx)
        
        assert load.shape == (8,)
        assert float(load[0]) == 0.5  # 4/8
        assert float(load.sum()) == pytest.approx(1.0)
    
    def test_signature_diversity(self):
        from trix.native import NativeProgrammableTileBank
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        
        div = bank.get_signature_diversity()
        assert 0 <= div <= 1
    
    def test_freeze_all(self):
        from trix.native import NativeProgrammableTileBank
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        
        bank.freeze_all()
        assert all(t.is_frozen for t in bank.tiles)
        
        bank.unfreeze_all()
        assert all(not t.is_frozen for t in bank.tiles)


class TestNativeTrainingObserver:
    """Tests for training observer."""
    
    def test_creation(self):
        from trix.native import NativeProgrammableTileBank, NativeTrainingObserver
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        observer = NativeTrainingObserver(bank)
        
        assert observer.tile_bank is bank
    
    def test_observe(self):
        from trix.native import NativeProgrammableTileBank, NativeTrainingObserver
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        observer = NativeTrainingObserver(bank)
        
        tile_idx = cp.array([0, 1, 2, 3, 4, 5, 6, 7])
        obs = observer.observe(tile_idx)
        
        assert 'load_balance' in obs
        assert 'diversity' in obs
        assert 'num_unused' in obs
    
    def test_intervention_on_collapse(self):
        from trix.native import NativeProgrammableTileBank, NativeTrainingObserver
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        
        # Force signature collapse
        base_sig = bank.tiles[0].signature.copy()
        for tile in bank.tiles[1:]:
            tile.signature = base_sig.copy()
        
        observer = NativeTrainingObserver(bank, diversity_threshold=0.5)
        
        x = cp.random.randn(16, 64).astype(cp.float32)
        _ = bank.forward(x)
        
        obs = observer.step(bank._cached_tile_idx)
        
        # Should have low diversity and intervene
        assert obs['diversity'] < 0.5
        assert obs['intervened'] is True
        assert obs['intervention_reason'] == 'signature_collapse'
    
    def test_no_intervention_when_healthy(self):
        from trix.native import NativeProgrammableTileBank, NativeTrainingObserver
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        observer = NativeTrainingObserver(bank)
        
        x = cp.random.randn(64, 64).astype(cp.float32)  # More samples for balance
        _ = bank.forward(x)
        
        obs = observer.step(bank._cached_tile_idx)
        
        # Random initialization should be diverse
        if obs['diversity'] > observer.diversity_threshold:
            assert obs['intervened'] is False


class TestIntegration:
    """Integration tests with native trainer."""
    
    def test_training_loop(self):
        from trix.native import (
            NativeProgrammableTileBank,
            NativeTrainingObserver,
            AdamOptimizer,
            mse_loss,
        )
        
        bank = NativeProgrammableTileBank(d_model=32, d_hidden=64, num_tiles=4)
        observer = NativeTrainingObserver(bank)
        optimizer = AdamOptimizer(bank.get_params(), lr=0.01)
        
        # Simple training loop
        losses = []
        for step in range(10):
            bank.zero_grad()
            
            x = cp.random.randn(8, 32).astype(cp.float32)
            target = cp.random.randn(8, 32).astype(cp.float32)
            
            out = bank.forward(x)
            loss, d_loss = mse_loss(out, target)
            bank.backward(d_loss)
            
            optimizer.step(bank.get_grads())
            
            obs = observer.step(bank._cached_tile_idx)
            losses.append(float(loss))
        
        assert len(losses) == 10
        assert len(observer.observations) == 10
    
    def test_params_and_grads(self):
        from trix.native import NativeProgrammableTileBank
        
        bank = NativeProgrammableTileBank(d_model=32, d_hidden=64, num_tiles=4)
        
        params = bank.get_params()
        assert len(params) == 4 * 4  # 4 tiles * 4 params each
        
        x = cp.random.randn(8, 32).astype(cp.float32)
        out = bank.forward(x)
        d_out = cp.random.randn(8, 32).astype(cp.float32)
        bank.backward(d_out)
        
        grads = bank.get_grads()
        assert len(grads) == 4 * 4
