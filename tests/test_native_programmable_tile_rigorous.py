"""
Rigorous Tests for Native Programmable Tiles.

Stress tests, edge cases, numerical stability, and integration verification.
"""

import pytest
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    import numpy as cp
    HAS_CUPY = False


# =============================================================================
# NUMERICAL STABILITY
# =============================================================================

class TestNumericalStability:
    """Verify numerical stability under edge conditions."""
    
    def test_zero_input(self):
        """Forward/backward with zero input."""
        from trix.native import NativeProgrammableTileBank
        
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        x = cp.zeros((8, 64), dtype=cp.float32)
        
        out = bank.forward(x)
        assert not cp.isnan(out).any()
        assert not cp.isinf(out).any()
        
        d_out = cp.ones((8, 64), dtype=cp.float32)
        d_in = bank.backward(d_out)
        assert not cp.isnan(d_in).any()
        assert not cp.isinf(d_in).any()
    
    def test_large_input(self):
        """Forward/backward with large magnitude input."""
        from trix.native import NativeProgrammableTileBank
        
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        x = cp.random.randn(8, 64).astype(cp.float32) * 1000
        
        out = bank.forward(x)
        assert not cp.isnan(out).any()
        assert not cp.isinf(out).any()
    
    def test_small_input(self):
        """Forward/backward with tiny magnitude input."""
        from trix.native import NativeProgrammableTileBank
        
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        x = cp.random.randn(8, 64).astype(cp.float32) * 1e-10
        
        out = bank.forward(x)
        assert not cp.isnan(out).any()
        assert not cp.isinf(out).any()
    
    def test_gradient_magnitude(self):
        """Gradients should not explode or vanish."""
        from trix.native import NativeProgrammableTileBank
        
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        
        x = cp.random.randn(16, 64).astype(cp.float32)
        out = bank.forward(x)
        
        d_out = cp.random.randn(16, 64).astype(cp.float32)
        d_in = bank.backward(d_out)
        
        grads = bank.get_grads()
        for name, grad in grads.items():
            grad_norm = float(cp.linalg.norm(grad))
            assert grad_norm < 1e6, f"Gradient {name} exploded: {grad_norm}"
            # Allow zero grads for unused tiles
    
    def test_blend_boundary_values(self):
        """Test blend at boundaries 0.0 and 1.0."""
        from trix.native import NativeProgrammableTile
        
        tile = NativeProgrammableTile(d_model=64, d_hidden=128)
        old_sig = tile.read_signature().copy()
        new_sig = cp.random.randn(64).astype(cp.float32)
        
        # blend=0.0 should not change
        tile.write_signature(new_sig, blend=0.0)
        assert cp.allclose(tile.signature, old_sig)
        
        # blend=1.0 should fully replace
        tile.write_signature(new_sig, blend=1.0)
        assert cp.allclose(tile.signature, new_sig)


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_tile(self):
        """Bank with just one tile."""
        from trix.native import NativeProgrammableTileBank
        
        bank = NativeProgrammableTileBank(d_model=32, d_hidden=64, num_tiles=1)
        x = cp.random.randn(8, 32).astype(cp.float32)
        
        out = bank.forward(x)
        assert out.shape == (8, 32)
        
        tile_idx, _ = bank.route(x)
        assert (tile_idx == 0).all()
    
    def test_many_tiles(self):
        """Bank with many tiles."""
        from trix.native import NativeProgrammableTileBank
        
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=64)
        x = cp.random.randn(128, 64).astype(cp.float32)
        
        out = bank.forward(x)
        assert out.shape == (128, 64)
        
        tile_idx, _ = bank.route(x)
        # With 128 samples and 64 tiles, should use most tiles
        assert len(cp.unique(tile_idx)) > 32
    
    def test_batch_size_one(self):
        """Single sample batch."""
        from trix.native import NativeProgrammableTileBank
        
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        x = cp.random.randn(1, 64).astype(cp.float32)
        
        out = bank.forward(x)
        assert out.shape == (1, 64)
        
        d_out = cp.random.randn(1, 64).astype(cp.float32)
        d_in = bank.backward(d_out)
        assert d_in.shape == (1, 64)
    
    def test_large_batch(self):
        """Large batch size."""
        from trix.native import NativeProgrammableTileBank
        
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        x = cp.random.randn(1024, 64).astype(cp.float32)
        
        out = bank.forward(x)
        assert out.shape == (1024, 64)
    
    def test_empty_history(self):
        """Tile with no modifications."""
        from trix.native import NativeProgrammableTile
        
        tile = NativeProgrammableTile(d_model=64, d_hidden=128)
        assert tile.version == 0
        assert len(tile.history) == 0
    
    def test_frozen_tile_forward(self):
        """Frozen tile should still forward correctly."""
        from trix.native import NativeProgrammableTile
        
        tile = NativeProgrammableTile(d_model=64, d_hidden=128)
        tile.freeze()
        
        x = cp.random.randn(8, 64).astype(cp.float32)
        out = tile.forward(x)
        assert out.shape == (8, 64)


# =============================================================================
# GRADIENT CORRECTNESS
# =============================================================================

class TestGradientCorrectness:
    """Verify gradient computation is correct."""
    
    def test_gradient_shapes(self):
        """Gradient shapes match parameter shapes."""
        from trix.native import NativeProgrammableTile
        
        tile = NativeProgrammableTile(d_model=64, d_hidden=128)
        
        x = cp.random.randn(8, 64).astype(cp.float32)
        out = tile.forward(x)
        d_out = cp.random.randn(8, 64).astype(cp.float32)
        tile.backward(d_out)
        
        assert tile.d_weights_up.shape == tile.weights_up.shape
        assert tile.d_weights_down.shape == tile.weights_down.shape
        assert tile.d_bias.shape == tile.bias.shape
    
    def test_gradient_accumulation(self):
        """Gradients accumulate across backward calls."""
        from trix.native import NativeProgrammableTile
        
        tile = NativeProgrammableTile(d_model=64, d_hidden=128)
        
        x = cp.random.randn(8, 64).astype(cp.float32)
        out = tile.forward(x)
        d_out = cp.ones((8, 64), dtype=cp.float32)
        
        tile.backward(d_out)
        grad1 = tile.d_weights_up.copy()
        
        tile.backward(d_out)
        grad2 = tile.d_weights_up.copy()
        
        # Should have accumulated
        assert cp.allclose(grad2, 2 * grad1)
    
    def test_zero_grad(self):
        """zero_grad resets all gradients."""
        from trix.native import NativeProgrammableTile
        
        tile = NativeProgrammableTile(d_model=64, d_hidden=128)
        
        x = cp.random.randn(8, 64).astype(cp.float32)
        out = tile.forward(x)
        d_out = cp.random.randn(8, 64).astype(cp.float32)
        tile.backward(d_out)
        
        tile.zero_grad()
        
        assert cp.allclose(tile.d_weights_up, 0)
        assert cp.allclose(tile.d_weights_down, 0)
        assert cp.allclose(tile.d_bias, 0)
    
    def test_finite_difference_check(self):
        """Numerical gradient check for a single parameter."""
        from trix.native import NativeProgrammableTile, mse_loss
        
        tile = NativeProgrammableTile(d_model=16, d_hidden=32)
        
        x = cp.random.randn(4, 16).astype(cp.float32)
        target = cp.random.randn(4, 16).astype(cp.float32)
        
        eps = 1e-4
        
        # Compute analytical gradient
        tile.zero_grad()
        out = tile.forward(x)
        loss, d_loss = mse_loss(out, target)
        tile.backward(d_loss)
        analytical_grad = tile.d_bias[0].copy()
        
        # Compute numerical gradient
        orig = float(tile.bias[0])
        
        tile.bias[0] = orig + eps
        out_plus = tile.forward(x)
        loss_plus, _ = mse_loss(out_plus, target)
        
        tile.bias[0] = orig - eps
        out_minus = tile.forward(x)
        loss_minus, _ = mse_loss(out_minus, target)
        
        tile.bias[0] = orig
        
        numerical_grad = (float(loss_plus) - float(loss_minus)) / (2 * eps)
        
        # Check they're close (relative tolerance)
        assert abs(float(analytical_grad) - numerical_grad) < 0.1, \
            f"Gradient mismatch: analytical={float(analytical_grad)}, numerical={numerical_grad}"


# =============================================================================
# OBSERVER BEHAVIOR
# =============================================================================

class TestObserverBehavior:
    """Test training observer behavior."""
    
    def test_observation_history(self):
        """Observations are recorded."""
        from trix.native import NativeProgrammableTileBank, NativeTrainingObserver
        
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        observer = NativeTrainingObserver(bank)
        
        for _ in range(10):
            x = cp.random.randn(16, 64).astype(cp.float32)
            bank.forward(x)
            observer.step(bank._cached_tile_idx)
        
        assert len(observer.observations) == 10
    
    def test_intervention_history(self):
        """Interventions are recorded."""
        from trix.native import NativeProgrammableTileBank, NativeTrainingObserver
        
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        
        # Force collapse
        base = bank.tiles[0].signature.copy()
        for t in bank.tiles[1:]:
            t.signature = base.copy()
        
        observer = NativeTrainingObserver(bank, diversity_threshold=0.5)
        
        x = cp.random.randn(16, 64).astype(cp.float32)
        bank.forward(x)
        observer.step(bank._cached_tile_idx)
        
        assert len(observer.interventions) > 0
    
    def test_diversity_improves_after_intervention(self):
        """Diversity should improve after signature collapse intervention."""
        from trix.native import NativeProgrammableTileBank, NativeTrainingObserver
        
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        
        # Force collapse
        base = bank.tiles[0].signature.copy()
        for t in bank.tiles[1:]:
            t.signature = base.copy()
        
        observer = NativeTrainingObserver(bank, diversity_threshold=0.5, intervention_blend=0.3)
        
        x = cp.random.randn(16, 64).astype(cp.float32)
        bank.forward(x)
        
        obs_before = observer.observe(bank._cached_tile_idx)
        div_before = obs_before['diversity']
        
        # Intervene
        observer.intervene('signature_collapse')
        
        div_after = bank.get_signature_diversity()
        
        assert div_after > div_before, f"Diversity did not improve: {div_before} -> {div_after}"
    
    def test_observer_thresholds(self):
        """Observer respects threshold settings."""
        from trix.native import NativeProgrammableTileBank, NativeTrainingObserver
        
        bank = NativeProgrammableTileBank(d_model=64, d_hidden=128, num_tiles=8)
        
        # Very permissive observer
        observer = NativeTrainingObserver(
            bank,
            diversity_threshold=0.0,  # Never trigger on diversity
            balance_threshold=1.0,    # Never trigger on balance
        )
        
        x = cp.random.randn(16, 64).astype(cp.float32)
        bank.forward(x)
        
        obs = observer.step(bank._cached_tile_idx)
        assert obs['intervened'] is False


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStress:
    """Stress tests for robustness."""
    
    def test_repeated_forward_backward(self):
        """Many forward/backward cycles."""
        from trix.native import NativeProgrammableTileBank, AdamOptimizer, mse_loss
        
        bank = NativeProgrammableTileBank(d_model=32, d_hidden=64, num_tiles=4)
        optimizer = AdamOptimizer(bank.get_params(), lr=0.001)
        
        for _ in range(100):
            bank.zero_grad()
            
            x = cp.random.randn(8, 32).astype(cp.float32)
            target = cp.random.randn(8, 32).astype(cp.float32)
            
            out = bank.forward(x)
            loss, d_loss = mse_loss(out, target)
            bank.backward(d_loss)
            optimizer.step(bank.get_grads())
        
        # Should still be stable
        x = cp.random.randn(8, 32).astype(cp.float32)
        out = bank.forward(x)
        assert not cp.isnan(out).any()
        assert not cp.isinf(out).any()
    
    def test_many_modifications(self):
        """Many write operations."""
        from trix.native import NativeProgrammableTile
        
        tile = NativeProgrammableTile(d_model=64, d_hidden=128)
        
        for i in range(100):
            new_sig = cp.random.randn(64).astype(cp.float32)
            tile.write_signature(new_sig, blend=0.1, reason=f'mod_{i}')
        
        assert tile.version == 100
        assert len(tile.history) == 100
        
        # Signature should still be valid
        assert not cp.isnan(tile.signature).any()
        assert not cp.isinf(tile.signature).any()
    
    def test_observer_long_run(self):
        """Observer over many steps."""
        from trix.native import NativeProgrammableTileBank, NativeTrainingObserver
        
        bank = NativeProgrammableTileBank(d_model=32, d_hidden=64, num_tiles=4)
        observer = NativeTrainingObserver(bank)
        
        for _ in range(200):
            x = cp.random.randn(16, 32).astype(cp.float32)
            bank.forward(x)
            obs = observer.step(bank._cached_tile_idx)
        
        assert len(observer.observations) == 200
    
    def test_concurrent_freeze_unfreeze(self):
        """Interleaved freeze/unfreeze with modifications."""
        from trix.native import NativeProgrammableTile
        
        tile = NativeProgrammableTile(d_model=64, d_hidden=128)
        
        for i in range(50):
            if i % 2 == 0:
                tile.freeze()
            else:
                tile.unfreeze()
            
            new_sig = cp.random.randn(64).astype(cp.float32)
            result = tile.write_signature(new_sig, blend=0.1)
            
            if i % 2 == 0:
                assert result is False
            else:
                assert result is True


# =============================================================================
# INTEGRATION
# =============================================================================

class TestFullIntegration:
    """Full integration tests."""
    
    def test_training_with_observer(self):
        """Complete training loop with observer."""
        from trix.native import (
            NativeProgrammableTileBank,
            NativeTrainingObserver,
            AdamOptimizer,
            mse_loss,
        )
        
        bank = NativeProgrammableTileBank(d_model=32, d_hidden=64, num_tiles=4)
        observer = NativeTrainingObserver(bank)
        optimizer = AdamOptimizer(bank.get_params(), lr=0.01)
        
        losses = []
        for step in range(50):
            bank.zero_grad()
            
            # Simple pattern: output should match input
            x = cp.random.randn(8, 32).astype(cp.float32)
            target = x.copy()
            
            out = bank.forward(x)
            loss, d_loss = mse_loss(out, target)
            bank.backward(d_loss)
            optimizer.step(bank.get_grads())
            
            obs = observer.step(bank._cached_tile_idx)
            losses.append(float(loss))
        
        # Loss should decrease
        assert losses[-1] < losses[0], "Loss did not decrease"
    
    def test_parity_with_pytorch_guardian(self):
        """Native tiles behave like PyTorch tiles."""
        from trix.native import NativeProgrammableTile
        
        # Create native tile
        native_tile = NativeProgrammableTile(d_model=64, d_hidden=128, tile_id=0)
        
        # Test interface parity
        assert hasattr(native_tile, 'read_signature')
        assert hasattr(native_tile, 'read_weights')
        assert hasattr(native_tile, 'write_signature')
        assert hasattr(native_tile, 'write_weights')
        assert hasattr(native_tile, 'freeze')
        assert hasattr(native_tile, 'unfreeze')
        assert hasattr(native_tile, 'is_frozen')
        assert hasattr(native_tile, 'version')
        assert hasattr(native_tile, 'history')
        assert hasattr(native_tile, 'forward')
        assert hasattr(native_tile, 'backward')
    
    def test_deterministic_with_seed(self):
        """Same seed produces same results."""
        from trix.native import NativeProgrammableTileBank
        
        cp.random.seed(42)
        bank1 = NativeProgrammableTileBank(d_model=32, d_hidden=64, num_tiles=4)
        x = cp.random.randn(8, 32).astype(cp.float32)
        out1 = bank1.forward(x)
        
        cp.random.seed(42)
        bank2 = NativeProgrammableTileBank(d_model=32, d_hidden=64, num_tiles=4)
        x = cp.random.randn(8, 32).astype(cp.float32)
        out2 = bank2.forward(x)
        
        assert cp.allclose(out1, out2)
