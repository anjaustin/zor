"""
Integration tests for SDPMX Visualizer.

Tests the complete observation → storage → display pipeline.
"""

import pytest
import torch
import time
import threading

from trix.nn import SDPMXPipeline
from trix.viz import (
    create_observer,
    LogLevel,
    ObserverConfig,
    SDPMXObserver,
    RingBuffer,
    Aggregates,
    ObservationStore,
    compute_delta,
    compute_health,
    compute_mask_stats,
    HealthScore,
    MaskStats,
    SimpleTUI,
)


# =============================================================================
# METRICS TESTS
# =============================================================================

class TestMetrics:
    """Test metric computations."""

    def test_compute_delta(self):
        """Delta computation works correctly."""
        x_in = torch.randn(8, 32)
        x_out = x_in + 0.1 * torch.randn(8, 32)

        delta = compute_delta(x_in, x_out, 'test')

        assert delta.name == 'test'
        assert delta.l2_norm > 0
        assert delta.linf_norm > 0
        assert delta.mean_abs > 0

    def test_compute_mask_stats(self):
        """Mask statistics work correctly."""
        # All ones
        mask = torch.ones(8, 32)
        stats = compute_mask_stats(mask)
        assert stats.density == 1.0
        assert stats.num_active == 256
        assert stats.is_dense

        # Half
        mask = torch.zeros(8, 32)
        mask[:, :16] = 1.0
        stats = compute_mask_stats(mask)
        assert stats.density == 0.5

        # Sparse
        mask = torch.zeros(8, 32)
        mask[:, 0] = 1.0
        stats = compute_mask_stats(mask)
        assert stats.is_sparse

    def test_health_score_properties(self):
        """Health score has correct status mappings."""
        # Healthy
        h = HealthScore(value=0.95, distance=0.1, subspace_id=0, trend=0.05)
        assert h.status == 'healthy'
        assert h.trend_arrow == '↗'

        # Stuck
        h = HealthScore(value=0.3, distance=0.8, subspace_id=0, trend=-0.1)
        assert h.status == 'stuck'
        assert h.trend_arrow == '↘'


# =============================================================================
# BUFFER TESTS
# =============================================================================

class TestBuffer:
    """Test buffer infrastructure."""

    def test_ring_buffer_basic(self):
        """Ring buffer stores and retrieves correctly."""
        from trix.viz.metrics import Observation

        buffer = RingBuffer(maxsize=10)

        # Add observations
        for i in range(5):
            obs = Observation(
                step=i,
                timestamp=time.time(),
                health=HealthScore(0.8, 0.1, 0, 0.0),
                mask_stats=MaskStats(0.5, 16, 32),
                intervention_magnitude=0.1,
                deltas={},
            )
            buffer.push(obs)

        assert len(buffer) == 5
        assert buffer.latest.step == 4

    def test_ring_buffer_overflow(self):
        """Ring buffer drops old data when full."""
        from trix.viz.metrics import Observation

        buffer = RingBuffer(maxsize=5)

        for i in range(10):
            obs = Observation(
                step=i,
                timestamp=time.time(),
                health=HealthScore(0.8, 0.1, 0, 0.0),
                mask_stats=MaskStats(0.5, 16, 32),
                intervention_magnitude=0.1,
                deltas={},
            )
            buffer.push(obs)

        # Should only have last 5
        assert len(buffer) == 5
        assert buffer.get_all()[0].step == 5
        assert buffer.latest.step == 9

    def test_aggregates_update(self):
        """Running aggregates compute correctly."""
        from trix.viz.metrics import Observation

        agg = Aggregates()

        for i in range(100):
            obs = Observation(
                step=i,
                timestamp=time.time(),
                health=HealthScore(0.8 + 0.001 * i, 0.1, i % 4, 0.0),
                mask_stats=MaskStats(0.5, 16, 32),
                intervention_magnitude=0.1,
                deltas={},
                loss=1.0 - 0.005 * i,
            )
            agg.update(obs)

        assert agg.health.count == 100
        assert 0.8 < agg.health.mean < 0.9
        assert agg.loss.mean < 1.0  # Should have decreased
        assert len(agg.subspace_routing.counts) == 4  # Used 4 subspaces


# =============================================================================
# OBSERVER TESTS
# =============================================================================

class TestObserver:
    """Test observer functionality."""

    def test_observer_creation(self):
        """Observer can be created."""
        pipeline = SDPMXPipeline(dim=32, grid_size=8, num_subspaces=4)
        observer = create_observer(pipeline, level=LogLevel.TRAJECTORY)

        assert observer.step == 0
        assert observer.config.level == LogLevel.TRAJECTORY

    def test_observer_manual_observe(self):
        """Manual observation works."""
        pipeline = SDPMXPipeline(dim=32, grid_size=8, num_subspaces=4)
        observer = create_observer(pipeline, level=LogLevel.TRAJECTORY, sample_rate=1)

        x = torch.randn(8, 32)
        x_new, info = pipeline(x, return_intermediates=True)

        observer.observe(x, x_new, info, loss=0.5)

        assert observer.step == 1
        # Give async worker time to process
        time.sleep(0.2)
        assert observer.store.aggregates.health.count >= 1

    def test_observer_attach_hook(self):
        """Attached hooks capture observations."""
        pipeline = SDPMXPipeline(dim=32, grid_size=8, num_subspaces=4)
        observer = create_observer(pipeline, level=LogLevel.TRAJECTORY, sample_rate=1)

        observer.attach()

        # Forward passes should be observed
        for _ in range(10):
            x = torch.randn(8, 32)
            pipeline(x)

        observer.detach()
        observer.stop()

        assert observer.step == 10

    def test_observer_context_manager(self):
        """Observer works as context manager."""
        pipeline = SDPMXPipeline(dim=32, grid_size=8, num_subspaces=4)

        with create_observer(pipeline, sample_rate=1) as observer:
            for _ in range(5):
                x = torch.randn(8, 32)
                pipeline(x)

            assert observer.step == 5

    def test_observer_sampling(self):
        """Observer respects sample rate."""
        pipeline = SDPMXPipeline(dim=32, grid_size=8, num_subspaces=4)
        # Disable adaptive sampling for predictable counts
        config = ObserverConfig(sample_rate=10, adaptive_sampling=False)
        observer = SDPMXObserver(pipeline, config)

        observer.attach()

        for _ in range(100):
            x = torch.randn(8, 32)
            pipeline(x)

        observer.stop()

        # Should have observed ~10 times (every 10 steps)
        time.sleep(0.2)  # Let async worker process
        assert 8 <= observer.store.aggregates.health.count <= 12

    def test_observer_summary(self):
        """Summary contains expected fields."""
        pipeline = SDPMXPipeline(dim=32, grid_size=8, num_subspaces=4)
        observer = create_observer(pipeline, sample_rate=1)

        observer.attach()
        x = torch.randn(8, 32)
        pipeline(x)
        observer.set_training_context(loss=0.5)
        pipeline(x)
        observer.stop()

        time.sleep(0.2)
        summary = observer.summary

        assert 'step' in summary
        assert 'health' in summary
        assert 'health_trend' in summary
        assert 'mask_density' in summary
        assert 'subspace_histogram' in summary

    def test_sparkline_generation(self):
        """Sparklines generate correctly."""
        pipeline = SDPMXPipeline(dim=32, grid_size=8, num_subspaces=4)
        config = ObserverConfig(sample_rate=1, async_logging=False)
        observer = SDPMXObserver(pipeline, config)

        observer.attach()
        for _ in range(20):
            x = torch.randn(8, 32)
            pipeline(x)
        observer.stop()

        spark = observer.get_sparkline('health', length=10)
        assert len(spark) == 10
        assert all(c in "▁▂▃▄▅▆▇█" for c in spark)


# =============================================================================
# TUI TESTS
# =============================================================================

class TestTUI:
    """Test terminal UI components."""

    def test_simple_tui_format(self):
        """Simple TUI formats correctly."""
        pipeline = SDPMXPipeline(dim=32, grid_size=8, num_subspaces=4)
        config = ObserverConfig(sample_rate=1, async_logging=False)
        observer = SDPMXObserver(pipeline, config)

        observer.attach()
        x = torch.randn(8, 32)
        pipeline(x)
        observer.stop()

        tui = SimpleTUI(observer)
        line = tui.format_line()

        assert 'Step' in line
        assert 'Health' in line
        assert 'Mask' in line

    def test_make_bar(self):
        """Progress bar generation works."""
        from trix.viz.tui import make_bar

        bar = make_bar(0.5, width=10)
        assert len(bar) == 10
        assert bar.count('█') == 5
        assert bar.count('░') == 5

        bar = make_bar(1.0, width=10)
        assert bar == '█' * 10

        bar = make_bar(0.0, width=10)
        assert bar == '░' * 10


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestFullIntegration:
    """Full pipeline integration tests."""

    def test_training_loop_with_observer(self):
        """Observer works in a training loop."""
        pipeline = SDPMXPipeline(dim=32, grid_size=8, num_subspaces=4)
        optimizer = torch.optim.Adam(pipeline.parameters(), lr=0.01)

        observer = create_observer(pipeline, sample_rate=5)
        observer.attach()

        for i in range(50):
            x = torch.randn(16, 32)
            target = torch.tanh(x)

            optimizer.zero_grad()
            x_new, loss, _ = pipeline.forward_with_loss(x, target)
            loss.backward()
            optimizer.step()

            observer.set_training_context(loss=loss.item())

        observer.stop()
        time.sleep(0.2)

        # Should have observations
        assert observer.store.aggregates.health.count > 0
        assert observer.store.aggregates.loss.count > 0

        # Health should be tracked
        summary = observer.summary
        assert summary['health'] > 0

    def test_zero_impact_at_level_0(self):
        """Level 0 logging has minimal overhead."""
        pipeline = SDPMXPipeline(dim=64, grid_size=16, num_subspaces=8)

        # Warmup to stabilize CPU frequency (important for DVFS platforms like Jetson)
        for _ in range(200):
            x = torch.randn(32, 64)
            pipeline(x)

        # Run multiple trials and take median to reduce DVFS noise
        overheads = []
        for _ in range(3):
            # Time without observer
            start = time.time()
            for _ in range(200):
                x = torch.randn(32, 64)
                pipeline(x)
            time_without = time.time() - start

            # Time with level 0 observer
            observer = create_observer(pipeline, level=LogLevel.HEARTBEAT, sample_rate=100)
            observer.attach()

            start = time.time()
            for _ in range(200):
                x = torch.randn(32, 64)
                pipeline(x)
            time_with = time.time() - start

            observer.stop()

            overheads.append((time_with - time_without) / time_without)

        # Use median overhead to reduce DVFS noise on embedded platforms
        median_overhead = sorted(overheads)[len(overheads) // 2]
        assert median_overhead < 0.20, f"Median overhead too high: {median_overhead*100:.1f}%"

    def test_concurrent_access(self):
        """Observer handles concurrent reads/writes."""
        pipeline = SDPMXPipeline(dim=32, grid_size=8, num_subspaces=4)
        observer = create_observer(pipeline, sample_rate=1)
        observer.attach()

        errors = []

        def writer():
            try:
                for _ in range(50):
                    x = torch.randn(8, 32)
                    pipeline(x)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    _ = observer.summary
                    _ = observer.get_sparkline('health')
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        observer.stop()

        assert len(errors) == 0, f"Concurrent access errors: {errors}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
