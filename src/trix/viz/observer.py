"""
Observer: How we capture without disturbing.

Hooks into SDPMX pipeline. Samples. Derives. Queues.
The ghost that watches computation heal itself.
"""

import time
import threading
import queue
from typing import Optional, Dict, Any, Callable
from enum import IntEnum
from dataclasses import dataclass

import torch

from .metrics import extract_observation, Observation
from .buffer import ObservationStore


class LogLevel(IntEnum):
    """Hierarchical logging levels."""
    HEARTBEAT = 0   # Just loss and step
    TRAJECTORY = 1  # + health, mask, routing
    DIAGNOSTIC = 2  # + per-dim stats, gradients
    FULL = 3        # + all tensors


@dataclass
class ObserverConfig:
    """Configuration for the observer."""
    level: LogLevel = LogLevel.TRAJECTORY
    sample_rate: int = 10           # Log every N steps
    ring_size: int = 1000           # Ring buffer size
    queue_maxsize: int = 100        # Async queue depth
    async_logging: bool = True      # Use background thread
    adaptive_sampling: bool = True  # More samples on anomalies


class SDPMXObserver:
    """
    Observer that attaches to SDPMX pipeline.

    Zero-impact at Level 0. Full visibility at Level 3.
    """

    def __init__(
        self,
        pipeline,  # SDPMXPipeline instance
        config: Optional[ObserverConfig] = None,
    ):
        self.pipeline = pipeline
        self.config = config or ObserverConfig()

        # Storage
        self.store = ObservationStore(
            ring_size=self.config.ring_size,
            num_subspaces=getattr(pipeline.P, 'num_subspaces', 8),
        )

        # State
        self.step = 0
        self.prev_health = None
        self.current_loss = None
        self.current_lr = None

        # Async infrastructure
        self._queue: queue.Queue = queue.Queue(maxsize=self.config.queue_maxsize)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Hooks
        self._hooks = []

        # Start async worker if enabled
        if self.config.async_logging:
            self._start_worker()

    def _start_worker(self) -> None:
        """Start background logging thread."""
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Background thread that processes observations."""
        while not self._stop_event.is_set():
            try:
                obs = self._queue.get(timeout=0.1)
                self.store.record(obs)
            except queue.Empty:
                continue

    def _should_sample(self) -> bool:
        """Decide whether to sample this step."""
        # Always sample at rate
        if self.step % self.config.sample_rate == 0:
            return True

        # Adaptive: sample more on anomalies
        if self.config.adaptive_sampling and self.store.latest:
            latest = self.store.latest
            # Sample if health dropped significantly
            if latest.health.trend < -0.1:
                return True
            # Sample if mask suddenly dense
            if latest.mask_stats.density > 0.8:
                return True

        return False

    def observe(
        self,
        x_input: torch.Tensor,
        x_output: torch.Tensor,
        intermediates: Dict[str, Any],
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        """
        Record an observation from a forward pass.

        Call this after pipeline.forward() with the intermediates.
        """
        self.step += 1

        # Check sampling
        if not self._should_sample():
            return

        # Extract observation
        obs = extract_observation(
            step=self.step,
            timestamp=time.time(),
            x_input=x_input,
            intermediates=intermediates,
            x_output=x_output,
            projection_operator=self.pipeline.P,
            prev_health=self.prev_health,
            loss=loss or self.current_loss,
            learning_rate=learning_rate or self.current_lr,
            include_tensors=(self.config.level >= LogLevel.FULL),
        )

        # Update state
        self.prev_health = obs.health.value

        # Record (async or sync)
        if self.config.async_logging:
            try:
                self._queue.put_nowait(obs)
            except queue.Full:
                pass  # Drop if queue full (zero-impact policy)
        else:
            self.store.record(obs)

    def set_training_context(
        self,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        """Update training context for next observation."""
        if loss is not None:
            self.current_loss = loss
        if learning_rate is not None:
            self.current_lr = learning_rate

    def attach(self) -> None:
        """Attach hooks to pipeline (alternative to manual observe calls)."""
        # Store original forward
        original_forward = self.pipeline.forward

        def hooked_forward(x, return_intermediates=True, **kwargs):
            # Always request intermediates for observation
            result = original_forward(x, return_intermediates=True, **kwargs)
            x_output, info = result

            # Observe
            self.observe(
                x_input=x,
                x_output=x_output,
                intermediates=info,
            )

            # Return in original format
            if return_intermediates:
                return x_output, info
            else:
                return x_output, {}

        self.pipeline.forward = hooked_forward
        self._hooks.append(('forward', original_forward))

    def detach(self) -> None:
        """Remove hooks from pipeline."""
        for name, original in self._hooks:
            if name == 'forward':
                self.pipeline.forward = original
        self._hooks.clear()

    def stop(self) -> None:
        """Stop observer and clean up."""
        self.detach()
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)

    # Convenience accessors
    @property
    def latest(self) -> Optional[Observation]:
        return self.store.latest

    @property
    def summary(self) -> Dict[str, Any]:
        return self.store.get_summary()

    def get_sparkline(self, metric: str, length: int = 50) -> str:
        """Get ASCII sparkline for a metric."""
        data = self.store.get_sparkline_data(metric, length)
        if not data:
            return ""

        # Normalize to 0-1
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val > min_val else 1.0

        normalized = [(v - min_val) / range_val for v in data]

        # Map to sparkline characters
        chars = "▁▂▃▄▅▆▇█"
        return "".join(chars[min(7, int(v * 7.99))] for v in normalized)

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *args):
        self.stop()


def create_observer(
    pipeline,
    level: LogLevel = LogLevel.TRAJECTORY,
    sample_rate: int = 10,
) -> SDPMXObserver:
    """Factory function for creating observers."""
    config = ObserverConfig(
        level=level,
        sample_rate=sample_rate,
    )
    return SDPMXObserver(pipeline, config)
