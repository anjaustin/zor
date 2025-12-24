"""
Buffer: How we store observations.

Ring buffer for recent history. Running aggregates for all-time stats.
Zero-copy where possible. Memory-bounded always.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator
import threading
import time

from .metrics import Observation, HealthScore, MaskStats


@dataclass
class RunningStats:
    """Running statistics that update incrementally."""
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0  # For Welford's online variance
    min_val: float = float('inf')
    max_val: float = float('-inf')

    def update(self, value: float) -> None:
        """Update stats with new value using Welford's algorithm."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return self.variance ** 0.5

    def __repr__(self):
        return f"μ={self.mean:.4f} σ={self.std:.4f} [{self.min_val:.4f}, {self.max_val:.4f}]"


@dataclass
class SubspaceCounter:
    """Count routing decisions per subspace."""
    counts: Dict[int, int] = field(default_factory=dict)
    total: int = 0

    def update(self, subspace_id: int) -> None:
        self.counts[subspace_id] = self.counts.get(subspace_id, 0) + 1
        self.total += 1

    def get_distribution(self) -> Dict[int, float]:
        """Get normalized distribution."""
        if self.total == 0:
            return {}
        return {k: v / self.total for k, v in sorted(self.counts.items())}

    def get_histogram(self, num_subspaces: int) -> List[float]:
        """Get histogram as list (for display)."""
        return [self.counts.get(i, 0) / max(1, self.total) for i in range(num_subspaces)]


@dataclass
class Aggregates:
    """All-time aggregate statistics."""
    health: RunningStats = field(default_factory=RunningStats)
    mask_density: RunningStats = field(default_factory=RunningStats)
    intervention: RunningStats = field(default_factory=RunningStats)
    loss: RunningStats = field(default_factory=RunningStats)
    subspace_routing: SubspaceCounter = field(default_factory=SubspaceCounter)

    # Per-operator delta stats
    delta_stats: Dict[str, RunningStats] = field(default_factory=dict)

    def update(self, obs: Observation) -> None:
        """Update all aggregates with new observation."""
        self.health.update(obs.health.value)
        self.mask_density.update(obs.mask_stats.density)
        self.intervention.update(obs.intervention_magnitude)

        if obs.loss is not None:
            self.loss.update(obs.loss)

        if obs.health.subspace_id >= 0:
            self.subspace_routing.update(obs.health.subspace_id)

        for name, delta in obs.deltas.items():
            if name not in self.delta_stats:
                self.delta_stats[name] = RunningStats()
            self.delta_stats[name].update(delta.l2_norm)


class RingBuffer:
    """
    Fixed-size circular buffer for recent observations.

    Thread-safe for single-producer, single-consumer pattern.
    """

    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self._buffer: deque = deque(maxlen=maxsize)
        self._lock = threading.Lock()

    def push(self, obs: Observation) -> None:
        """Add observation. Oldest is dropped if full."""
        with self._lock:
            self._buffer.append(obs)

    def get_recent(self, n: int = 100) -> List[Observation]:
        """Get last n observations."""
        with self._lock:
            return list(self._buffer)[-n:]

    def get_all(self) -> List[Observation]:
        """Get all observations in buffer."""
        with self._lock:
            return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> Iterator[Observation]:
        with self._lock:
            return iter(list(self._buffer))

    @property
    def latest(self) -> Optional[Observation]:
        """Get most recent observation."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def get_health_series(self) -> List[float]:
        """Get health values as time series."""
        with self._lock:
            return [obs.health.value for obs in self._buffer]

    def get_mask_density_series(self) -> List[float]:
        """Get mask density as time series."""
        with self._lock:
            return [obs.mask_stats.density for obs in self._buffer]

    def get_intervention_series(self) -> List[float]:
        """Get intervention magnitude as time series."""
        with self._lock:
            return [obs.intervention_magnitude for obs in self._buffer]

    def get_loss_series(self) -> List[float]:
        """Get loss values as time series (filtering None)."""
        with self._lock:
            return [obs.loss for obs in self._buffer if obs.loss is not None]


class ObservationStore:
    """
    Complete observation storage system.

    Combines ring buffer (recent) with aggregates (all-time).
    """

    def __init__(
        self,
        ring_size: int = 1000,
        num_subspaces: int = 8,
    ):
        self.ring = RingBuffer(maxsize=ring_size)
        self.aggregates = Aggregates()
        self.num_subspaces = num_subspaces

        # Timing
        self.start_time = time.time()
        self.last_step = 0
        self.last_time = self.start_time

    def record(self, obs: Observation) -> None:
        """Record a new observation."""
        self.ring.push(obs)
        self.aggregates.update(obs)

        self.last_step = obs.step
        self.last_time = obs.timestamp

    @property
    def steps_per_second(self) -> float:
        """Compute current throughput."""
        elapsed = self.last_time - self.start_time
        if elapsed < 0.001:
            return 0.0
        return self.last_step / elapsed

    @property
    def latest(self) -> Optional[Observation]:
        """Get most recent observation."""
        return self.ring.latest

    def get_summary(self) -> Dict[str, Any]:
        """Get summary for display."""
        latest = self.latest

        return {
            'step': latest.step if latest else 0,
            'health': latest.health.value if latest else 0.0,
            'health_trend': latest.health.trend_arrow if latest else '→',
            'health_status': latest.health.status if latest else 'unknown',
            'mask_density': latest.mask_stats.density if latest else 0.0,
            'intervention': latest.intervention_magnitude if latest else 0.0,
            'loss': latest.loss if latest else None,
            'steps_per_sec': self.steps_per_second,
            'subspace_histogram': self.aggregates.subspace_routing.get_histogram(
                self.num_subspaces
            ),
            'buffer_size': len(self.ring),
            'total_observations': self.aggregates.health.count,
        }

    def get_sparkline_data(self, metric: str, length: int = 50) -> List[float]:
        """Get data for sparkline visualization."""
        if metric == 'health':
            data = self.ring.get_health_series()
        elif metric == 'mask_density':
            data = self.ring.get_mask_density_series()
        elif metric == 'intervention':
            data = self.ring.get_intervention_series()
        elif metric == 'loss':
            data = self.ring.get_loss_series()
        else:
            data = []

        # Return last `length` points
        return data[-length:] if data else []
