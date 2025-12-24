# SDPMX Visualizer: Ghost in the Machine

**Watch self-repairing computation heal itself.**

The SDPMX Visualizer provides real-time observability into the SDPMX pipeline without impacting performance. It captures health metrics, intervention patterns, and subspace routing decisions - making the invisible visible.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Architecture](#architecture)
4. [API Reference](#api-reference)
5. [Logging Levels](#logging-levels)
6. [Metrics Explained](#metrics-explained)
7. [TUI Dashboard](#tui-dashboard)
8. [Best Practices](#best-practices)
9. [Examples](#examples)

---

## Quick Start

### Basic Usage

```python
from trix.nn import SDPMXPipeline
from trix.viz import create_observer, LogLevel

# Create pipeline and observer
pipeline = SDPMXPipeline(dim=64, grid_size=16, num_subspaces=8)
observer = create_observer(pipeline, level=LogLevel.TRAJECTORY)

# Attach hooks (automatic observation)
observer.attach()

# Training loop - observations happen automatically
for epoch in range(100):
    for x, target in dataloader:
        optimizer.zero_grad()
        x_new, loss, _ = pipeline.forward_with_loss(x, target)
        loss.backward()
        optimizer.step()

        # Update training context
        observer.set_training_context(loss=loss.item())

# Check current state
print(observer.summary)

# Clean up
observer.stop()
```

### Context Manager Usage

```python
from trix.viz import create_observer

with create_observer(pipeline) as observer:
    for batch in dataloader:
        pipeline(batch)

    print(f"Health: {observer.summary['health']:.2%}")
    print(f"Trend: {observer.summary['health_trend']}")
```

### Live Dashboard

```python
from trix.viz import create_observer, watch
import threading

observer = create_observer(pipeline)
observer.attach()

# Start training in background
def train():
    for epoch in range(100):
        for batch in dataloader:
            # ... training code ...
            observer.set_training_context(loss=loss.item())

training_thread = threading.Thread(target=train)
training_thread.start()

# Watch live (blocks until Ctrl+C)
watch(observer)
```

---

## Core Concepts

### The Observer Pattern

The visualizer uses a non-invasive observer pattern:

```
SDPMX Pipeline          Observer              Display
      │                    │                    │
      ├─ forward() ───────►│                    │
      │                    ├─ sample? ──────────┤
      │                    ├─ extract metrics ──┤
      │                    ├─ queue (async) ────┤
      │                    │                    ├─ render
      ├─ forward() ───────►│                    │
      │                    └─ ...               │
```

Key properties:
- **Zero-copy where possible**: Metrics are computed, not stored
- **Async logging**: Main thread never blocks on I/O
- **Sampling**: Not every step is logged (configurable)
- **Bounded memory**: Ring buffer with fixed size

### What We Observe

For each forward pass (when sampled), we capture:

| Metric | Source | Meaning |
|--------|--------|---------|
| Health | P operator | Alignment with healthy subspace |
| Health Trend | Δ Health | Improving (↗), stable (→), declining (↘) |
| Mask Density | M operator | Fraction of bits flagged for repair |
| Intervention | X operator | Magnitude of total state change |
| Subspace ID | P operator | Which repair strategy was used |
| Deltas | Each operator | Per-operator transformation magnitude |

---

## Architecture

### Module Structure

```
src/trix/viz/
├── __init__.py      # Public API exports
├── metrics.py       # Metric computation (what we measure)
├── buffer.py        # Storage (how we store)
├── observer.py      # Capture (how we observe)
└── tui.py           # Display (how we show)
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     OBSERVATION LAYER                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Sampler  │───►│ Metrics  │───►│  Queue   │              │
│  │ (rate)   │    │ (derive) │    │ (async)  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      STORAGE LAYER                           │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │ Ring Buffer  │         │  Aggregates  │                 │
│  │ (recent 1k)  │         │  (all-time)  │                 │
│  └──────────────┘         └──────────────┘                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      DISPLAY LAYER                           │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │  Rich TUI    │   OR    │  Simple TUI  │                 │
│  │ (dashboard)  │         │ (one-line)   │                 │
│  └──────────────┘         └──────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

---

## API Reference

### create_observer

```python
def create_observer(
    pipeline: SDPMXPipeline,
    level: LogLevel = LogLevel.TRAJECTORY,
    sample_rate: int = 10,
) -> SDPMXObserver:
    """
    Create an observer for an SDPMX pipeline.

    Args:
        pipeline: The SDPMX pipeline to observe
        level: Logging detail level (0-3)
        sample_rate: Log every N steps

    Returns:
        Configured SDPMXObserver instance
    """
```

### SDPMXObserver

```python
class SDPMXObserver:
    """Observer that attaches to SDPMX pipeline."""

    def attach(self) -> None:
        """Attach hooks to pipeline. Observations happen automatically."""

    def detach(self) -> None:
        """Remove hooks from pipeline."""

    def stop(self) -> None:
        """Stop observer and clean up resources."""

    def observe(
        self,
        x_input: torch.Tensor,
        x_output: torch.Tensor,
        intermediates: Dict[str, Any],
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        """Manually record an observation."""

    def set_training_context(
        self,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        """Update training context for next observation."""

    @property
    def summary(self) -> Dict[str, Any]:
        """Get current state summary."""

    def get_sparkline(self, metric: str, length: int = 50) -> str:
        """Get ASCII sparkline for a metric."""
```

### watch

```python
def watch(
    observer: SDPMXObserver,
    refresh_rate: float = 0.5,
    rich: bool = True,
) -> None:
    """
    Start live dashboard (blocks until Ctrl+C).

    Args:
        observer: The observer to visualize
        refresh_rate: Seconds between updates
        rich: Use Rich TUI if available
    """
```

### LogLevel

```python
class LogLevel(IntEnum):
    HEARTBEAT = 0   # Just loss and step (~10 bytes/step)
    TRAJECTORY = 1  # + health, mask, routing (~100 bytes/step)
    DIAGNOSTIC = 2  # + per-dim stats, gradients (~1KB/step)
    FULL = 3        # + all tensors (~100KB/step)
```

---

## Logging Levels

### Level 0: HEARTBEAT

Minimal overhead. Just tracks that the system is alive.

```python
observer = create_observer(pipeline, level=LogLevel.HEARTBEAT)
```

Captures:
- Step number
- Loss (if provided)
- Timestamp

Use for: Production, long runs, maximum performance.

### Level 1: TRAJECTORY (Default)

Balanced observability. Tracks health trends.

```python
observer = create_observer(pipeline, level=LogLevel.TRAJECTORY)
```

Captures:
- Everything in Level 0
- Health score and trend
- Mask density
- Intervention magnitude
- Subspace routing

Use for: Training, monitoring, debugging.

### Level 2: DIAGNOSTIC

Detailed statistics for investigation.

```python
observer = create_observer(pipeline, level=LogLevel.DIAGNOSTIC)
```

Captures:
- Everything in Level 1
- Per-operator deltas (S, D, P, M, X)
- Gradient norms
- Per-dimension histograms

Use for: Debugging specific issues, research.

### Level 3: FULL

Complete state capture. High overhead.

```python
observer = create_observer(pipeline, level=LogLevel.FULL)
```

Captures:
- Everything in Level 2
- Full intermediate tensors (smoothed, gradients, projected, mask)
- Spline parameters

Use for: Deep debugging, visualization, analysis.

---

## Metrics Explained

### Health Score (0.0 - 1.0)

**What it measures**: How well the current state aligns with the learned "healthy" subspace.

**Interpretation**:
- `0.9 - 1.0`: Healthy. Minimal intervention needed.
- `0.7 - 0.9`: Recovering. Some repair happening.
- `0.5 - 0.7`: Struggling. Significant intervention.
- `0.0 - 0.5`: Stuck. Major repair needed.

**Trend arrows**:
- `↗` Improving (health increasing)
- `→` Stable (health constant)
- `↘` Declining (health decreasing)

### Mask Density (0.0 - 1.0)

**What it measures**: Fraction of dimensions flagged for XOR repair.

**Interpretation**:
- `< 0.3`: Sparse. Targeted intervention.
- `0.3 - 0.7`: Moderate. Balanced repair.
- `> 0.7`: Dense. Broad intervention.

**Over training**: Should generally decrease as the system learns to stay healthy.

### Intervention Magnitude

**What it measures**: L2 norm of total state change (input → output).

**Interpretation**:
- Low values: Minimal correction needed.
- High values: Significant state modification.

### Subspace Routing

**What it measures**: Which of the N learned subspaces handled each input.

**Interpretation**:
- Balanced distribution: Subspaces are sharing work.
- Concentrated distribution: Few subspaces dominate.
- Dead subspaces (0%): Could be pruned.

---

## TUI Dashboard

### Rich Terminal UI

If `rich` is installed, you get a full dashboard:

```
┌─────────────────────────── SDPMX Observer ───────────────────────────┐
│ Health: ████████████████░░░░ 82.3% ↗ [recovering]  │  Step: 12,847  │
└──────────────────────────────────────────────────────────────────────┘
┌─ Metrics ────────────────────────────────────────────────────────────┐
│ Health  ▁▂▃▄▅▆▇█▇▆▇█▇█▇▆▇█▇█▇▆▅▆▇▆▇█▇█▇▆▇█  82.3%                   │
│ Mask    ▇▆▅▄▃▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁  12.1%                   │
│ Loss    ▇▆▅▄▄▃▃▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁  0.0234                  │
└──────────────────────────────────────────────────────────────────────┘
```

### Simple Terminal (Fallback)

Without `rich`, you get a single updating line:

```
Step:    12847 | Health: ████████░░ 82.3% ↗ | Mask: 12.1% | 1.2k/s
```

### Installation

```bash
# For Rich TUI (recommended)
pip install rich

# Or use simple TUI (no dependencies)
```

---

## Best Practices

### 1. Start with Level 1

Level 1 (TRAJECTORY) provides good visibility with minimal overhead:

```python
observer = create_observer(pipeline, level=LogLevel.TRAJECTORY)
```

### 2. Use Sampling for Long Runs

Don't log every step. Sample:

```python
observer = create_observer(pipeline, sample_rate=100)  # Every 100 steps
```

### 3. Set Training Context

Always update loss so it's captured:

```python
loss.backward()
observer.set_training_context(loss=loss.item())
```

### 4. Clean Up

Always stop the observer when done:

```python
observer.stop()

# Or use context manager
with create_observer(pipeline) as observer:
    ...
```

### 5. Check Overhead

For performance-critical code, verify overhead:

```python
# Time without observer
start = time.time()
for _ in range(1000):
    pipeline(x)
time_without = time.time() - start

# Time with observer (Level 0)
observer = create_observer(pipeline, level=LogLevel.HEARTBEAT)
observer.attach()
start = time.time()
for _ in range(1000):
    pipeline(x)
time_with = time.time() - start
observer.stop()

overhead = (time_with - time_without) / time_without
print(f"Overhead: {overhead:.1%}")  # Should be < 10%
```

---

## Examples

### Training with Health Monitoring

```python
from trix.nn import SDPMXPipeline
from trix.viz import create_observer, LogLevel

pipeline = SDPMXPipeline(dim=64, grid_size=16, num_subspaces=8)
optimizer = torch.optim.Adam(pipeline.parameters(), lr=0.001)
observer = create_observer(pipeline, level=LogLevel.TRAJECTORY)
observer.attach()

for epoch in range(100):
    for x, target in dataloader:
        optimizer.zero_grad()
        x_new, loss, _ = pipeline.forward_with_loss(x, target)
        loss.backward()
        optimizer.step()

        observer.set_training_context(loss=loss.item())

    # End of epoch summary
    s = observer.summary
    print(f"Epoch {epoch}: "
          f"Health={s['health']:.2%} {s['health_trend']} "
          f"Mask={s['mask_density']:.2%} "
          f"Loss={s['loss']:.4f}")

observer.stop()
```

### Health-Based Early Stopping

```python
observer = create_observer(pipeline)
observer.attach()

for epoch in range(1000):
    train_one_epoch(pipeline, dataloader, optimizer, observer)

    # Check health
    health = observer.summary['health']
    trend = observer.summary['health_trend']

    if health > 0.95 and trend == '→':
        print(f"Converged at epoch {epoch}")
        break

observer.stop()
```

### Comparing Runs

```python
def train_with_config(config):
    pipeline = SDPMXPipeline(**config)
    observer = create_observer(pipeline)
    observer.attach()

    train(pipeline, dataloader)

    summary = observer.summary
    observer.stop()

    return {
        'config': config,
        'final_health': summary['health'],
        'final_mask_density': summary['mask_density'],
    }

results = [
    train_with_config({'dim': 32, 'num_subspaces': 4}),
    train_with_config({'dim': 64, 'num_subspaces': 8}),
    train_with_config({'dim': 128, 'num_subspaces': 16}),
]

for r in results:
    print(f"{r['config']}: health={r['final_health']:.2%}")
```

---

## Troubleshooting

### High Overhead

**Symptom**: Training is significantly slower with observer attached.

**Solution**:
1. Lower the logging level: `level=LogLevel.HEARTBEAT`
2. Increase sample rate: `sample_rate=100`
3. Disable adaptive sampling: `ObserverConfig(adaptive_sampling=False)`

### No Observations Recorded

**Symptom**: `observer.summary` shows zeros.

**Solution**:
1. Make sure to call `observer.attach()` before training
2. Check that `sample_rate` isn't too high
3. Ensure async worker has time to process: `time.sleep(0.1)` before checking

### Rich TUI Not Showing

**Symptom**: Getting simple one-line output instead of dashboard.

**Solution**:
1. Install Rich: `pip install rich`
2. Ensure terminal supports ANSI colors
3. Try: `watch(observer, rich=True)`

---

## Philosophy

> *"The ghost that watches computation heal itself."*

The visualizer embodies several principles:

1. **Non-invasive**: Observe without affecting. Be a ghost.
2. **Hierarchical**: Different detail levels for different needs.
3. **Bounded**: Fixed memory. Predictable performance.
4. **Async**: Never block the main training loop.
5. **Beautiful**: Make the invisible visible. Make it joyful.

The best monitoring is like the best infrastructure: invisible when working, immediately visible when needed.

---

## See Also

- [SDPMX Pipeline Documentation](SDPMX_PIPELINE.md)
- [Master Index](../MASTER_INDEX.md)
- [Exploration Notes](/tmp/viz_exploration_*.md) (if still present)
