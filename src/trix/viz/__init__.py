"""
SDPMX Visualizer: Ghost in the Machine.

Watch self-repairing computation heal itself.

Usage:
    from trix.viz import create_observer, watch

    # Attach observer to pipeline
    observer = create_observer(pipeline, level=LogLevel.TRAJECTORY)
    observer.attach()

    # Run training...
    for batch in dataloader:
        x_new, info = pipeline(x, return_intermediates=True)
        observer.set_training_context(loss=loss.item())

    # Or watch live (blocks):
    watch(observer)

    # Or check summary:
    print(observer.summary)
"""

from .metrics import (
    OperatorDelta,
    HealthScore,
    MaskStats,
    Observation,
    compute_delta,
    compute_health,
    compute_mask_stats,
    compute_intervention_magnitude,
    extract_observation,
)

from .buffer import (
    RunningStats,
    SubspaceCounter,
    Aggregates,
    RingBuffer,
    ObservationStore,
)

from .observer import (
    LogLevel,
    ObserverConfig,
    SDPMXObserver,
    create_observer,
)

from .tui import (
    SimpleTUI,
    create_tui,
    watch,
)

# Conditional Rich export
try:
    from .tui import RichTUI
except ImportError:
    RichTUI = None


__all__ = [
    # Metrics
    'OperatorDelta',
    'HealthScore',
    'MaskStats',
    'Observation',
    'compute_delta',
    'compute_health',
    'compute_mask_stats',
    'compute_intervention_magnitude',
    'extract_observation',

    # Buffer
    'RunningStats',
    'SubspaceCounter',
    'Aggregates',
    'RingBuffer',
    'ObservationStore',

    # Observer
    'LogLevel',
    'ObserverConfig',
    'SDPMXObserver',
    'create_observer',

    # TUI
    'SimpleTUI',
    'RichTUI',
    'create_tui',
    'watch',
]
