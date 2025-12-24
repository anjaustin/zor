"""
Metrics: What we measure.

Health, deltas, sparsity - the vital signs of self-repairing computation.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class OperatorDelta:
    """Delta between operator input and output."""
    name: str
    l2_norm: float
    linf_norm: float
    mean_abs: float

    def __repr__(self):
        return f"Δ{self.name}={self.l2_norm:.4f}"


@dataclass
class HealthScore:
    """Health of a computation state."""
    value: float          # 0.0 (stuck) to 1.0 (healthy)
    distance: float       # Raw distance from healthy subspace
    subspace_id: int      # Which subspace is nearest
    trend: float          # Derivative: positive = improving

    @property
    def status(self) -> str:
        if self.value >= 0.9:
            return "healthy"
        elif self.value >= 0.7:
            return "recovering"
        elif self.value >= 0.5:
            return "struggling"
        else:
            return "stuck"

    @property
    def trend_arrow(self) -> str:
        if self.trend > 0.01:
            return "↗"
        elif self.trend < -0.01:
            return "↘"
        else:
            return "→"


@dataclass
class MaskStats:
    """Statistics about the intervention mask."""
    density: float        # Fraction of mask that's active
    num_active: int       # Count of active bits
    total_bits: int       # Total mask size

    @property
    def is_sparse(self) -> bool:
        return self.density < 0.3

    @property
    def is_dense(self) -> bool:
        return self.density > 0.7


@dataclass
class Observation:
    """Complete observation from one forward pass."""
    step: int
    timestamp: float

    # Core metrics
    health: HealthScore
    mask_stats: MaskStats
    intervention_magnitude: float

    # Per-operator deltas
    deltas: Dict[str, OperatorDelta]

    # Training context
    loss: Optional[float] = None
    learning_rate: Optional[float] = None

    # Raw tensors (only at high logging levels)
    tensors: Optional[Dict[str, torch.Tensor]] = None


def compute_delta(x_in: torch.Tensor, x_out: torch.Tensor, name: str) -> OperatorDelta:
    """Compute delta between operator input and output."""
    with torch.no_grad():
        diff = x_out - x_in
        return OperatorDelta(
            name=name,
            l2_norm=diff.norm(dim=-1).mean().item(),
            linf_norm=diff.abs().max().item(),
            mean_abs=diff.abs().mean().item(),
        )


def compute_health(
    projected: torch.Tensor,
    subspace_idx: torch.Tensor,
    projection_operator,  # ProjectionOperator instance
    prev_health: Optional[float] = None,
) -> HealthScore:
    """
    Compute health score from projection operator output.

    Health is inversely related to distance from the healthy subspace.
    Uses the projection magnitude as a proxy for alignment with the subspace.
    """
    with torch.no_grad():
        batch_size = projected.shape[0]
        dim = projected.shape[-1]

        # Health based on how "aligned" the projection is
        # High projection magnitude = well-aligned with subspace = healthy
        proj_norms = projected.norm(dim=-1)  # [batch]

        # Normalize by expected magnitude (sqrt(dim) for random unit vectors)
        max_norm = dim ** 0.5
        normalized_norms = proj_norms / max_norm

        # Health is the mean alignment (clamped to 0-1)
        health_value = normalized_norms.mean().clamp(0.0, 1.0).item()

        # Mean "distance" (inverse of projection strength)
        mean_distance = (1.0 - health_value) * max_norm

        # Compute trend if we have previous health
        trend = 0.0
        if prev_health is not None:
            trend = health_value - prev_health

        # Get most common subspace in batch
        if batch_size == 1:
            main_subspace = subspace_idx[0].item()
        else:
            # Mode of subspace indices
            main_subspace = subspace_idx.mode().values.item()

        return HealthScore(
            value=health_value,
            distance=mean_distance,
            subspace_id=main_subspace,
            trend=trend,
        )


def compute_mask_stats(mask: torch.Tensor) -> MaskStats:
    """Compute statistics about the intervention mask."""
    with torch.no_grad():
        total = mask.numel()
        active = (mask > 0.5).sum().item()

        return MaskStats(
            density=active / total if total > 0 else 0.0,
            num_active=int(active),
            total_bits=total,
        )


def compute_intervention_magnitude(
    x_in: torch.Tensor,
    x_out: torch.Tensor,
) -> float:
    """Compute magnitude of total intervention (input to final output)."""
    with torch.no_grad():
        return (x_out - x_in).norm(dim=-1).mean().item()


def extract_observation(
    step: int,
    timestamp: float,
    x_input: torch.Tensor,
    intermediates: Dict[str, Any],
    x_output: torch.Tensor,
    projection_operator,
    prev_health: Optional[float] = None,
    loss: Optional[float] = None,
    learning_rate: Optional[float] = None,
    include_tensors: bool = False,
) -> Observation:
    """
    Extract a complete observation from a forward pass.

    This is the main entry point for the metrics module.
    """
    # Compute per-operator deltas
    deltas = {}

    if 'smoothed' in intermediates:
        deltas['S'] = compute_delta(x_input, intermediates['smoothed'], 'S')

    if 'gradients' in intermediates and 'smoothed' in intermediates:
        deltas['D'] = compute_delta(intermediates['smoothed'], intermediates['gradients'], 'D')

    if 'projected' in intermediates and 'gradients' in intermediates:
        deltas['P'] = compute_delta(intermediates['gradients'], intermediates['projected'], 'P')

    # Compute health
    health = compute_health(
        intermediates.get('projected', x_output),
        intermediates.get('subspace_idx', torch.zeros(x_input.shape[0], dtype=torch.long)),
        projection_operator,
        prev_health,
    )

    # Compute mask stats
    mask_stats = compute_mask_stats(intermediates.get('mask', torch.ones_like(x_input)))

    # Compute total intervention
    intervention = compute_intervention_magnitude(x_input, x_output)

    # Optionally include tensors
    tensors = None
    if include_tensors:
        tensors = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in intermediates.items()
        }

    return Observation(
        step=step,
        timestamp=timestamp,
        health=health,
        mask_stats=mask_stats,
        intervention_magnitude=intervention,
        deltas=deltas,
        loss=loss,
        learning_rate=learning_rate,
        tensors=tensors,
    )
