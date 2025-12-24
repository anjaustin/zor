"""
Dynamics Signature - Capturing Learning Moments.

Mesa 16.2: The Doula

A DynamicsSignature captures the essential characteristics of a learning
moment - how routing is behaving, what shapes are active, whether
crystallization is occurring.

Glass box: all data visible, all patterns recorded.
"""

import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import math


@dataclass
class DynamicsSignature:
    """
    Signature of learning dynamics at a moment in time.

    Captures the essential "shape" of how learning is proceeding.
    Used for storage in VDB and similarity search.
    """

    # Identity
    step: int
    layer: int

    # Routing Entropy (chaos → order)
    shape_entropy: float        # How spread out are shape selections?
    partner_entropy: float      # How spread out are partner selections?

    # Crystallization Metrics
    routing_stability: float    # How much did routing change from last step?
    dominant_shapes: List[int]  # Which shapes are winning?
    shape_counts: Tensor        # Distribution over shapes

    # Phase Indicators
    exploration_ratio: float    # High entropy = exploring
    exploitation_ratio: float   # Low entropy = exploiting

    # Emergence Signals
    new_shape_activated: bool   # Shape that wasn't used before
    cluster_formed: bool        # Tokens routing together

    # Raw Statistics
    mean_distance: float        # Mean Hamming distance
    std_distance: float         # Std of Hamming distances

    # Optional detailed data
    partner_histogram: Optional[Tensor] = None
    shape_weights_snapshot: Optional[Tensor] = None

    def to_embedding(self, embedding_dim: int = 64) -> Tensor:
        """
        Convert signature to fixed-size embedding for VDB.

        The embedding captures the essential dynamics in a form
        suitable for similarity search.
        """
        # Core metrics
        core = torch.tensor([
            self.shape_entropy,
            self.partner_entropy,
            self.routing_stability,
            self.exploration_ratio,
            self.exploitation_ratio,
            self.mean_distance,
            self.std_distance,
            float(self.new_shape_activated),
            float(self.cluster_formed),
        ])

        # Shape distribution (normalized)
        if self.shape_counts is not None:
            shape_dist = self.shape_counts.float()
            shape_dist = shape_dist / (shape_dist.sum() + 1e-8)
            # Pad or truncate to fixed size
            target_len = embedding_dim - len(core)
            if len(shape_dist) < target_len:
                shape_dist = torch.cat([
                    shape_dist,
                    torch.zeros(target_len - len(shape_dist))
                ])
            else:
                shape_dist = shape_dist[:target_len]
        else:
            shape_dist = torch.zeros(embedding_dim - len(core))

        return torch.cat([core, shape_dist])

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'step': self.step,
            'layer': self.layer,
            'shape_entropy': self.shape_entropy,
            'partner_entropy': self.partner_entropy,
            'routing_stability': self.routing_stability,
            'dominant_shapes': self.dominant_shapes,
            'shape_counts': self.shape_counts.tolist() if self.shape_counts is not None else None,
            'exploration_ratio': self.exploration_ratio,
            'exploitation_ratio': self.exploitation_ratio,
            'new_shape_activated': self.new_shape_activated,
            'cluster_formed': self.cluster_formed,
            'mean_distance': self.mean_distance,
            'std_distance': self.std_distance,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'DynamicsSignature':
        """Create from dictionary."""
        shape_counts = torch.tensor(d['shape_counts']) if d.get('shape_counts') else None
        return cls(
            step=d['step'],
            layer=d['layer'],
            shape_entropy=d['shape_entropy'],
            partner_entropy=d['partner_entropy'],
            routing_stability=d['routing_stability'],
            dominant_shapes=d['dominant_shapes'],
            shape_counts=shape_counts,
            exploration_ratio=d['exploration_ratio'],
            exploitation_ratio=d['exploitation_ratio'],
            new_shape_activated=d['new_shape_activated'],
            cluster_formed=d['cluster_formed'],
            mean_distance=d['mean_distance'],
            std_distance=d['std_distance'],
        )


@dataclass
class DoulaEntry:
    """
    One observation in the Doula VDB.

    Contains both the embedding (for search) and full signature (for inspection).
    Glass box: everything is accessible.
    """

    # Identity
    id: str

    # Embedding (for similarity search)
    dynamics_embedding: Tensor

    # Metadata (for filtering)
    step: int
    layer: int
    epoch: int
    phase: str  # 'exploration', 'transition', 'crystallization', 'stable'

    # Full Signature (for deep inspection)
    signature: DynamicsSignature

    # Training Context
    loss_at_step: float
    accuracy_at_step: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'dynamics_embedding': self.dynamics_embedding.tolist(),
            'step': self.step,
            'layer': self.layer,
            'epoch': self.epoch,
            'phase': self.phase,
            'signature': self.signature.to_dict(),
            'loss_at_step': self.loss_at_step,
            'accuracy_at_step': self.accuracy_at_step,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'DoulaEntry':
        """Create from dictionary."""
        return cls(
            id=d['id'],
            dynamics_embedding=torch.tensor(d['dynamics_embedding']),
            step=d['step'],
            layer=d['layer'],
            epoch=d['epoch'],
            phase=d['phase'],
            signature=DynamicsSignature.from_dict(d['signature']),
            loss_at_step=d['loss_at_step'],
            accuracy_at_step=d.get('accuracy_at_step'),
        )


@dataclass
class DoulaGuidance:
    """
    Suggestions from the Doula based on past observations.

    The wisdom distilled from witnessing many births.
    """

    # Phase Detection
    current_phase: str          # Where are we in the birth?
    expected_next_phase: str    # What's coming?
    steps_to_transition: int    # Estimated steps until phase change

    # Actionable Suggestions
    temperature_adjustment: float   # Warmer = more exploration
    priority_shapes: List[int]      # Shapes that helped before

    # Signals
    training_complete: bool     # "The birth is done"
    bottleneck_detected: bool   # "Layer X is struggling"
    emergence_imminent: bool    # "Something new is forming"

    # Evidence
    similar_past_moments: List[DoulaEntry] = field(default_factory=list)
    confidence: float = 0.0     # How confident in these suggestions?

    def summary(self) -> str:
        """Human-readable summary of guidance."""
        lines = [
            f"Phase: {self.current_phase} → {self.expected_next_phase}",
            f"Steps to transition: ~{self.steps_to_transition}",
        ]

        if self.training_complete:
            lines.append("Signal: Training complete")
        if self.bottleneck_detected:
            lines.append("Signal: Bottleneck detected")
        if self.emergence_imminent:
            lines.append("Signal: Emergence imminent")

        if self.temperature_adjustment != 0:
            direction = "warmer" if self.temperature_adjustment > 0 else "cooler"
            lines.append(f"Suggestion: Run {direction} (Δ={self.temperature_adjustment:.2f})")

        if self.priority_shapes:
            lines.append(f"Priority shapes: {self.priority_shapes}")

        return "\n".join(lines)
