"""
Doula Function - The Birth Attendant.

Mesa 16.2: The Doula

The DoulaFunction is integrated into the TriX training loop.
It observes, aggregates, stores, and guides.

Not a separate model - a function that writes to a VDB.
Glass box: all data captured, all patterns recorded.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple
import math

from .signature import DynamicsSignature, DoulaEntry, DoulaGuidance
from .phase import PhaseDetector
from .vdb import DoulaVDB


class DoulaFunction:
    """
    The Doula: observes, aggregates, stores, guides.

    Integrated into TriX training loop.
    No black boxes - everything is recorded.

    Usage:
        doula = DoulaFunction(vdb_path="training_dynamics.json")

        for step, batch in enumerate(dataloader):
            output, states, info = model(batch)
            loss = compute_loss(output, targets)

            # Doula observes
            for layer_idx, layer_info in enumerate(info['layers']):
                signature = doula.aggregate(
                    routing_info=layer_info['mixer'],
                    step=step,
                    layer=layer_idx,
                    loss=loss.item(),
                )
                doula.store(signature, loss.item())

            # Optional: get guidance
            if step % 100 == 0:
                guidance = doula.query(signature)
                if guidance.training_complete:
                    break
    """

    def __init__(
        self,
        vdb_path: Optional[str] = None,
        embedding_dim: int = 64,
        history_size: int = 100,
        domain: Optional[str] = None,
    ):
        """
        Initialize the Doula.

        Args:
            vdb_path: Path for VDB persistence (None = in-memory)
            embedding_dim: Dimension of dynamics embeddings
            history_size: Number of recent signatures to keep
            domain: Name of the training domain/task
        """
        self.vdb = DoulaVDB(vdb_path, embedding_dim)
        if domain:
            self.vdb.set_domain(domain)

        self.phase_detector = PhaseDetector()
        self.embedding_dim = embedding_dim
        self.history_size = history_size

        # Recent history for stability calculation
        self.history: List[DynamicsSignature] = []

        # Track shapes that have been seen
        self.seen_shapes: set = set()

        # Steps per epoch (set during training)
        self.steps_per_epoch: int = 1

    def aggregate(
        self,
        routing_info: Dict,
        step: int,
        layer: int,
        loss: float,
    ) -> DynamicsSignature:
        """
        Extract dynamics signature from routing info.

        Args:
            routing_info: From TriX forward pass, containing:
                - shape_idx: [B, T] which shape each token used
                - shape_weights: [B, T, num_shapes] soft routing weights
                - partner_idx: [B, T] partner selections
                - distances: [B, T, T] Hamming distances
            step: Training step
            layer: Layer index
            loss: Current loss value

        Returns:
            DynamicsSignature capturing this moment
        """
        # Extract from routing info
        shape_idx = routing_info.get('shape_idx')
        shape_weights = routing_info.get('shape_weights')
        partner_idx = routing_info.get('partner_idx')
        distances = routing_info.get('distances')

        # Compute entropy metrics
        shape_entropy = self._compute_shape_entropy(shape_weights)
        partner_entropy = self._compute_partner_entropy(partner_idx)

        # Compute stability (compare to recent history)
        routing_stability = self._compute_stability(shape_idx, layer)

        # Shape distribution
        num_shapes = shape_weights.shape[-1] if shape_weights is not None else 8
        if shape_idx is not None:
            shape_counts = shape_idx.flatten().bincount(minlength=num_shapes).float()
        else:
            shape_counts = torch.zeros(num_shapes)

        dominant_shapes = shape_counts.topk(min(3, num_shapes)).indices.tolist()

        # Phase indicators
        max_entropy = math.log(num_shapes) if num_shapes > 1 else 1.0
        exploration_ratio = shape_entropy / max_entropy if max_entropy > 0 else 0.0
        exploitation_ratio = 1 - exploration_ratio

        # Emergence signals
        new_shape_activated = self._detect_new_shape(shape_idx)
        cluster_formed = self._detect_cluster(partner_idx)

        # Distance statistics
        mean_distance, std_distance = self._compute_distance_stats(distances)

        signature = DynamicsSignature(
            step=step,
            layer=layer,
            shape_entropy=shape_entropy,
            partner_entropy=partner_entropy,
            routing_stability=routing_stability,
            dominant_shapes=dominant_shapes,
            shape_counts=shape_counts,
            exploration_ratio=exploration_ratio,
            exploitation_ratio=exploitation_ratio,
            new_shape_activated=new_shape_activated,
            cluster_formed=cluster_formed,
            mean_distance=mean_distance,
            std_distance=std_distance,
            partner_histogram=self._partner_histogram(partner_idx) if partner_idx is not None else None,
        )

        # Update history
        self.history.append(signature)
        if len(self.history) > self.history_size:
            self.history.pop(0)

        return signature

    def store(
        self,
        signature: DynamicsSignature,
        loss: float,
        accuracy: Optional[float] = None,
    ):
        """
        Store signature to VDB.

        Args:
            signature: The dynamics signature to store
            loss: Current loss value
            accuracy: Optional accuracy metric
        """
        embedding = signature.to_embedding(self.embedding_dim)
        phase = self.phase_detector.detect(signature)
        epoch = signature.step // max(1, self.steps_per_epoch)

        entry = DoulaEntry(
            id=f"step{signature.step}_layer{signature.layer}",
            dynamics_embedding=embedding,
            step=signature.step,
            layer=signature.layer,
            epoch=epoch,
            phase=phase,
            signature=signature,
            loss_at_step=loss,
            accuracy_at_step=accuracy,
        )

        self.vdb.add(entry)

    def query(
        self,
        current_signature: DynamicsSignature,
        k: int = 5,
    ) -> DoulaGuidance:
        """
        Query VDB for guidance based on current state.

        Args:
            current_signature: Current dynamics signature
            k: Number of similar past moments to consider

        Returns:
            DoulaGuidance with suggestions
        """
        # Find similar past moments
        similar = self.vdb.search_by_signature(current_signature, k=k)

        # Detect current phase
        current_phase, confidence = self.phase_detector.detect_with_history(current_signature)

        # Predict next phase
        expected_next = self.phase_detector.predict_next_phase(current_phase)

        # Estimate transition time
        steps_to_transition = self.phase_detector.estimate_steps_to_transition(
            current_signature,
            self.history,
        )

        # Suggest temperature adjustment
        temperature_adj = self._suggest_temperature(current_signature, current_phase)

        # Suggest priority shapes from similar moments
        priority_shapes = self._suggest_shapes(similar)

        # Signals
        training_complete = self.phase_detector.is_training_complete(current_signature)
        bottleneck_detected = self.phase_detector.detect_bottleneck(current_signature)
        emergence_imminent = self.phase_detector.detect_emergence(current_signature)

        return DoulaGuidance(
            current_phase=current_phase,
            expected_next_phase=expected_next,
            steps_to_transition=steps_to_transition,
            temperature_adjustment=temperature_adj,
            priority_shapes=priority_shapes,
            training_complete=training_complete,
            bottleneck_detected=bottleneck_detected,
            emergence_imminent=emergence_imminent,
            similar_past_moments=similar,
            confidence=confidence,
        )

    # =========================================================================
    # Private: Metric Computation
    # =========================================================================

    def _compute_shape_entropy(self, shape_weights: Optional[Tensor]) -> float:
        """Compute entropy of shape distribution."""
        if shape_weights is None:
            return 0.0

        # Average over batch and sequence
        avg_weights = shape_weights.mean(dim=(0, 1))  # [num_shapes]
        avg_weights = avg_weights + 1e-8  # Avoid log(0)

        entropy = -torch.sum(avg_weights * torch.log(avg_weights))
        return entropy.item()

    def _compute_partner_entropy(self, partner_idx: Optional[Tensor]) -> float:
        """Compute entropy of partner selection distribution."""
        if partner_idx is None:
            return 0.0

        B, T = partner_idx.shape
        # Count how often each position is chosen as partner
        counts = partner_idx.flatten().bincount(minlength=T).float()
        probs = counts / counts.sum()
        probs = probs + 1e-8

        entropy = -torch.sum(probs * torch.log(probs))
        return entropy.item()

    def _compute_stability(self, shape_idx: Optional[Tensor], layer: int) -> float:
        """Compute routing stability (agreement with recent history)."""
        if shape_idx is None or len(self.history) < 2:
            return 0.0

        # Get recent signatures for this layer
        recent = [s for s in self.history[-10:] if s.layer == layer]
        if len(recent) < 2:
            return 0.5

        # Compare shape distributions
        current_dist = shape_idx.flatten().bincount(
            minlength=shape_idx.max().item() + 1
        ).float()
        current_dist = current_dist / current_dist.sum()

        stabilities = []
        for sig in recent:
            if sig.shape_counts is not None:
                past_dist = sig.shape_counts / (sig.shape_counts.sum() + 1e-8)
                # Pad to same length
                max_len = max(len(current_dist), len(past_dist))
                curr = F.pad(current_dist, (0, max_len - len(current_dist)))
                past = F.pad(past_dist, (0, max_len - len(past_dist)))
                # Cosine similarity
                sim = F.cosine_similarity(curr.unsqueeze(0), past.unsqueeze(0))
                stabilities.append(sim.item())

        return sum(stabilities) / len(stabilities) if stabilities else 0.5

    def _detect_new_shape(self, shape_idx: Optional[Tensor]) -> bool:
        """Detect if a new shape was activated."""
        if shape_idx is None:
            return False

        unique_shapes = shape_idx.unique().tolist()
        new_shapes = set(unique_shapes) - self.seen_shapes

        if new_shapes:
            self.seen_shapes.update(new_shapes)
            return True
        return False

    def _detect_cluster(self, partner_idx: Optional[Tensor]) -> bool:
        """Detect if tokens are clustering (routing to same partners)."""
        if partner_idx is None:
            return False

        B, T = partner_idx.shape
        if T < 4:
            return False

        # Check if many tokens route to same partner
        for b in range(B):
            counts = partner_idx[b].bincount()
            if counts.max() > T * 0.3:  # 30%+ tokens share a partner
                return True
        return False

    def _compute_distance_stats(self, distances: Optional[Tensor]) -> Tuple[float, float]:
        """Compute mean and std of Hamming distances."""
        if distances is None:
            return 0.0, 0.0

        # Mask out inf values
        valid = distances[distances != float('inf')]
        if valid.numel() == 0:
            return 0.0, 0.0

        return valid.mean().item(), valid.std().item()

    def _partner_histogram(self, partner_idx: Tensor) -> Tensor:
        """Create histogram of partner selections."""
        B, T = partner_idx.shape
        histogram = torch.zeros(T)
        for b in range(B):
            histogram += partner_idx[b].bincount(minlength=T).float()
        return histogram / B

    def _suggest_temperature(self, signature: DynamicsSignature, phase: str) -> float:
        """Suggest temperature adjustment based on phase."""
        if phase == 'exploration':
            # Already exploring, maybe cool down a bit
            return -0.1 if signature.exploration_ratio > 0.8 else 0.0
        elif phase == 'transition':
            # In transition, stay neutral
            return 0.0
        elif phase == 'crystallization':
            # Crystallizing, warm up if stuck
            if signature.routing_stability < 0.5:
                return 0.1
            return 0.0
        else:  # stable
            return 0.0

    def _suggest_shapes(self, similar: List[DoulaEntry]) -> List[int]:
        """Suggest priority shapes based on similar past moments."""
        if not similar:
            return []

        # Aggregate dominant shapes from similar moments
        from collections import Counter
        shape_counts = Counter()
        for entry in similar:
            for shape in entry.signature.dominant_shapes:
                shape_counts[shape] += 1

        # Return top 3
        return [s for s, _ in shape_counts.most_common(3)]

    # =========================================================================
    # Configuration
    # =========================================================================

    def set_steps_per_epoch(self, steps: int):
        """Set steps per epoch for epoch calculation."""
        self.steps_per_epoch = steps

    def reset(self):
        """Reset for new training run."""
        self.history = []
        self.seen_shapes = set()
        self.phase_detector.reset()

    def save(self):
        """Save VDB to disk."""
        self.vdb.save()

    def get_vdb(self) -> DoulaVDB:
        """Get direct access to VDB for inspection."""
        return self.vdb

    def summarize(self) -> Dict:
        """Get summary of observations."""
        return self.vdb.summarize()
