"""
Phase Detection - Recognizing Stages of Learning.

Mesa 16.2: The Doula

Learning proceeds through phases:
    exploration → transition → crystallization → stable

The PhaseDetector recognizes these phases from dynamics signatures.
"""

from typing import List, Tuple, Optional
from .signature import DynamicsSignature


class PhaseDetector:
    """
    Detect which phase of learning we're in.

    Phases:
        exploration: High entropy, low stability - trying many things
        transition: Decreasing entropy, increasing stability - finding what works
        crystallization: Low entropy, high stability, still changing - locking in
        stable: Low entropy, very high stability, converged - done

    The detector uses thresholds that can be tuned per domain.
    """

    def __init__(
        self,
        exploration_entropy_threshold: float = 0.7,
        transition_entropy_threshold: float = 0.3,
        crystallization_stability_threshold: float = 0.95,
        stability_window: int = 10,
    ):
        """
        Initialize phase detector.

        Args:
            exploration_entropy_threshold: Above this = exploring
            transition_entropy_threshold: Below this, check stability
            crystallization_stability_threshold: Above this = stable
            stability_window: How many steps to average for stability
        """
        self.exploration_threshold = exploration_entropy_threshold
        self.transition_threshold = transition_entropy_threshold
        self.crystallization_threshold = crystallization_stability_threshold
        self.stability_window = stability_window

        # History for smoothed detection
        self.history: List[DynamicsSignature] = []

    def detect(self, signature: DynamicsSignature) -> str:
        """
        Detect the current learning phase.

        Args:
            signature: Current dynamics signature

        Returns:
            Phase name: 'exploration', 'transition', 'crystallization', 'stable'
        """
        entropy = signature.exploration_ratio
        stability = signature.routing_stability

        if entropy > self.exploration_threshold:
            return 'exploration'
        elif entropy > self.transition_threshold and stability < 0.7:
            return 'transition'
        elif stability < self.crystallization_threshold:
            return 'crystallization'
        else:
            return 'stable'

    def detect_with_history(self, signature: DynamicsSignature) -> Tuple[str, float]:
        """
        Detect phase using historical smoothing.

        Returns:
            (phase, confidence) tuple
        """
        self.history.append(signature)
        if len(self.history) > self.stability_window:
            self.history.pop(0)

        # Current detection
        current_phase = self.detect(signature)

        # Confidence based on consistency
        if len(self.history) < 3:
            return current_phase, 0.5

        recent_phases = [self.detect(s) for s in self.history[-5:]]
        agreement = sum(1 for p in recent_phases if p == current_phase) / len(recent_phases)

        return current_phase, agreement

    def predict_next_phase(self, current_phase: str) -> str:
        """
        Predict the next phase in the natural progression.

        Learning typically flows: exploration → transition → crystallization → stable
        """
        progression = {
            'exploration': 'transition',
            'transition': 'crystallization',
            'crystallization': 'stable',
            'stable': 'stable',  # Terminal
        }
        return progression.get(current_phase, 'exploration')

    def estimate_steps_to_transition(
        self,
        current_signature: DynamicsSignature,
        history: Optional[List[DynamicsSignature]] = None,
    ) -> int:
        """
        Estimate steps until next phase transition.

        Uses rate of change in entropy/stability to project.
        """
        if history is None:
            history = self.history

        if len(history) < 3:
            return 100  # Default estimate

        current_phase = self.detect(current_signature)

        # Calculate rate of change
        recent_entropies = [s.exploration_ratio for s in history[-10:]]
        recent_stabilities = [s.routing_stability for s in history[-10:]]

        if len(recent_entropies) < 2:
            return 100

        entropy_rate = (recent_entropies[-1] - recent_entropies[0]) / len(recent_entropies)
        stability_rate = (recent_stabilities[-1] - recent_stabilities[0]) / len(recent_stabilities)

        # Estimate based on current phase
        if current_phase == 'exploration':
            # How long until entropy drops below threshold?
            if entropy_rate >= 0:
                return 500  # Not decreasing, long wait
            steps = (self.exploration_threshold - current_signature.exploration_ratio) / abs(entropy_rate)
            return max(1, int(steps))

        elif current_phase == 'transition':
            # How long until stability rises?
            if stability_rate <= 0:
                return 500
            steps = (0.7 - current_signature.routing_stability) / stability_rate
            return max(1, int(steps))

        elif current_phase == 'crystallization':
            # How long until fully stable?
            if stability_rate <= 0:
                return 500
            steps = (self.crystallization_threshold - current_signature.routing_stability) / stability_rate
            return max(1, int(steps))

        else:  # stable
            return 0  # Already there

    def is_training_complete(
        self,
        signature: DynamicsSignature,
        min_stable_steps: int = 50,
    ) -> bool:
        """
        Determine if training is complete.

        Training is complete when we've been stable for a while.
        """
        if self.detect(signature) != 'stable':
            return False

        # Check how long we've been stable
        stable_count = 0
        for s in reversed(self.history):
            if self.detect(s) == 'stable':
                stable_count += 1
            else:
                break

        return stable_count >= min_stable_steps

    def detect_bottleneck(self, signature: DynamicsSignature) -> bool:
        """
        Detect if learning is stuck.

        Bottleneck = in transition but not making progress.
        """
        if self.detect(signature) != 'transition':
            return False

        if len(self.history) < 10:
            return False

        # Check if stability is not increasing
        recent_stabilities = [s.routing_stability for s in self.history[-10:]]
        improvement = recent_stabilities[-1] - recent_stabilities[0]

        return improvement < 0.01  # Less than 1% improvement over window

    def detect_emergence(self, signature: DynamicsSignature) -> bool:
        """
        Detect if new computational mode is emerging.

        Emergence = new shape activated or sudden cluster formation.
        """
        return signature.new_shape_activated or signature.cluster_formed

    def reset(self):
        """Clear history for new training run."""
        self.history = []
