"""
TriX Native Router

The routing layer - the ONLY learnable component in frozen shape systems.

"Computation is topology. Learning is routing."

The shapes are frozen mathematical truths.
The router learns WHICH shape to use for each input.
"""

from typing import Dict, List, Optional

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    import numpy as cp
    HAS_CUPY = False

from .frozen import FrozenShapes, FrozenALU, int_to_bits


# =============================================================================
# NATIVE ROUTER
# =============================================================================

class NativeRouter:
    """
    Native GPU routing network.

    Maps input patterns (e.g., opcodes) to shape IDs using softmax routing.
    This is the ONLY learnable component.

    Args:
        num_inputs: Number of input categories (e.g., opcodes)
        num_shapes: Number of available shapes
        lr: Learning rate

    Example:
        >>> router = NativeRouter(num_opcodes=11, num_shapes=16)
        >>> shapes, probs = router.sample_shape(opcode_ids)
        >>> router.update(opcode_ids, shapes, rewards)
    """

    def __init__(
        self,
        num_inputs: int = 11,
        num_shapes: int = 16,
        lr: float = 0.5,
    ):
        self.num_inputs = num_inputs
        self.num_shapes = num_shapes
        self.lr = lr

        # Routing logits: [num_inputs, num_shapes]
        self.logits = cp.random.randn(num_inputs, num_shapes).astype(cp.float32) * 0.1

    def softmax(self, logits):
        """Stable softmax."""
        logits_max = logits.max(axis=-1, keepdims=True)
        exp_logits = cp.exp(logits - logits_max)
        return exp_logits / (exp_logits.sum(axis=-1, keepdims=True) + 1e-10)

    def sample_shape(self, input_ids):
        """
        Sample a shape for each input using current probabilities.

        Args:
            input_ids: Array of input category IDs [batch]

        Returns:
            (sampled_shapes, probabilities) tuple
        """
        batch_logits = self.logits[input_ids]  # [batch, num_shapes]
        probs = self.softmax(batch_logits)

        # Sample from categorical distribution
        cumprobs = cp.cumsum(probs, axis=-1)
        u = cp.random.uniform(0, 1, (len(input_ids), 1))
        sampled_shapes = (u < cumprobs).argmax(axis=-1)

        return sampled_shapes, probs

    def get_hard_routing(self):
        """Get argmax routing for all inputs."""
        return cp.argmax(self.logits, axis=-1)

    def route(self, input_ids):
        """
        Get deterministic routing (argmax) for inputs.

        Args:
            input_ids: Array of input category IDs [batch]

        Returns:
            Shape IDs for each input [batch]
        """
        hard_routing = self.get_hard_routing()
        return hard_routing[input_ids]

    def update(self, input_ids, sampled_shapes, rewards):
        """
        Policy gradient update.

        Args:
            input_ids: Input category IDs [batch]
            sampled_shapes: Shapes that were sampled [batch]
            rewards: 1 if correct, 0 otherwise [batch]
        """
        batch_logits = self.logits[input_ids]
        probs = self.softmax(batch_logits)

        for i in range(len(input_ids)):
            input_id = int(input_ids[i])
            shape = int(sampled_shapes[i])
            reward = float(rewards[i])

            if reward > 0:
                # Correct! Increase this shape's probability
                grad = cp.zeros(self.num_shapes, dtype=cp.float32)
                grad[shape] = 1 - probs[i, shape]
                for s in range(self.num_shapes):
                    if s != shape:
                        grad[s] = -probs[i, s]
                self.logits[input_id] += self.lr * reward * grad
            else:
                # Wrong! Decrease this shape's probability
                grad = cp.zeros(self.num_shapes, dtype=cp.float32)
                grad[shape] = -probs[i, shape]
                self.logits[input_id] += self.lr * 0.1 * grad

        # Clip logits to prevent explosion
        self.logits = cp.clip(self.logits, -10, 10)

    def supervised_update(self, input_id: int, correct_shape: int):
        """
        Direct supervised update for a single input->shape mapping.

        Args:
            input_id: Input category ID
            correct_shape: Correct shape ID
        """
        self.logits[input_id, correct_shape] += self.lr
        for s in range(self.num_shapes):
            if s != correct_shape:
                self.logits[input_id, s] -= self.lr / (self.num_shapes - 1)

    def routing_entropy(self) -> float:
        """Compute entropy of routing (lower = more confident)."""
        probs = self.softmax(self.logits)
        entropy = -(probs * cp.log(probs + 1e-10)).sum(axis=-1).mean()
        return float(entropy)

    def param_count(self) -> int:
        """Number of learnable parameters."""
        return int(self.logits.size)


# =============================================================================
# NATIVE FROZEN HYBRID
# =============================================================================

class NativeFrozenHybrid:
    """
    Frozen shapes + Native routing = 100% accuracy.

    Architecture:
        - Frozen shapes: 0 learnable parameters (mathematical truths)
        - Routing: num_inputs * num_shapes learnable parameters

    The shapes ARE the computation.
    The routing learns to find them.

    Args:
        num_opcodes: Number of opcodes to route
        num_shapes: Number of available shapes
        lr: Router learning rate

    Example:
        >>> model = NativeFrozenHybrid()
        >>> model.train_supervised(ground_truth_mapping)
        >>> accuracy = model.evaluate(test_data)
    """

    # Opcode to ground truth shape mapping (6502 ALU)
    OPCODE_TO_SHAPE = {
        0: 0,   # ADC -> RIPPLE_ADD
        1: 1,   # SBC -> RIPPLE_SUB
        2: 2,   # AND -> PARALLEL_AND
        3: 3,   # ORA -> PARALLEL_OR
        4: 4,   # EOR -> PARALLEL_XOR
        5: 5,   # ASL -> SHIFT_LEFT
        6: 6,   # LSR -> SHIFT_RIGHT
        7: 7,   # ROL -> ROTATE_LEFT
        8: 8,   # ROR -> ROTATE_RIGHT
        9: 9,   # INC -> INCREMENT
        10: 10, # DEC -> DECREMENT
    }

    OPCODE_NAMES = ['ADC', 'SBC', 'AND', 'ORA', 'EOR', 'ASL', 'LSR', 'ROL', 'ROR', 'INC', 'DEC']

    def __init__(
        self,
        num_opcodes: int = 11,
        num_shapes: int = 16,
        lr: float = 0.5,
    ):
        self.router = NativeRouter(
            num_inputs=num_opcodes,
            num_shapes=num_shapes,
            lr=lr
        )

    def forward(self, opcode_ids, a_bits, b_bits, c_in):
        """
        Forward pass through frozen shapes with learned routing.

        Args:
            opcode_ids: Opcode IDs [batch]
            a_bits: First operand bits [batch, 8]
            b_bits: Second operand bits [batch, 8]
            c_in: Carry in [batch]

        Returns:
            (result_bits, carry_out) tuple
        """
        batch = len(opcode_ids)
        routing = self.router.route(opcode_ids)

        result_bits = cp.zeros((batch, 8), dtype=cp.float32)
        carry_out = cp.zeros(batch, dtype=cp.float32)

        for i in range(batch):
            shape_id = int(routing[i])
            result, carry = FrozenALU.execute_shape(
                shape_id,
                a_bits[i:i+1],
                b_bits[i:i+1],
                c_in[i:i+1]
            )
            result_bits[i] = result[0]
            carry_out[i] = carry[0]

        return result_bits, carry_out

    def train_step(self, opcode_ids, a_bits, b_bits, c_in, target_bits, target_carry):
        """
        Policy gradient training step.

        Args:
            opcode_ids: Opcode IDs [batch]
            a_bits: First operand bits [batch, 8]
            b_bits: Second operand bits [batch, 8]
            c_in: Carry in [batch]
            target_bits: Expected result bits [batch, 8]
            target_carry: Expected carry out [batch]

        Returns:
            Accuracy for this batch
        """
        batch = len(opcode_ids)

        # Sample shapes from current policy
        sampled_shapes, probs = self.router.sample_shape(opcode_ids)

        # Execute sampled shapes and compute rewards
        rewards = cp.zeros(batch, dtype=cp.float32)
        total_correct = 0

        for i in range(batch):
            shape_id = int(sampled_shapes[i])

            # Execute frozen shape
            result, carry = FrozenALU.execute_shape(
                shape_id,
                a_bits[i:i+1],
                b_bits[i:i+1],
                c_in[i:i+1]
            )

            # Round to binary
            result_bin = (result[0] > 0.5).astype(cp.float32)
            carry_bin = float(carry[0] > 0.5)

            # Check if correct
            bits_match = bool((result_bin == target_bits[i]).all())
            carry_match = bool(carry_bin == target_carry[i])

            if bits_match and carry_match:
                rewards[i] = 1.0
                total_correct += 1

        # Update router with policy gradient
        self.router.update(opcode_ids, sampled_shapes, rewards)

        return total_correct / batch

    def train_supervised(self, ground_truth: Optional[Dict[int, int]] = None, epochs: int = 50):
        """
        Train router with direct supervision.

        Args:
            ground_truth: Mapping of input_id -> correct_shape_id
                         Uses default OPCODE_TO_SHAPE if None
            epochs: Number of training epochs
        """
        if ground_truth is None:
            ground_truth = self.OPCODE_TO_SHAPE

        for epoch in range(epochs):
            for input_id, correct_shape in ground_truth.items():
                self.router.supervised_update(input_id, correct_shape)

            # Check convergence
            accuracy = self.get_routing_accuracy(ground_truth)
            if accuracy == 1.0:
                break

    def get_routing_accuracy(self, ground_truth: Optional[Dict[int, int]] = None) -> float:
        """Check if routing has converged to ground truth."""
        if ground_truth is None:
            ground_truth = self.OPCODE_TO_SHAPE

        hard_routing = self.router.get_hard_routing()

        correct = 0
        total = len(ground_truth)

        for input_id, true_shape in ground_truth.items():
            if int(hard_routing[input_id]) == true_shape:
                correct += 1

        return correct / total

    def get_routing_table(self) -> List[Dict]:
        """Get current routing as a readable table."""
        hard_routing = self.router.get_hard_routing()

        table = []
        for opcode, true_shape in self.OPCODE_TO_SHAPE.items():
            pred_shape = int(hard_routing[opcode])
            correct = "OK" if pred_shape == true_shape else "!!"
            table.append({
                'opcode': self.OPCODE_NAMES[opcode],
                'true_shape': FrozenALU.SHAPE_NAMES[true_shape],
                'pred_shape': FrozenALU.SHAPE_NAMES[pred_shape],
                'correct': correct,
            })

        return table

    def param_count(self) -> int:
        """Total learnable parameters (routing only, shapes are frozen)."""
        return self.router.param_count()
