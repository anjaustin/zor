"""
Hybrid 6502 Model: Frozen Shapes + Learnable Routing

The shapes are frozen (0 learnable parameters in computation).
Only the routing is learned (which shape to use for which opcode).

If the hypothesis is correct, the routing should converge to the
analytically-correct mapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

import sys
sys.path.insert(0, '/workspace/trix_latest/TriXO/src')

from trix.nn.frozen_6502 import (
    FrozenShapes6502,
    FlagComputer,
    ShapeID,
    int_to_bits,
    bits_to_int,
)

NUM_SHAPES = 16
NUM_REGISTERS = 4  # A, X, Y, M
NUM_OPCODES = 16   # Phase 1


class Frozen6502Executor(nn.Module):
    """
    Executes frozen shapes with soft input selection.

    Each shape is a pure mathematical function with 0 parameters.
    The executor just routes inputs to shapes and blends outputs.
    """

    def __init__(self):
        super().__init__()
        # No parameters - shapes are pure functions

    def forward(
        self,
        input_a: torch.Tensor,
        input_b: torch.Tensor,
        carry_in: torch.Tensor,
        shape_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute all shapes and blend by probability.

        Args:
            input_a: [batch, 8] first operand bits
            input_b: [batch, 8] second operand bits
            carry_in: [batch] carry input
            shape_probs: [batch, 16] probability for each shape

        Returns:
            result: [batch, 8] blended result
            carry: [batch] blended carry output
        """
        batch = input_a.shape[0]
        device = input_a.device

        result = torch.zeros(batch, 8, device=device)
        carry_out = torch.zeros(batch, device=device)

        # Execute each shape and blend
        shape_outputs = self._execute_all_shapes(input_a, input_b, carry_in)

        for shape_id in range(NUM_SHAPES):
            weight = shape_probs[:, shape_id].unsqueeze(-1)  # [batch, 1]
            result = result + weight * shape_outputs[shape_id]['result']

            if 'carry' in shape_outputs[shape_id]:
                carry_out = carry_out + shape_probs[:, shape_id] * shape_outputs[shape_id]['carry']

        return {'result': result, 'carry': carry_out}

    def _execute_all_shapes(
        self,
        input_a: torch.Tensor,
        input_b: torch.Tensor,
        carry_in: torch.Tensor,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Execute all 16 shapes and return outputs."""
        outputs = {}

        # Ensure carry_in has right shape
        if carry_in.dim() == 0:
            carry_in = carry_in.unsqueeze(0)

        # RIPPLE_ADD (0)
        outputs[0] = FrozenShapes6502.ripple_add(input_a, input_b, carry_in)

        # RIPPLE_SUB (1)
        outputs[1] = FrozenShapes6502.ripple_sub(input_a, input_b, carry_in)

        # PARALLEL_AND (2)
        outputs[2] = FrozenShapes6502.parallel_and(input_a, input_b)

        # PARALLEL_OR (3)
        outputs[3] = FrozenShapes6502.parallel_or(input_a, input_b)

        # PARALLEL_XOR (4)
        outputs[4] = FrozenShapes6502.parallel_xor(input_a, input_b)

        # SHIFT_LEFT (5)
        outputs[5] = FrozenShapes6502.shift_left(input_a)

        # SHIFT_RIGHT (6)
        outputs[6] = FrozenShapes6502.shift_right(input_a)

        # ROTATE_LEFT (7)
        outputs[7] = FrozenShapes6502.rotate_left(input_a, carry_in)

        # ROTATE_RIGHT (8)
        outputs[8] = FrozenShapes6502.rotate_right(input_a, carry_in)

        # INCREMENT (9)
        outputs[9] = FrozenShapes6502.increment(input_a)

        # DECREMENT (10)
        outputs[10] = FrozenShapes6502.decrement(input_a)

        # TRANSFER (11)
        outputs[11] = FrozenShapes6502.transfer(input_a)

        # LOAD (12)
        outputs[12] = FrozenShapes6502.load(input_a)

        # STORE (13)
        outputs[13] = FrozenShapes6502.store(input_a)

        # BIT_TEST (14) - uses both inputs
        outputs[14] = FrozenShapes6502.bit_test(input_a, input_b)

        # IDENTITY (15)
        outputs[15] = FrozenShapes6502.identity(input_a)

        return outputs


class HybridRouting(nn.Module):
    """
    Learnable routing layer.

    Maps opcode → (shape, input_a, input_b, output, uses_carry)

    This is the ONLY learnable part of the model.
    """

    def __init__(self, num_opcodes: int = NUM_OPCODES):
        super().__init__()

        # Initialize with small random values for symmetry breaking
        self.shape_logits = nn.Parameter(torch.randn(num_opcodes, NUM_SHAPES) * 0.01)
        self.input_a_logits = nn.Parameter(torch.randn(num_opcodes, NUM_REGISTERS) * 0.01)
        self.input_b_logits = nn.Parameter(torch.randn(num_opcodes, NUM_REGISTERS) * 0.01)
        self.output_logits = nn.Parameter(torch.randn(num_opcodes, NUM_REGISTERS) * 0.01)
        self.uses_carry_logit = nn.Parameter(torch.zeros(num_opcodes))

    def forward(
        self,
        opcode_id: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Get routing distributions for given opcodes.

        Args:
            opcode_id: [batch] opcode indices
            tau: Gumbel-Softmax temperature
            hard: Use hard (argmax) routing

        Returns:
            Dictionary with probability distributions
        """
        if hard:
            shape_probs = F.one_hot(
                self.shape_logits[opcode_id].argmax(-1), NUM_SHAPES
            ).float()
            input_a_probs = F.one_hot(
                self.input_a_logits[opcode_id].argmax(-1), NUM_REGISTERS
            ).float()
            input_b_probs = F.one_hot(
                self.input_b_logits[opcode_id].argmax(-1), NUM_REGISTERS
            ).float()
            output_probs = F.one_hot(
                self.output_logits[opcode_id].argmax(-1), NUM_REGISTERS
            ).float()
        else:
            shape_probs = F.gumbel_softmax(
                self.shape_logits[opcode_id], tau=tau, hard=False
            )
            input_a_probs = F.gumbel_softmax(
                self.input_a_logits[opcode_id], tau=tau, hard=False
            )
            input_b_probs = F.gumbel_softmax(
                self.input_b_logits[opcode_id], tau=tau, hard=False
            )
            output_probs = F.gumbel_softmax(
                self.output_logits[opcode_id], tau=tau, hard=False
            )

        uses_carry = torch.sigmoid(self.uses_carry_logit[opcode_id])

        return {
            'shape': shape_probs,
            'input_a': input_a_probs,
            'input_b': input_b_probs,
            'output': output_probs,
            'uses_carry': uses_carry,
        }

    def get_hard_routing(self) -> Dict[str, torch.Tensor]:
        """Get the argmax routing for all opcodes."""
        with torch.no_grad():
            return {
                'shape': self.shape_logits.argmax(dim=-1),
                'input_a': self.input_a_logits.argmax(dim=-1),
                'input_b': self.input_b_logits.argmax(dim=-1),
                'output': self.output_logits.argmax(dim=-1),
                'uses_carry': self.uses_carry_logit > 0,
            }

    def routing_entropy(self) -> float:
        """Compute average entropy of shape routing (lower = more confident)."""
        with torch.no_grad():
            probs = F.softmax(self.shape_logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
            return entropy.item()


class Hybrid6502(nn.Module):
    """
    Complete hybrid model: frozen shapes + learnable routing.

    The shapes compute exactly. The routing is learned.
    If the shapes are "natural", the routing should converge
    to the analytically-correct mapping.
    """

    def __init__(self, num_opcodes: int = NUM_OPCODES):
        super().__init__()

        self.executor = Frozen6502Executor()
        self.routing = HybridRouting(num_opcodes)
        self.num_opcodes = num_opcodes

    def forward(
        self,
        opcode_id: torch.Tensor,
        registers: torch.Tensor,
        carry_in: torch.Tensor,
        tau: float = 1.0,
        hard: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            opcode_id: [batch] opcode indices
            registers: [batch, 4, 8] register bits (A, X, Y, M)
            carry_in: [batch] carry flag
            tau: Gumbel-Softmax temperature
            hard: Use hard routing

        Returns:
            result: [batch, 8] output bits (for output register)
            new_registers: [batch, 4, 8] updated registers
            carry: [batch] output carry
            routing: routing probabilities
        """
        batch = opcode_id.shape[0]
        device = registers.device

        # Get routing probabilities
        routing = self.routing(opcode_id, tau=tau, hard=hard)

        # Select inputs via soft routing: registers @ probs
        # registers: [batch, 4, 8], probs: [batch, 4]
        input_a = torch.einsum('brd,br->bd', registers, routing['input_a'])
        input_b = torch.einsum('brd,br->bd', registers, routing['input_b'])

        # Carry input (masked by uses_carry)
        c_in = carry_in * routing['uses_carry']

        # Execute frozen shapes
        exec_out = self.executor(input_a, input_b, c_in, routing['shape'])

        result = exec_out['result']
        carry_out = exec_out['carry']

        # Update registers via soft routing
        # new_registers = registers + output_probs * (result - current_output_register)
        new_registers = registers.clone()
        for r in range(NUM_REGISTERS):
            weight = routing['output'][:, r].unsqueeze(-1)  # [batch, 1]
            new_registers[:, r] = registers[:, r] + weight * (result - registers[:, r])

        # Compute flags
        z_flag = FlagComputer.zero(result)
        n_flag = FlagComputer.negative(result)

        return {
            'result': result,
            'new_registers': new_registers,
            'carry': carry_out,
            'z_flag': z_flag,
            'n_flag': n_flag,
            'routing': routing,
        }

    def get_routing_matrix(self) -> Dict[str, torch.Tensor]:
        """Get the learned routing as argmax indices."""
        return self.routing.get_hard_routing()

    def routing_entropy(self) -> float:
        """Get routing entropy (confidence measure)."""
        return self.routing.routing_entropy()

    def param_count(self) -> int:
        """Count learnable parameters."""
        return sum(p.numel() for p in self.parameters())


def compare_routing(learned: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compare learned routing to ground truth.

    Returns accuracy for each routing dimension.
    """
    results = {}

    for key in ['shape', 'input_a', 'input_b', 'output']:
        if key in learned and key in ground_truth:
            match = (learned[key] == ground_truth[key]).float().mean().item()
            results[key] = match

    if 'uses_carry' in learned and 'uses_carry' in ground_truth:
        match = (learned['uses_carry'] == ground_truth['uses_carry']).float().mean().item()
        results['uses_carry'] = match

    return results


if __name__ == "__main__":
    # Test model
    print("Testing Hybrid6502 model...")

    model = Hybrid6502()
    print(f"Parameters: {model.param_count()}")

    # Test forward pass
    batch = 4
    opcodes = torch.randint(0, NUM_OPCODES, (batch,))
    registers = torch.randint(0, 2, (batch, 4, 8)).float()
    carry = torch.randint(0, 2, (batch,)).float()

    out = model(opcodes, registers, carry, tau=1.0, hard=False)
    print(f"Result shape: {out['result'].shape}")
    print(f"New registers shape: {out['new_registers'].shape}")
    print(f"Routing entropy: {model.routing_entropy():.3f}")

    # Test hard routing
    out_hard = model(opcodes, registers, carry, tau=0.1, hard=True)
    print(f"Hard result shape: {out_hard['result'].shape}")

    print("\nRouting matrix (initial, random):")
    routing = model.get_routing_matrix()
    print(f"Shape routing: {routing['shape'].tolist()}")
