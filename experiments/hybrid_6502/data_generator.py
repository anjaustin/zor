"""
Data Generator for Hybrid 6502 Experiment

Generates (opcode, state_before, state_after) tuples using frozen shapes
with known-correct routing. The training model must rediscover this routing.

This tests the hypothesis: given the right shapes, the routing is FORCED.
"""

import torch
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import IntEnum

import sys
sys.path.insert(0, '/workspace/trix_latest/TriXO/src')

from trix.nn.frozen_6502 import (
    FrozenShapes6502,
    FlagComputer,
    PureMath,
    ShapeID,
    int_to_bits,
    bits_to_int,
)


class RegisterID(IntEnum):
    """Register indices for Phase 1."""
    A = 0
    X = 1
    Y = 2
    M = 3  # Memory operand


# The 16 Phase 1 opcodes with their correct mappings
# This is the GROUND TRUTH that the model must discover
PHASE1_OPCODES = {
    0:  {'name': 'ADC', 'shape': ShapeID.RIPPLE_ADD, 'input_a': RegisterID.A, 'input_b': RegisterID.M, 'output': RegisterID.A, 'uses_carry': True, 'binary': True},
    1:  {'name': 'SBC', 'shape': ShapeID.RIPPLE_SUB, 'input_a': RegisterID.A, 'input_b': RegisterID.M, 'output': RegisterID.A, 'uses_carry': True, 'binary': True},
    2:  {'name': 'AND', 'shape': ShapeID.PARALLEL_AND, 'input_a': RegisterID.A, 'input_b': RegisterID.M, 'output': RegisterID.A, 'uses_carry': False, 'binary': True},
    3:  {'name': 'ORA', 'shape': ShapeID.PARALLEL_OR, 'input_a': RegisterID.A, 'input_b': RegisterID.M, 'output': RegisterID.A, 'uses_carry': False, 'binary': True},
    4:  {'name': 'EOR', 'shape': ShapeID.PARALLEL_XOR, 'input_a': RegisterID.A, 'input_b': RegisterID.M, 'output': RegisterID.A, 'uses_carry': False, 'binary': True},
    5:  {'name': 'ASL', 'shape': ShapeID.SHIFT_LEFT, 'input_a': RegisterID.A, 'input_b': RegisterID.A, 'output': RegisterID.A, 'uses_carry': False, 'binary': False},
    6:  {'name': 'LSR', 'shape': ShapeID.SHIFT_RIGHT, 'input_a': RegisterID.A, 'input_b': RegisterID.A, 'output': RegisterID.A, 'uses_carry': False, 'binary': False},
    7:  {'name': 'ROL', 'shape': ShapeID.ROTATE_LEFT, 'input_a': RegisterID.A, 'input_b': RegisterID.A, 'output': RegisterID.A, 'uses_carry': True, 'binary': False},
    8:  {'name': 'ROR', 'shape': ShapeID.ROTATE_RIGHT, 'input_a': RegisterID.A, 'input_b': RegisterID.A, 'output': RegisterID.A, 'uses_carry': True, 'binary': False},
    9:  {'name': 'CMP', 'shape': ShapeID.RIPPLE_SUB, 'input_a': RegisterID.A, 'input_b': RegisterID.M, 'output': None, 'uses_carry': False, 'binary': True, 'flags_only': True},
    10: {'name': 'CPX', 'shape': ShapeID.RIPPLE_SUB, 'input_a': RegisterID.X, 'input_b': RegisterID.M, 'output': None, 'uses_carry': False, 'binary': True, 'flags_only': True},
    11: {'name': 'CPY', 'shape': ShapeID.RIPPLE_SUB, 'input_a': RegisterID.Y, 'input_b': RegisterID.M, 'output': None, 'uses_carry': False, 'binary': True, 'flags_only': True},
    12: {'name': 'INX', 'shape': ShapeID.INCREMENT, 'input_a': RegisterID.X, 'input_b': RegisterID.X, 'output': RegisterID.X, 'uses_carry': False, 'binary': False},
    13: {'name': 'DEX', 'shape': ShapeID.DECREMENT, 'input_a': RegisterID.X, 'input_b': RegisterID.X, 'output': RegisterID.X, 'uses_carry': False, 'binary': False},
    14: {'name': 'INY', 'shape': ShapeID.INCREMENT, 'input_a': RegisterID.Y, 'input_b': RegisterID.Y, 'output': RegisterID.Y, 'uses_carry': False, 'binary': False},
    15: {'name': 'DEY', 'shape': ShapeID.DECREMENT, 'input_a': RegisterID.Y, 'input_b': RegisterID.Y, 'output': RegisterID.Y, 'uses_carry': False, 'binary': False},
}

NUM_OPCODES = len(PHASE1_OPCODES)
NUM_REGISTERS = 4
NUM_SHAPES = 16


class Hybrid6502DataGenerator:
    """
    Generate training data from frozen shapes with known routing.

    The model will only see (opcode, input_state, output_state).
    It must discover which shape produces the correct output.
    """

    def __init__(self, num_samples_per_opcode: int = 10000, seed: int = 42):
        self.num_samples = num_samples_per_opcode
        self.seed = seed

    def generate_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate complete dataset.

        Returns:
            opcodes: [N] opcode indices
            registers_before: [N, 4, 8] register bits before
            carry_before: [N] carry flag before
            registers_after: [N, 4, 8] register bits after
            carry_after: [N] carry flag after
        """
        torch.manual_seed(self.seed)

        all_opcodes = []
        all_regs_before = []
        all_carry_before = []
        all_regs_after = []
        all_carry_after = []

        for opcode_id in range(NUM_OPCODES):
            spec = PHASE1_OPCODES[opcode_id]

            for _ in range(self.num_samples):
                # Random input state
                reg_values = torch.randint(0, 256, (4,))
                carry_in = torch.randint(0, 2, (1,)).float().item()

                # Convert to bits
                regs_before = torch.stack([int_to_bits(v.unsqueeze(0))[0] for v in reg_values])

                # Execute with correct routing
                regs_after, carry_out = self._execute(spec, regs_before, carry_in)

                all_opcodes.append(opcode_id)
                all_regs_before.append(regs_before)
                all_carry_before.append(carry_in)
                all_regs_after.append(regs_after)
                all_carry_after.append(carry_out)

        return (
            torch.tensor(all_opcodes, dtype=torch.long),
            torch.stack(all_regs_before),
            torch.tensor(all_carry_before),
            torch.stack(all_regs_after),
            torch.tensor(all_carry_after),
        )

    def _execute(self, spec: Dict, regs: torch.Tensor, carry_in: float) -> Tuple[torch.Tensor, float]:
        """Execute a single instruction using the correct shape."""
        shape = spec['shape']

        # Get inputs
        input_a = regs[spec['input_a']].unsqueeze(0)  # [1, 8]
        input_b = regs[spec['input_b']].unsqueeze(0)  # [1, 8]
        c_in = torch.tensor([carry_in]) if spec['uses_carry'] else torch.tensor([0.0])

        # Execute the correct shape
        if shape == ShapeID.RIPPLE_ADD:
            out = FrozenShapes6502.ripple_add(input_a, input_b, c_in)
        elif shape == ShapeID.RIPPLE_SUB:
            # For CMP/CPX/CPY, carry is always 1 (no borrow)
            if spec.get('flags_only'):
                c_in = torch.tensor([1.0])
            out = FrozenShapes6502.ripple_sub(input_a, input_b, c_in)
        elif shape == ShapeID.PARALLEL_AND:
            out = FrozenShapes6502.parallel_and(input_a, input_b)
        elif shape == ShapeID.PARALLEL_OR:
            out = FrozenShapes6502.parallel_or(input_a, input_b)
        elif shape == ShapeID.PARALLEL_XOR:
            out = FrozenShapes6502.parallel_xor(input_a, input_b)
        elif shape == ShapeID.SHIFT_LEFT:
            out = FrozenShapes6502.shift_left(input_a)
        elif shape == ShapeID.SHIFT_RIGHT:
            out = FrozenShapes6502.shift_right(input_a)
        elif shape == ShapeID.ROTATE_LEFT:
            out = FrozenShapes6502.rotate_left(input_a, c_in)
        elif shape == ShapeID.ROTATE_RIGHT:
            out = FrozenShapes6502.rotate_right(input_a, c_in)
        elif shape == ShapeID.INCREMENT:
            out = FrozenShapes6502.increment(input_a)
        elif shape == ShapeID.DECREMENT:
            out = FrozenShapes6502.decrement(input_a)
        else:
            raise ValueError(f"Unknown shape: {shape}")

        result = out['result'][0]  # [8]
        carry_out = out.get('carry', torch.tensor([0.0]))
        if carry_out.dim() > 0:
            carry_out = carry_out.squeeze()

        # Build output registers
        regs_after = regs.clone()
        if spec.get('output') is not None:
            regs_after[spec['output']] = result

        return regs_after, carry_out.item()

    def generate_train_test_split(self, test_ratio: float = 0.1):
        """Generate data with train/test split."""
        opcodes, regs_b, carry_b, regs_a, carry_a = self.generate_dataset()

        n = len(opcodes)
        n_test = int(n * test_ratio)

        # Shuffle
        perm = torch.randperm(n)
        opcodes = opcodes[perm]
        regs_b = regs_b[perm]
        carry_b = carry_b[perm]
        regs_a = regs_a[perm]
        carry_a = carry_a[perm]

        # Split
        train_data = (
            opcodes[n_test:],
            regs_b[n_test:],
            carry_b[n_test:],
            regs_a[n_test:],
            carry_a[n_test:],
        )
        test_data = (
            opcodes[:n_test],
            regs_b[:n_test],
            carry_b[:n_test],
            regs_a[:n_test],
            carry_a[:n_test],
        )

        return train_data, test_data


def get_ground_truth_routing():
    """Return the known-correct routing for validation."""
    shape_routing = torch.zeros(NUM_OPCODES, dtype=torch.long)
    input_a_routing = torch.zeros(NUM_OPCODES, dtype=torch.long)
    input_b_routing = torch.zeros(NUM_OPCODES, dtype=torch.long)
    output_routing = torch.zeros(NUM_OPCODES, dtype=torch.long)
    uses_carry = torch.zeros(NUM_OPCODES, dtype=torch.bool)

    for opcode_id, spec in PHASE1_OPCODES.items():
        shape_routing[opcode_id] = spec['shape'].value
        input_a_routing[opcode_id] = spec['input_a']
        input_b_routing[opcode_id] = spec['input_b']
        output_routing[opcode_id] = spec['output'] if spec['output'] is not None else 0
        uses_carry[opcode_id] = spec['uses_carry']

    return {
        'shape': shape_routing,
        'input_a': input_a_routing,
        'input_b': input_b_routing,
        'output': output_routing,
        'uses_carry': uses_carry,
    }


if __name__ == "__main__":
    # Test data generation
    print("Testing data generator...")
    gen = Hybrid6502DataGenerator(num_samples_per_opcode=100, seed=42)
    train_data, test_data = gen.generate_train_test_split()

    print(f"Train: {len(train_data[0])} samples")
    print(f"Test: {len(test_data[0])} samples")

    # Verify a few samples
    print("\nSample verification:")
    for i in range(3):
        opcode = train_data[0][i].item()
        spec = PHASE1_OPCODES[opcode]
        print(f"  Opcode {opcode} ({spec['name']}): shape={spec['shape'].name}")

    print("\nGround truth routing:")
    gt = get_ground_truth_routing()
    for opcode_id in range(NUM_OPCODES):
        spec = PHASE1_OPCODES[opcode_id]
        print(f"  {opcode_id:2d} {spec['name']:4s}: shape={gt['shape'][opcode_id]:2d}, "
              f"in_a={gt['input_a'][opcode_id]}, in_b={gt['input_b'][opcode_id]}, "
              f"out={gt['output'][opcode_id]}, carry={gt['uses_carry'][opcode_id]}")
