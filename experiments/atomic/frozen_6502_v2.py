"""
Frozen 6502 v2: The Final Prototype

Based on Lincoln Manifold reflection:
- Level 0: Pure math primitives (0 params)
- Level 1: Frozen shapes (0 params)
- Level 2: Meaning layer with Gumbel-Softmax routing (~2,500 params)

Key insight: Computation is topology. Learning is routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import IntEnum


# =============================================================================
# CONFIGURATION
# =============================================================================

class ShapeID(IntEnum):
    """All frozen computation shapes."""
    RIPPLE_ADD = 0      # ADC
    RIPPLE_SUB = 1      # SBC, CMP, CPX, CPY
    PARALLEL_AND = 2    # AND
    PARALLEL_OR = 3     # ORA
    PARALLEL_XOR = 4    # EOR
    SHIFT_LEFT = 5      # ASL
    SHIFT_RIGHT = 6     # LSR
    ROTATE_LEFT = 7     # ROL
    ROTATE_RIGHT = 8    # ROR
    INCREMENT = 9       # INC, INX, INY
    DECREMENT = 10      # DEC, DEX, DEY
    TRANSFER = 11       # TAX, TXA, TAY, TYA, TSX, TXS
    LOAD = 12           # LDA, LDX, LDY
    STORE = 13          # STA, STX, STY
    BIT_TEST = 14       # BIT
    IDENTITY = 15       # NOP, flags, branches (no ALU op)


class RegisterID(IntEnum):
    """6502 registers."""
    A = 0       # Accumulator
    X = 1       # X index
    Y = 2       # Y index
    SP = 3      # Stack pointer
    PC_LO = 4   # Program counter low
    PC_HI = 5   # Program counter high
    P = 6       # Processor status
    MEM = 7     # Memory operand


NUM_SHAPES = 16
NUM_REGISTERS = 8
NUM_OPCODES = 56  # Unique 6502 operations (ignoring addressing modes)


# =============================================================================
# LEVEL 0: Pure Math Primitives (0 params)
# =============================================================================

class PureMath:
    """Mathematical primitives - no learning, just formulas."""

    @staticmethod
    def xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """XOR: a + b - 2ab"""
        return a + b - 2 * a * b

    @staticmethod
    def and_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """AND: ab"""
        return a * b

    @staticmethod
    def or_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """OR: a + b - ab"""
        return a + b - a * b

    @staticmethod
    def not_op(a: torch.Tensor) -> torch.Tensor:
        """NOT: 1 - a"""
        return 1 - a

    @staticmethod
    def full_adder(a: torch.Tensor, b: torch.Tensor,
                   c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full adder: (a, b, c_in) -> (sum, c_out)"""
        p = PureMath.xor(a, b)
        sum_bit = PureMath.xor(p, c)
        c_out = PureMath.or_op(PureMath.and_op(a, b), PureMath.and_op(c, p))
        return sum_bit, c_out


# =============================================================================
# LEVEL 1: Frozen Shapes (0 params)
# =============================================================================

class FrozenShapes:
    """Fixed computation topologies - pure math, no learning."""

    @staticmethod
    def ripple_add(a_bits: torch.Tensor, b_bits: torch.Tensor,
                   c_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        """8-bit ripple carry adder."""
        result = []
        carry = c_in.squeeze(-1) if c_in.dim() > 1 else c_in

        for i in range(8):
            s, carry = PureMath.full_adder(a_bits[:, i], b_bits[:, i], carry)
            result.append(s)

        return {
            'result': torch.stack(result, dim=1),
            'carry': carry
        }

    @staticmethod
    def ripple_sub(a_bits: torch.Tensor, b_bits: torch.Tensor,
                   c_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        """8-bit subtract (a - b with borrow). SBC uses inverted borrow."""
        b_inv = PureMath.not_op(b_bits)
        return FrozenShapes.ripple_add(a_bits, b_inv, c_in)

    @staticmethod
    def parallel_and(a_bits: torch.Tensor,
                     b_bits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """8-bit parallel AND."""
        return {'result': PureMath.and_op(a_bits, b_bits)}

    @staticmethod
    def parallel_or(a_bits: torch.Tensor,
                    b_bits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """8-bit parallel OR."""
        return {'result': PureMath.or_op(a_bits, b_bits)}

    @staticmethod
    def parallel_xor(a_bits: torch.Tensor,
                     b_bits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """8-bit parallel XOR."""
        return {'result': PureMath.xor(a_bits, b_bits)}

    @staticmethod
    def shift_left(a_bits: torch.Tensor,
                   _: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """ASL: shift left, bit7 -> carry."""
        zeros = torch.zeros_like(a_bits[:, :1])
        result = torch.cat([zeros, a_bits[:, :7]], dim=1)
        return {'result': result, 'carry': a_bits[:, 7]}

    @staticmethod
    def shift_right(a_bits: torch.Tensor,
                    _: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """LSR: shift right, bit0 -> carry."""
        zeros = torch.zeros_like(a_bits[:, :1])
        result = torch.cat([a_bits[:, 1:], zeros], dim=1)
        return {'result': result, 'carry': a_bits[:, 0]}

    @staticmethod
    def rotate_left(a_bits: torch.Tensor,
                    c_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ROL: rotate left through carry."""
        c = c_in.unsqueeze(-1) if c_in.dim() == 1 else c_in
        result = torch.cat([c, a_bits[:, :7]], dim=1)
        return {'result': result, 'carry': a_bits[:, 7]}

    @staticmethod
    def rotate_right(a_bits: torch.Tensor,
                     c_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ROR: rotate right through carry."""
        c = c_in.unsqueeze(-1) if c_in.dim() == 1 else c_in
        result = torch.cat([a_bits[:, 1:], c], dim=1)
        return {'result': result, 'carry': a_bits[:, 0]}

    @staticmethod
    def increment(a_bits: torch.Tensor,
                  _: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """INC: add 1."""
        b = torch.zeros_like(a_bits)
        b[:, 0] = 1.0
        c_in = torch.zeros(a_bits.shape[0], device=a_bits.device)
        return FrozenShapes.ripple_add(a_bits, b, c_in)

    @staticmethod
    def decrement(a_bits: torch.Tensor,
                  _: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """DEC: subtract 1."""
        b = torch.ones_like(a_bits)  # 0xFF = -1 in two's complement
        c_in = torch.zeros(a_bits.shape[0], device=a_bits.device)
        return FrozenShapes.ripple_add(a_bits, b, c_in)

    @staticmethod
    def transfer(a_bits: torch.Tensor,
                 _: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Transfer: pass through unchanged."""
        return {'result': a_bits}

    @staticmethod
    def load(m_bits: torch.Tensor,
             _: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Load: memory -> result."""
        return {'result': m_bits}

    @staticmethod
    def store(a_bits: torch.Tensor,
              _: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Store: register -> result (for writing to memory)."""
        return {'result': a_bits}

    @staticmethod
    def bit_test(a_bits: torch.Tensor,
                 m_bits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """BIT: test bits, set N/V from memory, Z from AND."""
        and_result = PureMath.and_op(a_bits, m_bits)
        return {
            'result': and_result,
            'n_from_m': m_bits[:, 7],
            'v_from_m': m_bits[:, 6]
        }

    @staticmethod
    def identity(a_bits: torch.Tensor,
                 _: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Identity: no operation (NOP, flag ops, branches)."""
        return {'result': a_bits}


# =============================================================================
# Flag Computation (0 params)
# =============================================================================

class FlagComputer:
    """Compute status flags from results."""

    @staticmethod
    def zero(result: torch.Tensor) -> torch.Tensor:
        """Z = 1 if all bits are 0."""
        or_all = result[:, 0]
        for i in range(1, 8):
            or_all = PureMath.or_op(or_all, result[:, i])
        return PureMath.not_op(or_all)

    @staticmethod
    def negative(result: torch.Tensor) -> torch.Tensor:
        """N = bit 7."""
        return result[:, 7]

    @staticmethod
    def overflow(a7: torch.Tensor, b7: torch.Tensor,
                 r7: torch.Tensor) -> torch.Tensor:
        """V = (A7 XOR R7) AND (B7 XOR R7)."""
        return PureMath.and_op(PureMath.xor(a7, r7), PureMath.xor(b7, r7))


# =============================================================================
# LEVEL 2: Meaning Layer (Learned)
# =============================================================================

class MeaningLayer(nn.Module):
    """
    The ONLY learned component. Maps opcode to routing decisions.

    Uses Gumbel-Softmax for differentiable discrete selection during training,
    hard argmax during inference.
    """

    def __init__(self, num_opcodes: int = NUM_OPCODES, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

        # Shape selection: which frozen shape to use
        self.shape_logits = nn.Parameter(torch.zeros(num_opcodes, NUM_SHAPES))

        # Input A selection: which register feeds input A
        self.input_a_logits = nn.Parameter(torch.zeros(num_opcodes, NUM_REGISTERS))

        # Input B selection: which register feeds input B
        self.input_b_logits = nn.Parameter(torch.zeros(num_opcodes, NUM_REGISTERS))

        # Output destination: where result goes
        self.output_logits = nn.Parameter(torch.zeros(num_opcodes, NUM_REGISTERS))

        # Flag update mask: which flags to update (N, Z, C, V)
        self.flag_mask = nn.Parameter(torch.zeros(num_opcodes, 4))

        # Uses carry input
        self.uses_carry = nn.Parameter(torch.zeros(num_opcodes, 1))

    def forward(self, opcode_id: torch.Tensor, training: bool = True
                ) -> Dict[str, torch.Tensor]:
        """
        Route opcode to shape and register selections.

        Returns soft probabilities during training, hard selections during inference.
        """
        if training:
            # Gumbel-Softmax for differentiable discrete selection
            shape_probs = F.gumbel_softmax(
                self.shape_logits[opcode_id], tau=self.temperature, hard=False
            )
            input_a_probs = F.gumbel_softmax(
                self.input_a_logits[opcode_id], tau=self.temperature, hard=False
            )
            input_b_probs = F.gumbel_softmax(
                self.input_b_logits[opcode_id], tau=self.temperature, hard=False
            )
            output_probs = F.gumbel_softmax(
                self.output_logits[opcode_id], tau=self.temperature, hard=False
            )
        else:
            # Hard selection for inference
            shape_probs = F.one_hot(
                self.shape_logits[opcode_id].argmax(dim=-1), NUM_SHAPES
            ).float()
            input_a_probs = F.one_hot(
                self.input_a_logits[opcode_id].argmax(dim=-1), NUM_REGISTERS
            ).float()
            input_b_probs = F.one_hot(
                self.input_b_logits[opcode_id].argmax(dim=-1), NUM_REGISTERS
            ).float()
            output_probs = F.one_hot(
                self.output_logits[opcode_id].argmax(dim=-1), NUM_REGISTERS
            ).float()

        flag_mask = torch.sigmoid(self.flag_mask[opcode_id])
        uses_carry = torch.sigmoid(self.uses_carry[opcode_id])

        return {
            'shape': shape_probs,
            'input_a': input_a_probs,
            'input_b': input_b_probs,
            'output': output_probs,
            'flag_mask': flag_mask,
            'uses_carry': uses_carry,
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Register File
# =============================================================================

class RegisterFile(nn.Module):
    """
    Differentiable register file with soft selection.

    Registers: A, X, Y, SP, PC_LO, PC_HI, P, MEM
    """

    def __init__(self):
        super().__init__()

    def select(self, registers: torch.Tensor,
               selector: torch.Tensor) -> torch.Tensor:
        """
        Select register value using soft probabilities.

        Args:
            registers: [batch, num_registers, 8] all register values
            selector: [batch, num_registers] selection probabilities

        Returns:
            [batch, 8] selected register bits
        """
        # Weighted sum over registers: [batch, reg, bits] * [batch, reg] -> [batch, bits]
        return torch.einsum('brd,br->bd', registers, selector)

    def scatter(self, registers: torch.Tensor, value: torch.Tensor,
                selector: torch.Tensor) -> torch.Tensor:
        """
        Write value to register using soft probabilities.

        Args:
            registers: [batch, num_registers, 8] current register values
            value: [batch, 8] value to write
            selector: [batch, num_registers] write probabilities

        Returns:
            [batch, num_registers, 8] updated registers
        """
        # Blend old and new values based on selector
        value_expanded = value.unsqueeze(1).expand_as(registers)
        selector_expanded = selector.unsqueeze(-1)
        return registers * (1 - selector_expanded) + value_expanded * selector_expanded


# =============================================================================
# Complete Frozen 6502 v2
# =============================================================================

class Frozen6502v2(nn.Module):
    """
    Complete 6502 with frozen shapes and learned meaning layer.

    Architecture:
    - Pure math primitives: 0 params
    - Frozen shapes: 0 params
    - Meaning layer: ~2,500 params (56 opcodes × ~45 features)

    Total: ~2,500 params vs ~100,000 for monolithic (40x compression)
    """

    SHAPE_FNS = [
        ('ripple_add', FrozenShapes.ripple_add),
        ('ripple_sub', FrozenShapes.ripple_sub),
        ('parallel_and', FrozenShapes.parallel_and),
        ('parallel_or', FrozenShapes.parallel_or),
        ('parallel_xor', FrozenShapes.parallel_xor),
        ('shift_left', FrozenShapes.shift_left),
        ('shift_right', FrozenShapes.shift_right),
        ('rotate_left', FrozenShapes.rotate_left),
        ('rotate_right', FrozenShapes.rotate_right),
        ('increment', FrozenShapes.increment),
        ('decrement', FrozenShapes.decrement),
        ('transfer', FrozenShapes.transfer),
        ('load', FrozenShapes.load),
        ('store', FrozenShapes.store),
        ('bit_test', FrozenShapes.bit_test),
        ('identity', FrozenShapes.identity),
    ]

    def __init__(self, num_opcodes: int = NUM_OPCODES, temperature: float = 1.0):
        super().__init__()
        self.meaning = MeaningLayer(num_opcodes, temperature)
        self.register_file = RegisterFile()
        self.num_shapes = len(self.SHAPE_FNS)

    def forward(self, opcode_id: torch.Tensor,
                registers: torch.Tensor,
                memory: torch.Tensor,
                carry_in: torch.Tensor,
                training: bool = True) -> Dict[str, torch.Tensor]:
        """
        Execute opcode on register state.

        Args:
            opcode_id: [batch] opcode indices
            registers: [batch, 7, 8] A, X, Y, SP, PC_LO, PC_HI, P bits
            memory: [batch, 8] memory operand bits
            carry_in: [batch] carry flag value
            training: use soft routing if True

        Returns:
            new_registers: [batch, 7, 8] updated registers
            result: [batch, 8] ALU result
            flags: dict of flag values
        """
        batch = opcode_id.shape[0]
        device = registers.device

        # Get routing from meaning layer
        routing = self.meaning(opcode_id, training=training)

        # Build full register bank including memory
        full_registers = torch.cat([registers, memory.unsqueeze(1)], dim=1)

        # Select inputs using soft routing
        input_a = self.register_file.select(full_registers, routing['input_a'])
        input_b = self.register_file.select(full_registers, routing['input_b'])

        # Prepare carry input
        c_in = carry_in * routing['uses_carry'].squeeze(-1)

        # Execute all shapes and blend by probability
        result = torch.zeros(batch, 8, device=device)
        carry_out = torch.zeros(batch, device=device)

        for shape_id, (name, fn) in enumerate(self.SHAPE_FNS):
            weight = routing['shape'][:, shape_id]  # [batch]

            # Execute shape
            if name in ['ripple_add', 'ripple_sub']:
                out = fn(input_a, input_b, c_in)
            elif name in ['rotate_left', 'rotate_right']:
                out = fn(input_a, c_in)
            elif name == 'bit_test':
                out = fn(input_a, input_b)
            elif name in ['parallel_and', 'parallel_or', 'parallel_xor']:
                out = fn(input_a, input_b)
            else:
                out = fn(input_a, None)

            # Blend results
            result = result + weight.unsqueeze(-1) * out['result']
            if 'carry' in out:
                carry_out = carry_out + weight * out['carry']

        # Compute flags
        z_flag = FlagComputer.zero(result)
        n_flag = FlagComputer.negative(result)
        v_flag = FlagComputer.overflow(input_a[:, 7], input_b[:, 7], result[:, 7])

        # Apply flag mask
        flag_mask = routing['flag_mask']  # [batch, 4] for N, Z, C, V

        # Write result to destination register
        new_registers = self.register_file.scatter(
            registers, result, routing['output'][:, :7]  # Exclude MEM from write
        )

        return {
            'registers': new_registers,
            'result': result,
            'carry': carry_out,
            'z_flag': z_flag,
            'n_flag': n_flag,
            'v_flag': v_flag,
            'flag_mask': flag_mask,
            'routing': routing,
        }

    def param_count(self) -> int:
        return self.meaning.param_count()

    def init_from_spec(self, opcode_specs: Dict[int, Dict]):
        """
        Initialize meaning layer from explicit opcode specifications.

        This allows preloading known 6502 behavior for faster convergence.
        """
        with torch.no_grad():
            for opcode_id, spec in opcode_specs.items():
                if 'shape' in spec:
                    self.meaning.shape_logits[opcode_id, spec['shape']] = 10.0
                if 'input_a' in spec:
                    self.meaning.input_a_logits[opcode_id, spec['input_a']] = 10.0
                if 'input_b' in spec:
                    self.meaning.input_b_logits[opcode_id, spec['input_b']] = 10.0
                if 'output' in spec:
                    self.meaning.output_logits[opcode_id, spec['output']] = 10.0
                if 'uses_carry' in spec and spec['uses_carry']:
                    self.meaning.uses_carry[opcode_id, 0] = 10.0  # High = uses carry


# =============================================================================
# Testing
# =============================================================================

def int_to_bits(x: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """Convert integer tensor to bit representation."""
    bits = []
    for i in range(num_bits):
        bits.append(((x >> i) & 1).float())
    return torch.stack(bits, dim=-1)


def bits_to_int(bits: torch.Tensor) -> torch.Tensor:
    """Convert bit tensor to integer."""
    powers = torch.tensor([2**i for i in range(bits.shape[-1])],
                          device=bits.device, dtype=bits.dtype)
    return (bits * powers).sum(dim=-1)


def test_frozen_6502_v2():
    """Test the frozen 6502 v2 model."""
    print("=" * 60)
    print("FROZEN 6502 v2 - FINAL PROTOTYPE")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Create model
    model = Frozen6502v2(num_opcodes=16, temperature=0.5)
    model = model.to(device)

    print(f"\nTotal parameters: {model.param_count()}")
    print(f"  Shape selection: {16 * NUM_SHAPES} params")
    print(f"  Input routing:   {16 * NUM_REGISTERS * 2} params")
    print(f"  Output routing:  {16 * NUM_REGISTERS} params")
    print(f"  Flag/carry:      {16 * 5} params")

    # Define opcode specs for testing
    opcode_specs = {
        0: {'shape': ShapeID.RIPPLE_ADD, 'input_a': RegisterID.A,
            'input_b': RegisterID.MEM, 'output': RegisterID.A, 'uses_carry': True},  # ADC
        1: {'shape': ShapeID.RIPPLE_SUB, 'input_a': RegisterID.A,
            'input_b': RegisterID.MEM, 'output': RegisterID.A, 'uses_carry': True},  # SBC
        2: {'shape': ShapeID.PARALLEL_AND, 'input_a': RegisterID.A,
            'input_b': RegisterID.MEM, 'output': RegisterID.A},  # AND
        3: {'shape': ShapeID.PARALLEL_OR, 'input_a': RegisterID.A,
            'input_b': RegisterID.MEM, 'output': RegisterID.A},  # ORA
        4: {'shape': ShapeID.PARALLEL_XOR, 'input_a': RegisterID.A,
            'input_b': RegisterID.MEM, 'output': RegisterID.A},  # EOR
        5: {'shape': ShapeID.SHIFT_LEFT, 'input_a': RegisterID.A,
            'input_b': RegisterID.A, 'output': RegisterID.A},     # ASL A
        6: {'shape': ShapeID.SHIFT_RIGHT, 'input_a': RegisterID.A,
            'input_b': RegisterID.A, 'output': RegisterID.A},     # LSR A
        7: {'shape': ShapeID.INCREMENT, 'input_a': RegisterID.X,
            'input_b': RegisterID.X, 'output': RegisterID.X},     # INX
        8: {'shape': ShapeID.DECREMENT, 'input_a': RegisterID.X,
            'input_b': RegisterID.X, 'output': RegisterID.X},     # DEX
        9: {'shape': ShapeID.TRANSFER, 'input_a': RegisterID.A,
            'input_b': RegisterID.A, 'output': RegisterID.X},     # TAX
        10: {'shape': ShapeID.TRANSFER, 'input_a': RegisterID.X,
             'input_b': RegisterID.X, 'output': RegisterID.A},    # TXA
        11: {'shape': ShapeID.LOAD, 'input_a': RegisterID.MEM,
             'input_b': RegisterID.MEM, 'output': RegisterID.A},  # LDA
        12: {'shape': ShapeID.LOAD, 'input_a': RegisterID.MEM,
             'input_b': RegisterID.MEM, 'output': RegisterID.X},  # LDX
    }

    model.init_from_spec(opcode_specs)

    # Test each operation
    print("\n" + "-" * 40)
    print("Testing Operations (with initialized routing)")
    print("-" * 40)

    test_cases = [
        (0, 'ADC', lambda a, m, c: (a + m + c) % 256),
        (2, 'AND', lambda a, m, c: a & m),
        (3, 'ORA', lambda a, m, c: a | m),
        (4, 'EOR', lambda a, m, c: a ^ m),
        (5, 'ASL', lambda a, m, c: (a << 1) & 0xFF),
        (6, 'LSR', lambda a, m, c: a >> 1),
        (7, 'INX', lambda a, m, c: (a + 1) & 0xFF),  # Note: uses X not A
        (8, 'DEX', lambda a, m, c: (a - 1) & 0xFF),
    ]

    batch_size = 1000
    model.eval()

    for op_id, op_name, gt_fn in test_cases:
        # Generate random test values
        a_val = torch.randint(0, 256, (batch_size,), device=device)
        m_val = torch.randint(0, 256, (batch_size,), device=device)
        c_val = torch.randint(0, 2, (batch_size,), device=device).float()

        # Build register state (A, X, Y, SP, PC_LO, PC_HI, P)
        registers = torch.zeros(batch_size, 7, 8, device=device)
        registers[:, RegisterID.A, :] = int_to_bits(a_val)
        registers[:, RegisterID.X, :] = int_to_bits(a_val)  # Use same value for X ops

        opcode_ids = torch.full((batch_size,), op_id, dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(
                opcode_ids,
                registers,
                int_to_bits(m_val),
                c_val,
                training=False
            )

        # Get predicted result
        pred_int = bits_to_int((out['result'] > 0.5).float()).long()

        # Compute ground truth
        if op_name in ['INX', 'DEX']:
            gt_int = gt_fn(a_val, m_val, c_val.long()).long() % 256
        else:
            gt_int = gt_fn(a_val.long(), m_val.long(), c_val.long()).long()

        accuracy = (pred_int == gt_int).float().mean()
        print(f"{op_name:4s}: {accuracy.item()*100:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model parameters: {model.param_count()}")
    print(f"Pure math primitives: 0 params (XOR, AND, OR, NOT)")
    print(f"Frozen shapes: 0 params (16 topologies)")
    print(f"Meaning layer: {model.param_count()} params")
    print(f"\nCompression vs monolithic (~100,000): {100000/model.param_count():.0f}x")
    print("\nKey insight: Computation is topology. Learning is routing.")

    return model


if __name__ == '__main__':
    model = test_frozen_6502_v2()
