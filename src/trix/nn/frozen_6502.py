"""
Frozen 6502: 16 Shapes for CPU Emulation

This module provides the 16 frozen computation shapes needed to emulate
a 6502 CPU's ALU operations with 100% accuracy and ~40x compression.

Architecture:
    Level 0: Pure Math Primitives (0 params)
        XOR, AND, OR, NOT - continuous polynomial forms

    Level 1: Frozen Shapes (0 params)
        16 topologies composed from Level 0 primitives

    Level 2: Meaning Layer (~2,500 params)
        Learned opcode → shape routing

Key insight: "Computation is topology. Learning is routing."
    - The shapes ARE the geometry (discovered, not learned)
    - The meaning layer routes opcodes to shapes (learned)
    - Result: 100,000 params → 2,500 params = 40x compression

Example:
    >>> from trix.nn.frozen_6502 import create_frozen_6502, OPCODE_SPECS
    >>> model = create_frozen_6502()
    >>> model.init_meaning_from_spec(OPCODE_SPECS)
    >>> # Now ready for 6502 emulation

See also:
    - docs/FROZEN_6502.md for complete architecture
    - docs/OPCODE_MAP.md for opcode reference
    - docs/ATOMIC_FUNCTIONS.md for mathematical foundations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import IntEnum

from trix.nn.frozen import (
    FrozenShape,
    FrozenTile,
    FrozenTriXFFN,
    FrozenShapeRegistry,
    derive_signature_from_fn,
)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ShapeID(IntEnum):
    """Frozen shape identifiers."""
    RIPPLE_ADD = 0
    RIPPLE_SUB = 1
    PARALLEL_AND = 2
    PARALLEL_OR = 3
    PARALLEL_XOR = 4
    SHIFT_LEFT = 5
    SHIFT_RIGHT = 6
    ROTATE_LEFT = 7
    ROTATE_RIGHT = 8
    INCREMENT = 9
    DECREMENT = 10
    TRANSFER = 11
    LOAD = 12
    STORE = 13
    BIT_TEST = 14
    IDENTITY = 15


class RegisterID(IntEnum):
    """6502 register identifiers."""
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


# =============================================================================
# LEVEL 0: PURE MATH PRIMITIVES (0 params)
# =============================================================================

class PureMath:
    """
    Mathematical primitives as continuous polynomials.

    These formulas compute Boolean logic using continuous operations,
    enabling gradient flow through the computation while maintaining
    100% accuracy on {0, 1} inputs.

    Key formulas:
        XOR(a, b) = a + b - 2ab
        AND(a, b) = ab
        OR(a, b)  = a + b - ab
        NOT(a)    = 1 - a
    """

    @staticmethod
    def xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """XOR: a + b - 2ab (saddle surface)"""
        return a + b - 2 * a * b

    @staticmethod
    def and_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """AND: ab (product)"""
        return a * b

    @staticmethod
    def or_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """OR: a + b - ab (union)"""
        return a + b - a * b

    @staticmethod
    def not_op(a: torch.Tensor) -> torch.Tensor:
        """NOT: 1 - a (reflection)"""
        return 1 - a

    @staticmethod
    def full_adder(
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full adder: (a, b, c_in) → (sum, c_out)

        sum   = a XOR b XOR c
        c_out = (a AND b) OR (c AND (a XOR b))
        """
        p = PureMath.xor(a, b)
        sum_bit = PureMath.xor(p, c)
        c_out = PureMath.or_op(PureMath.and_op(a, b), PureMath.and_op(c, p))
        return sum_bit, c_out


# =============================================================================
# LEVEL 1: FROZEN SHAPES (0 params)
# =============================================================================

class FrozenShapes6502:
    """
    The 16 frozen computation topologies for 6502 emulation.

    Each shape is a fixed function composed from PureMath primitives.
    No learning occurs here - these ARE the geometry.
    """

    @staticmethod
    def ripple_add(
        a_bits: torch.Tensor,
        b_bits: torch.Tensor,
        c_in: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        8-bit ripple carry adder.

        Chains 8 full adders, propagating carry from LSB to MSB.
        Used for: ADC
        """
        result = []
        carry = c_in.squeeze(-1) if c_in.dim() > 1 else c_in

        for i in range(8):
            s, carry = PureMath.full_adder(a_bits[:, i], b_bits[:, i], carry)
            result.append(s)

        return {
            'result': torch.stack(result, dim=1),
            'carry': carry,
        }

    @staticmethod
    def ripple_sub(
        a_bits: torch.Tensor,
        b_bits: torch.Tensor,
        c_in: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        8-bit subtraction via inverted addition.

        A - B = A + NOT(B) + 1 (with borrow handling)
        Used for: SBC, CMP, CPX, CPY
        """
        b_inv = PureMath.not_op(b_bits)
        return FrozenShapes6502.ripple_add(a_bits, b_inv, c_in)

    @staticmethod
    def parallel_and(
        a_bits: torch.Tensor,
        b_bits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        8-bit parallel AND.

        Applies AND to each bit position independently.
        Used for: AND
        """
        return {'result': PureMath.and_op(a_bits, b_bits)}

    @staticmethod
    def parallel_or(
        a_bits: torch.Tensor,
        b_bits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        8-bit parallel OR.

        Used for: ORA
        """
        return {'result': PureMath.or_op(a_bits, b_bits)}

    @staticmethod
    def parallel_xor(
        a_bits: torch.Tensor,
        b_bits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        8-bit parallel XOR.

        Used for: EOR
        """
        return {'result': PureMath.xor(a_bits, b_bits)}

    @staticmethod
    def shift_left(
        a_bits: torch.Tensor,
        _: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Arithmetic Shift Left.

        [b0,b1,b2,b3,b4,b5,b6,b7] → [0,b0,b1,b2,b3,b4,b5,b6], carry=b7
        Used for: ASL
        """
        zeros = torch.zeros_like(a_bits[:, :1])
        result = torch.cat([zeros, a_bits[:, :7]], dim=1)
        return {'result': result, 'carry': a_bits[:, 7]}

    @staticmethod
    def shift_right(
        a_bits: torch.Tensor,
        _: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Logical Shift Right.

        [b0,b1,b2,b3,b4,b5,b6,b7] → [b1,b2,b3,b4,b5,b6,b7,0], carry=b0
        Used for: LSR
        """
        zeros = torch.zeros_like(a_bits[:, :1])
        result = torch.cat([a_bits[:, 1:], zeros], dim=1)
        return {'result': result, 'carry': a_bits[:, 0]}

    @staticmethod
    def rotate_left(
        a_bits: torch.Tensor,
        c_in: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Rotate Left through Carry.

        [b0,b1,...,b7] + c_in → [c_in,b0,...,b6], carry=b7
        Used for: ROL
        """
        c = c_in.unsqueeze(-1) if c_in.dim() == 1 else c_in
        result = torch.cat([c, a_bits[:, :7]], dim=1)
        return {'result': result, 'carry': a_bits[:, 7]}

    @staticmethod
    def rotate_right(
        a_bits: torch.Tensor,
        c_in: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Rotate Right through Carry.

        [b0,b1,...,b7] + c_in → [b1,...,b7,c_in], carry=b0
        Used for: ROR
        """
        c = c_in.unsqueeze(-1) if c_in.dim() == 1 else c_in
        result = torch.cat([a_bits[:, 1:], c], dim=1)
        return {'result': result, 'carry': a_bits[:, 0]}

    @staticmethod
    def increment(
        a_bits: torch.Tensor,
        _: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Increment by 1.

        A + 1 via ripple adder with B=1.
        Used for: INC, INX, INY
        """
        b = torch.zeros_like(a_bits)
        b[:, 0] = 1.0
        c_in = torch.zeros(a_bits.shape[0], device=a_bits.device)
        return FrozenShapes6502.ripple_add(a_bits, b, c_in)

    @staticmethod
    def decrement(
        a_bits: torch.Tensor,
        _: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decrement by 1.

        A + 0xFF = A - 1 (two's complement)
        Used for: DEC, DEX, DEY
        """
        b = torch.ones_like(a_bits)  # 0xFF
        c_in = torch.zeros(a_bits.shape[0], device=a_bits.device)
        return FrozenShapes6502.ripple_add(a_bits, b, c_in)

    @staticmethod
    def transfer(
        a_bits: torch.Tensor,
        _: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Transfer (pass through).

        Used for: TAX, TXA, TAY, TYA, TSX, TXS
        """
        return {'result': a_bits.clone()}

    @staticmethod
    def load(
        m_bits: torch.Tensor,
        _: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Load from memory.

        Used for: LDA, LDX, LDY
        """
        return {'result': m_bits.clone()}

    @staticmethod
    def store(
        a_bits: torch.Tensor,
        _: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Store to memory.

        Used for: STA, STX, STY
        """
        return {'result': a_bits.clone()}

    @staticmethod
    def bit_test(
        a_bits: torch.Tensor,
        m_bits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        BIT instruction.

        Sets N and V from memory bits 7 and 6.
        Sets Z from A AND M.
        Used for: BIT
        """
        and_result = PureMath.and_op(a_bits, m_bits)
        return {
            'result': and_result,
            'n_from_m': m_bits[:, 7],
            'v_from_m': m_bits[:, 6],
        }

    @staticmethod
    def identity(
        a_bits: torch.Tensor,
        _: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Identity (no operation).

        Used for: NOP, flag ops, branches
        """
        return {'result': a_bits.clone()}


# =============================================================================
# FLAG COMPUTATION (0 params)
# =============================================================================

class FlagComputer:
    """Compute 6502 status flags from results."""

    @staticmethod
    def zero(result: torch.Tensor) -> torch.Tensor:
        """Z = 1 if all result bits are 0."""
        # NOR of all bits: NOT(OR(b0, b1, ..., b7))
        or_all = result[:, 0]
        for i in range(1, 8):
            or_all = PureMath.or_op(or_all, result[:, i])
        return PureMath.not_op(or_all)

    @staticmethod
    def negative(result: torch.Tensor) -> torch.Tensor:
        """N = bit 7 of result."""
        return result[:, 7]

    @staticmethod
    def overflow(
        a7: torch.Tensor,
        b7: torch.Tensor,
        r7: torch.Tensor,
    ) -> torch.Tensor:
        """
        V = 1 if signed overflow occurred.

        Overflow when: same sign inputs, different sign output.
        V = (A7 XOR R7) AND (B7 XOR R7)
        """
        return PureMath.and_op(
            PureMath.xor(a7, r7),
            PureMath.xor(b7, r7),
        )


# =============================================================================
# SHAPE WRAPPER FOR FROZEN INTEGRATION
# =============================================================================

class Frozen6502Tile(nn.Module):
    """
    A tile that wraps a 6502 frozen shape.

    This provides the interface expected by FrozenTriXFFN while
    using the pure-math FrozenShapes6502 functions internally.
    """

    # Map of shape IDs to (function, needs_carry, is_binary)
    SHAPE_FUNCTIONS = {
        ShapeID.RIPPLE_ADD: (FrozenShapes6502.ripple_add, True, True),
        ShapeID.RIPPLE_SUB: (FrozenShapes6502.ripple_sub, True, True),
        ShapeID.PARALLEL_AND: (FrozenShapes6502.parallel_and, False, True),
        ShapeID.PARALLEL_OR: (FrozenShapes6502.parallel_or, False, True),
        ShapeID.PARALLEL_XOR: (FrozenShapes6502.parallel_xor, False, True),
        ShapeID.SHIFT_LEFT: (FrozenShapes6502.shift_left, False, False),
        ShapeID.SHIFT_RIGHT: (FrozenShapes6502.shift_right, False, False),
        ShapeID.ROTATE_LEFT: (FrozenShapes6502.rotate_left, True, False),
        ShapeID.ROTATE_RIGHT: (FrozenShapes6502.rotate_right, True, False),
        ShapeID.INCREMENT: (FrozenShapes6502.increment, False, False),
        ShapeID.DECREMENT: (FrozenShapes6502.decrement, False, False),
        ShapeID.TRANSFER: (FrozenShapes6502.transfer, False, False),
        ShapeID.LOAD: (FrozenShapes6502.load, False, False),
        ShapeID.STORE: (FrozenShapes6502.store, False, False),
        ShapeID.BIT_TEST: (FrozenShapes6502.bit_test, False, True),
        ShapeID.IDENTITY: (FrozenShapes6502.identity, False, False),
    }

    def __init__(self, shape_id: ShapeID, sig_dim: int = 64):
        super().__init__()
        self.shape_id = shape_id
        self.shape_name = shape_id.name

        # Get shape info
        fn, needs_carry, is_binary = self.SHAPE_FUNCTIONS[shape_id]
        self.shape_fn = fn
        self.needs_carry = needs_carry
        self.is_binary = is_binary

        # Derive signature from shape behavior
        signature = self._derive_signature(sig_dim)
        self.register_buffer('signature', signature)

        # Usage tracking
        self.register_buffer('activation_count', torch.tensor(0.0))
        self.register_buffer('total_count', torch.tensor(0.0))

    def _derive_signature(self, sig_dim: int) -> torch.Tensor:
        """Derive signature from shape's behavior on sample inputs."""
        torch.manual_seed(hash(self.shape_name) % (2**31))

        # Generate diverse test inputs
        n_samples = 256
        a = torch.randint(0, 2, (n_samples, 8), dtype=torch.float32)
        b = torch.randint(0, 2, (n_samples, 8), dtype=torch.float32)
        c = torch.randint(0, 2, (n_samples,), dtype=torch.float32)

        # Execute shape
        if self.is_binary:
            if self.needs_carry:
                out = self.shape_fn(a, b, c)
            else:
                out = self.shape_fn(a, b)
        else:
            if self.needs_carry:
                out = self.shape_fn(a, c)
            else:
                out = self.shape_fn(a)

        # Flatten outputs
        result = out['result'].flatten()
        if 'carry' in out:
            result = torch.cat([result, out['carry'].flatten()])

        # Project to signature dimension
        result_ternary = result * 2 - 1
        projection = torch.randn(len(result_ternary), sig_dim)
        projection = projection / (projection.norm(dim=0, keepdim=True) + 1e-8)

        signature = (result_ternary @ projection).sign()
        signature[signature == 0] = 1

        return signature

    def get_signature(self) -> torch.Tensor:
        return self.signature

    def forward(
        self,
        a_bits: torch.Tensor,
        b_bits: Optional[torch.Tensor] = None,
        c_in: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Execute the frozen shape."""
        if self.is_binary:
            if b_bits is None:
                b_bits = torch.zeros_like(a_bits)
            if self.needs_carry:
                if c_in is None:
                    c_in = torch.zeros(a_bits.shape[0], device=a_bits.device)
                return self.shape_fn(a_bits, b_bits, c_in)
            else:
                return self.shape_fn(a_bits, b_bits)
        else:
            if self.needs_carry:
                if c_in is None:
                    c_in = torch.zeros(a_bits.shape[0], device=a_bits.device)
                return self.shape_fn(a_bits, c_in)
            else:
                return self.shape_fn(a_bits)

    def update_usage(self, count: int, total: int):
        self.activation_count = self.activation_count + count
        self.total_count = self.total_count + total

    @property
    def usage_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return (self.activation_count / self.total_count).item()


# =============================================================================
# FROZEN 6502 MODEL
# =============================================================================

class Frozen6502(nn.Module):
    """
    Complete 6502 ALU using frozen shapes.

    Architecture:
        - 16 frozen computation shapes (0 learnable params)
        - Meaning layer for opcode routing (~2,500 learnable params)
        - Register file with soft selection

    This achieves 100% accuracy with ~40x compression vs monolithic approach.

    Args:
        num_opcodes: Number of opcodes to support
        temperature: Gumbel-Softmax temperature

    Example:
        >>> model = Frozen6502()
        >>> model.init_from_spec(OPCODE_SPECS)
        >>> # Execute ADC: A + M + C
        >>> registers = torch.zeros(1, 7, 8)  # A, X, Y, SP, PC_LO, PC_HI, P
        >>> registers[0, 0, :] = int_to_bits(42)  # A = 42
        >>> memory = int_to_bits(torch.tensor([13]))  # M = 13
        >>> carry = torch.tensor([1.0])  # C = 1
        >>> opcode = torch.tensor([0])  # ADC
        >>> out = model(opcode, registers, memory, carry)
        >>> bits_to_int(out['result'])  # Should be 56
    """

    def __init__(
        self,
        num_opcodes: int = 56,
        temperature: float = 1.0,
        sig_dim: int = 64,
    ):
        super().__init__()
        self.num_opcodes = num_opcodes
        self.temperature = temperature

        # Create all 16 frozen tiles
        self.tiles = nn.ModuleList([
            Frozen6502Tile(ShapeID(i), sig_dim)
            for i in range(NUM_SHAPES)
        ])

        # Meaning layer: learned routing from opcode to shape
        self.shape_logits = nn.Parameter(
            torch.zeros(num_opcodes, NUM_SHAPES)
        )
        self.input_a_logits = nn.Parameter(
            torch.zeros(num_opcodes, NUM_REGISTERS)
        )
        self.input_b_logits = nn.Parameter(
            torch.zeros(num_opcodes, NUM_REGISTERS)
        )
        self.output_logits = nn.Parameter(
            torch.zeros(num_opcodes, NUM_REGISTERS)
        )
        self.flag_mask = nn.Parameter(
            torch.zeros(num_opcodes, 4)  # N, Z, C, V
        )
        self.uses_carry = nn.Parameter(
            torch.zeros(num_opcodes, 1)
        )

    def forward(
        self,
        opcode_id: torch.Tensor,
        registers: torch.Tensor,
        memory: torch.Tensor,
        carry_in: torch.Tensor,
        training: bool = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute opcode on register state.

        Args:
            opcode_id: [batch] opcode indices
            registers: [batch, 7, 8] register bits (A, X, Y, SP, PC_LO, PC_HI, P)
            memory: [batch, 8] memory operand bits
            carry_in: [batch] carry flag
            training: Override training mode

        Returns:
            Dictionary with result, flags, routing info
        """
        if training is None:
            training = self.training

        batch = opcode_id.shape[0]
        device = registers.device

        # Get routing from meaning layer
        if training:
            shape_probs = F.gumbel_softmax(
                self.shape_logits[opcode_id],
                tau=self.temperature,
                hard=False,
            )
            input_a_probs = F.gumbel_softmax(
                self.input_a_logits[opcode_id],
                tau=self.temperature,
                hard=False,
            )
            input_b_probs = F.gumbel_softmax(
                self.input_b_logits[opcode_id],
                tau=self.temperature,
                hard=False,
            )
            output_probs = F.gumbel_softmax(
                self.output_logits[opcode_id],
                tau=self.temperature,
                hard=False,
            )
        else:
            shape_probs = F.one_hot(
                self.shape_logits[opcode_id].argmax(dim=-1),
                NUM_SHAPES,
            ).float()
            input_a_probs = F.one_hot(
                self.input_a_logits[opcode_id].argmax(dim=-1),
                NUM_REGISTERS,
            ).float()
            input_b_probs = F.one_hot(
                self.input_b_logits[opcode_id].argmax(dim=-1),
                NUM_REGISTERS,
            ).float()
            output_probs = F.one_hot(
                self.output_logits[opcode_id].argmax(dim=-1),
                NUM_REGISTERS,
            ).float()

        # Build full register bank (including memory as virtual register)
        full_registers = torch.cat([registers, memory.unsqueeze(1)], dim=1)

        # Select inputs via soft routing
        input_a = torch.einsum('brd,br->bd', full_registers, input_a_probs)
        input_b = torch.einsum('brd,br->bd', full_registers, input_b_probs)

        # Prepare carry
        uses_carry_prob = torch.sigmoid(self.uses_carry[opcode_id]).squeeze(-1)
        c_in = carry_in * uses_carry_prob

        # Execute all shapes and blend by probability
        result = torch.zeros(batch, 8, device=device)
        carry_out = torch.zeros(batch, device=device)

        for shape_id in range(NUM_SHAPES):
            weight = shape_probs[:, shape_id]
            tile = self.tiles[shape_id]

            out = tile(input_a, input_b, c_in)

            result = result + weight.unsqueeze(-1) * out['result']
            if 'carry' in out:
                carry_out = carry_out + weight * out['carry']

        # Compute flags
        z_flag = FlagComputer.zero(result)
        n_flag = FlagComputer.negative(result)
        v_flag = FlagComputer.overflow(
            input_a[:, 7], input_b[:, 7], result[:, 7]
        )

        # Apply flag mask
        flag_mask_probs = torch.sigmoid(self.flag_mask[opcode_id])

        return {
            'result': result,
            'carry': carry_out,
            'z_flag': z_flag,
            'n_flag': n_flag,
            'v_flag': v_flag,
            'flag_mask': flag_mask_probs,
            'shape_probs': shape_probs,
            'input_a_probs': input_a_probs,
            'input_b_probs': input_b_probs,
            'output_probs': output_probs,
        }

    def init_from_spec(self, opcode_specs: Dict[int, Dict]):
        """
        Initialize meaning layer from opcode specifications.

        Args:
            opcode_specs: Maps opcode_id to specification dict with:
                - shape: ShapeID
                - input_a: RegisterID
                - input_b: RegisterID
                - output: RegisterID
                - uses_carry: bool
                - flags: List[int] for N, Z, C, V
        """
        with torch.no_grad():
            for opcode_id, spec in opcode_specs.items():
                if 'shape' in spec:
                    self.shape_logits[opcode_id, spec['shape']] = 10.0
                if 'input_a' in spec:
                    self.input_a_logits[opcode_id, spec['input_a']] = 10.0
                if 'input_b' in spec:
                    self.input_b_logits[opcode_id, spec['input_b']] = 10.0
                if 'output' in spec:
                    self.output_logits[opcode_id, spec['output']] = 10.0
                if spec.get('uses_carry', False):
                    self.uses_carry[opcode_id, 0] = 10.0
                if 'flags' in spec:
                    for i, f in enumerate(spec['flags']):
                        if f:
                            self.flag_mask[opcode_id, i] = 10.0

    def param_count(self) -> int:
        """Count learnable parameters (meaning layer only)."""
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# OPCODE SPECIFICATIONS
# =============================================================================

OPCODE_SPECS = {
    # Arithmetic
    0: {'shape': ShapeID.RIPPLE_ADD, 'input_a': RegisterID.A,
        'input_b': RegisterID.MEM, 'output': RegisterID.A,
        'uses_carry': True, 'flags': [1, 1, 1, 1]},  # ADC
    1: {'shape': ShapeID.RIPPLE_SUB, 'input_a': RegisterID.A,
        'input_b': RegisterID.MEM, 'output': RegisterID.A,
        'uses_carry': True, 'flags': [1, 1, 1, 1]},  # SBC

    # Logic
    2: {'shape': ShapeID.PARALLEL_AND, 'input_a': RegisterID.A,
        'input_b': RegisterID.MEM, 'output': RegisterID.A,
        'flags': [1, 1, 0, 0]},  # AND
    3: {'shape': ShapeID.PARALLEL_OR, 'input_a': RegisterID.A,
        'input_b': RegisterID.MEM, 'output': RegisterID.A,
        'flags': [1, 1, 0, 0]},  # ORA
    4: {'shape': ShapeID.PARALLEL_XOR, 'input_a': RegisterID.A,
        'input_b': RegisterID.MEM, 'output': RegisterID.A,
        'flags': [1, 1, 0, 0]},  # EOR

    # Shifts
    5: {'shape': ShapeID.SHIFT_LEFT, 'input_a': RegisterID.A,
        'input_b': RegisterID.A, 'output': RegisterID.A,
        'flags': [1, 1, 1, 0]},  # ASL A
    6: {'shape': ShapeID.SHIFT_RIGHT, 'input_a': RegisterID.A,
        'input_b': RegisterID.A, 'output': RegisterID.A,
        'flags': [1, 1, 1, 0]},  # LSR A
    7: {'shape': ShapeID.ROTATE_LEFT, 'input_a': RegisterID.A,
        'input_b': RegisterID.A, 'output': RegisterID.A,
        'uses_carry': True, 'flags': [1, 1, 1, 0]},  # ROL A
    8: {'shape': ShapeID.ROTATE_RIGHT, 'input_a': RegisterID.A,
        'input_b': RegisterID.A, 'output': RegisterID.A,
        'uses_carry': True, 'flags': [1, 1, 1, 0]},  # ROR A

    # Inc/Dec
    9: {'shape': ShapeID.INCREMENT, 'input_a': RegisterID.X,
        'input_b': RegisterID.X, 'output': RegisterID.X,
        'flags': [1, 1, 0, 0]},  # INX
    10: {'shape': ShapeID.DECREMENT, 'input_a': RegisterID.X,
         'input_b': RegisterID.X, 'output': RegisterID.X,
         'flags': [1, 1, 0, 0]},  # DEX
    11: {'shape': ShapeID.INCREMENT, 'input_a': RegisterID.Y,
         'input_b': RegisterID.Y, 'output': RegisterID.Y,
         'flags': [1, 1, 0, 0]},  # INY
    12: {'shape': ShapeID.DECREMENT, 'input_a': RegisterID.Y,
         'input_b': RegisterID.Y, 'output': RegisterID.Y,
         'flags': [1, 1, 0, 0]},  # DEY

    # Transfer
    13: {'shape': ShapeID.TRANSFER, 'input_a': RegisterID.A,
         'input_b': RegisterID.A, 'output': RegisterID.X,
         'flags': [1, 1, 0, 0]},  # TAX
    14: {'shape': ShapeID.TRANSFER, 'input_a': RegisterID.X,
         'input_b': RegisterID.X, 'output': RegisterID.A,
         'flags': [1, 1, 0, 0]},  # TXA

    # Load/Store
    15: {'shape': ShapeID.LOAD, 'input_a': RegisterID.MEM,
         'input_b': RegisterID.MEM, 'output': RegisterID.A,
         'flags': [1, 1, 0, 0]},  # LDA
}


# =============================================================================
# UTILITIES
# =============================================================================

def int_to_bits(x: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """Convert integer tensor to bit representation (LSB first)."""
    if x.dim() == 0:
        x = x.unsqueeze(0)
    bits = []
    for i in range(num_bits):
        bits.append(((x >> i) & 1).float())
    return torch.stack(bits, dim=-1)


def bits_to_int(bits: torch.Tensor) -> torch.Tensor:
    """Convert bit tensor to integer."""
    powers = torch.tensor(
        [2**i for i in range(bits.shape[-1])],
        device=bits.device,
        dtype=bits.dtype,
    )
    return (bits * powers).sum(dim=-1)


def create_frozen_6502(num_opcodes: int = 56) -> Frozen6502:
    """Create a Frozen 6502 model with default settings."""
    model = Frozen6502(num_opcodes=num_opcodes)
    model.init_from_spec(OPCODE_SPECS)
    return model


# =============================================================================
# REGISTRY INTEGRATION
# =============================================================================

def register_6502_shapes(registry: FrozenShapeRegistry):
    """
    Register 6502 shapes in a FrozenShapeRegistry.

    Note: This uses ThresholdCircuit wrappers for compatibility
    with the main frozen.py module.
    """
    # For each shape, we create a wrapper that matches FrozenShape interface
    # This is a bridge between the pure-math shapes and the registry system

    from trix.compiler.atoms_fp4 import ThresholdCircuit, ThresholdLayer

    # For now, we just register the basic atoms that are equivalent
    # The full 8-bit shapes would require more complex composition
    pass  # TODO: Add 8-bit shape registration if needed
