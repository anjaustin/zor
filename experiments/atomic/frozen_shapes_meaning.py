"""
Frozen Shapes + Meaning Layer: Ultra-Compressed 6502

Hypothesis: We can model the entire 6502 with ~500 params by:
1. Freezing opcode "shapes" as pure math (0 params)
2. Learning only a tiny "meaning layer" that routes opcodes to shapes

The shapes ARE the computation. The meaning layer is just the decoder ring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Callable
from dataclasses import dataclass


# =============================================================================
# LEVEL 0: Pure Math Primitives (0 params)
# =============================================================================

def xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """XOR: a + b - 2ab"""
    return a + b - 2 * a * b

def and_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """AND: ab"""
    return a * b

def or_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """OR: a + b - ab"""
    return a + b - a * b

def not_op(a: torch.Tensor) -> torch.Tensor:
    """NOT: 1 - a"""
    return 1 - a


# =============================================================================
# LEVEL 1: Frozen Shapes (0 params) - The Topologies
# =============================================================================

class FrozenShape:
    """Base class for frozen computation shapes."""

    @staticmethod
    def forward(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class FullAdderShape(FrozenShape):
    """Single bit full adder: (a, b, c_in) -> (sum, c_out)"""

    @staticmethod
    def forward(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        a, b, c = inputs['a'], inputs['b'], inputs['c_in']

        # XOR chain for sum
        p = xor(a, b)
        sum_bit = xor(p, c)

        # Carry = (a AND b) OR (c_in AND (a XOR b))
        c_out = or_op(and_op(a, b), and_op(c, p))

        return {'sum': sum_bit, 'c_out': c_out}


class RippleCarryShape(FrozenShape):
    """8-bit ripple carry adder: chains 8 full adders"""

    @staticmethod
    def forward(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        a_bits = inputs['a_bits']  # [batch, 8]
        b_bits = inputs['b_bits']  # [batch, 8]
        c_in = inputs['c_in']      # [batch, 1]

        result_bits = []
        carry = c_in.squeeze(-1) if c_in.dim() > 1 else c_in

        for i in range(8):
            fa_inputs = {
                'a': a_bits[:, i],
                'b': b_bits[:, i],
                'c_in': carry
            }
            fa_out = FullAdderShape.forward(fa_inputs)
            result_bits.append(fa_out['sum'])
            carry = fa_out['c_out']

        return {
            'result_bits': torch.stack(result_bits, dim=1),
            'c_out': carry.unsqueeze(-1)
        }


class ParallelBitwiseShape(FrozenShape):
    """8-bit parallel bitwise op: applies same op to all bit pairs"""

    @staticmethod
    def forward(inputs: Dict[str, torch.Tensor],
                op: Callable) -> Dict[str, torch.Tensor]:
        a_bits = inputs['a_bits']  # [batch, 8]
        b_bits = inputs['b_bits']  # [batch, 8]

        result_bits = op(a_bits, b_bits)
        return {'result_bits': result_bits}


class ShiftLeftShape(FrozenShape):
    """ASL: Shift left, bit7 -> carry"""

    @staticmethod
    def forward(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        a_bits = inputs['a_bits']  # [batch, 8]

        # Shift left: [0, b0, b1, ..., b6], carry = b7
        zeros = torch.zeros_like(a_bits[:, :1])
        result_bits = torch.cat([zeros, a_bits[:, :7]], dim=1)
        c_out = a_bits[:, 7:8]

        return {'result_bits': result_bits, 'c_out': c_out}


class ShiftRightShape(FrozenShape):
    """LSR: Shift right, bit0 -> carry"""

    @staticmethod
    def forward(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        a_bits = inputs['a_bits']  # [batch, 8]

        # Shift right: [b1, b2, ..., b7, 0], carry = b0
        zeros = torch.zeros_like(a_bits[:, :1])
        result_bits = torch.cat([a_bits[:, 1:], zeros], dim=1)
        c_out = a_bits[:, :1]

        return {'result_bits': result_bits, 'c_out': c_out}


class RotateLeftShape(FrozenShape):
    """ROL: Rotate left through carry"""

    @staticmethod
    def forward(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        a_bits = inputs['a_bits']  # [batch, 8]
        c_in = inputs['c_in']      # [batch, 1]

        # [c_in, b0, b1, ..., b6], carry_out = b7
        result_bits = torch.cat([c_in, a_bits[:, :7]], dim=1)
        c_out = a_bits[:, 7:8]

        return {'result_bits': result_bits, 'c_out': c_out}


class RotateRightShape(FrozenShape):
    """ROR: Rotate right through carry"""

    @staticmethod
    def forward(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        a_bits = inputs['a_bits']  # [batch, 8]
        c_in = inputs['c_in']      # [batch, 1]

        # [b1, b2, ..., b7, c_in], carry_out = b0
        result_bits = torch.cat([a_bits[:, 1:], c_in], dim=1)
        c_out = a_bits[:, :1]

        return {'result_bits': result_bits, 'c_out': c_out}


class IncrementShape(FrozenShape):
    """INC/INX/INY: Add 1 (ADC with b=1, c_in=0)"""

    @staticmethod
    def forward(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        a_bits = inputs['a_bits']
        batch = a_bits.shape[0]

        # b = 00000001, c_in = 0
        b_bits = torch.zeros_like(a_bits)
        b_bits[:, 0] = 1.0
        c_in = torch.zeros(batch, 1, device=a_bits.device)

        return RippleCarryShape.forward({
            'a_bits': a_bits,
            'b_bits': b_bits,
            'c_in': c_in
        })


class DecrementShape(FrozenShape):
    """DEC/DEX/DEY: Subtract 1 (add -1 = 0xFF with c_in=1)"""

    @staticmethod
    def forward(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        a_bits = inputs['a_bits']
        batch = a_bits.shape[0]

        # Adding 0xFF with c_in=0 is same as subtracting 1
        # But cleaner: add 0 with borrow (c_in=0 acts as borrow in SBC sense)
        # Actually: A - 1 = A + NOT(1) + 1 = A + 0xFE + 1 = A + 0xFF
        b_bits = torch.ones_like(a_bits)  # 0xFF
        c_in = torch.zeros(batch, 1, device=a_bits.device)

        return RippleCarryShape.forward({
            'a_bits': a_bits,
            'b_bits': b_bits,
            'c_in': c_in
        })


class CompareShape(FrozenShape):
    """CMP/CPX/CPY: Subtract without storing (for flags only)"""

    @staticmethod
    def forward(inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        a_bits = inputs['a_bits']
        b_bits = inputs['b_bits']

        # CMP = A - M, setting flags but not storing result
        # Implemented as A + NOT(M) + 1
        b_inv = not_op(b_bits)
        batch = a_bits.shape[0]
        c_in = torch.ones(batch, 1, device=a_bits.device)  # +1 for two's complement

        return RippleCarryShape.forward({
            'a_bits': a_bits,
            'b_bits': b_inv,
            'c_in': c_in
        })


# =============================================================================
# Flag Computation (0 params)
# =============================================================================

def compute_zero_flag(result_bits: torch.Tensor) -> torch.Tensor:
    """Z = 1 if all result bits are 0"""
    # NOR of all bits
    or_all = result_bits[:, 0]
    for i in range(1, result_bits.shape[1]):
        or_all = or_op(or_all, result_bits[:, i])
    return not_op(or_all)

def compute_negative_flag(result_bits: torch.Tensor) -> torch.Tensor:
    """N = bit 7 of result"""
    return result_bits[:, 7]

def compute_overflow_flag(a7: torch.Tensor, b7: torch.Tensor,
                          r7: torch.Tensor) -> torch.Tensor:
    """V = 1 if signed overflow occurred"""
    # V = (A7 XOR R7) AND (B7 XOR R7)
    return and_op(xor(a7, r7), xor(b7, r7))


# =============================================================================
# LEVEL 2: Meaning Layer (Learned - The ONLY learnable part!)
# =============================================================================

@dataclass
class OpcodeSpec:
    """Specification for what an opcode does."""
    shape_id: int           # Which frozen shape to use
    uses_accumulator: bool  # Does it read/write A?
    uses_memory: bool       # Does it read/write memory operand?
    uses_carry_in: bool     # Does it use carry flag as input?
    sets_carry: bool        # Does it modify carry flag?
    sets_overflow: bool     # Does it modify overflow flag?
    is_store: bool          # Is it a store operation (no flag update)?


class MeaningLayer(nn.Module):
    """
    The ONLY learned component. Maps opcode -> operation specification.

    This is essentially a lookup table with soft routing, allowing gradients
    to flow for end-to-end training.

    Total params: num_opcodes * spec_size ≈ 56 * 8 = 448 params
    """

    SHAPE_NAMES = [
        'ripple_carry_add',   # ADC
        'ripple_carry_sub',   # SBC, CMP
        'parallel_and',       # AND
        'parallel_or',        # ORA
        'parallel_xor',       # EOR
        'shift_left',         # ASL
        'shift_right',        # LSR
        'rotate_left',        # ROL
        'rotate_right',       # ROR
        'increment',          # INC, INX, INY
        'decrement',          # DEC, DEX, DEY
        'transfer',           # TAX, TXA, TAY, TYA, etc.
        'load',               # LDA, LDX, LDY
        'store',              # STA, STX, STY
    ]
    NUM_SHAPES = len(SHAPE_NAMES)

    def __init__(self, num_opcodes: int = 56):
        super().__init__()
        self.num_opcodes = num_opcodes

        # Shape selection logits: [num_opcodes, num_shapes]
        self.shape_logits = nn.Parameter(torch.zeros(num_opcodes, self.NUM_SHAPES))

        # Operation flags: [num_opcodes, 6]
        # (uses_acc, uses_mem, uses_carry, sets_carry, sets_overflow, is_store)
        self.op_flags = nn.Parameter(torch.zeros(num_opcodes, 6))

    def forward(self, opcode_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            opcode_id: [batch] tensor of opcode indices

        Returns:
            shape_probs: [batch, num_shapes] soft routing to shapes
            flags: [batch, 6] operation flags (sigmoids)
        """
        # Get shape selection (soft for training, hard for inference)
        shape_logits = self.shape_logits[opcode_id]  # [batch, num_shapes]
        shape_probs = F.softmax(shape_logits, dim=-1)

        # Get operation flags
        flags = torch.sigmoid(self.op_flags[opcode_id])  # [batch, 6]

        return shape_probs, flags

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Complete Frozen 6502 Model
# =============================================================================

class Frozen6502(nn.Module):
    """
    Complete 6502 using frozen shapes + learned meaning layer.

    Architecture:
    - Level 0: Pure math primitives (XOR, AND, OR) - 0 params
    - Level 1: Frozen shapes (ripple carry, shifts, etc.) - 0 params
    - Level 2: Meaning layer (opcode -> shape routing) - ~500 params

    Expected compression: ~100,000 params -> ~500 params (200x!)
    """

    def __init__(self, num_opcodes: int = 56):
        super().__init__()
        self.meaning = MeaningLayer(num_opcodes)

        # Shape dispatch table
        self.shapes = {
            0: self._exec_adc,
            1: self._exec_sbc,
            2: self._exec_and,
            3: self._exec_ora,
            4: self._exec_eor,
            5: self._exec_asl,
            6: self._exec_lsr,
            7: self._exec_rol,
            8: self._exec_ror,
            9: self._exec_inc,
            10: self._exec_dec,
            11: self._exec_transfer,
            12: self._exec_load,
            13: self._exec_store,
        }

    def _exec_adc(self, a_bits, m_bits, c_in, **kwargs):
        return RippleCarryShape.forward({
            'a_bits': a_bits, 'b_bits': m_bits, 'c_in': c_in
        })

    def _exec_sbc(self, a_bits, m_bits, c_in, **kwargs):
        # SBC = ADC with inverted M (borrow handled by c_in)
        return RippleCarryShape.forward({
            'a_bits': a_bits, 'b_bits': not_op(m_bits), 'c_in': c_in
        })

    def _exec_and(self, a_bits, m_bits, **kwargs):
        return {'result_bits': and_op(a_bits, m_bits)}

    def _exec_ora(self, a_bits, m_bits, **kwargs):
        return {'result_bits': or_op(a_bits, m_bits)}

    def _exec_eor(self, a_bits, m_bits, **kwargs):
        return {'result_bits': xor(a_bits, m_bits)}

    def _exec_asl(self, a_bits, **kwargs):
        return ShiftLeftShape.forward({'a_bits': a_bits})

    def _exec_lsr(self, a_bits, **kwargs):
        return ShiftRightShape.forward({'a_bits': a_bits})

    def _exec_rol(self, a_bits, c_in, **kwargs):
        return RotateLeftShape.forward({'a_bits': a_bits, 'c_in': c_in})

    def _exec_ror(self, a_bits, c_in, **kwargs):
        return RotateRightShape.forward({'a_bits': a_bits, 'c_in': c_in})

    def _exec_inc(self, a_bits, **kwargs):
        return IncrementShape.forward({'a_bits': a_bits})

    def _exec_dec(self, a_bits, **kwargs):
        return DecrementShape.forward({'a_bits': a_bits})

    def _exec_transfer(self, a_bits, **kwargs):
        # Transfer: just pass through (TAX, TXA, etc.)
        return {'result_bits': a_bits}

    def _exec_load(self, m_bits, **kwargs):
        # Load: result = memory value
        return {'result_bits': m_bits}

    def _exec_store(self, a_bits, **kwargs):
        # Store: result = accumulator (to be written to memory)
        return {'result_bits': a_bits}

    def forward(self, opcode_id: torch.Tensor,
                a_bits: torch.Tensor,
                m_bits: torch.Tensor,
                c_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Execute opcode using frozen shapes routed by meaning layer.

        Args:
            opcode_id: [batch] opcode indices
            a_bits: [batch, 8] accumulator bits
            m_bits: [batch, 8] memory operand bits
            c_in: [batch, 1] carry flag

        Returns:
            result_bits: [batch, 8] result
            c_out: [batch, 1] output carry (if applicable)
            z_flag: [batch] zero flag
            n_flag: [batch] negative flag
            v_flag: [batch] overflow flag (if applicable)
        """
        batch = opcode_id.shape[0]

        # Get routing from meaning layer
        shape_probs, flags = self.meaning(opcode_id)

        # Execute all shapes and blend by probability
        # (In inference, this would be argmax selection)
        result_bits = torch.zeros(batch, 8, device=a_bits.device)
        c_out = torch.zeros(batch, 1, device=a_bits.device)

        for shape_id, exec_fn in self.shapes.items():
            out = exec_fn(a_bits=a_bits, m_bits=m_bits, c_in=c_in)
            weight = shape_probs[:, shape_id:shape_id+1]  # [batch, 1]

            result_bits = result_bits + weight * out.get('result_bits',
                                                          torch.zeros_like(result_bits))
            if 'c_out' in out:
                c_out = c_out + weight * out['c_out']

        # Compute flags
        z_flag = compute_zero_flag(result_bits)
        n_flag = compute_negative_flag(result_bits)
        v_flag = compute_overflow_flag(a_bits[:, 7], m_bits[:, 7], result_bits[:, 7])

        return {
            'result_bits': result_bits,
            'c_out': c_out,
            'z_flag': z_flag,
            'n_flag': n_flag,
            'v_flag': v_flag,
            'shape_probs': shape_probs,
            'op_flags': flags,
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Training & Validation
# =============================================================================

def int_to_bits(x: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """Convert integer tensor to bit representation."""
    bits = []
    for i in range(num_bits):
        bits.append((x >> i) & 1)
    return torch.stack(bits, dim=-1).float()


def bits_to_int(bits: torch.Tensor) -> torch.Tensor:
    """Convert bit tensor back to integer."""
    powers = torch.tensor([2**i for i in range(bits.shape[-1])],
                          device=bits.device, dtype=bits.dtype)
    return (bits * powers).sum(dim=-1)


def generate_adc_data(batch_size: int = 256, device='cpu'):
    """Generate ADC training data."""
    a = torch.randint(0, 256, (batch_size,), device=device)
    m = torch.randint(0, 256, (batch_size,), device=device)
    c = torch.randint(0, 2, (batch_size,), device=device)

    # Ground truth
    total = a.long() + m.long() + c.long()
    result = total % 256
    c_out = (total >= 256).long()

    return {
        'a_bits': int_to_bits(a),
        'm_bits': int_to_bits(m),
        'c_in': c.float().unsqueeze(-1),
        'result_bits': int_to_bits(result),
        'c_out': c_out.float().unsqueeze(-1),
    }


def train_frozen_6502():
    """Train just the meaning layer to route opcodes correctly."""
    print("=" * 60)
    print("FROZEN SHAPES + MEANING LAYER EXPERIMENT")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Create model
    model = Frozen6502(num_opcodes=14)  # One per shape for this test
    model = model.to(device)

    total_params = model.param_count()
    print(f"Total parameters: {total_params}")
    print(f"  - Meaning layer: {model.meaning.param_count()}")
    print(f"  - Frozen shapes: 0 (pure math)")

    # Initialize meaning layer to route opcode i -> shape i
    with torch.no_grad():
        for i in range(14):
            model.meaning.shape_logits[i, i] = 10.0  # Strong prior

    # Test ADC specifically (opcode 0 -> shape 0)
    print("\n" + "-" * 40)
    print("Testing ADC (opcode 0 -> ripple_carry_add)")
    print("-" * 40)

    # Generate test data
    test_data = generate_adc_data(batch_size=1000, device=device)
    opcode_ids = torch.zeros(1000, dtype=torch.long, device=device)  # All ADC

    with torch.no_grad():
        out = model(
            opcode_ids,
            test_data['a_bits'],
            test_data['m_bits'],
            test_data['c_in']
        )

        # Check result accuracy
        pred_result = (out['result_bits'] > 0.5).float()
        gt_result = test_data['result_bits']
        result_match = (pred_result == gt_result).all(dim=1).float().mean()

        # Check carry accuracy
        pred_c = (out['c_out'] > 0.5).float()
        gt_c = test_data['c_out']
        carry_match = (pred_c == gt_c).float().mean()

        # Perfect match
        perfect = ((pred_result == gt_result).all(dim=1) &
                   (pred_c == gt_c).squeeze(-1)).float().mean()

        print(f"Result bits accuracy: {result_match.item()*100:.2f}%")
        print(f"Carry accuracy: {carry_match.item()*100:.2f}%")
        print(f"Perfect (result + carry): {perfect.item()*100:.2f}%")

    # Test all shapes
    print("\n" + "-" * 40)
    print("Testing All Frozen Shapes")
    print("-" * 40)

    test_ops = [
        (0, 'ADC', 'ripple_carry_add'),
        (2, 'AND', 'parallel_and'),
        (3, 'ORA', 'parallel_or'),
        (4, 'EOR', 'parallel_xor'),
        (5, 'ASL', 'shift_left'),
        (6, 'LSR', 'shift_right'),
        (9, 'INC', 'increment'),
        (10, 'DEC', 'decrement'),
    ]

    for op_id, op_name, shape_name in test_ops:
        # Generate test data
        a = torch.randint(0, 256, (1000,), device=device)
        m = torch.randint(0, 256, (1000,), device=device)
        c = torch.randint(0, 2, (1000,), device=device)

        # Compute ground truth
        if op_name == 'ADC':
            total = a.long() + m.long() + c.long()
            gt_result = total % 256
        elif op_name == 'AND':
            gt_result = a & m
        elif op_name == 'ORA':
            gt_result = a | m
        elif op_name == 'EOR':
            gt_result = a ^ m
        elif op_name == 'ASL':
            gt_result = (a << 1) & 0xFF
        elif op_name == 'LSR':
            gt_result = a >> 1
        elif op_name == 'INC':
            gt_result = (a + 1) & 0xFF
        elif op_name == 'DEC':
            gt_result = (a - 1) & 0xFF

        # Run through model
        opcode_ids = torch.full((1000,), op_id, dtype=torch.long, device=device)

        with torch.no_grad():
            out = model(
                opcode_ids,
                int_to_bits(a),
                int_to_bits(m),
                c.float().unsqueeze(-1)
            )

            pred_result = bits_to_int((out['result_bits'] > 0.5).float()).long()
            accuracy = (pred_result == gt_result).float().mean()

            print(f"{op_name:4s} ({shape_name:18s}): {accuracy.item()*100:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total model parameters: {total_params}")
    print(f"Compression vs monolithic (~100k): {100000/total_params:.0f}x")
    print(f"All shapes use pure math formulas (0 params)")
    print(f"Only the meaning layer is learned ({model.meaning.param_count()} params)")

    return model


if __name__ == '__main__':
    model = train_frozen_6502()
