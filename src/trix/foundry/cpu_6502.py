"""
Full MOS 6502 CPU - Frozen Shapes Architecture.

Mesa 16.4: Complete 6502 from Frozen Geometry

32 frozen shapes:
- 16 ALU operations (0-15)
- 10 addressing modes (16-25)
- 4 control flow (26-29)
- 2 stack operations (30-31)

All computation is frozen geometry. Only routing is learned.

"The geometry was always there. We just learned to route to it."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import IntEnum
import time

from trix.foundry.pipeline_6502 import (
    PureMath, FrozenShapes, SHAPE_FUNCTIONS,
    int_to_bits, bits_to_int,
)
from trix.doula import DoulaFunction


# =============================================================================
# EXTENDED SHAPE IDS (32 total)
# =============================================================================

class ShapeID(IntEnum):
    """All 32 frozen shapes for complete 6502."""

    # ALU Operations (0-15) - from pipeline_6502
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

    # Addressing Modes (16-25)
    ADDR_IMMEDIATE = 16
    ADDR_ZERO_PAGE = 17
    ADDR_ZERO_PAGE_X = 18
    ADDR_ZERO_PAGE_Y = 19
    ADDR_ABSOLUTE = 20
    ADDR_ABSOLUTE_X = 21
    ADDR_ABSOLUTE_Y = 22
    ADDR_INDIRECT = 23
    ADDR_INDEXED_INDIRECT = 24
    ADDR_INDIRECT_INDEXED = 25

    # Control Flow (26-29)
    BRANCH = 26
    JUMP = 27
    JSR = 28
    RTS = 29

    # Stack (30-31)
    PUSH = 30
    PULL = 31


# =============================================================================
# 6502 REGISTERS (State)
# =============================================================================

@dataclass
class CPUState:
    """6502 CPU register state."""
    A: Tensor      # Accumulator (8 bits)
    X: Tensor      # Index X (8 bits)
    Y: Tensor      # Index Y (8 bits)
    SP: Tensor     # Stack Pointer (8 bits)
    PC: Tensor     # Program Counter (16 bits)
    P: Tensor      # Status flags (8 bits: NV-BDIZC)

    @classmethod
    def reset(cls, batch_size: int = 1, device: str = 'cpu') -> 'CPUState':
        """Reset CPU to initial state."""
        return cls(
            A=torch.zeros(batch_size, 8, device=device),
            X=torch.zeros(batch_size, 8, device=device),
            Y=torch.zeros(batch_size, 8, device=device),
            SP=torch.ones(batch_size, 8, device=device),  # SP = 0xFF
            PC=torch.zeros(batch_size, 16, device=device),
            P=torch.zeros(batch_size, 8, device=device),
        )

    @property
    def N(self) -> Tensor:
        """Negative flag."""
        return self.P[:, 7]

    @property
    def V(self) -> Tensor:
        """Overflow flag."""
        return self.P[:, 6]

    @property
    def B(self) -> Tensor:
        """Break flag."""
        return self.P[:, 4]

    @property
    def D(self) -> Tensor:
        """Decimal flag."""
        return self.P[:, 3]

    @property
    def I(self) -> Tensor:
        """Interrupt disable flag."""
        return self.P[:, 2]

    @property
    def Z(self) -> Tensor:
        """Zero flag."""
        return self.P[:, 1]

    @property
    def C(self) -> Tensor:
        """Carry flag."""
        return self.P[:, 0]


# =============================================================================
# ADDRESSING MODE SHAPES (Frozen)
# =============================================================================

class AddressingModes:
    """Frozen addressing mode computations."""

    @staticmethod
    def immediate(operand: Tensor, _x: Tensor, _y: Tensor) -> Tensor:
        """Immediate: operand is the value."""
        return operand

    @staticmethod
    def zero_page(addr: Tensor, _x: Tensor, _y: Tensor) -> Tensor:
        """Zero page: addr is 8-bit address."""
        return addr[:, :8]  # Just the low byte

    @staticmethod
    def zero_page_x(addr: Tensor, x: Tensor, _y: Tensor) -> Tensor:
        """Zero page,X: (addr + X) & 0xFF."""
        # Add with wrap at 256
        result, _ = FrozenShapes.ripple_add(addr[:, :8], x, torch.zeros(addr.shape[0], device=addr.device))
        return result

    @staticmethod
    def zero_page_y(addr: Tensor, _x: Tensor, y: Tensor) -> Tensor:
        """Zero page,Y: (addr + Y) & 0xFF."""
        result, _ = FrozenShapes.ripple_add(addr[:, :8], y, torch.zeros(addr.shape[0], device=addr.device))
        return result

    @staticmethod
    def absolute(addr: Tensor, _x: Tensor, _y: Tensor) -> Tensor:
        """Absolute: 16-bit address."""
        return addr[:, :16]

    @staticmethod
    def absolute_x(addr: Tensor, x: Tensor, _y: Tensor) -> Tensor:
        """Absolute,X: addr + X (16-bit)."""
        # Extend X to 16 bits
        x_16 = torch.cat([x, torch.zeros_like(x)], dim=-1)
        result, _ = FrozenShapes.ripple_add(addr[:, :16], x_16, torch.zeros(addr.shape[0], device=addr.device))
        return result

    @staticmethod
    def absolute_y(addr: Tensor, _x: Tensor, y: Tensor) -> Tensor:
        """Absolute,Y: addr + Y (16-bit)."""
        y_16 = torch.cat([y, torch.zeros_like(y)], dim=-1)
        result, _ = FrozenShapes.ripple_add(addr[:, :16], y_16, torch.zeros(addr.shape[0], device=addr.device))
        return result

    @staticmethod
    def indirect(addr: Tensor, _x: Tensor, _y: Tensor) -> Tensor:
        """Indirect: address points to address (for JMP)."""
        # This requires memory access - return the pointer address
        return addr[:, :16]

    @staticmethod
    def indexed_indirect(addr: Tensor, x: Tensor, _y: Tensor) -> Tensor:
        """Indexed indirect: (addr + X) & 0xFF points to address."""
        result, _ = FrozenShapes.ripple_add(addr[:, :8], x, torch.zeros(addr.shape[0], device=addr.device))
        return result

    @staticmethod
    def indirect_indexed(addr: Tensor, _x: Tensor, y: Tensor) -> Tensor:
        """Indirect indexed: addr points to base, add Y."""
        # First dereference gives base, then add Y
        # For now return the zero page address (memory access needed)
        return addr[:, :8]


# =============================================================================
# CONTROL FLOW SHAPES (Frozen)
# =============================================================================

class ControlFlow:
    """Frozen control flow computations."""

    @staticmethod
    def branch(pc: Tensor, offset: Tensor, condition: Tensor) -> Tensor:
        """
        Branch: PC + offset if condition is true.

        offset is signed 8-bit (-128 to +127)
        """
        # Sign extend offset to 16 bits
        sign = offset[:, 7:8]
        offset_16 = torch.cat([offset, sign.expand(-1, 8)], dim=-1)

        # Calculate target
        target, _ = FrozenShapes.ripple_add(
            pc, offset_16,
            torch.zeros(pc.shape[0], device=pc.device)
        )

        # Select based on condition
        condition = condition.unsqueeze(-1)
        return pc * (1 - condition) + target * condition

    @staticmethod
    def jump(target: Tensor) -> Tensor:
        """Jump: PC = target."""
        return target

    @staticmethod
    def jsr(pc: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        """
        JSR: Push PC+2, jump to target.
        Returns (new_pc, return_address).
        """
        # PC + 2 (actually PC + 3 - 1 for 6502 quirk)
        two = torch.zeros_like(pc)
        two[:, 1] = 1.0  # 2 in binary
        return_addr, _ = FrozenShapes.ripple_add(
            pc, two,
            torch.zeros(pc.shape[0], device=pc.device)
        )
        return target, return_addr

    @staticmethod
    def rts(return_addr: Tensor) -> Tensor:
        """RTS: PC = return_address + 1."""
        one = torch.zeros_like(return_addr)
        one[:, 0] = 1.0
        result, _ = FrozenShapes.ripple_add(
            return_addr, one,
            torch.zeros(return_addr.shape[0], device=return_addr.device)
        )
        return result


# =============================================================================
# STACK SHAPES (Frozen)
# =============================================================================

class StackOps:
    """Frozen stack operations."""

    @staticmethod
    def push_addr(sp: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Calculate stack address and decrement SP.
        Returns (stack_addr, new_sp).
        Stack is at 0x0100 + SP.
        """
        # Stack address = 0x0100 | SP
        stack_addr = torch.cat([sp, torch.zeros_like(sp)], dim=-1)
        stack_addr[:, 8] = 1.0  # Set bit 8 for 0x0100

        # Decrement SP
        new_sp, _ = FrozenShapes.decrement(sp, sp, sp)  # dummy args

        return stack_addr, new_sp

    @staticmethod
    def pull_addr(sp: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Increment SP and calculate stack address.
        Returns (stack_addr, new_sp).
        """
        # Increment SP first
        new_sp, _ = FrozenShapes.increment(sp, sp, sp)  # dummy args

        # Stack address = 0x0100 | new_SP
        stack_addr = torch.cat([new_sp, torch.zeros_like(new_sp)], dim=-1)
        stack_addr[:, 8] = 1.0

        return stack_addr, new_sp


# =============================================================================
# FLAG COMPUTATION (Frozen)
# =============================================================================

class FlagCompute:
    """Frozen flag computations."""

    @staticmethod
    def compute_nz(result: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute N and Z flags from result."""
        n = result[:, 7]  # Bit 7
        z = (result.sum(dim=-1) == 0).float()  # All zeros
        return n, z

    @staticmethod
    def compute_v(a: Tensor, b: Tensor, result: Tensor) -> Tensor:
        """Compute overflow flag for ADC/SBC."""
        # V = (A[7] == B[7]) && (A[7] != R[7])
        a7, b7, r7 = a[:, 7], b[:, 7], result[:, 7]
        same_sign = PureMath.xor(a7, b7)  # 0 if same
        diff_result = PureMath.xor(a7, r7)  # 1 if different
        return PureMath.and_op(1 - same_sign, diff_result)

    @staticmethod
    def apply_flags(
        p: Tensor,
        n: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        z: Optional[Tensor] = None,
        c: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply flag updates with mask."""
        new_p = p.clone()
        if n is not None:
            new_p[:, 7] = n
        if v is not None:
            new_p[:, 6] = v
        if z is not None:
            new_p[:, 1] = z
        if c is not None:
            new_p[:, 0] = c
        return new_p


# =============================================================================
# INSTRUCTION DECODER (Learned Routing)
# =============================================================================

@dataclass
class DecodedInstruction:
    """Decoded instruction fields."""
    alu_shape: int         # Which ALU operation (0-15)
    addr_mode: int         # Which addressing mode (16-25)
    src_a: str             # Source for A input ('A', 'X', 'Y', 'M', 'IMM')
    src_b: str             # Source for B input ('A', 'X', 'Y', 'M', 'IMM')
    dst: str               # Destination ('A', 'X', 'Y', 'M', 'SP', 'PC')
    flags_mask: int        # Which flags to update (NV-BDIZC as bits)
    is_branch: bool        # Branch instruction
    is_jump: bool          # Jump instruction
    is_stack: bool         # Stack operation
    bytes: int             # Instruction length


class InstructionDecoder(nn.Module):
    """
    Learned instruction decoder.

    Maps 256 opcodes to instruction fields.
    This is the ONLY learned component - everything else is frozen.
    """

    def __init__(self, d_hidden: int = 64):
        super().__init__()

        # Opcode embedding
        self.opcode_embed = nn.Embedding(256, d_hidden)

        # Decode heads
        self.alu_head = nn.Linear(d_hidden, 16)      # 16 ALU shapes
        self.addr_head = nn.Linear(d_hidden, 10)     # 10 addressing modes
        self.src_a_head = nn.Linear(d_hidden, 5)     # 5 sources
        self.src_b_head = nn.Linear(d_hidden, 5)
        self.dst_head = nn.Linear(d_hidden, 6)       # 6 destinations
        self.flags_head = nn.Linear(d_hidden, 8)     # 8 flag bits
        self.control_head = nn.Linear(d_hidden, 3)   # branch/jump/stack
        self.bytes_head = nn.Linear(d_hidden, 3)     # 1/2/3 bytes

        self.temperature = 0.5

    def forward(self, opcode: Tensor) -> Dict[str, Tensor]:
        """Decode opcodes to instruction fields."""
        h = self.opcode_embed(opcode)

        if self.training:
            alu = F.softmax(self.alu_head(h) / self.temperature, dim=-1)
            addr = F.softmax(self.addr_head(h) / self.temperature, dim=-1)
            src_a = F.softmax(self.src_a_head(h) / self.temperature, dim=-1)
            src_b = F.softmax(self.src_b_head(h) / self.temperature, dim=-1)
            dst = F.softmax(self.dst_head(h) / self.temperature, dim=-1)
        else:
            alu = F.one_hot(self.alu_head(h).argmax(dim=-1), 16).float()
            addr = F.one_hot(self.addr_head(h).argmax(dim=-1), 10).float()
            src_a = F.one_hot(self.src_a_head(h).argmax(dim=-1), 5).float()
            src_b = F.one_hot(self.src_b_head(h).argmax(dim=-1), 5).float()
            dst = F.one_hot(self.dst_head(h).argmax(dim=-1), 6).float()

        flags = torch.sigmoid(self.flags_head(h))
        control = torch.sigmoid(self.control_head(h))
        bytes_logits = F.softmax(self.bytes_head(h) / self.temperature, dim=-1)

        return {
            'alu': alu,
            'addr': addr,
            'src_a': src_a,
            'src_b': src_b,
            'dst': dst,
            'flags': flags,
            'control': control,  # [branch, jump, stack]
            'bytes': bytes_logits,
        }

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# FULL 6502 CPU
# =============================================================================

class MOS6502(nn.Module):
    """
    Complete MOS 6502 CPU.

    32 frozen shapes + learned decoder.
    All computation is geometry. Only routing is learned.
    """

    def __init__(self, d_hidden: int = 64):
        super().__init__()

        self.decoder = InstructionDecoder(d_hidden)

        # Register ALU shape functions (frozen)
        self.alu_shapes = [
            FrozenShapes.ripple_add,
            FrozenShapes.ripple_sub,
            FrozenShapes.parallel_and,
            FrozenShapes.parallel_or,
            FrozenShapes.parallel_xor,
            FrozenShapes.shift_left,
            FrozenShapes.shift_right,
            FrozenShapes.rotate_left,
            FrozenShapes.rotate_right,
            FrozenShapes.increment,
            FrozenShapes.decrement,
            FrozenShapes.transfer,
            FrozenShapes.load,
            FrozenShapes.store,
            FrozenShapes.bit_test,
            FrozenShapes.identity,
        ]

        # Register addressing mode functions (frozen)
        self.addr_modes = [
            AddressingModes.immediate,
            AddressingModes.zero_page,
            AddressingModes.zero_page_x,
            AddressingModes.zero_page_y,
            AddressingModes.absolute,
            AddressingModes.absolute_x,
            AddressingModes.absolute_y,
            AddressingModes.indirect,
            AddressingModes.indexed_indirect,
            AddressingModes.indirect_indexed,
        ]

    def execute_alu(
        self,
        a: Tensor,
        b: Tensor,
        c: Tensor,
        alu_weights: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Execute weighted sum of ALU operations."""
        results = []
        carries = []

        for shape_fn in self.alu_shapes:
            r, car = shape_fn(a, b, c)
            results.append(r)
            carries.append(car)

        results = torch.stack(results, dim=-1)  # [B, 8, 16]
        carries = torch.stack(carries, dim=-1)  # [B, 16]

        # Weighted sum
        result = (results * alu_weights.unsqueeze(1)).sum(dim=-1)
        carry = (carries * alu_weights).sum(dim=-1)

        return result, carry

    def forward(
        self,
        opcode: Tensor,
        operand: Tensor,
        state: CPUState,
        memory: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """
        Execute one instruction.

        Args:
            opcode: Instruction opcode [B]
            operand: Instruction operand [B, 16] (up to 2 bytes)
            state: Current CPU state
            memory: Optional memory tensor for loads/stores

        Returns:
            result: ALU result [B, 8]
            new_state: Updated CPU state
            info: Routing and debug info
        """
        B = opcode.shape[0]

        # Decode instruction
        decoded = self.decoder(opcode)

        # Get source values based on routing
        # src encoding: 0=A, 1=X, 2=Y, 3=M, 4=IMM
        sources = torch.stack([state.A, state.X, state.Y, operand[:, :8], operand[:, :8]], dim=-1)

        src_a = (sources * decoded['src_a'].unsqueeze(1)).sum(dim=-1)
        src_b = (sources * decoded['src_b'].unsqueeze(1)).sum(dim=-1)

        # Execute ALU
        result, carry = self.execute_alu(src_a, src_b, state.C, decoded['alu'])

        # Compute flags
        n, z = FlagCompute.compute_nz(result)
        v = FlagCompute.compute_v(src_a, src_b, result)

        # Apply flags based on mask
        new_p = FlagCompute.apply_flags(state.P, n=n, v=v, z=z, c=carry)
        # Mask out flags that shouldn't be updated
        new_p = state.P * (1 - decoded['flags']) + new_p * decoded['flags']

        info = {
            'decoded': decoded,
            'shape_idx': decoded['alu'].argmax(dim=-1, keepdim=True),
            'shape_weights': decoded['alu'].unsqueeze(1),
            'partner_idx': torch.zeros(B, 1, dtype=torch.long, device=opcode.device),
            'distances': torch.zeros(B, 1, 1, device=opcode.device),
        }

        return result, carry, info

    @property
    def num_params(self) -> int:
        return self.decoder.num_params


# =============================================================================
# TRAINING
# =============================================================================

def train_6502_decoder(
    opcodes: Tensor,
    operands: Tensor,
    states: CPUState,
    expected_results: Tensor,
    expected_flags: Tensor,
    max_steps: int = 200,
    lr: float = 0.05,
    doula: Optional[DoulaFunction] = None,
) -> Tuple[MOS6502, Dict]:
    """
    Train the 6502 instruction decoder.

    Given (opcode, operand, state) → expected (result, flags),
    learn the routing that produces correct outputs.
    """
    cpu = MOS6502()
    optimizer = torch.optim.Adam(cpu.parameters(), lr=lr)

    start_time = time.time()
    history = []

    for step in range(max_steps):
        result, carry, info = cpu(opcodes, operands, states)

        # Loss on result bits
        result_loss = F.mse_loss(result, expected_results)

        # Loss on flags (if provided)
        if expected_flags is not None:
            flag_loss = F.mse_loss(info['decoded']['flags'], expected_flags)
        else:
            flag_loss = 0

        loss = result_loss + 0.1 * flag_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        with torch.no_grad():
            acc = (result.round() == expected_results).all(dim=1).float().mean().item()

        history.append({
            'step': step,
            'loss': loss.item(),
            'accuracy': acc,
        })

        # Doula observation
        if doula:
            sig = doula.aggregate(info, step, 0, loss.item())
            doula.store(sig, loss.item(), acc)

        # Early stopping
        if acc >= 1.0 and loss.item() < 1e-6:
            break

    elapsed_ms = (time.time() - start_time) * 1000

    return cpu, {
        'steps': step + 1,
        'final_loss': loss.item(),
        'final_accuracy': acc,
        'time_ms': elapsed_ms,
        'history': history,
        'num_params': cpu.num_params,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("MOS 6502 CPU - Frozen Shapes Architecture")
    print("=" * 50)

    cpu = MOS6502()
    print(f"Total parameters: {cpu.num_params}")
    print(f"ALU shapes: 16 (0 params)")
    print(f"Addressing modes: 10 (0 params)")
    print(f"Decoder: {cpu.num_params} params")

    # Quick test
    B = 4
    state = CPUState.reset(B)
    opcode = torch.randint(0, 256, (B,))
    operand = torch.zeros(B, 16)

    result, carry, info = cpu(opcode, operand, state)
    print(f"\nTest execution:")
    print(f"  Opcodes: {opcode.tolist()}")
    print(f"  Result shape: {result.shape}")
    print(f"  Routing to shapes: {info['shape_idx'].squeeze().tolist()}")
