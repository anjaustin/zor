"""
Intel 80486 - 32-bit Neural ALU via FrozenFoundry.

The 80486 (1989) added to the 386:
- On-chip FPU (x87)
- On-chip L1 cache (8KB)
- Pipelined execution
- XADD, CMPXCHG instructions

For the integer ALU, it's the same as 386 + two new instructions.
1.2 million transistors. Let's build it.
"""

import torch
from trix.foundry.frozen_foundry import FrozenFoundry, PureMath
import time


def build_486():
    """Build an Intel 486 ALU using FrozenFoundry."""

    print("=" * 70)
    print("Intel 80486 - 32-bit Neural ALU")
    print("=" * 70)
    print()

    # Create 32-bit foundry
    foundry = FrozenFoundry(bit_width=32)

    # =========================================================================
    # CUSTOM SHAPES FOR x86-32
    # =========================================================================

    # NEG - Two's complement negate: -a = ~a + 1
    def negate(a, b, c):
        inverted = PureMath.not_op(a)
        one = torch.zeros_like(a)
        one[:, 0] = 1.0
        carry = torch.zeros(a.shape[0], device=a.device)
        result = []
        for i in range(32):
            s, carry = PureMath.full_adder(inverted[:, i], one[:, i], carry)
            result.append(s)
        result = torch.stack(result, dim=1)
        cf = 1 - PureMath.not_op(a.sum(dim=1).clamp(0, 1))
        return result, cf

    foundry.register_shape("negate", negate, n_inputs=1, produces_carry=True)

    # ADD without carry
    def add_no_carry(a, b, c):
        result = []
        carry = torch.zeros(a.shape[0], device=a.device)
        for i in range(32):
            s, carry = PureMath.full_adder(a[:, i], b[:, i], carry)
            result.append(s)
        return torch.stack(result, dim=1), carry

    foundry.register_shape("add_no_carry", add_no_carry, n_inputs=2, produces_carry=True)

    # SAR - Arithmetic Shift Right (preserves sign)
    def shift_right_arith(a, b, c):
        sign = a[:, 31:32]
        shifted = a[:, 1:]
        result = torch.cat([shifted, sign], dim=1)
        return result, a[:, 0]

    foundry.register_shape("shift_right_arith", shift_right_arith, n_inputs=1, produces_carry=True)

    # ROL - Rotate Left (no carry in)
    def rotate_left_no_carry(a, b, c):
        high_bit = a[:, 31:32]
        lower_bits = a[:, :31]
        result = torch.cat([high_bit, lower_bits], dim=1)
        return result, a[:, 31]

    foundry.register_shape("rotate_left_no_carry", rotate_left_no_carry, n_inputs=1, produces_carry=True)

    # ROR - Rotate Right (no carry in)
    def rotate_right_no_carry(a, b, c):
        low_bit = a[:, 0:1]
        upper_bits = a[:, 1:]
        result = torch.cat([upper_bits, low_bit], dim=1)
        return result, a[:, 0]

    foundry.register_shape("rotate_right_no_carry", rotate_right_no_carry, n_inputs=1, produces_carry=True)

    # Return second operand (for XCHG)
    def return_second(a, b, c):
        return b.clone(), torch.zeros(a.shape[0], device=a.device)

    foundry.register_shape("return_second", return_second, n_inputs=2)

    # MOVSX - Sign extend byte to dword
    def sign_extend_8_32(a, b, c):
        sign = a[:, 7:8]
        high_bits = sign.expand(-1, 24)
        result = torch.cat([a[:, :8], high_bits], dim=1)
        return result, torch.zeros(a.shape[0], device=a.device)

    foundry.register_shape("sign_extend_8_32", sign_extend_8_32, n_inputs=1)

    # MOVSX - Sign extend word to dword
    def sign_extend_16_32(a, b, c):
        sign = a[:, 15:16]
        high_bits = sign.expand(-1, 16)
        result = torch.cat([a[:, :16], high_bits], dim=1)
        return result, torch.zeros(a.shape[0], device=a.device)

    foundry.register_shape("sign_extend_16_32", sign_extend_16_32, n_inputs=1)

    # CDQ - Sign extend EAX to EDX:EAX (returns sign extension)
    def sign_to_dword(a, b, c):
        sign = a[:, 31:32]
        result = sign.expand(-1, 32)
        return result, torch.zeros(a.shape[0], device=a.device)

    foundry.register_shape("sign_to_dword", sign_to_dword, n_inputs=1)

    # BSWAP - Byte swap (386 specific)
    def bswap(a, b, c):
        b0 = a[:, 0:8]
        b1 = a[:, 8:16]
        b2 = a[:, 16:24]
        b3 = a[:, 24:32]
        result = torch.cat([b3, b2, b1, b0], dim=1)
        return result, torch.zeros(a.shape[0], device=a.device)

    foundry.register_shape("bswap", bswap, n_inputs=1)

    # Zero extend byte to dword
    def zero_extend_8_32(a, b, c):
        zeros = torch.zeros(a.shape[0], 24, device=a.device)
        result = torch.cat([a[:, :8], zeros], dim=1)
        return result, torch.zeros(a.shape[0], device=a.device)

    foundry.register_shape("zero_extend_8_32", zero_extend_8_32, n_inputs=1)

    # Zero extend word to dword
    def zero_extend_16_32(a, b, c):
        zeros = torch.zeros(a.shape[0], 16, device=a.device)
        result = torch.cat([a[:, :16], zeros], dim=1)
        return result, torch.zeros(a.shape[0], device=a.device)

    foundry.register_shape("zero_extend_16_32", zero_extend_16_32, n_inputs=1)

    # Conditional select: if c then b else a (MUX)
    def conditional_select(a, b, c):
        c_expanded = c.unsqueeze(-1).expand(-1, 32)
        result = a * (1 - c_expanded) + b * c_expanded
        return result, torch.zeros(a.shape[0], device=a.device)

    foundry.register_shape("conditional_select", conditional_select, n_inputs=2)

    # =========================================================================
    # ARITHMETIC (x86-32)
    # =========================================================================

    MAX32 = 0xFFFFFFFF

    # ADD
    foundry.register("ADD", lambda a, b, c: (
        (a + b) & MAX32,
        int((a + b) > MAX32)
    ))

    # ADC
    foundry.register("ADC", lambda a, b, c: (
        (a + b + c) & MAX32,
        int((a + b + c) > MAX32)
    ))

    # SUB
    foundry.register("SUB", lambda a, b, c: (
        (a - b) & MAX32,
        int(a >= b)
    ))

    # SBB
    foundry.register("SBB", lambda a, b, c: (
        (a - b - (1 - c)) & MAX32,
        int(a >= b + (1 - c))
    ))

    # INC
    foundry.register("INC", lambda a, b, c: ((a + 1) & MAX32, c))

    # DEC
    foundry.register("DEC", lambda a, b, c: ((a - 1) & MAX32, c))

    # NEG
    foundry.register("NEG", lambda a, b, c: ((-a) & MAX32, int(a != 0)))

    # CMP
    foundry.register("CMP", lambda a, b, c: ((a - b) & MAX32, int(a >= b)))

    # =========================================================================
    # LOGIC (x86-32)
    # =========================================================================

    foundry.register("AND", lambda a, b, c: (a & b, 0))
    foundry.register("OR", lambda a, b, c: (a | b, 0))
    foundry.register("XOR", lambda a, b, c: (a ^ b, 0))
    foundry.register("NOT", lambda a, b, c: (a ^ MAX32, c))
    foundry.register("TEST", lambda a, b, c: (a & b, 0))

    # =========================================================================
    # SHIFTS (x86-32)
    # =========================================================================

    foundry.register("SHL", lambda a, b, c: ((a << 1) & MAX32, (a >> 31) & 1))
    foundry.register("SHR", lambda a, b, c: (a >> 1, a & 1))
    foundry.register("SAR", lambda a, b, c: ((a >> 1) | (a & 0x80000000), a & 1))

    foundry.register("ROL", lambda a, b, c: (((a << 1) | (a >> 31)) & MAX32, (a >> 31) & 1))
    foundry.register("ROR", lambda a, b, c: (((a >> 1) | (a << 31)) & MAX32, a & 1))

    foundry.register("RCL", lambda a, b, c: (((a << 1) | c) & MAX32, (a >> 31) & 1))
    foundry.register("RCR", lambda a, b, c: ((a >> 1) | (c << 31), a & 1))

    # =========================================================================
    # DATA MOVEMENT (x86-32)
    # =========================================================================

    foundry.register("MOV", lambda a, b, c: (a, c))
    foundry.register("XCHG", lambda a, b, c: (b, c))

    # MOVSX byte to dword
    foundry.register("MOVSX8", lambda a, b, c: (
        (a & 0xFF) | (0xFFFFFF00 if (a & 0x80) else 0),
        c
    ))

    # MOVSX word to dword
    foundry.register("MOVSX16", lambda a, b, c: (
        (a & 0xFFFF) | (0xFFFF0000 if (a & 0x8000) else 0),
        c
    ))

    # MOVZX byte to dword
    foundry.register("MOVZX8", lambda a, b, c: (a & 0xFF, c))

    # MOVZX word to dword
    foundry.register("MOVZX16", lambda a, b, c: (a & 0xFFFF, c))

    # CDQ - Convert dword to qword (sign extend EAX to EDX:EAX)
    foundry.register("CDQ", lambda a, b, c: (
        MAX32 if (a & 0x80000000) else 0,
        c
    ))

    # BSWAP - Byte swap (386+)
    foundry.register("BSWAP", lambda a, b, c: (
        ((a & 0xFF) << 24) |
        ((a & 0xFF00) << 8) |
        ((a & 0xFF0000) >> 8) |
        ((a >> 24) & 0xFF),
        c
    ))

    # =========================================================================
    # FLAGS (x86-32)
    # =========================================================================

    foundry.register("CLC", lambda a, b, c: (a, 0))
    foundry.register("STC", lambda a, b, c: (a, 1))
    foundry.register("CMC", lambda a, b, c: (a, 1 - c))

    # =========================================================================
    # 386/486 SPECIFIC
    # =========================================================================

    # BT - Bit Test
    foundry.register("BT", lambda a, b, c: (a, a & 1))

    # XADD - Exchange and Add (486+)
    # Exchanges a and b, then adds them, stores in a
    # Returns: (a + b, old_a) - we return the sum
    foundry.register("XADD", lambda a, b, c: ((a + b) & MAX32, int((a + b) > MAX32)))

    # CMPXCHG - Compare and Exchange (486+)
    # If c (comparison flag) is set, return b; else return a
    # c acts as the "equal" flag from a prior comparison
    foundry.register("CMPXCHG", lambda a, b, c: (b if c == 1 else a, 0))

    # =========================================================================
    # BUILD
    # =========================================================================

    print(f"Registered {len(foundry.operations)} x86-32 operations")
    print()

    start = time.time()
    result = foundry.build(verbose=True, n_samples_per_op=300)
    elapsed = time.time() - start

    # =========================================================================
    # VALIDATE
    # =========================================================================

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    val_acc = foundry.validate(n_samples=50000)
    print(f"Validation accuracy: {val_acc * 100:.4f}%")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print("=" * 70)
    print("INTEL 80486 SUMMARY")
    print("=" * 70)
    print(f"  Architecture:     x86 (32-bit)")
    print(f"  Year:             1989")
    print(f"  Transistors:      1,200,000")
    print(f"  Operations:       {len(foundry.operations)}")
    print(f"  Frozen shapes:    {len(foundry.library)}")
    print(f"  Routing params:   {result.num_params}")
    print(f"  Training steps:   {result.training_steps}")
    print(f"  Training time:    {result.training_time_ms:.1f} ms")
    print(f"  Final accuracy:   {result.accuracy * 100:.2f}%")
    print(f"  Validation:       {val_acc * 100:.2f}%")
    print()

    if val_acc >= 0.9999:
        print("  THE INTEL 486 IS NOW A NEURAL NETWORK.")
    else:
        unmatched = [op.name for op in foundry.operations.values()
                     if op.match_accuracy < 1.0]
        print(f"  Unmatched ops: {unmatched}")

    print()
    print("=" * 70)

    return foundry, result


if __name__ == "__main__":
    foundry, result = build_486()
