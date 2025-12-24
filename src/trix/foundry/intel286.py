"""
Intel 80286 - 16-bit Neural ALU via FrozenFoundry.

The 80286 (1982) was Intel's second-generation x86 processor:
- 16-bit data bus
- 24-bit address bus
- 134,000 transistors
- No FPU (separate 80287)
- Protected mode

Let's see how close we are.
"""

import torch
from trix.foundry.frozen_foundry import FrozenFoundry
import time


def build_286():
    """Build an Intel 286 ALU using FrozenFoundry."""

    print("=" * 70)
    print("Intel 80286 - 16-bit Neural ALU")
    print("=" * 70)
    print()

    # Create 16-bit foundry
    foundry = FrozenFoundry(bit_width=16)

    # =========================================================================
    # CUSTOM SHAPES FOR x86
    # =========================================================================

    from trix.foundry.frozen_foundry import PureMath

    # Byte swap (XCHG AL, AH equivalent)
    def byte_swap(a, b, c):
        low = a[:, :8]
        high = a[:, 8:]
        result = torch.cat([high, low], dim=1)
        return result, torch.zeros(a.shape[0], device=a.device)

    foundry.register_shape("byte_swap", byte_swap, n_inputs=1)

    # Sign extend (CBW: byte to word)
    def sign_extend_8_16(a, b, c):
        sign = a[:, 7:8]
        high_byte = sign.expand(-1, 8)
        result = torch.cat([a[:, :8], high_byte], dim=1)
        return result, torch.zeros(a.shape[0], device=a.device)

    foundry.register_shape("sign_extend", sign_extend_8_16, n_inputs=1)

    # NEG - Two's complement negate: -a = ~a + 1
    def negate(a, b, c):
        # NOT then add 1
        inverted = PureMath.not_op(a)
        one = torch.zeros_like(a)
        one[:, 0] = 1.0
        carry = torch.zeros(a.shape[0], device=a.device)
        # Ripple add
        result = []
        for i in range(16):
            s, carry = PureMath.full_adder(inverted[:, i], one[:, i], carry)
            result.append(s)
        result = torch.stack(result, dim=1)
        # Carry flag is set if a != 0
        cf = 1 - PureMath.not_op(a.sum(dim=1).clamp(0, 1))
        return result, cf

    foundry.register_shape("negate", negate, n_inputs=1, produces_carry=True)

    # ADD without carry (x86 ADD doesn't use carry input)
    def add_no_carry(a, b, c):
        result = []
        carry = torch.zeros(a.shape[0], device=a.device)
        for i in range(16):
            s, carry = PureMath.full_adder(a[:, i], b[:, i], carry)
            result.append(s)
        return torch.stack(result, dim=1), carry

    foundry.register_shape("add_no_carry", add_no_carry, n_inputs=2, produces_carry=True)

    # SAR - Arithmetic Shift Right (preserves sign)
    def shift_right_arith(a, b, c):
        sign = a[:, 15:16]  # Keep sign bit
        shifted = a[:, 1:]  # Shift right
        result = torch.cat([shifted, sign], dim=1)
        return result, a[:, 0]

    foundry.register_shape("shift_right_arith", shift_right_arith, n_inputs=1, produces_carry=True)

    # ROL - Rotate Left (no carry in, different from RCL)
    # bit 15 -> bit 0, bit 0 -> bit 1, ..., bit 14 -> bit 15
    # CF = old bit 15
    def rotate_left_no_carry(a, b, c):
        # a[:, 0] is bit 0 (LSB), a[:, 15] is bit 15 (MSB)
        # After ROL: new bit 0 = old bit 15, new bit 1 = old bit 0, etc.
        high_bit = a[:, 15:16]
        lower_bits = a[:, :15]
        result = torch.cat([high_bit, lower_bits], dim=1)
        return result, a[:, 15]

    foundry.register_shape("rotate_left_no_carry", rotate_left_no_carry, n_inputs=1, produces_carry=True)

    # ROR - Rotate Right (no carry in, different from RCR)
    # bit 0 -> bit 15, bit 1 -> bit 0, ..., bit 15 -> bit 14
    # CF = old bit 0
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

    # Sign extend to all 1s or all 0s (for CWD)
    def sign_to_word(a, b, c):
        sign = a[:, 15:16]
        result = sign.expand(-1, 16)
        return result, torch.zeros(a.shape[0], device=a.device)

    foundry.register_shape("sign_to_word", sign_to_word, n_inputs=1)

    # =========================================================================
    # ARITHMETIC (x86)
    # =========================================================================

    # ADD - Add
    foundry.register("ADD", lambda a, b, c: (
        (a + b) & 0xFFFF,
        int((a + b) > 0xFFFF)
    ))

    # ADC - Add with Carry
    foundry.register("ADC", lambda a, b, c: (
        (a + b + c) & 0xFFFF,
        int((a + b + c) > 0xFFFF)
    ))

    # SUB - Subtract
    foundry.register("SUB", lambda a, b, c: (
        (a - b) & 0xFFFF,
        int(a >= b)
    ))

    # SBB - Subtract with Borrow
    foundry.register("SBB", lambda a, b, c: (
        (a - b - (1 - c)) & 0xFFFF,
        int(a >= b + (1 - c))
    ))

    # INC - Increment
    foundry.register("INC", lambda a, b, c: ((a + 1) & 0xFFFF, c))  # Preserves CF

    # DEC - Decrement
    foundry.register("DEC", lambda a, b, c: ((a - 1) & 0xFFFF, c))  # Preserves CF

    # NEG - Two's complement negate
    foundry.register("NEG", lambda a, b, c: ((-a) & 0xFFFF, int(a != 0)))

    # CMP - Compare (subtract without storing)
    foundry.register("CMP", lambda a, b, c: (
        (a - b) & 0xFFFF,
        int(a >= b)
    ))

    # =========================================================================
    # LOGIC (x86)
    # =========================================================================

    # AND
    foundry.register("AND", lambda a, b, c: (a & b, 0))

    # OR
    foundry.register("OR", lambda a, b, c: (a | b, 0))

    # XOR
    foundry.register("XOR", lambda a, b, c: (a ^ b, 0))

    # NOT - One's complement
    foundry.register("NOT", lambda a, b, c: (a ^ 0xFFFF, c))  # Preserves flags

    # TEST - AND without storing
    foundry.register("TEST", lambda a, b, c: (a & b, 0))

    # =========================================================================
    # SHIFTS (x86)
    # =========================================================================

    # SHL/SAL - Shift Left
    foundry.register("SHL", lambda a, b, c: (
        (a << 1) & 0xFFFF,
        (a >> 15) & 1
    ))

    # SHR - Shift Right (logical)
    foundry.register("SHR", lambda a, b, c: (
        a >> 1,
        a & 1
    ))

    # SAR - Shift Right (arithmetic, preserves sign)
    foundry.register("SAR", lambda a, b, c: (
        (a >> 1) | (a & 0x8000),  # Keep sign bit
        a & 1
    ))

    # ROL - Rotate Left (CF = old bit 15)
    foundry.register("ROL", lambda a, b, c: (
        ((a << 1) | (a >> 15)) & 0xFFFF,
        (a >> 15) & 1
    ))

    # ROR - Rotate Right (CF = old bit 0)
    foundry.register("ROR", lambda a, b, c: (
        ((a >> 1) | (a << 15)) & 0xFFFF,
        a & 1
    ))

    # RCL - Rotate through Carry Left
    foundry.register("RCL", lambda a, b, c: (
        ((a << 1) | c) & 0xFFFF,
        (a >> 15) & 1
    ))

    # RCR - Rotate through Carry Right
    foundry.register("RCR", lambda a, b, c: (
        (a >> 1) | (c << 15),
        a & 1
    ))

    # =========================================================================
    # DATA MOVEMENT (x86)
    # =========================================================================

    # MOV - Move (just identity for ALU purposes)
    foundry.register("MOV", lambda a, b, c: (a, c))

    # XCHG - Exchange (for single operand, it's identity)
    foundry.register("XCHG", lambda a, b, c: (b, c))  # Returns b

    # CBW - Convert Byte to Word (sign extend low byte to word)
    # Takes AL (low 8 bits), sign extends bit 7 to AH (high 8 bits)
    foundry.register("CBW", lambda a, b, c: (
        (a & 0xFF) | (0xFF00 if (a & 0x80) else 0),
        c
    ))

    # CWD - Convert Word to Doubleword (sign extend to DX:AX)
    # For 16-bit, this just returns the sign extension
    foundry.register("CWD", lambda a, b, c: (
        0xFFFF if (a & 0x8000) else 0,
        c
    ))

    # =========================================================================
    # FLAGS (x86)
    # =========================================================================

    # CLC - Clear Carry
    foundry.register("CLC", lambda a, b, c: (a, 0))

    # STC - Set Carry
    foundry.register("STC", lambda a, b, c: (a, 1))

    # CMC - Complement Carry
    foundry.register("CMC", lambda a, b, c: (a, 1 - c))

    # =========================================================================
    # STACK-LIKE (for completeness)
    # =========================================================================

    # PUSH/POP are memory operations, but for ALU we model as identity
    foundry.register("PUSH", lambda a, b, c: (a, c))
    foundry.register("POP", lambda a, b, c: (a, c))

    # =========================================================================
    # 286-SPECIFIC
    # =========================================================================

    # BOUND - Check array bounds (compare operation)
    foundry.register("BOUND", lambda a, b, c: (a, int(a >= 0 and a <= b)))

    # ENTER - Make stack frame (for ALU, just compute)
    foundry.register("ENTER", lambda a, b, c: ((a - b) & 0xFFFF, c))

    # LEAVE - Destroy stack frame
    foundry.register("LEAVE", lambda a, b, c: (a, c))

    # =========================================================================
    # BUILD
    # =========================================================================

    print(f"Registered {len(foundry.operations)} x86 operations")
    print()

    start = time.time()
    result = foundry.build(verbose=True, n_samples_per_op=500)
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
    print("INTEL 80286 SUMMARY")
    print("=" * 70)
    print(f"  Architecture:     x86 (16-bit)")
    print(f"  Year:             1982")
    print(f"  Operations:       {len(foundry.operations)}")
    print(f"  Frozen shapes:    {len(foundry.library)}")
    print(f"  Routing params:   {result.num_params}")
    print(f"  Training steps:   {result.training_steps}")
    print(f"  Training time:    {result.training_time_ms:.1f} ms")
    print(f"  Final accuracy:   {result.accuracy * 100:.2f}%")
    print(f"  Validation:       {val_acc * 100:.2f}%")
    print()

    if val_acc >= 0.9999:
        print("  THE INTEL 286 IS NOW A NEURAL NETWORK.")
    else:
        unmatched = [op.name for op in foundry.operations.values()
                     if op.match_accuracy < 1.0]
        print(f"  Unmatched ops: {unmatched}")

    print()
    print("=" * 70)

    return foundry, result


if __name__ == "__main__":
    foundry, result = build_286()
