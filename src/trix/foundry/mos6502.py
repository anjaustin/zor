"""
MOS 6502 - 8-bit Neural ALU via FrozenFoundry.

The MOS 6502 (1975) was one of the most influential CPUs in computing history:
- Used in Apple II, Commodore 64, NES, Atari 2600
- 8-bit data bus
- 3,510 transistors
- Simple but elegant instruction set

Let's build it with FrozenFoundry.
"""

from trix.foundry.frozen_foundry import FrozenFoundry
import time


def build_6502():
    """Build a MOS 6502 ALU using FrozenFoundry."""

    print("=" * 70)
    print("MOS 6502 - 8-bit Neural ALU")
    print("=" * 70)
    print()

    # Create 8-bit foundry
    foundry = FrozenFoundry(bit_width=8)

    # =========================================================================
    # ARITHMETIC OPERATIONS
    # =========================================================================

    # ADC - Add with Carry
    foundry.register("ADC", lambda a, b, c: (
        (a + b + c) & 0xFF,
        int((a + b + c) > 255)
    ))

    # SBC - Subtract with Carry (6502 uses inverted borrow)
    foundry.register("SBC", lambda a, b, c: (
        (a + (b ^ 0xFF) + c) & 0xFF,
        int((a + (b ^ 0xFF) + c) > 255)
    ))

    # =========================================================================
    # LOGIC OPERATIONS
    # =========================================================================

    # AND - Logical AND
    foundry.register("AND", lambda a, b, c: (a & b, 0))

    # ORA - Logical OR
    foundry.register("ORA", lambda a, b, c: (a | b, 0))

    # EOR - Exclusive OR
    foundry.register("EOR", lambda a, b, c: (a ^ b, 0))

    # =========================================================================
    # SHIFT OPERATIONS
    # =========================================================================

    # ASL - Arithmetic Shift Left
    foundry.register("ASL", lambda a, b, c: (
        (a << 1) & 0xFF,
        (a >> 7) & 1
    ))

    # LSR - Logical Shift Right
    foundry.register("LSR", lambda a, b, c: (
        a >> 1,
        a & 1
    ))

    # ROL - Rotate Left through Carry
    foundry.register("ROL", lambda a, b, c: (
        ((a << 1) | c) & 0xFF,
        (a >> 7) & 1
    ))

    # ROR - Rotate Right through Carry
    foundry.register("ROR", lambda a, b, c: (
        (a >> 1) | (c << 7),
        a & 1
    ))

    # =========================================================================
    # INCREMENT/DECREMENT
    # =========================================================================

    # INC - Increment
    foundry.register("INC", lambda a, b, c: ((a + 1) & 0xFF, 0))

    # DEC - Decrement
    foundry.register("DEC", lambda a, b, c: ((a - 1) & 0xFF, 0))

    # INX/INY
    foundry.register("INX", lambda a, b, c: ((a + 1) & 0xFF, 0))
    foundry.register("INY", lambda a, b, c: ((a + 1) & 0xFF, 0))

    # DEX/DEY
    foundry.register("DEX", lambda a, b, c: ((a - 1) & 0xFF, 0))
    foundry.register("DEY", lambda a, b, c: ((a - 1) & 0xFF, 0))

    # =========================================================================
    # TRANSFER OPERATIONS
    # =========================================================================

    foundry.register("TAX", lambda a, b, c: (a, 0))
    foundry.register("TXA", lambda a, b, c: (a, 0))
    foundry.register("TAY", lambda a, b, c: (a, 0))
    foundry.register("TYA", lambda a, b, c: (a, 0))
    foundry.register("TSX", lambda a, b, c: (a, 0))
    foundry.register("TXS", lambda a, b, c: (a, 0))

    # =========================================================================
    # LOAD/STORE
    # =========================================================================

    foundry.register("LDA", lambda a, b, c: (a, 0))
    foundry.register("LDX", lambda a, b, c: (a, 0))
    foundry.register("LDY", lambda a, b, c: (a, 0))
    foundry.register("STA", lambda a, b, c: (a, 0))
    foundry.register("STX", lambda a, b, c: (a, 0))
    foundry.register("STY", lambda a, b, c: (a, 0))

    # =========================================================================
    # COMPARE
    # =========================================================================

    foundry.register("CMP", lambda a, b, c: (
        (a - b) & 0xFF,
        int(a >= b)
    ))

    foundry.register("CPX", lambda a, b, c: (
        (a - b) & 0xFF,
        int(a >= b)
    ))

    foundry.register("CPY", lambda a, b, c: (
        (a - b) & 0xFF,
        int(a >= b)
    ))

    # =========================================================================
    # BUILD
    # =========================================================================

    print(f"Registered {len(foundry.operations)} operations")
    print()

    start = time.time()
    result = foundry.build(verbose=True, n_samples_per_op=1000)
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
    print("MOS 6502 SUMMARY")
    print("=" * 70)
    print(f"  Architecture:     8-bit")
    print(f"  Year:             1975")
    print(f"  Transistors:      3,510")
    print(f"  Operations:       {len(foundry.operations)}")
    print(f"  Frozen shapes:    {len(foundry.library)}")
    print(f"  Routing params:   {result.num_params}")
    print(f"  Training steps:   {result.training_steps}")
    print(f"  Training time:    {result.training_time_ms:.1f} ms")
    print(f"  Final accuracy:   {result.accuracy * 100:.2f}%")
    print(f"  Validation:       {val_acc * 100:.2f}%")
    print()

    if val_acc >= 0.9999:
        print("  THE MOS 6502 IS NOW A NEURAL NETWORK.")
    else:
        unmatched = [op.name for op in foundry.operations.values()
                     if op.match_accuracy < 1.0]
        print(f"  Unmatched ops: {unmatched}")

    print()
    print("=" * 70)

    return foundry, result


if __name__ == "__main__":
    foundry, result = build_6502()
