"""
TriX Native Frozen Shapes

Mathematical shapes for exact computation. 0 learnable parameters.

The four axioms:
    XOR(a, b) = a + b - 2ab  (the saddle)
    AND(a, b) = ab           (the product)
    OR(a, b)  = a + b - ab   (the union)
    NOT(a)    = 1 - a        (the reflection)

These are mathematical truths - not approximations.
100% accurate on binary inputs {0, 1}.
Differentiable for gradient flow during training.

"Computation is geometry. Learning is routing."
"""

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    import numpy as cp
    HAS_CUPY = False


# =============================================================================
# FROZEN SHAPES - The Geometry (0 learnable parameters)
# =============================================================================

class FrozenShapes:
    """
    Frozen mathematical shapes for exact binary computation.

    All operations are exact on binary inputs {0, 1}.
    Polynomial form allows gradients to flow during training.

    Example:
        >>> a = cp.array([0, 0, 1, 1])
        >>> b = cp.array([0, 1, 0, 1])
        >>> FrozenShapes.xor(a, b)  # [0, 1, 1, 0]
    """

    @staticmethod
    def xor(a, b):
        """
        XOR: The primitive of difference.

        XOR(a, b) = a + b - 2ab

        Truth table:
            0 XOR 0 = 0
            0 XOR 1 = 1
            1 XOR 0 = 1
            1 XOR 1 = 0
        """
        return a + b - 2 * a * b

    @staticmethod
    def and_op(a, b):
        """
        AND: The product.

        AND(a, b) = ab

        Truth table:
            0 AND 0 = 0
            0 AND 1 = 0
            1 AND 0 = 0
            1 AND 1 = 1
        """
        return a * b

    @staticmethod
    def or_op(a, b):
        """
        OR: The union.

        OR(a, b) = a + b - ab

        Truth table:
            0 OR 0 = 0
            0 OR 1 = 1
            1 OR 0 = 1
            1 OR 1 = 1
        """
        return a + b - a * b

    @staticmethod
    def not_op(a):
        """
        NOT: The reflection.

        NOT(a) = 1 - a

        Truth table:
            NOT 0 = 1
            NOT 1 = 0
        """
        return 1 - a

    @staticmethod
    def nand(a, b):
        """NAND: NOT(AND(a, b)) = 1 - ab"""
        return 1 - a * b

    @staticmethod
    def nor(a, b):
        """NOR: NOT(OR(a, b)) = 1 - a - b + ab"""
        return 1 - a - b + a * b

    @staticmethod
    def xnor(a, b):
        """XNOR: NOT(XOR(a, b)) = 1 - a - b + 2ab"""
        return 1 - a - b + 2 * a * b

    @staticmethod
    def half_adder(a, b):
        """
        Half adder: (a, b) -> (sum, carry)

        sum = a XOR b
        carry = a AND b
        """
        sum_bit = FrozenShapes.xor(a, b)
        carry = FrozenShapes.and_op(a, b)
        return sum_bit, carry

    @staticmethod
    def full_adder(a, b, c):
        """
        Full adder: (a, b, c_in) -> (sum, c_out)

        sum = a XOR b XOR c
        c_out = (a AND b) OR (c AND (a XOR b))
        """
        p = FrozenShapes.xor(a, b)
        sum_bit = FrozenShapes.xor(p, c)
        c_out = FrozenShapes.or_op(
            FrozenShapes.and_op(a, b),
            FrozenShapes.and_op(c, p)
        )
        return sum_bit, c_out


# =============================================================================
# FROZEN ALU - 16 shapes for 6502 operations
# =============================================================================

class FrozenALU:
    """
    Frozen 6502 ALU operations.

    16 shapes covering all ALU operations.
    Each is a pure mathematical function.
    0 learnable parameters.

    Shape IDs:
        0: RIPPLE_ADD     (ADC)
        1: RIPPLE_SUB     (SBC)
        2: PARALLEL_AND   (AND)
        3: PARALLEL_OR    (ORA)
        4: PARALLEL_XOR   (EOR)
        5: SHIFT_LEFT     (ASL)
        6: SHIFT_RIGHT    (LSR)
        7: ROTATE_LEFT    (ROL)
        8: ROTATE_RIGHT   (ROR)
        9: INCREMENT      (INC)
        10: DECREMENT     (DEC)
        11: TRANSFER      (TAX/TXA/etc)
        12: LOAD          (LDA/LDX/LDY)
        13: STORE         (STA/STX/STY)
        14: BIT_TEST      (BIT)
        15: IDENTITY      (NOP)
    """

    NUM_SHAPES = 16
    SHAPE_NAMES = [
        'RIPPLE_ADD',    # 0: ADC
        'RIPPLE_SUB',    # 1: SBC
        'PARALLEL_AND',  # 2: AND
        'PARALLEL_OR',   # 3: ORA
        'PARALLEL_XOR',  # 4: EOR
        'SHIFT_LEFT',    # 5: ASL
        'SHIFT_RIGHT',   # 6: LSR
        'ROTATE_LEFT',   # 7: ROL
        'ROTATE_RIGHT',  # 8: ROR
        'INCREMENT',     # 9: INC
        'DECREMENT',     # 10: DEC
        'TRANSFER',      # 11: TAX/TXA/etc
        'LOAD',          # 12: LDA/LDX/LDY
        'STORE',         # 13: STA/STX/STY
        'BIT_TEST',      # 14: BIT
        'IDENTITY',      # 15: NOP
    ]

    @staticmethod
    def ripple_add(a_bits, b_bits, c_in):
        """8-bit addition with carry. Shape 0 (ADC)."""
        result = []
        carry = c_in

        for i in range(8):
            s, carry = FrozenShapes.full_adder(a_bits[:, i], b_bits[:, i], carry)
            result.append(s)

        return cp.stack(result, axis=1), carry

    @staticmethod
    def ripple_sub(a_bits, b_bits, c_in):
        """
        8-bit subtraction. Shape 1 (SBC).

        6502 semantics: A - B - (1 - C) = A + NOT(B) + C
        """
        b_inv = FrozenShapes.not_op(b_bits)
        result, c_out = FrozenALU.ripple_add(a_bits, b_inv, c_in)
        return result, c_out

    @staticmethod
    def parallel_and(a_bits, b_bits):
        """Bitwise AND. Shape 2."""
        return FrozenShapes.and_op(a_bits, b_bits), cp.zeros(a_bits.shape[0])

    @staticmethod
    def parallel_or(a_bits, b_bits):
        """Bitwise OR. Shape 3."""
        return FrozenShapes.or_op(a_bits, b_bits), cp.zeros(a_bits.shape[0])

    @staticmethod
    def parallel_xor(a_bits, b_bits):
        """Bitwise XOR. Shape 4."""
        return FrozenShapes.xor(a_bits, b_bits), cp.zeros(a_bits.shape[0])

    @staticmethod
    def shift_left(a_bits, _unused=None):
        """Arithmetic shift left. Shape 5 (ASL)."""
        carry_out = a_bits[:, 7]  # MSB goes to carry
        result = cp.zeros_like(a_bits)
        result[:, 1:8] = a_bits[:, 0:7]
        result[:, 0] = 0
        return result, carry_out

    @staticmethod
    def shift_right(a_bits, _unused=None):
        """Logical shift right. Shape 6 (LSR)."""
        carry_out = a_bits[:, 0]  # LSB goes to carry
        result = cp.zeros_like(a_bits)
        result[:, 0:7] = a_bits[:, 1:8]
        result[:, 7] = 0
        return result, carry_out

    @staticmethod
    def rotate_left(a_bits, c_in):
        """Rotate left through carry. Shape 7 (ROL)."""
        carry_out = a_bits[:, 7]  # MSB goes to carry
        result = cp.zeros_like(a_bits)
        result[:, 1:8] = a_bits[:, 0:7]
        result[:, 0] = c_in  # Carry comes in at LSB
        return result, carry_out

    @staticmethod
    def rotate_right(a_bits, c_in):
        """Rotate right through carry. Shape 8 (ROR)."""
        carry_out = a_bits[:, 0]  # LSB goes to carry
        result = cp.zeros_like(a_bits)
        result[:, 0:7] = a_bits[:, 1:8]
        result[:, 7] = c_in  # Carry comes in at MSB
        return result, carry_out

    @staticmethod
    def increment(a_bits, _unused=None):
        """Increment by 1. Shape 9 (INC)."""
        one_bits = cp.zeros_like(a_bits)
        one_bits[:, 0] = 1
        result, _ = FrozenALU.ripple_add(a_bits, one_bits, cp.zeros(a_bits.shape[0]))
        return result, cp.zeros(a_bits.shape[0])  # INC doesn't affect carry

    @staticmethod
    def decrement(a_bits, _unused=None):
        """Decrement by 1. Shape 10 (DEC)."""
        ff_bits = cp.ones_like(a_bits)  # All 1s = 0xFF
        result, _ = FrozenALU.ripple_add(a_bits, ff_bits, cp.zeros(a_bits.shape[0]))
        return result, cp.zeros(a_bits.shape[0])  # DEC doesn't affect carry

    @staticmethod
    def transfer(a_bits, _unused=None):
        """Transfer (copy). Shape 11."""
        return a_bits.copy(), cp.zeros(a_bits.shape[0])

    @staticmethod
    def load(a_bits, _unused=None):
        """Load (same as transfer). Shape 12."""
        return a_bits.copy(), cp.zeros(a_bits.shape[0])

    @staticmethod
    def store(a_bits, _unused=None):
        """Store (same as transfer). Shape 13."""
        return a_bits.copy(), cp.zeros(a_bits.shape[0])

    @staticmethod
    def bit_test(a_bits, b_bits):
        """BIT test. Shape 14."""
        return FrozenShapes.and_op(a_bits, b_bits), cp.zeros(a_bits.shape[0])

    @staticmethod
    def identity(a_bits, _unused=None):
        """Identity/NOP. Shape 15."""
        return a_bits.copy(), cp.zeros(a_bits.shape[0])

    @staticmethod
    def execute_shape(shape_id: int, a_bits, b_bits, c_in):
        """Execute a shape by ID."""
        if shape_id == 0:
            return FrozenALU.ripple_add(a_bits, b_bits, c_in)
        elif shape_id == 1:
            return FrozenALU.ripple_sub(a_bits, b_bits, c_in)
        elif shape_id == 2:
            return FrozenALU.parallel_and(a_bits, b_bits)
        elif shape_id == 3:
            return FrozenALU.parallel_or(a_bits, b_bits)
        elif shape_id == 4:
            return FrozenALU.parallel_xor(a_bits, b_bits)
        elif shape_id == 5:
            return FrozenALU.shift_left(a_bits)
        elif shape_id == 6:
            return FrozenALU.shift_right(a_bits)
        elif shape_id == 7:
            return FrozenALU.rotate_left(a_bits, c_in)
        elif shape_id == 8:
            return FrozenALU.rotate_right(a_bits, c_in)
        elif shape_id == 9:
            return FrozenALU.increment(a_bits)
        elif shape_id == 10:
            return FrozenALU.decrement(a_bits)
        elif shape_id == 11:
            return FrozenALU.transfer(a_bits)
        elif shape_id == 12:
            return FrozenALU.load(a_bits)
        elif shape_id == 13:
            return FrozenALU.store(a_bits)
        elif shape_id == 14:
            return FrozenALU.bit_test(a_bits, b_bits)
        else:
            return FrozenALU.identity(a_bits)

    @staticmethod
    def execute_all_shapes(a_bits, b_bits, c_in):
        """Execute all shapes, return stacked results."""
        results = []
        carries = []

        for shape_id in range(FrozenALU.NUM_SHAPES):
            result, carry = FrozenALU.execute_shape(shape_id, a_bits, b_bits, c_in)
            results.append(result)
            carries.append(carry)

        return cp.stack(results, axis=1), cp.stack(carries, axis=1)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def int_to_bits(values, num_bits=8):
    """
    Convert integer values to bit representation (little-endian).

    Args:
        values: Integer array [batch]
        num_bits: Number of bits per value

    Returns:
        Bit array [batch, num_bits]
    """
    batch = values.shape[0]
    bits = cp.zeros((batch, num_bits), dtype=cp.float32)
    for i in range(num_bits):
        bits[:, i] = ((values >> i) & 1).astype(cp.float32)
    return bits


def bits_to_int(bits):
    """
    Convert bit representation to integers (little-endian).

    Args:
        bits: Bit array [batch, num_bits]

    Returns:
        Integer array [batch]
    """
    result = cp.zeros(bits.shape[0], dtype=cp.int32)
    for i in range(bits.shape[1]):
        result += ((bits[:, i] > 0.5).astype(cp.int32)) << i
    return result
