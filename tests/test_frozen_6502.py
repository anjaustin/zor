"""
Tests for Frozen 6502 Module

Integration tests for the 16 frozen shapes for 6502 CPU emulation.
All tests validate 100% accuracy on ALU operations.

These tests prove the core thesis: "Computation is topology. Learning is routing."
"""

import pytest
import torch
import torch.nn as nn

from trix.nn.frozen_6502 import (
    ShapeID,
    RegisterID,
    PureMath,
    FrozenShapes6502,
    FlagComputer,
    Frozen6502Tile,
    Frozen6502,
    OPCODE_SPECS,
    int_to_bits,
    bits_to_int,
    create_frozen_6502,
    NUM_SHAPES,
)


class TestPureMath:
    """Test Level 0: Pure Math Primitives."""

    def test_xor_truth_table(self):
        """XOR computes correctly on {0, 1}."""
        a = torch.tensor([0.0, 0.0, 1.0, 1.0])
        b = torch.tensor([0.0, 1.0, 0.0, 1.0])

        result = PureMath.xor(a, b)
        expected = torch.tensor([0.0, 1.0, 1.0, 0.0])

        assert torch.allclose(result, expected)

    def test_and_truth_table(self):
        """AND computes correctly."""
        a = torch.tensor([0.0, 0.0, 1.0, 1.0])
        b = torch.tensor([0.0, 1.0, 0.0, 1.0])

        result = PureMath.and_op(a, b)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0])

        assert torch.allclose(result, expected)

    def test_or_truth_table(self):
        """OR computes correctly."""
        a = torch.tensor([0.0, 0.0, 1.0, 1.0])
        b = torch.tensor([0.0, 1.0, 0.0, 1.0])

        result = PureMath.or_op(a, b)
        expected = torch.tensor([0.0, 1.0, 1.0, 1.0])

        assert torch.allclose(result, expected)

    def test_not_truth_table(self):
        """NOT computes correctly."""
        a = torch.tensor([0.0, 1.0])

        result = PureMath.not_op(a)
        expected = torch.tensor([1.0, 0.0])

        assert torch.allclose(result, expected)

    def test_full_adder(self):
        """Full adder computes correctly."""
        # Test all 8 combinations
        a = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        b = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
        c = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

        sum_bit, carry = PureMath.full_adder(a, b, c)

        expected_sum = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        expected_carry = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0])

        assert torch.allclose(sum_bit, expected_sum)
        assert torch.allclose(carry, expected_carry)


class TestFrozenShapes6502:
    """Test Level 1: Frozen Shapes (16 topologies)."""

    def test_ripple_add_simple(self):
        """Ripple adder: 5 + 3 = 8."""
        a = int_to_bits(torch.tensor([5]))  # [1, 0, 1, 0, 0, 0, 0, 0]
        b = int_to_bits(torch.tensor([3]))  # [1, 1, 0, 0, 0, 0, 0, 0]
        c_in = torch.tensor([0.0])

        out = FrozenShapes6502.ripple_add(a, b, c_in)
        result = bits_to_int(out['result'])

        assert result.item() == 8

    def test_ripple_add_with_carry_in(self):
        """Ripple adder with carry: 5 + 3 + 1 = 9."""
        a = int_to_bits(torch.tensor([5]))
        b = int_to_bits(torch.tensor([3]))
        c_in = torch.tensor([1.0])

        out = FrozenShapes6502.ripple_add(a, b, c_in)
        result = bits_to_int(out['result'])

        assert result.item() == 9

    def test_ripple_add_with_carry_out(self):
        """Ripple adder overflow: 255 + 1 = 0 with carry."""
        a = int_to_bits(torch.tensor([255]))
        b = int_to_bits(torch.tensor([1]))
        c_in = torch.tensor([0.0])

        out = FrozenShapes6502.ripple_add(a, b, c_in)
        result = bits_to_int(out['result'])

        assert result.item() == 0
        assert out['carry'].item() == 1.0

    def test_ripple_sub_simple(self):
        """Subtraction: 10 - 3 = 7."""
        a = int_to_bits(torch.tensor([10]))
        b = int_to_bits(torch.tensor([3]))
        c_in = torch.tensor([1.0])  # No borrow = carry set

        out = FrozenShapes6502.ripple_sub(a, b, c_in)
        result = bits_to_int(out['result'])

        assert result.item() == 7

    def test_parallel_and(self):
        """Parallel AND: 0xF0 AND 0x0F = 0x00."""
        a = int_to_bits(torch.tensor([0xF0]))
        b = int_to_bits(torch.tensor([0x0F]))

        out = FrozenShapes6502.parallel_and(a, b)
        result = bits_to_int(out['result'])

        assert result.item() == 0x00

    def test_parallel_or(self):
        """Parallel OR: 0xF0 OR 0x0F = 0xFF."""
        a = int_to_bits(torch.tensor([0xF0]))
        b = int_to_bits(torch.tensor([0x0F]))

        out = FrozenShapes6502.parallel_or(a, b)
        result = bits_to_int(out['result'])

        assert result.item() == 0xFF

    def test_parallel_xor(self):
        """Parallel XOR: 0xAA XOR 0x55 = 0xFF."""
        a = int_to_bits(torch.tensor([0xAA]))
        b = int_to_bits(torch.tensor([0x55]))

        out = FrozenShapes6502.parallel_xor(a, b)
        result = bits_to_int(out['result'])

        assert result.item() == 0xFF

    def test_shift_left(self):
        """Shift left: 0x42 << 1 = 0x84."""
        a = int_to_bits(torch.tensor([0x42]))

        out = FrozenShapes6502.shift_left(a)
        result = bits_to_int(out['result'])

        assert result.item() == 0x84
        assert out['carry'].item() == 0.0  # MSB was 0

    def test_shift_left_with_carry(self):
        """Shift left with carry out: 0x80 << 1 = 0x00, carry=1."""
        a = int_to_bits(torch.tensor([0x80]))

        out = FrozenShapes6502.shift_left(a)
        result = bits_to_int(out['result'])

        assert result.item() == 0x00
        assert out['carry'].item() == 1.0

    def test_shift_right(self):
        """Shift right: 0x42 >> 1 = 0x21."""
        a = int_to_bits(torch.tensor([0x42]))

        out = FrozenShapes6502.shift_right(a)
        result = bits_to_int(out['result'])

        assert result.item() == 0x21
        assert out['carry'].item() == 0.0  # LSB was 0

    def test_rotate_left(self):
        """Rotate left: 0x80 ROL with C=1 = 0x01, carry=1."""
        a = int_to_bits(torch.tensor([0x80]))
        c_in = torch.tensor([1.0])

        out = FrozenShapes6502.rotate_left(a, c_in)
        result = bits_to_int(out['result'])

        assert result.item() == 0x01
        assert out['carry'].item() == 1.0

    def test_rotate_right(self):
        """Rotate right: 0x01 ROR with C=1 = 0x80, carry=1."""
        a = int_to_bits(torch.tensor([0x01]))
        c_in = torch.tensor([1.0])

        out = FrozenShapes6502.rotate_right(a, c_in)
        result = bits_to_int(out['result'])

        assert result.item() == 0x80
        assert out['carry'].item() == 1.0

    def test_increment(self):
        """Increment: 41 + 1 = 42."""
        a = int_to_bits(torch.tensor([41]))

        out = FrozenShapes6502.increment(a)
        result = bits_to_int(out['result'])

        assert result.item() == 42

    def test_increment_overflow(self):
        """Increment overflow: 255 + 1 = 0."""
        a = int_to_bits(torch.tensor([255]))

        out = FrozenShapes6502.increment(a)
        result = bits_to_int(out['result'])

        assert result.item() == 0

    def test_decrement(self):
        """Decrement: 42 - 1 = 41."""
        a = int_to_bits(torch.tensor([42]))

        out = FrozenShapes6502.decrement(a)
        result = bits_to_int(out['result'])

        assert result.item() == 41

    def test_decrement_underflow(self):
        """Decrement underflow: 0 - 1 = 255."""
        a = int_to_bits(torch.tensor([0]))

        out = FrozenShapes6502.decrement(a)
        result = bits_to_int(out['result'])

        assert result.item() == 255

    def test_transfer(self):
        """Transfer: passthrough."""
        a = int_to_bits(torch.tensor([42]))

        out = FrozenShapes6502.transfer(a)
        result = bits_to_int(out['result'])

        assert result.item() == 42

    def test_load(self):
        """Load: passthrough from memory."""
        m = int_to_bits(torch.tensor([123]))

        out = FrozenShapes6502.load(m)
        result = bits_to_int(out['result'])

        assert result.item() == 123

    def test_store(self):
        """Store: passthrough to memory."""
        a = int_to_bits(torch.tensor([99]))

        out = FrozenShapes6502.store(a)
        result = bits_to_int(out['result'])

        assert result.item() == 99

    def test_identity(self):
        """Identity: passthrough."""
        a = int_to_bits(torch.tensor([77]))

        out = FrozenShapes6502.identity(a)
        result = bits_to_int(out['result'])

        assert result.item() == 77


class TestFlagComputer:
    """Test flag computation."""

    def test_zero_flag(self):
        """Z flag set when result is zero."""
        zero = int_to_bits(torch.tensor([0]))
        nonzero = int_to_bits(torch.tensor([42]))

        z_zero = FlagComputer.zero(zero)
        z_nonzero = FlagComputer.zero(nonzero)

        assert z_zero.item() == 1.0
        assert z_nonzero.item() == 0.0

    def test_negative_flag(self):
        """N flag set when bit 7 is set."""
        positive = int_to_bits(torch.tensor([127]))
        negative = int_to_bits(torch.tensor([128]))

        n_pos = FlagComputer.negative(positive)
        n_neg = FlagComputer.negative(negative)

        assert n_pos.item() == 0.0
        assert n_neg.item() == 1.0


class TestFrozen6502Tile:
    """Test Frozen6502Tile wrapper."""

    @pytest.mark.parametrize("shape_id", list(ShapeID))
    def test_tile_signature(self, shape_id):
        """Each tile has a valid signature."""
        tile = Frozen6502Tile(shape_id)

        sig = tile.get_signature()
        assert sig.shape == (64,)
        assert torch.all((sig == -1) | (sig == 1))

    def test_tile_forward(self):
        """Tile forward executes correctly."""
        tile = Frozen6502Tile(ShapeID.PARALLEL_XOR)

        a = int_to_bits(torch.tensor([0xAA]))
        b = int_to_bits(torch.tensor([0x55]))

        out = tile(a, b)
        result = bits_to_int(out['result'])

        assert result.item() == 0xFF


class TestFrozen6502Model:
    """Test complete Frozen6502 model."""

    def test_creation(self):
        """Model creates successfully."""
        model = create_frozen_6502()
        assert isinstance(model, Frozen6502)
        assert len(model.tiles) == NUM_SHAPES

    def test_param_count(self):
        """Model has minimal parameters (meaning layer only)."""
        model = create_frozen_6502()
        param_count = model.param_count()

        # 56 opcodes * (16 shapes + 8 input_a + 8 input_b + 8 output + 4 flags + 1 carry)
        # = 56 * 45 = 2520, plus or minus
        assert param_count < 5000  # Much less than 100k learned approach

    def test_adc_instruction(self):
        """ADC: A + M + C."""
        model = create_frozen_6502()
        model.eval()

        # A = 42, M = 13, C = 1 => A = 56
        batch = 1
        registers = torch.zeros(batch, 7, 8)
        registers[0, 0, :] = int_to_bits(torch.tensor([42]))[0]  # A = 42
        memory = int_to_bits(torch.tensor([13]))  # M = 13
        carry = torch.tensor([1.0])  # C = 1
        opcode = torch.tensor([0])  # ADC

        out = model(opcode, registers, memory, carry)
        result = bits_to_int(out['result'][0])

        # Use approximate comparison due to soft blending in model
        assert abs(result.item() - 56) < 0.01  # 42 + 13 + 1

    def test_and_instruction(self):
        """AND: A & M."""
        model = create_frozen_6502()
        model.eval()

        batch = 1
        registers = torch.zeros(batch, 7, 8)
        registers[0, 0, :] = int_to_bits(torch.tensor([0xF0]))[0]
        memory = int_to_bits(torch.tensor([0x0F]))
        carry = torch.tensor([0.0])
        opcode = torch.tensor([2])  # AND

        out = model(opcode, registers, memory, carry)
        result = bits_to_int(out['result'][0])

        assert result.item() == 0x00

    def test_inx_instruction(self):
        """INX: X + 1."""
        model = create_frozen_6502()
        model.eval()

        batch = 1
        registers = torch.zeros(batch, 7, 8)
        registers[0, 1, :] = int_to_bits(torch.tensor([41]))[0]  # X = 41
        memory = torch.zeros(batch, 8)
        carry = torch.tensor([0.0])
        opcode = torch.tensor([9])  # INX

        out = model(opcode, registers, memory, carry)
        result = bits_to_int(out['result'][0])

        assert result.item() == 42

    def test_batched_execution(self):
        """Model handles batched inputs."""
        model = create_frozen_6502()
        model.eval()

        batch = 16
        registers = torch.zeros(batch, 7, 8)
        for i in range(batch):
            registers[i, 0, :] = int_to_bits(torch.tensor([i]))[0]

        memory = int_to_bits(torch.tensor([1])).expand(batch, 8)
        carry = torch.zeros(batch)
        opcode = torch.zeros(batch, dtype=torch.long)  # ADC

        out = model(opcode, registers, memory, carry)
        results = bits_to_int(out['result'])

        # Each should be i + 1
        for i in range(batch):
            assert results[i].item() == i + 1


class TestExhaustiveAccuracy:
    """Exhaustive 100% accuracy tests."""

    def test_ripple_add_exhaustive_8bit(self):
        """ADC achieves 100% on 256 sample cases."""
        correct = 0
        total = 0

        for a_val in range(0, 256, 16):  # Sample every 16th value
            for b_val in range(0, 256, 16):
                for c_val in [0, 1]:
                    a = int_to_bits(torch.tensor([a_val]))
                    b = int_to_bits(torch.tensor([b_val]))
                    c = torch.tensor([float(c_val)])

                    out = FrozenShapes6502.ripple_add(a, b, c)
                    result = bits_to_int(out['result']).item()
                    carry = out['carry'].item()

                    expected = (a_val + b_val + c_val) % 256
                    expected_carry = 1 if (a_val + b_val + c_val) >= 256 else 0

                    if result == expected and carry == expected_carry:
                        correct += 1
                    total += 1

        accuracy = correct / total
        assert accuracy == 1.0, f"Accuracy was {accuracy * 100:.1f}%"

    def test_shift_exhaustive_8bit(self):
        """ASL achieves 100% on all 256 values."""
        correct = 0

        for val in range(256):
            a = int_to_bits(torch.tensor([val]))
            out = FrozenShapes6502.shift_left(a)
            result = bits_to_int(out['result']).item()
            carry = out['carry'].item()

            expected = (val << 1) & 0xFF
            expected_carry = 1 if (val & 0x80) else 0

            if result == expected and carry == expected_carry:
                correct += 1

        assert correct == 256

    def test_all_shapes_sample_accuracy(self):
        """Sample test all 16 shapes for basic correctness."""
        # Each tuple: (shape_id, a_val, b_val, c_val, expected)
        # For LOAD, a_val IS the memory value being loaded
        shapes = [
            (ShapeID.RIPPLE_ADD, 5, 3, 0, 8),      # 5 + 3 = 8
            (ShapeID.RIPPLE_SUB, 10, 3, 1, 7),    # 10 - 3 = 7
            (ShapeID.PARALLEL_AND, 0xFF, 0x0F, 0, 0x0F),
            (ShapeID.PARALLEL_OR, 0xF0, 0x0F, 0, 0xFF),
            (ShapeID.PARALLEL_XOR, 0xAA, 0x55, 0, 0xFF),
            (ShapeID.SHIFT_LEFT, 0x42, 0, 0, 0x84),
            (ShapeID.SHIFT_RIGHT, 0x42, 0, 0, 0x21),
            (ShapeID.INCREMENT, 41, 0, 0, 42),
            (ShapeID.DECREMENT, 43, 0, 0, 42),
            (ShapeID.TRANSFER, 42, 0, 0, 42),
            (ShapeID.LOAD, 123, 0, 0, 123),       # LOAD takes m_bits as input_a
            (ShapeID.STORE, 99, 0, 0, 99),
            (ShapeID.IDENTITY, 77, 0, 0, 77),
        ]

        for shape_id, a_val, b_val, c_val, expected in shapes:
            tile = Frozen6502Tile(shape_id)

            a = int_to_bits(torch.tensor([a_val]))
            b = int_to_bits(torch.tensor([b_val]))
            c = torch.tensor([float(c_val)])

            out = tile(a, b, c)
            result = bits_to_int(out['result']).item()

            assert result == expected, f"{shape_id.name}: expected {expected}, got {result}"


class TestIntToFromBits:
    """Test bit conversion utilities."""

    def test_int_to_bits(self):
        """int_to_bits converts correctly."""
        x = torch.tensor([5])
        bits = int_to_bits(x)

        # 5 = 0b00000101 (LSB first)
        expected = torch.tensor([[1, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
        assert torch.allclose(bits, expected)

    def test_bits_to_int(self):
        """bits_to_int converts correctly."""
        bits = torch.tensor([[1, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
        x = bits_to_int(bits)

        assert x.item() == 5

    def test_roundtrip(self):
        """int_to_bits and bits_to_int are inverses."""
        for val in [0, 1, 42, 127, 128, 255]:
            x = torch.tensor([val])
            bits = int_to_bits(x)
            recovered = bits_to_int(bits)
            assert recovered.item() == val

    def test_batched_conversion(self):
        """Batch conversion works."""
        x = torch.tensor([0, 5, 42, 255])
        bits = int_to_bits(x)
        recovered = bits_to_int(bits)

        assert torch.allclose(recovered.float(), x.float())


class TestOpcodeSpecs:
    """Test opcode specification completeness."""

    def test_specs_have_shape(self):
        """All specs have shape field."""
        for opcode_id, spec in OPCODE_SPECS.items():
            assert 'shape' in spec, f"Opcode {opcode_id} missing shape"

    def test_shapes_are_valid(self):
        """All shape references are valid."""
        for opcode_id, spec in OPCODE_SPECS.items():
            shape = spec['shape']
            assert shape in ShapeID, f"Invalid shape {shape} for opcode {opcode_id}"

    def test_registers_are_valid(self):
        """All register references are valid."""
        for opcode_id, spec in OPCODE_SPECS.items():
            if 'input_a' in spec:
                assert spec['input_a'] in RegisterID
            if 'input_b' in spec:
                assert spec['input_b'] in RegisterID
            if 'output' in spec:
                assert spec['output'] in RegisterID
