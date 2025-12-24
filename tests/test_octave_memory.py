"""
Tests for OctaveMemory - Scale-free memory with geometric addressing.

The geometry is in add_octave (frozen shapes chained).
The storage is simple (bytearray).
The carry is the resonance (coupling between octaves).
"""

import pytest
from trix.sim import OctaveMemory, add_octave, int_to_bytes, bytes_to_int


# =============================================================================
# TEST: add_octave (the geometry)
# =============================================================================

class TestAddOctave:
    """Test the chained add function - this is where frozen geometry lives."""

    def test_simple_add_one_octave(self):
        """Basic 8-bit addition."""
        result, carry = add_octave([5], [3])
        assert result == [8]
        assert carry == 0

    def test_simple_add_one_octave_larger(self):
        """8-bit addition with larger values."""
        result, carry = add_octave([100], [50])
        assert result == [150]
        assert carry == 0

    def test_carry_within_octave(self):
        """Overflow within single octave produces carry."""
        result, carry = add_octave([255], [1])
        assert result == [0]
        assert carry == 1

    def test_carry_propagation_two_octaves(self):
        """$00FF + $0001 = $0100 - carry propagates from low to high byte."""
        result, carry = add_octave([0xFF, 0x00], [0x01, 0x00])
        assert result == [0x00, 0x01]
        assert carry == 0

    def test_full_overflow_two_octaves(self):
        """$FFFF + $0001 = $0000 with carry out."""
        result, carry = add_octave([0xFF, 0xFF], [0x01, 0x00])
        assert result == [0x00, 0x00]
        assert carry == 1

    def test_three_octaves_simple(self):
        """24-bit addition without carry."""
        result, carry = add_octave([0x01, 0x02, 0x03], [0x04, 0x05, 0x06])
        assert result == [0x05, 0x07, 0x09]
        assert carry == 0

    def test_three_octaves_carry_chain(self):
        """$00FFFF + $000001 = $010000 - carry chains through two octaves."""
        result, carry = add_octave([0xFF, 0xFF, 0x00], [0x01, 0x00, 0x00])
        assert result == [0x00, 0x00, 0x01]
        assert carry == 0

    def test_initial_carry(self):
        """Adding with initial carry set."""
        result, carry = add_octave([0x00, 0x00], [0x00, 0x00], carry=1)
        assert result == [0x01, 0x00]
        assert carry == 0

    def test_initial_carry_propagates(self):
        """Initial carry causes chain reaction."""
        result, carry = add_octave([0xFF, 0x00], [0x00, 0x00], carry=1)
        assert result == [0x00, 0x01]
        assert carry == 0


# =============================================================================
# TEST: Byte/Int conversion utilities
# =============================================================================

class TestConversions:
    """Test int_to_bytes and bytes_to_int."""

    def test_int_to_bytes_8bit(self):
        assert int_to_bytes(0x42, 1) == [0x42]

    def test_int_to_bytes_16bit(self):
        assert int_to_bytes(0x1234, 2) == [0x34, 0x12]

    def test_int_to_bytes_24bit(self):
        assert int_to_bytes(0x123456, 3) == [0x56, 0x34, 0x12]

    def test_bytes_to_int_8bit(self):
        assert bytes_to_int([0x42]) == 0x42

    def test_bytes_to_int_16bit(self):
        assert bytes_to_int([0x34, 0x12]) == 0x1234

    def test_bytes_to_int_24bit(self):
        assert bytes_to_int([0x56, 0x34, 0x12]) == 0x123456

    def test_roundtrip(self):
        """Convert to bytes and back."""
        for value in [0, 1, 127, 255, 256, 1000, 65535, 0x123456]:
            width = 3
            assert bytes_to_int(int_to_bytes(value, width)) == value


# =============================================================================
# TEST: OctaveMemory construction
# =============================================================================

class TestOctaveMemoryConstruction:
    """Test OctaveMemory initialization at different octaves."""

    def test_octave_0_size(self):
        """Octave 0 = 8-bit = 256 bytes."""
        mem = OctaveMemory(octave=0)
        assert mem.size == 256
        assert mem.width == 1
        assert mem.address_bits == 8
        assert len(mem) == 256

    def test_octave_1_size(self):
        """Octave 1 = 16-bit = 64KB (classic 6502)."""
        mem = OctaveMemory(octave=1)
        assert mem.size == 65536
        assert mem.width == 2
        assert mem.address_bits == 16

    def test_octave_2_size(self):
        """Octave 2 = 24-bit = 16MB (65816 style)."""
        mem = OctaveMemory(octave=2)
        assert mem.size == 16777216
        assert mem.width == 3
        assert mem.address_bits == 24

    def test_octave_3_size(self):
        """Octave 3 = 32-bit = 4GB."""
        mem = OctaveMemory(octave=3)
        assert mem.size == 4294967296
        assert mem.width == 4

    def test_default_octave(self):
        """Default is octave 1 (64KB)."""
        mem = OctaveMemory()
        assert mem.octave == 1
        assert mem.size == 65536

    def test_negative_octave_raises(self):
        """Negative octave is invalid."""
        with pytest.raises(ValueError):
            OctaveMemory(octave=-1)

    def test_too_large_octave_raises(self):
        """Octave > 7 not yet supported (would need lazy allocation)."""
        with pytest.raises(ValueError):
            OctaveMemory(octave=8)

    def test_repr(self):
        """String representation."""
        mem = OctaveMemory(octave=1)
        assert "OctaveMemory" in repr(mem)
        assert "octave=1" in repr(mem)


# =============================================================================
# TEST: Read/Write operations
# =============================================================================

class TestReadWrite:
    """Test basic read and write operations."""

    def test_write_read_integer_address(self):
        """Write and read using integer address."""
        mem = OctaveMemory(octave=1)
        mem.write(0x42, 0x1234)
        assert mem.read(0x1234) == 0x42

    def test_write_read_byte_address(self):
        """Write and read using byte tuple (little-endian)."""
        mem = OctaveMemory(octave=1)
        mem.write(0x42, 0x34, 0x12)
        assert mem.read(0x34, 0x12) == 0x42

    def test_mixed_addressing(self):
        """Write with int, read with bytes (and vice versa)."""
        mem = OctaveMemory(octave=1)
        mem.write(0xAB, 0x1234)
        assert mem.read(0x34, 0x12) == 0xAB

        mem.write(0xCD, 0x00, 0x10)
        assert mem.read(0x1000) == 0xCD

    def test_value_truncation(self):
        """Values > 255 are truncated to byte."""
        mem = OctaveMemory(octave=1)
        mem.write(0x1FF, 0x0000)  # Should store 0xFF
        assert mem.read(0x0000) == 0xFF

    def test_address_wrap(self):
        """Addresses wrap at memory size boundary."""
        mem = OctaveMemory(octave=1)  # 64KB
        mem.write(0x42, 0x10000)  # Wraps to 0x0000
        assert mem.read(0x0000) == 0x42

    def test_initial_memory_is_zero(self):
        """Memory starts as all zeros."""
        mem = OctaveMemory(octave=0)
        for addr in range(256):
            assert mem.read(addr) == 0


# =============================================================================
# TEST: Word operations
# =============================================================================

class TestWordOperations:
    """Test 16-bit word read/write."""

    def test_read_word(self):
        """Read 16-bit word (little-endian)."""
        mem = OctaveMemory(octave=1)
        mem.write(0x34, 0x1000)  # Low byte
        mem.write(0x12, 0x1001)  # High byte
        assert mem.read_word(0x1000) == 0x1234

    def test_write_word(self):
        """Write 16-bit word (little-endian)."""
        mem = OctaveMemory(octave=1)
        mem.write_word(0x1234, 0x1000)
        assert mem.read(0x1000) == 0x34
        assert mem.read(0x1001) == 0x12

    def test_word_roundtrip(self):
        """Write and read back word."""
        mem = OctaveMemory(octave=1)
        mem.write_word(0xABCD, 0x2000)
        assert mem.read_word(0x2000) == 0xABCD

    def test_word_at_boundary(self):
        """Word at memory boundary wraps correctly."""
        mem = OctaveMemory(octave=1)  # 64KB
        mem.write_word(0x1234, 0xFFFF)
        assert mem.read(0xFFFF) == 0x34  # Low byte at $FFFF
        assert mem.read(0x0000) == 0x12  # High byte wraps to $0000


# =============================================================================
# TEST: Effective address (geometric addressing)
# =============================================================================

class TestEffectiveAddress:
    """Test effective address computation using frozen geometry."""

    def test_simple_effective_address(self):
        """Base + offset without carry."""
        mem = OctaveMemory(octave=1)
        addr = mem.effective_address(0x1200, 0x34)
        assert addr == 0x1234

    def test_effective_address_with_carry(self):
        """Carry propagates from low to high byte."""
        mem = OctaveMemory(octave=1)
        addr = mem.effective_address(0x12FF, 0x01)
        assert addr == 0x1300

    def test_effective_address_large_offset(self):
        """Larger offset."""
        mem = OctaveMemory(octave=1)
        addr = mem.effective_address(0x1000, 0x0234)
        assert addr == 0x1234

    def test_effective_address_overflow(self):
        """Address overflow wraps."""
        mem = OctaveMemory(octave=1)
        addr = mem.effective_address(0xFFFF, 0x0001)
        assert addr == 0x0000

    def test_effective_address_octave_0(self):
        """Effective address in 8-bit space."""
        mem = OctaveMemory(octave=0)
        addr = mem.effective_address(0x80, 0x10)
        assert addr == 0x90

    def test_effective_address_octave_0_wrap(self):
        """8-bit address wraps at 256."""
        mem = OctaveMemory(octave=0)
        addr = mem.effective_address(0xFF, 0x02)
        assert addr == 0x01

    def test_effective_address_octave_2(self):
        """24-bit effective address."""
        mem = OctaveMemory(octave=2)
        addr = mem.effective_address(0x00FFFF, 0x000001)
        assert addr == 0x010000


# =============================================================================
# TEST: Bulk operations
# =============================================================================

class TestBulkOperations:
    """Test fill, load, dump operations."""

    def test_fill_entire_memory(self):
        """Fill entire memory with a value."""
        mem = OctaveMemory(octave=0)  # 256 bytes
        mem.fill(0xAA)
        for addr in range(256):
            assert mem.read(addr) == 0xAA

    def test_fill_region(self):
        """Fill a specific region."""
        mem = OctaveMemory(octave=0)
        mem.fill(0xFF, start=0x10, end=0x20)
        assert mem.read(0x0F) == 0x00
        assert mem.read(0x10) == 0xFF
        assert mem.read(0x1F) == 0xFF
        assert mem.read(0x20) == 0x00

    def test_load_bytes(self):
        """Load bytes into memory."""
        mem = OctaveMemory(octave=1)
        data = bytes([0x01, 0x02, 0x03, 0x04])
        mem.load(data, 0x0200)
        assert mem.read(0x0200) == 0x01
        assert mem.read(0x0201) == 0x02
        assert mem.read(0x0202) == 0x03
        assert mem.read(0x0203) == 0x04

    def test_dump_region(self):
        """Dump a memory region."""
        mem = OctaveMemory(octave=1)
        data = bytes([0x01, 0x02, 0x03, 0x04])
        mem.load(data, 0x0200)
        assert mem.dump(0x0200, 4) == data

    def test_load_dump_roundtrip(self):
        """Load and dump are inverse operations."""
        mem = OctaveMemory(octave=1)
        original = bytes(range(256))
        mem.load(original, 0x1000)
        result = mem.dump(0x1000, 256)
        assert result == original


# =============================================================================
# TEST: Integration with frozen shapes
# =============================================================================

class TestGeometryIntegration:
    """Verify frozen shapes are actually used in addressing."""

    def test_effective_address_uses_add_with_carry(self):
        """
        The effective_address method should use add_with_carry internally.
        We verify this by checking carry propagation behavior.
        """
        mem = OctaveMemory(octave=1)

        # This computation requires carry propagation
        # $00FF + $0001 = $0100
        addr = mem.effective_address(0x00FF, 0x0001)
        assert addr == 0x0100

        # More complex: $0FFF + $0001 = $1000
        addr = mem.effective_address(0x0FFF, 0x0001)
        assert addr == 0x1000

    def test_multi_octave_consistency(self):
        """Same simple computation works at different octaves."""
        for octave in [0, 1, 2]:
            mem = OctaveMemory(octave=octave)
            # Simple add that fits in any octave
            addr = mem.effective_address(0x05, 0x03)
            assert addr == 0x08

    def test_indexed_memory_access(self):
        """Simulate indexed addressing mode: base + X."""
        mem = OctaveMemory(octave=1)

        # Store values at $1000, $1001, $1002
        mem.write(0xAA, 0x1000)
        mem.write(0xBB, 0x1001)
        mem.write(0xCC, 0x1002)

        # Access with base + offset (like LDA $1000,X)
        base = 0x1000
        for x in range(3):
            addr = mem.effective_address(base, x)
            expected = [0xAA, 0xBB, 0xCC][x]
            assert mem.read(addr) == expected


# =============================================================================
# TEST: Edge cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_address(self):
        """Reading/writing address 0."""
        mem = OctaveMemory(octave=1)
        mem.write(0x42, 0x0000)
        assert mem.read(0x0000) == 0x42

    def test_max_address(self):
        """Reading/writing maximum address."""
        mem = OctaveMemory(octave=1)
        mem.write(0x42, 0xFFFF)
        assert mem.read(0xFFFF) == 0x42

    def test_zero_value(self):
        """Writing and reading zero."""
        mem = OctaveMemory(octave=1)
        mem.write(0xFF, 0x1000)
        mem.write(0x00, 0x1000)
        assert mem.read(0x1000) == 0x00

    def test_effective_address_zero_offset(self):
        """Effective address with zero offset."""
        mem = OctaveMemory(octave=1)
        addr = mem.effective_address(0x1234, 0x0000)
        assert addr == 0x1234

    def test_effective_address_zero_base(self):
        """Effective address with zero base."""
        mem = OctaveMemory(octave=1)
        addr = mem.effective_address(0x0000, 0x1234)
        assert addr == 0x1234
