"""
Tests for TriXc Binary Format
==============================

The TriXc format is the portable binary format for frozen shapes.

Format:
  Header (64 bytes) + Shape Data (N bytes) + Footer (16 bytes checksum)
"""

import pytest
from pathlib import Path
import tempfile

from foundry.export.trixc import TriXcFormat, ChipSpec, ChipType


class TestTriXcFormat:
    """Tests for the TriXc binary format handler."""

    def test_magic_bytes(self):
        """TriXc files start with 'TRIX' magic bytes."""
        spec = ChipSpec(
            name="test",
            chip_type=ChipType.ATOM,
            bits=8,
            description="Test chip",
            polynomial="a",
            coefficients=[1],
            inputs=["a"],
        )
        data = TriXcFormat.pack(spec)
        assert data[:4] == b"TRIX"

    def test_version_bytes(self):
        """Version is encoded as major.minor after magic."""
        spec = ChipSpec(
            name="test",
            chip_type=ChipType.ATOM,
            bits=8,
            description="",
            polynomial="a",
            coefficients=[1],
            inputs=["a"],
        )
        data = TriXcFormat.pack(spec)
        assert data[4] == 1  # Major version
        assert data[5] == 0  # Minor version

    def test_header_size(self):
        """Header is exactly 64 bytes."""
        assert TriXcFormat.HEADER_SIZE == 64

    def test_footer_size(self):
        """Footer (checksum) is exactly 16 bytes."""
        assert TriXcFormat.FOOTER_SIZE == 16

    def test_pack_unpack_roundtrip(self):
        """Pack and unpack should produce equivalent spec."""
        original = ChipSpec(
            name="XOR",
            chip_type=ChipType.ATOM,
            bits=1,
            description="XOR gate",
            polynomial="a + b - 2ab",
            coefficients=[1, 1, -2],
            inputs=["a", "b"],
        )

        packed = TriXcFormat.pack(original)
        unpacked, _ = TriXcFormat.unpack(packed)

        assert unpacked.name == original.name
        assert unpacked.chip_type == original.chip_type
        assert unpacked.bits == original.bits
        assert unpacked.coefficients == original.coefficients

    def test_file_save_load_roundtrip(self):
        """Save and load from file should work."""
        spec = ChipSpec(
            name="AND",
            chip_type=ChipType.ATOM,
            bits=1,
            description="AND gate",
            polynomial="a * b",
            coefficients=[0, 0, 1],
            inputs=["a", "b"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "and.trixc"
            TriXcFormat.save(spec, path)

            assert path.exists()

            loaded = TriXcFormat.load(path)
            assert loaded.name == spec.name
            assert loaded.coefficients == spec.coefficients

    def test_checksum_verification(self):
        """Corrupted files should fail checksum verification."""
        spec = ChipSpec(
            name="test",
            chip_type=ChipType.ATOM,
            bits=8,
            description="",
            polynomial="a",
            coefficients=[1, 2, 3],
            inputs=["a"],
        )

        data = bytearray(TriXcFormat.pack(spec))
        # Corrupt a byte in the middle
        data[70] ^= 0xFF

        with pytest.raises(ValueError, match="Checksum mismatch"):
            TriXcFormat.unpack(bytes(data))

    def test_invalid_magic(self):
        """Invalid magic bytes should raise error."""
        data = b"NOTX" + b"\x00" * 76  # Wrong magic

        with pytest.raises(ValueError, match="Invalid magic"):
            TriXcFormat.unpack(data)

    def test_minimum_file_size(self):
        """Files smaller than header + footer should fail."""
        data = b"TRIX" + b"\x00" * 10  # Too small

        with pytest.raises(ValueError, match="too small"):
            TriXcFormat.unpack(data)

    def test_chip_types(self):
        """All chip types should pack/unpack correctly."""
        for chip_type in ChipType:
            spec = ChipSpec(
                name=f"test_{chip_type.name}",
                chip_type=chip_type,
                bits=8,
                description="",
                polynomial="a",
                coefficients=[1],
                inputs=["a"],
            )
            packed = TriXcFormat.pack(spec)
            unpacked, _ = TriXcFormat.unpack(packed)
            assert unpacked.chip_type == chip_type

    def test_negative_coefficients(self):
        """Negative coefficients should survive roundtrip."""
        spec = ChipSpec(
            name="test",
            chip_type=ChipType.ATOM,
            bits=8,
            description="",
            polynomial="a - b",
            coefficients=[1, -1, -128, 127, -2],
            inputs=["a", "b"],
        )

        packed = TriXcFormat.pack(spec)
        unpacked, _ = TriXcFormat.unpack(packed)

        assert unpacked.coefficients == spec.coefficients

    def test_long_name_truncation(self):
        """Names longer than 47 chars should be truncated."""
        long_name = "A" * 100
        spec = ChipSpec(
            name=long_name,
            chip_type=ChipType.ATOM,
            bits=8,
            description="",
            polynomial="a",
            coefficients=[1],
            inputs=["a"],
        )

        packed = TriXcFormat.pack(spec)
        unpacked, _ = TriXcFormat.unpack(packed)

        assert len(unpacked.name) <= 47
        assert unpacked.name == long_name[:47]

    def test_file_size_calculation(self):
        """File size should be header + coefficients + footer."""
        coefficients = [1, 2, 3, 4, 5]
        spec = ChipSpec(
            name="test",
            chip_type=ChipType.ATOM,
            bits=8,
            description="",
            polynomial="",
            coefficients=coefficients,
            inputs=[],
        )

        packed = TriXcFormat.pack(spec)
        expected_size = 64 + len(coefficients) + 16
        assert len(packed) == expected_size


class TestChipSpec:
    """Tests for the ChipSpec dataclass."""

    def test_size_property(self):
        """Size should return coefficient count."""
        spec = ChipSpec(
            name="test",
            chip_type=ChipType.ATOM,
            bits=8,
            description="",
            polynomial="",
            coefficients=[1, 2, 3, 4, 5],
            inputs=[],
        )
        assert spec.size == 5

    def test_default_validated(self):
        """Chips should default to validated=True."""
        spec = ChipSpec(
            name="test",
            chip_type=ChipType.ATOM,
            bits=8,
            description="",
            polynomial="",
            coefficients=[],
            inputs=[],
        )
        assert spec.validated is True
