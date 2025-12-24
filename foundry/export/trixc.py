"""
TriXc Binary Format
===================

The portable binary format for frozen shapes.

Format v1.0:
  Header (64 bytes):
    Magic:      "TRIX" (4 bytes)
    Version:    Major.Minor (2 bytes)
    Type:       ATOM|MOLECULE|PROTEIN (1 byte)
    Bits:       8|16|32 (1 byte)
    Shape size: N bytes (4 bytes)
    Reserved:   (4 bytes)
    Name:       Null-terminated (48 bytes)

  Shape Data (N bytes):
    Frozen polynomial coefficients

  Footer (16 bytes):
    Checksum:   SHA-256 truncated (16 bytes)
"""

import struct
import hashlib
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path


class ChipType(IntEnum):
    """Chip complexity tier."""
    ATOM = 1      # Primitive operations (XOR, AND, ADD)
    MOLECULE = 2  # Compositions (half adder, full adder)
    PROTEIN = 3   # Complete units (6502 CPU)
    PATHWAY = 4   # Multi-chip systems


@dataclass
class ChipSpec:
    """Specification for a chip in the catalog."""
    name: str
    chip_type: ChipType
    bits: int
    description: str
    polynomial: str
    coefficients: List[int]
    inputs: List[str]
    validated: bool = True

    @property
    def size(self) -> int:
        """Size of shape data in bytes."""
        return len(self.coefficients)


class TriXcFormat:
    """
    Handler for the TriXc binary format.

    The .trixc file IS the chip. Load it, run it.
    """

    MAGIC = b"TRIX"
    VERSION_MAJOR = 1
    VERSION_MINOR = 0
    HEADER_SIZE = 64
    FOOTER_SIZE = 16
    NAME_SIZE = 48

    @classmethod
    def pack(cls, spec: ChipSpec) -> bytes:
        """
        Pack a chip specification into TriXc binary format.

        Returns the complete .trixc file contents.
        """
        # Shape data: pack coefficients as signed bytes
        shape_data = bytes([(c + 256) % 256 for c in spec.coefficients])

        # Build header
        name_bytes = spec.name.encode('utf-8')[:cls.NAME_SIZE - 1]
        name_padded = name_bytes + b'\x00' * (cls.NAME_SIZE - len(name_bytes))

        header = struct.pack(
            '<4sBBBBI4s48s',
            cls.MAGIC,
            cls.VERSION_MAJOR,
            cls.VERSION_MINOR,
            spec.chip_type,
            spec.bits,
            len(shape_data),
            b'\x00' * 4,  # Reserved
            name_padded
        )

        assert len(header) == cls.HEADER_SIZE, f"Header size mismatch: {len(header)}"

        # Calculate checksum over header + shape data
        full_data = header + shape_data
        checksum = hashlib.sha256(full_data).digest()[:cls.FOOTER_SIZE]

        return full_data + checksum

    @classmethod
    def unpack(cls, data: bytes) -> Tuple[ChipSpec, bytes]:
        """
        Unpack TriXc binary data into a chip specification.

        Returns (ChipSpec, raw_coefficients).
        """
        if len(data) < cls.HEADER_SIZE + cls.FOOTER_SIZE:
            raise ValueError(f"Data too small: {len(data)} bytes")

        # Parse header
        header = data[:cls.HEADER_SIZE]
        magic, ver_major, ver_minor, chip_type, bits, shape_size, _, name_bytes = struct.unpack(
            '<4sBBBBI4s48s',
            header
        )

        if magic != cls.MAGIC:
            raise ValueError(f"Invalid magic: {magic}")

        # Extract name (null-terminated)
        name = name_bytes.rstrip(b'\x00').decode('utf-8')

        # Extract shape data
        shape_data = data[cls.HEADER_SIZE:cls.HEADER_SIZE + shape_size]
        coefficients = [b if b < 128 else b - 256 for b in shape_data]

        # Verify checksum
        expected_checksum = data[-cls.FOOTER_SIZE:]
        actual_checksum = hashlib.sha256(data[:-cls.FOOTER_SIZE]).digest()[:cls.FOOTER_SIZE]

        if expected_checksum != actual_checksum:
            raise ValueError("Checksum mismatch - file may be corrupted")

        spec = ChipSpec(
            name=name,
            chip_type=ChipType(chip_type),
            bits=bits,
            description="",  # Not stored in binary
            polynomial="",   # Not stored in binary
            coefficients=coefficients,
            inputs=[],       # Not stored in binary
            validated=True   # If it's in binary form, it was validated
        )

        return spec, shape_data

    @classmethod
    def save(cls, spec: ChipSpec, path: Path) -> int:
        """Save chip to .trixc file. Returns bytes written."""
        data = cls.pack(spec)
        path.write_bytes(data)
        return len(data)

    @classmethod
    def load(cls, path: Path) -> ChipSpec:
        """Load chip from .trixc file."""
        data = path.read_bytes()
        spec, _ = cls.unpack(data)
        return spec
