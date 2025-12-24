"""
NGE - Neural-Geometric Executable format.

The NGE is a self-describing binary format for compiled TRIX models:
- Header with magic, version, flags
- Metadata section (JSON)
- Code section (native machine code or bytecode)
- Data section (weights)

NGE files can:
- Self-report their contents
- Run on their target platform
- Emit Glassbox events during inference
"""

import struct
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, BinaryIO
from pathlib import Path


# Magic bytes
NGE_MAGIC = b"TRIX"
NGE_VERSION = (1, 0)


class NGEFlags:
    """Bit flags for NGE header."""
    NONE = 0x0000
    GLASSBOX = 0x0001      # Glassbox events enabled
    METADATA = 0x0002      # Metadata section present
    DEBUG = 0x0004         # Debug symbols included
    COMPRESSED = 0x0008    # Data section compressed
    ENCRYPTED = 0x0010     # Data section encrypted


@dataclass
class NGEHeader:
    """
    NGE file header (64 bytes fixed size).

    Layout:
        0-3:   Magic ("TRIX")
        4-5:   Version major
        6-7:   Version minor
        8-9:   Flags
        10-11: Reserved
        12-15: Metadata offset
        16-19: Metadata size
        20-23: Code offset
        24-27: Code size
        28-31: Data offset
        32-35: Data size
        36-51: Target (16 bytes, null-padded)
        52-63: Reserved
    """
    magic: bytes = NGE_MAGIC
    version_major: int = NGE_VERSION[0]
    version_minor: int = NGE_VERSION[1]
    flags: int = NGEFlags.GLASSBOX | NGEFlags.METADATA
    metadata_offset: int = 64  # Right after header
    metadata_size: int = 0
    code_offset: int = 0
    code_size: int = 0
    data_offset: int = 0
    data_size: int = 0
    target: str = "c"

    def has_flag(self, flag: int) -> bool:
        """Check if a flag is set."""
        return (self.flags & flag) != 0

    def set_flag(self, flag: int) -> None:
        """Set a flag."""
        self.flags |= flag

    def clear_flag(self, flag: int) -> None:
        """Clear a flag."""
        self.flags &= ~flag

    def pack(self) -> bytes:
        """Pack header to bytes."""
        target_bytes = self.target.encode("utf-8")[:16].ljust(16, b"\x00")

        return struct.pack(
            "<4sHHHH IIII IIII 16s 12s",
            self.magic,
            self.version_major,
            self.version_minor,
            self.flags,
            0,  # reserved
            self.metadata_offset,
            self.metadata_size,
            self.code_offset,
            self.code_size,
            self.data_offset,
            self.data_size,
            target_bytes,
            b"\x00" * 12  # reserved
        )

    @classmethod
    def unpack(cls, data: bytes) -> "NGEHeader":
        """Unpack header from bytes."""
        if len(data) < 64:
            raise ValueError("Header too short")

        (magic, ver_major, ver_minor, flags, _,
         meta_off, meta_size, code_off, code_size,
         data_off, data_size, target_bytes, _) = struct.unpack(
            "<4sHHHH IIII IIII 16s 12s", data[:64]
        )

        if magic != NGE_MAGIC:
            raise ValueError(f"Invalid magic: {magic}")

        target = target_bytes.rstrip(b"\x00").decode("utf-8")

        return cls(
            magic=magic,
            version_major=ver_major,
            version_minor=ver_minor,
            flags=flags,
            metadata_offset=meta_off,
            metadata_size=meta_size,
            code_offset=code_off,
            code_size=code_size,
            data_offset=data_off,
            data_size=data_size,
            target=target
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "magic": self.magic.decode("utf-8"),
            "version": f"{self.version_major}.{self.version_minor}",
            "flags": {
                "glassbox": self.has_flag(NGEFlags.GLASSBOX),
                "metadata": self.has_flag(NGEFlags.METADATA),
                "debug": self.has_flag(NGEFlags.DEBUG),
                "compressed": self.has_flag(NGEFlags.COMPRESSED),
            },
            "target": self.target,
            "sections": {
                "metadata": {"offset": self.metadata_offset, "size": self.metadata_size},
                "code": {"offset": self.code_offset, "size": self.code_size},
                "data": {"offset": self.data_offset, "size": self.data_size},
            }
        }


@dataclass
class NGEMetadata:
    """Metadata section of NGE file."""
    model_name: str = "unnamed"
    model_version: str = "1.0"
    created: float = field(default_factory=time.time)
    target: str = "c"
    shapes: List[Dict[str, Any]] = field(default_factory=list)
    levers: Dict[str, Any] = field(default_factory=dict)
    input_shape: List[int] = field(default_factory=list)
    output_shape: List[int] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "model": self.model_name,
            "version": self.model_version,
            "created": self.created,
            "target": self.target,
            "shapes": self.shapes,
            "levers": self.levers,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            **self.extra
        })

    @classmethod
    def from_json(cls, data: str) -> "NGEMetadata":
        """Create from JSON string."""
        d = json.loads(data)
        return cls(
            model_name=d.get("model", "unnamed"),
            model_version=d.get("version", "1.0"),
            created=d.get("created", 0),
            target=d.get("target", "c"),
            shapes=d.get("shapes", []),
            levers=d.get("levers", {}),
            input_shape=d.get("input_shape", []),
            output_shape=d.get("output_shape", []),
            extra={k: v for k, v in d.items()
                   if k not in ("model", "version", "created", "target",
                               "shapes", "levers", "input_shape", "output_shape")}
        )


class NGEFile:
    """
    Neural-Geometric Executable file handler.

    Usage:
        # Create
        nge = NGEFile()
        nge.set_metadata(model_name="tiny_mlp", target="arm64-linux")
        nge.set_code(compiled_code)
        nge.set_data(weight_data)
        nge.save("model.nge")

        # Load
        nge = NGEFile.load("model.nge")
        print(nge.metadata.model_name)
        print(nge.header.target)
    """

    def __init__(self):
        self.header = NGEHeader()
        self.metadata = NGEMetadata()
        self.code: bytes = b""
        self.data: bytes = b""

    def set_metadata(self, **kwargs) -> None:
        """Set metadata fields."""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)

    def set_code(self, code: bytes) -> None:
        """Set the code section."""
        if isinstance(code, str):
            code = code.encode("utf-8")
        self.code = code

    def set_data(self, data: bytes) -> None:
        """Set the data section (weights)."""
        self.data = data

    def save(self, path: str) -> None:
        """Save NGE file to disk."""
        # Prepare metadata
        meta_json = self.metadata.to_json().encode("utf-8")

        # Calculate offsets
        self.header.metadata_offset = 64  # After header
        self.header.metadata_size = len(meta_json)
        self.header.code_offset = self.header.metadata_offset + self.header.metadata_size
        self.header.code_size = len(self.code)
        self.header.data_offset = self.header.code_offset + self.header.code_size
        self.header.data_size = len(self.data)
        self.header.target = self.metadata.target

        # Set flags
        self.header.flags = NGEFlags.METADATA | NGEFlags.GLASSBOX

        with open(path, "wb") as f:
            f.write(self.header.pack())
            f.write(meta_json)
            f.write(self.code)
            f.write(self.data)

    @classmethod
    def load(cls, path: str) -> "NGEFile":
        """Load NGE file from disk."""
        nge = cls()

        with open(path, "rb") as f:
            # Read header
            header_data = f.read(64)
            nge.header = NGEHeader.unpack(header_data)

            # Read metadata
            if nge.header.has_flag(NGEFlags.METADATA):
                f.seek(nge.header.metadata_offset)
                meta_json = f.read(nge.header.metadata_size).decode("utf-8")
                nge.metadata = NGEMetadata.from_json(meta_json)

            # Read code
            if nge.header.code_size > 0:
                f.seek(nge.header.code_offset)
                nge.code = f.read(nge.header.code_size)

            # Read data
            if nge.header.data_size > 0:
                f.seek(nge.header.data_offset)
                nge.data = f.read(nge.header.data_size)

        return nge

    def info(self) -> Dict[str, Any]:
        """Get summary information about the NGE."""
        return {
            "model": self.metadata.model_name,
            "version": self.metadata.model_version,
            "target": self.header.target,
            "size": {
                "total": 64 + self.header.metadata_size + self.header.code_size + self.header.data_size,
                "header": 64,
                "metadata": self.header.metadata_size,
                "code": self.header.code_size,
                "data": self.header.data_size,
            },
            "flags": {
                "glassbox": self.header.has_flag(NGEFlags.GLASSBOX),
                "debug": self.header.has_flag(NGEFlags.DEBUG),
            },
            "input_shape": self.metadata.input_shape,
            "output_shape": self.metadata.output_shape,
        }

    def __repr__(self) -> str:
        total = 64 + self.header.metadata_size + self.header.code_size + self.header.data_size
        return f"NGEFile({self.metadata.model_name}, {self.header.target}, {total} bytes)"
