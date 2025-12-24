"""
Binary Shape Format - Frozen Shape Executable (.fsh)

The Frozen Shape format stores shapes in a binary-executable form that
Thor hardware can load and execute directly.

Format:
    Header (32 bytes):
        0-3:   Magic "FSHP"
        4-5:   Version (major.minor)
        6:     Kingdom ID
        7:     Shape Type (elemental=0, compound=1)
        8:     Arity (unary=1, binary=2, nary=0xFF)
        9:     Opcode (hardware instruction mapping)
        10-11: Flags
        12-15: Component count (for compounds)
        16-31: Shape name (null-padded)

    Body (variable):
        For compounds: list of component opcodes
        Implementation bytecode (future)

"The shape describes itself. The hardware executes it."
"""

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Dict, Any
from pathlib import Path


# Magic bytes
FSH_MAGIC = b"FSHP"
FSH_VERSION = (1, 0)


class KingdomID(IntEnum):
    """Kingdom identifiers for binary format."""
    LOGIC = 0x01
    ARITHMETIC = 0x02
    ACTIVATION = 0x03
    NORMALIZATION = 0x04
    LINEAR = 0x05
    ATTENTION = 0x06
    POOLING = 0x07


class ShapeTypeID(IntEnum):
    """Shape type identifiers."""
    ELEMENTAL = 0x00
    COMPOUND = 0x01


class ArityID(IntEnum):
    """Arity identifiers."""
    UNARY = 0x01
    BINARY = 0x02
    TERNARY = 0x03
    NARY = 0xFF


class Opcode(IntEnum):
    """
    Hardware opcodes for frozen shapes.

    These map to Thor instruction codes.
    Range 0x00-0x1F: Logic
    Range 0x20-0x3F: Arithmetic
    Range 0x40-0x5F: Activation
    Range 0x60-0x7F: Normalization
    Range 0x80-0x9F: Pooling
    Range 0xA0-0xBF: Linear
    Range 0xC0-0xDF: Attention
    Range 0xE0-0xFF: Reserved/Compound
    """
    # Logic Kingdom (0x00-0x1F)
    XOR = 0x00
    AND = 0x01
    OR = 0x02
    NOT = 0x03
    NAND = 0x04
    NOR = 0x05
    XNOR = 0x06

    # Arithmetic Kingdom (0x20-0x3F)
    ADD = 0x20
    SUB = 0x21
    MUL = 0x22
    NEG = 0x23
    POPCOUNT = 0x24

    # Activation Kingdom (0x40-0x5F)
    RELU = 0x40
    SIGMOID = 0x41
    TANH = 0x42
    GELU = 0x43
    SWISH = 0x44
    SOFTMAX = 0x45
    LEAKY_RELU = 0x46

    # Normalization Kingdom (0x60-0x7F)
    LAYER_NORM = 0x60
    RMS_NORM = 0x61

    # Pooling Kingdom (0x80-0x9F)
    MAX_POOL = 0x80
    AVG_POOL = 0x81
    SUM_POOL = 0x82
    MIN_POOL = 0x83
    ARGMIN = 0x84
    ARGMAX = 0x85

    # Linear Kingdom (0xA0-0xBF)
    MATMUL = 0xA0
    MATVEC = 0xA1
    DOT = 0xA2

    # Attention Kingdom (0xC0-0xDF)
    ATTENTION = 0xC0
    CROSS_ATTENTION = 0xC1

    # Compound Shapes (0xE0-0xFF)
    HALF_ADDER = 0xE0
    FULL_ADDER = 0xE1
    HAMMING = 0xE2
    ZIT = 0xE3  # The core recognition circuit: popcount(S XOR vx) < theta


class FSHFlags(IntEnum):
    """Binary flags for shape properties."""
    NONE = 0x0000
    DIFFERENTIABLE = 0x0001    # Smooth/differentiable implementation
    FROZEN = 0x0002            # No learnable parameters
    PARALLEL = 0x0004          # Can execute in parallel
    VECTORIZED = 0x0008        # Has vectorized implementation
    INPLACE = 0x0010           # Can operate in-place


@dataclass
class FrozenShapeHeader:
    """
    Frozen Shape file header (32 bytes).
    """
    magic: bytes = FSH_MAGIC
    version_major: int = FSH_VERSION[0]
    version_minor: int = FSH_VERSION[1]
    kingdom: KingdomID = KingdomID.LOGIC
    shape_type: ShapeTypeID = ShapeTypeID.ELEMENTAL
    arity: ArityID = ArityID.BINARY
    opcode: int = 0x00
    flags: int = FSHFlags.FROZEN
    component_count: int = 0
    name: str = ""

    def pack(self) -> bytes:
        """Pack header to 32 bytes."""
        name_bytes = self.name.encode("utf-8")[:16].ljust(16, b"\x00")

        return struct.pack(
            "<4s BB BBB B HI 16s",
            self.magic,
            self.version_major,
            self.version_minor,
            self.kingdom,
            self.shape_type,
            self.arity,
            self.opcode,
            self.flags,
            self.component_count,
            name_bytes
        )

    @classmethod
    def unpack(cls, data: bytes) -> "FrozenShapeHeader":
        """Unpack header from bytes."""
        if len(data) < 32:
            raise ValueError(f"Header too short: {len(data)} bytes")

        (magic, ver_major, ver_minor, kingdom, shape_type,
         arity, opcode, flags, component_count, name_bytes) = struct.unpack(
            "<4s BB BBB B HI 16s", data[:32]
        )

        if magic != FSH_MAGIC:
            raise ValueError(f"Invalid magic: {magic}")

        return cls(
            magic=magic,
            version_major=ver_major,
            version_minor=ver_minor,
            kingdom=KingdomID(kingdom),
            shape_type=ShapeTypeID(shape_type),
            arity=ArityID(arity) if arity in [1, 2, 3, 0xFF] else ArityID.BINARY,
            opcode=opcode,
            flags=flags,
            component_count=component_count,
            name=name_bytes.rstrip(b"\x00").decode("utf-8")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "magic": self.magic.decode("utf-8"),
            "version": f"{self.version_major}.{self.version_minor}",
            "kingdom": self.kingdom.name,
            "type": self.shape_type.name,
            "arity": self.arity.name,
            "opcode": f"0x{self.opcode:02X}",
            "flags": self.flags,
            "components": self.component_count,
            "name": self.name,
        }


@dataclass
class FrozenShape:
    """
    A frozen shape in binary form.

    Can be saved to .fsh files and loaded by Thor hardware.
    """
    header: FrozenShapeHeader
    components: List[int] = None  # Component opcodes for compounds

    def __post_init__(self):
        if self.components is None:
            self.components = []

    def pack(self) -> bytes:
        """Pack shape to bytes."""
        data = self.header.pack()

        # Add component opcodes for compound shapes
        if self.header.shape_type == ShapeTypeID.COMPOUND:
            for opcode in self.components:
                data += struct.pack("<B", opcode)

        return data

    @classmethod
    def unpack(cls, data: bytes) -> "FrozenShape":
        """Unpack shape from bytes."""
        header = FrozenShapeHeader.unpack(data[:32])

        components = []
        if header.shape_type == ShapeTypeID.COMPOUND:
            offset = 32
            for _ in range(header.component_count):
                if offset < len(data):
                    components.append(data[offset])
                    offset += 1

        return cls(header=header, components=components)

    def save(self, path: str) -> None:
        """Save shape to .fsh file."""
        with open(path, "wb") as f:
            f.write(self.pack())

    @classmethod
    def load(cls, path: str) -> "FrozenShape":
        """Load shape from .fsh file."""
        with open(path, "rb") as f:
            return cls.unpack(f.read())

    def info(self) -> str:
        """Human-readable info."""
        lines = [
            f"┌─────────────────────────────────────┐",
            f"│ {self.header.name.upper():^35} │",
            f"├─────────────────────────────────────┤",
            f"│ Kingdom: {self.header.kingdom.name:<26}│",
            f"│ Type:    {self.header.shape_type.name:<26}│",
            f"│ Arity:   {self.header.arity.name:<26}│",
            f"│ Opcode:  0x{self.header.opcode:02X}{'':<24}│",
        ]

        if self.components:
            comp_str = ", ".join(f"0x{c:02X}" for c in self.components)
            lines.append(f"│ Built From: {comp_str:<23}│")

        lines.append(f"└─────────────────────────────────────┘")
        return "\n".join(lines)


# =============================================================================
# Shape Definitions - All frozen shapes with their binary representations
# =============================================================================

SHAPES: Dict[str, FrozenShape] = {}

def register_shape(
    name: str,
    kingdom: KingdomID,
    opcode: int,
    arity: ArityID = ArityID.BINARY,
    shape_type: ShapeTypeID = ShapeTypeID.ELEMENTAL,
    components: List[int] = None,
    flags: int = FSHFlags.FROZEN,
) -> FrozenShape:
    """Register a frozen shape."""
    header = FrozenShapeHeader(
        kingdom=kingdom,
        shape_type=shape_type,
        arity=arity,
        opcode=opcode,
        flags=flags,
        component_count=len(components) if components else 0,
        name=name,
    )
    shape = FrozenShape(header=header, components=components or [])
    SHAPES[name] = shape
    return shape


# -----------------------------------------------------------------------------
# Logic Kingdom (0x00-0x1F)
# -----------------------------------------------------------------------------

register_shape("xor", KingdomID.LOGIC, Opcode.XOR, ArityID.BINARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

register_shape("and", KingdomID.LOGIC, Opcode.AND, ArityID.BINARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

register_shape("or", KingdomID.LOGIC, Opcode.OR, ArityID.BINARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

register_shape("not", KingdomID.LOGIC, Opcode.NOT, ArityID.UNARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

register_shape("nand", KingdomID.LOGIC, Opcode.NAND, ArityID.BINARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

register_shape("nor", KingdomID.LOGIC, Opcode.NOR, ArityID.BINARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

register_shape("xnor", KingdomID.LOGIC, Opcode.XNOR, ArityID.BINARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

# -----------------------------------------------------------------------------
# Arithmetic Kingdom (0x20-0x3F)
# -----------------------------------------------------------------------------

register_shape("add", KingdomID.ARITHMETIC, Opcode.ADD, ArityID.BINARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

register_shape("sub", KingdomID.ARITHMETIC, Opcode.SUB, ArityID.BINARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

register_shape("mul", KingdomID.ARITHMETIC, Opcode.MUL, ArityID.BINARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

register_shape("neg", KingdomID.ARITHMETIC, Opcode.NEG, ArityID.UNARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

register_shape("popcount", KingdomID.ARITHMETIC, Opcode.POPCOUNT, ArityID.UNARY,
               flags=FSHFlags.FROZEN | FSHFlags.PARALLEL)

# -----------------------------------------------------------------------------
# Activation Kingdom (0x40-0x5F)
# -----------------------------------------------------------------------------

register_shape("relu", KingdomID.ACTIVATION, Opcode.RELU, ArityID.UNARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL | FSHFlags.INPLACE)

register_shape("sigmoid", KingdomID.ACTIVATION, Opcode.SIGMOID, ArityID.UNARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

register_shape("tanh", KingdomID.ACTIVATION, Opcode.TANH, ArityID.UNARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

register_shape("gelu", KingdomID.ACTIVATION, Opcode.GELU, ArityID.UNARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

register_shape("swish", KingdomID.ACTIVATION, Opcode.SWISH, ArityID.UNARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

register_shape("softmax", KingdomID.ACTIVATION, Opcode.SOFTMAX, ArityID.NARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.INPLACE)

register_shape("leaky_relu", KingdomID.ACTIVATION, Opcode.LEAKY_RELU, ArityID.UNARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL | FSHFlags.INPLACE)

# -----------------------------------------------------------------------------
# Normalization Kingdom (0x60-0x7F)
# -----------------------------------------------------------------------------

register_shape("layer_norm", KingdomID.NORMALIZATION, Opcode.LAYER_NORM, ArityID.NARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.INPLACE)

register_shape("rms_norm", KingdomID.NORMALIZATION, Opcode.RMS_NORM, ArityID.NARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.INPLACE)

# -----------------------------------------------------------------------------
# Pooling Kingdom (0x80-0x9F)
# -----------------------------------------------------------------------------

register_shape("max_pool", KingdomID.POOLING, Opcode.MAX_POOL, ArityID.NARY,
               flags=FSHFlags.FROZEN | FSHFlags.PARALLEL)

register_shape("avg_pool", KingdomID.POOLING, Opcode.AVG_POOL, ArityID.NARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

register_shape("sum_pool", KingdomID.POOLING, Opcode.SUM_POOL, ArityID.NARY,
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

register_shape("min_pool", KingdomID.POOLING, Opcode.MIN_POOL, ArityID.NARY,
               flags=FSHFlags.FROZEN | FSHFlags.PARALLEL)

register_shape("argmin", KingdomID.POOLING, Opcode.ARGMIN, ArityID.NARY,
               flags=FSHFlags.FROZEN | FSHFlags.PARALLEL)

register_shape("argmax", KingdomID.POOLING, Opcode.ARGMAX, ArityID.NARY,
               flags=FSHFlags.FROZEN | FSHFlags.PARALLEL)

# -----------------------------------------------------------------------------
# Compound Shapes (0xE0-0xFF)
# -----------------------------------------------------------------------------

register_shape("half_adder", KingdomID.ARITHMETIC, Opcode.HALF_ADDER, ArityID.BINARY,
               shape_type=ShapeTypeID.COMPOUND,
               components=[Opcode.XOR, Opcode.AND],
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

register_shape("full_adder", KingdomID.ARITHMETIC, Opcode.FULL_ADDER, ArityID.TERNARY,
               shape_type=ShapeTypeID.COMPOUND,
               components=[Opcode.XOR, Opcode.XOR, Opcode.AND, Opcode.AND, Opcode.OR],
               flags=FSHFlags.FROZEN | FSHFlags.DIFFERENTIABLE | FSHFlags.PARALLEL)

register_shape("hamming", KingdomID.ARITHMETIC, Opcode.HAMMING, ArityID.BINARY,
               shape_type=ShapeTypeID.COMPOUND,
               components=[Opcode.XOR, Opcode.POPCOUNT],
               flags=FSHFlags.FROZEN | FSHFlags.PARALLEL)

# ZIT - The core recognition circuit of the NGP
# Zit = popcount(S XOR vx) < theta
# This is Hamming distance + threshold comparison
register_shape("zit", KingdomID.POOLING, Opcode.ZIT, ArityID.BINARY,
               shape_type=ShapeTypeID.COMPOUND,
               components=[Opcode.XOR, Opcode.POPCOUNT],  # + comparator (implicit)
               flags=FSHFlags.FROZEN | FSHFlags.PARALLEL)


# =============================================================================
# API Functions
# =============================================================================

def get_shape(name: str) -> Optional[FrozenShape]:
    """Get a shape by name."""
    return SHAPES.get(name)


def list_shapes() -> List[str]:
    """List all registered shapes."""
    return list(SHAPES.keys())


def list_by_kingdom(kingdom: KingdomID) -> List[str]:
    """List shapes in a kingdom."""
    return [name for name, shape in SHAPES.items()
            if shape.header.kingdom == kingdom]


def get_opcode(name: str) -> Optional[int]:
    """Get the opcode for a shape."""
    shape = SHAPES.get(name)
    return shape.header.opcode if shape else None


def save_all(directory: str) -> None:
    """Save all shapes to a directory as .fsh files."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    for name, shape in SHAPES.items():
        shape.save(str(path / f"{name}.fsh"))


def load_directory(directory: str) -> Dict[str, FrozenShape]:
    """Load all .fsh files from a directory."""
    shapes = {}
    path = Path(directory)

    for fsh_file in path.glob("*.fsh"):
        shape = FrozenShape.load(str(fsh_file))
        shapes[shape.header.name] = shape

    return shapes


def print_opcode_table() -> None:
    """Print the opcode mapping table."""
    print("┌────────────────────────────────────────────────────────────┐")
    print("│              FROZEN SHAPE OPCODE TABLE                     │")
    print("├────────────────────────────────────────────────────────────┤")
    print("│ Opcode │ Name         │ Kingdom      │ Type               │")
    print("├────────┼──────────────┼──────────────┼────────────────────┤")

    for name, shape in sorted(SHAPES.items(), key=lambda x: x[1].header.opcode):
        h = shape.header
        type_str = "COMPOUND" if h.shape_type == ShapeTypeID.COMPOUND else "ELEMENT"
        print(f"│  0x{h.opcode:02X}  │ {name:<12} │ {h.kingdom.name:<12} │ {type_str:<18} │")

    print("└────────────────────────────────────────────────────────────┘")
