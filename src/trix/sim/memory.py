"""
Octave Memory - Scale-free memory with geometric addressing.

The storage is simple (bytearray).
The addressing is geometric (frozen shapes).
The octaves are harmonic (carry propagation).

The carry is the resonance between octaves - it couples adjacent
frequency bands the same way harmonics couple in sound.

Example:
    >>> mem = OctaveMemory(octave=1)  # 16-bit, 64KB
    >>> mem.write(0x42, 0x0200)
    >>> mem.read(0x0200)
    66

    >>> mem = OctaveMemory(octave=2)  # 24-bit, 16MB
    >>> mem.write(0xFF, 0x010000)

Octave reference:
    0 = 8-bit   = 256 bytes
    1 = 16-bit  = 64 KB       (classic 6502)
    2 = 24-bit  = 16 MB       (65816 style)
    3 = 32-bit  = 4 GB
    n = (n+1)*8-bit = 256^(n+1) bytes

Providence: Storage is simple. Addressing is music.
"""

from typing import Tuple, List, Union
from trix.shapes import add_with_carry


# =============================================================================
# OCTAVE ARITHMETIC
# =============================================================================

def add_octave(a_bytes: List[int], b_bytes: List[int], carry: int = 0) -> Tuple[List[int], int]:
    """
    Chain 8-bit adds across octaves using frozen geometry.

    The carry is the resonance - it couples adjacent octaves.
    This is where the geometry lives.

    Args:
        a_bytes: First operand as list of bytes (little-endian)
        b_bytes: Second operand as list of bytes (little-endian)
        carry: Initial carry (0 or 1)

    Returns:
        (result_bytes, final_carry)

    Example:
        >>> add_octave([0xFF, 0x00], [0x01, 0x00])  # $00FF + $0001
        ([0, 1], 0)  # = $0100

        >>> add_octave([0xFF, 0xFF], [0x01, 0x00])  # $FFFF + $0001
        ([0, 0], 1)  # = $0000 with carry (overflow)
    """
    result = []
    for a, b in zip(a_bytes, b_bytes):
        r, carry = add_with_carry(a, b, carry)
        result.append(r)
    return result, carry


def int_to_bytes(value: int, width: int) -> List[int]:
    """
    Convert integer to little-endian byte list.

    Args:
        value: Integer value
        width: Number of bytes (octave + 1)

    Returns:
        List of bytes, least significant first

    Example:
        >>> int_to_bytes(0x1234, 2)
        [0x34, 0x12]
    """
    return [(value >> (8 * i)) & 0xFF for i in range(width)]


def bytes_to_int(byte_list: List[int]) -> int:
    """
    Convert little-endian byte list to integer.

    Args:
        byte_list: List of bytes, least significant first

    Returns:
        Integer value

    Example:
        >>> bytes_to_int([0x34, 0x12])
        0x1234
    """
    return sum(b << (8 * i) for i, b in enumerate(byte_list))


# =============================================================================
# OCTAVE MEMORY
# =============================================================================

class OctaveMemory:
    """
    Scale-free memory with geometric addressing.

    Storage: Simple bytearray (the bucket).
    Addressing: Frozen shapes compute effective addresses (the music).
    Octaves: Carry propagation between chained operations (the resonance).

    Args:
        octave: Address width in octaves (default 1 = 16-bit = 64KB)
                0 = 8-bit (256 bytes)
                1 = 16-bit (64KB) - classic 6502
                2 = 24-bit (16MB) - 65816 style
                3 = 32-bit (4GB)
                n = (n+1)*8-bit

    Example:
        >>> mem = OctaveMemory(octave=1)
        >>> mem.write(0x42, 0x1234)
        >>> mem.read(0x1234)
        66

        >>> # Effective address via frozen geometry
        >>> mem.effective_address(0x1200, 0x34)
        0x1234
    """

    def __init__(self, octave: int = 1):
        """
        Create memory with specified octave (address width).

        Args:
            octave: 0=256B, 1=64KB, 2=16MB, 3=4GB, ...

        Raises:
            ValueError: If octave < 0 or > 7
        """
        if octave < 0:
            raise ValueError("Octave must be non-negative")
        if octave > 7:
            raise ValueError("Octave > 7 (64-bit) requires lazy allocation (not yet implemented)")

        self.octave = octave
        self.width = octave + 1  # Bytes per address
        self.address_bits = 8 * self.width
        self.size = 256 ** self.width
        self.data = bytearray(self.size)

    def _normalize_addr(self, *args) -> int:
        """Convert address args to integer, normalized to valid range."""
        if len(args) == 1 and isinstance(args[0], int):
            return args[0] % self.size
        # Multiple args = individual bytes (little-endian)
        return bytes_to_int(list(args)) % self.size

    # -------------------------------------------------------------------------
    # Core Read/Write
    # -------------------------------------------------------------------------

    def read(self, *addr) -> int:
        """
        Read a byte from memory.

        Args:
            *addr: Address as single int OR individual bytes (little-endian)

        Returns:
            Byte value (0-255)

        Examples:
            >>> mem.read(0x1234)       # Integer address
            >>> mem.read(0x34, 0x12)   # Bytes, little-endian
        """
        return self.data[self._normalize_addr(*addr)]

    def write(self, value: int, *addr) -> None:
        """
        Write a byte to memory.

        Args:
            value: Byte value to write (0-255)
            *addr: Address as single int OR individual bytes

        Examples:
            >>> mem.write(0x42, 0x1234)
            >>> mem.write(0x42, 0x34, 0x12)
        """
        self.data[self._normalize_addr(*addr)] = value & 0xFF

    def read_word(self, *addr) -> int:
        """
        Read a 16-bit word (little-endian).

        Args:
            *addr: Address of low byte

        Returns:
            16-bit value
        """
        base = self._normalize_addr(*addr)
        lo = self.data[base]
        hi = self.data[(base + 1) % self.size]
        return lo | (hi << 8)

    def write_word(self, value: int, *addr) -> None:
        """
        Write a 16-bit word (little-endian).

        Args:
            value: 16-bit value
            *addr: Address of low byte
        """
        base = self._normalize_addr(*addr)
        self.data[base] = value & 0xFF
        self.data[(base + 1) % self.size] = (value >> 8) & 0xFF

    # -------------------------------------------------------------------------
    # Geometric Addressing
    # -------------------------------------------------------------------------

    def effective_address(self, base: int, offset: int) -> int:
        """
        Compute effective address using frozen geometry.

        This is where the shapes live. The addition is done via
        chained add_with_carry operations - the same frozen shapes
        that compute ALU operations.

        The carry propagates between octaves like harmonics.

        Args:
            base: Base address
            offset: Offset to add

        Returns:
            Effective address (wrapped to valid range)

        Example:
            >>> mem = OctaveMemory(octave=1)
            >>> mem.effective_address(0x1200, 0x34)
            0x1234

            >>> mem.effective_address(0x12FF, 0x01)  # Carry propagates
            0x1300
        """
        base_bytes = int_to_bytes(base, self.width)
        offset_bytes = int_to_bytes(offset, self.width)
        result_bytes, _ = add_octave(base_bytes, offset_bytes)
        return bytes_to_int(result_bytes) % self.size

    # -------------------------------------------------------------------------
    # Bulk Operations
    # -------------------------------------------------------------------------

    def fill(self, value: int, start: int = 0, end: int = None) -> None:
        """
        Fill a memory region with a value.

        Args:
            value: Byte value to fill
            start: Start address (default 0)
            end: End address (default: full memory)
        """
        if end is None:
            end = self.size
        for addr in range(start, min(end, self.size)):
            self.data[addr] = value & 0xFF

    def load(self, data: bytes, start: int = 0) -> None:
        """
        Load bytes into memory starting at address.

        Args:
            data: Bytes to load
            start: Start address
        """
        for i, byte in enumerate(data):
            if start + i < self.size:
                self.data[start + i] = byte

    def dump(self, start: int, length: int) -> bytes:
        """
        Dump a region of memory as bytes.

        Args:
            start: Start address
            length: Number of bytes

        Returns:
            Memory contents as bytes
        """
        end = min(start + length, self.size)
        return bytes(self.data[start:end])

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"OctaveMemory(octave={self.octave}, size={self.size:,} bytes)"

    def __len__(self) -> int:
        return self.size
