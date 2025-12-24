# TriXc Binary Format Specification

Version 1.0

## Overview

The TriXc format (`.trixc`) is the portable binary format for frozen Neural Geometric Deterministic shapes. A `.trixc` file IS the chip - load it, run it.

## File Structure

```
┌─────────────────────────────────────────────────────────────┐
│                      HEADER (64 bytes)                       │
├─────────────────────────────────────────────────────────────┤
│                     SHAPE DATA (N bytes)                     │
├─────────────────────────────────────────────────────────────┤
│                      FOOTER (16 bytes)                       │
└─────────────────────────────────────────────────────────────┘
```

## Header Format (64 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | Magic | `"TRIX"` (0x54 0x52 0x49 0x58) |
| 4 | 1 | Version Major | Format major version (1) |
| 5 | 1 | Version Minor | Format minor version (0) |
| 6 | 1 | Chip Type | See Chip Type Values |
| 7 | 1 | Bits | Bit width (1, 8, 16, or 32) |
| 8 | 4 | Shape Size | Size of shape data in bytes (little-endian) |
| 12 | 4 | Reserved | Reserved for future use (zeros) |
| 16 | 48 | Name | Null-terminated UTF-8 chip name |

### Chip Type Values

| Value | Name | Description |
|-------|------|-------------|
| 1 | ATOM | Primitive operations (XOR, AND, ADD) |
| 2 | MOLECULE | Compositions (half adder, full adder) |
| 3 | PROTEIN | Complete functional units (6502 CPU) |
| 4 | PATHWAY | Multi-chip systems |

## Shape Data (N bytes)

The shape data contains the frozen polynomial coefficients, stored as signed 8-bit integers. The interpretation of coefficients depends on the chip type and polynomial structure.

For standard logic gates, the coefficients represent terms in the multilinear polynomial:

- **XOR**: `a + b - 2ab` → coefficients `[1, 1, -2]`
- **AND**: `ab` → coefficients `[0, 0, 1]`
- **OR**: `a + b - ab` → coefficients `[1, 1, -1]`

Coefficient encoding:
- Values 0-127: stored as-is
- Values -128 to -1: stored as two's complement (128-255)

## Footer Format (16 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 16 | Checksum | SHA-256 of (header + shape data), truncated to 16 bytes |

The checksum provides integrity verification. If the checksum doesn't match, the file may be corrupted.

## Example: XOR Gate

```
Offset  Hex                                              ASCII
──────  ───────────────────────────────────────────────  ─────
0000    54 52 49 58 01 00 01 01 03 00 00 00 00 00 00 00  TRIX............
0010    58 4F 52 00 00 00 00 00 00 00 00 00 00 00 00 00  XOR.............
0020    00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
0030    00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  ................
0040    01 01 FE                                         ... (coefficients: 1, 1, -2)
0043    XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX XX  (16-byte checksum)
```

Total size: 64 (header) + 3 (shape) + 16 (footer) = 83 bytes

## Validation

When loading a `.trixc` file:

1. Verify magic bytes are `"TRIX"`
2. Check version compatibility
3. Read shape size from header
4. Extract shape data
5. Compute SHA-256 of header + shape data
6. Compare first 16 bytes of hash with footer checksum

## Implementation Notes

### Reading

```python
from foundry.export.trixc import TriXcFormat
from pathlib import Path

spec = TriXcFormat.load(Path("xor.trixc"))
print(f"Name: {spec.name}")
print(f"Coefficients: {spec.coefficients}")
```

### Writing

```python
from foundry.export.trixc import TriXcFormat, ChipSpec, ChipType
from pathlib import Path

spec = ChipSpec(
    name="XOR",
    chip_type=ChipType.ATOM,
    bits=1,
    description="XOR gate",
    polynomial="a + b - 2ab",
    coefficients=[1, 1, -2],
    inputs=["a", "b"],
)

TriXcFormat.save(spec, Path("xor.trixc"))
```

## Future Versions

Reserved fields and version numbers allow for future extensions:

- Additional metadata (creation date, author, license)
- Extended coefficient formats (16-bit, 32-bit)
- Compression
- Encryption

Readers should check version numbers and fail gracefully on unsupported versions.
