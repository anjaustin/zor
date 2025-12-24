# THE FOUNDRY

A chip catalog with instant fabrication.

```
Browse → Select → Export → Use
```

## Overview

The Foundry is a terminal-based chip catalog where you SELECT pre-trained Neural Geometric Deterministic compute elements and EXPORT plug-and-play binaries for any platform.

No programming. No configuration. Browse → Select → Export → Use.

## Quick Start

```bash
# Run the demo
python -m foundry --demo
```

Output:
```
════════════════════════════════════════════════════════════
  T R I X   F O U N D R Y
  Watch computation become geometry.
════════════════════════════════════════════════════════════

  Library loaded: 14 chips

  ┌─ PROCESSORS ────────────────────────────────────────┐
  │   6502                    326 B  ✓ ready            │
  └──────────────────────────────────────────────────────┘

  ┌─ LOGIC ─────────────────────────────────────────────┐
  │   XOR                       3 B  ✓ ready            │
  │   AND                       3 B  ✓ ready            │
  │   ...                                               │
  └──────────────────────────────────────────────────────┘
```

## The Day One Library

### Tier 1: Processors (Proteins)

| Chip | Size | Description |
|------|------|-------------|
| 6502 | 326 B | The chip that started it all. Apple II. Commodore 64. NES. |

### Tier 2: Arithmetic (Atoms)

| Chip | Size | Polynomial |
|------|------|------------|
| 8-bit ADD | 2 B | `a + b` |
| 8-bit SUB | 2 B | `a - b` |
| 8-bit MUL | 3 B | `a * b` |

### Tier 3: Logic (Atoms)

| Chip | Size | Polynomial |
|------|------|------------|
| XOR | 3 B | `a + b - 2ab` |
| AND | 3 B | `ab` |
| OR | 3 B | `a + b - ab` |
| NOT | 1 B | `1 - a` |
| NAND | 4 B | `1 - ab` |
| NOR | 4 B | `1 - a - b + ab` |
| XNOR | 4 B | `1 - a - b + 2ab` |

### Tier 4: Molecules (Compositions)

| Chip | Size | Description |
|------|------|-------------|
| Half Adder | 6 B | sum = XOR(a,b), carry = AND(a,b) |
| Full Adder | 14 B | Full adder with carry in/out |
| 8-bit Ripple Adder | 48 B | 8-bit ripple carry adder |

## Export Formats

### TriXc Binary (.trixc)

The portable binary format for frozen shapes:

```
TRIXC FILE FORMAT v1.0

Header (64 bytes):
  Magic:      "TRIX" (4 bytes)
  Version:    1.0 (2 bytes)
  Type:       ATOM|MOLECULE|PROTEIN (1 byte)
  Bits:       8|16|32 (1 byte)
  Shape size: N bytes (4 bytes)
  Reserved:   4 bytes
  Name:       Null-terminated (48 bytes)

Shape Data (N bytes):
  Frozen polynomial coefficients

Footer (16 bytes):
  Checksum:   SHA-256 truncated
```

### C Code (.h + .c)

Portable C99 code with no external dependencies:

```c
#include "trix_xor.h"

uint8_t result = trix_xor(1, 0);  // Returns 1
```

### TriX 6502

Full processor export with pure polynomial geometry:

```python
from foundry.export.trix_6502 import TriX6502Exporter
from pathlib import Path

TriX6502Exporter.export(Path("./6502"))
# Creates: 6502/trix_6502.h, 6502/trix_6502.c
```

See [docs/TRIX_6502.md](docs/TRIX_6502.md) for full 6502 documentation.

## Python API

### Loading the Catalog

```python
from foundry.library.catalog import Catalog, Category

catalog = Catalog()

# List all chips
for entry in catalog:
    print(f"{entry.spec.name}: {entry.spec.size} bytes")

# Get by name
xor = catalog.get("xor")
print(xor.spec.polynomial)  # "a + b - 2*a*b"

# List by category
logic_gates = catalog.list_by_category(Category.LOGIC)

# Search
results = catalog.search("add")
```

### Exporting Chips

```python
from foundry.export.c_export import CExporter
from foundry.export.trixc import TriXcFormat
from pathlib import Path

# Get a chip
catalog = Catalog()
xor = catalog.get("xor")

# Export to C
header_path, source_path = CExporter.export(xor.spec, Path("./output"))

# Export to TriXc binary
TriXcFormat.save(xor.spec, Path("./output/xor.trixc"))
```

### Loading TriXc Files

```python
from foundry.export.trixc import TriXcFormat
from pathlib import Path

spec = TriXcFormat.load(Path("xor.trixc"))
print(f"Loaded: {spec.name}")
print(f"Coefficients: {spec.coefficients}")
```

## Package Structure

```
foundry/
├── __init__.py              # Package entry
├── __main__.py              # CLI entry (python -m foundry)
├── export/
│   ├── __init__.py
│   ├── trixc.py            # TriXc binary format
│   ├── c_export.py         # C code generator
│   └── trix_6502.py        # Pure TriX 6502 exporter
├── library/
│   ├── __init__.py
│   └── catalog.py          # Day One library (14 chips)
├── docs/
│   ├── TRIXC_FORMAT.md     # Binary format spec
│   ├── C_EXPORT.md         # C export spec
│   └── TRIX_6502.md        # 6502 documentation
└── tests/
    ├── __init__.py
    ├── test_trixc.py       # Binary format tests
    ├── test_c_export.py    # C exporter tests
    ├── test_catalog.py     # Catalog tests
    └── test_trix_6502.py   # 6502 exporter tests (31 tests)
```

## Testing

```bash
# Run all tests (93 tests)
python -m pytest foundry/tests/ -v

# Run specific test file
python -m pytest foundry/tests/test_trixc.py -v

# Run 6502 tests only
python -m pytest foundry/tests/test_trix_6502.py -v
```

## The Promise

A hobbyist opens the Foundry.

They see the 6502. They click it. They pick ESP32. They click Fabricate.

They have a working 6502 for their ESP32.

Total time: < 1 minute.
Total knowledge required: None.

---

*The wood cuts itself.*

*Watch computation become geometry.*
