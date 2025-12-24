# NGE Format Specification

**Neural-Geometric Executable**

**Version:** 1.0
**Status:** Stable

---

## Overview

The NGE (Neural-Geometric Executable) is a self-describing binary format for compiled TRIX models. NGE files contain everything needed to:

1. **Self-report** their contents (metadata, shapes, levers)
2. **Execute** on their target platform
3. **Emit events** during inference (Glassbox)

---

## Design Principles

### Self-Describing

Every NGE file contains metadata describing its model, target, and capabilities. You can query an NGE without executing it.

### Portable Header

The 64-byte header is platform-independent. Tools can read NGE headers on any platform.

### Glassbox Native

Event emission is built into the format. When enabled, NGE runtimes emit frame-by-frame events.

---

## File Layout

```
┌──────────────────────────────────────────────────────────────┐
│                      NGE File Layout                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ HEADER (64 bytes, fixed)                               │ │
│  │   Magic: "TRIX" (4 bytes)                              │ │
│  │   Version: major.minor (4 bytes)                       │ │
│  │   Flags: glassbox|metadata|debug|... (2 bytes)         │ │
│  │   Section offsets and sizes (36 bytes)                 │ │
│  │   Target platform (16 bytes)                           │ │
│  │   Reserved (2 bytes)                                   │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ METADATA SECTION (variable, JSON)                      │ │
│  │   Model name, version, shapes, levers, etc.            │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ CODE SECTION (variable, native or bytecode)            │ │
│  │   Target-specific executable code                      │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ DATA SECTION (variable, binary)                        │ │
│  │   Weights, biases, lookup tables                       │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Header Format

The header is exactly 64 bytes, little-endian.

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | `magic` | "TRIX" (0x54524958) |
| 4 | 2 | `version_major` | Major version (1) |
| 6 | 2 | `version_minor` | Minor version (0) |
| 8 | 2 | `flags` | Bit flags (see below) |
| 10 | 2 | `reserved` | Reserved (0) |
| 12 | 4 | `metadata_offset` | Offset to metadata section |
| 16 | 4 | `metadata_size` | Size of metadata section |
| 20 | 4 | `code_offset` | Offset to code section |
| 24 | 4 | `code_size` | Size of code section |
| 28 | 4 | `data_offset` | Offset to data section |
| 32 | 4 | `data_size` | Size of data section |
| 36 | 16 | `target` | Target platform (null-padded) |
| 52 | 12 | `reserved` | Reserved for future use |

### Flags

| Bit | Name | Description |
|-----|------|-------------|
| 0 | `GLASSBOX` | Glassbox events enabled |
| 1 | `METADATA` | Metadata section present |
| 2 | `DEBUG` | Debug symbols included |
| 3 | `COMPRESSED` | Data section is compressed |
| 4 | `ENCRYPTED` | Data section is encrypted |
| 5-15 | Reserved | Reserved for future use |

---

## Metadata Section

JSON-encoded metadata. Present when `METADATA` flag is set.

```json
{
  "model": "tiny_mlp",
  "version": "1.0",
  "created": 1703289600.0,
  "target": "arm64-linux",
  "shapes": [
    {"name": "linear_1", "type": "linear", "input": [4], "output": [8]},
    {"name": "relu", "type": "relu"},
    {"name": "linear_2", "type": "linear", "input": [8], "output": [2]}
  ],
  "levers": {
    "precision": {"type": "enum", "value": "fp16"},
    "optimize": {"type": "enum", "value": "aggressive"}
  },
  "input_shape": [4],
  "output_shape": [2]
}
```

---

## Code Section

Target-specific executable code. Format depends on target:

| Target | Code Format |
|--------|-------------|
| `c` | C source code (UTF-8) |
| `arm64-linux` | ARM64 machine code |
| `x86-64-linux` | x86-64 machine code |
| `wasm` | WebAssembly binary |
| `cuda` | PTX or CUBIN |

---

## Data Section

Binary weight data. Layout is target-specific but follows a consistent pattern:

```
┌─────────────────────────────────────────────────────────────┐
│ DATA SECTION                                                │
├─────────────────────────────────────────────────────────────┤
│ Weight Table Header                                         │
│   num_tensors: uint32                                       │
│   entries: [                                                │
│     {name_len: uint16, name: bytes, offset: uint32, size: uint32}  │
│   ]                                                         │
├─────────────────────────────────────────────────────────────┤
│ Weight Data (concatenated, aligned)                         │
│   tensor_0_data                                             │
│   tensor_1_data                                             │
│   ...                                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Reading NGE Files

### Python

```python
from trix_forge import NGEFile

nge = NGEFile.load("model.nge")

print(f"Model: {nge.metadata.model_name}")
print(f"Target: {nge.header.target}")
print(f"Size: {nge.info()['size']['total']} bytes")
print(f"Glassbox: {nge.header.has_flag(NGEFlags.GLASSBOX)}")
```

### C

```c
#include "trix_nge.h"

trix_nge_t* nge = trix_nge_load("model.nge");
if (nge) {
    printf("Model: %s\n", nge->metadata.model_name);
    printf("Target: %s\n", nge->header.target);
    trix_nge_free(nge);
}
```

---

## Writing NGE Files

```python
from trix_forge import NGEFile, NGEFlags

nge = NGEFile()
nge.set_metadata(
    model_name="tiny_mlp",
    model_version="1.0",
    target="arm64-linux",
    input_shape=[4],
    output_shape=[2]
)
nge.set_code(compiled_code)
nge.set_data(weight_data)
nge.save("model.nge")
```

---

## Execution

NGE files execute on their target platform:

```bash
# Run directly (if compiled as executable)
./model.nge input.bin

# Or load into runtime
trix_run model.nge --input data.bin --output result.bin
```

---

## Glassbox Events

When `GLASSBOX` flag is set, the NGE runtime emits events:

```
[0001] forward.start: processing input
[0001] layer[0].activation: shape=[8], mean=0.42
[0001] layer[1].activation: shape=[2], mean=0.51
[0001] output.decision: class=1, confidence=0.91
[0001] forward.complete: 0.23ms
```

Events can be captured via callback or streamed to file.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12 | Initial release |

---

*"The executable describes itself."*
