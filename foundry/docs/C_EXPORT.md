# C Code Export Specification

## Overview

The Foundry exports chips as portable C99 code with no external dependencies. Each chip produces two files:

- `trix_{name}.h` - Header with API declarations
- `trix_{name}.c` - Implementation with polynomial evaluation

## Generated API

### Header File Structure

```c
#ifndef TRIX_{NAME}_H
#define TRIX_{NAME}_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Chip state structure */
typedef struct {
    uint8_t coefficients[N];
    uint8_t state[16];
} trix_{name}_t;

/* Initialize chip */
void trix_{name}_init(trix_{name}_t *chip);

/* Compute function */
uint8_t trix_{name}(uint8_t a, uint8_t b);

/* Step function (for stateful chips) */
void trix_{name}_step(trix_{name}_t *chip);

#ifdef __cplusplus
}
#endif

#endif
```

### Source File Structure

```c
#include "trix_{name}.h"

/* Frozen coefficients */
static const int8_t COEFFICIENTS[N] = {
    /* coefficient values */
};

void trix_{name}_init(trix_{name}_t *chip) {
    /* Load coefficients into chip structure */
}

uint8_t trix_{name}(uint8_t a, uint8_t b) {
    /* Polynomial evaluation */
    return (uint8_t)(polynomial_expression);
}

void trix_{name}_step(trix_{name}_t *chip) {
    /* For stateful chips only */
}
```

## Polynomial Mappings

The exporter knows the polynomials for standard logic gates:

| Gate | Polynomial | C Expression |
|------|------------|--------------|
| XOR | `a + b - 2ab` | `a + b - 2*a*b` |
| AND | `ab` | `a * b` |
| OR | `a + b - ab` | `a + b - a*b` |
| NOT | `1 - a` | `1 - a` |
| NAND | `1 - ab` | `1 - a*b` |
| NOR | `1 - a - b + ab` | `1 - a - b + a*b` |
| XNOR | `1 - a - b + 2ab` | `1 - a - b + 2*a*b` |
| ADD | `a + b` | `a + b` |
| SUB | `a - b` | `a - b` |
| MUL | `ab` | `a * b` |

## Usage Examples

### Basic Usage

```c
#include "trix_xor.h"

int main() {
    // Direct computation
    uint8_t result = trix_xor(1, 0);  // Returns 1

    // With state structure
    trix_xor_t chip;
    trix_xor_init(&chip);

    return 0;
}
```

### Compilation

```bash
gcc -c trix_xor.c -o trix_xor.o
gcc main.c trix_xor.o -o program
```

### Integration

```c
// Include multiple chips
#include "trix_xor.h"
#include "trix_and.h"
#include "trix_or.h"

// Build a half adder from primitives
uint8_t half_adder_sum(uint8_t a, uint8_t b) {
    return trix_xor(a, b);
}

uint8_t half_adder_carry(uint8_t a, uint8_t b) {
    return trix_and(a, b);
}
```

## Configuration

The `ExportConfig` class controls export behavior:

```python
from foundry.export.c_export import CExporter, ExportConfig

config = ExportConfig(
    prefix="trix",           # Function name prefix
    include_runtime=True,    # Include runtime support
    target="generic",        # Target platform
)

header, source = CExporter.generate(spec, config)
```

### Prefix Customization

```python
config = ExportConfig(prefix="myproject")
# Generates: myproject_xor(), myproject_xor_t, etc.
```

## Type Mappings

| Bit Width | C Type |
|-----------|--------|
| 1 | `uint8_t` |
| 8 | `uint8_t` |
| 16 | `uint16_t` |
| 32 | `uint32_t` |

## Portability

Generated code is:

- **C99 compliant** - Works with any C99 compiler
- **No dependencies** - Only requires `<stdint.h>`
- **Self-contained** - Each chip is independent
- **Cross-platform** - Works on any architecture

## Target-Specific Runtimes (Planned)

Future versions will include optimized runtimes for:

- ESP32 (Xtensa LX6)
- Raspberry Pi (ARM)
- Arduino (AVR)
- x86 (SSE/AVX)
- FPGA (Verilog generation)
