# Mesa 15: NG6502 - Next Generation 6502 Framework

*Vision document for a complete 6502 development environment on frozen geometry.*

---

## Overview

Build a framework where users can create full programs - even an OS - on top of frozen shapes. Every computation remains pure geometry (`XOR = a + b - 2ab`), but we add the infrastructure to route complex paths through that geometry.

---

## Current State (Mesa 14)

What we have:
- **Frozen Shapes**: 16 ALU operations, 0 learnable parameters
- **asm.py**: Immediate mode assembler (`LDA #$05`)
- **Registers**: A, X, Y, C, Z, N flags
- **shapes.py**: Function-call interface (`add(42, 13)`)

---

## The Gap (P0)

What's missing to make real programs:

| Component | Description | Priority |
|-----------|-------------|----------|
| **Memory** | 64KB RAM simulation | P0 |
| **Addressing Modes** | Absolute, zero-page, indexed, indirect | P0 |
| **Stack** | PHA, PLA, PHP, PLP, JSR, RTS | P0 |
| **Control Flow** | JMP, branches (BEQ, BNE, BCC, etc.) | P0 |

---

## Vision: Three Layers

### Layer 1: Full CPU Simulation

```python
from trix.sim import Machine

machine = Machine(ram=64*1024)
machine.load(binary, at=0x0800)
machine.run(start=0x0800)

# Inspect state
print(machine.A, machine.X, machine.Y)
print(machine.memory[0x0200:0x0210])
```

**Components:**
- `Memory` - 64KB address space, read/write
- `CPU` - Full instruction set, all addressing modes
- `Devices` - Virtual hardware (screen, keyboard, storage)

### Layer 2: Macro Assembler + Toolchain

```python
from trix.asm import assemble, link

# Modules with imports/exports
math_mod = assemble("""
.module math
.export multiply

multiply:
    ; A * X -> A (low), Y (high)
    ; Uses frozen shapes for all arithmetic
    RTS
""")

main = assemble("""
.import multiply from math

.org $0800
start:
    LDA #$07
    LDX #$06
    JSR multiply
    BRK
""")

binary = link(main, math_mod)
```

**Components:**
- `Assembler` - Full 6502 instruction set, macros, labels
- `Linker` - Combine modules, resolve imports
- `Stdlib` - Standard library (math, string, I/O)

### Layer 3: OS Components

```
os/
├── kernel.asm      # Core: vectors, IRQ handling, memory map
├── shell.asm       # Command interpreter
├── fs.asm          # Simple filesystem
└── drivers/
    ├── console.asm # Text I/O
    └── storage.asm # Block device
```

**The OS would be:**
- **Deterministic** - Same input, same output, always
- **Verifiable** - Every operation is pure geometry
- **Portable** - Could export to ONNX?

---

## Proposed File Structure

```
src/trix/
├── shapes.py           # [EXISTS] Function-call shapes
├── asm.py              # [EXISTS] Basic assembler
├── sim/                # [NEW] Simulation layer
│   ├── __init__.py
│   ├── memory.py       # 64KB RAM
│   ├── cpu.py          # Full 6502 CPU
│   ├── devices.py      # Virtual hardware
│   └── machine.py      # Unified interface
├── asm/                # [NEW] Full toolchain
│   ├── __init__.py
│   ├── assembler.py    # Macro assembler
│   ├── linker.py       # Module linker
│   ├── parser.py       # ASM parser
│   └── stdlib/         # Standard library
│       ├── math.asm
│       ├── string.asm
│       └── io.asm
└── os/                 # [NEW] OS components
    ├── kernel.asm
    ├── shell.asm
    └── fs.asm
```

---

## Key Insight: Shapes Stay Frozen

Even with a full OS:
- Every ADD is still chained full-adders
- Every XOR is still `a + b - 2ab`
- Every AND is still `ab`

The geometry doesn't change. We're building **routing infrastructure** on top of **frozen computation**.

> *"Computation is geometry. OS is routing."*

---

## UX Goals

### Beginner (30 seconds)
```python
from trix import add
add(5, 3)  # 8
```

### Intermediate (5 minutes)
```python
from trix.asm import run
run("""
    LDA #$05
    ADC #$03
""")
```

### Advanced (1 hour)
```python
from trix.sim import Machine
from trix.asm import assemble

prog = assemble("""
.org $0800
    LDX #$00
loop:
    TXA
    STA $0200,X
    INX
    BNE loop
    BRK
""")

m = Machine()
m.load(prog, 0x0800)
m.run(0x0800)
# Memory $0200-$02FF now contains 0-255
```

### Expert (1 day)
```bash
# Build and run an OS
trix-asm kernel.asm -o kernel.bin
trix-asm shell.asm -o shell.bin
trix-link kernel.bin shell.bin -o os.bin
trix-run os.bin
```

---

## Success Criteria

- [ ] Full 6502 instruction set (56 opcodes, all addressing modes)
- [ ] 64KB memory simulation
- [ ] Stack operations (JSR/RTS, PHA/PLA)
- [ ] Branch instructions (BEQ, BNE, BCC, BCS, etc.)
- [ ] Macro assembler with labels, imports, exports
- [ ] Module linker
- [ ] Standard library
- [ ] At least one working "OS" (boot + shell)
- [ ] All computation still uses frozen shapes

---

## Open Questions

1. **How to handle I/O?** Virtual devices? Callbacks? Memory-mapped?
2. **Interrupts?** Do we need IRQ/NMI for an OS?
3. **ONNX export?** Can we export a running program as ONNX?
4. **Debugging?** Step-through, breakpoints, memory watch?
5. **Performance?** Is Python fast enough, or do we need Rust/C?

---

*Mesa 15: Where frozen geometry becomes a living system.*
