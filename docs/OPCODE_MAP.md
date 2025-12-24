# 6502 Opcode Mapping for Frozen Shapes

This document provides the complete mapping from 6502 opcodes to frozen shapes and routing configurations.

---

## Quick Reference

### Shape IDs

| ID | Shape | Used By |
|----|-------|---------|
| 0 | `ripple_add` | ADC |
| 1 | `ripple_sub` | SBC, CMP, CPX, CPY |
| 2 | `parallel_and` | AND |
| 3 | `parallel_or` | ORA |
| 4 | `parallel_xor` | EOR |
| 5 | `shift_left` | ASL |
| 6 | `shift_right` | LSR |
| 7 | `rotate_left` | ROL |
| 8 | `rotate_right` | ROR |
| 9 | `increment` | INC, INX, INY |
| 10 | `decrement` | DEC, DEX, DEY |
| 11 | `transfer` | TAX, TXA, TAY, TYA, TSX, TXS, PHA, PLA, PHP, PLP |
| 12 | `load` | LDA, LDX, LDY |
| 13 | `store` | STA, STX, STY |
| 14 | `bit_test` | BIT |
| 15 | `identity` | NOP, branches, jumps, flag ops |

### Register IDs

| ID | Register | Purpose |
|----|----------|---------|
| 0 | A | Accumulator |
| 1 | X | X index |
| 2 | Y | Y index |
| 3 | SP | Stack pointer |
| 4 | PC_LO | Program counter low |
| 5 | PC_HI | Program counter high |
| 6 | P | Processor status |
| 7 | MEM | Memory operand |

### Flag Mask

| Index | Flag | Bit in P |
|-------|------|----------|
| 0 | N | 7 |
| 1 | Z | 1 |
| 2 | C | 0 |
| 3 | V | 6 |

---

## Complete Opcode Table

### Arithmetic Operations

| Mnemonic | Shape | In A | In B | Out | Carry | Flags | Description |
|----------|-------|------|------|-----|-------|-------|-------------|
| ADC | 0 (add) | A | MEM | A | Yes | NZCV | Add with carry |
| SBC | 1 (sub) | A | MEM | A | Yes | NZCV | Subtract with carry |

### Compare Operations

| Mnemonic | Shape | In A | In B | Out | Carry | Flags | Description |
|----------|-------|------|------|-----|-------|-------|-------------|
| CMP | 1 (sub) | A | MEM | - | Yes | NZC | Compare A with memory |
| CPX | 1 (sub) | X | MEM | - | Yes | NZC | Compare X with memory |
| CPY | 1 (sub) | Y | MEM | - | Yes | NZC | Compare Y with memory |

*Note: Compare operations discard the result, only flags are updated.*

### Logic Operations

| Mnemonic | Shape | In A | In B | Out | Carry | Flags | Description |
|----------|-------|------|------|-----|-------|-------|-------------|
| AND | 2 (and) | A | MEM | A | No | NZ | Logical AND |
| ORA | 3 (or) | A | MEM | A | No | NZ | Logical OR |
| EOR | 4 (xor) | A | MEM | A | No | NZ | Exclusive OR |
| BIT | 14 (bit) | A | MEM | - | No | NZV | Test bits |

*Note: BIT sets N and V from memory bits 7 and 6, Z from AND result.*

### Shift Operations (Accumulator)

| Mnemonic | Shape | In A | In B | Out | Carry | Flags | Description |
|----------|-------|------|------|-----|-------|-------|-------------|
| ASL A | 5 (shl) | A | - | A | No | NZC | Shift left |
| LSR A | 6 (shr) | A | - | A | No | NZC | Shift right |
| ROL A | 7 (rol) | A | - | A | Yes | NZC | Rotate left through C |
| ROR A | 8 (ror) | A | - | A | Yes | NZC | Rotate right through C |

### Shift Operations (Memory)

| Mnemonic | Shape | In A | In B | Out | Carry | Flags | Description |
|----------|-------|------|------|-----|-------|-------|-------------|
| ASL mem | 5 (shl) | MEM | - | MEM | No | NZC | Shift left |
| LSR mem | 6 (shr) | MEM | - | MEM | No | NZC | Shift right |
| ROL mem | 7 (rol) | MEM | - | MEM | Yes | NZC | Rotate left through C |
| ROR mem | 8 (ror) | MEM | - | MEM | Yes | NZC | Rotate right through C |

### Increment/Decrement

| Mnemonic | Shape | In A | In B | Out | Carry | Flags | Description |
|----------|-------|------|------|-----|-------|-------|-------------|
| INC | 9 (inc) | MEM | - | MEM | No | NZ | Increment memory |
| DEC | 10 (dec) | MEM | - | MEM | No | NZ | Decrement memory |
| INX | 9 (inc) | X | - | X | No | NZ | Increment X |
| DEX | 10 (dec) | X | - | X | No | NZ | Decrement X |
| INY | 9 (inc) | Y | - | Y | No | NZ | Increment Y |
| DEY | 10 (dec) | Y | - | Y | No | NZ | Decrement Y |

### Transfer Operations

| Mnemonic | Shape | In A | In B | Out | Carry | Flags | Description |
|----------|-------|------|------|-----|-------|-------|-------------|
| TAX | 11 (xfer) | A | - | X | No | NZ | A → X |
| TXA | 11 (xfer) | X | - | A | No | NZ | X → A |
| TAY | 11 (xfer) | A | - | Y | No | NZ | A → Y |
| TYA | 11 (xfer) | Y | - | A | No | NZ | Y → A |
| TSX | 11 (xfer) | SP | - | X | No | NZ | SP → X |
| TXS | 11 (xfer) | X | - | SP | No | - | X → SP |

### Load Operations

| Mnemonic | Shape | In A | In B | Out | Carry | Flags | Description |
|----------|-------|------|------|-----|-------|-------|-------------|
| LDA | 12 (load) | MEM | - | A | No | NZ | Load A from memory |
| LDX | 12 (load) | MEM | - | X | No | NZ | Load X from memory |
| LDY | 12 (load) | MEM | - | Y | No | NZ | Load Y from memory |

### Store Operations

| Mnemonic | Shape | In A | In B | Out | Carry | Flags | Description |
|----------|-------|------|------|-----|-------|-------|-------------|
| STA | 13 (store) | A | - | MEM | No | - | Store A to memory |
| STX | 13 (store) | X | - | MEM | No | - | Store X to memory |
| STY | 13 (store) | Y | - | MEM | No | - | Store Y to memory |

### Stack Operations

| Mnemonic | Shape | In A | In B | Out | Carry | Flags | Description |
|----------|-------|------|------|-----|-------|-------|-------------|
| PHA | 11 (xfer) | A | - | MEM* | No | - | Push A |
| PLA | 11 (xfer) | MEM* | - | A | No | NZ | Pull A |
| PHP | 11 (xfer) | P | - | MEM* | No | - | Push P |
| PLP | 11 (xfer) | MEM* | - | P | No | All | Pull P |

*MEM in stack ops refers to stack memory at SP.*

### Branch Operations

| Mnemonic | Shape | Condition | Description |
|----------|-------|-----------|-------------|
| BEQ | 15 (id) | Z = 1 | Branch if equal |
| BNE | 15 (id) | Z = 0 | Branch if not equal |
| BCS | 15 (id) | C = 1 | Branch if carry set |
| BCC | 15 (id) | C = 0 | Branch if carry clear |
| BMI | 15 (id) | N = 1 | Branch if minus |
| BPL | 15 (id) | N = 0 | Branch if plus |
| BVS | 15 (id) | V = 1 | Branch if overflow set |
| BVC | 15 (id) | V = 0 | Branch if overflow clear |

*Branches use identity shape; PC manipulation is separate.*

### Jump/Subroutine Operations

| Mnemonic | Shape | Description |
|----------|-------|-------------|
| JMP | 15 (id) | Jump to address |
| JSR | 15 (id) | Jump to subroutine (push PC) |
| RTS | 15 (id) | Return from subroutine (pull PC) |
| RTI | 15 (id) | Return from interrupt (pull P, PC) |
| BRK | 15 (id) | Software interrupt |

### Flag Operations

| Mnemonic | Shape | Flags | Description |
|----------|-------|-------|-------------|
| CLC | 15 (id) | C=0 | Clear carry |
| SEC | 15 (id) | C=1 | Set carry |
| CLD | 15 (id) | D=0 | Clear decimal |
| SED | 15 (id) | D=1 | Set decimal |
| CLI | 15 (id) | I=0 | Clear interrupt disable |
| SEI | 15 (id) | I=1 | Set interrupt disable |
| CLV | 15 (id) | V=0 | Clear overflow |

### Other

| Mnemonic | Shape | Description |
|----------|-------|-------------|
| NOP | 15 (id) | No operation |

---

## Implementation Dictionary

Python dictionary for initializing the meaning layer:

```python
OPCODE_SPECS = {
    # Arithmetic
    'ADC': {'shape': 0, 'in_a': 0, 'in_b': 7, 'out': 0, 'carry': True, 'flags': [1,1,1,1]},
    'SBC': {'shape': 1, 'in_a': 0, 'in_b': 7, 'out': 0, 'carry': True, 'flags': [1,1,1,1]},

    # Compare
    'CMP': {'shape': 1, 'in_a': 0, 'in_b': 7, 'out': -1, 'carry': True, 'flags': [1,1,1,0]},
    'CPX': {'shape': 1, 'in_a': 1, 'in_b': 7, 'out': -1, 'carry': True, 'flags': [1,1,1,0]},
    'CPY': {'shape': 1, 'in_a': 2, 'in_b': 7, 'out': -1, 'carry': True, 'flags': [1,1,1,0]},

    # Logic
    'AND': {'shape': 2, 'in_a': 0, 'in_b': 7, 'out': 0, 'carry': False, 'flags': [1,1,0,0]},
    'ORA': {'shape': 3, 'in_a': 0, 'in_b': 7, 'out': 0, 'carry': False, 'flags': [1,1,0,0]},
    'EOR': {'shape': 4, 'in_a': 0, 'in_b': 7, 'out': 0, 'carry': False, 'flags': [1,1,0,0]},
    'BIT': {'shape': 14, 'in_a': 0, 'in_b': 7, 'out': -1, 'carry': False, 'flags': [1,1,0,1]},

    # Shifts (Accumulator)
    'ASL_A': {'shape': 5, 'in_a': 0, 'in_b': 0, 'out': 0, 'carry': False, 'flags': [1,1,1,0]},
    'LSR_A': {'shape': 6, 'in_a': 0, 'in_b': 0, 'out': 0, 'carry': False, 'flags': [1,1,1,0]},
    'ROL_A': {'shape': 7, 'in_a': 0, 'in_b': 0, 'out': 0, 'carry': True, 'flags': [1,1,1,0]},
    'ROR_A': {'shape': 8, 'in_a': 0, 'in_b': 0, 'out': 0, 'carry': True, 'flags': [1,1,1,0]},

    # Shifts (Memory)
    'ASL_M': {'shape': 5, 'in_a': 7, 'in_b': 7, 'out': 7, 'carry': False, 'flags': [1,1,1,0]},
    'LSR_M': {'shape': 6, 'in_a': 7, 'in_b': 7, 'out': 7, 'carry': False, 'flags': [1,1,1,0]},
    'ROL_M': {'shape': 7, 'in_a': 7, 'in_b': 7, 'out': 7, 'carry': True, 'flags': [1,1,1,0]},
    'ROR_M': {'shape': 8, 'in_a': 7, 'in_b': 7, 'out': 7, 'carry': True, 'flags': [1,1,1,0]},

    # Inc/Dec
    'INC': {'shape': 9, 'in_a': 7, 'in_b': 7, 'out': 7, 'carry': False, 'flags': [1,1,0,0]},
    'DEC': {'shape': 10, 'in_a': 7, 'in_b': 7, 'out': 7, 'carry': False, 'flags': [1,1,0,0]},
    'INX': {'shape': 9, 'in_a': 1, 'in_b': 1, 'out': 1, 'carry': False, 'flags': [1,1,0,0]},
    'DEX': {'shape': 10, 'in_a': 1, 'in_b': 1, 'out': 1, 'carry': False, 'flags': [1,1,0,0]},
    'INY': {'shape': 9, 'in_a': 2, 'in_b': 2, 'out': 2, 'carry': False, 'flags': [1,1,0,0]},
    'DEY': {'shape': 10, 'in_a': 2, 'in_b': 2, 'out': 2, 'carry': False, 'flags': [1,1,0,0]},

    # Transfer
    'TAX': {'shape': 11, 'in_a': 0, 'in_b': 0, 'out': 1, 'carry': False, 'flags': [1,1,0,0]},
    'TXA': {'shape': 11, 'in_a': 1, 'in_b': 1, 'out': 0, 'carry': False, 'flags': [1,1,0,0]},
    'TAY': {'shape': 11, 'in_a': 0, 'in_b': 0, 'out': 2, 'carry': False, 'flags': [1,1,0,0]},
    'TYA': {'shape': 11, 'in_a': 2, 'in_b': 2, 'out': 0, 'carry': False, 'flags': [1,1,0,0]},
    'TSX': {'shape': 11, 'in_a': 3, 'in_b': 3, 'out': 1, 'carry': False, 'flags': [1,1,0,0]},
    'TXS': {'shape': 11, 'in_a': 1, 'in_b': 1, 'out': 3, 'carry': False, 'flags': [0,0,0,0]},

    # Load
    'LDA': {'shape': 12, 'in_a': 7, 'in_b': 7, 'out': 0, 'carry': False, 'flags': [1,1,0,0]},
    'LDX': {'shape': 12, 'in_a': 7, 'in_b': 7, 'out': 1, 'carry': False, 'flags': [1,1,0,0]},
    'LDY': {'shape': 12, 'in_a': 7, 'in_b': 7, 'out': 2, 'carry': False, 'flags': [1,1,0,0]},

    # Store
    'STA': {'shape': 13, 'in_a': 0, 'in_b': 0, 'out': 7, 'carry': False, 'flags': [0,0,0,0]},
    'STX': {'shape': 13, 'in_a': 1, 'in_b': 1, 'out': 7, 'carry': False, 'flags': [0,0,0,0]},
    'STY': {'shape': 13, 'in_a': 2, 'in_b': 2, 'out': 7, 'carry': False, 'flags': [0,0,0,0]},

    # Stack
    'PHA': {'shape': 11, 'in_a': 0, 'in_b': 0, 'out': 7, 'carry': False, 'flags': [0,0,0,0]},
    'PLA': {'shape': 11, 'in_a': 7, 'in_b': 7, 'out': 0, 'carry': False, 'flags': [1,1,0,0]},
    'PHP': {'shape': 11, 'in_a': 6, 'in_b': 6, 'out': 7, 'carry': False, 'flags': [0,0,0,0]},
    'PLP': {'shape': 11, 'in_a': 7, 'in_b': 7, 'out': 6, 'carry': False, 'flags': [1,1,1,1]},

    # Flag ops (identity shape, direct flag manipulation)
    'CLC': {'shape': 15, 'in_a': 0, 'in_b': 0, 'out': -1, 'carry': False, 'flags': [0,0,1,0], 'set_c': 0},
    'SEC': {'shape': 15, 'in_a': 0, 'in_b': 0, 'out': -1, 'carry': False, 'flags': [0,0,1,0], 'set_c': 1},
    'CLV': {'shape': 15, 'in_a': 0, 'in_b': 0, 'out': -1, 'carry': False, 'flags': [0,0,0,1], 'set_v': 0},

    # NOP
    'NOP': {'shape': 15, 'in_a': 0, 'in_b': 0, 'out': -1, 'carry': False, 'flags': [0,0,0,0]},
}
```

---

## Addressing Modes

The frozen shapes handle computation. Addressing modes handle operand fetching.

| Mode | Syntax | Operand Source |
|------|--------|----------------|
| Immediate | `#$nn` | Next byte after opcode |
| Zero Page | `$nn` | Memory at address $00nn |
| Zero Page,X | `$nn,X` | Memory at address ($00nn + X) |
| Zero Page,Y | `$nn,Y` | Memory at address ($00nn + Y) |
| Absolute | `$nnnn` | Memory at address $nnnn |
| Absolute,X | `$nnnn,X` | Memory at address ($nnnn + X) |
| Absolute,Y | `$nnnn,Y` | Memory at address ($nnnn + Y) |
| Indirect | `($nnnn)` | Memory at address stored at $nnnn |
| (Indirect,X) | `($nn,X)` | Memory at address stored at ($nn + X) |
| (Indirect),Y | `($nn),Y` | Memory at (address stored at $nn) + Y |
| Accumulator | `A` | Accumulator register |
| Implied | - | No operand |
| Relative | `$nn` | PC + signed offset (branches) |

Addressing mode decoding is a **separate routing layer** that determines how to populate the MEM register before the frozen shape executes.

---

## Summary Statistics

| Category | Count | Shapes Used |
|----------|-------|-------------|
| Arithmetic | 2 | add, sub |
| Compare | 3 | sub |
| Logic | 4 | and, or, xor, bit |
| Shift/Rotate | 8 | shl, shr, rol, ror |
| Inc/Dec | 6 | inc, dec |
| Transfer | 6 | transfer |
| Load/Store | 6 | load, store |
| Stack | 4 | transfer |
| Branch | 8 | identity |
| Jump | 5 | identity |
| Flag | 7 | identity |
| **Total** | **56** | **16 shapes** |

---

*Each opcode is just an address into frozen geometry.*
