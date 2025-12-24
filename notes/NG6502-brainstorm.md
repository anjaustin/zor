# NG6502 Brainstorm: P0 Gaps

*December 2025 - Infrastructure needed for real programs*

---

## Current State

We have:
- **Frozen Shapes**: 16 ALU operations, 0 learnable parameters
- **asm.py**: Immediate mode only (`LDA #$05`)
- **Registers**: A, X, Y, C, Z, N flags
- **shapes.py**: Function-call interface (`add(42, 13)`)

---

## The Gap

What's missing to make real programs:

| Component | Description | Priority |
|-----------|-------------|----------|
| **Memory** | 64KB RAM simulation | P0 |
| **Addressing Modes** | Absolute, zero-page, indexed, indirect | P0 |
| **Stack** | PHA, PLA, PHP, PLP, JSR, RTS | P0 |
| **Control Flow** | JMP, branches (BEQ, BNE, BCC, etc.) | P0 |

---

## 1. Memory

**Implementation:**
```python
class Memory:
    def __init__(self, size=65536):
        self.data = bytearray(size)

    def read(self, addr): return self.data[addr]
    def write(self, addr, val): self.data[addr] = val & 0xFF
```

**Key areas:**
| Address | Purpose |
|---------|---------|
| `$0000-$00FF` | Zero Page (fast access) |
| `$0100-$01FF` | Stack |
| `$0200-$FFFF` | General purpose / ROM |

**Open question:** Do we simulate ROM (read-only regions)?

---

## 2. Addressing Modes

The 6502 has 13 addressing modes. We have 1.

| Mode | Syntax | Effective Address |
|------|--------|-------------------|
| Immediate | `LDA #$05` | Value is operand ✓ HAVE |
| Zero Page | `LDA $05` | `mem[$05]` |
| Zero Page,X | `LDA $05,X` | `mem[($05 + X) & $FF]` |
| Zero Page,Y | `LDX $05,Y` | `mem[($05 + Y) & $FF]` |
| Absolute | `LDA $1234` | `mem[$1234]` |
| Absolute,X | `LDA $1234,X` | `mem[$1234 + X]` |
| Absolute,Y | `LDA $1234,Y` | `mem[$1234 + Y]` |
| Indirect | `JMP ($1234)` | `mem[word at $1234]` |
| (Indirect,X) | `LDA ($05,X)` | `mem[word at ($05+X)&$FF]` |
| (Indirect),Y | `LDA ($05),Y` | `mem[(word at $05) + Y]` |

**Key insight:** Addressing is pure routing. The **shapes** only activate when we actually compute (ADC, SBC, AND, etc.).

---

## 3. Stack

```
$01FF  ← SP starts here (empty stack)
$01FE  ← First push goes here
...
$0100  ← Stack bottom (full)
```

**Operations:**
| Opcode | Action |
|--------|--------|
| `PHA` | Push A: `mem[$0100 + SP] = A; SP--` |
| `PLA` | Pull A: `SP++; A = mem[$0100 + SP]` |
| `PHP` | Push flags (with B flag set) |
| `PLP` | Pull flags |
| `JSR $1234` | Push (PC+2) high then low, PC = $1234 |
| `RTS` | Pull low then high, PC = pulled + 1 |

**Key:** JSR/RTS enable subroutines. This is how we get modularity.

---

## 4. Control Flow

**Unconditional:**
| Opcode | Action |
|--------|--------|
| `JMP $1234` | PC = $1234 |
| `JMP ($1234)` | PC = word at $1234 (indirect) |

**Conditional branches (relative, -128 to +127):**
| Opcode | Condition |
|--------|-----------|
| `BEQ rel` | Z = 1 (equal/zero) |
| `BNE rel` | Z = 0 (not equal) |
| `BCC rel` | C = 0 (carry clear) |
| `BCS rel` | C = 1 (carry set) |
| `BMI rel` | N = 1 (minus/negative) |
| `BPL rel` | N = 0 (plus/positive) |
| `BVS rel` | V = 1 (overflow set) |
| `BVC rel` | V = 0 (overflow clear) |

**Key:** Branches use flags set by frozen shapes (CMP, ADC, etc.).

---

## Architecture Decision

Two approaches:

**A) Extend asm.py incrementally**
- Add memory to CPUState
- Add addressing modes to parser
- Add stack pointer
- Add new instructions
- Pros: One file, simple
- Cons: Gets messy, mixes assembler + executor

**B) Build fresh sim/ module**
- Clean separation: `asm.py` = assembler, `sim/` = execution
- More modular, easier to test
- Pros: Clean architecture
- Cons: More files

**Recommendation:** Option B - keep `asm.py` as the simple on-ramp, build `sim/` as the full system.

---

## Proposed Implementation Order

1. **Memory** - Foundation for everything
2. **Addressing Modes** - Makes memory useful
3. **Stack + JSR/RTS** - Enables subroutines
4. **Branches + JMP** - Enables loops and conditionals

Alternative: **Vertical slice** - one complete program that uses all four, then fill in gaps.

---

## The Core Insight

All of this is **routing infrastructure**. The actual computation still uses frozen shapes:

- `ADC $1234` → fetch from memory → RIPPLE_ADD shape
- `AND $05,X` → fetch from zero page + X → PARALLEL_AND shape
- `CMP #$10` → RIPPLE_SUB shape (for flags) → enables BEQ/BNE

> *"Computation is geometry. Infrastructure is routing."*

---

## Full 6502 Instruction Set Reference

For completeness, here's what a full implementation needs:

**Load/Store:** LDA, LDX, LDY, STA, STX, STY
**Transfer:** TAX, TXA, TAY, TYA, TSX, TXS
**Stack:** PHA, PHP, PLA, PLP
**Arithmetic:** ADC, SBC, INC, DEC, INX, DEX, INY, DEY
**Logic:** AND, ORA, EOR
**Shift:** ASL, LSR, ROL, ROR
**Compare:** CMP, CPX, CPY, BIT
**Branch:** BCC, BCS, BEQ, BMI, BNE, BPL, BVC, BVS
**Jump:** JMP, JSR, RTS, RTI, BRK
**Flags:** CLC, SEC, CLD, SED, CLI, SEI, CLV
**No-op:** NOP

56 official opcodes, ~150 opcode/addressing-mode combinations.

---

*Ready to build.*
