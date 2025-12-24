# NAND

*NOT AND — The universal gate*

```
┌─────────────────────────────────────────────────────────────┐
│ NAND                                                        │
├─────────────────────────────────────────────────────────────┤
│ Kingdom: Logic                                              │
│ Type: Elemental                                             │
│ Arity: Binary                                               │
│ Frozen: Yes                                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Definition

### Mathematical

```
NAND(a, b) = ¬(a ∧ b) = 1 - a·b
```

### Prose

NAND outputs 0 only when both inputs are 1; otherwise it outputs 1. It's AND with an inverted output. NAND is **functionally complete** — any Boolean function can be constructed using only NAND gates.

---

## Visual

```
     ┌─────┐
a ───┤     │
     │ ⊼   ├─── ¬(a ∧ b)
b ───┤     │
     └─────┘

Standard logic gate symbol:

    a ──┐
        D>o── output
    b ──┘

The bubble indicates inversion of AND.
```

---

## Truth Table

| a | b | NAND |
|---|---|------|
| 0 | 0 | 1    |
| 0 | 1 | 1    |
| 1 | 0 | 1    |
| 1 | 1 | 0    |

---

## Examples

### Discrete

```
NAND(0, 0) = 1   # Not both → 1
NAND(0, 1) = 1   # Not both → 1
NAND(1, 0) = 1   # Not both → 1
NAND(1, 1) = 0   # Both → 0
```

### Continuous

```
NAND(0.0, 0.0) = 1.0
NAND(0.5, 0.5) = 0.75
NAND(0.8, 0.9) = 0.28
NAND(1.0, 1.0) = 0.0
```

---

## Implementation

### Python

```python
def nand(a: float, b: float) -> float:
    """Differentiable NAND for continuous inputs in [0, 1]."""
    return 1.0 - a * b

def nand_discrete(a: int, b: int) -> int:
    """Discrete NAND for boolean inputs."""
    return 1 - (a & b)
```

### C

```c
static inline float nand_shape(float a, float b) {
    return 1.0f - a * b;
}

static inline int nand_discrete(int a, int b) {
    return !(a & b);
}
```

---

## Relationships

### Built From

Conceptually: `NOT(AND(a, b))`

But we treat NAND as elemental due to its importance as a universal gate.

### Used In

- **All other gates** — NAND can construct any Boolean function
- **Flash memory** — NAND flash is named for its gate structure
- **Minimal gate sets** — Hardware often uses NAND-only designs

### See Also

- **[AND](and.md)** — NAND = NOT(AND)
- **[NOR](nor.md)** — The other universal gate
- **[NOT](not.md)** — NOT(a) = NAND(a, a)

---

## Universality

NAND can build any other gate:

```
NOT(a)    = NAND(a, a)
AND(a,b)  = NAND(NAND(a,b), NAND(a,b))
OR(a,b)   = NAND(NAND(a,a), NAND(b,b))
XOR(a,b)  = NAND(NAND(a,NAND(a,b)), NAND(b,NAND(a,b)))
```

This makes NAND **functionally complete**. With enough NAND gates, you can compute anything.

---

## Use Cases

1. **Universal Construction**: Any digital circuit can be built from NAND gates alone.

2. **Flash Memory**: NAND flash is the dominant storage technology — your SSD likely uses NAND cells.

3. **Minimal Hardware**: When optimizing for gate count, NAND-only designs reduce manufacturing complexity.

4. **Theoretical Foundations**: Proving NAND completeness is a cornerstone of computability theory.

---

## Properties

- **Commutative**: `NAND(a, b) = NAND(b, a)`
- **NOT via self**: `NAND(a, a) = NOT(a)`
- **Not associative**: `NAND(NAND(a, b), c) ≠ NAND(a, NAND(b, c))` in general

---

## The Universal Gate

NAND's universality is profound. It means that *one* gate type suffices for *all* computation. Every processor, every neural network, every digital system could (in principle) be built from NAND alone.

This is the power of a single, well-chosen primitive.

---

*"Like I told my last wife, I says, 'Honey, I never compute the same circuit twice.'"*
