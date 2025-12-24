# NOR

*NOT OR — The other universal gate*

```
┌─────────────────────────────────────────────────────────────┐
│ NOR                                                         │
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
NOR(a, b) = ¬(a ∨ b) = 1 - (a + b - a·b) = (1-a)·(1-b)
```

Simplified:
```
NOR(a, b) = 1 - a - b + a·b
```

### Prose

NOR outputs 1 only when both inputs are 0; otherwise it outputs 0. It's OR with an inverted output. Like NAND, NOR is **functionally complete**.

---

## Visual

```
     ┌─────┐
a ───┤     │
     │ ⊽   ├─── ¬(a ∨ b)
b ───┤     │
     └─────┘

Standard logic gate symbol:

    a ──┐
        )>o── output
    b ──┘

The bubble indicates inversion of OR.
```

---

## Truth Table

| a | b | NOR |
|---|---|-----|
| 0 | 0 | 1   |
| 0 | 1 | 0   |
| 1 | 0 | 0   |
| 1 | 1 | 0   |

---

## Examples

### Discrete

```
NOR(0, 0) = 1   # Neither → 1
NOR(0, 1) = 0   # One → 0
NOR(1, 0) = 0   # One → 0
NOR(1, 1) = 0   # Both → 0
```

### Continuous

```
NOR(0.0, 0.0) = 1.0
NOR(0.5, 0.5) = 0.25
NOR(0.0, 1.0) = 0.0
NOR(1.0, 1.0) = 0.0
```

---

## Implementation

### Python

```python
def nor(a: float, b: float) -> float:
    """Differentiable NOR for continuous inputs in [0, 1]."""
    return (1.0 - a) * (1.0 - b)

def nor_discrete(a: int, b: int) -> int:
    """Discrete NOR for boolean inputs."""
    return 1 - (a | b)
```

### C

```c
static inline float nor_shape(float a, float b) {
    return (1.0f - a) * (1.0f - b);
}

static inline int nor_discrete(int a, int b) {
    return !(a | b);
}
```

---

## Relationships

### Built From

Conceptually: `NOT(OR(a, b))`

But we treat NOR as elemental due to its importance as a universal gate.

### Used In

- **All other gates** — NOR can construct any Boolean function
- **Early computers** — Apollo Guidance Computer used NOR gates exclusively
- **Minimal gate sets** — Alternative to NAND-only designs

### See Also

- **[OR](or.md)** — NOR = NOT(OR)
- **[NAND](nand.md)** — The other universal gate
- **[NOT](not.md)** — NOT(a) = NOR(a, a)

---

## Universality

NOR can build any other gate:

```
NOT(a)    = NOR(a, a)
OR(a,b)   = NOR(NOR(a,b), NOR(a,b))
AND(a,b)  = NOR(NOR(a,a), NOR(b,b))
```

Like NAND, NOR is **functionally complete**.

---

## Use Cases

1. **Universal Construction**: Any digital circuit can be built from NOR gates alone.

2. **Apollo Guidance Computer**: NASA chose NOR gates for reliability — the AGC used about 5,600 NOR gates.

3. **De Morgan Dual**: NOR is the De Morgan dual of NAND. Any NAND circuit can be transformed to NOR.

---

## Properties

- **Commutative**: `NOR(a, b) = NOR(b, a)`
- **NOT via self**: `NOR(a, a) = NOT(a)`
- **De Morgan**: `NOR(a, b) = AND(NOT(a), NOT(b))`

---

## Historical Note

The Apollo Guidance Computer (1966) was built entirely from NOR gates. This wasn't because NOR was better than NAND — both are universal — but because the Fairchild integrated circuits available at the time implemented NOR.

The AGC guided astronauts to the Moon using just:
- ~5,600 NOR gates
- 72 KB of ROM
- 4 KB of RAM

Proof that simple, frozen computation can do extraordinary things.

---

*"Everybody relax, I'm here."*
