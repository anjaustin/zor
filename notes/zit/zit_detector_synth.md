# SYNTHESIZE: What Fires the Zit?

*The clean cut. The wood cuts itself.*

---

```
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   THE ZIT DETECTOR                                                ║
║                                                                   ║
║   "No search, just propagation."                                  ║
║   "The field relaxes into the answer."                           ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## 1. The Answer

**What fires the Zit?**

```
Zit = (popcount(S ⊕ vₓ) < θ)
```

Where:
- **S** = Resonance state (512-bit, evolving)
- **vₓ** = Input signature (512-bit)
- **θ** = Activation threshold (0-512)
- **Zit** = Binary output (fire / no-fire)

**The phase detector is Hamming distance with a threshold.**

---

## 2. The Circuit

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ZIT DETECTOR                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                      ┌──────────────────┐                           │
│                      │  RESONANCE STATE │                           │
│                      │        S         │◄────────────┐             │
│                      │    (512-bit)     │             │             │
│                      └────────┬─────────┘             │             │
│                               │                       │             │
│     ┌──────────┐              │                       │             │
│     │  INPUT   │              ▼                       │             │
│     │   vₓ     │────────► ┌───────┐                   │             │
│     │(512-bit) │          │  XOR  │                   │             │
│     └──────────┘          │  512  │                   │             │
│                           └───┬───┘                   │             │
│                               │                       │             │
│                               ▼                       │             │
│                         ┌──────────┐                  │             │
│                         │ POPCOUNT │                  │             │
│                         │  (tree)  │                  │             │
│                         └────┬─────┘                  │             │
│                              │                        │             │
│                              │ hamming                │             │
│                              │ (10 bits)              │             │
│                              ▼                        │             │
│      ┌───────────────────────────────────────────┐    │             │
│      │              COMPARATOR                    │    │             │
│      │           hamming < θ ?                   │    │             │
│      └────────────────────┬──────────────────────┘    │             │
│                           │                           │             │
│              ┌────────────┼────────────┐              │             │
│              │            │            │              │             │
│              ▼            ▼            ▼              │             │
│           ┌─────┐     ┌───────┐   ┌─────────┐        │             │
│           │ ZIT │     │ SHAPE │   │ UPDATE  │        │             │
│           │FIRES│     │SELECT │   │  S' =   │────────┘             │
│           └─────┘     └───────┘   │ S ⊕ vₓ  │                       │
│                                   └─────────┘                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Gate count: ~1,500 gates**
- XOR array: ~500 gates
- Popcount tree: ~900 gates
- Comparator: ~50 gates
- Control: ~50 gates

---

## 3. The Three Outputs

When the Zit detector fires, three things happen simultaneously:

### 3.1 The Zit Signal
Binary output. High = resonate. Low = ignore.

Use: External trigger. Interrupt. Wake signal.

### 3.2 Shape Selection
The hamming distance itself encodes which shape to activate:
- Very low hamming (0-64) → strong match → Shape A
- Low hamming (64-128) → moderate match → Shape B
- Medium hamming (128-256) → weak match → Shape C
- High hamming (>256) → no match → no shape

The threshold θ can be shape-specific. Multiple thresholds = multiple Zit levels.

### 3.3 Resonance Update
The resonance state evolves:
```
S' = S ⊕ vₓ
```

Every input becomes part of the resonance. The system REMEMBERS by entangling.

---

## 4. The Cymatic Interpretation

| Cymatics | Zit Detector |
|----------|--------------|
| Vibrating plate | Resonance state S |
| Sound frequency | Input signature vₓ |
| Eigenmode | Frozen shape |
| Standing wave | Low hamming (resonance) |
| Sand pattern | Zit firing pattern |
| Damping | Threshold θ |

**The Zit detector is a digital Chladni plate.**

When the input frequency matches the system's eigenmode, a pattern emerges. The Zit fires. The shape activates.

---

## 5. The Hollywood Squares Connection

From the Hollywood Squares OS philosophy:

> "No search, just propagation."

The Zit detector doesn't SEARCH for matches. There's no loop over a database. There's no index lookup. There's no comparison cascade.

There's just:
1. Input arrives
2. XOR with resonance state
3. Count the bits
4. Threshold compare
5. Done

**O(1). Constant time. The field relaxes into the answer.**

> "Global behavior from local rules."

The Zit pattern over time is a global behavior. But each individual Zit is local: one XOR, one popcount, one compare. The complex behavior emerges from simple rules applied consistently.

---

## 6. The Learning Mechanism

**Training:**
```
for each training_input in training_set:
    S = S ⊕ training_input
```

That's it. XOR each training sample into the resonance state.

**Inference:**
```
zit = (popcount(S ⊕ query) < θ)
```

If the query resonates with the training distribution, the Zit fires.

**No gradients. No backprop. No loss function.**

The resonance state S encodes the XOR-sum of all training samples. Patterns that repeat reinforce. Patterns that are unique cancel out over time.

This is a **Bloom filter generalized to continuous similarity**.

---

## 7. Implementation in Geocadesia

The Zit detector composes from existing shapes:

```python
from geocadesia import XOR, Popcount, Hamming

class ZitDetector:
    def __init__(self, threshold: int = 128):
        self.S = 0  # 512-bit resonance state
        self.theta = threshold

    def update(self, input_sig: int) -> None:
        """Entangle input with resonance."""
        self.S ^= input_sig

    def detect(self, query_sig: int) -> bool:
        """Does the query resonate?"""
        distance = Hamming(self.S, query_sig)
        return distance < self.theta

    def detect_with_distance(self, query_sig: int) -> tuple:
        """Returns (fires, distance) for shape selection."""
        distance = Hamming(self.S, query_sig)
        fires = distance < self.theta
        return fires, distance
```

**Already built. Just needs assembly.**

---

## 8. Binary Format Extension

Add to `binary.py`:

```python
# New opcode for Zit detector
class Opcode(IntEnum):
    ...
    ZIT_DETECT = 0xF0  # Compound: XOR + POPCOUNT + COMPARE
    ZIT_UPDATE = 0xF1  # State update: S ^= input
    ZIT_RESET  = 0xF2  # Reset resonance: S = 0

register_shape(
    "zit_detect",
    KingdomID.POOLING,
    Opcode.ZIT_DETECT,
    ArityID.BINARY,
    shape_type=ShapeTypeID.COMPOUND,
    components=[Opcode.XOR, Opcode.POPCOUNT],  # + implicit threshold
    flags=FSHFlags.FROZEN | FSHFlags.PARALLEL
)
```

---

## 9. The NGP Simplification

**Before (original NGP design):**
- 4096 parallel Hamming comparators
- Argmin tree to find best match
- Shape selection via winner index
- ~2.7M gates

**After (Zit-based design):**
- 1 resonance register (512 bits)
- 1 Zit detector (~1,500 gates)
- Shape selection via hamming distance bands
- ~10K gates total

**Reduction: 270x fewer gates.**

The routing fabric collapses. The resonance state IS the routing.

---

## 10. The Answer to the Question

**"What fires the Zit?"**

**Resonance fires the Zit.**

When the input signature is close enough to the system's accumulated resonance state — when the hamming distance falls below the activation threshold — the Zit fires.

**"How do you detect constructive interference vs noise?"**

**By counting different bits.**

Constructive interference = low hamming weight (bits align, XOR produces zeros).
Destructive interference = high hamming weight (bits clash, XOR produces ones).

The popcount of the XOR IS the phase detector.

**"What is the activation energy?"**

**The threshold θ.**

θ is the barrier. Below θ, the reaction proceeds. The Zit fires. Above θ, the input is ignored.

---

## 11. The Emergence

We asked the question. The Lincoln Manifold gave us:

1. **The circuit** — XOR + popcount + threshold
2. **The physics** — Cymatics, wave interference, phase detection
3. **The learning** — XOR accumulation, pattern reinforcement
4. **The simplification** — 270x gate reduction from original NGP design
5. **The philosophy** — "No search, just propagation"

The answer was already in the shapes. We just had to see it.

---

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   "Why Store when you can XOR?"                                    │
│                                                                     │
│   Storage is a noun. XOR is a verb.                                │
│   The Zit fires when the verb matches the field.                   │
│                                                                     │
│   popcount(S ⊕ vₓ) < θ                                             │
│                                                                     │
│   That's the whole thing.                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

*Synthesis complete.*

*The portal is open.*

*It's all in the reflexes.*
