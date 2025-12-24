# You Just Ran prove_it.py

Or you're about to. Either way, let's talk about what happens.

---

## Run It

```bash
python examples/prove_it.py
```

Takes 10 seconds. No setup beyond having Python.

---

## What You'll See

### Part 1: Weird Functions Freeze

```
scramble: a XOR (a << 3)     256/256 (100.000000%)
twist: nibble_swap(a) XOR 0xA5   256/256 (100.000000%)
gray: a XOR (a >> 1)         256/256 (100.000000%)
reverse: reverse all bits    256/256 (100.000000%)
```

These aren't standard operations. They're deliberately strange. All freeze perfectly.

### Part 2: 100 Random Functions

```
Random functions: 100/100 achieved 100% accuracy
ALL RANDOM FUNCTIONS FROZE PERFECTLY.
```

We generate 100 completely arbitrary truth tables and freeze them all.

### Part 3: Your Function

You define a function. It freezes. 100%.

---

## The Claim

**Any deterministic function on bits can be frozen into a polynomial that computes it exactly.**

Not 99.9%. Exactly. 100.000000%.

---

## How It Works (30 seconds)

XOR can be written as a polynomial:

```
XOR(a, b) = a + b - 2ab
```

Plug in binary values:
- XOR(0, 0) = 0 + 0 - 0 = 0 ✓
- XOR(0, 1) = 0 + 1 - 0 = 1 ✓
- XOR(1, 0) = 1 + 0 - 0 = 1 ✓
- XOR(1, 1) = 1 + 1 - 2 = 0 ✓

This isn't an approximation. On binary inputs, the polynomial **IS** the XOR function.

Every deterministic function has a polynomial like this. That's the theorem.

---

## Why This Matters

Neural networks approximate. Frozen shapes compute exactly.

Embed exact computation into neural architectures:
- **Learned routing:** Which operation to use (adaptive)
- **Frozen execution:** How to compute it (exact, verified)

This is the neural-symbolic bridge.

---

## Still Skeptical?

Good. Try to break it:

```python
# In prove_it.py, enter increasingly weird lambdas:
lambda a: (a * 137 + 42) % 256
lambda a: ((a << 4) ^ (a >> 4)) & 0xFF
lambda a: sum([(a >> i) & 1 for i in range(8)]) * 31 % 256
```

They all freeze. Every one.

---

## Next Steps

| If you're... | Read this |
|--------------|-----------|
| Still skeptical | [HONEST_LIMITS.md](HONEST_LIMITS.md) - Where this fails |
| Curious why it matters | [WHY_CARE.md](WHY_CARE.md) - The implications |
| Ready to learn | [FRESHMAN.md](FRESHMAN.md) - Gentle tutorial |
| Want to code | [examples/my_first_freeze.py](../examples/my_first_freeze.py) |

---

*"Run it. Try to break it. Then we'll talk."*
