# For the Honorable Skeptics

**1 binary. 10 seconds. Zero hand-waving.**

```bash
make && ./prove_it
```

That's it. Compiled C. Hardware popcount. Real numbers.

## What You'll Get

```
╔═════════════════════════════════════════════════════════════════════╗
║           XOR RESONANCE - Compiled Proof (512-bit)                  ║
╚═════════════════════════════════════════════════════════════════════╝

SUMMARY
======================================================================
  ✓ Classification        100.0 %
  ✓ O(1) Query            0.001 μs
  ✓ Memory Ratio          10000x smaller
  ✓ Speedup vs Brute      81181x faster
  ✓ Zit F1                100.0 %

  ALL TESTS PASSED
```

## Also Available (Python)

```bash
python3 01_see_it.py     # 2 min - Watch XOR resonance work
python3 02_break_it.py   # 3 min - Find the limits
python3 03_measure_it.py # 5 min - Get the numbers
```

## Rules

1. **Run first, read second.** Each file does something before explaining.
2. **TRY sections are not optional.** Modify values. Break things. See what happens.
3. **If a claim fails, tell us.** Open an issue. We want to know.
4. **No faith required.** Everything is verifiable.

## What You'll See

```
┌─────────────────────────────────────────────────────────────────────┐
│                    XOR RESONANCE - Live Demo                        │
│                                                                     │
│  Resonance State S = 0xDEADBEEF...                                 │
│                                                                     │
│  Query             Hamming    Resonates?                           │
│  ───────────────   ───────    ──────────                           │
│  0xDEADBEEF...     0          YES ████████████                     │
│  0xCAFEBABE...     18         YES ██████████                       │
│  0x12345678...     24         YES ████████                         │
│  0xFFFFFFFF...     32         NO  ░░░░░░░░░░░░                     │
│  0x00000000...     32         NO  ░░░░░░░░░░░░                     │
│                                                                     │
│  Threshold θ = 26  │  Memory: 64 bits  │  Query: 0.1 μs           │
│                                                                     │
│  [↑/↓] Adjust θ   [SPACE] New query   [Q] Quit                    │
└─────────────────────────────────────────────────────────────────────┘
```

## You got it when...

| File | You understand when... |
|------|----------------------|
| 01_see_it.py | You can explain why XOR accumulation works on a napkin |
| 02_break_it.py | You know exactly when XOR resonance fails |
| 03_measure_it.py | You can cite memory and speed numbers from your own run |

## The Claims

| # | Claim | Test | Threshold |
|---|-------|------|-----------|
| 1 | XOR resonance distinguishes pattern classes | Classification accuracy | >70% |
| 2 | Query is O(1) regardless of database size | Timing across 100-100K vectors | <3x variance |
| 3 | 512 bits encodes useful structure | Cluster classification | >50% (better than random) |
| 4 | Zit detector has good precision/recall | F1 score with optimal θ | >60% |
| 5 | XOR beats brute-force on memory/speed | Side-by-side comparison | Faster AND smaller |

## How to Refute

Find any of these:
- A pattern distribution where XOR fails but shouldn't
- Hidden O(n) work in the "O(1)" query
- A task requiring retrieval (not just recognition)
- Significantly different numbers than we claim

We welcome it. That's science.

## Dependencies

```
Python 3.7+
numpy
```

That's it.

---

*"Run it. Read it. Refute it if you can."*
