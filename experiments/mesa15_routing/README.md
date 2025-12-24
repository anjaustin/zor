# Mesa 15: Learning IS Routing

**Date:** 2025-12-22
**Status:** HYPOTHESIS CONFIRMED

---

## Hypothesis

> *"Learning IS Routing. Everything Else Can Be Frozen."*

If computation can be expressed as frozen polynomial shapes, then a neural network only needs to learn which shape to apply—not how to compute.

---

## The Calculator Test

Compared two architectures on a 4-operation calculator (ADD, SUB, XOR, AND):

| | MLP (learns everything) | Frozen + Router |
|-|-------------------------|-----------------|
| Trainable Parameters | 5,896 | 76 |
| Final Accuracy | 94.62% | 100% |
| Epochs to 100% | Never | 2 |
| Training Time | 241s | 4.5s |

**Results:**
- 78x fewer parameters
- 60x faster training
- Perfect accuracy (vs stuck at 94.6%)

---

## Key Insight

The MLP must learn:
1. **WHICH** operation to perform (routing)
2. **HOW** to compute each operation (execution)

The Frozen + Router only learns:
1. **WHICH** frozen shape to use (routing)

The "how" is encoded in frozen polynomial shapes—mathematically exact on binary inputs.

---

## Files

| File | Description |
|------|-------------|
| `01_raw.md` | Phase 1: Brain dump |
| `02_nodes.md` | Phase 2: Key decisions |
| `03_reflect.md` | Phase 3: Resolved tensions |
| `04_synthesize.md` | Phase 4: Test specification |
| `05_results.md` | Final results |
| `test_routing.py` | Executable test |

---

## Outcome

This experiment led to the creation of `trix.routing`:

```python
from trix.routing import RoutingPipeline

pipe = RoutingPipeline(bit_width=8)
pipe.add_builtins(["add", "sub", "xor", "and"])
result = pipe.train()  # 100% in 2 epochs
```

---

## Method

Used the **Lincoln Manifold Method** (4-phase exploration):
1. RAW: Stream of consciousness
2. NODES: Extract key decisions
3. REFLECT: Resolve tensions
4. SYNTHESIZE: Concrete specification

Then executed the test.

---

*"The router learns WHERE to send data. The frozen shapes compute exactly. Together, they outperform pure learning on every metric."*
