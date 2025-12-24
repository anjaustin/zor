# TriX Test Suite

*Ensuring the architecture provides itself - reliably.*

**Total Tests:** 400+
**Rigorous Tests:** 67
**Providence Tests:** 237

---

## Quick Start

```bash
# From project root
cd /path/to/TriXO

# Run quick smoke test (~30 seconds)
python scripts/run_tests.py quick

# Run all Providence tests
python scripts/run_tests.py providence

# Run rigorous stress tests
python scripts/run_tests.py rigorous

# Run everything
python scripts/run_tests.py
```

---

## Test Suites

| Suite | Tests | Time | Description |
|-------|-------|------|-------------|
| `quick` | ~3 | 30s | Smoke test - verify basic functionality |
| `core` | ~80 | 1min | Core TriX architecture |
| `providence` | 237 | 2min | Providence unified architecture |
| `rigorous` | 67 | 20s | Stress tests, edge cases, invariants |
| `frozen` | ~100 | 1min | Frozen shapes and 6502 |
| `memory` | ~40 | 30s | Octave and Providence memory |
| `all` | 400+ | 5min | Everything |

### Running Individual Suites

```bash
# Quick verification that imports work
python scripts/run_tests.py quick

# After making changes to core architecture
python scripts/run_tests.py core

# After making changes to Providence
python scripts/run_tests.py providence

# Before committing - run rigorous tests
python scripts/run_tests.py rigorous
```

---

## Test Categories

### Rigorous Tests (`test_providence_rigorous.py`)

67 tests across 8 categories designed to prove invariants:

| Category | Tests | What It Proves |
|----------|-------|----------------|
| **Edge Cases** | 12 | Handles batch=1 to 1024, d_model=4 to 512 |
| **Numerical Stability** | 7 | No NaN/Inf, bounded gradients |
| **Mathematical Correctness** | 10 | 100% accuracy on frozen shapes |
| **Routing Invariants** | 7 | Indices in range, deterministic eval |
| **State Consistency** | 8 | State persists across time |
| **Gradient Integrity** | 7 | STE works, all params get gradients |
| **Equivalence** | 5 | Soft→Hard, 2D↔3D consistent |
| **Stress** | 6 | 256 tiles, 1000 tokens, combined stress |

### Core Architecture Tests

| File | Component | Coverage |
|------|-----------|----------|
| `test_nn.py` | TriXLinear, base layers | Forward/backward, shapes |
| `test_hierarchical.py` | HierarchicalTriXFFN | Routing, clustering |
| `test_sparse.py` | SparseTriXFFN | Tile selection |
| `test_sparse_lookup.py` | SparseLookupFFN | Routing as computation |
| `test_kernel.py` | 2-bit packing | Pack/unpack, NEON |

### Providence Architecture Tests

| File | Component | Coverage |
|------|-----------|----------|
| `test_xor_ffn.py` | XOR routing | Hamming distance, STE |
| `test_frozen_shapes.py` | Frozen shapes library | All 22 shapes, accuracy |
| `test_hierarchical_temporal.py` | Temporal state | State persistence, transitions |
| `test_providence_ffn.py` | ProvidenceFFN | Unified architecture |
| `test_providence_rigorous.py` | Rigorous tests | Edge cases, stress, invariants |

### Memory Architecture Tests

| File | Component | Coverage |
|------|-----------|----------|
| `test_octave_memory.py` | OctaveMemory | Carry chains, addressing |
| `test_providence_memory.py` | ProvidenceMemory | XOR matching, CAM |

---

## Running Tests Manually

If you prefer using pytest directly:

```bash
# Set PYTHONPATH
export PYTHONPATH=src

# Run all tests
python -m pytest tests/ -v

# Run specific file
python -m pytest tests/test_providence_rigorous.py -v

# Run specific test class
python -m pytest tests/test_providence_rigorous.py::TestEdgeCases -v

# Run specific test
python -m pytest tests/test_providence_rigorous.py::TestEdgeCases::test_single_sample -v

# Run with short traceback
python -m pytest tests/ -v --tb=short

# Run with coverage
python -m pytest tests/ -v --cov=src/trix --cov-report=html
```

---

## Understanding Test Output

### Successful Run

```
tests/test_providence_rigorous.py::TestEdgeCases::test_single_sample PASSED [ 1%]
tests/test_providence_rigorous.py::TestEdgeCases::test_large_batch PASSED [ 2%]
...
============================= 67 passed in 19.44s ==============================
```

### Failed Test

```
FAILED tests/test_providence_rigorous.py::TestEdgeCases::test_something
E   AssertionError: Expected X but got Y
```

When a test fails:
1. Read the assertion message
2. Check the test code to understand what's being tested
3. Check if your changes affected that invariant

---

## Writing New Tests

### Test Naming Convention

```python
class TestComponentName:
    def test_what_it_does(self):
        """Descriptive docstring explaining the test."""
        pass
```

### Test Template

```python
import torch
import pytest
from trix.nn import ProvidenceFFN

class TestMyFeature:
    """Tests for my new feature."""

    def test_basic_functionality(self):
        """Basic forward pass works."""
        ffn = ProvidenceFFN(d_model=64, num_tiles=16)
        x = torch.randn(8, 64)
        state = ffn.init_state(8)
        output, new_state, routing_info, aux = ffn(x, state)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_edge_case_single_sample(self):
        """Works with batch size 1."""
        ffn = ProvidenceFFN(d_model=64, num_tiles=16)
        x = torch.randn(1, 64)
        state = ffn.init_state(1)
        output, _, _, _ = ffn(x, state)

        assert output.shape == (1, 64)

    def test_gradient_flow(self):
        """Gradients flow correctly."""
        ffn = ProvidenceFFN(d_model=64, num_tiles=16)
        ffn.train()

        x = torch.randn(8, 64, requires_grad=True)
        state = ffn.init_state(8)
        output, _, _, _ = ffn(x, state)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
```

### Rigorous Test Categories

When adding rigorous tests, categorize them:

```python
class TestEdgeCases:
    """Boundary conditions that often break systems."""
    pass

class TestNumericalStability:
    """No NaN/Inf, bounded gradients."""
    pass

class TestMathematicalCorrectness:
    """The math must be exact where claimed."""
    pass

class TestRoutingInvariants:
    """Properties that must ALWAYS hold."""
    pass

class TestStateConsistency:
    """Temporal state behaves correctly."""
    pass

class TestGradientIntegrity:
    """Training works correctly."""
    pass

class TestEquivalence:
    """Different modes are consistent."""
    pass

class TestStress:
    """System under pressure."""
    pass
```

---

## Continuous Integration

Tests run on every commit. The CI pipeline:

1. Runs `python scripts/run_tests.py quick` for PRs
2. Runs `python scripts/run_tests.py all` on merge to main
3. Generates coverage reports

---

## Test Philosophy

> "Tests are not about finding bugs. Tests are about proving invariants."

Our tests verify:
1. **Boundaries** - Edge cases that break systems first
2. **Stability** - Production runs at scale without NaN/Inf
3. **Correctness** - Mathematical claims hold exactly
4. **Invariants** - Properties that must ALWAYS be true
5. **Pressure** - System behavior under stress

---

## Troubleshooting

### Import Errors

```bash
# Make sure PYTHONPATH is set
export PYTHONPATH=src

# Or use the test runner which sets it automatically
python scripts/run_tests.py quick
```

### Slow Tests

```bash
# Run only quick smoke test
python scripts/run_tests.py quick

# Skip slow tests with pytest marker (if marked)
python -m pytest tests/ -v -m "not slow"
```

### Memory Issues

```bash
# Reduce batch sizes in stress tests
# Edit test_providence_rigorous.py if needed
```

---

*"Rigorous testing is an act of respect for the work."*

*"The architecture provides itself - and the tests prove it."*
