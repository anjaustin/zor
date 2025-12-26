# Testing Guide

Comprehensive documentation of the TriX test suite.

---

## Overview

The test suite verifies that TriX architectures maintain their mathematical properties, handle edge cases correctly, and perform as expected across different modes.

```
Total Python Tests: 200+
Total C Tests:      30+

Key Test Files:
  tests/test_octave.py            TrueOctaveFFN functionality
  tests/test_octave_rigorous.py   Mathematical invariants
  tests/test_multiscale.py        MultiScaleTriXFFN
  tests/test_hierarchical.py      HierarchicalTriXFFN
  tests/test_sparse_lookup.py     SparseLookupFFN
```

---

## Running Tests

### All Tests

```bash
# Python tests
PYTHONPATH=src pytest tests/ -v

# C tests
cd src/trix/native/ops && make test

# Both via Makefile
make test
```

### Specific Test Files

```bash
# TrueOctaveFFN tests
PYTHONPATH=src pytest tests/test_octave.py tests/test_octave_rigorous.py -v

# Hierarchical tests
PYTHONPATH=src pytest tests/test_hierarchical.py -v

# SparseLookup tests
PYTHONPATH=src pytest tests/test_sparse_lookup.py -v
```

### Specific Test Categories

```bash
# Only rigorous invariant tests
PYTHONPATH=src pytest tests/test_octave_rigorous.py -v

# Only derivation tests
PYTHONPATH=src pytest tests/test_octave_rigorous.py::TestDerivationInvariants -v

# Only numerical stability tests
PYTHONPATH=src pytest tests/test_octave_rigorous.py::TestNumericalStability -v
```

### With Coverage

```bash
PYTHONPATH=src pytest tests/ --cov=src/trix --cov-report=term-missing
```

---

## Test Categories

### 1. TrueOctaveFFN Tests

**File:** `tests/test_octave.py` (31 tests)

Basic functionality tests for the True Octave architecture.

| Test Class | Tests | Purpose |
|------------|-------|---------|
| `TestFrozenTile` | 6 | Single tile weights, scales, gradients |
| `TestOctave` | 3 | Octave routing (soft/hard) |
| `TestDeriveOctave` | 4 | Octave derivation correctness |
| `TestTrueOctaveFFN` | 10 | Full FFN behavior |
| `TestTrueOctaveBlock` | 2 | Transformer block integration |
| `TestIntegration` | 3 | Training loops, mode switching |
| `TestPhilosophy` | 3 | Conceptual properties |

**Key Tests:**

```python
def test_coarse_is_view_of_fine():
    """Coarse octave should be a compressed view of fine, not independent."""
    
def test_same_architecture_two_behaviors():
    """Same frozen structure should support both exact and fuzzy behavior."""
```

---

### 2. Rigorous Invariant Tests

**File:** `tests/test_octave_rigorous.py` (45 tests)

Mathematical invariants that MUST hold for the architecture to be correct.

#### 2.1 Derivation Invariants (4 tests)

The core property: coarse octaves are derived from fine octaves.

```python
def test_medium_signature_equals_pooled_fine():
    """Medium[i].signature == sign(mean(Fine[4i:4i+4].signatures))"""

def test_coarse_signature_equals_pooled_medium():
    """Coarse[j].signature == sign(mean(Medium[4j:4j+4].signatures))"""

def test_derivation_holds_for_various_pool_factors():
    """Derivation must work for pool_factor in {2, 4}."""

def test_rederive_restores_derivation():
    """After corruption, rederive() must restore correct derivation."""
```

**Why these matter:** If derivation fails, octaves become independent banks instead of multi-resolution views. The architecture loses its conceptual foundation.

---

#### 2.2 Ternary Invariants (4 tests)

All frozen weights must be in {-1, 0, +1}.

```python
def test_fine_weights_are_ternary():
    """Fine octave weights in {-1, 0, +1}."""

def test_medium_weights_are_ternary():
    """Medium octave weights in {-1, 0, +1}."""

def test_coarse_weights_are_ternary():
    """Coarse octave weights in {-1, 0, +1}."""

def test_signatures_are_ternary():
    """All signatures in {-1, 0, +1}."""
```

**Why these matter:** Ternary weights enable efficient computation (add/subtract only, no multiply). Non-ternary weights break this property.

---

#### 2.3 Frozen Invariants (3 tests)

Frozen structure must not change during training.

```python
def test_weights_unchanged_after_forward_backward():
    """Weights identical before and after backward pass."""

def test_weights_unchanged_after_optimizer_step():
    """Weights identical before and after optimizer.step()."""

def test_weights_have_no_grad():
    """Frozen weights have requires_grad=False."""
```

**Why these matter:** If frozen weights change, we're back to STE-based learning. The whole point of Gradient Truth is that structure is discovered, not learned.

---

#### 2.4 Mode Invariants (6 tests)

Deterministic mode must be exact. Generative mode must be soft.

```python
def test_deterministic_blend_is_one_hot():
    """Blend weights are one-hot in deterministic mode."""

def test_deterministic_entropy_is_zero():
    """Entropy is 0 in deterministic mode (no uncertainty)."""

def test_generative_blend_is_soft():
    """Blend weights are NOT one-hot in generative mode."""

def test_generative_entropy_is_positive():
    """Entropy > 0 in generative mode (uncertainty exists)."""

def test_deterministic_is_reproducible():
    """Same input gives exact same output in deterministic mode."""

def test_mode_switch_changes_output():
    """Different modes produce different outputs."""
```

**Why these matter:** The architecture's claim is that it can do both exact (6502) and fuzzy (LLM) computation. These tests verify that claim.

---

#### 2.5 Gradient Invariants (4 tests)

Gradients must flow only to learned parameters.

```python
def test_gradients_flow_to_scales():
    """Tile scales receive gradients."""

def test_gradients_flow_to_blend_network():
    """Blend network receives gradients."""

def test_gradients_flow_to_output_scale():
    """Output scale receives gradients."""

def test_no_gradients_to_frozen_weights():
    """Frozen weights do NOT receive gradients."""
```

**Why these matter:** This is Gradient Truth in action. Gradients should flow through continuous parameters (scales), not discrete structure (ternary weights).

---

#### 2.6 Numerical Stability (6 tests)

No NaN, no Inf, bounded outputs.

```python
def test_no_nan_in_forward():
    """Forward pass produces no NaN."""

def test_no_inf_in_forward():
    """Forward pass produces no Inf."""

def test_no_nan_with_large_input():
    """No NaN with input * 100."""

def test_no_nan_with_small_input():
    """No NaN with input * 1e-6."""

def test_no_nan_in_backward():
    """Backward pass produces no NaN in gradients."""

def test_output_bounded():
    """Output magnitude is bounded."""
```

**Why these matter:** Numerical instability is silent and deadly. These tests catch it early.

---

#### 2.7 Edge Cases (8 tests)

Boundary conditions that must be handled correctly.

```python
def test_batch_size_one():
    """Works with batch_size=1."""

def test_sequence_length_one():
    """Works with seq_len=1."""

def test_single_token():
    """Works with batch=1, seq=1."""

def test_large_batch():
    """Works with batch_size=128."""

def test_long_sequence():
    """Works with seq_len=512."""

def test_zero_input():
    """Handles all-zero input."""

def test_constant_input():
    """Handles constant input."""

def test_minimum_tiles():
    """Works with minimum configuration (4 fine, 2 medium, 1 coarse)."""
```

**Why these matter:** Edge cases are where bugs hide. Testing them explicitly prevents silent failures.

---

#### 2.8 Reproducibility (3 tests)

Determinism where expected.

```python
def test_same_seed_same_init():
    """Same random seed produces identical initialization."""

def test_deterministic_same_output():
    """Deterministic mode produces identical output on repeated calls."""

def test_eval_mode_deterministic():
    """Eval mode is deterministic (no dropout)."""
```

**Why these matter:** Reproducibility is essential for debugging and scientific rigor.

---

#### 2.9 Training Correctness (3 tests)

Training actually works.

```python
def test_loss_decreases():
    """Loss decreases during training."""

def test_scales_change_during_training():
    """Tile scales are updated by optimizer."""

def test_blend_network_changes_during_training():
    """Blend network is updated by optimizer."""
```

**Why these matter:** A model that doesn't train is useless. These tests verify the training loop works.

---

#### 2.10 Block Integration (4 tests)

TrueOctaveBlock integrates correctly.

```python
def test_block_forward_shape():
    """Block preserves input shape."""

def test_block_mode_propagates():
    """set_mode propagates to FFN."""

def test_stacked_blocks():
    """Multiple blocks can be stacked."""

def test_block_gradient_flow():
    """Gradients flow through entire block."""
```

**Why these matter:** The FFN is used inside transformer blocks. Integration must work correctly.

---

### 3. HierarchicalTriXFFN Tests

**File:** `tests/test_hierarchical.py` (32 tests)

Tests for the ternary matmul architecture with Gradient Truth.

| Test Class | Tests | Purpose |
|------------|-------|---------|
| `TestTriXTile` | 8 | Individual tile behavior |
| `TestHierarchicalTriXFFN` | 12 | Full FFN functionality |
| `TestHierarchicalTriXBlock` | 6 | Transformer block |
| `TestIntegration` | 6 | Training, stacking, modes |

**Key Properties Tested:**
- Frozen ternary weights
- Learned scales only
- Hierarchical routing (cluster → tile)
- Gradient Truth mode (default) vs STE mode (deprecated)

---

### 4. SparseLookupFFN Tests

**File:** `tests/test_sparse_lookup.py` (24 tests)

Tests for the MatMul-free architecture.

| Test Class | Tests | Purpose |
|------------|-------|---------|
| `TestTernarySpline2D` | 5 | Spline with ternary coefficients |
| `TestFloatSpline2D` | 3 | Float spline (baseline) |
| `TestSparseLookupFFN` | 10 | Full FFN functionality |
| `TestSparseLookupBlock` | 3 | Transformer block |
| `TestIntegration` | 3 | Training, stacking |

**Key Properties Tested:**
- Frozen ternary directions
- Frozen ternary spline coefficients
- Learned scales and compression network
- No matrix multiply in hot path

---

### 5. MultiScaleTriXFFN Tests

**File:** `tests/test_multiscale.py` (21 tests)

Tests for the scaffold architecture (independent octaves).

| Test Class | Tests | Purpose |
|------------|-------|---------|
| `TestOctaveTile` | 5 | Individual tile |
| `TestOctave` | 3 | Octave routing |
| `TestMultiScaleTriXFFN` | 8 | Full FFN |
| `TestMultiScaleTriXBlock` | 2 | Transformer block |
| `TestIntegration` | 3 | Training, modes |

**Note:** This is the scaffold version with independent octaves. See `TrueOctaveFFN` for the correct derived-octave implementation.

---

## Writing New Tests

### Test Structure

```python
class TestMyFeature:
    """Tests for MyFeature."""
    
    def test_basic_functionality(self):
        """Description of what this tests."""
        # Arrange
        model = MyModel(...)
        x = torch.randn(...)
        
        # Act
        out = model(x)
        
        # Assert
        assert out.shape == expected_shape
    
    def test_invariant_property(self):
        """Mathematical property that MUST hold."""
        # ... test code ...
        assert property_holds, "Explanation of what failed"
```

### Test Naming Convention

```
test_<what>_<expected_behavior>

Examples:
  test_weights_are_ternary
  test_gradients_flow_to_scales
  test_deterministic_is_reproducible
```

### Adding to Rigorous Tests

When adding to `test_octave_rigorous.py`, put tests in the appropriate category:

```python
class TestDerivationInvariants:
    """Coarse octaves MUST be derived from fine octaves."""
    
    def test_new_derivation_property(self):
        """New derivation invariant."""
        ...

class TestNumericalStability:
    """No NaN, no Inf, bounded outputs."""
    
    def test_new_stability_check(self):
        """New stability check."""
        ...
```

---

## Continuous Integration

Tests are run on every commit. The CI pipeline:

1. Runs all Python tests with pytest
2. Runs all C tests with make
3. Reports coverage
4. Fails on any test failure

### Pre-commit Verification

Before committing, run:

```bash
PYTHONPATH=src pytest tests/ -v --tb=short
```

All tests must pass.

---

## Test Philosophy

### What We Test

1. **Invariants** — Mathematical properties that must hold
2. **Edge cases** — Boundary conditions
3. **Integration** — Components work together
4. **Training** — The system actually learns

### What We Don't Test

1. **Performance** — Separate benchmarking suite
2. **Specific loss values** — Too brittle
3. **Exact output values** — Random initialization varies

### Test Independence

Each test should:
- Create its own model instance
- Not depend on other tests
- Not leave global state

### Test Speed

- Individual tests should complete in < 1 second
- Full suite should complete in < 30 seconds
- Use small dimensions (d_model=64, not 512)

---

## Debugging Failed Tests

### Verbose Output

```bash
PYTHONPATH=src pytest tests/test_octave_rigorous.py -v --tb=long
```

### Run Single Test

```bash
PYTHONPATH=src pytest tests/test_octave_rigorous.py::TestDerivationInvariants::test_medium_signature_equals_pooled_fine -v
```

### Print Debugging

```python
def test_something(self):
    model = MyModel(...)
    print(f"Model structure: {model}")
    print(f"Weights: {model.weights}")
    # ... rest of test
```

Run with `-s` to see print output:

```bash
PYTHONPATH=src pytest tests/test_foo.py -v -s
```

---

## References

- [TRUE_OCTAVE.md](TRUE_OCTAVE.md) — TrueOctaveFFN architecture
- [GRADIENT_TRUTH.md](GRADIENT_TRUTH.md) — Gradient Truth paradigm
- [pytest documentation](https://docs.pytest.org/)
