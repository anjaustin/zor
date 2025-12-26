# Lincoln Manifold: Gradient Truth

These artifacts document the discovery of the Gradient Truth paradigm using the [Lincoln Manifold Method](../../LINCOLN_MANIFOLD_METHOD.md).

## The Question

*"Is there a more elegant way to train discrete/ternary networks than the Straight-Through Estimator?"*

## The Phases

1. **[gradient_truth_raw.md](gradient_truth_raw.md)** - Initial brain dump, fears, first instincts
2. **[gradient_truth_nodes.md](gradient_truth_nodes.md)** - 14 key nodes extracted
3. **[gradient_truth_reflect.md](gradient_truth_reflect.md)** - Deep patterns, resolved tensions
4. **[gradient_truth_synth.md](gradient_truth_synth.md)** - The clean specification

## The Result

A three-layer decomposition that eliminates STE entirely:
- **Routing** (continuous, learned)
- **Shape Bank** (discrete, frozen)
- **Magnitude** (continuous, learned)

Implemented in: `src/trix/nn/gradient_truth.py`
Documentation: `docs/GRADIENT_TRUTH.md`
Tests: `tests/test_gradient_truth.py` (32 tests)

## Date

December 2025
