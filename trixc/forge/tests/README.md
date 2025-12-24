# TRIX Forge Test Suite

**Rigorous tests for the skeptics. Production-grade validation.**

> *"No one's laughing us out of anywhere."*

---

## Test Summary

| Suite | Tests | Coverage |
|-------|-------|----------|
| **Python Rigorous** | 57 | Model IR, Levers, Events, Compilation, NGE, E2E |
| **C Output Verification** | 48 | Frozen shapes, MatMul, Softmax, LayerNorm, Full Adder |
| **Integration** | 20 | TRIXC compat, Platforms, Robustness, Error handling |
| **Total** | **125** | **Full stack** |

---

## Quick Start

```bash
# Run all Python tests
python -m pytest . -v

# Run C tests
gcc -O2 -Wall -o test_forge_c_output test_forge_c_output.c -lm
./test_forge_c_output

# Run specific suite
python -m pytest test_forge_rigorous.py -v
python -m pytest test_forge_integration.py -v
```

---

## Test Categories

### 1. Model IR Tests (`test_forge_rigorous.py::TestModelIR`)

Validates the Intermediate Representation:
- Create empty/populated models
- Save/load roundtrip
- to_dict/from_dict serialization
- Validation (valid models, missing tensors)
- Complex multi-layer models

### 2. Lever System Tests (`test_forge_rigorous.py::TestLeverSystem`)

Validates the parameter control system:
- Create levers with all properties
- Set valid/invalid enum values
- Set valid/invalid int ranges
- Set valid/invalid float ranges
- Reset to default
- Serialization (to_dict)
- Integration with models

### 3. Event Spine Tests (`test_forge_rigorous.py::TestEventSpine`)

Validates the Glassbox protocol:
- Emit single/multiple events
- Frame tracking
- Subscribe to all/filtered events
- Unsubscribe
- Query by frame/type/source
- Export to JSON
- Event ordering
- Clear events

### 4. Compilation Tests (`test_forge_rigorous.py::TestCompilation`)

Validates code generation:
- List available targets
- Target properties (arch, features)
- Compile to C
- Include/exclude metadata
- Event emission during compile
- Invalid target handling
- Complex model compilation

### 5. NGE Format Tests (`test_forge_rigorous.py::TestNGEFormat`)

Validates Neural-Geometric Executable format:
- Magic bytes ("TRIX")
- Version encoding
- Flags encoding (glassbox, metadata, debug)
- Metadata JSON structure

### 6. End-to-End Tests (`test_forge_rigorous.py::TestEndToEnd`)

Full pipeline validation:
- Define → Validate → Save → Load → Compile
- Pipeline with lever modifications
- Pipeline with event tracking
- Multi-target compilation

### 7. Edge Cases (`test_forge_rigorous.py::TestEdgeCases`)

Boundary conditions:
- Empty models
- Very large tensors (1024³)
- Unicode in names (中文, 🔥)
- Special characters
- Deeply nested models (100 layers)
- Concurrent event emission (10,000 events)
- Lever boundary values
- Float precision

### 8. Determinism Tests (`test_forge_rigorous.py::TestDeterminism`)

Reproducibility:
- Same input → same hash
- Compilation determinism (3 runs)
- Event ID uniqueness

### 9. C Output Verification (`test_forge_c_output.c`)

Tests the actual generated C code works:
- NGE header format
- Frozen shapes (XOR, AND, OR, NOT)
- De Morgan's laws
- Activation functions (ReLU, Sigmoid, GELU)
- MatMul correctness
- Softmax properties
- Layer normalization
- Argmax
- Full adder (all 8 cases)
- MLP forward pass
- Determinism (100 runs)
- Performance (128×128 matmul < 100ms)

### 10. Integration Tests (`test_forge_integration.py`)

Validates TRIXC ecosystem integration:
- Generated code compiles with GCC
- Correct header includes
- Shape compatibility
- Precision levels match APU
- Can represent TRIXC Pi models
- All targets produce output
- Event flow through pipeline
- Lever state persistence
- Large model handling (50 layers)
- Many levers (100)
- Rapid event emission
- Error message quality

---

## Running Individual Tests

```bash
# Run specific test class
python -m pytest test_forge_rigorous.py::TestModelIR -v

# Run specific test
python -m pytest test_forge_rigorous.py::TestModelIR::test_save_and_load -v

# Run with coverage
python -m pytest test_forge_rigorous.py --cov=. --cov-report=term-missing

# Run C tests in verbose mode
./test_forge_c_output 2>&1 | grep -E "(PASS|FAIL)"
```

---

## Test Philosophy

### 1. Exhaustive

We don't sample. We test all cases when feasible:
- All 8 full adder input combinations
- All logic gate truth tables
- All built-in targets

### 2. Deterministic

Every test produces the same result on every run:
- No random data without seeds
- Explicit ordering checks
- Hash consistency verification

### 3. Fast

The full suite runs in < 1 second:
- No unnecessary I/O
- Temp files cleaned up
- Efficient data structures

### 4. Self-Documenting

Tests serve as documentation:
- Clear test names
- Example patterns in test code
- Categories match system components

---

## Adding New Tests

### Python Test Template

```python
def test_new_feature(self):
    """Test description."""
    # Setup
    model = ModelIR(name="test")

    # Action
    result = model.some_operation()

    # Assert
    assert result == expected
```

### C Test Template

```c
TEST(new_feature) {
    /* Setup */
    float input = 1.0f;

    /* Action */
    float result = some_function(input);

    /* Assert */
    ASSERT_FLOAT_EQ(result, expected, 1e-6f);
}
```

---

## CI Integration

```yaml
# GitHub Actions example
test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v2

    - name: Python tests
      run: |
        cd trixc/forge/tests
        python -m pytest . -v --tb=short

    - name: C tests
      run: |
        cd trixc/forge/tests
        gcc -O2 -Wall -o test_c test_forge_c_output.c -lm
        ./test_c
```

---

## Results

```
TRIX Forge Test Suite
=====================
Python Rigorous:    57/57 passed (0.26s)
C Verification:     48/48 passed (instant)
Integration:        20/20 passed (0.24s)
=====================
Total:             125/125 passed

"It's all in the reflexes."
```

---

*No one's laughing us out of anywhere.*
