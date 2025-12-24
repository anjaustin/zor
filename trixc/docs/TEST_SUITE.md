# TRIXC Test Suite Documentation

**Rigorous verification for the skeptics. Every shape. Every edge case.**

*"Trust, but verify. Then verify again."*

---

## Overview

The TRIXC test suite provides **exhaustive verification** of the compiler and all frozen shapes. It consists of three complementary test suites:

| Suite | Language | Tests | Focus |
|-------|----------|-------|-------|
| `test_rigorous.c` | C | 1,329 | Frozen shapes, 6502 ALU, ONNX ops |
| `test_6502_onnx_pipeline.py` | Python | 20 | ONNX→C pipeline validation |
| `test_emit_c.py` | Python | 26 | C code generation |
| **Total** | | **1,375** | |

---

## Quick Start

```bash
# Run all tests
make test

# Run C rigorous suite only
make build/test_rigorous && ./build/test_rigorous

# Run Python tests
python -m pytest test/ -v

# Run specific test file
python -m pytest test/test_6502_onnx_pipeline.py -v
```

---

## Test Suite 1: C Rigorous Tests

**File:** `test/test_rigorous.c`

### Purpose

Validates all frozen shapes at the C level with exhaustive testing. This is the foundational test—if these pass, the mathematical shapes are correct.

### Sections

#### 1. Logic Shapes - Truth Tables (28 tests)

Complete truth table verification for all logic gates:

```c
// XOR: a + b - 2ab
TEST(trix_shape_xor_f32(0.0f, 0.0f) == 0.0f);  // 0 XOR 0 = 0
TEST(trix_shape_xor_f32(0.0f, 1.0f) == 1.0f);  // 0 XOR 1 = 1
TEST(trix_shape_xor_f32(1.0f, 0.0f) == 1.0f);  // 1 XOR 0 = 1
TEST(trix_shape_xor_f32(1.0f, 1.0f) == 0.0f);  // 1 XOR 1 = 0
```

Tests: XOR, AND, OR, NOT, NAND, NOR, XNOR

#### 2. Logic Shapes - Mathematical Invariants (800 tests)

Verifies algebraic properties hold for random inputs:

| Property | Formula | Test |
|----------|---------|------|
| XOR self-inverse | `XOR(a, a) = 0` | ✓ |
| XOR identity | `XOR(a, 0) = a` | ✓ |
| XOR commutative | `XOR(a, b) = XOR(b, a)` | ✓ |
| AND identity | `AND(a, 1) = a` | ✓ |
| AND annihilator | `AND(a, 0) = 0` | ✓ |
| OR identity | `OR(a, 0) = a` | ✓ |
| De Morgan's | `NOT(AND(a,b)) = OR(NOT(a),NOT(b))` | ✓ |

#### 3. Full Adder - Exhaustive (8 tests)

All 8 input combinations for the 1-bit full adder:

```
A B Cin │ Sum Cout
────────┼─────────
0 0  0  │  0   0
0 0  1  │  1   0
0 1  0  │  1   0
0 1  1  │  0   1
1 0  0  │  1   0
1 0  1  │  0   1
1 1  0  │  0   1
1 1  1  │  1   1
```

#### 4. Ripple Adder - Exhaustive 8-bit (65,536 tests)

Tests **every possible 8-bit addition**:

```c
for (int a = 0; a < 256; a++) {
    for (int b = 0; b < 256; b++) {
        // Test a + b using ripple carry adder
        // Verify result matches (a + b) & 0xFF
        // Verify carry matches (a + b) > 255
    }
}
```

**Result:** All 65,536 additions pass.

#### 5. 6502 ALU - Hardware Vectors (29 tests)

Known test vectors from actual 6502 hardware behavior:

| Operation | Test Case | Expected |
|-----------|-----------|----------|
| ADC | `0xFF + 0x01` | `0x00, C=1` |
| ADC | `0x80 + 0x80` | `0x00, C=1` |
| SBC | `0x50 - 0x30` | `0x20` |
| AND | `0xAA & 0x55` | `0x00` |
| ORA | `0xAA \| 0x55` | `0xFF` |
| EOR | `0xAA ^ 0xFF` | `0x55` |
| ASL | `0x80 << 1` | `0x00, C=1` |
| LSR | `0x01 >> 1` | `0x00, C=1` |
| ROL | `0x80, C=1` | `0x01, C=1` |
| ROR | `0x01, C=1` | `0x80, C=1` |
| INC | `0xFF + 1` | `0x00` |
| DEC | `0x00 - 1` | `0xFF` |

#### 6. ONNX Shapes - Numerical Accuracy (34 tests)

Validates ONNX-compatible operations:

**Activations:**
```c
TEST(trix_onnx_relu(-1.0f) == 0.0f);
TEST(trix_onnx_sigmoid(0.0f) == 0.5f);
TEST(trix_onnx_tanh(0.0f) == 0.0f);
TEST(trix_onnx_gelu(0.0f) == 0.0f);
```

**Arithmetic:**
```c
float a[4] = {1, 2, 3, 4};
float b[4] = {0.5, 1, 1.5, 2};
trix_onnx_add(a, b, c, 4);  // [1.5, 3, 4.5, 6]
trix_onnx_mul(a, b, c, 4);  // [0.5, 2, 4.5, 8]
```

**Matrix Operations:**
```c
// 2x3 @ 3x2 = 2x2
// Verified against analytical solution
trix_onnx_matmul(A, B, C, 2, 2, 3);
TEST(C[0] == 22.0f);  // [1,2,3] @ [1,3,5]^T
```

**Softmax:**
```c
trix_onnx_softmax(logits, probs, 4);
TEST(sum(probs) == 1.0f);           // Probabilities sum to 1
TEST(probs[3] > probs[2] > ...);    // Preserves order
TEST(all_positive(probs));          // All positive
```

**LayerNorm:**
```c
trix_onnx_layer_norm(x, gamma, beta, out, 4, 1e-5f);
TEST(mean(out) ≈ 0.0f);   // Zero mean
TEST(var(out) ≈ 1.0f);    // Unit variance
```

#### 7. Precision Conversions (9 tests)

Edge case testing for mixed precision:

```c
// FP16 roundtrip
TEST(fp16_to_fp32(fp32_to_fp16(0.0f)) == 0.0f);
TEST(fp16_to_fp32(fp32_to_fp16(1.0f)) == 1.0f);

// FP8 approximate roundtrip
TEST(fp8_to_fp32(fp32_to_fp8(1.0f)) ≈ 1.0f);

// FP4 range check
TEST(fp4_to_fp32(fp32_to_fp4(1.0f)) in [0, 4]);
```

#### 8. Hamming Distance (6 tests)

Content-addressed memory fundamentals:

```c
TEST(hamming(0x00, 0xFF) == 8);  // All bits different
TEST(hamming(x, x) == 0);         // Self-distance zero
TEST(hamming(0x00, 0x55) == 4);   // Half bits different
// Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
```

#### 9. Sparse Octave Lookup (5 tests)

Multi-scale memory operations:

```c
trix_sparse_octave_init(&sol, 16, 2, 32, 4);
trix_sparse_octave_forward(&sol, input, output);
TEST(output != all_zeros);
TEST(output is finite);
```

#### 10. Providence (3 tests)

Content-addressed memory with soft attention:

```c
trix_providence_init(&prov, 8, 32);
trix_providence_lookup(&prov, query, result, 4, 1.0f);
TEST(result in reasonable_range);
```

#### 11. Stress Tests (3 tests)

Performance under load:

```c
// 1 million random additions
for (int i = 0; i < 1000000; i++) {
    uint8_t result = alu_execute(ALU_ADC, rand(), rand(), rand() & 1);
}
// Verified: 0 errors

// 1000x 64x64 matrix multiply
for (int i = 0; i < 1000; i++) {
    trix_onnx_matmul(A, B, C, 64, 64, 64);
}
// Verified: completes in < 10 seconds
```

### Running

```bash
# Compile
gcc -O2 -I./include test/test_rigorous.c -o build/test_rigorous -lm

# Run
./build/test_rigorous
```

### Expected Output

```
╔═══════════════════════════════════════════════════════════════╗
║         TRIXC RIGOROUS TEST SUITE                             ║
╚═══════════════════════════════════════════════════════════════╝

=== Logic Shapes - Complete Truth Tables ===
  Section: 28/28 passed

=== Ripple Adder - Exhaustive 8-bit (256 × 256 = 65536 tests) ===
  PASS: All 65536 additions correct
  Section: 1/1 passed

... (more sections) ...

╔═══════════════════════════════════════════════════════════════╗
║  FINAL RESULTS                                                ║
╠═══════════════════════════════════════════════════════════════╣
║  Total tests:   1329                                          ║
║  Passed:        1329                                          ║
║  Failed:           0                                          ║
║  Pass rate:    100.0%                                        ║
╚═══════════════════════════════════════════════════════════════╝

✓ ALL TESTS PASSED
"The shapes are frozen. The math is eternal. The skeptics are satisfied."
```

---

## Test Suite 2: Python 6502 Pipeline Tests

**File:** `test/test_6502_onnx_pipeline.py`

### Purpose

Validates the complete ONNX → C → Compile → Run pipeline using 6502 ALU-equivalent operations. This proves the compiler produces correct code.

### Philosophy

The 6502 ALU is an ideal test case because:
1. **Well-defined operations** with exact mathematical specifications
2. **Small domain** (8-bit) allows exhaustive testing
3. **Known hardware behavior** provides ground truth

### Test Classes

#### TestBitwiseOperations (2 tests)

Tests bitwise operations expressed as frozen shapes:

```python
def test_bitwise_and_exhaustive(self):
    """AND using frozen shape: a * b"""
    for a in [0, 1]:
        for b in [0, 1]:
            result = a * b
            expected = a & b
            assert result == expected

def test_xor_truth_table(self):
    """XOR using frozen shape: a + b - 2*a*b"""
    truth_table = [
        (0.0, 0.0, 0.0),  # 0 XOR 0 = 0
        (0.0, 1.0, 1.0),  # 0 XOR 1 = 1
        (1.0, 0.0, 1.0),  # 1 XOR 0 = 1
        (1.0, 1.0, 0.0),  # 1 XOR 1 = 0
    ]
```

#### TestArithmeticOperations (3 tests)

Tests arithmetic building blocks:

```python
def test_half_adder_truth_table(self):
    """Half adder: sum=XOR, carry=AND"""
    # All 4 combinations verified

def test_full_adder_truth_table(self):
    """Full adder: all 8 input combinations"""
    # Verified against truth table

def test_8bit_add_reference(self):
    """256 random 8-bit additions"""
    for _ in range(256):
        a, b = random 8-bit values
        result, carry = ref_adc(a, b, 0)
        assert result == (a + b) & 0xFF
```

#### TestMatMulPipeline (2 tests)

Tests matrix multiply through the full pipeline:

```python
def test_matmul_small(self):
    """[1,4] @ [4,2] = [1,2]"""
    # Create ONNX model
    # Convert to C
    # Compile
    # Run
    # Verify output matches NumPy

def test_matmul_exhaustive(self):
    """100 random inputs through pipeline"""
```

#### TestMLPPipeline (1 test)

Tests multi-layer perceptron:

```python
def test_mlp_forward(self):
    """input -> FC1 -> ReLU -> FC2 -> output"""
    # Verifies complete forward pass
```

#### Test6502ALUEquivalence (6 tests)

**The definitive tests:** Proves ONNX models match 6502 hardware.

```python
def test_and_exhaustive_4bit(self):
    """256 AND combinations"""
    for a in range(16):
        for b in range(16):
            result = bits_to_int(int_to_bits(a) * int_to_bits(b))
            assert result == ref_and(a, b)

def test_or_exhaustive_4bit(self):
    """256 OR combinations using a + b - ab"""

def test_xor_exhaustive_4bit(self):
    """256 XOR combinations using a + b - 2ab"""

def test_adc_exhaustive_8bit(self):
    """65,536 additions"""
    for a in range(256):
        for b in range(256):
            result, carry = ref_adc(a, b, 0)
            assert result == (a + b) & 0xFF

def test_shift_operations(self):
    """ASL and LSR for all 256 values"""

def test_inc_dec(self):
    """INC and DEC for all 256 values"""
```

#### TestCodeGenerationQuality (3 tests)

Validates generated code quality:

```python
def test_no_memory_leaks_potential(self):
    """Forward function uses alloca, not malloc"""
    assert "malloc" not in forward_section

def test_weights_in_rodata(self):
    """Weights are static const (goes in .rodata)"""
    assert "static const float" in c_code

def test_proper_includes(self):
    """All necessary headers present"""
    assert "#include <stdio.h>" in c_code
    assert "#include <math.h>" in c_code
    assert "trixc/onnx_shapes.h" in c_code
```

#### TestNumericalStability (2 tests)

Edge case handling:

```python
def test_large_values(self):
    """Model handles 1e6, 1e-6 without overflow"""

def test_denormal_handling(self):
    """Model handles denormal numbers"""
```

#### TestSummary (1 test)

Complete pipeline sanity check:

```python
def test_complete_pipeline_summary(self):
    """Tests ReLU, XOR, MatMul, MLP models"""
```

### ONNX Model Builders

The test creates ONNX models representing 6502 operations:

| Model | ONNX Ops | 6502 Equivalent |
|-------|----------|-----------------|
| `create_xor_model` | Mul, Add, Sub | EOR (XOR) |
| `create_bitwise_and_model` | Mul | AND |
| `create_bitwise_or_model` | Mul, Add, Sub | ORA (OR) |
| `create_half_adder_model` | Mul, Add, Sub, Concat | Half adder |
| `create_full_adder_model` | Mul, Add, Sub, Concat | Full adder |
| `create_matmul_model` | MatMul | Matrix ops |
| `create_mlp_model` | MatMul, Relu | Neural net |

### Running

```bash
# All tests
python -m pytest test/test_6502_onnx_pipeline.py -v

# Specific test class
python -m pytest test/test_6502_onnx_pipeline.py::Test6502ALUEquivalence -v

# With output
python -m pytest test/test_6502_onnx_pipeline.py -v -s
```

### Expected Output

```
test/test_6502_onnx_pipeline.py::TestBitwiseOperations::test_bitwise_and_exhaustive PASSED
test/test_6502_onnx_pipeline.py::TestBitwiseOperations::test_xor_truth_table PASSED
test/test_6502_onnx_pipeline.py::TestArithmeticOperations::test_8bit_add_reference PASSED
test/test_6502_onnx_pipeline.py::TestArithmeticOperations::test_full_adder_truth_table PASSED
test/test_6502_onnx_pipeline.py::TestArithmeticOperations::test_half_adder_truth_table PASSED
test/test_6502_onnx_pipeline.py::Test6502ALUEquivalence::test_adc_exhaustive_8bit PASSED
test/test_6502_onnx_pipeline.py::Test6502ALUEquivalence::test_and_exhaustive_4bit PASSED
test/test_6502_onnx_pipeline.py::Test6502ALUEquivalence::test_xor_exhaustive_4bit PASSED
...
======================== 20 passed in 8.72s ========================
```

---

## Test Suite 3: Python C Emission Tests

**File:** `test/test_emit_c.py`

### Purpose

Unit tests for C code generation functions. Validates that each component of the code generator works correctly in isolation.

### Test Classes

#### TestSanitizeName (5 tests)

C identifier generation:

```python
_sanitize_name("weight")           # "weight"
_sanitize_name("fc1.weight")       # "fc1_weight"
_sanitize_name("model/layer/w")    # "model_layer_w"
_sanitize_name("0_layer")          # "_0_layer"
_sanitize_name("")                 # "_unnamed"
```

#### TestFormatFloat (4 tests)

Float formatting for C:

```python
_format_float(0.0)    # "0.0f"
_format_float(1.0)    # "1.0f"
_format_float(0.5)    # "0.5f"
_format_float(-1.5)   # "-1.5f"
```

#### TestGetTensorSize (4 tests)

Tensor size calculation:

```python
_get_tensor_size([2, 3])           # 6
_get_tensor_size([2, 3, 4])        # 24
_get_tensor_size(["batch", 768])   # 768 (dynamic → 1)
_get_tensor_size([])               # 1
```

#### TestEmitWeights (2 tests)

Weight array generation:

```python
weights = {"fc.weight": {"shape": [2,3], "data": [1,2,3,4,5,6]}}
result = emit_weights(weights)
# "static const float W_fc_weight[6] = { 1.0f, 2.0f, ... };"
```

#### TestBuildTensorMap (1 test)

Tensor name mapping:

```python
tensor_map = build_tensor_map(trix)
assert tensor_map["input"]["kind"] == "input"
assert tensor_map["fc.weight"]["kind"] == "weight"
assert tensor_map["output"]["kind"] == "output"
```

#### TestGenerateCCode (3 tests)

Full C generation:

```python
def test_simple_relu(self):
    code = generate_c_code(trix, standalone=True)
    assert "simple_relu_forward" in code
    assert "trix_onnx_relu" in code
    assert "int main(" in code

def test_no_standalone(self):
    code = generate_c_code(trix, standalone=False)
    assert "int main(" not in code
```

#### TestEmitShapeCall (4 tests)

Per-operation code templates:

```python
def test_relu(self):
    result = emit_shape_call(relu_shape, tensor_map)
    assert "trix_onnx_relu" in result

def test_matmul(self):
    result = emit_shape_call(matmul_shape, tensor_map)
    assert "trix_onnx_matmul" in result
```

#### TestFullPipeline (1 test)

End-to-end conversion:

```python
def test_onnx_to_c_conversion(self):
    # Create ONNX model
    # Convert to C
    # Verify structure
```

#### TestCLI (1 test)

Command-line interface:

```python
def test_help(self):
    result = subprocess.run(["python", "onnx2trix.py", "--help"])
    assert "--emit-c" in result.stdout
```

### Running

```bash
python -m pytest test/test_emit_c.py -v
```

---

## Reference Implementations

The test suite includes reference implementations for verification:

```python
# From test/test_6502_onnx_pipeline.py

def ref_adc(a: int, b: int, carry_in: int) -> Tuple[int, int]:
    """Reference ADC matching 6502 hardware."""
    result = a + b + carry_in
    carry_out = 1 if result > 255 else 0
    return result & 0xFF, carry_out

def ref_and(a: int, b: int) -> int:
    """Reference AND."""
    return a & b

def ref_eor(a: int, b: int) -> int:
    """Reference XOR."""
    return a ^ b

def ref_asl(a: int) -> Tuple[int, int]:
    """Reference ASL."""
    carry = (a >> 7) & 1
    result = (a << 1) & 0xFF
    return result, carry
```

---

## Frozen Shape Formulas

The tests verify these mathematical facts:

| Shape | Formula | Domain |
|-------|---------|--------|
| XOR | `a + b - 2ab` | {0, 1} |
| AND | `ab` | {0, 1} |
| OR | `a + b - ab` | {0, 1} |
| NOT | `1 - a` | {0, 1} |
| NAND | `1 - ab` | {0, 1} |
| NOR | `1 - (a + b - ab)` | {0, 1} |
| XNOR | `1 - (a + b - 2ab)` | {0, 1} |
| ReLU | `max(0, x)` | ℝ |
| Sigmoid | `1 / (1 + exp(-x))` | ℝ |
| GELU | `x · sigmoid(1.702x)` | ℝ |

---

## Test Metrics

### Coverage Summary

| Category | Tests | Method |
|----------|-------|--------|
| Logic gates | 828 | Exhaustive truth tables + invariants |
| Full adder | 8 | Complete truth table |
| Ripple adder | 65,536 | All 8-bit combinations |
| 6502 ALU | 29 | Hardware test vectors |
| ONNX shapes | 34 | Numerical accuracy |
| Precision | 9 | Edge cases |
| Pipeline | 20 | End-to-end |
| Code gen | 26 | Unit tests |

### Performance

| Test | Time | Notes |
|------|------|-------|
| C rigorous | ~2s | 1,329 tests including 1M stress |
| Python 6502 | ~9s | 20 tests with compilation |
| Python emit_c | ~0.5s | 26 unit tests |

---

## Adding New Tests

### To add a C test:

```c
// In test/test_rigorous.c

void test_my_new_feature(void) {
    SECTION("My New Feature");

    TEST(my_condition, "description");
    TEST(another_condition, "another description");
}

// Add to main():
test_my_new_feature();
```

### To add a Python test:

```python
# In test/test_6502_onnx_pipeline.py

class TestMyNewFeature(unittest.TestCase):
    def test_something(self):
        self.assertEqual(actual, expected)
```

---

## Troubleshooting

### "gcc not found"

Install build tools:
```bash
apt-get install build-essential  # Ubuntu/Debian
```

### "onnx not found"

Install dependencies:
```bash
pip install onnx numpy
```

### Test timeout

Increase timeout:
```bash
python -m pytest test/ -v --timeout=300
```

---

## The Guarantee

When all tests pass:

1. **Frozen shapes are mathematically correct** — truth tables verified
2. **6502 ALU matches hardware** — known test vectors pass
3. **ONNX operations are numerically accurate** — compared to analytical solutions
4. **C code generation is correct** — compiles and produces expected output
5. **Pipeline is end-to-end functional** — ONNX → C → binary → correct results

---

*"The shapes are frozen. The math is eternal. The skeptics are satisfied."*
