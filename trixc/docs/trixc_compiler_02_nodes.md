# Nodes of Interest: TRIXC ONNX → Executable Pipeline

Extracted from RAW phase.

---

## Node 1: The Pipeline Already Exists in Pieces

We have:
- `onnx2trix.py` → produces `.trix` JSON with shapes, weights, routing
- `onnx_shapes.h` → 40+ frozen C implementations
- Working examples: `sparse_octave.h`, `alu6502.h`

**What's missing:** The glue that reads `.trix` and emits the C code.

---

## Node 2: Extend Python vs New C Tool

**Tension:** Should the compiler be Python or C?

| Python | C |
|--------|---|
| Already has onnx2trix.py | Self-contained |
| Easy JSON parsing | Painful JSON parsing |
| Familiar | Lower dependency |
| Can emit .c/.h files easily | Would need to bootstrap |

**Observation:** The output is C code. The tool that generates it doesn't need to be C.

---

## Node 3: The Generated Code Pattern

Looking at `sparse_octave.h`, the pattern is clear:

```c
typedef struct {
    // weights and config
} model_t;

void model_init(model_t* m, ...);
void model_forward(const model_t* m, const float* in, float* out);
void model_free(model_t* m);
```

This is the template. The compiler fills in the blanks.

---

## Node 4: Weight Handling Strategy

Three options:

| Strategy | Pros | Cons |
|----------|------|------|
| Embed in C | Simple, self-contained | Large source files |
| External binary | Small source, large model OK | Runtime loading complexity |
| Hybrid | Best of both | More code paths |

**For v1:** Embed weights. Keeps it simple. Handle large models later.

---

## Node 5: Buffer Management for Intermediates

The forward pass needs temp buffers between shapes.

Options:
- `alloca()` - stack allocation, fast, auto-cleanup, size limited
- `malloc()/free()` - heap, any size, manual cleanup
- Static arrays - no alloc, fixed size, thread-unsafe

**For v1:** `alloca()` for simplicity. Add heap fallback for large tensors.

---

## Node 6: Shape Dimensions Must Be Known

Each shape call needs dimensions:
```c
trix_onnx_matmul(A, B, C, M, N, K);  // Need M, N, K
trix_onnx_gelu(x, y, N);             // Need N
```

**Source of truth:** The `.trix` file has shape info from ONNX.

**Tension:** Dynamic dimensions (batch size, seq length) vs fixed.

**For v1:** Require fixed dimensions. Dynamic = v2.

---

## Node 7: Graph Topology is Already Sorted

ONNX graphs are topologically sorted. The `.trix` file preserves this order.

**Implication:** Just iterate through shapes in order. No need to re-sort.

---

## Node 8: The generate_c_header() Stub

`onnx2trix.py` already has:
```python
def generate_c_header(trix: Dict[str, Any]) -> str:
    # ... stub that just emits forward declaration
```

**This is the natural extension point.** Expand this function.

---

## Node 9: Minimal Viable Compiler Scope

What's the simplest thing that could work?

1. Sequential MLP (no branches, no residuals)
2. Fixed input/output dimensions
3. Embedded weights
4. Single-threaded
5. Float32 only

Covers: classifier heads, small encoders, basic transformers.

---

## Node 10: The Include Strategy

Generated code should include:
```c
#include "trixc/onnx_shapes.h"  // All the frozen shapes
#include "trixc/apu.h"          // Precision helpers
```

**No need to duplicate shape implementations.** Just call them.

---

## Node 11: Memory Layout Assumption

ONNX uses row-major (C-style) layout. Our shapes assume this too.

**Assumption:** Weights in `.trix` are row-major. If not, transpose on load.

---

## Node 12: Error Handling Philosophy

Frozen shapes don't fail (they're math). But:
- Memory allocation can fail
- File loading can fail
- Dimension mismatches are bugs

**For v1:** Assert on errors. No graceful handling. Fail fast.

---

## Node 13: Single File Output vs Multi-File

Options:
- Single `.c` file with everything
- `.h` for declarations, `.c` for implementation
- `.h` for model, separate `weights.bin`

**For v1:** Single `.c` file. Simplest to compile and deploy.

---

## Node 14: Broadcasting Complexity

Element-wise ops (Add, Mul) may need broadcasting:
```
[1, 768] + [768] → broadcast [768] to [1, 768]
```

**This is complex.** ONNX broadcasting rules are non-trivial.

**For v1:** Require matching dimensions. Error on broadcast needed.

---

## Node 15: The Happy Path

```bash
# Convert
python onnx2trix.py model.onnx model.trix

# Generate C
python onnx2trix.py model.onnx --emit-c model.c

# Compile
gcc -O3 model.c -o model -lm

# Run
./model
```

**One command to go from ONNX to executable would be ideal.**

---

## Summary of Tensions

| Tension | Resolution for v1 |
|---------|-------------------|
| Python vs C compiler | Python (extend onnx2trix.py) |
| Embedded vs external weights | Embedded |
| Stack vs heap buffers | Stack (alloca) |
| Dynamic vs fixed shapes | Fixed |
| Broadcasting vs exact match | Exact match |
| Single vs multi-file output | Single .c file |
