# Raw Thoughts: TRIXC ONNX → Executable Pipeline

## Stream of Consciousness

So we need to complete the pipeline from ONNX to native executable. What do we actually have?

We have `onnx2trix.py` which takes an ONNX model and produces a `.trix` file (JSON format, Octave IR). This contains:
- Shape list (the operations: MATMUL, GELU, LAYERNORM, etc.)
- Routing table (order of execution)
- Weights (optionally embedded as JSON arrays)
- Input/output specs

We have the frozen shapes in C headers:
- `onnx_shapes.h` - 40+ ONNX-compatible ops (relu, gelu, matmul, softmax, layer_norm, etc.)
- `shapes.h` - Low-level shapes (XOR, AND, OR, full adder, ripple add)
- `apu.h` - Mixed precision management
- `sparse_octave.h` - Multi-scale memory (a working example!)
- `alu6502.h` - 6502 ALU (another working example!)

The gap is: who reads the .trix JSON and emits the C code that wires it all together?

Looking at sparse_octave.h - it's a self-contained header with all the logic. The pattern is:
1. Define data structures (trix_sparse_octave_t)
2. Init function allocates memory
3. Forward function calls frozen shapes in order
4. Free function cleans up

So the compiler needs to generate something similar but from the .trix spec.

Wait - do we even need a separate compiler? Could we:
1. Generate a single .h file that IS the model?
2. Include it in a trivial main.c?
3. Compile with gcc?

That's the simplest path. The .trix file describes the graph. We emit a .h file that:
- Declares the weight arrays as static const
- Declares a forward() function that chains the shapes

But there are complications:
- Weight data could be huge (GPT-2 is 500MB+)
- Need to handle dynamic shapes (batch size, sequence length)
- Memory management for intermediates
- Broadcasting for element-wise ops

What about just supporting a restricted subset first?
- Fixed shapes (no dynamic dimensions)
- Small models (weights embedded in C)
- Sequential graphs (no branches)

That would cover: simple MLPs, small transformers, basic CNNs.

Actually looking at the existing onnx2trix.py, there's already a `generate_c_header()` function but it's just a stub. That's the natural place to expand.

The real question: what does the generated code look like?

Looking at how sparse_octave_forward works:
```c
void trix_sparse_octave_forward(
    const trix_sparse_octave_t* sol,
    const float* input,
    float* output
) {
    // For each octave...
    // Compute distances (frozen shape)
    // Find top-k (frozen shape)
    // Blend values (frozen shape)
}
```

So for an ONNX model, it would be:
```c
void model_forward(
    const model_t* model,
    const float* input,
    float* output
) {
    // Temp buffers
    float* t0 = alloca(N * sizeof(float));
    float* t1 = alloca(M * sizeof(float));

    // Layer 1: MatMul
    trix_onnx_matmul(input, model->weight0, t0, B, H, D);

    // Layer 2: GELU
    for (int i = 0; i < N; i++) t1[i] = trix_onnx_gelu(t0[i]);

    // Layer 3: LayerNorm
    trix_onnx_layer_norm(t1, model->ln_weight, model->ln_bias, output, N, 1e-5);
}
```

That's not that complicated! The compiler just needs to:
1. Parse the .trix JSON
2. Topologically sort the shapes (already done by ONNX)
3. Allocate temp buffers for intermediates
4. Emit the shape calls in order
5. Wire outputs of one shape to inputs of next

## Questions Arising

- Should the compiler be Python or C?
  - Python: easier JSON parsing, already have onnx2trix.py
  - C: self-contained, but parsing JSON in C is painful

- How to handle weights?
  - Embed as C arrays (simple, but large files)
  - External binary file loaded at runtime
  - Memory-mapped file

- How to handle dynamic shapes?
  - Ignore for v1 (require fixed dimensions)
  - Pass dimensions as runtime parameters

- Intermediate buffer management?
  - alloca() for small models (stack)
  - malloc() for large (heap)
  - Static buffers (no alloc, but fixed size)

- What about in-place operations?
  - Some ops can reuse input buffer as output
  - Optimization for later

## First Instincts

1. Extend onnx2trix.py to emit full C code, not just header stub
2. Start with simplest case: sequential MLP with fixed shapes
3. Weights as static const arrays in C
4. alloca() for intermediates
5. Single generated .c file that includes the headers and contains everything

The "6502 ALU in 6KB" proves this is viable. A small MLP should be similar size.

## What Scares Me

- Large models with huge weight files
- Dynamic shapes requiring runtime dimension handling
- Broadcasting semantics for element-wise ops
- Memory layout differences (row-major vs column-major)
- Getting the shape inference right for intermediates

## What Would Be the Win?

```bash
python onnx2trix.py model.onnx --emit-c model.c
gcc -O3 model.c -o model -lm
./model < input.bin > output.bin
```

Model runs. No Python. No ONNX runtime. Just a native binary.
