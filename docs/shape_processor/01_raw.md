# Raw Thoughts: Shape Processor

## Stream of Consciousness

A chip that processes shapes. Not a CPU that runs shape code - a chip WHERE shapes ARE the computation. What does that even mean?

Current state: We generate C code that evaluates polynomials. The C runs on a CPU. The CPU does fetch-decode-execute. Massive overhead. We're simulating math on a logic machine.

What if the chip WAS the polynomial? Like an FPGA, but instead of LUTs and routing, it's multiply-accumulate units wired for polynomial evaluation.

Wait. What are we actually computing?

```
XOR(a,b) = a + b - 2ab
AND(a,b) = ab
OR(a,b) = a + b - ab
NOT(a) = 1 - a
```

These are ALL just multiply-accumulate with small integer coefficients. The coefficients are always -2, -1, 0, 1, 2. That's it. We don't need a general multiplier. We need:
- Multiply by -2, -1, 0, 1, 2
- Add results

Multiply by 2 is a shift. Multiply by -1 is negate. Multiply by 0 is zero. This is trivial in hardware.

So a "Polynomial Engine" is just:
- Two inputs (a, b)
- Hardwired coefficient selection (-2, -1, 0, 1, 2)
- Adder tree
- Output

For XOR: output = a*1 + b*1 + ab*(-2) = a + b - 2ab

This is ONE CYCLE if we pipeline it right. No instruction fetch. No decode. Just data flow.

But wait - shapes are compositions. XOR of XOR of AND of... How do we chain them?

Option 1: Cascade units. Output of PE1 feeds PE2. Like a pipeline.
Option 2: Reconfigurable routing. Like an FPGA.
Option 3: Time-multiplexed. Same PE, multiple cycles.

Option 1 is fastest but inflexible - chip is hardwired to one shape.
Option 2 is flexible but complex - we're just building another FPGA.
Option 3 is a middle ground - programmable but sequential.

Actually, what's the use case? We want to:
1. Load a shape (the polynomial structure)
2. Feed inputs
3. Get outputs
4. Repeat for different inputs

The shape doesn't change often. The inputs change constantly. So we want:
- Slow configuration (load shape)
- Fast evaluation (process inputs)

This is like a GPU shader. You compile the shader (slow), then run it on millions of pixels (fast).

Shape Processor = Polynomial Shader Processor?

What about the carry chain problem? For adders, we have ripple carry. Each bit depends on the previous. This is inherently sequential. Can we parallelize?

Carry-lookahead! We can compute carries in parallel with O(log n) depth instead of O(n). But that's more hardware.

Or... we keep the polynomial representation. In polynomial form, there IS no carry chain. The polynomial for sum bit 15 is a closed-form expression. It's huge, but it's parallel.

Wait. That's the insight. The expanded polynomial is FULLY PARALLEL. No dependencies between terms. We can evaluate all terms simultaneously and sum them.

The factored form (carry variables) is sequential.
The expanded form is parallel but huge.
Trade-off: space vs time.

For a dedicated chip, we might want the expanded form. Burn silicon for parallelism.

But the expanded form for a 32-bit adder is megabytes of polynomial. That's millions of terms. We can't have millions of multiply units.

Unless... we use the STRUCTURE. The polynomial has patterns. Symmetry. We don't need to evaluate every term independently - we can share computation.

This is like FFT. Naive DFT is O(n^2). FFT exploits structure for O(n log n). Can we find the "FFT of polynomials"?

Hmm. The ripple carry structure IS the factored form. Each carry depends on previous. That's O(n) depth.

Carry-lookahead gives O(log n) depth but O(n log n) hardware.

Is there something better for polynomials specifically?

Actually, for our use case (evaluating shapes on binary inputs), we have a constraint: all inputs are 0 or 1. This means:
- a^2 = a (idempotent)
- ab = min(a,b) for binary
- a + b - ab = max(a,b) for binary

Can we exploit this? Instead of general polynomial evaluation, we're doing Boolean polynomial evaluation.

In hardware, binary signals are just wires. a*b is an AND gate. a + b - ab is an OR gate. We're... back to logic gates?

No wait. The polynomial form lets us do something different. We can evaluate on ANALOG signals, not just digital. Or on probabilistic bits. Or on error-corrected encoded bits.

But for now, assume binary. Then polynomial evaluation = gate evaluation. Our "Shape Processor" is just... a CPU? An FPGA?

No. The difference is HOW we specify the computation. We don't write gate netlists. We write polynomials. The chip interprets polynomials directly.

Think of it as a polynomial interpreter in hardware. The "instruction" is a polynomial coefficient. The "execution" is multiply-accumulate.

Data format for a shape:
```
SHAPE := list of TERMS
TERM := coefficient, list of VARIABLES
VARIABLES := indices into input vector
```

Example: XOR(a,b) = a + b - 2ab
```
SHAPE = [
  (1, [0]),      # +1 * a
  (1, [1]),      # +1 * b
  (-2, [0, 1])   # -2 * a * b
]
```

The processor:
1. Reads SHAPE from memory
2. For each TERM: multiply variables, scale by coefficient
3. Sum all terms
4. Output result

This is a dot product! SHAPE is a sparse vector of coefficients. INPUT is a vector of variable products. Output is dot product.

Dot products are HIGHLY optimized in hardware. GPUs do billions per second. TPUs are built for this.

Are we just building a TPU for polynomials?

Kind of. But with a twist:
- Coefficients are tiny (-2 to 2)
- Variables are binary (0 or 1)
- Structure is sparse (most terms are zero)

This is more like a sparse binary neural network than a dense float matrix multiply.

Hmm. What if we use existing TPU/GPU hardware and encode shapes as sparse matrices?

SHAPE as a matrix:
- Rows = output bits
- Columns = input bit products (1, a, b, ab, c, ac, bc, abc, ...)
- Values = coefficients (-2, -1, 0, 1, 2)

Input vector: all products of input bits (exponentially many, but sparse - only evaluate non-zero terms).

Matrix-vector multiply gives output.

But this is exponential in input size. 32-bit adder has 64 input bits. 2^64 possible products. That's not sparse, that's impossible.

Back to structure. We MUST exploit structure. The factored form does this - O(n) intermediates instead of O(2^n) terms.

So the chip architecture should support:
1. Polynomial primitives (XOR, AND, OR, NOT)
2. Composition (output of one feeds input of next)
3. Named intermediates (carry variables)

This is a dataflow architecture. Like a systolic array but for polynomials.

Systolic array for shapes:
```
IN → [PE] → [PE] → [PE] → [PE] → OUT
       ↓      ↓      ↓      ↓
      c1     c2     c3     c4    (carry outputs)
```

Each PE computes one bit position. Takes a, b, carry_in. Outputs sum, carry_out.

The PEs are hardwired for the polynomial:
```
sum = XOR(XOR(a, b), c_in)
c_out = OR(AND(a, b), AND(c_in, XOR(a, b)))
```

This is just a hardware adder! We've reinvented the wheel!

Or have we? The difference is we can RECONFIGURE the PEs for different polynomials. Not just adders - any shape.

A reconfigurable PE might have:
- Inputs: a, b, c, d (4 inputs)
- Operation select: XOR, AND, OR, NOT, ADD, MUL
- Coefficient select: -2, -1, 0, 1, 2 for each input
- Routing: which output goes where

This is... a coarse-grained reconfigurable array (CGRA). These exist. Xilinx, Intel, startups have them.

What's our angle? What's different about a SHAPE processor vs a generic CGRA?

1. Native polynomial representation (not gates)
2. Algebraic optimization at configuration time
3. Built-in certification (proof by structure)
4. Optimized for the specific primitives we use

Actually, the certification angle is interesting. If the chip is DESIGNED around proven-correct polynomial primitives, the hardware itself is provably correct. Not just the shapes - the silicon.

Formally verified hardware that executes formally verified shapes. Correctness all the way down.

## Questions
- What's the minimum viable architecture?
- FPGA soft-core first, or straight to ASIC?
- What's the market? Who buys this?
- How do we compete with GPUs/TPUs for matrix ops?
- Is this actually better than just running C code on a normal CPU?

## First Instinct
Start with a Verilog soft-core that implements a simple dataflow polynomial evaluator. Prove it works on FPGA. Then consider ASIC if there's demand.

The killer app might be: run an entire frozen CPU (6502, Z80) as a shape on our shape processor. Meta-level: chip running a chip.
