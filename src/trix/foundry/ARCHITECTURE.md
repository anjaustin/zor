# Hollywood Squares Foundry - Architecture

Deep dive into the system design and core principles.

## Core Principle: Position IS Computation

Traditional neural networks compute:
```
output = matmul(input, weights) + bias
```

Hollywood Squares computes:
```
output = input + lookup[position(input)]
```

Where:
- **position** is determined by content and spatial location
- **lookup** is frozen geometry (spline coefficients, directions)
- Values flow through structure like electrons through transistors

## System Overview

```
                        ┌─────────────────────────────────────────┐
                        │            User Application             │
                        └─────────────────────────────────────────┘
                                           │
                        ┌──────────────────┼──────────────────┐
                        ▼                  ▼                  ▼
              ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
              │ NativeHollywood │ │    Trainer      │ │   HollywoodS.   │
              │    Squares      │ │                 │ │   Emergence     │
              │   (Training)    │ │  (Loss + Loop)  │ │  (Inference)    │
              └─────────────────┘ └─────────────────┘ └─────────────────┘
                        │                  │                  │
                        └──────────────────┼──────────────────┘
                                           ▼
              ┌─────────────────────────────────────────────────────────┐
              │                    CUDA Kernels                          │
              │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │
              │  │  Forward   │  │  Backward  │  │  Ternary Packing   │  │
              │  │  Kernel    │  │  Kernel    │  │  INT8 Quantization │  │
              │  └────────────┘  └────────────┘  └────────────────────┘  │
              └─────────────────────────────────────────────────────────┘
                                           │
              ┌─────────────────────────────────────────────────────────┐
              │                    CuPy Arrays                           │
              │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐  │
              │  │ Signatures │  │ Directions │  │  Spline Coeffs     │  │
              │  │  (uint32)  │  │   (int8)   │  │     (float32)      │  │
              │  └────────────┘  └────────────┘  └────────────────────┘  │
              └─────────────────────────────────────────────────────────┘
                                           │
              ┌─────────────────────────────────────────────────────────┐
              │                     NVIDIA GPU                           │
              │        Shared Memory │ Global Memory │ Registers          │
              └─────────────────────────────────────────────────────────┘
```

## Data Flow

### Forward Pass

```
Input x [batch, d_model]
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ 1. CONTENT SCORING                                                    │
│                                                                       │
│    For each tile t in [0, num_tiles):                                │
│        content_score[t] = dot(x, signature[t])                        │
│                                                                       │
│    signature[t] is ternary {-1, 0, +1}, packed as 2 bits             │
│    16 values per uint32 = 16x compression                            │
│                                                                       │
│    ┌──────────────────────────────────────────────────┐              │
│    │ uint32 packed: [01][10][00][01][10][01][00][10]... │              │
│    │ Meaning:       [-1][+1][ 0][-1][+1][-1][ 0][+1]... │              │
│    └──────────────────────────────────────────────────┘              │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ 2. SPATIAL WEIGHTING                                                  │
│                                                                       │
│    position = [0, 1, 2, ...] for each sample in batch                │
│                                                                       │
│    For each tile t:                                                   │
│        tile_center = t * (num_tiles / spread)                        │
│        dist = (position - tile_center) / spread                       │
│        spatial_weight[t] = cubic_bspline(dist)                       │
│                                                                       │
│    Cubic B-spline:                                                    │
│    ┌────────────────────────────────────────────────────┐            │
│    │                     ╭───╮                          │            │
│    │                   ╭─╯   ╰─╮                        │            │
│    │                 ╭─╯       ╰─╮                      │            │
│    │               ╭─╯           ╰─╮                    │            │
│    │   ──────────╯                 ╰────────────        │            │
│    │  -2        -1         0         1         2        │            │
│    └────────────────────────────────────────────────────┘            │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ 3. TILE SELECTION                                                     │
│                                                                       │
│    combined_score[t] = content_score[t] * spatial_weight[t]          │
│    best_tile = argmax(combined_score)                                 │
│                                                                       │
│    Content-spatial balance:                                           │
│    - High content score = tile "understands" this input              │
│    - High spatial weight = tile "owns" this position                 │
│    - Best tile = expert for this (content, position) pair            │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ 4. SPLINE LOOKUP                                                      │
│                                                                       │
│    Compress input to 2D coordinates:                                  │
│        a = tanh(x[0] * 0.1)  ∈ [-1, 1]                               │
│        b = tanh(x[1] * 0.1)  ∈ [-1, 1]                               │
│                                                                       │
│    Map to grid indices:                                               │
│        u = (a + 1) / 2 * (grid_size - 1)                             │
│        v = (b + 1) / 2 * (grid_size - 1)                             │
│                                                                       │
│    Bilinear interpolation:                                            │
│        scale = interpolate(spline_coeffs[best_tile], u, v)           │
│                                                                       │
│    ┌────────────────────────────────────────┐                        │
│    │  grid_size x grid_size coefficients    │                        │
│    │                                         │                        │
│    │    ●───●───●───●───●                   │                        │
│    │    │   │   │   │   │                   │                        │
│    │    ●───●───●───●───●                   │                        │
│    │    │   │ ◆ │   │   │  ◆ = (u,v)        │                        │
│    │    ●───●───●───●───●                   │                        │
│    │    │   │   │   │   │                   │                        │
│    │    ●───●───●───●───●                   │                        │
│    └────────────────────────────────────────┘                        │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ 5. OUTPUT WITH RESIDUAL                                               │
│                                                                       │
│    direction = directions[best_tile]  # INT8 quantized, 4x compressed│
│    output = x + scale * direction     # Residual connection          │
│                                                                       │
│    Residual ensures:                                                  │
│    - Identity mapping possible (when scale ≈ 0)                      │
│    - Gradient flows through (d_input = d_output)                     │
│    - Stable training (no vanishing gradients)                        │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
Output y [batch, d_model]
```

### Backward Pass

```
Gradient d_output [batch, d_model]
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ 1. DIRECTION GRADIENTS                                                │
│                                                                       │
│    For each sample i:                                                 │
│        tile = best_tile[i]  (from forward pass)                      │
│        scale = cached_scale[i]                                        │
│                                                                       │
│        d_directions[tile] += d_output[i] * scale                     │
│                             (atomic add for thread safety)            │
│                                                                       │
│    Gradient interpretation:                                           │
│    - Direction gradient = "which way should this tile point?"        │
│    - Accumulated from all samples that used this tile                │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ 2. SPLINE COEFFICIENT GRADIENTS                                       │
│                                                                       │
│    Using cached (u, v) coordinates from forward:                      │
│                                                                       │
│        d_scale = dot(d_output[i], directions[tile])                  │
│        d_coeffs = d_scale * bilinear_basis(u, v)                     │
│                                                                       │
│    Bilinear basis distributes gradient to 4 neighbors:               │
│        d_coeffs[j,k] += d_scale * (1-du) * (1-dv)                    │
│        d_coeffs[j+1,k] += d_scale * du * (1-dv)                      │
│        d_coeffs[j,k+1] += d_scale * (1-du) * dv                      │
│        d_coeffs[j+1,k+1] += d_scale * du * dv                        │
│                                                                       │
│    (All with atomicAdd)                                               │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│ 3. INPUT GRADIENTS (Residual Pass-Through)                            │
│                                                                       │
│    d_input = d_output  (identity, from residual connection)          │
│                                                                       │
│    Note: We do NOT backprop through:                                  │
│    - Signature computation (ternary, non-differentiable)             │
│    - Tile selection (argmax, non-differentiable)                     │
│    - Position encoding (discrete, non-differentiable)                │
│                                                                       │
│    Only trainable parameters receive gradients:                       │
│    - directions [num_tiles, d_model]                                 │
│    - spline_coeffs [num_tiles, grid_size, grid_size]                 │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
Gradient d_input [batch, d_model]
```

## Weight Compression

### Ternary Signature Packing

```
Original: float32 signatures [num_tiles, d_model]
          Each value ∈ {-1.0, 0.0, +1.0}

Compressed: uint32 packed_signatures [num_tiles, d_model // 16]
            16 values per uint32 (2 bits each)

Encoding:
  -1.0 → 0b00  (value 0)
   0.0 → 0b01  (value 1)
  +1.0 → 0b10  (value 2)

Compression: 32 bits / 2 bits = 16x

Decoding:
  bits = (packed >> (idx * 2)) & 0x3
  value = bits - 1  // Gives {-1, 0, +1}
```

### INT8 Direction Quantization

```
Original: float32 directions [num_tiles, d_model]

Step 1: Find scale
  max_abs = max(|directions|)
  scale = 127.0 / max_abs

Step 2: Quantize
  int8_directions = round(directions * scale)
  int8_directions = clamp(int8_directions, -127, 127)

Step 3: Dequantize (at runtime)
  directions_approx = int8_directions / scale

Compression: 32 bits / 8 bits = 4x
Error: <1% typical (max 127/128 = 0.78% per step)
```

## Memory Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GPU GLOBAL MEMORY                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  packed_signatures: [num_tiles × (d_model/16)]                      │
│  ┌──────────────────────────────────────────┐                       │
│  │ uint32 │ uint32 │ uint32 │ ... │ uint32 │  × num_tiles           │
│  └──────────────────────────────────────────┘                       │
│  Size: num_tiles × d_model / 16 × 4 bytes                           │
│  Example: 16 tiles × 128 dims = 512 bytes (vs 8KB uncompressed)     │
│                                                                      │
│  int8_directions: [num_tiles × d_model]                             │
│  ┌──────────────────────────────────────────┐                       │
│  │ int8 │ int8 │ int8 │ ... │ int8 │  × num_tiles                   │
│  └──────────────────────────────────────────┘                       │
│  Size: num_tiles × d_model bytes                                    │
│  Example: 16 × 128 = 2KB (vs 8KB uncompressed)                      │
│                                                                      │
│  spline_coeffs: [num_tiles × grid_size × grid_size]                 │
│  ┌──────────────────────────────────────────┐                       │
│  │ float32 │ float32 │ ... │  × grid_size²  │  × num_tiles          │
│  └──────────────────────────────────────────┘                       │
│  Size: num_tiles × grid_size² × 4 bytes                             │
│  Example: 16 × 8 × 8 × 4 = 4KB                                      │
│                                                                      │
│  TOTAL WEIGHTS: ~6.5 KB (vs 24 KB uncompressed) = 3.7x compression  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         GPU SHARED MEMORY                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Per-block tile cache:                                               │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ signatures_cache: [num_tiles × (d_model/16)] = 512 bytes       │ │
│  │ directions_cache: [num_tiles × d_model] = 2KB                  │ │
│  │ spline_cache: [num_tiles × grid_size²] = 4KB                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  Total shared per block: ~6.5 KB                                    │
│  Max shared memory: 48 KB (typical)                                 │
│  Headroom: 41.5 KB for other uses                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## CUDA Kernel Design

### Thread Block Organization

```
Grid: (num_blocks,)
  where num_blocks = (batch_size + block_size - 1) // block_size

Block: (block_size,)
  where block_size = 256 (typical)

Each thread processes one sample:
  sample_idx = blockIdx.x * blockDim.x + threadIdx.x

Cooperative loading (first N threads load shared memory):
  if threadIdx.x < num_tiles:
      load_tile_data(threadIdx.x)
  __syncthreads()
```

### Atomic Gradient Accumulation

```cuda
// Multiple threads may update same tile
// Use atomicAdd for correctness

// Direction gradients
for (int d = 0; d < d_model; d++) {
    atomicAdd(&d_directions[tile * d_model + d],
              d_output[d] * scale);
}

// Spline coefficient gradients
atomicAdd(&d_coeffs[tile * grid_size * grid_size + j * grid_size + k],
          d_scale * (1.0f - du) * (1.0f - dv));
```

## Adam Optimizer

Pure CuPy implementation:

```python
class AdamOptimizer:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.t = 0

        # Initialize moments
        self.m = {k: cp.zeros_like(v) for k, v in params.items()}
        self.v = {k: cp.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.t += 1

        for name, param in self.params.items():
            grad = grads[name]

            # Update biased moments
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad**2

            # Bias correction
            m_hat = self.m[name] / (1 - self.beta1**self.t)
            v_hat = self.v[name] / (1 - self.beta2**self.t)

            # Update parameters
            param -= self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)
```

## Why Native Beats PyTorch

### PyTorch Overhead

```
PyTorch forward pass:
  1. Create computation graph node for input
  2. Allocate output tensor
  3. Dispatch to CUDA kernel
  4. Record graph edge
  5. Repeat for each operation

For SparseLookupFFNv4:
  - 20+ operations
  - 20+ tensor allocations
  - 20+ graph nodes
  - Dynamic dispatch overhead
```

### Native Approach

```
Native forward pass:
  1. Launch single fused kernel
  2. All computation in registers/shared memory
  3. Single output write

Advantages:
  - No graph construction
  - No intermediate allocations
  - No dispatch overhead
  - Memory coalesced access
  - All in one kernel launch
```

### Measured Results

```
Training:
  Native:  368,000 samples/sec
  PyTorch:   4,300 samples/sec
  Speedup: 85x

Inference:
  Native:  44.5M tokens/sec
  PyTorch:  5.7M tokens/sec
  Speedup: 7.8x
```

## Parameter Count

```
Trainable parameters:
  directions:    num_tiles × d_model
  spline_coeffs: num_tiles × grid_size × grid_size

Example (16 tiles, 128 dims, 8×8 grid):
  directions:    16 × 128 = 2,048
  spline_coeffs: 16 × 64 = 1,024
  TOTAL: 3,072 parameters

Compared to MLP with same dims:
  hidden = 4 × d_model = 512
  W1: 128 × 512 = 65,536
  W2: 512 × 128 = 65,536
  TOTAL: 131,072 parameters

Hollywood Squares: 43x fewer parameters!
```

## File Structure

```
src/trix/foundry/
├── native_training.py           # Complete training system
│   ├── NativeHollywoodSquares   # Model class
│   ├── Trainer                  # Training loop
│   ├── AdamOptimizer            # Pure CuPy Adam
│   ├── mse_loss()               # MSE with gradients
│   └── cross_entropy_loss()     # Cross-entropy with gradients
│
├── hollywood_squares_emergence.py  # Optimized inference
│   └── HollywoodSquaresEmergence   # Compressed weights
│
├── hollywood_squares_ffn_fast.py   # Shared memory variant
│   └── HollywoodSquaresFFNFast     # Block-level parallelism
│
├── hollywood_squares_ffn.py        # Basic GPU kernel
│   └── HollywoodSquaresFFN         # Reference implementation
│
├── test_hollywood_squares.py       # 12 rigorous tests
├── ab_test_harness.py              # Native vs PyTorch A/B tests
├── benchmark_harness.py            # Reproducible benchmarks
├── benchmark_training.py           # Training speed comparison
│
├── README.md                       # Overview documentation
├── QUICKSTART.md                   # Getting started guide
├── ARCHITECTURE.md                 # This file
├── API.md                          # API reference
├── GAPS.md                         # Known limitations
└── CHANGELOG.md                    # Version history
```
