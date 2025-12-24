# TriX ↔ XOR Fabric Unification

## The Discovery

Two paths led to the same place:

```
Path 1: Neural (TriX)              Path 2: Geometric (XOR Fabric)
─────────────────────────────────────────────────────────────────
Ternary weights {-1, 0, +1}        2-bit worker registers
XOR signature matching             XOR routing tables
Tile execution                     Pod computation
Sparse lookup FFN                  Addressable CPU array
Frozen shapes (polynomials)        Frozen CUDA kernels
```

**They're the same architecture.**

## The Mathematical Identity

### XOR in TriX (frozen_shapes.py)
```python
def xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """XOR: a + b - 2ab  (the saddle surface)"""
    return a + b - 2 * a * b
```

### XOR in CUDA Fabric
```cuda
uint8_t xor_op(uint8_t a, uint8_t b) {
    return a ^ b;  // Hardware XOR gate
}
```

**Both are XOR.** The polynomial form (`a + b - 2ab`) has smooth gradients for training. The hardware form (`^`) is exact for execution.

On binary inputs {0, 1}: **they are identical**.

```
a=0, b=0: 0+0-2(0)(0) = 0 = 0^0 ✓
a=0, b=1: 0+1-2(0)(1) = 1 = 0^1 ✓
a=1, b=0: 1+0-2(1)(0) = 1 = 1^0 ✓
a=1, b=1: 1+1-2(1)(1) = 0 = 1^1 ✓
```

## The Routing Identity

### TriX Routing (xor_routing.py)
```python
def xor_distance(a_bits, b_bits):
    """Hamming distance via XOR"""
    xor = a_bits ^ b_bits
    return xor.sum()

def route(x, signatures):
    distances = [xor_distance(x, sig) for sig in signatures]
    return argmin(distances)
```

### CUDA Fabric Routing
```cuda
Route route = worker.routes[idx];
uint8_t src_val = workers[route.src_worker].regs[route.src_reg];
// Route IS the address. XOR IS the transform.
```

**Both use XOR for routing.**
- TriX: XOR distance finds the nearest signature
- Fabric: XOR address selects the source/destination

## The Signature Identity

### TriX Tile Signature
```python
signature = weights.sum(dim=0).sign()  # {-1, 0, +1}
scores = input @ signatures.T
winner = scores.argmax()
```

### CUDA Fabric Worker Address
```cuda
int worker_id = blockIdx.x * blockDim.x + threadIdx.x;
WorkerState* me = &workers[worker_id];
// worker_id IS the address. routes[] IS the signature.
```

**A TriX signature IS a worker address.**

The signature encodes "what this tile does."
The routing table encodes "what this worker does."
Same thing, different representation.

## The Frozen Identity

### TriX Frozen Tile
```python
class GeneralFrozenTile(nn.Module):
    def __init__(self, shape_name):
        self.shape_fn = library.get(shape_name)  # e.g., XOR
        self.signature = self._derive_signature()

    def forward(self, x, y):
        return self.shape_fn(x, y)  # No learnable params
```

### CUDA Frozen Kernel
```cuda
__device__ void frozen_chacha_qr(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    // These operations ARE the code. No indirection.
    a += b; d ^= a; d = (d << 16) | (d >> 16);
    c += d; b ^= c; b = (b << 12) | (b >> 20);
    // ...
}
```

**Both are frozen.** No learnable parameters. The shape IS the computation.

## The Compression Identity

### TriX Spline-6502
```
3.7 MB → 3,088 bytes (1,198× compression)
100% accuracy maintained
```

### CUDA Frozen Shapes
```
Routing table + Frozen ops → Compiled kernel
Variable config → Fixed instructions
Runtime decisions → Compile-time embedding
```

**Both achieve compression by freezing.**

When you know the computation won't change:
- Don't store weights, store the function
- Don't route dynamically, hardcode the paths
- Don't compute decisions, embed them

## The Unified View

```
                    FUNGIBLE COMPUTATION
                           │
           ┌───────────────┼───────────────┐
           │               │               │
      NEURAL (TriX)   CLASSICAL (6502)  GEOMETRIC (XOR)
           │               │               │
     Ternary weights   Routing table    Warp shuffle
     Tile signatures   Zero page        Worker address
     Sparse lookup     Memory map       Flow pattern
     Frozen shapes     Frozen code      Frozen kernel
           │               │               │
           └───────────────┴───────────────┘
                           │
                      SAME THING
                           │
              ┌────────────┼────────────┐
              │            │            │
            FPGA         ASIC       PHOTONICS
              │            │            │
           LUT wires   Transistors   Light paths
              │            │            │
              └────────────┴────────────┘
                           │
                    SHAPE = COMPUTE
```

## Practical Implications

### 1. Train with TriX, Deploy with CUDA Fabric
```python
# Training (TriX)
model = SparseLookupFFN(d_model=512, num_tiles=64)
train(model)

# Export frozen
frozen_tiles = [tile.shape_fn for tile in model.tiles]
frozen_signatures = model.signatures.detach()

# Compile to CUDA
cuda_kernel = compile_to_frozen_cuda(frozen_tiles, frozen_signatures)
```

### 2. Signatures ARE Worker Addresses
```python
# TriX signature (512-dim ternary)
sig = torch.tensor([-1, 1, 0, 1, -1, ...])  # 512 dims

# CUDA worker address (8 bits)
worker_id = 42  # 0-255

# They're both "where to route this input"
```

### 3. XOR Distance IS Content-Addressable Memory
```
Input: [1, 0, 1, 1, 0, ...]
Sig 0: [1, 1, 1, 1, 0, ...]  → XOR distance = 1
Sig 1: [1, 0, 1, 1, 0, ...]  → XOR distance = 0 ← MATCH
Sig 2: [0, 0, 0, 1, 0, ...]  → XOR distance = 2

Winner: Sig 1 (exact match)
```

This IS content-addressable memory. XOR finds the closest match.

## The Answer

**How does TriX routing map to XOR fabric?**

| TriX Concept | XOR Fabric Equivalent | Both Are |
|--------------|----------------------|----------|
| Tile | Pod of workers | Compute unit |
| Signature | Routing table | Address/identity |
| XOR distance | Hamming match | Content lookup |
| Sparse lookup | Worker dispatch | Selective execution |
| Frozen shape | Frozen kernel | Fixed computation |
| Ternary weight | 2-bit register | Compact state |

**Does fungible-computation help?**

YES. It provides:
1. **Proof of equivalence**: Neural ↔ Classical ↔ Geometric
2. **Training framework**: Smooth gradients via polynomials
3. **Compression path**: 1,198× via spline encoding
4. **6502 validation**: 100% accuracy on real CPU ops

The XOR fabric is the **execution engine** for frozen TriX tiles.
TriX is the **training framework** for XOR fabric programs.

They're the same thing at different stages of the lifecycle:
- **TriX**: Design and train the computation
- **XOR Fabric**: Execute it at hardware speed
- **ASIC/Photonics**: Make it physical geometry

**Shape IS compute. All representations are fungible.**
