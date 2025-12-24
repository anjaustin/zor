# Frozen Shapes: Computation as Geometry

*Mesa 14: The First Neural Net of Frozen Shapes*

---

## Overview

Frozen Shapes are fixed mathematical structures that compute exactly. Unlike learned neural network weights, frozen shapes:

- Have **zero learned parameters** in the computation path
- Achieve **100% accuracy** by construction
- Derive **signatures from structure** (not learned)
- Enable **40x compression** on applicable domains

**Core Thesis:** *"Computation is topology. Learning is routing."*

The shapes ARE the geometry - discovered, not learned. Only the routing (meaning layer) is learned.

---

## Architecture

```
Level 0: Pure Math Primitives (0 params)
    XOR, AND, OR, NOT - continuous polynomial forms

Level 1: Frozen Shapes (0 params)
    16 topologies composed from Level 0 primitives

Level 2: Meaning Layer (~2,500 params)
    Learned opcode/content -> shape routing
```

### Module Structure

```
src/trix/nn/
├── frozen.py           # Core infrastructure
│   ├── FrozenShape     # Verified computation primitive
│   ├── FrozenTile      # nn.Module wrapper
│   ├── FrozenShapeRegistry  # Shape catalog
│   └── FrozenTriXFFN   # Routing layer
│
└── frozen_6502.py      # 6502-specific shapes
    ├── PureMath        # Boolean polynomials
    ├── FrozenShapes6502  # 16 topologies
    ├── Frozen6502Tile  # Tile wrapper
    └── Frozen6502      # Complete model
```

---

## Theory

### Why Frozen Shapes Work

Boolean logic can be expressed as continuous polynomials that preserve gradients while computing exactly on {0, 1}:

```python
XOR(a, b) = a + b - 2ab    # Saddle surface
AND(a, b) = ab             # Product
OR(a, b)  = a + b - ab     # Union
NOT(a)    = 1 - a          # Reflection
```

These formulas:
1. Compute correctly on bit inputs
2. Are differentiable everywhere
3. Compose to form complex circuits

### The Signature Insight

Frozen shapes derive signatures from their **truth tables**:

```python
def derive_signature(circuit, sig_dim=64):
    # 1. Generate truth table
    truth_table = circuit(all_inputs)

    # 2. Project to fixed dimension
    projection = random_projection(truth_table, sig_dim)

    # 3. Sign to get ternary
    signature = projection.sign()

    return signature
```

This makes signatures:
- **Deterministic** - Same circuit, same signature
- **Semantic** - Signature reflects what the function computes
- **Content-addressable** - Similar functions, similar signatures

### FP4 Atoms ARE Frozen Shapes

The compiler's FP4 atoms (`atoms_fp4.py`) are the same concept:

| FP4 Atom | Frozen Equivalent |
|----------|------------------|
| `make_AND()` | `PureMath.and_op()` |
| `make_XOR()` | `PureMath.xor()` |
| `make_FULLADDER()` | `FrozenShapes6502.ripple_add` (single bit) |

The infrastructure unifies them with a common interface.

---

## API Reference

### FrozenShape

```python
@dataclass
class FrozenShape:
    name: str                    # Unique identifier
    circuit: ThresholdCircuit    # FP4 computation
    signature: torch.Tensor      # [sig_dim] ternary
    num_inputs: int
    num_outputs: int
    description: str = ""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute frozen computation."""

    def verify(self) -> Tuple[bool, float]:
        """Verify 100% accuracy."""
```

### FrozenTile

```python
class FrozenTile(nn.Module):
    """TriX-compatible tile wrapper."""

    def __init__(self, shape: FrozenShape, tile_id: int = 0):
        ...

    def get_signature(self) -> torch.Tensor:
        """Return fixed signature for routing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute frozen computation."""

    @property
    def usage_rate(self) -> float:
        """Get activation frequency."""
```

### FrozenShapeRegistry

```python
class FrozenShapeRegistry:
    """Catalog of available shapes."""

    def __init__(self, sig_dim: int = 64):
        ...

    def register(self, name: str, circuit: ThresholdCircuit) -> FrozenShape:
        """Register a new shape."""

    def get(self, name: str) -> FrozenShape:
        """Get shape by name."""

    def list_shapes(self) -> List[str]:
        """List all registered shapes."""

# Global default
registry = get_default_registry()
```

### FrozenTriXFFN

```python
class FrozenTriXFFN(nn.Module):
    """FFN with frozen tiles and content/opcode routing."""

    def __init__(
        self,
        tiles: List[FrozenTile],
        use_meaning_layer: bool = False,
        num_opcodes: int = 56,
    ):
        ...

    def forward(
        self,
        x: torch.Tensor,
        opcode_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Route and execute."""

    def init_meaning_from_spec(self, spec: Dict[int, int]):
        """Initialize routing from opcode->tile spec."""

    def param_count(self) -> int:
        """Count learnable parameters (meaning layer only)."""
```

---

## Quick Start

### Basic Usage

```python
from trix.nn import (
    get_default_registry,
    FrozenTile,
    create_frozen_ffn_from_names,
)

# Get builtin shapes
registry = get_default_registry()
print(registry.list_shapes())  # ['AND', 'OR', 'XOR', ...]

# Create a frozen tile
xor_shape = registry.get("XOR")
tile = FrozenTile(xor_shape)

# Execute
x = torch.tensor([[0, 1], [1, 1]], dtype=torch.float32)
y = tile(x)  # [[1], [0]]

# Create a frozen FFN with multiple tiles
ffn = create_frozen_ffn_from_names(["AND", "OR", "XOR"])
out = ffn(x)  # Routes to appropriate tile
```

### 6502 Emulation

```python
from trix.nn import create_frozen_6502, int_to_bits, bits_to_int

# Create 6502 model
model = create_frozen_6502()
model.eval()

# Execute ADC: A + M + C
registers = torch.zeros(1, 7, 8)
registers[0, 0, :] = int_to_bits(torch.tensor([42]))[0]  # A = 42
memory = int_to_bits(torch.tensor([13]))  # M = 13
carry = torch.tensor([1.0])  # C = 1
opcode = torch.tensor([0])  # ADC

out = model(opcode, registers, memory, carry)
result = bits_to_int(out['result'][0])  # 56 = 42 + 13 + 1
```

### Custom Shapes

```python
from trix.nn import FrozenShapeRegistry
from trix.compiler.atoms_fp4 import truth_table_to_circuit

# Create custom registry
registry = FrozenShapeRegistry(auto_register_builtins=False)

# Register custom shape from truth table
truth_table = {
    (0, 0): 0,
    (0, 1): 1,
    (1, 0): 1,
    (1, 1): 0,
}
circuit = truth_table_to_circuit("MY_XOR", 2, truth_table)
shape = registry.register("MY_XOR", circuit)

# Use it
tile = FrozenTile(shape)
```

---

## The 16 6502 Shapes

| ID | Name | Function | Used By |
|----|------|----------|---------|
| 0 | RIPPLE_ADD | 8-bit addition | ADC |
| 1 | RIPPLE_SUB | 8-bit subtraction | SBC, CMP |
| 2 | PARALLEL_AND | 8-bit AND | AND |
| 3 | PARALLEL_OR | 8-bit OR | ORA |
| 4 | PARALLEL_XOR | 8-bit XOR | EOR |
| 5 | SHIFT_LEFT | ASL | ASL |
| 6 | SHIFT_RIGHT | LSR | LSR |
| 7 | ROTATE_LEFT | ROL | ROL |
| 8 | ROTATE_RIGHT | ROR | ROR |
| 9 | INCREMENT | +1 | INC, INX, INY |
| 10 | DECREMENT | -1 | DEC, DEX, DEY |
| 11 | TRANSFER | passthrough | TAX, TXA, etc. |
| 12 | LOAD | from memory | LDA, LDX, LDY |
| 13 | STORE | to memory | STA, STX, STY |
| 14 | BIT_TEST | A AND M + flags | BIT |
| 15 | IDENTITY | no-op | NOP, flags |

---

## Compression Analysis

### Traditional Approach
- 56 opcodes x ~2000 params each = ~100,000 parameters
- Learns separate weights for each operation
- Accuracy depends on training

### Frozen Shapes
- 16 shapes x 0 learned params = 0 computation params
- Meaning layer: 56 x 45 = ~2,500 params
- **Total: ~2,500 params (40x compression)**
- 100% accuracy by construction

---

## Integration with TriX

Frozen tiles implement the same interface as learned TriX tiles:

```python
# Same interface
tile.get_signature()  # For routing
tile.forward(x)       # For computation
tile.usage_rate       # For monitoring
```

This enables:
1. **Hybrid architectures** - Mix frozen and learned tiles
2. **Progressive freezing** - Freeze tiles as they converge
3. **Routing reuse** - Use HierarchicalTriXFFN routing
4. **Guardian monitoring** - Track frozen tile usage

---

## Testing

Run frozen shape tests:

```bash
# Unit tests
pytest tests/test_frozen.py -v

# Integration tests (100% accuracy validation)
pytest tests/test_frozen_6502.py -v

# Exhaustive 6502 tests
pytest tests/test_frozen_6502.py -k exhaustive -v
```

---

## Performance

| Metric | Value |
|--------|-------|
| Inference latency | ~1ms (CPU, batch=1) |
| Parameter count | ~2,500 (6502 model) |
| Memory footprint | <1MB |
| Accuracy | 100% by construction |

---

## ONNX Export

Frozen shapes can be exported to ONNX for portable execution.

### Export the 6502 Network

```python
from trix.nn import Frozen6502Net
import torch

net = Frozen6502Net()
# ... see FROZEN_6502_NET.md for full export code
torch.onnx.export(wrapper, inputs, "frozen_6502.onnx")
```

### What's in the ONNX

The 73KB file contains 915 nodes:

| Node Type | Count | Purpose |
|-----------|-------|---------|
| Mul | 243 | `ab` (AND), `2ab` (XOR term) |
| Add | 131 | Sum terms |
| Sub | 109 | `a + b - 2ab` (XOR), `a + b - ab` (OR) |
| Constant | 179 | Fixed values |

### Geometry Size Analysis

Individual shape sizes when exported separately:

```
ripple_add       : 12 KB  ← 8 full-adders chained
ripple_sub       : 12 KB
increment        : 15 KB  ← Largest (add with constant)
decrement        : 13 KB
parallel_xor     : 443 B  ← Just: a + b - 2ab per bit
parallel_or      : 307 B
parallel_and     : 215 B
shift/rotate     : ~900 B
transfer/identity: ~200 B
```

The carry-chain operations dominate because a full-adder is ~16 arithmetic ops per bit × 8 bits = 128 operations.

See [FROZEN_6502_NET.md](FROZEN_6502_NET.md) for complete ONNX documentation.

---

## Design Decisions

### Why FP4 Atoms?

The compiler's FP4 atoms provide:
- Verified threshold circuits
- Hardware-mappable representations
- Existing composition primitives

Building on them avoids reinvention.

### Why Truth Table Signatures?

Options considered:
1. Random fixed - arbitrary, no semantics
2. Learned - defeats "frozen" property
3. **Truth table projection** - deterministic, semantic

Truth tables uniquely identify function behavior, making them ideal for content-addressable routing.

### Why Pure Math Polynomials?

For 6502 shapes, we use PureMath instead of FP4 circuits because:
- 8-bit operations compose better with polynomial forms
- Gradients flow through for training the meaning layer
- Simpler implementation for common patterns

The two approaches (FP4 and PureMath) are isomorphic for bit inputs.

---

## Future Work

1. **Continuous domain shapes** - Extend beyond bit operations
2. **Shape discovery** - Learn new shapes from data
3. **Progressive freezing** - Auto-freeze converged tiles
4. **Guardian integration** - Monitor shape activation patterns
5. **Compiler unification** - Single representation for all shapes

---

## References

- `docs/FROZEN_6502.md` - 6502 architecture details
- `docs/FROZEN_6502_NET.md` - Neural network API and ONNX export
- `docs/OPCODE_MAP.md` - Complete opcode reference
- `docs/ATOMIC_FUNCTIONS.md` - Mathematical foundations
- `docs/LINCOLN_MANIFOLD_METHOD.md` - Design methodology
- `src/trix/nn/frozen_6502_net.py` - Neural network implementation
- `experiments/frozen_emulator/` - Python emulator and ONNX export

---

*"The wood cuts itself when you understand the grain."*
