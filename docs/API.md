# TriX API Reference

Complete API documentation for the TriX library.

## Table of Contents

1. [Core Modules](#core-modules)
2. [Neural Network Layers](#neural-network-layers)
3. [XOR Superposition](#xor-superposition)
4. [Providence: The Unified Architecture](#providence-the-unified-architecture)
5. [XOR Routing](#xor-routing)
6. [Frozen Shapes Library](#frozen-shapes-library)
7. [Hierarchical Temporal](#hierarchical-temporal)
8. [Kernel Operations](#kernel-operations)
9. [Quantization-Aware Training](#quantization-aware-training)
10. [Compiler](#compiler)

---

## Core Modules

### Top-Level Imports

```python
from trix import (
    # Recommended - Hierarchical Architecture
    HierarchicalTriXFFN,
    HierarchicalTriXBlock,
    TriXTile,
    
    # Sparse Lookup Architecture
    SparseLookupFFN,
    SparseLookupBlock,
    TernarySpline2D,
    
    # Simple Sparse Architecture
    SparseTriXFFN,
    SparseTriXBlock,
    
    # Classic Emergent Routing
    TriXFFN,
    TriXBlock,
    TriXStack,
    
    # Low-Level Kernel
    TriXLinear,
    STESign,
    pack_weights,
    unpack_weights,
    
    # QAT Utilities
    TernaryQuantizer,
    QATTrainer,
)
```

---

## Neural Network Layers

### HierarchicalTriXFFN

Two-level hierarchical routing FFN for large-scale deployments.

```python
class HierarchicalTriXFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        d_hidden: int = None,  # Default: 4 * d_model // num_tiles
        dropout: float = 0.1,
        aux_weight: float = 0.01,
    ):
        """
        Args:
            d_model: Input/output dimension
            num_tiles: Total number of specialist tiles
            tiles_per_cluster: Tiles per routing cluster
            d_hidden: Hidden dimension per tile (auto-computed if None)
            dropout: Dropout probability
            aux_weight: Weight for auxiliary losses
        """
```

**Forward Pass:**

```python
def forward(
    self,
    x: Tensor,                    # (batch, seq, d_model)
    labels: Optional[Tensor] = None,  # For claim tracking
) -> Tuple[Tensor, Dict, Dict]:
    """
    Returns:
        output: (batch, seq, d_model)
        routing_info: {
            'cluster_indices': (batch, seq),
            'tile_indices': (batch, seq),
            'global_indices': (batch, seq),
            'cluster_scores': (batch, seq, num_clusters),
            'tile_scores': (batch, seq, tiles_per_cluster),
        }
        aux_losses: {
            'load_balance': Tensor,
            'entropy': Tensor,
            'total_aux': Tensor,
        }
    """
```

**Methods:**

```python
def get_signatures(self) -> Tensor:
    """Return all tile signatures. Shape: (num_tiles, d_model)"""

def get_routing_stats(self) -> Dict[str, Tensor]:
    """Return routing statistics from last forward pass."""

def pack_weights(self) -> Dict[str, Tensor]:
    """Pack all weights to 2-bit representation."""

def unpack_weights(self, packed: Dict[str, Tensor]):
    """Restore weights from packed representation."""
```

---

### HierarchicalTriXBlock

Full transformer block with hierarchical FFN.

```python
class HierarchicalTriXBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            num_tiles: Number of FFN tiles
            tiles_per_cluster: Tiles per cluster
            dropout: Dropout probability
            causal: Use causal attention mask
        """
```

---

### SparseLookupFFN

"Routing IS the computation" architecture.

```python
class SparseLookupFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_tiles: int = 64,
        spline_knots: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Input/output dimension
            num_tiles: Number of direction tiles
            spline_knots: Knot points for magnitude spline
            dropout: Dropout probability
        """
```

**Concept:**
- Routing selects a **direction** (tile signature)
- Spline modulates **magnitude** based on input norm
- No hidden layer computation in the hot path

---

### SparseLookupFFNv2

Enhanced with surgery API and regularization.

```python
class SparseLookupFFNv2(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_tiles: int = 64,
        spline_knots: int = 8,
        dropout: float = 0.1,
        island_weight: float = 0.01,  # Signature diversity loss
        ternary_weight: float = 0.01, # Ternarization loss
    ):
        """
        Additional Args:
            island_weight: Weight for island regularization
            ternary_weight: Weight for ternary convergence
        """
```

**Surgery API:**

```python
def surgery_replace_tile(self, tile_idx: int, new_signature: Tensor):
    """Replace a tile's signature."""

def surgery_merge_tiles(self, tile_a: int, tile_b: int):
    """Merge two tiles into one."""

def surgery_split_tile(self, tile_idx: int):
    """Split a tile into two."""
```

---

### TemporalTileLayer

State-aware routing for sequential tasks.

```python
class TemporalTileLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        num_tiles: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Input/output dimension
            d_state: State vector dimension
            num_tiles: Number of tiles
            dropout: Dropout probability
        """
```

**Forward Pass:**

```python
def forward(
    self,
    x: Tensor,                      # (batch, d_model)
    state: Optional[Tensor] = None, # (batch, d_state)
) -> Tuple[Tensor, Tensor, Dict]:
    """
    Single-step forward.
    
    Returns:
        output: (batch, d_model)
        new_state: (batch, d_state)
        routing_info: {...}
    """

def forward_sequence(
    self,
    x: Tensor,  # (batch, seq, d_model)
) -> Tuple[Tensor, Tensor, List[Dict]]:
    """
    Process full sequence.
    
    Returns:
        output: (batch, seq, d_model)
        final_state: (batch, d_state)
        routing_infos: List of per-step routing info
    """

def init_state(self, batch_size: int) -> Tensor:
    """Initialize state for new sequence."""
```

---

### CompiledDispatch

O(1) inference via precomputed routes.

```python
class CompiledDispatch(nn.Module):
    def __init__(self, base_ffn: SparseLookupFFNv2):
        """
        Args:
            base_ffn: The FFN to compile
        """
```

**Methods:**

```python
def profile_class(
    self,
    class_id: int,
    samples: Tensor,
) -> ProfileStats:
    """Profile routing for a class."""

def compile_stable(
    self,
    threshold: float = 0.9,
) -> int:
    """Compile all classes with stability > threshold."""

def forward(
    self,
    x: Tensor,
    class_hint: Optional[int] = None,
    confidence: float = 0.9,
) -> Tuple[Tensor, Dict, Dict]:
    """
    Forward with optional compiled dispatch.
    
    Args:
        x: Input tensor
        class_hint: Known class ID (enables O(1) dispatch)
        confidence: Required confidence for compiled path
    """
```

---

## XOR Superposition

Signature compression and deterministic routing via XOR operations.

### CompressedSignatures

XOR superposition storage for ternary signatures.

```python
from trix.nn import CompressedSignatures

class CompressedSignatures:
    def compress(self, signatures: Tensor) -> 'CompressedSignatures':
        """
        Compress signatures via XOR superposition.

        Args:
            signatures: (num_sigs, d_model) ternary tensor

        Returns:
            self (for chaining)
        """

    def decompress(self, index: int) -> Tensor:
        """Decompress single signature by index."""

    def decompress_all(self) -> Tensor:
        """Decompress all signatures. Returns (num_sigs, d_model)."""

    def get_compression_stats(self) -> CompressionStats:
        """Get compression statistics."""
```

**CompressionStats:**

```python
class CompressionStats(NamedTuple):
    original_bytes: int
    compressed_bytes: int
    compression_ratio: float
    mean_delta_sparsity: float
    max_delta_sparsity: float
    num_signatures: int
```

---

### SuperpositionRouter

Hamming-distance router with compression support.

```python
from trix.nn import SuperpositionRouter

class SuperpositionRouter(nn.Module):
    def __init__(
        self,
        num_tiles: int,
        d_model: int,
    ):
        """
        Args:
            num_tiles: Number of routing tiles
            d_model: Model dimension
        """

    def route(
        self,
        x: Tensor,
        return_scores: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Route input to best-matching tile.

        Returns:
            tile_idx: (...) winning tile indices
            scores_or_distances: (..., num_tiles)
        """

    def compress(self):
        """Compress signatures for inference."""

    def decompress(self):
        """Decompress for training."""

    def verify_routing_equivalence(
        self,
        x: Tensor,
        tolerance: float = 0.0,
    ) -> bool:
        """Verify compressed routing matches uncompressed."""

    def get_compression_stats(self) -> Optional[CompressionStats]:
        """Get compression stats if compressed."""
```

---

### XORSuperpositionFFN

Drop-in FFN replacement with compression lifecycle.

```python
from trix.nn import XORSuperpositionFFN

class XORSuperpositionFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_tiles: int = 16,
        d_hidden: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension
            num_tiles: Number of routing tiles
            d_hidden: Hidden dimension (default: 4 * d_model)
            dropout: Dropout probability
        """

    def forward(
        self,
        x: Tensor,
        return_routing_info: bool = False,
    ) -> Tuple[Tensor, Optional[Dict]]:
        """
        Forward pass with routing.

        Args:
            x: (batch, seq, d_model) or (batch, d_model)
            return_routing_info: Include routing details

        Returns:
            output: Same shape as input
            routing_info: Optional dict with tile_idx, weights, entropy
        """

    def compress(self):
        """Compress router for inference."""

    def decompress(self):
        """Decompress for training."""

    def get_compression_stats(self) -> Optional[CompressionStats]:
        """Get router compression stats."""
```

---

### HierarchicalTriXFFN Compression Methods

Extended methods on `HierarchicalTriXFFN` for signature compression.

```python
# After training, compress for inference
ffn.compress_signatures()

# Get compression statistics
stats = ffn.get_compression_stats()
# {'tile': CompressionStats(...), 'cluster': CompressionStats(...)}

# Decompress for fine-tuning
ffn.decompress_signatures()
```

---

### Utility Functions

```python
from trix.nn import (
    pack_ternary_to_uint8,
    unpack_uint8_to_ternary,
    hamming_distance_packed,
    hamming_distance_batch,
)

def pack_ternary_to_uint8(ternary: Tensor) -> Tensor:
    """
    Pack ternary {-1, 0, +1} to 2-bit uint8.

    Encoding: +1 → 01, -1 → 10, 0 → 00
    4 values per byte.

    Args:
        ternary: (..., dim) tensor

    Returns:
        (..., (dim+3)//4) uint8 tensor
    """

def unpack_uint8_to_ternary(packed: Tensor, dim: int) -> Tensor:
    """Unpack uint8 back to ternary. Inverse of pack_ternary_to_uint8."""

def hamming_distance_batch(
    query: Tensor,       # (batch, packed_dim) uint8
    signatures: Tensor,  # (num_sigs, packed_dim) uint8
) -> Tensor:
    """
    Compute Hamming distances from query to all signatures.

    Returns:
        (batch, num_sigs) distances
    """
```

---

## Providence: The Unified Architecture

The culmination of all FFN variants into a single unified architecture.

### ProvidenceFFN

```python
from trix.nn import ProvidenceFFN, create_providence_ffn

class ProvidenceFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int = None,
        d_state: int = 16,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        d_global_state: int = 32,
        dropout: float = 0.1,
        use_state_routing: bool = True,
        use_frozen_shapes: bool = False,
        frozen_shape_names: Optional[List[str]] = None,
        use_soft_routing: bool = True,
        temperature: float = 1.0,
        mode: str = 'hierarchical',  # or 'flat'
    ):
        """
        The unified architecture: FFN as content-addressable memory.

        Args:
            d_model: Model dimension
            d_hidden: Hidden dimension per tile
            d_state: State dimension per tile (temporal memory)
            num_tiles: Total number of tiles
            tiles_per_cluster: Tiles per cluster (for hierarchical mode)
            d_global_state: Global state dimension
            dropout: Dropout probability
            use_state_routing: Include state in routing decision
            use_frozen_shapes: Use frozen shapes (0 compute params)
            frozen_shape_names: Specific shapes to use
            use_soft_routing: Soft routing during training
            temperature: Softmax temperature
            mode: 'hierarchical' (O(√n)) or 'flat' (O(n))
        """
```

**Forward Pass:**

```python
def forward(
    self,
    x: Tensor,                              # (batch, d_model) or (batch, seq, d_model)
    state: Optional[Dict[str, Tensor]] = None,
    track_transitions: bool = True,
) -> Tuple[Tensor, Dict, Dict, Dict]:
    """
    Returns:
        output: Same shape as input
        new_state: {
            'global_state': (batch, d_global_state),
            'tile_states': (batch, num_tiles, d_state),
            'prev_tile': (batch,),
        }
        routing_info: {
            'tile_idx': (batch,) or (batch, seq),
            'cluster_idx': (batch,) or (batch, seq),
            'gate': (batch, num_tiles) or (batch, seq, num_tiles),
        }
        aux_losses: {
            'tile_balance': Tensor,
            'cluster_balance': Tensor,
            'total_aux': Tensor,
        }
    """

def init_state(self, batch_size: int) -> Dict[str, Tensor]:
    """Initialize state for new sequence."""

def get_routing_stats(self) -> Dict:
    """Get routing statistics."""

def get_transition_matrix(self, normalize: bool = True) -> Tensor:
    """Get tile transition matrix."""

def get_regime_analysis(self) -> Dict:
    """Analyze learned regime structure."""

def count_parameters(self) -> Dict[str, int]:
    """Count parameters by category (routing, compute, state)."""
```

**Convenience Functions:**

```python
# Standard Providence FFN
ffn = create_providence_ffn(d_model=128, num_tiles=64, d_state=16)

# Frozen shapes only (0 compute params)
from trix.nn import create_frozen_providence_ffn
ffn = create_frozen_providence_ffn(d_model=128, num_tiles=16)
```

---

### ProvidenceBlock

Transformer block with Providence FFN.

```python
from trix.nn import ProvidenceBlock

class ProvidenceBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        num_tiles: int = 64,
        d_state: int = 16,
        dropout: float = 0.1,
        use_frozen_shapes: bool = False,
    ):
        """
        Standard transformer block: LayerNorm → Attention → LayerNorm → ProvidenceFFN
        """

def forward(
    self,
    x: Tensor,                              # (batch, seq, d_model)
    state: Optional[Dict] = None,
    attn_mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Dict, Dict, Dict]:
    """Returns output, state, routing_info, aux_losses."""
```

---

### ProvidenceTile

Individual tile with signature + shape + state.

```python
from trix.nn import ProvidenceTile

class ProvidenceTile(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        d_state: int,
        tile_id: int = 0,
        frozen_shape: Optional[Callable] = None,
        shape_name: str = None,
    ):
        """
        The atom of Providence.

        Args:
            d_model: Model dimension
            d_hidden: Hidden dimension
            d_state: State dimension
            tile_id: Unique identifier
            frozen_shape: Frozen shape function (0 compute params)
            shape_name: Name for registry
        """

def get_signature(self) -> Tensor:
    """Get ternary routing signature."""

def forward(self, x: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
    """(input, state) → (output, new_state)"""
```

---

## XOR Routing

Hamming distance routing primitives.

### XORRoutingFFN

```python
from trix.nn import XORRoutingFFN

class XORRoutingFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int = None,
        num_tiles: int = 16,
        dropout: float = 0.1,
        use_soft_training: bool = True,
        temperature: float = 1.0,
    ):
        """
        FFN with XOR-based (Hamming distance) routing.

        Instead of: winner = argmax(x @ signatures.T)
        We use:     winner = argmin(hamming(binarize(x), signatures))
        """
```

### HierarchicalXORRoutingFFN

```python
from trix.nn import HierarchicalXORRoutingFFN

class HierarchicalXORRoutingFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int = None,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        dropout: float = 0.1,
        use_soft_training: bool = True,
        temperature: float = 1.0,
    ):
        """
        Hierarchical XOR routing for O(√n) scaling.
        """
```

### Core Functions

```python
from trix.nn import (
    binarize_ste,
    ternarize_ste,
    xor_hamming_distance,
    soft_hamming_distance,
)

def binarize_ste(x: Tensor) -> Tensor:
    """Binarize to {-1, +1} with straight-through estimator."""

def ternarize_ste(x: Tensor, threshold: float = 0.3) -> Tensor:
    """Ternarize to {-1, 0, +1} with straight-through estimator."""

def xor_hamming_distance(a: Tensor, b: Tensor) -> Tensor:
    """Compute Hamming distance between ternary vectors."""

def soft_hamming_distance(a: Tensor, b: Tensor) -> Tensor:
    """Differentiable soft Hamming distance for training."""
```

---

## Frozen Shapes Library

Mathematical shapes with 0 learnable parameters.

### FrozenShapeLibrary

```python
from trix.nn import FrozenShapeLibrary, get_frozen_shape_library

library = get_frozen_shape_library()

# List available shapes
all_shapes = library.list_shapes()
logic_shapes = library.list_shapes('logic')
activation_shapes = library.list_shapes('activation')

# Get a shape function
xor_fn = library.get('xor')
result = xor_fn(a, b)  # XOR: a + b - 2ab

# Get shape info
info = library.info('xor')
# ShapeInfo(name='xor', category='primitive', n_inputs=2, ...)
```

### Shape Categories

```python
from trix.nn import (
    Primitives,      # xor, and, or, not
    LogicShapes,     # nand, nor, xnor, implies, full_adder, majority
    ComparisonShapes,# eq, lt, gt, max, min, clamp
    ActivationShapes,# relu, gelu, swish, sigmoid, tanh, softplus
    ArithmeticShapes,# add_8bit, sub_8bit
    CompositionShapes,# identity, swap
)

# Direct usage
result = Primitives.xor(a, b)      # a + b - 2ab
result = Primitives.and_op(a, b)   # ab
result = ActivationShapes.relu(x)  # max(0, x) polynomial approx
```

### GeneralFrozenTile

```python
from trix.nn import GeneralFrozenTile

tile = GeneralFrozenTile(shape_name='xor', d_model=64)
signature = tile.get_signature()  # Derived from shape behavior
output = tile(x)
```

---

## Hierarchical Temporal

State-aware hierarchical routing.

### HierarchicalTemporalFFN

```python
from trix.nn import HierarchicalTemporalFFN, create_hierarchical_temporal_ffn

class HierarchicalTemporalFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int = None,
        d_state: int = 16,
        num_tiles: int = 64,
        tiles_per_cluster: int = 8,
        d_global_state: int = 32,
        dropout: float = 0.1,
        use_state_routing: bool = True,
        temperature: float = 1.0,
    ):
        """
        Hierarchical O(√n) routing with temporal state persistence.
        """

def init_state(self, batch_size: int) -> Dict[str, Tensor]:
    """Initialize states."""

def forward(
    self,
    x: Tensor,
    state: Optional[Dict] = None,
    track_transitions: bool = True,
) -> Tuple[Tensor, Dict, Dict, Dict]:
    """Returns output, new_state, routing_info, aux_losses."""

def get_transition_matrix(self, normalize: bool = True) -> Tensor:
    """Get tile transition probabilities."""

def get_regime_analysis(self) -> Dict:
    """Analyze regime structure."""
```

### TemporalTile

```python
from trix.nn import TemporalTile

class TemporalTile(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        d_state: int,
        tile_id: int = 0,
    ):
        """Tile with persistent state."""

def get_signature(self) -> Tensor:
    """Get routing signature."""

def forward(self, x: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
    """(input, state) → (output, new_state)"""
```

---

## Kernel Operations

### TriXLinear

Base ternary linear layer.

```python
class TriXLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Include bias term (default: False)
        """
```

**Methods:**

```python
def get_ternary_weights(self) -> Tensor:
    """Return quantized ternary weights."""

def get_signature(self) -> Tensor:
    """Return weight signature."""
```

---

### Weight Packing

```python
def pack_weights(weights: Tensor) -> Tensor:
    """
    Pack ternary weights to 2-bit representation.
    
    Args:
        weights: Ternary tensor with values in {-1, 0, +1}
        
    Returns:
        Packed tensor (4x smaller)
    """

def unpack_weights(packed: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """
    Unpack 2-bit weights to ternary.
    
    Args:
        packed: Packed weight tensor
        shape: Original weight shape
        
    Returns:
        Ternary tensor with values in {-1, 0, +1}
    """
```

---

### NEON Acceleration

```python
def trix_forward(
    input: Tensor,
    packed_weights: Tensor,
    output_features: int,
) -> Tensor:
    """
    NEON-accelerated ternary matrix multiply.
    
    Only available on ARM platforms with NEON support.
    Falls back to PyTorch on other platforms.
    """

def is_neon_available() -> bool:
    """Check if NEON acceleration is available."""
```

---

## Quantization-Aware Training

### TernaryQuantizer

```python
class TernaryQuantizer(nn.Module):
    def __init__(
        self,
        threshold: float = 0.5,
    ):
        """
        Args:
            threshold: Values with |w| < threshold become 0
        """
    
    def forward(self, weights: Tensor) -> Tensor:
        """Quantize weights to ternary."""
```

### QATTrainer

```python
class QATTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        quantizer: TernaryQuantizer,
    ):
        """
        Args:
            model: Model to train
            optimizer: Optimizer
            quantizer: Ternary quantizer
        """
    
    def train_step(
        self,
        batch: Tensor,
        labels: Tensor,
    ) -> Dict[str, float]:
        """Perform one training step with QAT."""
```

### Progressive Schedule

```python
def progressive_quantization_schedule(
    epoch: int,
    total_epochs: int,
    start_threshold: float = 0.1,
    end_threshold: float = 0.5,
) -> float:
    """
    Compute quantization threshold for progressive training.
    
    Starts with soft quantization, progressively hardens.
    """
```

---

## Compiler

### TriXCompiler

```python
from trix.compiler import TriXCompiler

compiler = TriXCompiler(
    use_fp4: bool = False,  # Use FP4 atoms (exact by construction)
    cache_dir: str = None,  # Cache for trained atoms
    verbose: bool = True,   # Print progress
)
```

**Methods:**

```python
def compile(
    self,
    spec_or_name: Union[CircuitSpec, str],
    output_dir: Optional[str] = None,
) -> CompilationResult:
    """
    Compile a circuit specification.
    
    Args:
        spec_or_name: CircuitSpec or template name 
                     ('full_adder', 'adder_8bit', etc.)
        output_dir: Directory for emitted files
        
    Returns:
        CompilationResult with topology, verification, and executor
    """
```

### CircuitSpec

```python
from trix.compiler import CircuitSpec

spec = CircuitSpec(name="my_circuit", description="...")

spec.add_input("A", width=8)
spec.add_output("Y", width=8)
spec.add_atom("gate1", "AND", inputs=["A[0]", "A[1]"], outputs=["Y[0]"])
```

---

## Type Definitions

```python
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor

RoutingInfo = Dict[str, Tensor]
AuxLosses = Dict[str, Tensor]
ForwardOutput = Tuple[Tensor, RoutingInfo, AuxLosses]
```

---

## Error Handling

All TriX components raise standard PyTorch exceptions:

- `ValueError`: Invalid arguments
- `RuntimeError`: Computation errors
- `TypeError`: Type mismatches

```python
# Example: Invalid tile count
try:
    ffn = HierarchicalTriXFFN(d_model=512, num_tiles=7)  # Not divisible
except ValueError as e:
    print(f"Configuration error: {e}")
```
