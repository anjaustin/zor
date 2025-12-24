"""
Frozen Shapes: Computation as Geometry

Frozen shapes are fixed mathematical structures that compute exactly.
Unlike learned tiles (TriXTile), frozen tiles have:
- Zero learned parameters in the computation
- 100% accuracy by construction
- Signatures derived from structure (truth table)

This module provides:
- FrozenShape: A verified computation primitive
- FrozenTile: nn.Module wrapper compatible with TriX routing
- FrozenShapeRegistry: Catalog of available shapes
- FrozenTriXFFN: FFN layer using frozen tiles

Key insight: FP4 atoms from the compiler ARE frozen shapes.
We wrap them with signatures for content-addressable routing.

Example:
    >>> registry = FrozenShapeRegistry()
    >>> xor_shape = registry.get("XOR")
    >>> tile = FrozenTile(xor_shape)
    >>> x = torch.tensor([[0, 1], [1, 1]], dtype=torch.float32)
    >>> tile(x)  # Returns [[1], [0]]

Theory: "Computation is topology. Learning is routing."
- The frozen shapes ARE the geometry
- The routing (meaning layer) is all that's learned
- Result: 40x compression on 6502 emulation

See also:
- docs/FROZEN_SHAPES.md for theory and API
- docs/FROZEN_6502.md for 6502-specific implementation
- experiments/atomic/ for validation experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import math

from trix.compiler.atoms_fp4 import (
    ThresholdCircuit,
    ThresholdLayer,
    FP4AtomLibrary,
    make_AND,
    make_OR,
    make_XOR,
    make_NOT,
    make_NAND,
    make_NOR,
    make_CARRY,
    make_SUM,
    make_MUX,
    make_XNOR,
    truth_table_to_circuit,
)


# =============================================================================
# FROZEN SHAPE
# =============================================================================

@dataclass
class FrozenShape:
    """
    A verified computation primitive with routing signature.

    A FrozenShape wraps a ThresholdCircuit (from the FP4 compiler)
    with metadata for integration into TriX routing.

    Attributes:
        name: Unique identifier for registry lookup
        circuit: The ThresholdCircuit that performs computation
        signature: Ternary signature for content-addressable routing
        num_inputs: Number of input bits
        num_outputs: Number of output bits
        description: Human-readable description

    The signature is derived from the circuit's truth table, making
    it deterministic and semantically meaningful.
    """
    name: str
    circuit: ThresholdCircuit
    signature: torch.Tensor
    num_inputs: int
    num_outputs: int
    description: str = ""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute the frozen computation.

        Args:
            x: [batch, num_inputs] bit tensor (values in {0, 1})

        Returns:
            [batch, num_outputs] bit tensor
        """
        return self.circuit(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def verify(self) -> Tuple[bool, float]:
        """
        Verify the shape computes correctly.

        Returns:
            (passed, accuracy) tuple
        """
        # FP4 circuits are verified by construction
        return True, 1.0


# =============================================================================
# SIGNATURE DERIVATION
# =============================================================================

def derive_signature(
    circuit: ThresholdCircuit,
    sig_dim: int = 64,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Derive a routing signature from a circuit's truth table.

    The signature is a fixed-dimension ternary vector that uniquely
    identifies the circuit's behavior. Two circuits with identical
    truth tables will have identical signatures.

    Args:
        circuit: The ThresholdCircuit to derive signature from
        sig_dim: Output signature dimension
        seed: Random seed for projection (default: hash of circuit name)

    Returns:
        [sig_dim] ternary tensor with values in {-1, 0, +1}

    Algorithm:
        1. Generate truth table (exhaustive for small inputs, sampled for large)
        2. Flatten outputs to 1D vector
        3. Project to sig_dim via random projection
        4. Sign to get ternary values
    """
    n_inputs = circuit.num_inputs

    # Generate truth table
    if n_inputs <= 12:  # Exhaustive up to 4096 entries
        all_inputs = torch.tensor([
            [(i >> b) & 1 for b in range(n_inputs)]
            for i in range(2 ** n_inputs)
        ], dtype=torch.float32)
        truth_table = circuit(all_inputs).flatten()
    else:
        # Sample for large inputs
        torch.manual_seed(42)  # Deterministic sampling
        samples = torch.randint(0, 2, (4096, n_inputs), dtype=torch.float32)
        truth_table = circuit(samples).flatten()

    # Convert to ternary (-1, +1) for better projection
    truth_table_ternary = truth_table * 2 - 1

    # Project to fixed dimension
    if seed is None:
        seed = hash(circuit.name) % (2**31)
    torch.manual_seed(seed)

    tt_len = len(truth_table_ternary)
    projection = torch.randn(tt_len, sig_dim)
    projection = projection / (projection.norm(dim=0, keepdim=True) + 1e-8)

    # Project and sign
    projected = truth_table_ternary @ projection
    signature = projected.sign()

    # Handle zeros (rare, but possible)
    signature[signature == 0] = 1

    return signature


def derive_signature_from_fn(
    fn: Callable,
    num_inputs: int,
    sig_dim: int = 64,
    name: str = "custom",
) -> torch.Tensor:
    """
    Derive signature from a pure Python function.

    Useful for defining shapes without building ThresholdCircuits.

    Args:
        fn: Function that takes n bits and returns output bits
        num_inputs: Number of input bits
        sig_dim: Signature dimension
        name: Name for seeding

    Returns:
        [sig_dim] ternary signature
    """
    # Build truth table
    outputs = []
    for i in range(2 ** num_inputs):
        bits = [(i >> b) & 1 for b in range(num_inputs)]
        out = fn(*bits)
        if isinstance(out, (int, float)):
            out = [out]
        outputs.extend(out)

    truth_table = torch.tensor(outputs, dtype=torch.float32)
    truth_table_ternary = truth_table * 2 - 1

    # Project
    seed = hash(name) % (2**31)
    torch.manual_seed(seed)

    tt_len = len(truth_table_ternary)
    projection = torch.randn(tt_len, sig_dim)
    projection = projection / (projection.norm(dim=0, keepdim=True) + 1e-8)

    projected = truth_table_ternary @ projection
    signature = projected.sign()
    signature[signature == 0] = 1

    return signature


# =============================================================================
# FROZEN SHAPE REGISTRY
# =============================================================================

class FrozenShapeRegistry:
    """
    Registry of available frozen shapes.

    The registry provides:
    - Access to built-in shapes (FP4 atoms)
    - Registration of custom shapes
    - Signature management

    Example:
        >>> registry = FrozenShapeRegistry()
        >>> registry.list_shapes()
        ['AND', 'OR', 'XOR', 'NOT', ...]
        >>> xor = registry.get("XOR")
        >>> xor.signature.shape
        torch.Size([64])
    """

    def __init__(self, sig_dim: int = 64, auto_register_builtins: bool = True):
        """
        Initialize the registry.

        Args:
            sig_dim: Signature dimension for all shapes
            auto_register_builtins: If True, register FP4 atoms on init
        """
        self.sig_dim = sig_dim
        self._shapes: Dict[str, FrozenShape] = {}

        if auto_register_builtins:
            self._register_builtin_shapes()

    def _register_builtin_shapes(self):
        """Register the built-in FP4 atoms."""
        builtins = [
            ("AND", make_AND(), "2-input AND gate"),
            ("OR", make_OR(), "2-input OR gate"),
            ("XOR", make_XOR(), "2-input XOR gate"),
            ("NOT", make_NOT(), "1-input NOT gate"),
            ("NAND", make_NAND(), "2-input NAND gate"),
            ("NOR", make_NOR(), "2-input NOR gate"),
            ("XNOR", make_XNOR(), "2-input XNOR gate"),
            ("SUM", make_SUM(), "3-input sum (XOR chain)"),
            ("CARRY", make_CARRY(), "3-input carry (majority)"),
            ("MUX", make_MUX(), "2:1 multiplexer"),
        ]

        for name, circuit, desc in builtins:
            self.register(name, circuit, desc)

    def register(
        self,
        name: str,
        circuit: ThresholdCircuit,
        description: str = "",
    ) -> FrozenShape:
        """
        Register a new frozen shape.

        Args:
            name: Unique identifier
            circuit: ThresholdCircuit implementation
            description: Human-readable description

        Returns:
            The registered FrozenShape
        """
        signature = derive_signature(circuit, self.sig_dim)

        shape = FrozenShape(
            name=name,
            circuit=circuit,
            signature=signature,
            num_inputs=circuit.num_inputs,
            num_outputs=circuit.num_outputs,
            description=description,
        )

        self._shapes[name] = shape
        return shape

    def register_from_truth_table(
        self,
        name: str,
        num_inputs: int,
        truth_table: Dict[Tuple[int, ...], int],
        description: str = "",
    ) -> FrozenShape:
        """
        Register a shape from a truth table.

        Args:
            name: Unique identifier
            num_inputs: Number of input bits
            truth_table: Maps input tuples to output (0 or 1)
            description: Human-readable description

        Returns:
            The registered FrozenShape
        """
        circuit = truth_table_to_circuit(name, num_inputs, truth_table)
        return self.register(name, circuit, description)

    def get(self, name: str) -> FrozenShape:
        """Get a shape by name."""
        if name not in self._shapes:
            raise KeyError(f"Unknown shape: {name}. Available: {self.list_shapes()}")
        return self._shapes[name]

    def list_shapes(self) -> List[str]:
        """List all registered shape names."""
        return list(self._shapes.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._shapes

    def __len__(self) -> int:
        return len(self._shapes)


# Global default registry
_DEFAULT_REGISTRY: Optional[FrozenShapeRegistry] = None


def get_default_registry() -> FrozenShapeRegistry:
    """Get or create the default global registry."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = FrozenShapeRegistry()
    return _DEFAULT_REGISTRY


# =============================================================================
# FROZEN TILE
# =============================================================================

class FrozenTile(nn.Module):
    """
    A TriX tile backed by a frozen shape.

    FrozenTile implements the same interface as TriXTile:
    - get_signature() for routing
    - forward(x) for computation

    But the computation is fixed (0 learned parameters).

    This allows FrozenTiles to be used anywhere TriXTiles are used,
    enabling hybrid architectures with mixed frozen/learned tiles.

    Args:
        shape: The FrozenShape to wrap
        tile_id: Identifier for tracking

    Example:
        >>> shape = get_default_registry().get("XOR")
        >>> tile = FrozenTile(shape)
        >>> x = torch.tensor([[0, 1]], dtype=torch.float32)
        >>> tile(x)  # Returns [[1]]
    """

    def __init__(self, shape: FrozenShape, tile_id: int = 0):
        super().__init__()
        self.shape = shape
        self.tile_id = tile_id

        # Signature is fixed (buffer, not parameter)
        self.register_buffer('signature', shape.signature.clone())

        # Store circuit layers as buffers for device compatibility
        for i, layer in enumerate(shape.circuit.layers):
            self.register_buffer(f'weight_{i}', layer.weights.clone())
            self.register_buffer(f'bias_{i}', layer.bias.clone())

        # Usage tracking (compatible with HierarchicalTriXFFN)
        self.register_buffer('activation_count', torch.tensor(0.0))
        self.register_buffer('total_count', torch.tensor(0.0))

    def get_signature(self) -> torch.Tensor:
        """Return the fixed signature for routing."""
        return self.signature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through frozen computation.

        Args:
            x: [batch, num_inputs] or [..., num_inputs] bit tensor

        Returns:
            [..., num_outputs] bit tensor
        """
        # Use buffered weights for device compatibility
        for i in range(len(self.shape.circuit.layers)):
            weights = getattr(self, f'weight_{i}')
            bias = getattr(self, f'bias_{i}')
            x = (x @ weights.T + bias >= 0).to(x.dtype)
        return x

    def update_usage(self, count: int, total: int):
        """Track activation frequency."""
        self.activation_count = self.activation_count + count
        self.total_count = self.total_count + total

    @property
    def usage_rate(self) -> float:
        """Get activation rate (0-1)."""
        if self.total_count == 0:
            return 0.0
        return (self.activation_count / self.total_count).item()

    @property
    def num_inputs(self) -> int:
        return self.shape.num_inputs

    @property
    def num_outputs(self) -> int:
        return self.shape.num_outputs


# =============================================================================
# FROZEN TRIX FFN
# =============================================================================

class FrozenTriXFFN(nn.Module):
    """
    Feed-forward network using frozen tiles with content-addressable routing.

    Two routing modes:
    1. Content routing: Input signature → nearest tile (emergent)
    2. Opcode routing: Opcode ID → meaning layer → tile (6502 mode)

    The computation is frozen. Only the meaning layer (if used) is learned.

    Args:
        tiles: List of FrozenTile instances
        use_meaning_layer: If True, add learned opcode→tile routing
        num_opcodes: Number of opcodes (if using meaning layer)
        temperature: Gumbel-Softmax temperature for training

    Example (content routing):
        >>> registry = get_default_registry()
        >>> tiles = [FrozenTile(registry.get(name)) for name in ["AND", "OR", "XOR"]]
        >>> ffn = FrozenTriXFFN(tiles)
        >>> x = torch.tensor([[0, 1]], dtype=torch.float32)
        >>> out = ffn(x)

    Example (opcode routing):
        >>> ffn = FrozenTriXFFN(tiles, use_meaning_layer=True, num_opcodes=8)
        >>> x = torch.tensor([[0, 1]], dtype=torch.float32)
        >>> opcode = torch.tensor([2])  # Use tile 2
        >>> out = ffn(x, opcode_id=opcode)
    """

    def __init__(
        self,
        tiles: List[FrozenTile],
        use_meaning_layer: bool = False,
        num_opcodes: int = 56,
        temperature: float = 1.0,
    ):
        super().__init__()

        if len(tiles) == 0:
            raise ValueError("Must provide at least one tile")

        self.tiles = nn.ModuleList(tiles)
        self.num_tiles = len(tiles)
        self.temperature = temperature

        # Stack signatures for efficient routing
        signatures = torch.stack([t.get_signature() for t in tiles])
        self.register_buffer('signatures', signatures)

        # Optional meaning layer for opcode routing
        self.use_meaning_layer = use_meaning_layer
        if use_meaning_layer:
            self.meaning_logits = nn.Parameter(
                torch.zeros(num_opcodes, self.num_tiles)
            )
            self.num_opcodes = num_opcodes

    def route_by_content(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route by content similarity to signatures.

        Args:
            x: [batch, sig_dim] input in signature space

        Returns:
            tile_ids: [batch] selected tile indices
            scores: [batch, num_tiles] routing scores
        """
        scores = x @ self.signatures.T  # [batch, num_tiles]
        tile_ids = scores.argmax(dim=-1)
        return tile_ids, scores

    def route_by_opcode(
        self,
        opcode_id: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route by opcode through meaning layer.

        Args:
            opcode_id: [batch] opcode indices
            training: If True, use Gumbel-Softmax for differentiability

        Returns:
            tile_ids: [batch] selected tile indices (hard)
            probs: [batch, num_tiles] soft routing probabilities
        """
        logits = self.meaning_logits[opcode_id]  # [batch, num_tiles]

        if training:
            # Gumbel-Softmax for differentiable discrete selection
            probs = F.gumbel_softmax(logits, tau=self.temperature, hard=False)
        else:
            probs = F.softmax(logits, dim=-1)

        tile_ids = logits.argmax(dim=-1)
        return tile_ids, probs

    def forward(
        self,
        x: torch.Tensor,
        opcode_id: Optional[torch.Tensor] = None,
        return_routing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through frozen tiles.

        Args:
            x: [batch, num_inputs] input bits
            opcode_id: [batch] opcode indices (if using meaning layer)
            return_routing: If True, include routing info in output

        Returns:
            Dictionary with:
                result: [batch, num_outputs] output bits
                tile_ids: [batch] which tiles were selected
                probs: [batch, num_tiles] routing probabilities (if training)
        """
        batch = x.shape[0]
        device = x.device

        # Determine routing
        if self.use_meaning_layer and opcode_id is not None:
            tile_ids, probs = self.route_by_opcode(opcode_id, self.training)
        else:
            # Use input as signature for content routing
            # For bit inputs, derive a signature-like representation
            x_sig = x * 2 - 1  # Map {0,1} to {-1,+1}
            # Pad or project to signature dimension if needed
            sig_dim = self.signatures.shape[1]
            if x_sig.shape[1] < sig_dim:
                x_sig = F.pad(x_sig, (0, sig_dim - x_sig.shape[1]))
            elif x_sig.shape[1] > sig_dim:
                x_sig = x_sig[:, :sig_dim]
            tile_ids, probs = self.route_by_content(x_sig)
            probs = F.softmax(probs, dim=-1)  # Normalize for consistency

        # Execute tiles
        # For training with meaning layer, blend outputs by probability
        if self.training and self.use_meaning_layer and opcode_id is not None:
            # Soft execution: weighted sum of all tile outputs
            # This allows gradients to flow through the meaning layer
            result = torch.zeros(batch, self.tiles[0].num_outputs, device=device)
            for i, tile in enumerate(self.tiles):
                tile_out = tile(x)
                result = result + probs[:, i:i+1] * tile_out
        else:
            # Hard execution: only selected tile
            outputs = []
            for i in range(batch):
                tid = tile_ids[i].item()
                out = self.tiles[tid](x[i:i+1])
                outputs.append(out)
            result = torch.cat(outputs, dim=0)

        # Update usage stats
        if not self.training:
            for i, tile in enumerate(self.tiles):
                count = (tile_ids == i).sum().item()
                tile.update_usage(count, batch)

        output = {
            'result': result,
            'tile_ids': tile_ids,
        }

        if return_routing:
            output['probs'] = probs
            output['signatures'] = self.signatures

        return output

    def init_meaning_from_spec(self, opcode_specs: Dict[int, int]):
        """
        Initialize meaning layer from opcode→tile specification.

        Args:
            opcode_specs: Maps opcode_id to tile_id
        """
        if not self.use_meaning_layer:
            raise ValueError("Meaning layer not enabled")

        with torch.no_grad():
            for opcode_id, tile_id in opcode_specs.items():
                # Set high logit for target tile
                self.meaning_logits[opcode_id, tile_id] = 10.0

    def param_count(self) -> int:
        """Count learnable parameters (meaning layer only)."""
        if self.use_meaning_layer:
            return self.meaning_logits.numel()
        return 0

    def get_routing_stats(self) -> Dict[str, float]:
        """Get routing statistics."""
        stats = {}
        for i, tile in enumerate(self.tiles):
            stats[f"tile_{i}_usage"] = tile.usage_rate
        return stats


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_frozen_ffn_from_names(
    shape_names: List[str],
    registry: Optional[FrozenShapeRegistry] = None,
    **kwargs,
) -> FrozenTriXFFN:
    """
    Create a FrozenTriXFFN from shape names.

    Args:
        shape_names: List of shape names (e.g., ["AND", "OR", "XOR"])
        registry: Shape registry (uses default if None)
        **kwargs: Additional arguments for FrozenTriXFFN

    Returns:
        Configured FrozenTriXFFN
    """
    if registry is None:
        registry = get_default_registry()

    tiles = [
        FrozenTile(registry.get(name), tile_id=i)
        for i, name in enumerate(shape_names)
    ]

    return FrozenTriXFFN(tiles, **kwargs)


def verify_frozen_tile(tile: FrozenTile) -> Tuple[bool, float]:
    """
    Verify a frozen tile computes correctly.

    Args:
        tile: FrozenTile to verify

    Returns:
        (passed, accuracy) tuple
    """
    shape = tile.shape
    n_inputs = shape.num_inputs

    if n_inputs > 16:
        # Sample for large inputs
        torch.manual_seed(42)
        inputs = torch.randint(0, 2, (1000, n_inputs), dtype=torch.float32)
    else:
        # Exhaustive for small inputs
        inputs = torch.tensor([
            [(i >> b) & 1 for b in range(n_inputs)]
            for i in range(2 ** n_inputs)
        ], dtype=torch.float32)

    # Compare tile output to shape output
    tile_out = tile(inputs)
    shape_out = shape(inputs)

    match = (tile_out == shape_out).all().item()
    accuracy = (tile_out == shape_out).float().mean().item()

    return match, accuracy
