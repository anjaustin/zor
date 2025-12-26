"""
TriX Native - PyTorch-Free Training and Inference

Primary API for TriX. No framework dependencies.
Requires CuPy for training, NumPy-only for inference.

Components:
    - NativeHollywoodSquares: Full native model with CUDA kernels
    - Trainer: Training loop with loss functions
    - AdamOptimizer: Pure CuPy Adam implementation
    - FrozenShapes: Mathematical primitives (XOR, AND, OR, NOT)
    - FrozenALU: 16 frozen shapes for 6502 operations
    - NativeRouter: Learnable routing layer
    - NativeFrozenHybrid: Frozen shapes + learned routing
    - Providence: Content-addressed memory with Hamming distance
    - SparseOctaveLookupFFN: Multi-scale sparse memory for FFN replacement
    - SparseOctaveTransformerFFN: Drop-in Transformer FFN replacement

Quick Start:
    >>> from trix.native import NativeHollywoodSquares, Trainer
    >>> model = NativeHollywoodSquares(d_model=128, num_tiles=16)
    >>> trainer = Trainer(model)
    >>> trainer.train(train_x, train_y, epochs=100)

Frozen Shapes + Routing:
    >>> from trix.native import NativeFrozenHybrid
    >>> model = NativeFrozenHybrid()
    >>> model.train_supervised()  # Learns opcode->shape mapping
    >>> # 0 learnable params in shapes, 176 in routing
    >>> # 100% accuracy on 6502 ALU operations

Sparse Octave Lookup:
    >>> from trix.native import SparseOctaveLookupFFN
    >>> ffn = SparseOctaveLookupFFN(d_model=768, n_octaves=3)
    >>> # Multi-scale sparse lookup instead of dense matrix multiply
    >>> # Interpretable: see which octave (scale) contributes what

"No PyTorch. No TensorFlow. No frameworks. Pure CUDA."
"""

# Model
from .model import NativeHollywoodSquares

# Training
from .trainer import (
    Trainer,
    AdamOptimizer,
    mse_loss,
    cross_entropy_loss,
    binary_cross_entropy_loss,
)

# Frozen shapes
from .frozen import (
    FrozenShapes,
    FrozenALU,
    int_to_bits,
    bits_to_int,
)

# Routing
from .router import (
    NativeRouter,
    NativeFrozenHybrid,
)

# CUDA kernels (for advanced users)
from .kernels import NATIVE_KERNELS

# Sparse Octave Lookup (multi-scale memory)
from .sparse_octave import (
    Providence,
    SparseOctaveLookupFFN,
    SparseOctaveTransformerFFN,
    SparseOctaveTrainer,
)

# Programmable Tiles (native Guardian interface)
from .programmable_tile import (
    NativeProgrammableTile,
    NativeProgrammableTileBank,
    NativeTrainingObserver,
    TileModification,
)

__all__ = [
    # Model
    'NativeHollywoodSquares',

    # Training
    'Trainer',
    'AdamOptimizer',
    'mse_loss',
    'cross_entropy_loss',
    'binary_cross_entropy_loss',

    # Frozen shapes
    'FrozenShapes',
    'FrozenALU',
    'int_to_bits',
    'bits_to_int',

    # Routing
    'NativeRouter',
    'NativeFrozenHybrid',

    # Sparse Octave Lookup
    'Providence',
    'SparseOctaveLookupFFN',
    'SparseOctaveTransformerFFN',
    'SparseOctaveTrainer',

    # Kernels
    'NATIVE_KERNELS',

    # Programmable Tiles
    'NativeProgrammableTile',
    'NativeProgrammableTileBank',
    'NativeTrainingObserver',
    'TileModification',
]
