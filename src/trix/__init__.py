"""
TriX - 2-Bit Sparse Ternary Neural Networks

Native-first. PyTorch-free training and inference.
"Computation is geometry. Learning is routing."

=============================================================================
PRIMARY API: Native (PyTorch-free)
=============================================================================

    from trix.native import NativeHollywoodSquares, Trainer

    # Create model (no PyTorch required)
    model = NativeHollywoodSquares(d_model=128, num_tiles=16)

    # Train with pure CuPy
    trainer = Trainer(model)
    trainer.train(train_x, train_y, epochs=100)

    # Save/load
    model.save("model.npz")

Frozen Shapes (0 learnable parameters):
    from trix.native import FrozenShapes, FrozenALU, NativeFrozenHybrid

    # Mathematical truths - 100% accurate on binary inputs
    result = FrozenShapes.xor(a, b)  # a + b - 2ab

    # Full 6502 ALU with learned routing
    model = NativeFrozenHybrid()
    model.train_supervised()  # 176 params learns opcode->shape mapping

=============================================================================
LEGACY API: PyTorch (for backward compatibility)
=============================================================================

    from trix import HierarchicalTriXFFN  # PyTorch module

    ffn = HierarchicalTriXFFN(d_model=512, num_tiles=16)
    output, routing_info, aux_losses = ffn(x)

Core Principle:
    Don't learn what you can read.
    Ternary weights encode preferences.
    Preferences enable routing.
    Routing enables sparsity.
    Sparsity enables speed.

"Train with gradients. Execute with geometry. Shape IS compute."
"""

__version__ = "0.5.0"  # Native-first release

# =============================================================================
# PRIMARY: Native Training and Inference (PyTorch-free)
# =============================================================================

from . import native
from .native import (
    # Model
    NativeHollywoodSquares,
    # Training
    Trainer,
    AdamOptimizer,
    mse_loss,
    cross_entropy_loss,
    # Frozen shapes
    FrozenShapes,
    FrozenALU,
    # Routing
    NativeRouter,
    NativeFrozenHybrid,
)

# =============================================================================
# LEGACY: PyTorch Modules (for backward compatibility)
# =============================================================================

from . import pytorch  # Alias with deprecation warnings

# =============================================================================
# RECOMMENDED: Hierarchical Content-Addressable Memory (The Big Leap)
# =============================================================================

from .nn import (
    # The main components - use these
    HierarchicalTriXFFN,    # FFN with 2-level hierarchical routing
    HierarchicalTriXBlock,  # Full transformer block
    TriXTile,               # Individual 2-bit specialist
)

# =============================================================================
# NEW: Sparse Lookup (Routing IS The Computation)
# =============================================================================

from .nn import (
    SparseLookupFFN,        # Routing selects direction, spline selects magnitude
    SparseLookupBlock,      # Full transformer block with SparseLookup
    TernarySpline2D,        # 2D spline with ternary coefficients
)

# =============================================================================
# MULTISCALE: Exact where exact, fuzzy where fuzzy (scaffold)
# =============================================================================

from .nn import (
    MultiScaleTriXFFN,      # Multi-octave FFN: fine/medium/coarse (scaffold)
    MultiScaleTriXBlock,    # Full transformer block with multi-scale
)

# =============================================================================
# TRUE OCTAVE: Derived multi-resolution (the real thing)
# =============================================================================

from .nn import (
    TrueOctaveFFN,          # Derived octaves: coarse = pool(fine)
    TrueOctaveBlock,        # Full transformer block with true octaves
)

# =============================================================================
# SIMPLE: Sparse Training (Option B) - 4 tiles, proven
# =============================================================================

from .nn import (
    SparseTriXFFN,    # Simple sparse FFN
    SparseTriXBlock,  # Simple transformer block
)

# =============================================================================
# CLASSIC: Original emergent routing (reference implementation)
# =============================================================================

from .nn import (
    TriXFFN,
    TriXBlock,
    TriXStack,
)

# =============================================================================
# CORE: Low-level kernel components
# =============================================================================

from .kernel import (
    TriXLinear,       # Base ternary linear layer
    STESign,          # Straight-through estimator for sign()
    pack_weights,     # Pack to 2-bit
    unpack_weights,   # Unpack from 2-bit
    trix_forward,     # NEON-accelerated forward
    is_neon_available,
)

# =============================================================================
# TRAINING: Quantization-aware training utilities
# =============================================================================

from .qat import (
    TernaryQuantizer,
    SoftTernaryQuantizer,
    TriXLinearQAT,
    progressive_quantization_schedule,
    QATTrainer,
)

# =============================================================================
# LEGACY: Learned routing (alternative approach)
# =============================================================================

from .nn import (
    Top1Gate,
    GatedFFN,
    TriXTransformerBlock,
)

# =============================================================================
# FROZEN 6502: Computation as Geometry (Mesa 14)
# =============================================================================

from .nn import (
    Frozen6502Net,
    CPUState,
    CPUStateWithFlags,
    FrozenShapeBank,
    OPCODE_TABLE,
    int_to_bits,
    bits_to_int,
)

# Shapes module - function-call interface
from . import shapes
from .shapes import (
    add, add_with_carry, sub, inc, dec,
    and_op, or_op, xor,
    asl, lsr, rol, ror,
)

# Assembly module - minimal assembler
from . import asm
from .asm import run as run_asm

# Simulation module - Providence (geometric memory)
from . import sim
from .sim import (
    OctaveMemory, add_octave,
    ProvidenceMemory, HierarchicalProvidenceMemory, hamming_distance,
)

# =============================================================================
# MESA 16: Sacred Foundry - Unified Providence (Attention = FFN = Routing)
# =============================================================================

from . import foundry
from .foundry import (
    # Token Mixing
    TokenMixer,
    create_token_mixer,
    hamming_distance as foundry_hamming_distance,
    binarize_ste,
    # Frozen Shapes
    MixingShapes,
    MixingShapeInfo,
    FrozenMixingLibrary,
    get_mixing_library,
    # Unified Blocks
    UnifiedProvidenceBlock,
    UnifiedProvidenceTransformer,
    create_unified_block,
    create_unified_transformer,
)

# =============================================================================
# MESA 16.2: Doula - Glass Box Meta-Learning (The Birth Attendant)
# =============================================================================

from . import doula
from .doula import (
    # Signatures
    DynamicsSignature,
    DoulaEntry,
    DoulaGuidance,
    # Core
    PhaseDetector,
    DoulaVDB,
    DoulaFunction,
    # Training integration
    DoulaTrainingConfig,
    DoulaCallback,
    DoulaTrainer,
    train_with_doula,
)

__all__ = [
    # Version
    "__version__",

    # PRIMARY: Native (PyTorch-free)
    "native",
    "NativeHollywoodSquares",
    "Trainer",
    "AdamOptimizer",
    "mse_loss",
    "cross_entropy_loss",
    "FrozenShapes",
    "FrozenALU",
    "NativeRouter",
    "NativeFrozenHybrid",

    # LEGACY: PyTorch alias
    "pytorch",

    # Recommended - Hierarchical (The Big Leap)
    "HierarchicalTriXFFN",
    "HierarchicalTriXBlock",
    "TriXTile",
    
    # New - Sparse Lookup (Routing IS Computation)
    "SparseLookupFFN",
    "SparseLookupBlock",
    "TernarySpline2D",
    
    # MultiScale - Exact where exact, fuzzy where fuzzy (scaffold)
    "MultiScaleTriXFFN",
    "MultiScaleTriXBlock",
    
    # TrueOctave - Derived multi-resolution (the real thing)
    "TrueOctaveFFN",
    "TrueOctaveBlock",
    
    # Simple - Sparse Training
    "SparseTriXFFN",
    "SparseTriXBlock",
    
    # Classic - Emergent Routing
    "TriXFFN",
    "TriXBlock",
    "TriXStack",
    
    # Core - Kernel
    "TriXLinear",
    "STESign",
    "pack_weights",
    "unpack_weights",
    "trix_forward",
    "is_neon_available",
    
    # Training - QAT
    "TernaryQuantizer",
    "SoftTernaryQuantizer",
    "TriXLinearQAT",
    "progressive_quantization_schedule",
    "QATTrainer",
    
    # Legacy - Learned Routing
    "Top1Gate",
    "GatedFFN",
    "TriXTransformerBlock",

    # Frozen 6502 - Computation as Geometry
    "Frozen6502Net",
    "CPUState",
    "CPUStateWithFlags",
    "FrozenShapeBank",
    "OPCODE_TABLE",
    "int_to_bits",
    "bits_to_int",

    # Shapes - Function interface
    "shapes",
    "add", "add_with_carry", "sub", "inc", "dec",
    "and_op", "or_op", "xor",
    "asl", "lsr", "rol", "ror",

    # Assembly - Minimal assembler
    "asm",
    "run_asm",

    # Simulation - Providence (Geometric Memory)
    "sim",
    "OctaveMemory",
    "add_octave",
    "ProvidenceMemory",
    "HierarchicalProvidenceMemory",
    "hamming_distance",

    # Mesa 16 - Sacred Foundry (Unified Providence)
    "foundry",
    "TokenMixer",
    "create_token_mixer",
    "foundry_hamming_distance",
    "binarize_ste",
    "MixingShapes",
    "MixingShapeInfo",
    "FrozenMixingLibrary",
    "get_mixing_library",
    "UnifiedProvidenceBlock",
    "UnifiedProvidenceTransformer",
    "create_unified_block",
    "create_unified_transformer",

    # Mesa 16.2 - Doula (Glass Box Meta-Learning)
    "doula",
    "DynamicsSignature",
    "DoulaEntry",
    "DoulaGuidance",
    "PhaseDetector",
    "DoulaVDB",
    "DoulaFunction",
    "DoulaTrainingConfig",
    "DoulaCallback",
    "DoulaTrainer",
    "train_with_doula",
]
