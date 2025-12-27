"""
TriX Neural Network Layers

High-level neural network components built on TriX sparse layers.

Recommended (emergent routing):
- TriXFFN: Feed-forward network with signature-based routing
- TriXBlock: Transformer block with TriXFFN
- TriXStack: Stack of transformer blocks

Alternative (learned routing):
- GatedFFN: Feed-forward network with learned gate network
"""

from .layers import (
    Top1Gate,
    GatedFFN,
    TriXTransformerBlock,
)

from .emergent import (
    EmergentGatedFFN,
    EmergentTransformerBlock,
)

from .trix import (
    TriXFFN,
    TriXBlock,
    TriXStack,
)

from .sparse import (
    SparseTriXFFN,
    SparseTriXBlock,
)

from .hierarchical import (
    TriXTile,
    HierarchicalTriXFFN,
    HierarchicalTriXBlock,
)

from .multiscale import (
    MultiScaleTriXFFN,
    MultiScaleTriXBlock,
    Octave as MultiScaleOctave,
    OctaveTile as MultiScaleOctaveTile,
    MultiScaleRoutingInfo,
)

from .octave import (
    TrueOctaveFFN,
    TrueOctaveBlock,
    Octave,
    FrozenTile,
    derive_octave,
    OctaveRoutingInfo,
)

from .sparse_lookup import (
    SparseLookupFFN,
    SparseLookupBlock,
    TernarySpline2D,
    FloatSpline2D,
)

from .sparse_lookup_v2 import (
    SparseLookupFFNv2,
    SparseLookupBlockV2,
    ScoreCalibrationSpline,
)

from .temporal_tiles import (
    TemporalTileLayer,
    TemporalTileStack,
)
from .compiled_dispatch import (
    CompiledDispatch,
    CompiledEntry,
    ProfileStats,
)

from .xor_superposition import (
    SparseDelta,
    CompressedSignatures,
    CompressionStats,
    SuperpositionRouter,
    XORSuperpositionFFN,
    create_compressed_ffn,
    pack_ternary_to_uint8,
    unpack_uint8_to_ternary,
    hamming_distance_packed,
    hamming_distance_batch,
)

# =============================================================================
# XOR ROUTING (Providence - Content-Addressable FFN)
# =============================================================================

from .xor_ffn import (
    binarize_ste,
    ternarize_ste,
    hamming_distance as xor_hamming_distance,
    soft_hamming_distance,
    XORRouter,
    XORRoutingFFN,
    HierarchicalXORRoutingFFN,
)

# =============================================================================
# FROZEN SHAPES LIBRARY (General-Purpose Mathematical Shapes)
# =============================================================================

from .frozen_shapes import (
    Primitives,
    LogicShapes,
    ComparisonShapes,
    ActivationShapes,
    ArithmeticShapes,
    CompositionShapes,
    FrozenShapeLibrary,
    GeneralFrozenTile,
    get_library as get_frozen_shape_library,
)

# =============================================================================
# HIERARCHICAL TEMPORAL (State + O(√n) Routing)
# =============================================================================

from .hierarchical_temporal import (
    TemporalTile,
    HierarchicalTemporalFFN,
    create_hierarchical_temporal_ffn,
)

# =============================================================================
# PROVIDENCE (The Unified Architecture - FFN as Content-Addressable Memory)
# =============================================================================

from .providence import (
    ProvidenceTile,
    ProvidenceFFN,
    ProvidenceBlock,
    create_providence_ffn,
    create_frozen_providence_ffn,
)

# =============================================================================
# GRADIENT TRUTH (Training Beyond STE - Gradients Only Where Uncertain)
# =============================================================================

from .gradient_truth import (
    Shape,
    ShapeBank,
    PolynomialShapeBank,
    DistilledShapeBank,
    GradientTruthFFN,
    GradientTruthBlock,
    RoutingInfo,
    ShapeGenesis,
    create_gradient_truth_ffn,
    create_gradient_truth_block,
)

# =============================================================================
# KAN (Kolmogorov-Arnold Networks) - Track B
# =============================================================================

from .additive_kan import (
    Spline1D,
    AdditiveKAN,
    ProductSplineKAN,
    SharedAdditiveKAN,
    KANTile as AdditiveKANTile,
)

from .kan_hierarchical import (
    KANTile,
    HierarchicalKANFFN,
)

# =============================================================================
# SDPMX Pipeline (Vi's Synthesis - Hilbert Space Operators)
# =============================================================================

from .sdpmx_pipeline import (
    Spline1D as SDPMXSpline1D,
    SmoothingOperator,
    DifferentiationOperator,
    ProjectionOperator,
    MaskOperator,
    XOROperator,
    SDPMXPipeline,
)

# =============================================================================
# FROZEN SHAPES (Computation as Geometry)
# =============================================================================

from .frozen import (
    FrozenShape,
    FrozenTile,
    FrozenShapeRegistry,
    FrozenTriXFFN,
    derive_signature,
    derive_signature_from_fn,
    get_default_registry,
    create_frozen_ffn_from_names,
    verify_frozen_tile,
)

from .frozen_6502 import (
    ShapeID,
    RegisterID,
    PureMath,
    FrozenShapes6502,
    FlagComputer,
    Frozen6502Tile,
    Frozen6502,
    OPCODE_SPECS,
    int_to_bits,
    bits_to_int,
    create_frozen_6502,
    register_6502_shapes,
)

from .frozen_6502_net import (
    Frozen6502Net,
    CPUState,
    CPUStateWithFlags,
    FrozenShapeBank,
    OpcodeSpec,
    OPCODE_TABLE,
    Reg,
    get_opcode_id,
    get_opcode_name,
)

# =============================================================================
# ANCHORED DUAL-MODE (Partition First, Search Within)
# =============================================================================

from .anchored import (
    AnchoredDualModeFFN,
    AnchoredDualModeBlock,
    get_temperature_schedule,
)

__all__ = [
    # Recommended - emergent routing (zero parameters)
    "TriXFFN",
    "TriXBlock",
    "TriXStack",
    # Sparse training (Option B)
    "SparseTriXFFN",
    "SparseTriXBlock",
    # Hierarchical (The Big Leap - 64+ tiles)
    "TriXTile",
    "HierarchicalTriXFFN",
    "HierarchicalTriXBlock",
    # MultiScale (Exact where exact, fuzzy where fuzzy) - scaffold
    "MultiScaleTriXFFN",
    "MultiScaleTriXBlock",
    "MultiScaleRoutingInfo",
    # TrueOctave (Derived multi-resolution) - the real thing
    "TrueOctaveFFN",
    "TrueOctaveBlock",
    "Octave",
    "FrozenTile",
    "derive_octave",
    "OctaveRoutingInfo",
    # SparseLookup (Routing IS Computation)
    "SparseLookupFFN",
    "SparseLookupBlock",
    "TernarySpline2D",
    "FloatSpline2D",
    # SparseLookup v2 (with Surgery + Regularization)
    "SparseLookupFFNv2",
    "SparseLookupBlockV2",
    "ScoreCalibrationSpline",
    # Compiled Dispatch (Path Compilation)
    "CompiledDispatch",
    "CompiledEntry",
    "ProfileStats",
    # Temporal Tiles (Mesa 4)
    "TemporalTileLayer",
    "TemporalTileStack",
    # Alternative - learned routing
    "Top1Gate",
    "GatedFFN",
    "TriXTransformerBlock",
    # Experimental
    "EmergentGatedFFN",
    "EmergentTransformerBlock",
    # XOR Superposition (11.6x compression)
    "SparseDelta",
    "CompressedSignatures",
    "CompressionStats",
    "SuperpositionRouter",
    "XORSuperpositionFFN",
    "create_compressed_ffn",
    "pack_ternary_to_uint8",
    "unpack_uint8_to_ternary",
    "hamming_distance_packed",
    "hamming_distance_batch",
    # KAN (Kolmogorov-Arnold Networks) - Track B
    "Spline1D",
    "AdditiveKAN",
    "ProductSplineKAN",
    "SharedAdditiveKAN",
    "AdditiveKANTile",
    "KANTile",
    "HierarchicalKANFFN",
    # SDPMX Pipeline (Vi's Synthesis - Hilbert Space Operators)
    "SDPMXSpline1D",
    "SmoothingOperator",
    "DifferentiationOperator",
    "ProjectionOperator",
    "MaskOperator",
    "XOROperator",
    "SDPMXPipeline",
    # Frozen Shapes (Computation as Geometry - Mesa 14)
    "FrozenShape",
    "FrozenTile",
    "FrozenShapeRegistry",
    "FrozenTriXFFN",
    "derive_signature",
    "derive_signature_from_fn",
    "get_default_registry",
    "create_frozen_ffn_from_names",
    "verify_frozen_tile",
    # Frozen 6502 (16 shapes for CPU emulation)
    "ShapeID",
    "RegisterID",
    "PureMath",
    "FrozenShapes6502",
    "FlagComputer",
    "Frozen6502Tile",
    "Frozen6502",
    "OPCODE_SPECS",
    "int_to_bits",
    "bits_to_int",
    "create_frozen_6502",
    "register_6502_shapes",
    # Frozen 6502 Net (THE Neural Network That IS a 6502)
    "Frozen6502Net",
    "CPUState",
    "CPUStateWithFlags",
    "FrozenShapeBank",
    "OpcodeSpec",
    "OPCODE_TABLE",
    "Reg",
    "get_opcode_id",
    "get_opcode_name",
    # XOR Routing (Providence - Mesa 15)
    "binarize_ste",
    "ternarize_ste",
    "xor_hamming_distance",
    "soft_hamming_distance",
    "XORRouter",
    "XORRoutingFFN",
    "HierarchicalXORRoutingFFN",
    # Frozen Shapes Library (General-Purpose)
    "Primitives",
    "LogicShapes",
    "ComparisonShapes",
    "ActivationShapes",
    "ArithmeticShapes",
    "CompositionShapes",
    "FrozenShapeLibrary",
    "GeneralFrozenTile",
    "get_frozen_shape_library",
    # Hierarchical Temporal (State + O(√n) Routing - Mesa 15)
    "TemporalTile",
    "HierarchicalTemporalFFN",
    "create_hierarchical_temporal_ffn",
    # Providence (The Unified Architecture - Mesa 15)
    "ProvidenceTile",
    "ProvidenceFFN",
    "ProvidenceBlock",
    "create_providence_ffn",
    "create_frozen_providence_ffn",
    # Gradient Truth (Training Beyond STE)
    "Shape",
    "ShapeBank",
    "PolynomialShapeBank",
    "DistilledShapeBank",
    "GradientTruthFFN",
    "GradientTruthBlock",
    "RoutingInfo",
    "ShapeGenesis",
    "create_gradient_truth_ffn",
    "create_gradient_truth_block",
    # Anchored Dual-Mode (Partition First, Search Within)
    "AnchoredDualModeFFN",
    "AnchoredDualModeBlock",
    "get_temperature_schedule",
]
