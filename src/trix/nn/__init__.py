"""
TriX Neural Network Layers

Production-ready neural network components with frozen ternary computation.

CORE ARCHITECTURES (Production):
- TriXFFN: Base FFN with signature-based emergent routing
- HierarchicalTriXFFN: O(sqrt(n)) scaling for 64-1000+ tiles
- AnchoredDualModeFFN: Partition-first, dual deployment modes
- TrueOctaveFFN: Multi-resolution derived views
- ProvidenceFFN: Unified architecture (XOR + shapes + state)
- GradientTruthFFN: Explicit 3-layer decomposition

ROUTING:
- XORRoutingFFN: Hamming distance routing (O(1) vs O(d))
- SparseLookupFFN: Spline-based routing

SHAPES (Frozen Computation):
- FrozenTriXFFN: Content-addressable frozen shapes
- Frozen6502: CPU emulation (40x compression)

RESEARCH (Separate tracks):
- trix.nn.research.kan: Kolmogorov-Arnold Networks (Track B)
- trix.nn.research.sdpmx: Vi's Synthesis (SDPMX Pipeline)

ARCHIVE (Superseded, preserved for reference):
- trix.nn.archive: Deprecated modules with breadcrumbs

Consolidated: December 27, 2025
"""

import warnings

# =============================================================================
# CORE: Base FFN (Emergent Routing)
# =============================================================================

from .trix import (
    TriXFFN,
    TriXBlock,
    TriXStack,
)

# =============================================================================
# CORE: Hierarchical (O(sqrt(n)) Scaling)
# =============================================================================

from .hierarchical import (
    TriXTile,
    HierarchicalTriXFFN,
    HierarchicalTriXBlock,
)

# =============================================================================
# CORE: Anchored Dual-Mode (Partition First, Search Within)
# =============================================================================

from .anchored import (
    AnchoredDualModeFFN,
    AnchoredDualModeBlock,
    get_temperature_schedule,
)

# =============================================================================
# CORE: TrueOctave (Multi-Resolution Derived Views)
# =============================================================================

from .octave import (
    TrueOctaveFFN,
    TrueOctaveBlock,
    Octave,
    FrozenTile,
    derive_octave,
    OctaveRoutingInfo,
)

# =============================================================================
# CORE: Providence (The Unified Architecture)
# =============================================================================

from .providence import (
    ProvidenceTile,
    ProvidenceFFN,
    ProvidenceBlock,
    create_providence_ffn,
    create_frozen_providence_ffn,
)

# =============================================================================
# CORE: Gradient Truth (Training Beyond STE)
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
# ROUTING: XOR (Hamming Distance)
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
# ROUTING: SparseLookup (Spline-Based)
# =============================================================================

from .sparse_lookup import (
    SparseLookupFFN,
    SparseLookupBlock,
    TernarySpline2D,
    FloatSpline2D,
)

# =============================================================================
# SHAPES: Frozen Computation
# =============================================================================

from .frozen import (
    FrozenShape,
    FrozenTile as FrozenShapeTile,  # Alias to avoid conflict with octave.FrozenTile
    FrozenShapeRegistry,
    FrozenTriXFFN,
    derive_signature,
    derive_signature_from_fn,
    get_default_registry,
    create_frozen_ffn_from_names,
    verify_frozen_tile,
)

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
# UTILITY: XOR Superposition (Compression)
# =============================================================================

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
# UTILITY: Compiled Dispatch
# =============================================================================

from .compiled_dispatch import (
    CompiledDispatch,
    CompiledEntry,
    ProfileStats,
)

# =============================================================================
# UTILITY: Temporal Tiles
# =============================================================================

from .temporal_tiles import (
    TemporalTileLayer,
    TemporalTileStack,
)

# =============================================================================
# UTILITY: Hierarchical Temporal
# =============================================================================

from .hierarchical_temporal import (
    TemporalTile,
    HierarchicalTemporalFFN,
    create_hierarchical_temporal_ffn,
)

# =============================================================================
# LEGACY: Layers (Deprecated - kept for compatibility)
# =============================================================================

from .layers import (
    Top1Gate,
    GatedFFN,
    TriXTransformerBlock,
)

# =============================================================================
# RESEARCH: KAN (Track B) - Access via trix.nn.research.kan
# =============================================================================

from .research.kan import (
    Spline1D,
    AdditiveKAN,
    ProductSplineKAN,
    SharedAdditiveKAN,
    AdditiveKANTile,
    KANTile,
    HierarchicalKANFFN,
)

# =============================================================================
# RESEARCH: SDPMX (Vi's Synthesis) - Access via trix.nn.research.sdpmx
# =============================================================================

from .research.sdpmx import (
    SDPMXSpline1D,
    SmoothingOperator,
    DifferentiationOperator,
    ProjectionOperator,
    MaskOperator,
    XOROperator,
    SDPMXPipeline,
)

# =============================================================================
# ARCHIVE: Deprecated modules - Access via trix.nn.archive
# Preserved for reference. See archive/ARCHIVE_INDEX.md
# =============================================================================

# Lazy imports for archived classes (with deprecation warnings)
_archived_classes = {
    "EmergentGatedFFN": ("archive.emergent", "HierarchicalTriXFFN"),
    "EmergentTransformerBlock": ("archive.emergent", "HierarchicalTriXBlock"),
    "SparseTriXFFN": ("archive.sparse", "HierarchicalTriXFFN"),
    "SparseTriXBlock": ("archive.sparse", "HierarchicalTriXBlock"),
    "SparseLookupFFNv2": ("archive.sparse_lookup_v2", "SparseLookupFFN with use_gradient_truth=True"),
    "SparseLookupBlockV2": ("archive.sparse_lookup_v2", "SparseLookupBlock"),
    "ScoreCalibrationSpline": ("archive.sparse_lookup_v2", "TernarySpline2D"),
    "MultiScaleTriXFFN": ("archive.multiscale", "TrueOctaveFFN"),
    "MultiScaleTriXBlock": ("archive.multiscale", "TrueOctaveBlock"),
    "MultiScaleRoutingInfo": ("archive.multiscale", "OctaveRoutingInfo"),
    "MultiScaleOctave": ("archive.multiscale", "Octave"),
    "MultiScaleOctaveTile": ("archive.multiscale", "FrozenTile"),
}

def __getattr__(name):
    if name in _archived_classes:
        module_path, replacement = _archived_classes[name]
        warnings.warn(
            f"{name} is archived. Use {replacement} instead. "
            f"See trix.nn.archive.ARCHIVE_INDEX.md for details.",
            DeprecationWarning,
            stacklevel=2
        )
        # Import from archive
        parts = module_path.split(".")
        if parts[0] == "archive":
            from . import archive
            mod = getattr(archive, parts[1])
            return getattr(mod, name)
    raise AttributeError(f"module 'trix.nn' has no attribute '{name}'")

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # === CORE: Production Architectures ===
    # Base FFN
    "TriXFFN",
    "TriXBlock",
    "TriXStack",
    # Hierarchical
    "TriXTile",
    "HierarchicalTriXFFN",
    "HierarchicalTriXBlock",
    # Anchored
    "AnchoredDualModeFFN",
    "AnchoredDualModeBlock",
    "get_temperature_schedule",
    # Octave
    "TrueOctaveFFN",
    "TrueOctaveBlock",
    "Octave",
    "FrozenTile",
    "derive_octave",
    "OctaveRoutingInfo",
    # Providence
    "ProvidenceTile",
    "ProvidenceFFN",
    "ProvidenceBlock",
    "create_providence_ffn",
    "create_frozen_providence_ffn",
    # Gradient Truth
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

    # === ROUTING ===
    # XOR Routing
    "binarize_ste",
    "ternarize_ste",
    "xor_hamming_distance",
    "soft_hamming_distance",
    "XORRouter",
    "XORRoutingFFN",
    "HierarchicalXORRoutingFFN",
    # SparseLookup
    "SparseLookupFFN",
    "SparseLookupBlock",
    "TernarySpline2D",
    "FloatSpline2D",

    # === SHAPES (Frozen Computation) ===
    "FrozenShape",
    "FrozenShapeTile",
    "FrozenShapeRegistry",
    "FrozenTriXFFN",
    "derive_signature",
    "derive_signature_from_fn",
    "get_default_registry",
    "create_frozen_ffn_from_names",
    "verify_frozen_tile",
    # Shape Library
    "Primitives",
    "LogicShapes",
    "ComparisonShapes",
    "ActivationShapes",
    "ArithmeticShapes",
    "CompositionShapes",
    "FrozenShapeLibrary",
    "GeneralFrozenTile",
    "get_frozen_shape_library",
    # 6502
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
    "Frozen6502Net",
    "CPUState",
    "CPUStateWithFlags",
    "FrozenShapeBank",
    "OpcodeSpec",
    "OPCODE_TABLE",
    "Reg",
    "get_opcode_id",
    "get_opcode_name",

    # === UTILITY ===
    # Compression
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
    # Compiled Dispatch
    "CompiledDispatch",
    "CompiledEntry",
    "ProfileStats",
    # Temporal
    "TemporalTileLayer",
    "TemporalTileStack",
    "TemporalTile",
    "HierarchicalTemporalFFN",
    "create_hierarchical_temporal_ffn",

    # === LEGACY (Deprecated) ===
    "Top1Gate",
    "GatedFFN",
    "TriXTransformerBlock",

    # === RESEARCH ===
    # KAN (Track B)
    "Spline1D",
    "AdditiveKAN",
    "ProductSplineKAN",
    "SharedAdditiveKAN",
    "AdditiveKANTile",
    "KANTile",
    "HierarchicalKANFFN",
    # SDPMX
    "SDPMXSpline1D",
    "SmoothingOperator",
    "DifferentiationOperator",
    "ProjectionOperator",
    "MaskOperator",
    "XOROperator",
    "SDPMXPipeline",

    # === ARCHIVE (Deprecated - lazy loaded with warnings) ===
    # These will emit deprecation warnings when accessed
    "EmergentGatedFFN",
    "EmergentTransformerBlock",
    "SparseTriXFFN",
    "SparseTriXBlock",
    "SparseLookupFFNv2",
    "SparseLookupBlockV2",
    "ScoreCalibrationSpline",
    "MultiScaleTriXFFN",
    "MultiScaleTriXBlock",
    "MultiScaleRoutingInfo",
]
