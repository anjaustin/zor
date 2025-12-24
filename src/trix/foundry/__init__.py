"""
The Sacred Foundry - Mesa 16

A Self-Evolving Universal Substrate for Summoning Elemental Functions.

Core Insight:
    TILE = signature + shape + state
         = address + transform + memory
         = key + computation + value

    "We do not create intelligence. We summon it from the frozen
     elements that always were."

Components:
    - FrozenMixingLibrary: Frozen shapes for token mixing
    - TokenMixer: Tokens route to tokens via Hamming distance
    - UnifiedProvidenceBlock: Mixing + Transform as one mechanism
    - UnifiedProvidenceTransformer: Full model using only Providence

The Unification:
    Attention and FFN are dissolved into a single mechanism.
    Both are Providence routing to frozen shapes.
    Same architecture. Different shape libraries.

"Attention is soft Providence. Providence is hard attention."
"""

from .mixing_shapes import (
    MixingShapes,
    MixingShapeInfo,
    FrozenMixingLibrary,
    get_mixing_library,
)

from .token_mixer import (
    TokenMixer,
    create_token_mixer,
    hamming_distance,
    binarize_ste,
)

from .unified_block import (
    UnifiedProvidenceBlock,
    UnifiedProvidenceTransformer,
    create_unified_block,
    create_unified_transformer,
)

from .frozen_foundry import (
    FrozenFoundry,
    PureMath,
    ShapeLibrary,
    BuildResult,
)

from .mos6502 import build_6502
from .wdc65816 import build_65816
from .intel286 import build_286
from .intel386 import build_486

__all__ = [
    # Mixing shapes
    'MixingShapes',
    'MixingShapeInfo',
    'FrozenMixingLibrary',
    'get_mixing_library',
    # Token mixer
    'TokenMixer',
    'create_token_mixer',
    'hamming_distance',
    'binarize_ste',
    # Unified blocks
    'UnifiedProvidenceBlock',
    'UnifiedProvidenceTransformer',
    'create_unified_block',
    'create_unified_transformer',
    # FrozenFoundry
    'FrozenFoundry',
    'PureMath',
    'ShapeLibrary',
    'BuildResult',
    # CPU builders
    'build_6502',
    'build_65816',
    'build_286',
    'build_486',
]
