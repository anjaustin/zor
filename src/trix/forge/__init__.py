"""
trix.forge: Unified Neural-Geometric Deterministic Systems Foundry.

The forge compiles declarative specifications into frozen neural-geometric
models that compute exactly using polynomial shapes.

The Foundry Interface (Recommended):
    from trix.forge import Foundry

    foundry = Foundry(bits=8)
    foundry.atom("xor", lambda a, b: a ^ b)
    foundry.atom("and", lambda a, b: a & b)
    foundry.atom("add", lambda a, b: (a + b) & 0xFF)

    system = foundry.build()
    result = system.validate(exhaustive=True)
    assert result.all_passed()

    print(system.execute(42, 13, "xor"))  # 39
    system.export_cuda("output/")

The Chip DSL (Legacy):
    from trix.forge import Chip

    chip = Chip("alu_4op")
    chip.input("a", 8).input("b", 8).input("op", 2)
    chip.operation(0, "add")
    chip.operation(1, "sub")
    chip.operation(2, "xor")
    chip.operation(3, "and")
    chip.output("result", 8)
    chip.compile()

    print(chip.execute(17, 38, "add"))  # 55

The Three Atoms:
    All shapes are built from three polynomial primitives:
    - AND(a, b) = a * b
    - XOR(a, b) = a + b - 2*a*b
    - NOT(a)    = 1 - a

Everything else is composition.
"""

from .chip import Chip, CompiledChip, Port, Operation
from .xorpu_spec import XORPU, XORPUSpec, ShapeSpec, Shapes32, Truth32
from .term import (
    Term, BitTerms, ShapeTerms,
    xor_terms, and_terms, or_terms, not_terms, nand_terms, nor_terms, xnor_terms,
    generate_xor_shape, generate_and_shape, generate_or_shape, generate_not_shape,
    generate_nand_shape, generate_nor_shape, generate_xnor_shape, generate_nop_shape,
    generate_add_shape, generate_sub_shape,
    generate_sll_shape, generate_srl_shape, generate_sra_shape,
    generate_slt_shape, generate_sltu_shape,
    SHAPE_GENERATORS, generate_shape, generate_all_shapes,
    validate_terms_against_tensor,
    STANDARD_WIDTHS, generate_all_shapes_multiwidth, validate_multiwidth, multiwidth_summary,
    FAST_PATH_SHAPES, is_fast_path, compute_fast, compute_auto,
    evaluate_batch, evaluate_batch_fast, compression_stats,
)
from .verilog import shape_to_verilog, export_verilog, generate_testbench
from .cuda import shape_to_cuda, export_cuda
from .hardware import (
    HardwareEstimate,
    estimate_shape,
    estimate_xorpu,
    estimate_luts,
    estimate_ffs,
    estimate_cycles,
    estimate_power,
    estimate_summary,
)
from .backend import (
    Backend,
    BackendInfo,
    CPUBackend,
    CUDABackend,
    get_backend,
    list_backends,
    register_backend,
)
from .foundry import Foundry, generate_from_truth
from .system import System, SystemValidation
from .composition import (
    Composition,
    Seq, Par, Sel, Rep,
    seq, par, sel, rep,
    compose_sequential,
    compose_parallel,
    compose_selection,
    compose_repetition,
)
from .signature import (
    signature_from_terms,
    signature_distance,
    signature_similarity,
    match_signature,
    SignatureTable,
    derive_all_signatures,
    build_signature_table,
    DEFAULT_SIG_DIM,
)

__all__ = [
    # Chip specification
    "Chip",
    "CompiledChip",
    "Port",
    "Operation",

    # XORPU
    "XORPU",
    "XORPUSpec",
    "ShapeSpec",
    "Shapes32",
    "Truth32",

    # Explicit Terms
    "Term",
    "BitTerms",
    "ShapeTerms",

    # Primitive term generators
    "xor_terms",
    "and_terms",
    "or_terms",
    "not_terms",
    "nand_terms",
    "nor_terms",
    "xnor_terms",

    # Shape term generators
    "generate_xor_shape",
    "generate_and_shape",
    "generate_or_shape",
    "generate_not_shape",
    "generate_nand_shape",
    "generate_nor_shape",
    "generate_xnor_shape",
    "generate_nop_shape",
    "generate_add_shape",
    "generate_sub_shape",
    "generate_sll_shape",
    "generate_srl_shape",
    "generate_sra_shape",
    "generate_slt_shape",
    "generate_sltu_shape",

    # Registry
    "SHAPE_GENERATORS",
    "generate_shape",
    "generate_all_shapes",

    # Validation
    "validate_terms_against_tensor",

    # Multi-width
    "STANDARD_WIDTHS",
    "generate_all_shapes_multiwidth",
    "validate_multiwidth",
    "multiwidth_summary",

    # Optimization: Fast path
    "FAST_PATH_SHAPES",
    "is_fast_path",
    "compute_fast",
    "compute_auto",

    # Optimization: Batch evaluation
    "evaluate_batch",
    "evaluate_batch_fast",

    # Optimization: Compression
    "compression_stats",

    # Verilog Export
    "shape_to_verilog",
    "export_verilog",
    "generate_testbench",

    # CUDA Export
    "shape_to_cuda",
    "export_cuda",

    # Hardware Estimation
    "HardwareEstimate",
    "estimate_shape",
    "estimate_xorpu",
    "estimate_luts",
    "estimate_ffs",
    "estimate_cycles",
    "estimate_power",
    "estimate_summary",

    # Backend Layer
    "Backend",
    "BackendInfo",
    "CPUBackend",
    "CUDABackend",
    "get_backend",
    "list_backends",
    "register_backend",

    # Foundry (Unified Interface)
    "Foundry",
    "generate_from_truth",

    # System (Built Artifact)
    "System",
    "SystemValidation",

    # Composition Operators
    "Composition",
    "Seq",
    "Par",
    "Sel",
    "Rep",
    "seq",
    "par",
    "sel",
    "rep",
    "compose_sequential",
    "compose_parallel",
    "compose_selection",
    "compose_repetition",

    # Signature Derivation
    "signature_from_terms",
    "signature_distance",
    "signature_similarity",
    "match_signature",
    "SignatureTable",
    "derive_all_signatures",
    "build_signature_table",
    "DEFAULT_SIG_DIM",
]
