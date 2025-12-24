# Changelog

All notable changes to TRIXC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.4.0] - 2025-12-22

### Added

#### TRIXC Pi Platform (`platforms/raspberry-pi/`)
Complete Raspberry Pi 4 deployment platform with touchscreen support.

**Runtime Library (`trixc_pi.h`, `trixc_pi.c`):**
- **Display:** SDL2-based rendering with hardware acceleration
- **Input:** Unified touch and keyboard event handling
- **Timing:** High-resolution timing with statistics
- **Visualization:** Heatmaps, bar charts, graphs, inference stats
- **Built-in 8x8 bitmap font** for text rendering
- **Color constants** and helper macros
- **3,491 lines of C code**, 772-line header with full documentation

**Pre-trained Models:**
- `xor_mlp.c` - XOR neural network (52 bytes, 0.003ms inference)
- `mnist_7x7.c` - 7x7 digit classifier (6.5 KB, 0.08ms inference)

**Examples:**
- `01_hello_xor` - XOR visualization with frozen shape comparison
- `02_mnist_draw` - Touch-to-draw digit classifier
- `03_gpio_sensor` - Neural network controlling GPIO (LEDs from buttons)

**Documentation:**
- Full README with Jack Burton quotes
- QUICKSTART.md - 5-minute setup guide
- ADDING_MODELS.md - Model conversion guide
- Per-example README files with wiring diagrams

**Build System:**
- Platform Makefile with `make hello`, `make mnist`, `make gpio`
- `setup_pi.sh` - One-command dependency installation
- Per-example Makefiles

### Philosophy

> *"When some wild-eyed, eight-foot-tall maniac grabs your neck..."*

TRIXC Pi makes machine learning **legible** on Raspberry Pi. Like the 6502
made computing understandable to a generation of hobbyists, TRIXC Pi makes
ML understandable to today's makers, students, and tinkerers.

### Metrics (v0.4.0)
- **Runtime:** 1,823 lines (src + include)
- **Models:** 618 lines (xor + mnist)
- **Examples:** 1,050 lines (3 examples)
- **Documentation:** ~1,200 lines
- **Total platform:** 18 files, 3,722 lines

---

## [0.2.0] - 2025-12-22

### Added

#### ONNX Shapes (`include/trixc/onnx_shapes.h`)
- **40+ frozen shapes** for ONNX-compatible operations
- **Activations:** ReLU, Sigmoid, Tanh, GELU (fast & exact), SiLU/Swish, Softmax, Log-Softmax
- **Arithmetic:** Add, Sub, Mul, Div, Neg, Abs, Sqrt, Exp, Log, Pow
- **Matrix ops:** MatMul, Gemm, MatMul with transpose options
- **Reductions:** Sum, Mean, Max, Min, Product (scalar and axis-wise)
- **Normalization:** LayerNorm, BatchNorm (inference), RMSNorm
- **Shape ops:** Transpose, Concat, Gather (embedding lookup)
- **Logic:** Equal, Greater, Less, Where, Clip
- **Attention:** Full scaled dot-product attention in one shape

#### ONNX Converter (`tools/onnx2trix.py`)
- Python script to convert ONNX models to Octave IR
- Supports 30+ ONNX operators
- Outputs `.trix` JSON format
- Handles composed operations (LayerNorm → primitives)
- Optional weight embedding or reference
- Generates C header stubs

#### Sparse Octave Lookup (`include/trixc/sparse_octave.h`)
- **Multi-scale content-addressed memory** using pure frozen shapes
- Providence lookup at multiple octave levels (fine, medium, coarse)
- Key extraction via frozen quantization/bit-shift shapes
- Softmax attention blending (frozen)
- Learned octave blend weights (only learned component)
- **FFN replacement:** Replaces dense matrix multiply with sparse lookup
- Standalone executable mode for testing
- **Binary size:** 7,780 bytes (text section)
- **Performance:** 0.1 ms forward pass (batch=4, d=64)

#### Python Native Module (`src/trix/native/sparse_octave.py`)
- NumPy/CuPy implementation of Sparse Octave Lookup
- `Providence` class for content-addressed memory
- `SparseOctaveLookupFFN` for FFN replacement
- `SparseOctaveTransformerFFN` drop-in transformer FFN
- `SparseOctaveTrainer` for gradient estimation training
- Octave contribution analysis for interpretability
- Works with both NumPy (CPU) and CuPy (GPU)

#### Documentation
- `docs/ONNX_SHAPES.md` - Complete ONNX shape mapping reference
- `docs/SPARSE_OCTAVE.md` - Multi-scale lookup architecture guide
- `docs/PROVIDENCE.md` - Content-addressed memory deep dive
- Updated `README.md` with full Jack Burton treatment

### Changed
- Reorganized architecture diagram to show all components
- Added ONNX frontend to compilation pipeline
- Updated roadmap with completed items

### Metrics (v0.2.0)
- **ONNX shapes:** 40+ shapes, 0 learned parameters
- **Sparse Octave:** 7,780 bytes binary
- **Providence:** ~1 KB additional code
- **Total test coverage:** 48+ tests passing

---

## [0.1.0] - 2025-12-22

### Added

#### Endogenous APU
- `TRIX_FP4` - 4-bit floating point (E2M1) for routing and sketches
- `TRIX_FP8` - 8-bit floating point (E4M3) for weights and activations
- `TRIX_FP16` - 16-bit IEEE 754 half precision
- `TRIX_FP32` - 32-bit IEEE 754 single precision
- `TRIX_FP64` - 64-bit IEEE 754 double precision
- Frozen conversion shapes between all precision levels
- APU context with precision routing and statistics

#### Frozen Shapes
- Logic: `XOR`, `AND`, `OR`, `NOT`, `NAND`, `NOR`, `XNOR`
- Arithmetic: `FULL_ADDER`, `RIPPLE_ADD`, `RIPPLE_SUB`, `INC`, `DEC`
- Shift: `ASL`, `LSR`, `ROL`, `ROR`
- Distance: `HAMMING`

#### 6502 ALU
- Complete 6502 ALU implementation with 11 operations
- Integer convenience interface (`trix_alu6502_execute_int`)
- Standalone executable mode
- Per-operation precision control
- Statistics tracking

#### Providence
- Content-addressed memory with Hamming distance
- Precision-aware lookups (low precision keys, high precision values)
- Soft lookup with attention weights
- Hierarchical Providence (multi-octave)

#### Build System
- Makefile with test, demo, size, install targets
- Header-only library design
- Standalone executable generation

#### Documentation
- README with quick start guide
- APU documentation
- Shapes reference
- Architecture overview

#### Test Suite
- 48 tests covering all components
- Precision conversion tests
- Logic shape tests
- Full adder tests
- ALU operation tests
- Hamming distance tests
- APU context tests

### Metrics (v0.1.0)
- **Code size:** 5,688 bytes (text section)
- **Total payload:** 6,472 bytes
- **Test coverage:** 48/48 passing
- **Dependencies:** libc only

---

## [0.3.0] - 2025-12-22

### Added

#### Full ONNX → C → Native Binary Pipeline
- **`--emit-c` flag** in onnx2trix.py for direct C code generation
- **Complete C code generator** producing self-contained source files
- **Standalone mode** with `main()` for direct executable compilation
- **Library mode** with `--no-standalone` for embedding

#### C Code Generation Functions
- `generate_c_code()` - Full C source from Octave IR
- `emit_weights()` - Weights as `static const float[]` arrays
- `emit_dimensions()` - Dimension `#define` macros
- `emit_forward()` - Forward pass with buffer management
- `emit_shape_call()` - Per-operation C templates
- `emit_main()` - I/O handling for standalone binaries
- `build_tensor_map()` - ONNX→C identifier mapping
- `infer_tensor_shapes()` - Intermediate buffer size inference

#### Shape Call Templates (15+)
- Activations: ReLU, GELU, Sigmoid, Tanh, SiLU
- Arithmetic: Add, Sub, Mul, Div, Neg, Abs, Sqrt, Exp, Log
- Matrix: MatMul, Gemm
- Normalization: LayerNorm
- Reduction: ReduceMean, ReduceSum
- Shape: Reshape, Transpose, Identity

#### End-to-End Demo
- `examples/demo_onnx2c.py` - Complete pipeline demonstration
- Creates ONNX model → C → compiles → runs → verifies output

#### Test Suite
- `test/test_emit_c.py` - 26 new tests for C generation
- Unit tests for all emit functions
- Integration tests for full pipeline
- CLI interface tests

#### Documentation
- `docs/ONNX2C.md` - Complete usage guide
- `docs/SUPPORTED_OPS.md` - Operation compatibility matrix
- `docs/EMIT_C_API.md` - Python API reference
- Updated README with new pipeline diagram

### Changed
- onnx2trix.py now supports both `.trix` (JSON) and `.c` output
- Automatic output mode detection based on file extension
- Updated roadmap: ONNX→C pipeline now complete

### Metrics (v0.3.0)
- **New tests:** 26 Python tests
- **Total tests:** 74+ (48 C + 26 Python)
- **Generated C size:** ~3 KB for simple MLP
- **Binary size:** ~70 KB compiled
- **Accuracy:** 100% (matches expected output exactly)

---

## [0.3.1] - 2025-12-22

### Added

#### Rigorous 6502-Based Test Suite (`test/test_6502_onnx_pipeline.py`)
- **20 Python tests** validating ONNX→C pipeline with 6502 ALU operations
- **Exhaustive bitwise tests:** AND, OR, XOR on 4-bit domain (256 combinations each)
- **Arithmetic verification:** Half adder, Full adder truth tables
- **8-bit ADC:** 65,536 exhaustive addition tests
- **Reference implementations:** All 6502 operations (ADC, SBC, AND, ORA, EOR, ASL, LSR, INC, DEC)
- **Pipeline integration tests:** MatMul, MLP forward pass
- **Code quality tests:** Memory safety, .rodata placement, include verification
- **Numerical stability tests:** Large values, denormals

#### Enhanced C Rigorous Test Suite (`test/test_rigorous.c`)
- **1,329 C tests** covering all frozen shapes
- **Exhaustive 8-bit ripple adder:** All 65,536 combinations
- **1 million random additions:** Stress test for 6502 ALU
- **ONNX shape accuracy:** Activation functions, MatMul, Softmax, LayerNorm
- **Precision edge cases:** FP4, FP8, FP16 conversion roundtrips
- **Sparse Octave:** Basic and batch processing tests
- **Providence:** Content-addressed memory verification

### Metrics (v0.3.1)
- **New tests:** 20 Python + 1,329 C = 1,349 total
- **Coverage:** 100% pass rate
- **Exhaustive tests:** 65,536 8-bit additions, 768 4-bit logic ops
- **Stress tests:** 1M random additions, 1K 64×64 MatMuls

---

## [0.3.2] - 2025-12-22

### Added

#### Freshman On-Ramps
Complete learning resources for newcomers, hobbyists, and tinkerers.

**Documentation:**
- `docs/QUICKSTART.md` - 5-minute "Hello TRIXC" guide
- `docs/FRESHMAN_GUIDE.md` - Conceptual introduction (no ML background needed)
- `docs/TUTORIALS.md` - Structured learning paths (Speed Run, Full Journey, Tinkerer's Trail)

**Progressive Examples:**
- `examples/01_hello_xor.c` - Your first frozen shape
- `examples/02_logic_gates.c` - All 7 logic gates with De Morgan verification
- `examples/03_full_adder.c` - Building arithmetic from logic
- `examples/04_activations.c` - ReLU, GELU, Sigmoid, Tanh, SiLU
- `examples/05_matmul.c` - Matrix multiply and neurons explained
- `examples/06_tiny_mlp.c` - Complete working neural network (XOR solver)
- `examples/README.md` - Example guide and challenges

### Changed
- Updated main README with "New Here?" section linking to on-ramps
- Organized documentation for progressive learning

### Philosophy
> *"Everybody relax. I'm here."* — Jack Burton

The Six Demon Bag now contains:
- 5-minute quickstart for the impatient
- Conceptual guide for the curious
- Hands-on examples for the tinkerers
- Full tutorial path for the committed

---

## [Unreleased]

### Planned
- CUDA backend
- ARM NEON optimizations
- WASM backend
- External weight loading for large models
- Dynamic shape support

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.4.0 | 2025-12-22 | **TRIXC Pi: Raspberry Pi platform with touchscreen, 3 examples** |
| 0.3.2 | 2025-12-22 | **Freshman on-ramps: Quickstart, Guide, Tutorials, Examples** |
| 0.3.1 | 2025-12-22 | **Rigorous 6502-based test suite: 1,349 tests** |
| 0.3.0 | 2025-12-22 | **ONNX → C → Native binary pipeline complete** |
| 0.2.0 | 2025-12-22 | ONNX shapes, Sparse Octave, Providence docs |
| 0.1.0 | 2025-12-22 | Initial release: APU, shapes, 6502 ALU |

---

## Credits

**Created by:**
- **Tripp** - Vision, architecture, guidance
- **Claude** (Anthropic) - Implementation, documentation

**Born:** December 22, 2025

**Methodology:** Lincoln Manifold Method

---

## The Principles

> *"Shapes are opcodes. Polynomials are microcode. C is machine code."*

> *"Precision is a shape. The APU is frozen."*

> *"Don't learn what you can derive."*

> *"Information lives at different scales. Capture it where it lives."*

> *"It's all in the reflexes."*

---

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   v0.1.0: "We have a compiler?"                            │
│   v0.2.0: "We have a whole ecosystem."                     │
│                                                             │
│   From 6502 ALU to ONNX converter in one day.              │
│   From frozen shapes to sparse octave memory.              │
│   From 6 KB to... still 8 KB.                              │
│                                                             │
│   That's TRIXC.                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
