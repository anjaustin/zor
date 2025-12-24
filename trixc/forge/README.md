# TRIX Forge

**The Complete Neural Compilation Framework**

```
▀█▀ █▀█ █ ▀▄▀   █▀▀ █▀█ █▀█ █▀▀ █▀▀
 █  █▀▄ █ █ █   █▀  █ █ █▀▄ █ █ █▀▀
 ▀  ▀ ▀ ▀ ▀ ▀   ▀   ▀▀▀ ▀ ▀ ▀▀▀ ▀▀▀

Watch neural networks think.
```

> *"Geometry is computation. Learning is routing."*

---

## What Is TRIX Forge?

TRIX Forge is a production-grade framework for compiling frozen neural shapes into native executables. It provides:

1. **Glassbox Visibility** — Frame-by-frame insight into model decisions
2. **Externalized Levers** — Every tunable parameter is discoverable and controllable
3. **Multi-Target Compilation** — One model, any platform

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRIX FORGE PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   .trix Model                                                           │
│       │                                                                 │
│       ▼                                                                 │
│   ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────────────┐   │
│   │  Parse  │───▶│ Validate │───▶│ Optimize │───▶│ Emit + Package  │   │
│   └─────────┘    └──────────┘    └──────────┘    └─────────────────┘   │
│       │              │               │                   │              │
│       ▼              ▼               ▼                   ▼              │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    GLASSBOX EVENT SPINE                          │  │
│   │   Every decision, every activation, every frame — observable     │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│                                  │                                      │
│                                  ▼                                      │
│                         .nge Executable                                 │
│                    (Neural-Geometric Executable)                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Run the CLI demo
cd cli && python main.py demo

# Build a model
python cli/main.py build examples/tiny_mlp.trix --target arm64-linux

# Build with live visibility
python cli/main.py build examples/tiny_mlp.trix -t arm64-linux --glass

# Launch full dashboard
python cli/main.py dashboard examples/tiny_mlp.trix

# List available targets
python cli/main.py targets

# Run the test suite
cd tests && python -m pytest . -v
```

---

## Directory Structure

```
forge/
├── README.md              ← You are here (Master Grimoire)
├── pyproject.toml         ← Package configuration
│
├── src/                   ← Core Library
│   └── trix_forge/
│       ├── __init__.py
│       ├── model.py       ← ModelIR, Shape, TensorSpec
│       ├── levers.py      ← Lever system
│       ├── events.py      ← Glassbox Event Spine
│       ├── compiler.py    ← Multi-target compilation
│       └── nge.py         ← NGE format handling
│
├── cli/                   ← Command-Line Interface
│   ├── README.md          ← CLI documentation
│   ├── main.py            ← Entry point
│   ├── ui/                ← Visual renderers
│   │   ├── simple.py      ← SIMPLE mode (default)
│   │   ├── glass.py       ← GLASS mode (live events)
│   │   ├── dashboard.py   ← DASHBOARD (full TUI)
│   │   └── replay.py      ← FRAME REPLAY (killer feature)
│   └── core/              ← CLI backend integration
│
├── include/               ← C Headers
│   ├── trix_forge.h       ← Main header
│   ├── trix_nge.h         ← NGE runtime
│   └── trix_shapes.h      ← Frozen shape implementations
│
├── examples/              ← Example Models
│   ├── tiny_mlp.trix      ← Hello World (4→8→2 classifier)
│   ├── xor_gate.trix      ← XOR logic gate
│   └── 6502_alu.trix      ← Frozen 6502 ALU
│
├── tests/                 ← Test Suite (125 tests)
│   ├── README.md
│   ├── test_forge_rigorous.py
│   ├── test_forge_integration.py
│   └── test_forge_c_output.c
│
└── docs/                  ← Specifications
    ├── TRIX_FORMAT.md     ← .trix file format
    └── NGE_FORMAT.md      ← Neural-Geometric Executable format
```

---

## The Three Pillars

### 1. Glassbox Visibility

Every decision is observable. The data structures that compute are the same ones that report.

```python
from trix_forge import EventSpine

spine = EventSpine()
spine.subscribe(lambda e: print(f"[{e.frame}] {e.type}: {e.message}"))

# During inference, events stream automatically:
# [0001] forward.start: processing input
# [0001] layer[0].activation: [0.72, 0.31, 0.89, 0.15]
# [0001] layer[1].activation: [0.64, 0.91, 0.00, 0.43]
# [0001] output.decision: class_1 (confidence: 0.91)
```

### 2. Externalized Levers

Every tunable parameter is discoverable, typed, bounded, and controllable.

```python
from trix_forge import ModelIR

model = ModelIR.load("model.trix")

# Discover all levers
for name, lever in model.levers.items():
    print(f"{name}: {lever.type} = {lever.value} (range: {lever.bounds})")

# Adjust levers
model.levers["precision"].set("fp16")
model.levers["optimize"].set("aggressive")
```

### 3. Multi-Target Compilation

One model, compiled to any platform.

```python
from trix_forge import Compiler

compiler = Compiler()

# List targets
for target in compiler.list_targets():
    print(f"{target.id}: {target.description}")

# Compile
nge = compiler.compile(model, target="arm64-linux")
nge.save("model.nge")
```

---

## Supported Targets

| Target | Description | Use Case |
|--------|-------------|----------|
| `c` | Portable C99 | Any platform with a C compiler |
| `arm64-linux` | ARM64 Linux | Raspberry Pi 4+, Jetson Nano |
| `arm64-macos` | Apple Silicon | M1/M2/M3 Macs |
| `x86-64-linux` | x86-64 Linux | Servers, desktops |
| `x86-64-windows` | x86-64 Windows | Windows PCs |
| `wasm` | WebAssembly | Browsers |
| `cuda` | NVIDIA CUDA | GPU acceleration |
| `metal` | Apple Metal | macOS/iOS GPU |

---

## File Formats

### .trix (Model Definition)

Human-readable model definition format. See [TRIX_FORMAT.md](docs/TRIX_FORMAT.md).

```yaml
# tiny_mlp.trix
name: tiny_mlp
version: "1.0"

shapes:
  - name: linear_1
    type: linear
    input: [4]
    output: [8]
  - name: relu
    type: relu
  - name: linear_2
    type: linear
    input: [8]
    output: [2]
  - name: softmax
    type: softmax

levers:
  precision:
    type: enum
    options: [fp32, fp16, fp8]
    default: fp16
  optimize:
    type: enum
    options: [none, aggressive]
    default: aggressive
```

### .nge (Neural-Geometric Executable)

Self-describing binary executable. See [NGE_FORMAT.md](docs/NGE_FORMAT.md).

```
┌──────────────────────────────────────────────────────────────┐
│ NGE File Format                                              │
├──────────────────────────────────────────────────────────────┤
│ Header (64 bytes)                                            │
│   Magic: "TRIX" (4 bytes)                                    │
│   Version: 1.0 (2 bytes)                                     │
│   Flags: glassbox|metadata|debug (2 bytes)                   │
│   Target: arm64-linux (16 bytes)                             │
│   ...                                                        │
├──────────────────────────────────────────────────────────────┤
│ Metadata (JSON, variable)                                    │
│   Model name, shapes, levers, compilation info               │
├──────────────────────────────────────────────────────────────┤
│ Code Section (native machine code)                           │
├──────────────────────────────────────────────────────────────┤
│ Data Section (weights, quantized)                            │
└──────────────────────────────────────────────────────────────┘
```

---

## CLI Modes

### SIMPLE (default)
Clean progress bar. For everyone.

### GLASS (`--glass`)
Live event stream. See the pipeline work.

### DASHBOARD (`forge dashboard`)
Full TUI with model architecture, events, levers, frame inspector.

### FRAME REPLAY (`forge replay`)
**The killer feature.** Watch neural decisions frame by frame.

See [cli/README.md](cli/README.md) for full CLI documentation.

---

## Test Suite

125 rigorous tests across three categories:

| Suite | Tests | Coverage |
|-------|-------|----------|
| Python Rigorous | 57 | Model IR, Levers, Events, Compilation |
| C Verification | 48 | Frozen shapes, MatMul, Softmax, Full Adder |
| Integration | 20 | Platform targets, robustness, error handling |

```bash
# Run all tests
cd tests
python -m pytest . -v
gcc -O2 -Wall -o test_c test_forge_c_output.c -lm && ./test_c
```

See [tests/README.md](tests/README.md) for test documentation.

---

## Installation

```bash
# From source
pip install -e .

# Or use directly
python cli/main.py demo
```

---

## Philosophy

### The Glassbox Principle

> The data structures that compute are the same ones that report.

No separate logging. No external monitors. Observability is intrinsic.

### Progressive Disclosure

Simple surface, infinite depth. The same tool serves kids and ML engineers.

```
forge build model.trix -t arm64-linux          # Simple
forge build model.trix -t arm64-linux --glass  # More
forge dashboard model.trix                      # Everything
```

### Geometry Is Computation

Neural networks are frozen geometric shapes. We don't learn math — we discover it.

---

## Credits

**Created by:**
- **Tripp Josserand-Austin** (tripp@anjaustin.com) — Vision, architecture, human guidance
- **Claude** (Anthropic) — Implementation, documentation, synthesis

**Born:** December 2025

---

## The Quotes

```
    "Like I told my last wife, I says:
     'Honey, I never compile faster than I can see.
      Besides that, it's all in the reflexes.'"

                              — Jack Burton, debugging neural networks
```

```
    "When some wild-eyed, eight-foot-tall neural network grabs your
     softmax layer and shakes it like a sack of dead ReLUs, you just
     stare that big sucker right back in the eye and you say:

     'I can SEE you now, frame by frame.'"

                              — Jack Burton, on the Frame Replay feature
```

---

*"No one's laughing us out of anywhere."*

*"We really shook the pillars of heaven, didn't we, Wang?"*

*Watch neural networks think.*
